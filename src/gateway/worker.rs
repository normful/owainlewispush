//! Per-thread worker: runs one message through an agent backend, records the
//! outcome in canonical history, and delivers the reply.

use std::future::{pending, Future};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use crate::agent::{Request, RunError, RunOutput, Runner};
use crate::approval::{Choice, Question};
use crate::attachments::{DownloadedFile, PreparedFiles, MAX_FILE_BYTES, MAX_FILE_COUNT};
use crate::config::{AgentBackend, Config};
use crate::history::{DeliveryStatus, OutboundMessage, OutboundOrigin};
use crate::image::{PreparedImages, MAX_IMAGE_BYTES, MAX_IMAGE_COUNT};
use crate::pi::rpc::{UiFuture, UiHandler, UiRequest, UiResponse};
use crate::prompt::{ComposedPrompt, Composer};
use crate::util::now_ms;
use crate::voice::MAX_AUDIO_BYTES;

use super::{audit, audit_schedule_events, complete_row, Ctx, Job, WorkerState};

pub(super) const SESSION_SETUP_FAILURE: &str =
    "Push could not prepare this conversation. Check the local logs, then resend.";

/// Processes one thread's jobs strictly in order, exiting when the queue closes.
pub(super) async fn run(
    ctx: Ctx,
    mut rx: mpsc::Receiver<Job>,
    mut cancel: watch::Receiver<i64>,
    state: Arc<std::sync::Mutex<WorkerState>>,
) {
    while let Some(job) = rx.recv().await {
        let row_id = job.row_id;
        {
            let mut state = state.lock().unwrap();
            state.current_row = Some(row_id);
            state.retained_rows.remove(&row_id);
        }
        handle_with_interrupt(&ctx, job, interrupt(&mut cancel, row_id)).await;
        let completed = !ctx.ack.lock().unwrap().in_flight.contains(&row_id);
        let mut state = state.lock().unwrap();
        state.current_row = None;
        if completed {
            state.pending.retain(|job| job.row_id != row_id);
            state.retained_rows.remove(&row_id);
        } else {
            state.retained_rows.insert(row_id);
        }
    }
}

#[cfg(test)]
pub(super) async fn handle(ctx: &Ctx, job: Job) {
    handle_with_interrupt(ctx, job, pending()).await;
}

async fn handle_with_interrupt<I>(ctx: &Ctx, mut job: Job, interrupt: I)
where
    I: Future<Output = ()>,
{
    let existing_outbound = match ctx.history.lock().unwrap().outbound_for(job.inbound_id) {
        Ok(outbound) => outbound,
        Err(error) => {
            history_error(ctx, &job, "read outbound", error);
            return;
        }
    };
    if let Some(outbound) = existing_outbound {
        report_delivery(
            ctx,
            &job,
            deliver_stored(ctx, &job, &outbound).await,
            &outbound.content,
            "recovered_outbound",
            "recover outbound",
        );
        review_changed_schedules(ctx, &job).await;
        return;
    }

    if job.voice_attachment.is_some() {
        match prepare_voice(ctx, &job).await {
            Ok(transcript) => job.text = transcript,
            Err(VoicePreparationError::History(error)) => {
                history_error(ctx, &job, "store voice transcript", error);
                return;
            }
            Err(VoicePreparationError::User {
                event,
                reply,
                detail,
            }) => {
                warn!("[{}] {detail}", job.thread);
                audit(
                    ctx,
                    ctx.audit
                        .failed(event, job.row_id, &job.thread, Some(job.backend), detail),
                );
                let delivery = record_and_deliver(ctx, &job, OutboundOrigin::Gateway, reply).await;
                report_delivery(ctx, &job, delivery, reply, event, "deliver voice fallback");
                return;
            }
        }
    }

    if job.image_attachments.is_empty() && job.file_attachments.is_empty() {
        if let Some(reply) = command(ctx, &job) {
            let delivery = record_and_deliver(ctx, &job, OutboundOrigin::Gateway, &reply).await;
            if delivery.is_ok() {
                info!(
                    "[{}] command reply sent via {}",
                    job.thread,
                    ctx.channel.id()
                );
            }
            report_delivery(
                ctx,
                &job,
                delivery,
                &reply,
                "command",
                "record command reply",
            );
            return;
        }
    }

    let Some(runner) = ctx.runners.get(&job.backend) else {
        error!(
            "[{}] no runner configured for {}",
            job.thread,
            job.backend.as_str()
        );
        audit(
            ctx,
            ctx.audit.failed(
                "backend_run_failed",
                job.row_id,
                &job.thread,
                Some(job.backend),
                "no runner configured",
            ),
        );
        complete_job(ctx, &job, "missing_runner");
        return;
    };

    let session_result = {
        let initial_session_id = runner.initial_session_id();
        ctx.store.lock().unwrap().session_for(
            &job.thread,
            runner.backend().as_str(),
            initial_session_id,
        )
    };
    let (session_id, is_new) = match session_result {
        Ok(v) => v,
        Err(e) => {
            error!("[{}] session error: {e}", job.thread);
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_setup_failed",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    e.to_string(),
                ),
            );
            complete_setup_failure(ctx, &job, SESSION_SETUP_FAILURE).await;
            return;
        }
    };

    let work_dir = match std::fs::canonicalize(&ctx.assistant_dir) {
        Ok(path) => path.to_string_lossy().to_string(),
        Err(error) => {
            error!("[{}] assistant workspace error: {error}", job.thread);
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_setup_failed",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    error.to_string(),
                ),
            );
            complete_setup_failure(ctx, &job, SESSION_SETUP_FAILURE).await;
            return;
        }
    };

    if let Err(error) = ctx.cfg.backend_context_dir() {
        error!("[{}] assistant context error: {error:#}", job.thread);
        audit(
            ctx,
            ctx.audit.failed(
                "backend_setup_failed",
                job.row_id,
                &job.thread,
                Some(job.backend),
                error.to_string(),
            ),
        );
        complete_setup_failure(ctx, &job, SESSION_SETUP_FAILURE).await;
        return;
    }
    let composer = match Composer::load(&work_dir, &work_dir) {
        Ok(composer) => composer,
        Err(error) => {
            error!("[{}] prompt composition error: {error}", job.thread);
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_setup_failed",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    error.to_string(),
                ),
            );
            complete_setup_failure(ctx, &job, SESSION_SETUP_FAILURE).await;
            return;
        }
    };

    let prepared_images = if job.image_attachments.is_empty() {
        None
    } else {
        match prepare_images(ctx, &job).await {
            Ok(images) => Some(images),
            Err(error) => {
                warn!("[{}] {error:#}", job.thread);
                audit(
                    ctx,
                    ctx.audit.failed(
                        "image_preparation_failed",
                        job.row_id,
                        &job.thread,
                        Some(job.backend),
                        error.to_string(),
                    ),
                );
                let reply = "I could not use that image. Send up to 4 JPEG, PNG, or WebP images with a combined size under 6 MiB, then try again.";
                let delivery = record_and_deliver(ctx, &job, OutboundOrigin::Gateway, reply).await;
                report_delivery(
                    ctx,
                    &job,
                    delivery,
                    reply,
                    "image_preparation_failed",
                    "deliver image fallback",
                );
                return;
            }
        }
    };
    let mut attachment_paths = prepared_images
        .as_ref()
        .map_or_else(Vec::new, |images| images.paths().to_vec());
    let prepared_files = if job.file_attachments.is_empty() {
        None
    } else {
        match prepare_files(ctx, &job).await {
            Ok(files) => Some(files),
            Err(error) => {
                warn!("[{}] {error:#}", job.thread);
                audit(
                    ctx,
                    ctx.audit.failed(
                        "file_preparation_failed",
                        job.row_id,
                        &job.thread,
                        Some(job.backend),
                        error.to_string(),
                    ),
                );
                let reply = "I could not use that attachment. Send up to 4 files with a combined size under 8 MiB, then try again.";
                let delivery = record_and_deliver(ctx, &job, OutboundOrigin::Gateway, reply).await;
                report_delivery(
                    ctx,
                    &job,
                    delivery,
                    reply,
                    "file_preparation_failed",
                    "deliver file fallback",
                );
                return;
            }
        }
    };
    if let Some(files) = prepared_files.as_ref() {
        attachment_paths.extend_from_slice(files.paths());
    }

    let run = async {
        let mut session_id = session_id;
        let mut prompt = match conversation_prompt(ctx, &job, &composer, is_new) {
            Ok(prompt) => prompt,
            Err(error) => {
                return Err(RunError::Failed(format!(
                    "load canonical history for prompt: {error}"
                )));
            }
        };

        // Some backends let push choose the session id. Mark those before the
        // run so a post-create failure does not retry the same create call.
        if is_new && runner.mark_started_before_run() {
            let _ = ctx.store.lock().unwrap().mark_started(&job.thread, None);
        }
        audit(
            ctx,
            ctx.audit.backend_started(
                job.row_id,
                &job.thread,
                job.backend,
                is_new,
                prompt.rehydrated_messages,
            ),
        );
        info!(
            "[{}] sending message to {} (new_session={is_new}, rehydrated_messages={})",
            job.thread,
            runner.label(),
            prompt.rehydrated_messages
        );
        let mut result = dispatch(
            ctx,
            runner,
            &job,
            BackendTurn {
                session_id: &session_id,
                is_new,
                work_dir: &work_dir,
                prompt: &prompt,
                attachment_paths: &attachment_paths,
            },
        )
        .await;
        // If the session id already exists (e.g. left over from a previous run
        // or a different build), resume it instead of trying to create it again.
        if is_new {
            if let Err(RunError::Failed(msg)) = &result {
                if msg.to_lowercase().contains("already in use") {
                    warn!("[{}] session id already existed, resuming", job.thread);
                    prompt = conversation_prompt(ctx, &job, &composer, false).map_err(|error| {
                        RunError::Failed(format!("compose resumed prompt: {error}"))
                    })?;
                    result = dispatch(
                        ctx,
                        runner,
                        &job,
                        BackendTurn {
                            session_id: &session_id,
                            is_new: false,
                            work_dir: &work_dir,
                            prompt: &prompt,
                            attachment_paths: &attachment_paths,
                        },
                    )
                    .await;
                }
            }
        } else if matches!(&result, Err(RunError::SessionMissing(_))) {
            warn!(
                "[{}] backend session is missing; rotating and rehydrating",
                job.thread
            );
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_session_missing",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    "backend could not resume the stored session",
                ),
            );
            session_id = runner.initial_session_id();
            if let Err(error) = ctx.store.lock().unwrap().rotate(
                &job.thread,
                runner.backend().as_str(),
                session_id.clone(),
            ) {
                return Err(RunError::Failed(format!(
                    "rotate missing backend session: {error}"
                )));
            }
            prompt = match conversation_prompt(ctx, &job, &composer, true) {
                Ok(prompt) => prompt,
                Err(error) => {
                    return Err(RunError::Failed(format!(
                        "load canonical history for prompt: {error}"
                    )));
                }
            };
            if runner.mark_started_before_run() {
                let _ = ctx.store.lock().unwrap().mark_started(&job.thread, None);
            }
            audit(
                ctx,
                ctx.audit.backend_started(
                    job.row_id,
                    &job.thread,
                    job.backend,
                    true,
                    prompt.rehydrated_messages,
                ),
            );
            result = dispatch(
                ctx,
                runner,
                &job,
                BackendTurn {
                    session_id: &session_id,
                    is_new: true,
                    work_dir: &work_dir,
                    prompt: &prompt,
                    attachment_paths: &attachment_paths,
                },
            )
            .await;
        }
        result
    };
    let run = async {
        if let Some(refresh) = ctx.channel.typing_refresh() {
            let channel = ctx.channel.clone();
            let channel_id = channel.id();
            let target = job.target.clone();
            let thread = job.thread.clone();
            run_with_periodic_activity(run, refresh, move || {
                let channel = channel.clone();
                let target = target.clone();
                let thread = thread.clone();
                async move {
                    if let Err(e) = channel.send_typing(&target).await {
                        warn!("[{thread}] {channel_id} typing update failed: {e}");
                    }
                }
            })
            .await
        } else {
            run.await
        }
    };
    let mut run = Box::pin(run);
    tokio::pin!(interrupt);
    let result = tokio::select! {
        result = &mut run => Some(result),
        _ = &mut interrupt => None,
    };
    drop(run);
    drop(prepared_images);
    drop(prepared_files);

    match result {
        Some(Ok(out)) => {
            info!(
                "[{}] {} completed; reply_chars={}",
                job.thread,
                runner.label(),
                out.reply.chars().count()
            );
            audit(
                ctx,
                ctx.audit
                    .backend_completed(job.row_id, &job.thread, job.backend, &out.reply),
            );
            let outbound = match ctx.history.lock().unwrap().record_outbound(
                job.inbound_id,
                OutboundOrigin::Backend,
                Some(job.backend.as_str()),
                &out.reply,
            ) {
                Ok(outbound) => outbound,
                Err(error) => {
                    history_error(ctx, &job, "record backend reply", error);
                    return;
                }
            };
            if let Err(e) = ctx
                .store
                .lock()
                .unwrap()
                .mark_started(&job.thread, out.session_id.as_deref())
            {
                error!("[{}] session save error: {e}", job.thread);
                audit(
                    ctx,
                    ctx.audit.failed(
                        "backend_session_save_failed",
                        job.row_id,
                        &job.thread,
                        Some(job.backend),
                        e.to_string(),
                    ),
                );
                return;
            }
            let delivery = deliver_stored(ctx, &job, &outbound).await;
            if delivery.is_ok() {
                info!("[{}] reply sent via {}", job.thread, ctx.channel.id());
            }
            report_delivery(
                ctx,
                &job,
                delivery,
                &out.reply,
                "completed",
                "deliver backend reply",
            );
        }
        Some(Err(RunError::Timeout)) => {
            warn!("[{}] {} run timed out", job.thread, runner.label());
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_run_failed",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    format!("{} run timed out", runner.label()),
                ),
            );
            let reply = "That took too long and was stopped. Try again or simplify the request.";
            finish_run_with_gateway_reply(
                ctx,
                &job,
                reply,
                ReplyLabels {
                    record: "record timeout reply",
                    deliver: "deliver timeout reply",
                    completion: "timeout",
                },
            )
            .await;
        }
        None | Some(Err(RunError::Cancelled)) => {
            warn!("[{}] {} run interrupted", job.thread, runner.label());
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_run_interrupted",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    format!("{} run interrupted by user", runner.label()),
                ),
            );
            finish_run_with_gateway_reply(
                ctx,
                &job,
                "Stopped the current request.",
                ReplyLabels {
                    record: "record interrupted reply",
                    deliver: "deliver interrupted reply",
                    completion: "interrupted",
                },
            )
            .await;
        }
        Some(Err(RunError::Failed(msg) | RunError::SessionMissing(msg))) => {
            error!("[{}] {} error: {msg}", job.thread, runner.label());
            audit(
                ctx,
                ctx.audit.failed(
                    "backend_run_failed",
                    job.row_id,
                    &job.thread,
                    Some(job.backend),
                    msg.clone(),
                ),
            );
            let reply = format!("⚠️ {}", short(&msg));
            finish_run_with_gateway_reply(
                ctx,
                &job,
                &reply,
                ReplyLabels {
                    record: "record failure reply",
                    deliver: "deliver failure reply",
                    completion: "backend_failed",
                },
            )
            .await;
        }
    }
    review_changed_schedules(ctx, &job).await;
}

async fn review_changed_schedules(ctx: &Ctx, job: &Job) {
    let Some(destination) = &ctx.schedule_destination else {
        return;
    };
    let result = (|| {
        let catalog = crate::jobs::Catalog::load(&ctx.cfg)?;
        let mut ledger = crate::jobs::Ledger::open(&ctx.cfg.paths.database)?;
        let (_, events) = ledger.reconcile_schedule_reviews(
            &catalog,
            &destination.channel,
            &destination.target,
            crate::util::now_ms(),
        )?;
        let questions = ledger.schedule_review_questions(
            &job.approval_origin,
            &job.target,
            crate::util::now_ms(),
        )?;
        Ok::<_, anyhow::Error>((ledger, events, questions))
    })();
    let (mut ledger, _events, questions) = match result {
        Ok(result) => result,
        Err(error) => {
            error!(
                "[{}] reconcile schedule activation reviews: {error:#}",
                job.thread
            );
            audit(
                ctx,
                ctx.audit.failed(
                    "schedule_review_failed",
                    job.row_id,
                    &job.thread,
                    None,
                    error.to_string(),
                ),
            );
            return;
        }
    };
    audit_schedule_events(ctx, &mut ledger);
    for question in questions {
        let delivered =
            super::reply_to(ctx, &question.target, &question.render_correlated_text()).await;
        if let Err(error) = ledger.mark_schedule_question_delivery(
            &question.id,
            if delivered {
                crate::approval::DeliveryStatus::Delivered
            } else {
                crate::approval::DeliveryStatus::Failed
            },
            crate::util::now_ms(),
        ) {
            error!(
                "[{}] persist schedule review delivery: {error:#}",
                job.thread
            );
        }
        if !delivered {
            warn!("[{}] schedule review delivery failed", job.thread);
        }
    }
}

async fn interrupt(cancel: &mut watch::Receiver<i64>, row_id: i64) {
    while *cancel.borrow() != row_id {
        if cancel.changed().await.is_err() {
            pending::<()>().await;
        }
    }
}

enum VoicePreparationError {
    User {
        event: &'static str,
        reply: &'static str,
        detail: String,
    },
    History(anyhow::Error),
}

async fn prepare_voice(ctx: &Ctx, job: &Job) -> std::result::Result<String, VoicePreparationError> {
    let attachment = job
        .voice_attachment
        .as_ref()
        .expect("voice preparation requires an attachment");
    if attachment
        .file_size
        .is_some_and(|size| size > MAX_AUDIO_BYTES)
    {
        return Err(VoicePreparationError::User {
            event: "voice_too_large",
            reply: "That voice message is too large. The limit is 20 MB.",
            detail: "voice message exceeds the 20 MB limit".to_string(),
        });
    }
    let Some(voice) = &ctx.voice else {
        return Err(VoicePreparationError::User {
            event: "voice_not_configured",
            reply: "Voice transcription is unavailable right now. Please try again or send text instead.",
            detail: "the Gemini voice provider is not configured".to_string(),
        });
    };
    let clip = ctx
        .channel
        .download_voice(attachment)
        .await
        .map_err(|error| VoicePreparationError::User {
            event: "voice_download_failed",
            reply: "I could not download that voice message. Please try again or send text.",
            detail: format!("voice download failed: {error:#}"),
        })?;
    let transcript = voice
        .transcribe(clip)
        .await
        .map_err(|error| VoicePreparationError::User {
            event: "voice_transcription_failed",
            reply: "I could not transcribe that voice message. Please try again or send text.",
            detail: format!("voice transcription failed: {error:#}"),
        })?;
    ctx.history
        .lock()
        .unwrap()
        .replace_inbound_content(job.inbound_id, &transcript)
        .map_err(VoicePreparationError::History)?;
    Ok(transcript)
}

async fn prepare_images(ctx: &Ctx, job: &Job) -> Result<PreparedImages> {
    if job.image_attachments.len() > MAX_IMAGE_COUNT {
        anyhow::bail!("image message has more than {MAX_IMAGE_COUNT} attachments");
    }
    let declared_total = job
        .image_attachments
        .iter()
        .filter_map(|image| image.file_size)
        .try_fold(0usize, |total, size| total.checked_add(size))
        .context("image message size overflow")?;
    if declared_total > MAX_IMAGE_BYTES {
        anyhow::bail!("image message exceeds the 6 MiB limit");
    }

    let mut downloaded = Vec::with_capacity(job.image_attachments.len());
    let mut actual_total = 0usize;
    for attachment in &job.image_attachments {
        let image = ctx.channel.download_image(attachment).await?;
        actual_total = actual_total
            .checked_add(image.bytes.len())
            .context("image message size overflow")?;
        if actual_total > MAX_IMAGE_BYTES {
            anyhow::bail!("image message exceeds the 6 MiB limit");
        }
        downloaded.push(image);
    }
    PreparedImages::create(&ctx.cfg.paths.cache, downloaded)
}

async fn prepare_files(ctx: &Ctx, job: &Job) -> Result<PreparedFiles> {
    if job.file_attachments.len() > MAX_FILE_COUNT {
        anyhow::bail!("message has more than {MAX_FILE_COUNT} files");
    }
    let declared_total = job
        .file_attachments
        .iter()
        .filter_map(|file| file.file_size)
        .try_fold(0usize, |total, size| total.checked_add(size))
        .context("attachment size overflow")?;
    if declared_total > MAX_FILE_BYTES {
        anyhow::bail!("attachments exceed the 8 MiB limit");
    }

    let mut downloaded = Vec::with_capacity(job.file_attachments.len());
    let mut actual_total = 0usize;
    for attachment in &job.file_attachments {
        let file = ctx.channel.download_file(attachment).await?;
        actual_total = actual_total
            .checked_add(file.bytes.len())
            .context("attachment size overflow")?;
        if actual_total > MAX_FILE_BYTES {
            anyhow::bail!("attachments exceed the 8 MiB limit");
        }
        downloaded.push(DownloadedFile {
            bytes: file.bytes,
            filename: file.filename,
        });
    }
    PreparedFiles::create(&ctx.cfg.paths.cache, downloaded)
}

/// Error and completion labels for one gateway-authored reply flow.
struct ReplyLabels {
    record: &'static str,
    deliver: &'static str,
    completion: &'static str,
}

/// Records a gateway-authored reply, delivers it, and completes the row.
/// Shared by the timeout and failure arms of `handle`.
async fn finish_run_with_gateway_reply(ctx: &Ctx, job: &Job, reply: &str, labels: ReplyLabels) {
    let outbound = match ctx.history.lock().unwrap().record_outbound(
        job.inbound_id,
        OutboundOrigin::Gateway,
        Some(job.backend.as_str()),
        reply,
    ) {
        Ok(outbound) => outbound,
        Err(error) => return history_error(ctx, job, labels.record, error),
    };
    report_delivery(
        ctx,
        job,
        deliver_stored(ctx, job, &outbound).await,
        reply,
        labels.completion,
        labels.deliver,
    );
}

/// Audits `reply_sent` and completes the row when the reply reached the
/// channel; reports a canonical-history failure otherwise.
fn report_delivery(
    ctx: &Ctx,
    job: &Job,
    delivery: Result<DeliveryOutcome>,
    reply: &str,
    completion: &str,
    deliver_action: &str,
) {
    match delivery {
        Ok(DeliveryOutcome::Delivered | DeliveryOutcome::AlreadyDelivered) => {
            audit(
                ctx,
                ctx.audit.reply_sent(
                    job.row_id,
                    &job.thread,
                    &job.target,
                    Some(job.backend),
                    reply,
                ),
            );
            complete_job(ctx, job, completion);
        }
        Err(error) => history_error(ctx, job, deliver_action, error),
    }
}

pub(super) async fn run_with_periodic_activity<O, A, AF>(
    operation: O,
    refresh: Duration,
    mut activity: A,
) -> O::Output
where
    O: Future,
    A: FnMut() -> AF,
    AF: Future<Output = ()>,
{
    tokio::pin!(operation);
    let mut ticker = tokio::time::interval(refresh);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            output = &mut operation => return output,
            _ = ticker.tick() => {
                let activity = activity();
                tokio::pin!(activity);
                tokio::select! {
                    output = &mut operation => return output,
                    _ = &mut activity => {}
                }
            }
        }
    }
}

/// Setup failures are terminal for the current row. Try to tell the user what
/// happened, then complete the row so one bad setup step cannot wedge ack state.
/// The persisted notification is retried in bounded batches until delivery or
/// shutdown.
async fn complete_setup_failure(ctx: &Ctx, job: &Job, reply: &str) {
    #[cfg(test)]
    ctx.setup_failure_replies
        .lock()
        .unwrap()
        .push(reply.to_string());

    match record_and_deliver(ctx, job, OutboundOrigin::Gateway, reply).await {
        Ok(DeliveryOutcome::Delivered | DeliveryOutcome::AlreadyDelivered) => {
            audit(
                ctx,
                ctx.audit.reply_sent(
                    job.row_id,
                    &job.thread,
                    &job.target,
                    Some(job.backend),
                    reply,
                ),
            );
        }
        Err(error) => {
            history_error(ctx, job, "record setup failure reply", error);
            return;
        }
    }
    audit(ctx, ctx.audit.completed(job.row_id, "setup_failed"));
    complete_row(&ctx.store, &ctx.ack, &ctx.channel.store_key(), job.row_id);
}

fn complete_job(ctx: &Ctx, job: &Job, reason: &str) {
    audit(ctx, ctx.audit.completed(job.row_id, reason));
    complete_row(&ctx.store, &ctx.ack, &ctx.channel.store_key(), job.row_id);
}

/// Handles gateway-level slash commands before anything reaches the agent.
fn command(ctx: &Ctx, job: &Job) -> Option<String> {
    match job.text.trim().to_lowercase().as_str() {
        "/clear" | "/new" | "/reset" => match ctx.store.lock().unwrap().rotate(
            &job.thread,
            job.backend.as_str(),
            ctx.runners
                .get(&job.backend)
                .map(|r| r.initial_session_id())
                .unwrap_or_default(),
        ) {
            Ok(()) => {
                // A rotated session must not keep serving the old rpc child.
                ctx.pool.evict(&job.thread);
                Some("Started a fresh conversation.".to_string())
            }
            Err(_) => Some("Couldn't reset the conversation.".to_string()),
        },
        "/help" => Some(
            "Commands:\n/clear - start a fresh conversation\n/stop - stop the active request\n/help - this message"
                .to_string(),
        ),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeliveryOutcome {
    Delivered,
    AlreadyDelivered,
}

pub(super) async fn record_and_deliver(
    ctx: &Ctx,
    job: &Job,
    origin: OutboundOrigin,
    text: &str,
) -> Result<DeliveryOutcome> {
    let outbound = ctx.history.lock().unwrap().record_outbound(
        job.inbound_id,
        origin,
        Some(job.backend.as_str()),
        text,
    )?;
    deliver_stored(ctx, job, &outbound).await
}

/// Records an outbound reply and tries delivery once. Control-path replies use
/// this so a channel outage cannot block the polling loop indefinitely.
pub(super) async fn record_and_deliver_once(
    ctx: &Ctx,
    job: &Job,
    origin: OutboundOrigin,
    text: &str,
) -> Result<bool> {
    let mut outbound = ctx.history.lock().unwrap().record_outbound(
        job.inbound_id,
        origin,
        Some(job.backend.as_str()),
        text,
    )?;
    if outbound.status == DeliveryStatus::Delivered {
        return Ok(true);
    }
    let delivered = deliver_outbound_once(ctx, &job.target, &mut outbound).await?;
    ctx.history.lock().unwrap().mark_delivery(
        outbound.id,
        if delivered {
            DeliveryStatus::Delivered
        } else {
            DeliveryStatus::Failed
        },
    )?;
    Ok(delivered)
}

async fn deliver_stored(
    ctx: &Ctx,
    job: &Job,
    outbound: &OutboundMessage,
) -> Result<DeliveryOutcome> {
    if outbound.status == DeliveryStatus::Delivered {
        return Ok(DeliveryOutcome::AlreadyDelivered);
    }
    let mut outbound = outbound.clone();
    // PLAN §3 modality rule: exactly one copy, chosen deterministically. A
    // voice reply that fails to synthesize falls back to the text path below.
    if crate::voice::select_modality(job.reply_with_voice, &outbound.content)
        == crate::voice::ReplyModality::Voice
        && deliver_voice_reply(ctx, job, &outbound.content).await?
    {
        ctx.history
            .lock()
            .unwrap()
            .mark_delivery(outbound.id, DeliveryStatus::Delivered)?;
        return Ok(DeliveryOutcome::Delivered);
    }
    let semantics = ctx.channel.delivery_semantics();
    let mut attempt = 0;
    loop {
        attempt += 1;
        let delivered = deliver_outbound_once(ctx, &job.target, &mut outbound).await?;
        let status = if delivered {
            DeliveryStatus::Delivered
        } else {
            DeliveryStatus::Failed
        };
        ctx.history
            .lock()
            .unwrap()
            .mark_delivery(outbound.id, status)?;
        if delivered {
            return Ok(DeliveryOutcome::Delivered);
        }
        audit(
            ctx,
            ctx.audit.reply_failed(
                job.row_id,
                &job.thread,
                &job.target,
                Some(job.backend),
                "stored outbound delivery attempt failed",
            ),
        );
        if attempt < semantics.retry_attempts {
            warn!(
                "delivery attempt {attempt}/{} failed; retrying stored outbound",
                semantics.retry_attempts
            );
            #[cfg(not(test))]
            tokio::time::sleep(semantics.retry_delay).await;
        } else {
            warn!("delivery attempts exhausted; stored outbound remains failed and will retry");
            attempt = 0;
            #[cfg(not(test))]
            tokio::time::sleep(semantics.exhausted_retry_delay).await;
        }
        #[cfg(test)]
        tokio::task::yield_now().await;
    }
}

async fn deliver_outbound_once(
    ctx: &Ctx,
    target: &str,
    outbound: &mut OutboundMessage,
) -> Result<bool> {
    let chunks = ctx
        .channel
        .outbound_chunks(&outbound.content, &ctx.reply_marker);
    if outbound.delivery_chunk_index > chunks.len() {
        anyhow::bail!(
            "outbound {} has invalid delivery chunk index {} for {} chunks",
            outbound.id,
            outbound.delivery_chunk_index,
            chunks.len()
        );
    }
    for (index, chunk) in chunks
        .iter()
        .enumerate()
        .skip(outbound.delivery_chunk_index)
    {
        if let Err(error) = super::send_reply_chunk(ctx, target, chunk).await {
            error!(
                "outbound {} chunk {index} send error to {target}: {error}",
                outbound.id
            );
            return Ok(false);
        }
        let next_chunk = index + 1;
        ctx.history
            .lock()
            .unwrap()
            .checkpoint_delivery(outbound.id, next_chunk)?;
        outbound.delivery_chunk_index = next_chunk;
    }
    Ok(true)
}

/// Delivers a reply as a native voice message (PLAN §6). Returns `true` when
/// the voice message was delivered; `false` means the caller must fall back to
/// the text path.
async fn deliver_voice_reply(ctx: &Ctx, job: &Job, text: &str) -> Result<bool> {
    let Some(voice) = &ctx.voice else {
        return Ok(false);
    };
    let clip = match voice.synthesize(text).await {
        Ok(clip) => clip,
        Err(error) => {
            warn!(
                "[{}] voice reply synthesis failed; falling back to text: {error:#}",
                job.thread
            );
            return Ok(false);
        }
    };
    #[cfg(test)]
    {
        ctx.sent_voice_replies
            .lock()
            .unwrap()
            .push((job.target.clone(), clip.bytes));
        Ok(true)
    }
    #[cfg(not(test))]
    match ctx.channel.send_voice(&job.target, &clip).await {
        Ok(()) => Ok(true),
        Err(error) => {
            warn!(
                "[{}] voice reply delivery failed; falling back to text: {error:#}",
                job.thread
            );
            Ok(false)
        }
    }
}

fn history_error(ctx: &Ctx, job: &Job, action: &str, error: anyhow::Error) {
    error!(
        "[{}] canonical history {action} failed: {error}; refusing unrecorded delivery",
        job.thread
    );
    audit(
        ctx,
        ctx.audit.failed(
            "message_history_failed",
            job.row_id,
            &job.thread,
            Some(job.backend),
            format!("{action}: {error}"),
        ),
    );
}

/// Extracts a short, user-facing reason from an error message.
fn short(msg: &str) -> String {
    let s = msg.rsplit(": ").next().unwrap_or(msg).trim();
    if s.is_empty() {
        "couldn't reach the agent".to_string()
    } else {
        s.to_string()
    }
}

fn conversation_prompt(
    ctx: &Ctx,
    job: &Job,
    composer: &Composer,
    include_history: bool,
) -> Result<ComposedPrompt> {
    let messages = if include_history {
        ctx.history.lock().unwrap().recent_messages_before(
            ctx.channel.id(),
            &job.thread,
            job.inbound_id,
            crate::prompt::MAX_HISTORY_MESSAGES,
        )?
    } else {
        Vec::new()
    };
    let mut prompt = composer.conversation(
        ctx.channel.id(),
        &job.thread,
        if job.reply_with_voice {
            "voice"
        } else {
            "text"
        },
        &messages,
        &job.text,
    );
    if let Some(prompt_file) = job.prompt_file.as_deref() {
        prompt
            .instructions
            .push_str(&channel_instructions(&ctx.cfg.assistant_dir, prompt_file)?);
    }
    Ok(prompt)
}

/// Reads a per-channel system-prompt file (PLAN §3) and renders it as a
/// system-instruction section. The path is validated at config load to stay a
/// plain relative path under `assistant_root`.
pub(super) fn channel_instructions(assistant_dir: &str, prompt_file: &str) -> Result<String> {
    let prompt_file = prompt_file.trim();
    let path = std::path::Path::new(assistant_dir).join(prompt_file);
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("read channel prompt file {}", path.display()))?;
    Ok(format!(
        "\n\n# Channel instructions ({prompt_file})\n\n{}",
        content.trim_end()
    ))
}

fn backend_request<'a>(
    session_id: &'a str,
    is_new: bool,
    work_dir: &'a str,
    prompt: &'a ComposedPrompt,
    images: &'a [PathBuf],
    model: Option<&'a str>,
    thinking: Option<&'a str>,
) -> Request<'a> {
    Request {
        session_id,
        is_new,
        work_dir,
        instructions: &prompt.instructions,
        prompt: &prompt.content,
        images,
        model,
        thinking,
    }
}

/// One backend turn's inputs, bundled so [`dispatch`] stays readable.
struct BackendTurn<'a> {
    session_id: &'a str,
    is_new: bool,
    work_dir: &'a str,
    prompt: &'a ComposedPrompt,
    attachment_paths: &'a [PathBuf],
}

/// The inputs to the `pi --mode rpc` spec builder. Extracted from [`dispatch`]
/// so the Discord worker and the voice endpoint build byte-identical specs and
/// cannot drift in prompt, gate, or session handling.
pub(super) struct RpcSpecInput<'a> {
    pub session_id: &'a str,
    pub is_new: bool,
    pub work_dir: &'a str,
    pub prompt: &'a ComposedPrompt,
    pub images: &'a [PathBuf],
    pub model: Option<&'a str>,
    pub thinking: Option<&'a str>,
    /// True when the approval gate should load. The Discord worker passes
    /// `ctx.cfg.approval_gate_enabled()`; the voice endpoint always passes
    /// `false` (contract §7).
    pub gate: bool,
}

/// Builds one thread's rpc child spec. The only backend with an rpc runner is
/// Pi, so the binary is resolved from `AgentBackend::Pi` unconditionally. It
/// takes `&Config` rather than `&Ctx` because the voice endpoint has no
/// `Channel` and so cannot build a worker `Ctx`.
pub(super) fn rpc_spec(cfg: &Config, input: RpcSpecInput<'_>) -> crate::pi::PiRpcSpec {
    let RpcSpecInput {
        session_id,
        is_new,
        work_dir,
        prompt,
        images,
        model,
        thinking,
        gate,
    } = input;
    crate::pi::PiRpcSpec {
        bin: cfg.agent_bin(AgentBackend::Pi).to_string(),
        session_id: session_id.to_string(),
        is_new,
        work_dir: work_dir.to_string(),
        instructions: prompt.instructions.clone(),
        model: model.map(str::to_string),
        thinking: thinking.map(str::to_string),
        images: images.to_vec(),
        // The gate only loads when tools are configured; an ungated rpc child
        // can never prompt unexpectedly (PLAN §7).
        env: if gate {
            vec![("GATE_TOOLS".to_string(), cfg.approval_gated_tools.join(","))]
        } else {
            Vec::new()
        },
        extension: if gate {
            cfg.approval_gate_extension_path()
        } else {
            None
        },
    }
}

/// Runs one backend turn: the per-thread rpc pool for `Runner::PiRpc` (PLAN §7
/// step 2), the one-shot `Runner::run` for every other runner. The audit
/// events, prompt composition, and session persistence around the call are
/// identical either way, so flipping `pi_mode` cannot change the recorded rows.
async fn dispatch(
    ctx: &Ctx,
    runner: &Runner,
    job: &Job,
    turn: BackendTurn<'_>,
) -> Result<RunOutput, RunError> {
    let BackendTurn {
        session_id,
        is_new,
        work_dir,
        prompt,
        attachment_paths,
    } = turn;
    if !runner.is_rpc() {
        return runner
            .run(
                backend_request(
                    session_id,
                    is_new,
                    work_dir,
                    prompt,
                    attachment_paths,
                    job.model.as_deref(),
                    job.thinking.as_deref(),
                ),
                ctx.run_timeout,
            )
            .await;
    }
    let spec = rpc_spec(
        &ctx.cfg,
        RpcSpecInput {
            session_id,
            is_new,
            work_dir,
            prompt,
            images: attachment_paths,
            model: job.model.as_deref(),
            thinking: job.thinking.as_deref(),
            gate: ctx.cfg.approval_gate_enabled(),
        },
    );
    let mut handler = ApprovalHandler {
        ctx,
        job,
        timeout: ctx
            .cfg
            .approval_timeout_dur()
            .unwrap_or_else(|_| Duration::from_secs(600)),
    };
    ctx.pool
        .run(
            &job.thread,
            spec,
            &prompt.content,
            ctx.run_timeout,
            &mut handler,
            None,
        )
        .await
}

/// Translates a child's extension-UI request into push's durable question flow
/// and parks the run until the user answers (PLAN §7 / DESIGN §2).
///
/// It borrows the worker's `Ctx` and `Job`: creating the question, delivering
/// it, and polling `take_answer` are all short-lived operations that must not
/// hold the history lock across an await.
struct ApprovalHandler<'a> {
    ctx: &'a Ctx,
    job: &'a Job,
    timeout: Duration,
}

impl UiHandler for ApprovalHandler<'_> {
    fn handle<'a>(&'a mut self, request: &'a UiRequest) -> UiFuture<'a> {
        Box::pin(async move { self.ask(request).await })
    }
}

impl ApprovalHandler<'_> {
    async fn ask(&mut self, request: &UiRequest) -> UiResponse {
        let (prompt, choices, confirm) = match request {
            UiRequest::Select { title, options, .. } => {
                // A `select` carries the option strings verbatim; the value is
                // the string itself so push can map the number back without
                // guessing.
                let choices = options
                    .iter()
                    .map(|option| Choice {
                        label: option.clone(),
                        value: option.clone(),
                    })
                    .collect::<Vec<_>>();
                if !(2..=crate::approval::MAX_CHOICES).contains(&choices.len()) {
                    warn!(
                        "[{}] approval select had {} options; cancelling",
                        self.job.thread,
                        choices.len()
                    );
                    return UiResponse::Cancelled;
                }
                (title.clone(), choices, false)
            }
            UiRequest::Confirm { title, .. } => (
                title.clone(),
                vec![
                    Choice {
                        label: "Allow".to_string(),
                        value: "allow".to_string(),
                    },
                    Choice {
                        label: "Deny".to_string(),
                        value: "deny".to_string(),
                    },
                ],
                true,
            ),
            // `input`/`editor`/fire-and-forget methods: push has no UI for them,
            // and the spike confirms replying is protocol noise. Cancel.
            UiRequest::Other { .. } => return UiResponse::Cancelled,
        };

        let now = now_ms();
        let expiry = now.saturating_add(self.timeout.as_millis() as i64);
        let question = match Question::new(
            self.job.approval_origin.clone(),
            self.job.target.clone(),
            prompt,
            choices,
            expiry,
        ) {
            Ok(question) => question,
            Err(error) => {
                warn!(
                    "[{}] could not build approval question: {error}",
                    self.job.thread
                );
                return UiResponse::Cancelled;
            }
        };
        if let Err(error) =
            self.ctx
                .history
                .lock()
                .unwrap()
                .create_question(&question, Some(self.job.row_id), now)
        {
            warn!(
                "[{}] could not persist approval question: {error}",
                self.job.thread
            );
            return UiResponse::Cancelled;
        }

        let text = question.render_correlated_text();
        if !super::reply_to(self.ctx, &self.job.target, &text).await {
            let _ = self
                .ctx
                .history
                .lock()
                .unwrap()
                .cancel_question(&question.id, now_ms());
            warn!(
                "[{}] approval question delivery failed; cancelling the dialog",
                self.job.thread
            );
            return UiResponse::Cancelled;
        }
        let _ = self
            .ctx
            .history
            .lock()
            .unwrap()
            .mark_question_delivery(&question.id, crate::approval::DeliveryStatus::Delivered);

        let deadline = tokio::time::Instant::now() + self.timeout;
        loop {
            let answer = self
                .ctx
                .history
                .lock()
                .unwrap()
                .take_answer(&question.id, now_ms())
                .ok()
                .flatten();
            if let Some(answer) = answer {
                return if confirm {
                    UiResponse::Confirm(answer.value.eq_ignore_ascii_case("allow"))
                } else {
                    UiResponse::Select(answer.value)
                };
            }
            if tokio::time::Instant::now() >= deadline {
                let _ = self
                    .ctx
                    .history
                    .lock()
                    .unwrap()
                    .cancel_question(&question.id, now_ms());
                return UiResponse::Cancelled;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}
