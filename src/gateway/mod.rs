//! The message loop: poll, filter, and route each message to a per-thread
//! worker task that runs an agent backend and sends the reply.
//!
//! Design: each enabled channel has an independent loop, acknowledgement state,
//! and queue map. Channel loops share the durable store and history behind
//! short-lived locks. Each channel-qualified thread gets its own worker task,
//! which serializes that thread's messages.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{error, info, warn};

use crate::agent::Runner;
use crate::approval::{AnswerOrigin, AnswerOutcome};
use crate::audit::{AuditEvent, AuditLog};
use crate::channel::{Channel, InboundFile, InboundImage, InboundVoice, RawMessage};
use crate::config::{AgentBackend, ChannelSpec, Config, PrimaryDeliveryConfig};
use crate::history::{History, OutboundOrigin};
use crate::jobs;
use crate::store::Store;
use crate::util::now_ms;
use crate::voice::Voice;

const QUEUE_DEPTH: usize = 32;

#[cfg(test)]
type SentVoiceReplies = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

#[derive(Clone)]
struct Job {
    row_id: i64,
    inbound_id: i64,
    thread: String,
    target: String,
    backend: AgentBackend,
    /// Per-channel model/thinking resolved by the route (PLAN §3).
    model: Option<String>,
    thinking: Option<String>,
    /// Per-channel system-prompt file relative to assistant_root (PLAN §3).
    prompt_file: Option<String>,
    text: String,
    reply_with_voice: bool,
    voice_attachment: Option<InboundVoice>,
    image_attachments: Vec<InboundImage>,
    file_attachments: Vec<InboundFile>,
    approval_origin: AnswerOrigin,
}

/// Shared, cheaply cloneable context handed to each worker task.
#[derive(Clone)]
struct Ctx {
    cfg: Config,
    store: Arc<Mutex<Store>>,
    history: Arc<Mutex<History>>,
    ack: Arc<Mutex<AckState>>,
    runners: Arc<HashMap<AgentBackend, Runner>>,
    /// Per-thread long-lived `pi --mode rpc` children, shared by every channel
    /// like `runners` (DESIGN §1). Unused while `pi_mode = "print"`.
    pool: Arc<crate::pi::Pool>,
    channel: Channel,
    run_timeout: Duration,
    reply_marker: String,
    assistant_dir: String,
    audit: Arc<AuditLog>,
    voice: Option<Voice>,
    schedule_destination: Option<PrimaryDestination>,
    #[cfg(test)]
    setup_failure_replies: Arc<Mutex<Vec<String>>>,
    #[cfg(test)]
    sent_replies: Arc<Mutex<Vec<(String, String)>>>,
    #[cfg(test)]
    sent_attachments: Arc<Mutex<Vec<(String, String)>>>,
    #[cfg(test)]
    sent_voice_replies: SentVoiceReplies,
    #[cfg(test)]
    send_failures_remaining: Arc<Mutex<usize>>,
    #[cfg(test)]
    send_failure_after: Arc<Mutex<Option<usize>>>,
}

pub struct Gateway {
    channel: Channel,
    store: Arc<Mutex<Store>>,
    ack: Arc<Mutex<AckState>>,
    ctx: Ctx,
    cfg: Config,
    poll_interval: Duration,
    /// Set by [`Gateway::route`]/[`Gateway::stop`] during a tick; read and
    /// cleared by [`Gateway::tick`] to drive the adaptive poll interval.
    channel_active: bool,
    queues: HashMap<String, WorkerQueue>,
    handles: Vec<JoinHandle<()>>,
}

pub struct GatewayGroup {
    gateways: Vec<Gateway>,
    primary_delivery: Option<PrimaryDeliveryConfig>,
    cfg: Config,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryDestination {
    pub channel: String,
    pub target: String,
}

#[derive(Default)]
struct AckState {
    in_flight: BTreeSet<i64>,
    deferred: BTreeSet<i64>,
    persisting: BTreeSet<i64>,
    completed: BTreeSet<i64>,
}

struct WorkerQueue {
    jobs: mpsc::Sender<Job>,
    state: Arc<Mutex<WorkerState>>,
    cancel: watch::Sender<i64>,
}

struct WorkerState {
    pending: VecDeque<Job>,
    current_row: Option<i64>,
    retained_rows: BTreeSet<i64>,
}

impl GatewayGroup {
    pub fn new(cfg: Config) -> Result<Self> {
        jobs::Ledger::capture_legacy_schedule_baseline(&cfg)?;
        let enabled = cfg.enabled_channels()?;
        let store = Arc::new(Mutex::new(Store::open(&cfg.paths)?));
        let history = Arc::new(Mutex::new(
            History::open(&cfg.paths.database).with_context(|| {
                format!(
                    "open canonical history database {}",
                    cfg.paths.database.display()
                )
            })?,
        ));
        let runners = Arc::new(runners(&cfg));
        let pool = Arc::new(crate::pi::Pool::new(cfg.rpc_pool_config()?));
        let audit_lock = Arc::new(Mutex::new(()));
        let mut gateways = Vec::with_capacity(enabled.len());
        for spec in enabled {
            gateways.push(Gateway::new_with_shared(
                cfg.clone(),
                spec,
                store.clone(),
                history.clone(),
                runners.clone(),
                pool.clone(),
                audit_lock.clone(),
            )?);
        }
        let mut group = Self {
            gateways,
            primary_delivery: cfg.primary_delivery.clone(),
            cfg,
        };
        let destination = group.primary_destination().ok();
        let mut ledger = jobs::Ledger::open(&group.cfg.paths.database)?;
        if let Some(destination) = destination {
            let catalog = jobs::Catalog::load(&group.cfg)?;
            ledger.recover_answered_schedule_reviews(&group.cfg, now_ms())?;
            ledger.reconcile_schedule_reviews(
                &catalog,
                &destination.channel,
                &destination.target,
                now_ms(),
            )?;
            for gateway in &mut group.gateways {
                gateway.ctx.schedule_destination = Some(destination.clone());
            }
        } else {
            ledger.settle_legacy_schedule_migration_without_destination()?;
        }
        if let Some(gateway) = group.gateways.first() {
            audit_schedule_events(&gateway.ctx, &mut ledger);
        }
        Ok(group)
    }

    pub fn primary_destination(&self) -> Result<PrimaryDestination> {
        let configured = self
            .primary_delivery
            .as_ref()
            .context("primary delivery is not configured")?;
        let spec = self
            .cfg
            .resolve_channel_spec(&configured.channel)
            .context("invalid primary delivery channel")?;
        let gateway = self
            .gateways
            .iter()
            .find(|gateway| gateway.channel.store_key() == spec.store_key())
            .with_context(|| {
                format!(
                    "primary delivery channel {:?} is not enabled in channels",
                    configured.channel
                )
            })?;
        let target = gateway
            .channel
            .primary_target(&configured.target)
            .context("invalid primary delivery target")?;
        Ok(PrimaryDestination {
            channel: spec.store_key(),
            target,
        })
    }

    #[cfg(test)]
    pub async fn deliver_primary(&self, text: &str) -> Result<PrimaryDestination> {
        if text.trim().is_empty() {
            anyhow::bail!("primary delivery text cannot be empty");
        }
        let destination = self.primary_destination()?;
        let gateway = self
            .gateways
            .iter()
            .find(|gateway| gateway.channel.store_key() == destination.channel)
            .context("resolved primary delivery channel is unavailable")?;
        if !reply_to(&gateway.ctx, &destination.target, text).await {
            anyhow::bail!(
                "primary delivery to {} target {:?} failed",
                destination.channel,
                destination.target
            );
        }
        Ok(destination)
    }

    /// Cancels approval questions left pending by a previous process and tells
    /// each origin: the parked rpc run did not survive the restart, so the
    /// question cannot be resumed (DESIGN §4). Without this a stale pending row
    /// would linger and a late numbered reply would answer a dead run.
    async fn recover_pending_questions(&self) {
        let pending = {
            let Some(gateway) = self.gateways.first() else {
                return;
            };
            match gateway.ctx.history.lock().unwrap().pending_questions() {
                Ok(pending) => pending,
                Err(error) => {
                    warn!("approval restart recovery could not read pending questions: {error:#}");
                    return;
                }
            }
        };
        for question in pending {
            let Some(gateway) = self.gateways.iter().find(|gateway| {
                gateway.channel.store_key() == question.thread_key
                    || gateway.channel.id() == question.channel
            }) else {
                continue;
            };
            if let Err(error) = gateway
                .ctx
                .history
                .lock()
                .unwrap()
                .cancel_question(&question.id, now_ms())
            {
                warn!(
                    "could not cancel pending approval {}: {error:#}",
                    question.id
                );
                continue;
            }
            // Do not replay the abandoned run: advance the channel cursor past
            // its inbound row (and record it completed) so the parked trigger is
            // not re-polled and re-executed. Rows after it are still processed.
            if let Some(run_row_id) = question.run_row_id {
                let key = gateway.channel.store_key();
                let completed = gateway
                    .ctx
                    .store
                    .lock()
                    .unwrap()
                    .mark_row_completed(&key, run_row_id);
                let advanced = gateway
                    .ctx
                    .store
                    .lock()
                    .unwrap()
                    .set_cursor(&key, run_row_id);
                if let Err(error) = completed.and(advanced) {
                    warn!(
                        "could not guard abandoned approval run row {run_row_id} on {key}: {error:#}"
                    );
                }
            }
            let notice = "The assistant restarted before that request was approved, so it was cancelled. Ask again if you still want it.";
            if !reply_to(&gateway.ctx, &question.target, notice).await {
                warn!(
                    "approval restart notice delivery failed for {}",
                    question.id
                );
            }
        }
    }

    pub async fn run(self) -> Result<()> {
        self.run_until(shutdown_signal()).await
    }

    async fn run_until<S>(self, shutdown: S) -> Result<()>
    where
        S: Future<Output = ()>,
    {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let scheduler = match self.primary_destination() {
            Ok(destination) => {
                jobs::Scheduler::new(self.cfg.clone(), destination.channel, destination.target)
            }
            Err(error) => {
                warn!("scheduled jobs disabled: {error:#}");
                jobs::Scheduler::delivery_only(self.cfg.clone())
            }
        };
        let contexts = self
            .gateways
            .iter()
            .map(|gateway| (gateway.channel.store_key(), gateway.ctx.clone()))
            .collect::<HashMap<_, _>>();
        self.recover_pending_questions().await;
        // Every channel shares one pool, so one janitor reaps idle children for
        // the whole group (DESIGN §1). Only meaningful in rpc mode.
        let pool = if self.cfg.pi_rpc_enabled() {
            self.gateways
                .first()
                .map(|gateway| gateway.ctx.pool.clone())
        } else {
            None
        };
        let receiver = shutdown_rx.clone();
        let scheduler =
            tokio::spawn(async move { run_scheduler(scheduler, contexts, receiver).await });
        let janitor = pool.map(|pool| {
            let receiver = shutdown_rx.clone();
            tokio::spawn(async move { run_pool_janitor(pool, receiver).await })
        });
        let mut tasks = JoinSet::new();
        for gateway in self.gateways {
            let channel = gateway.channel.id();
            let receiver = shutdown_rx.clone();
            tasks.spawn(async move {
                gateway
                    .run_with_shutdown(receiver)
                    .await
                    .with_context(|| format!("{channel} channel stopped"))?;
                Ok(channel)
            });
        }
        drop(shutdown_rx);
        let result = coordinate_channel_tasks(tasks, shutdown_tx, shutdown).await;
        if let Some(janitor) = janitor {
            janitor.abort();
            let _ = janitor.await;
        }
        match scheduler.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => error!("job scheduler stopped: {error:#}"),
            Err(error) => error!("job scheduler task failed: {error}"),
        }
        result
    }
}

/// Reaps idle `pi --mode rpc` children on a fixed cadence. Runs on the group's
/// shared pool; skips any thread with a run in flight.
async fn run_pool_janitor(pool: Arc<crate::pi::Pool>, mut shutdown: watch::Receiver<bool>) {
    let idle = pool.idle_timeout();
    let cadence = (idle / 2).clamp(Duration::from_secs(1), Duration::from_secs(60));
    let mut ticker = tokio::time::interval(cadence);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // The first tick is immediate; skip it so startup does not sweep an
    // (empty) pool before anything has run.
    ticker.tick().await;
    loop {
        let sweep = async {
            ticker.tick().await;
            pool.sweep_idle().await;
        };
        if wait_for_channel_shutdown_or(&mut shutdown, sweep)
            .await
            .is_none()
        {
            break;
        }
    }
}

async fn run_scheduler(
    mut scheduler: jobs::Scheduler,
    contexts: HashMap<String, Ctx>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let contexts = Arc::new(contexts);
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            _ = ticker.tick() => {
                let contexts = contexts.clone();
                let audit_contexts = contexts.clone();
                let result = scheduler.tick(now_ms(), move |channel, target, text, start_chunk, progress| {
                    let contexts = contexts.clone();
                    async move {
                        let Some(ctx) = contexts.get(&channel) else {
                            return jobs::DeliveryAttempt::failed(
                                start_chunk,
                                format!("persisted delivery channel {channel:?} is not enabled"),
                            );
                        };
                        let target = match ctx.channel.primary_target(&target) {
                            Ok(target) => target,
                            Err(error) => {
                                return jobs::DeliveryAttempt::failed(
                                    start_chunk,
                                    format!(
                                        "persisted delivery target is no longer allowlisted: {error}"
                                    ),
                                )
                            }
                        };
                        scheduled_reply_to(ctx, &target, &text, start_chunk, progress).await
                    }
                }).await;
                if let Some(ctx) = audit_contexts.values().next() {
                    scheduler.take_schedule_events();
                    match jobs::Ledger::open(&ctx.cfg.paths.database) {
                        Ok(mut ledger) => audit_schedule_events(ctx, &mut ledger),
                        Err(error) => {
                            error!("open schedule audit outbox: {error:#}");
                        }
                    }
                }
                if let Err(error) = result {
                    error!("job scheduler tick failed: {error:#}");
                }
            }
        }
    }
    scheduler.shutdown().await;
    Ok(())
}

async fn coordinate_channel_tasks<S>(
    mut tasks: JoinSet<Result<&'static str>>,
    shutdown_tx: watch::Sender<bool>,
    shutdown: S,
) -> Result<()>
where
    S: Future<Output = ()>,
{
    let mut active = tasks.len();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            result = tasks.join_next(), if active > 0 => {
                active -= 1;
                match result {
                    Some(Ok(Ok(channel))) => warn!("{channel} channel exited before shutdown"),
                    Some(Ok(Err(error))) => error!("{error:#}"),
                    Some(Err(error)) => error!("channel task failed: {error}"),
                    None => {}
                }
                if active == 0 {
                    anyhow::bail!("all enabled reply channels stopped");
                }
            }
        }
    }

    let _ = shutdown_tx.send(true);
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(channel)) => info!("{channel} channel shut down cleanly"),
            Ok(Err(error)) => error!("{error:#}"),
            Err(error) => error!("channel task failed during shutdown: {error}"),
        }
    }
    Ok(())
}

impl Gateway {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(cfg: Config) -> Result<Self> {
        let store = Arc::new(Mutex::new(Store::open(&cfg.paths)?));
        let history = Arc::new(Mutex::new(
            History::open(&cfg.paths.database).with_context(|| {
                format!(
                    "open canonical history database {}",
                    cfg.paths.database.display()
                )
            })?,
        ));
        let runners = Arc::new(runners(&cfg));
        let pool = Arc::new(crate::pi::Pool::new(cfg.rpc_pool_config()?));
        let spec = cfg
            .enabled_channels()?
            .into_iter()
            .next()
            .context("at least one reply channel must be enabled")?;
        Self::new_with_shared(
            cfg,
            spec,
            store,
            history,
            runners,
            pool,
            Arc::new(Mutex::new(())),
        )
    }

    fn new_with_shared(
        cfg: Config,
        spec: ChannelSpec,
        store: Arc<Mutex<Store>>,
        history: Arc<Mutex<History>>,
        runners: Arc<HashMap<AgentBackend, Runner>>,
        pool: Arc<crate::pi::Pool>,
        audit_lock: Arc<Mutex<()>>,
    ) -> Result<Self> {
        let ack = Arc::new(Mutex::new(AckState::default()));
        let channel = Channel::new_for(&cfg, &spec)?;
        let audit = Arc::new(AuditLog::with_lock(
            cfg.paths.audit.clone(),
            cfg.audit_log_content,
            channel.id(),
            audit_lock,
        ));
        let ctx = Ctx {
            cfg: cfg.clone(),
            store: store.clone(),
            history,
            ack: ack.clone(),
            runners,
            pool,
            channel: channel.clone(),
            run_timeout: cfg.run_timeout_dur()?,
            reply_marker: crate::channel::REPLY_MARKER.to_string(),
            assistant_dir: cfg.assistant_dir.clone(),
            audit,
            schedule_destination: None,
            #[cfg(not(test))]
            voice: Voice::from_config(&cfg),
            #[cfg(test)]
            voice: None,
            #[cfg(test)]
            setup_failure_replies: Arc::new(Mutex::new(Vec::new())),
            #[cfg(test)]
            sent_replies: Arc::new(Mutex::new(Vec::new())),
            #[cfg(test)]
            sent_attachments: Arc::new(Mutex::new(Vec::new())),
            #[cfg(test)]
            sent_voice_replies: Arc::new(Mutex::new(Vec::new())),
            #[cfg(test)]
            send_failures_remaining: Arc::new(Mutex::new(0)),
            #[cfg(test)]
            send_failure_after: Arc::new(Mutex::new(None)),
        };
        let poll_interval = cfg.poll_interval_dur()?;
        Ok(Self {
            channel,
            store,
            ack,
            ctx,
            cfg,
            poll_interval,
            channel_active: false,
            queues: HashMap::new(),
            handles: Vec::new(),
        })
    }

    /// Delivers a durable approval question. Test-only seeding for the active
    /// inbound answer-resolution flow; nothing in production asks yet.
    #[cfg(test)]
    pub async fn ask_user(&self, question: crate::approval::Question) -> Result<String> {
        use crate::approval::DeliveryStatus as ApprovalDeliveryStatus;

        let id = question.id.clone();
        self.ctx
            .history
            .lock()
            .unwrap()
            .create_question(&question, None, now_ms())?;
        let delivered = reply_to(&self.ctx, &question.target, &question.render_text()).await;
        self.ctx.history.lock().unwrap().mark_question_delivery(
            &id,
            if delivered {
                ApprovalDeliveryStatus::Delivered
            } else {
                ApprovalDeliveryStatus::Failed
            },
        )?;
        if !delivered {
            anyhow::bail!("approval question {id} could not be delivered");
        }
        Ok(id)
    }

    async fn run_with_shutdown(mut self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        if let Some(result) = wait_for_channel_shutdown_or(&mut shutdown, self.skip_backlog()).await
        {
            result?;
            let active_interval = self.poll_interval;
            let idle_interval = self.cfg.poll_idle_interval_dur()?;
            let max_interval = self.cfg.poll_max_interval_dur()?;
            let mut backoff = PollBackoff::new(active_interval, idle_interval, max_interval);
            info!(
                "{} gateway running; polling every {:?} active / {:?}-{:?} idle",
                self.channel.id(),
                active_interval,
                idle_interval,
                max_interval
            );

            loop {
                let sleep_for = backoff.next();
                let poll = async {
                    tokio::time::sleep(sleep_for).await;
                    self.tick().await
                };
                let Some(active) = wait_for_channel_shutdown_or(&mut shutdown, poll).await else {
                    break;
                };
                backoff.record(active);
            }
        }
        info!(
            "shutdown requested for {}, draining in-flight runs",
            self.channel.id()
        );

        // Dropping the senders lets each worker finish its current job, drain
        // its queue, and exit. Then wait, bounded by a grace period.
        self.queues.clear();
        self.drain_workers().await;
        info!("shut down cleanly");
        Ok(())
    }

    async fn drain_workers(&mut self) {
        let deadline = tokio::time::Instant::now() + self.channel.shutdown_semantics().worker_grace;
        let mut workers = std::mem::take(&mut self.handles).into_iter();
        while let Some(mut worker) = workers.next() {
            if tokio::time::timeout_at(deadline, &mut worker)
                .await
                .is_err()
            {
                worker.abort();
                let _ = worker.await;
                for worker in workers {
                    worker.abort();
                    let _ = worker.await;
                }
                warn!("shutdown grace expired; remaining channel workers were aborted");
                return;
            }
        }
    }

    async fn skip_backlog(&self) -> Result<()> {
        // Durable state is keyed by the channel's store key, which a
        // multi-instance channel may widen beyond its provider id.
        let channel_id = self.channel.store_key();
        if self
            .store
            .lock()
            .unwrap()
            .has_cursor(&channel_id)
            .with_context(|| format!("check initial {channel_id} cursor"))?
        {
            return Ok(());
        }
        let max = self
            .channel
            .latest_cursor()
            .await
            .with_context(|| format!("read initial {channel_id} cursor"))?;
        self.store
            .lock()
            .unwrap()
            .set_cursor(&channel_id, max)
            .with_context(|| format!("persist initial {channel_id} cursor"))?;
        info!("starting {channel_id} from cursor {max} (skipping backlog)");
        Ok(())
    }

    async fn tick(&mut self) -> bool {
        self.channel_active = false;
        let store_key = self.channel.store_key();
        let since = match self.store.lock().unwrap().cursor(&store_key) {
            Ok(cursor) => cursor,
            Err(error) => {
                error!(
                    "{} cursor read error; polling paused: {error:#}",
                    self.channel.id()
                );
                return false;
            }
        };
        let msgs = match self.channel.poll(since).await {
            Ok(messages) => messages,
            Err(e) => {
                error!("{} poll error: {e}", self.channel.id());
                return false;
            }
        };

        let complete_snapshot = self.channel.poll_is_complete_snapshot();
        self.process_messages(msgs, complete_snapshot).await;
        self.channel_active
    }

    #[cfg(test)]
    async fn tick_fake(&mut self, msgs: Vec<RawMessage>) {
        self.process_messages(msgs, false).await;
    }

    #[cfg(test)]
    async fn tick_fake_complete(&mut self, msgs: Vec<RawMessage>) {
        self.process_messages(msgs, true).await;
    }

    async fn process_messages(&mut self, msgs: Vec<RawMessage>, complete_snapshot: bool) {
        self.recover_closed_workers();
        let store_key = self.channel.store_key();
        retry_completion_persistence(&self.store, &self.ack, &store_key);
        persist_cursor(&self.store, &self.ack, &store_key);
        let mut since = match self.store.lock().unwrap().cursor(&store_key) {
            Ok(cursor) => cursor,
            Err(error) => {
                error!(
                    "{} cursor read error; message processing paused: {error:#}",
                    self.channel.id()
                );
                return;
            }
        };
        if complete_snapshot {
            let visible = msgs
                .iter()
                .filter(|message| message.row_id > since)
                .map(|message| message.row_id)
                .collect::<BTreeSet<_>>();
            let missing = self
                .ack
                .lock()
                .unwrap()
                .deferred
                .iter()
                .filter(|row_id| !visible.contains(row_id))
                .copied()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                {
                    let mut ack = self.ack.lock().unwrap();
                    for row_id in &missing {
                        ack.deferred.remove(row_id);
                        ack.completed.insert(*row_id);
                        warn!(
                            "deferred {} row {row_id} disappeared before processing",
                            self.channel.id()
                        );
                    }
                }
                persist_cursor(&self.store, &self.ack, &store_key);
                since = match self.store.lock().unwrap().cursor(&store_key) {
                    Ok(cursor) => cursor,
                    Err(error) => {
                        error!(
                            "{} cursor read error after deferred reconciliation: {error:#}",
                            self.channel.id()
                        );
                        return;
                    }
                };
            }
        }
        let durable_completed = if complete_snapshot {
            match self
                .store
                .lock()
                .unwrap()
                .completed_rows_after(&store_key, since)
            {
                Ok(rows) => rows.into_iter().collect::<BTreeSet<_>>(),
                Err(error) => {
                    error!(
                        "{} completed row read error; polling paused: {error:#}",
                        self.channel.id()
                    );
                    return;
                }
            }
        } else {
            BTreeSet::new()
        };
        for m in &msgs {
            if m.row_id <= since {
                continue;
            }
            if durable_completed.contains(&m.row_id) {
                self.ack.lock().unwrap().completed.insert(m.row_id);
                persist_cursor(&self.store, &self.ack, &store_key);
                continue;
            }
            if self.ack.lock().unwrap().is_known(m.row_id) {
                continue;
            }
            if let Some((thread, target)) = self.channel.accept(m) {
                let deferred = {
                    let mut store = self.store.lock().unwrap();
                    self.channel.should_defer(m, &mut store)
                };
                let deferred = match deferred {
                    Ok(deferred) => deferred,
                    Err(error) => {
                        error!("[{thread}] attachment readiness check failed: {error:#}");
                        self.audit(self.ctx.audit.failed(
                            "message_defer_failed",
                            m.row_id,
                            &thread,
                            None,
                            error.to_string(),
                        ));
                        return;
                    }
                };
                if deferred {
                    self.ack.lock().unwrap().deferred.insert(m.row_id);
                    info!("[{thread}] waiting for iMessage attachment filename");
                    continue;
                }
                let reply_with_voice = m.voice.is_some();
                let message_text = if reply_with_voice {
                    "[Voice message]".to_string()
                } else if m.text.trim().is_empty() && !m.images.is_empty() {
                    "[Image attachment]".to_string()
                } else if m.text.trim().is_empty() && !m.files.is_empty() {
                    "[File attachment]".to_string()
                } else {
                    m.text.trim().to_string()
                };
                let approval_origin = self.channel.approval_origin(m, &thread);
                let approval = if reply_with_voice || !m.images.is_empty() || !m.files.is_empty() {
                    Ok(AnswerOutcome::NotAnAnswer)
                } else {
                    self.ctx.history.lock().unwrap().answer_question(
                        &approval_origin,
                        message_text.trim(),
                        now_ms(),
                    )
                };
                match approval {
                    Ok(AnswerOutcome::NotAnAnswer) => {}
                    Ok(outcome @ (AnswerOutcome::Selected(_) | AnswerOutcome::Duplicate(_))) => {
                        self.audit_approval(m.row_id, &thread, &outcome);
                        let correlation_id = match &outcome {
                            AnswerOutcome::Selected(answer) => &answer.correlation_id,
                            AnswerOutcome::Duplicate(id) => id,
                            _ => unreachable!(),
                        };
                        let reviewer = format!(
                            "channel={} thread={} sender={} chat={}",
                            approval_origin.channel,
                            approval_origin.thread_key,
                            approval_origin.sender_key,
                            approval_origin.chat_key
                        );
                        let decision =
                            jobs::Ledger::open(&self.cfg.paths.database).and_then(|mut ledger| {
                                let decision = ledger.resolve_schedule_answer(
                                    &self.cfg,
                                    correlation_id,
                                    &reviewer,
                                    now_ms(),
                                )?;
                                audit_schedule_events(&self.ctx, &mut ledger);
                                Ok(decision)
                            });
                        let decision = match decision {
                            Ok(decision) => decision,
                            Err(error) => {
                                error!("[{thread}] schedule review decision failed: {error:#}");
                                self.audit(self.ctx.audit.failed(
                                    "schedule_review_decision_failed",
                                    m.row_id,
                                    &thread,
                                    None,
                                    error.to_string(),
                                ));
                                return;
                            }
                        };
                        let confirmation = match decision {
                            jobs::ScheduleDecision::Approved {
                                job_name,
                                content_hash,
                                event: _,
                            } => {
                                Some(format!(
                                    "Approved schedule activation for `{job_name}` revision `{content_hash}`. It will activate on the next scheduler tick."
                                ))
                            }
                            jobs::ScheduleDecision::Rejected {
                                job_name,
                                content_hash,
                                event: _,
                            } => {
                                Some(format!(
                                    "Rejected schedule activation for `{job_name}` revision `{content_hash}`. The Markdown file remains available for manual use."
                                ))
                            }
                            jobs::ScheduleDecision::Invalidated {
                                job_name,
                                content_hash,
                                reason,
                                event: _,
                            } => {
                                Some(format!(
                                    "Could not activate `{job_name}` revision `{content_hash}` because it changed or became invalid: {reason}"
                                ))
                            }
                            jobs::ScheduleDecision::AlreadyHandled
                            | jobs::ScheduleDecision::NotScheduleReview => None,
                        };
                        if let Some(confirmation) = confirmation {
                            if !reply_to(&self.ctx, &target, &confirmation).await {
                                warn!("[{thread}] schedule review confirmation delivery failed");
                            }
                        }
                        self.complete_row(m.row_id, "approval_answer");
                        continue;
                    }
                    Ok(
                        outcome @ (AnswerOutcome::Ambiguous
                        | AnswerOutcome::Mismatched(_)
                        | AnswerOutcome::InvalidChoice(_)),
                    ) => {
                        // A stray or malformed numeric reply must not become a
                        // chat message (DESIGN §2/§8.7). Acknowledge and stop.
                        self.audit_approval(m.row_id, &thread, &outcome);
                        let notice = "I could not match that number to a question. Reply with the exact `<id> <number>` shown with the question.";
                        if !reply_to(&self.ctx, &target, notice).await {
                            warn!("[{thread}] unmatched approval notice delivery failed");
                        }
                        self.complete_row(m.row_id, "approval_answer_unmatched");
                        continue;
                    }
                    Ok(outcome) => {
                        let retired_job_approval = match &outcome {
                            AnswerOutcome::Cancelled(id) => self
                                .ctx
                                .history
                                .lock()
                                .unwrap()
                                .legacy_job_approval_was_retired(id),
                            _ => Ok(false),
                        };
                        let retired_job_approval = match retired_job_approval {
                            Ok(retired) => retired,
                            Err(error) => {
                                error!("[{thread}] legacy job approval lookup failed: {error}");
                                self.audit(self.ctx.audit.failed(
                                    "approval_answer_failed",
                                    m.row_id,
                                    &thread,
                                    None,
                                    error.to_string(),
                                ));
                                return;
                            }
                        };
                        self.audit_approval(m.row_id, &thread, &outcome);
                        if retired_job_approval
                            && !reply_to(
                                &self.ctx,
                                &target,
                                "Job approval is no longer used. Ask me to create the job again and I will write it directly to the assistant repository.",
                            )
                            .await
                        {
                            warn!("legacy job approval notice delivery failed");
                        }
                        self.complete_row(m.row_id, "approval_answer");
                        continue;
                    }
                    Err(error) => {
                        error!("[{thread}] approval lookup failed: {error}");
                        self.audit(self.ctx.audit.failed(
                            "approval_answer_failed",
                            m.row_id,
                            &thread,
                            None,
                            error.to_string(),
                        ));
                        return;
                    }
                }
                let inbound_id = match self.ctx.history.lock().unwrap().record_inbound(
                    m.channel,
                    &thread,
                    &m.event_id(),
                    message_text.trim(),
                ) {
                    Ok(id) => id,
                    Err(error) => {
                        error!(
                            "[{}] canonical history write failed for event {}: {error}; message will be retried",
                            thread,
                            m.event_id()
                        );
                        self.audit(self.ctx.audit.failed(
                            "message_history_failed",
                            m.row_id,
                            &thread,
                            None,
                            error.to_string(),
                        ));
                        return;
                    }
                };
                let route_threads = self.channel.route_thread_groups(&thread);
                let route = match self
                    .cfg
                    .route_for_message(m.channel, &store_key, &route_threads)
                {
                    Ok(v) => v,
                    Err(e) => {
                        error!("[{thread}] route error: {e}");
                        self.audit(self.ctx.audit.failed(
                            "message_route_failed",
                            m.row_id,
                            &thread,
                            None,
                            e.to_string(),
                        ));
                        self.complete_row(m.row_id, "route_error");
                        continue;
                    }
                };
                let backend = route.backend;
                self.audit(self.ctx.audit.accepted(m, &thread, backend));
                info!(
                    "[{thread}] new message accepted; routing to {}",
                    backend.as_str()
                );
                let job = Job {
                    row_id: m.row_id,
                    inbound_id,
                    thread,
                    target,
                    backend,
                    model: route.model,
                    thinking: route.thinking,
                    prompt_file: route.prompt_file,
                    text: message_text,
                    reply_with_voice,
                    voice_attachment: m.voice.clone(),
                    image_attachments: m.images.clone(),
                    file_attachments: m.files.clone(),
                    approval_origin,
                };
                if job.image_attachments.is_empty()
                    && job.file_attachments.is_empty()
                    && job.text.trim().eq_ignore_ascii_case("/stop")
                {
                    if !self.stop(job).await {
                        return;
                    }
                    continue;
                }
                if !self.route(job).await {
                    return;
                }
            } else {
                self.audit(self.ctx.audit.ignored(m, self.channel.reject_reason(m)));
                self.complete_row(m.row_id, "ignored");
            }
        }
    }

    async fn route(&mut self, job: Job) -> bool {
        self.channel_active = true;
        let row_id = job.row_id;
        let thread = job.thread.clone();
        let target = job.target.clone();
        let backend = job.backend;
        self.ack.lock().unwrap().in_flight.insert(row_id);
        if !self.queues.contains_key(&thread) {
            self.spawn_worker(thread, vec![job]);
            return true;
        }

        // Barge-in (PLAN §7, defect #4): a newer message interrupts the
        // running turn on this thread. The interrupt targets the *running* row
        // id (not the new one) and the new job still queues behind it, so the
        // worker emits "Stopped the current request." and then runs this job.
        if self.cfg.barge_in {
            if let Some(worker) = self.queues.get(&thread) {
                let current = worker.state.lock().unwrap().current_row;
                if let Some(current) = current.filter(|current| *current != row_id) {
                    if worker.cancel.send(current).is_ok() {
                        info!("[{thread}] barge-in: interrupting row {current}");
                    }
                }
            }
        }

        self.queues
            .get(&thread)
            .unwrap()
            .state
            .lock()
            .unwrap()
            .pending
            .push_back(job.clone());
        let enqueue = self.queues.get(&thread).unwrap().jobs.try_send(job.clone());
        match enqueue {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.recover_closed_worker(&thread);
                true
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.queues
                    .get(&thread)
                    .unwrap()
                    .state
                    .lock()
                    .unwrap()
                    .pending
                    .retain(|pending| pending.row_id != row_id);
                warn!("[{thread}] queue full, asking sender to resend");
                self.audit(self.ctx.audit.failed(
                    "message_queue_failed",
                    row_id,
                    &thread,
                    Some(backend),
                    "queue full",
                ));
                let reply = "I'm a bit behind on this thread - resend that in a moment.";
                match worker::record_and_deliver_once(
                    &self.ctx,
                    &job,
                    OutboundOrigin::Gateway,
                    reply,
                )
                .await
                {
                    Ok(delivered) => {
                        if delivered {
                            self.audit(self.ctx.audit.reply_sent(
                                row_id,
                                &thread,
                                &target,
                                Some(backend),
                                reply,
                            ));
                        } else {
                            self.audit(self.ctx.audit.reply_failed(
                                row_id,
                                &thread,
                                &target,
                                Some(backend),
                                "queue-full reply delivery failed",
                            ));
                        }
                        self.complete_row(row_id, "queue_full");
                        true
                    }
                    Err(error) => {
                        error!("[{thread}] canonical history write failed: {error}");
                        self.audit(self.ctx.audit.failed(
                            "message_history_failed",
                            row_id,
                            &thread,
                            Some(backend),
                            error.to_string(),
                        ));
                        self.ack.lock().unwrap().in_flight.remove(&row_id);
                        false
                    }
                }
            }
        }
    }

    fn spawn_worker(&mut self, thread: String, initial_jobs: Vec<Job>) {
        let (jobs, rx) = mpsc::channel::<Job>(QUEUE_DEPTH.max(initial_jobs.len()));
        let (cancel, cancel_rx) = watch::channel(0);
        let state = Arc::new(Mutex::new(WorkerState {
            pending: initial_jobs.iter().cloned().collect(),
            current_row: None,
            retained_rows: BTreeSet::new(),
        }));
        for job in initial_jobs {
            jobs.try_send(job)
                .expect("initial worker queue capacity must fit recovery jobs");
        }
        let ctx = self.ctx.clone();
        self.handles
            .push(tokio::spawn(worker::run(ctx, rx, cancel_rx, state.clone())));
        self.queues.insert(
            thread,
            WorkerQueue {
                jobs,
                state,
                cancel,
            },
        );
    }

    fn recover_closed_workers(&mut self) {
        let closed = self
            .queues
            .iter()
            .filter(|(_, worker)| worker.jobs.is_closed())
            .map(|(thread, _)| thread.clone())
            .collect::<Vec<_>>();
        for thread in closed {
            self.recover_closed_worker(&thread);
        }
    }

    fn recover_closed_worker(&mut self, thread: &str) {
        let Some(worker) = self.queues.remove(thread) else {
            return;
        };
        let pending = worker
            .state
            .lock()
            .unwrap()
            .pending
            .drain(..)
            .collect::<Vec<_>>();
        let Some(first) = pending.first() else {
            return;
        };
        warn!("[{thread}] worker queue closed unexpectedly; replaying retained jobs");
        self.audit(self.ctx.audit.failed(
            "message_queue_recovered",
            first.row_id,
            thread,
            Some(first.backend),
            format!(
                "worker queue closed; replaying {} retained jobs",
                pending.len()
            ),
        ));
        self.spawn_worker(thread.to_string(), pending);
    }

    async fn stop(&mut self, job: Job) -> bool {
        self.channel_active = true;
        let row_id = job.row_id;
        self.ack.lock().unwrap().in_flight.insert(row_id);
        let candidate_target = self.queues.get(&job.thread).and_then(|worker| {
            let state = worker.state.lock().unwrap();
            state.current_row.or_else(|| {
                state
                    .pending
                    .iter()
                    .find(|pending| !state.retained_rows.contains(&pending.row_id))
                    .map(|pending| pending.row_id)
            })
        });
        let target_row = match self
            .ctx
            .history
            .lock()
            .unwrap()
            .record_stop_target(job.inbound_id, candidate_target)
        {
            Ok(target_row) => target_row,
            Err(error) => {
                error!("[{}] stop target write failed: {error:#}", job.thread);
                self.audit(self.ctx.audit.failed(
                    "message_history_failed",
                    row_id,
                    &job.thread,
                    Some(job.backend),
                    error.to_string(),
                ));
                self.ack.lock().unwrap().in_flight.remove(&row_id);
                return false;
            }
        };
        let stopped = target_row.is_some();
        if let Some(target_row) = target_row {
            let signaled = self.queues.get(&job.thread).is_none_or(|worker| {
                let state = worker.state.lock().unwrap();
                let retained = state.retained_rows.contains(&target_row);
                let pending = state
                    .pending
                    .iter()
                    .any(|pending| pending.row_id == target_row);
                let target_is_runnable =
                    state.current_row == Some(target_row) || (pending && !retained);
                !target_is_runnable || worker.cancel.send(target_row).is_ok()
            });
            if !signaled {
                self.ack.lock().unwrap().in_flight.remove(&row_id);
                return false;
            }
        }
        let reply = if stopped {
            "Stop requested. Queued messages will continue."
        } else {
            "Nothing is currently running in this conversation."
        };
        match worker::record_and_deliver_once(&self.ctx, &job, OutboundOrigin::Gateway, reply).await
        {
            Ok(delivered) => {
                if delivered {
                    self.audit(self.ctx.audit.reply_sent(
                        row_id,
                        &job.thread,
                        &job.target,
                        Some(job.backend),
                        reply,
                    ));
                } else {
                    self.audit(self.ctx.audit.reply_failed(
                        row_id,
                        &job.thread,
                        &job.target,
                        Some(job.backend),
                        "stop acknowledgement delivery failed",
                    ));
                }
                self.complete_row(
                    row_id,
                    if stopped {
                        "stop_requested"
                    } else {
                        "nothing_to_stop"
                    },
                );
                true
            }
            Err(error) => {
                error!("[{}] stop reply failed: {error:#}", job.thread);
                self.audit(self.ctx.audit.failed(
                    "message_history_failed",
                    row_id,
                    &job.thread,
                    Some(job.backend),
                    error.to_string(),
                ));
                self.ack.lock().unwrap().in_flight.remove(&row_id);
                false
            }
        }
    }

    fn complete_row(&self, row_id: i64, reason: &str) {
        self.audit(self.ctx.audit.completed(row_id, reason));
        complete_row(&self.store, &self.ack, &self.channel.store_key(), row_id);
    }

    fn audit(&self, event: AuditEvent) {
        audit(&self.ctx, event);
    }

    fn audit_approval(&self, row_id: i64, thread: &str, outcome: &AnswerOutcome) {
        let (event, reason) = match outcome {
            AnswerOutcome::Selected(answer) => (
                "approval_answer_selected",
                format!(
                    "correlation_id={}, selected_number={}",
                    answer.correlation_id, answer.selected_number
                ),
            ),
            AnswerOutcome::Expired(id) => ("approval_answer_rejected", format!("expired:{id}")),
            AnswerOutcome::Duplicate(id) => ("approval_answer_rejected", format!("duplicate:{id}")),
            AnswerOutcome::Cancelled(id) => ("approval_answer_rejected", format!("cancelled:{id}")),
            AnswerOutcome::Mismatched(id) => {
                ("approval_answer_rejected", format!("mismatched:{id}"))
            }
            AnswerOutcome::InvalidChoice(id) => {
                ("approval_answer_rejected", format!("invalid_choice:{id}"))
            }
            AnswerOutcome::Ambiguous => (
                "approval_answer_rejected",
                "ambiguous_plain_number".to_string(),
            ),
            AnswerOutcome::NotAnAnswer => return,
        };
        self.audit(self.ctx.audit.approval(event, row_id, thread, reason));
    }
}

fn audit(ctx: &Ctx, event: AuditEvent) {
    if let Err(e) = ctx.audit.record(event) {
        error!("audit log error: {e}");
    }
}

fn audit_schedule_events(ctx: &Ctx, ledger: &mut jobs::Ledger) {
    if let Err(error) = ctx.audit.flush_schedule_reviews(ledger) {
        error!("schedule audit outbox error: {error:#}");
    }
}

async fn reply_to(ctx: &Ctx, target: &str, text: &str) -> bool {
    let chunks = ctx.channel.outbound_chunks(text, &ctx.reply_marker);
    if chunks.is_empty() {
        error!("send error to {target}: channel produced no outbound chunks");
        return false;
    }
    for chunk in chunks {
        if let Err(error) = send_reply_chunk(ctx, target, &chunk).await {
            error!("send error to {target}: {error}");
            return false;
        }
    }
    true
}

async fn send_reply_chunk(
    ctx: &Ctx,
    target: &str,
    chunk: &crate::channel::OutboundChunk,
) -> Result<()> {
    #[cfg(test)]
    {
        let should_fail = {
            let mut failures = ctx.send_failures_remaining.lock().unwrap();
            if *failures > 0 {
                *failures -= 1;
                true
            } else {
                let mut after = ctx.send_failure_after.lock().unwrap();
                match after.as_mut() {
                    Some(remaining) if *remaining == 0 => {
                        *after = None;
                        true
                    }
                    Some(remaining) => {
                        *remaining -= 1;
                        false
                    }
                    None => false,
                }
            }
        };
        if should_fail {
            anyhow::bail!("send failed");
        }
        if let Some(attachment) = &chunk.attachment {
            ctx.sent_attachments
                .lock()
                .unwrap()
                .push((target.to_string(), attachment.path.clone()));
        } else {
            ctx.sent_replies
                .lock()
                .unwrap()
                .push((target.to_string(), chunk.text.clone()));
        }
        Ok(())
    }
    #[cfg(not(test))]
    {
        let timeout = ctx.channel.delivery_semantics().send_timeout;
        let send = async {
            match &chunk.attachment {
                Some(attachment) => ctx.channel.send_attachment(target, attachment).await,
                None => ctx.channel.send_chunk(target, chunk).await,
            }
        };
        if timeout.is_zero() {
            send.await
        } else {
            match tokio::time::timeout(timeout, send).await {
                Ok(result) => result,
                Err(_) => anyhow::bail!("send timed out"),
            }
        }
    }
}

async fn scheduled_reply_to(
    ctx: &Ctx,
    target: &str,
    text: &str,
    start_chunk: usize,
    progress: jobs::DeliveryProgress,
) -> jobs::DeliveryAttempt {
    let chunks = ctx
        .channel
        .scheduled_outbound_chunks(text, &ctx.reply_marker);
    if chunks.is_empty() {
        let error = "channel produced no outbound chunks";
        tracing::error!("scheduled send error to {target}: {error}");
        return jobs::DeliveryAttempt::failed(0, error.to_string());
    }
    if start_chunk >= chunks.len() {
        return jobs::DeliveryAttempt::delivered(chunks.len());
    }
    for (index, chunk) in chunks.iter().enumerate().skip(start_chunk) {
        if let Err(error) = send_scheduled_chunk(ctx, target, chunk).await {
            tracing::error!("scheduled send error to {target}: {error}");
            return jobs::DeliveryAttempt::failed(index, error.to_string());
        }
        if let Err(error) = progress.checkpoint(index + 1).await {
            tracing::error!("persist scheduled delivery progress: {error:#}");
            return jobs::DeliveryAttempt::failed(index + 1, error.to_string());
        }
    }
    jobs::DeliveryAttempt::delivered(chunks.len())
}

async fn send_scheduled_chunk(
    ctx: &Ctx,
    target: &str,
    chunk: &crate::channel::OutboundChunk,
) -> Result<()> {
    send_reply_chunk(ctx, target, chunk).await
}

fn complete_row(store: &Arc<Mutex<Store>>, ack: &Arc<Mutex<AckState>>, channel: &str, row_id: i64) {
    let persisted = match store.lock().unwrap().mark_row_completed(channel, row_id) {
        Ok(()) => true,
        Err(error) => {
            error!("persist {channel} completed row {row_id}: {error:#}");
            false
        }
    };
    {
        let mut ack = ack.lock().unwrap();
        ack.in_flight.remove(&row_id);
        ack.deferred.remove(&row_id);
        if persisted {
            ack.persisting.remove(&row_id);
            ack.completed.insert(row_id);
        } else {
            ack.persisting.insert(row_id);
        }
    }
    persist_cursor(store, ack, channel);
}

fn retry_completion_persistence(
    store: &Arc<Mutex<Store>>,
    ack: &Arc<Mutex<AckState>>,
    channel: &str,
) {
    let pending = ack
        .lock()
        .unwrap()
        .persisting
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for row_id in pending {
        let persisted = store.lock().unwrap().mark_row_completed(channel, row_id);
        match persisted {
            Ok(()) => {
                let mut ack = ack.lock().unwrap();
                ack.persisting.remove(&row_id);
                ack.completed.insert(row_id);
            }
            Err(error) => error!("retry {channel} completed row {row_id}: {error:#}"),
        }
    }
}

fn persist_cursor(store: &Arc<Mutex<Store>>, ack: &Arc<Mutex<AckState>>, channel: &str) {
    let mut ack = ack.lock().unwrap();
    let Some(row_id) = ack.next_cursor() else {
        return;
    };
    match store.lock().unwrap().set_cursor(channel, row_id) {
        Ok(()) => ack.mark_persisted(row_id),
        Err(e) => error!("save state error: {e}"),
    }
}

fn runners(cfg: &Config) -> HashMap<AgentBackend, Runner> {
    [AgentBackend::Claude, AgentBackend::Codex, AgentBackend::Pi]
        .into_iter()
        .map(|backend| (backend, Runner::for_backend(backend, cfg)))
        .collect()
}

impl AckState {
    fn is_known(&self, row_id: i64) -> bool {
        self.in_flight.contains(&row_id)
            || self.persisting.contains(&row_id)
            || self.completed.contains(&row_id)
    }

    fn next_cursor(&self) -> Option<i64> {
        let limit = self
            .in_flight
            .first()
            .into_iter()
            .chain(self.deferred.first())
            .chain(self.persisting.first())
            .copied()
            .min()
            .unwrap_or(i64::MAX);
        self.completed
            .iter()
            .copied()
            .take_while(|id| *id < limit)
            .max()
    }

    fn mark_persisted(&mut self, row_id: i64) {
        self.completed.retain(|id| *id > row_id);
    }
}

/// Adaptive poll scheduling (PLAN §3): poll at the active floor while a channel
/// has traffic, then step the idle interval up (30s → 60s → cap) while it stays
/// quiet. Traffic snaps straight back to the active floor.
#[derive(Debug, Clone, Copy)]
struct PollBackoff {
    active: Duration,
    idle: Duration,
    max: Duration,
    idle_step: Duration,
    next: Duration,
}

impl PollBackoff {
    fn new(active: Duration, idle: Duration, max: Duration) -> Self {
        Self {
            active,
            idle,
            max,
            idle_step: idle,
            next: Duration::ZERO,
        }
    }

    /// The delay before the next poll. The first poll is immediate.
    fn next(&self) -> Duration {
        self.next
    }

    /// Fold in whether the poll that just ran saw traffic.
    fn record(&mut self, active: bool) {
        if active {
            self.next = self.active;
            self.idle_step = self.idle;
        } else {
            self.next = self.idle_step;
            self.idle_step = (self.idle_step * 2).min(self.max);
        }
    }
}

async fn wait_for_channel_shutdown_or<O>(
    shutdown: &mut watch::Receiver<bool>,
    operation: O,
) -> Option<O::Output>
where
    O: Future,
{
    if *shutdown.borrow() {
        return None;
    }
    tokio::pin!(operation);
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return None;
                }
            }
            output = &mut operation => return Some(output),
        }
    }
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(_) => {
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

mod worker;

#[cfg(test)]
pub(crate) mod tests;
