//! Loopback WebSocket turn endpoint for the voice bridge
//! (`spikes/voice-bridge/BRIDGE-PUSH-CONTRACT.md`).
//!
//! The bridge is a separate process. This module is the seam: it accepts one
//! WebSocket per voice conversation and runs each `turn` through the same pool,
//! session store, and prompt composition the Discord workers use, so a voice
//! turn is an ordinary Pi turn that simply never renders to Discord and never
//! uses the reply marker.
//!
//! Security is two independent gates: the listener binds a loopback address
//! only, and every upgrade must present the per-boot token. Loopback alone is
//! not authorisation — any local process can reach a loopback port.
//!
//! The Gemini side of the bridge is a later stage. For now the endpoint only
//! buffers audio frames so the framing is exercised; nothing consumes them.

use std::collections::VecDeque;
use std::future::Future;
use std::io::Write;
use std::net::SocketAddr;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::agent::{RunError, RunOutput};
use crate::audit::AuditLog;
use crate::config::{AgentBackend, Config};
use crate::pi::rpc::{Delta, FnHandler, UiRequest, UiResponse};
use crate::pi::Pool;
use crate::prompt::Composer;
use crate::store::Store;

use super::worker::{rpc_spec, RpcSpecInput};

/// The only wire version this build speaks. A mismatch fails loudly instead of
/// misparsing a future frame.
const PROTOCOL_VERSION: u64 = 1;

/// Header carrying the per-boot token on the upgrade request.
const TOKEN_HEADER: &str = "x-push-internal-token";

/// Binary frame type byte for microphone audio (bridge -> push).
const AUDIO_MIC: u8 = 0x01;
/// Binary frame type byte for playback audio (push -> bridge). Accepted and
/// dropped: sending it toward push is a client bug, not a fatal one.
const AUDIO_PLAYBACK: u8 = 0x02;

/// Bound on buffered audio. 24 kHz s16le PCM is ~48 KiB/s, so this is a little
/// over one second. The consumer arrives in a later stage; until then the bound
/// proves a slow or absent reader cannot grow memory without limit.
const AUDIO_MAX_BYTES: usize = 64 * 1024;
const AUDIO_MAX_FRAMES: usize = 64;

/// Messages a turn may queue before the reader loop drains them.
const DELTA_CHANNEL_DEPTH: usize = 32;
const UI_CHANNEL_DEPTH: usize = 16;

/// A boxed turn future. `Pool::run` borrows its handler and the delta sink for
/// the duration of the call, so the future owns both.
type TurnFuture = Pin<Box<dyn Future<Output = Result<RunOutput, RunError>> + Send>>;

/// Everything the endpoint shares with the Discord gateway. It is deliberately
/// *not* the worker's `Ctx`: `Ctx` requires a concrete `Channel`, and a voice
/// turn has no channel (contract §1 non-goals).
pub(super) struct InternalEndpoint {
    cfg: Config,
    store: Arc<Mutex<Store>>,
    pool: Arc<Pool>,
    audit: Arc<AuditLog>,
    /// Loaded once: instructions do not change per turn, and `Composer::load`
    /// canonicalizes paths and reads SOUL.md.
    composer: Composer,
    run_timeout: Duration,
    /// Canonical voice workspace.
    work_dir: String,
    listen: SocketAddr,
    /// Per-boot token. Never logged.
    token: String,
}

impl InternalEndpoint {
    /// Builds the endpoint when `[internal].listen` is set; `Ok(None)` when it
    /// is not, so an unconfigured push behaves exactly as before.
    ///
    /// Side effects at startup: create the voice work_dir (the composer
    /// canonicalizes it and fails on a missing directory) and overwrite the
    /// token file with a fresh 0600 token.
    pub(super) fn new(
        cfg: &Config,
        store: Arc<Mutex<Store>>,
        pool: Arc<Pool>,
        audit: Arc<AuditLog>,
    ) -> Result<Option<Self>> {
        let Some(listen) = cfg.internal_listen_addr()? else {
            return Ok(None);
        };
        let work_dir = cfg.internal_work_dir();
        std::fs::create_dir_all(&work_dir)
            .with_context(|| format!("create voice work_dir {}", work_dir.display()))?;
        let work_dir = std::fs::canonicalize(&work_dir)
            .with_context(|| format!("resolve voice work_dir {}", work_dir.display()))?;
        let composer = Composer::load(&cfg.assistant_dir, &work_dir.to_string_lossy())
            .context("load voice prompt composer")?;
        let token = generate_token();
        write_token_file(&cfg.paths.internal_token, &token)?;
        // Only the location is logged; the token itself never is.
        info!(
            "voice bridge endpoint token written to {}",
            cfg.paths.internal_token.display()
        );
        Ok(Some(Self {
            cfg: cfg.clone(),
            store,
            pool,
            audit,
            composer,
            run_timeout: cfg.run_timeout_dur()?,
            work_dir: work_dir.to_string_lossy().to_string(),
            listen,
            token,
        }))
    }

    /// Binds the listener. The caller awaits this at startup so a bind failure
    /// is a startup error, not a warning that hides an endpoint nobody can
    /// reach (contract §9).
    pub(super) async fn bind(self) -> Result<InternalListener> {
        let listener = TcpListener::bind(self.listen)
            .await
            .with_context(|| format!("bind internal endpoint on {}", self.listen))?;
        info!("voice bridge endpoint listening on {}", self.listen);
        Ok(InternalListener {
            ctx: Arc::new(self),
            listener,
        })
    }
}

/// A bound endpoint ready to accept connections.
pub(super) struct InternalListener {
    ctx: Arc<InternalEndpoint>,
    listener: TcpListener,
}

impl InternalListener {
    /// Accepts connections until `shutdown` flips, then aborts every connection
    /// task. Aborting drops each in-flight run future, which kills its child
    /// and clears the pool entry (contract §9).
    pub(super) async fn serve(self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        let InternalListener { ctx, listener } = self;
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }
                accepted = listener.accept() => {
                    let (stream, peer) = accepted.context("accept internal connection")?;
                    // The listener is loopback-bound, so a non-loopback peer
                    // means something forwarded the socket; reject it before
                    // any handshake work.
                    if !peer_is_allowed(peer) {
                        warn!("voice bridge endpoint rejected non-loopback peer {peer}");
                        continue;
                    }
                    let ctx = ctx.clone();
                    tasks.spawn(async move {
                        if let Err(error) = handle_connection(ctx, stream).await {
                            warn!("voice bridge connection ended: {error:#}");
                        }
                    });
                }
            }
        }
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        Ok(())
    }
}

/// Answers the upgrade only for a request carrying the per-boot token, then
/// runs the frame loop. Rejection is a `401` written before the upgrade
/// completes.
// The callback's error type is tungstenite's `ErrorResponse`, which is large by
// construction; the trait leaves no smaller option.
#[allow(clippy::result_large_err)]
async fn handle_connection(ctx: Arc<InternalEndpoint>, stream: TcpStream) -> Result<()> {
    let expected = ctx.token.clone();
    let socket = tokio_tungstenite::accept_hdr_async(
        stream,
        move |request: &Request, response: Response| {
            let presented = request
                .headers()
                .get(TOKEN_HEADER)
                .and_then(|value| value.to_str().ok());
            if presented == Some(expected.as_str()) {
                return Ok(response);
            }
            let mut rejection = ErrorResponse::new(None);
            *rejection.status_mut() = StatusCode::UNAUTHORIZED;
            Err(rejection)
        },
    )
    .await
    .context("voice bridge websocket handshake")?;

    let (mut sink, mut stream) = socket.split();

    // At most one turn is in flight. `id`/`thread` describe it for replies and
    // audit; the three channels carry its output until it settles.
    let mut run: Option<TurnFuture> = None;
    let mut deltas: Option<mpsc::Receiver<Delta>> = None;
    let mut ui: Option<mpsc::Receiver<Value>> = None;
    let mut current_id: Option<String> = None;
    let mut current_thread: Option<String> = None;
    // Audio that arrives with no consumer yet; see AUDIO_MAX_BYTES.
    let mut audio = AudioBuffer::default();

    loop {
        tokio::select! {
            incoming = stream.next() => {
                let Some(incoming) = incoming else { break };
                let message = match incoming {
                    Ok(message) => message,
                    Err(error) => {
                        warn!("voice bridge frame read failed: {error}");
                        break;
                    }
                };
                match message {
                    Message::Text(text) => {
                        let Ok(frame) = serde_json::from_str::<Value>(text.as_str()) else {
                            if !send_json(&mut sink, error_frame(None, "protocol", "control frame is not JSON")).await {
                                break;
                            }
                            continue;
                        };
                        if !handle_control(
                            &ctx,
                            &mut sink,
                            &mut run,
                            &mut deltas,
                            &mut ui,
                            &mut current_id,
                            &mut current_thread,
                            &frame,
                        )
                        .await
                        {
                            break;
                        }
                    }
                    Message::Binary(bytes) => {
                        audio.push(bytes.as_ref());
                    }
                    Message::Close(_) => break,
                    Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
                }
            }
            delta = recv_opt(&mut deltas) => {
                match delta {
                    Some(delta) => {
                        let id = current_id.clone().unwrap_or_default();
                        let frame = json!({
                            "type": "delta",
                            "v": PROTOCOL_VERSION,
                            "id": id,
                            "text": delta.text,
                            "boundary": delta.boundary,
                        });
                        if !send_json(&mut sink, frame).await {
                            break;
                        }
                    }
                    None => deltas = None,
                }
            }
            request = recv_opt(&mut ui) => {
                match request {
                    Some(frame) => {
                        if !send_json(&mut sink, frame).await {
                            break;
                        }
                    }
                    None => ui = None,
                }
            }
            outcome = poll_turn(&mut run) => {
                // Drain output the turn queued before it settled: a fast turn
                // can finish before the reader loop services every frame, and
                // losing a delta would break the streamed-answer guarantee.
                if let Some(mut rx) = deltas.take() {
                    while let Ok(delta) = rx.try_recv() {
                        let id = current_id.clone().unwrap_or_default();
                        let frame = json!({
                            "type": "delta",
                            "v": PROTOCOL_VERSION,
                            "id": id,
                            "text": delta.text,
                            "boundary": delta.boundary,
                        });
                        if !send_json(&mut sink, frame).await {
                            break;
                        }
                    }
                }
                if let Some(mut rx) = ui.take() {
                    while let Ok(frame) = rx.try_recv() {
                        if !send_json(&mut sink, frame).await {
                            break;
                        }
                    }
                }
                let id = current_id.take().unwrap_or_default();
                let thread = current_thread.take().unwrap_or_default();
                run = None;
                finish_turn(&ctx, &mut sink, &id, &thread, outcome).await;
            }
        }
    }
    // Dropping `run` here aborts the child and clears the pool entry, exactly
    // as a socket close should (contract §9).
    Ok(())
}

/// Handles one JSON control frame. Returns false when the connection should
/// close (a bad `hello` token or a send failure).
#[allow(clippy::too_many_arguments)]
async fn handle_control<S>(
    ctx: &InternalEndpoint,
    sink: &mut S,
    run: &mut Option<TurnFuture>,
    deltas: &mut Option<mpsc::Receiver<Delta>>,
    ui: &mut Option<mpsc::Receiver<Value>>,
    current_id: &mut Option<String>,
    current_thread: &mut Option<String>,
    frame: &Value,
) -> bool
where
    S: SinkExt<Message> + Unpin,
{
    let kind = frame
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    // `v` is present on every bridge frame so a version mismatch fails loudly.
    if frame.get("v").and_then(Value::as_u64) != Some(PROTOCOL_VERSION) {
        return send_json(
            sink,
            error_frame(
                frame.get("id").and_then(Value::as_str),
                "protocol",
                &format!("unsupported protocol version for {kind:?}"),
            ),
        )
        .await;
    }
    match kind {
        "hello" => {
            if let Some(presented) = frame.get("token").and_then(Value::as_str) {
                if presented != ctx.token {
                    let _ = send_json(sink, error_frame(None, "auth", "invalid token")).await;
                    return false;
                }
            }
            send_json(
                sink,
                json!({
                    "type": "ready",
                    "v": PROTOCOL_VERSION,
                    "server": format!("push/{}", env!("CARGO_PKG_VERSION")),
                }),
            )
            .await
        }
        "ping" => send_json(sink, json!({"type": "pong", "v": PROTOCOL_VERSION})).await,
        "turn" => {
            start_turn(
                ctx,
                sink,
                run,
                deltas,
                ui,
                current_id,
                current_thread,
                frame,
            )
            .await
        }
        "cancel" => {
            cancel_turn(
                ctx,
                sink,
                run,
                deltas,
                ui,
                current_id,
                current_thread,
                frame,
            )
            .await
        }
        other => {
            send_json(
                sink,
                error_frame(
                    frame.get("id").and_then(Value::as_str),
                    "protocol",
                    &format!("unsupported frame type {other:?}"),
                ),
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn start_turn<S>(
    ctx: &InternalEndpoint,
    sink: &mut S,
    run: &mut Option<TurnFuture>,
    deltas: &mut Option<mpsc::Receiver<Delta>>,
    ui: &mut Option<mpsc::Receiver<Value>>,
    current_id: &mut Option<String>,
    current_thread: &mut Option<String>,
    frame: &Value,
) -> bool
where
    S: SinkExt<Message> + Unpin,
{
    let id = frame.get("id").and_then(Value::as_str).unwrap_or_default();
    if run.is_some() {
        return send_json(
            sink,
            error_frame(Some(id), "busy", "a turn is already in flight"),
        )
        .await;
    }
    let Some(thread) = frame.get("thread").and_then(Value::as_str) else {
        return send_json(
            sink,
            error_frame(Some(id), "protocol", "turn is missing thread"),
        )
        .await;
    };
    // The store's own validation is the authority, but rejecting the shape here
    // keeps a bad frame out of the session table with a clearer error class.
    if !is_thread_shaped(thread) {
        return send_json(
            sink,
            error_frame(
                Some(id),
                "protocol",
                "thread must be channel-qualified, for example voice:desk",
            ),
        )
        .await;
    }
    let Some(text) = frame.get("text").and_then(Value::as_str) else {
        return send_json(
            sink,
            error_frame(Some(id), "protocol", "turn is missing text"),
        )
        .await;
    };

    // Voice threads are `voice:<device>`; the store accepts any non-empty
    // channel prefix, so no schema change is needed. Pi's initial id is empty
    // (the child discovers its own), matching the worker.
    let session =
        ctx.store
            .lock()
            .unwrap()
            .session_for(thread, AgentBackend::Pi.as_str(), String::new());
    let (session_id, is_new) = match session {
        Ok(session) => session,
        Err(error) => {
            return send_json(
                sink,
                error_frame(Some(id), "pi", &format!("session setup failed: {error}")),
            )
            .await;
        }
    };

    let prompt = ctx
        .composer
        .conversation("voice", thread, "text", &[], text);
    let spec = rpc_spec(
        &ctx.cfg,
        RpcSpecInput {
            session_id: &session_id,
            is_new,
            work_dir: &ctx.work_dir,
            prompt: &prompt,
            images: &[],
            model: frame.get("model").and_then(Value::as_str),
            thinking: frame.get("thinking").and_then(Value::as_str),
            // Voice runs ungated (contract §7): `extension: None` means the
            // child never loads the approval gate, and no GATE_TOOLS is set.
            gate: false,
        },
    );
    record(
        &ctx.audit,
        ctx.audit
            .backend_started(0, thread, AgentBackend::Pi, is_new, 0),
    );

    if !send_json(
        sink,
        json!({
            "type": "accepted",
            "v": PROTOCOL_VERSION,
            "id": id,
            "session": session_id,
            "is_new": is_new,
        }),
    )
    .await
    {
        return false;
    }

    let (delta_tx, delta_rx) = mpsc::channel::<Delta>(DELTA_CHANNEL_DEPTH);
    let (ui_tx, ui_rx) = mpsc::channel::<Value>(UI_CHANNEL_DEPTH);
    let ui_id = id.to_string();
    // Every UiRequest is answered Cancelled immediately so the turn never
    // parks; the frame is informational for the bridge's log (contract §7).
    let mut handler = FnHandler(move |request: &UiRequest| {
        let _ = ui_tx.try_send(ui_frame(&ui_id, request));
        UiResponse::Cancelled
    });
    let pool = ctx.pool.clone();
    let timeout = ctx.run_timeout;
    let key = thread.to_string();
    let future: TurnFuture = Box::pin(async move {
        pool.run(
            &key,
            spec,
            &prompt.content,
            timeout,
            &mut handler,
            Some(&delta_tx),
        )
        .await
    });

    *run = Some(future);
    *deltas = Some(delta_rx);
    *ui = Some(ui_rx);
    *current_id = Some(id.to_string());
    *current_thread = Some(thread.to_string());
    true
}

#[allow(clippy::too_many_arguments)]
async fn cancel_turn<S>(
    ctx: &InternalEndpoint,
    sink: &mut S,
    run: &mut Option<TurnFuture>,
    deltas: &mut Option<mpsc::Receiver<Delta>>,
    ui: &mut Option<mpsc::Receiver<Value>>,
    current_id: &mut Option<String>,
    current_thread: &mut Option<String>,
    frame: &Value,
) -> bool
where
    S: SinkExt<Message> + Unpin,
{
    let id = frame.get("id").and_then(Value::as_str);
    let matches = match (current_id.as_deref(), id) {
        (Some(current), Some(id)) => current == id,
        _ => false,
    };
    if !matches {
        return send_json(
            sink,
            error_frame(id, "protocol", "no matching turn to cancel"),
        )
        .await;
    }
    let id = current_id.clone().unwrap_or_default();
    let thread = current_thread.clone().unwrap_or_default();
    // Dropping the run future is the whole cancellation: it drops the child,
    // whose `kill_on_drop` reaps it, and the pool's `RunGuard` clears the
    // thread entry so the next turn respawns and resumes with `--session`
    // (contract §6). `Pool::run` does not expose the per-turn `Session`, so the
    // graceful `abort` message cannot be sent; the handler never parks on a
    // dialog, so there is no outstanding UI request to answer first either.
    *run = None;
    *deltas = None;
    *ui = None;
    *current_id = None;
    *current_thread = None;
    record(
        &ctx.audit,
        ctx.audit.failed(
            "voice_turn_cancelled",
            0,
            &thread,
            Some(AgentBackend::Pi),
            "cancelled by the bridge",
        ),
    );
    send_json(
        sink,
        json!({"type": "cancelled", "v": PROTOCOL_VERSION, "id": id}),
    )
    .await
}

/// Sends the terminal frame for a settled run and records it in the audit log.
async fn finish_turn<S>(
    ctx: &InternalEndpoint,
    sink: &mut S,
    id: &str,
    thread: &str,
    outcome: Result<RunOutput, RunError>,
) where
    S: SinkExt<Message> + Unpin,
{
    match outcome {
        Ok(output) => {
            record(
                &ctx.audit,
                ctx.audit
                    .backend_completed(0, thread, AgentBackend::Pi, &output.reply),
            );
            // Only a freshly created session reports its id, matching the
            // worker; a resumed turn leaves the stored row untouched.
            if let Err(error) = ctx
                .store
                .lock()
                .unwrap()
                .mark_started(thread, output.session_id.as_deref())
            {
                warn!("[{}] voice session save failed: {error}", thread);
            }
            send_json(
                sink,
                json!({
                    "type": "done",
                    "v": PROTOCOL_VERSION,
                    "id": id,
                    "text": output.reply,
                }),
            )
            .await;
        }
        Err(error) => {
            let class = error_class(&error);
            let message = error_message(&error);
            record(
                &ctx.audit,
                ctx.audit.failed(
                    "voice_turn_failed",
                    0,
                    thread,
                    Some(AgentBackend::Pi),
                    &message,
                ),
            );
            warn!("[{}] voice turn failed ({class}): {message}", thread);
            send_json(sink, error_frame(Some(id), class, &message)).await;
        }
    }
}

/// Sends one JSON control frame. False means the socket is gone.
async fn send_json<S>(sink: &mut S, value: Value) -> bool
where
    S: SinkExt<Message> + Unpin,
{
    sink.send(Message::Text(value.to_string().into()))
        .await
        .is_ok()
}

/// Resolves to the next item of an optional receiver, or never when absent, so
/// an idle channel parks a `select!` branch instead of spinning.
async fn recv_opt<T>(rx: &mut Option<mpsc::Receiver<T>>) -> Option<T> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// Awaits an optional turn future; never resolves while no turn is running.
async fn poll_turn(run: &mut Option<TurnFuture>) -> Result<RunOutput, RunError> {
    match run {
        Some(run) => run.as_mut().await,
        None => std::future::pending().await,
    }
}

fn error_frame(id: Option<&str>, class: &str, message: &str) -> Value {
    json!({
        "type": "error",
        "v": PROTOCOL_VERSION,
        "id": id,
        "class": class,
        "message": message,
    })
}

/// Informational frame for a UI request the endpoint refuses to render.
fn ui_frame(id: &str, request: &UiRequest) -> Value {
    let request = match request {
        UiRequest::Select { title, options, .. } => {
            json!({"kind": "select", "title": title, "options": options})
        }
        UiRequest::Confirm { title, .. } => json!({"kind": "confirm", "title": title}),
        UiRequest::Other { method, .. } => json!({"kind": "other", "method": method}),
    };
    json!({
        "type": "ui",
        "v": PROTOCOL_VERSION,
        "id": id,
        "request": request,
    })
}

fn error_class(error: &RunError) -> &'static str {
    match error {
        RunError::Timeout => "timeout",
        RunError::SessionMissing(_) => "pi",
        RunError::Failed(message) if message.to_lowercase().contains("spawn") => "spawn",
        RunError::Failed(_) => "pi",
    }
}

fn error_message(error: &RunError) -> String {
    match error {
        RunError::Timeout => "Pi turn timed out".to_string(),
        RunError::SessionMissing(message) | RunError::Failed(message) => message.clone(),
    }
}

fn is_thread_shaped(thread: &str) -> bool {
    thread
        .split_once(':')
        .is_some_and(|(channel, key)| !channel.trim().is_empty() && !key.trim().is_empty())
}

/// The only peers the endpoint serves. Loopback only, because the listener is
/// loopback-bound and anything else means the socket was forwarded.
fn peer_is_allowed(peer: SocketAddr) -> bool {
    peer.ip().is_loopback()
}

/// Audits a voice turn. Voice has no history row, so `row_id` is the existing
/// constructors' `0` sentinel.
///
/// Deviation from contract §9: the contract says `surface: "voice"`, but
/// `AuditEvent` has no `surface` field. The audit log is already channel-scoped,
/// so the channel is set to `"voice"` at construction and every event on this
/// endpoint carries it.
fn record(audit: &AuditLog, event: crate::audit::AuditEvent) {
    if let Err(error) = audit.record(event) {
        warn!("voice audit write failed: {error:#}");
    }
}

/// A bounded FIFO of audio that arrives before any consumer exists. The Gemini
/// side of the bridge is a later stage, so for now this only proves the binary
/// framing and keeps an absent reader from growing memory without limit.
#[derive(Default)]
struct AudioBuffer {
    frames: VecDeque<Vec<u8>>,
    bytes: usize,
}

impl AudioBuffer {
    fn push(&mut self, frame: &[u8]) {
        let Some((&kind, payload)) = frame.split_first() else {
            return;
        };
        match kind {
            AUDIO_MIC => {}
            // Playback is push -> bridge; a client sending it is a bug worth
            // ignoring rather than failing the connection over.
            AUDIO_PLAYBACK => return,
            _ => return,
        }
        self.bytes += payload.len();
        self.frames.push_back(payload.to_vec());
        while self.bytes > AUDIO_MAX_BYTES || self.frames.len() > AUDIO_MAX_FRAMES {
            let Some(dropped) = self.frames.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(dropped.len());
        }
    }
}

/// Two v4 UUIDs: 244 random bits, more than enough for a loopback bearer token
/// and available without a new dependency.
fn generate_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Overwrites any stale token file with a fresh 0600 one. The mode is set on
/// create and forced afterwards, because an existing file keeps its old mode.
fn write_token_file(path: &Path, token: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create token directory {}", parent.display()))?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("write internal token {}", path.display()))?;
    file.write_all(token.as_bytes())
        .context("write internal token")?;
    file.write_all(b"\n").context("write internal token")?;
    file.sync_all().context("sync internal token")?;
    crate::util::restrict_permissions(path, false)
        .with_context(|| format!("restrict internal token permissions {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests;
