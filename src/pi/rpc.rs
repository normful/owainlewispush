//! Long-lived `pi --mode rpc` session (PLAN §7, `spikes/runner-design/DESIGN.md`).
//!
//! This module owns the child process and the JSONL protocol; it holds no
//! gateway state and does not change the one-shot runner. The design lands it
//! first, behind nothing, so the risky protocol mechanics are proven before any
//! caller switches to it.
//!
//! Wire shape (verified against real `pi 0.86.1`, see
//! `spikes/runner-design/transcript.txt`):
//!
//! ```text
//! client  {"type":"prompt","message":"..."}                 host -> child (stdin)
//! client  {"type":"get_state"}                              host -> child
//! client  {"type":"get_last_assistant_text"}                host -> child
//! client  {"type":"abort"}
//! client  {"type":"extension_ui_response","id":..,"value"|"confirmed"|"cancelled":..}
//! server  {"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":".."}}
//! server  {"type":"extension_ui_request","id":..,"method":"select","title":..,"options":[..]}
//! server  {"type":"agent_settled"}
//! server  {"type":"response","command":"get_last_assistant_text","data":{"text":..}}
//! ```
//!
//! A malformed line is skipped, not fatal: the parser must tolerate an
//! unbounded stream, and the authoritative answer is fetched with
//! `get_last_assistant_text` rather than reassembled from the stream.

use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// Malformed lines tolerated before a run is declared broken, so a hostile or
/// corrupt stream cannot spin the read loop forever.
const MAX_SKIPPED_LINES: usize = 100;

/// Bytes of stderr kept for error classification. pi reports a missing
/// `--session` on stderr before exiting; stdout (the protocol stream) is silent.
const STDERR_TAIL_LIMIT: usize = 4096;

/// One extension-UI request the child is blocked waiting on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiRequest {
    /// A numbered choice. `options` are the child's raw option strings.
    Select {
        id: String,
        title: String,
        options: Vec<String>,
    },
    /// A yes/no prompt.
    Confirm { id: String, title: String },
    /// Any other UI method (`input`, `editor`, …) push does not render.
    Other { id: String, method: String },
}

impl UiRequest {
    pub fn id(&self) -> &str {
        match self {
            UiRequest::Select { id, .. }
            | UiRequest::Confirm { id, .. }
            | UiRequest::Other { id, .. } => id,
        }
    }
}

/// The host's answer to a [`UiRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiResponse {
    /// Pick one of the request's option strings (sent as `value`).
    Select(String),
    /// Answer a `confirm` request.
    Confirm(bool),
    /// Resolve the dialog as cancelled (also used for `Other`/fire-and-forget).
    Cancelled,
}

impl UiResponse {
    fn wire(&self, id: &str) -> Value {
        match self {
            UiResponse::Select(value) => {
                json!({"type": "extension_ui_response", "id": id, "value": value})
            }
            UiResponse::Confirm(confirmed) => {
                json!({"type": "extension_ui_response", "id": id, "confirmed": confirmed})
            }
            UiResponse::Cancelled => {
                json!({"type": "extension_ui_response", "id": id, "cancelled": true})
            }
        }
    }
}

/// A handler's answer to one [`UiRequest`], boxed so [`Session::turn`] can park
/// on a human reply without knowing the handler's concrete type.
pub type UiFuture<'a> = Pin<Box<dyn Future<Output = UiResponse> + Send + 'a>>;

/// Answers the child's extension-UI requests. Async because a real approval
/// must *park* the turn until the human replies (DESIGN §2); a `FnMut` closure
/// cannot express that, hence the boxed-future trait object.
pub trait UiHandler: Send {
    fn handle<'a>(&'a mut self, request: &'a UiRequest) -> UiFuture<'a>;
}

/// Adapts a synchronous closure to [`UiHandler`] for tests and callers that
/// never wait on a human.
pub struct FnHandler<F>(pub F);

impl<F> UiHandler for FnHandler<F>
where
    F: FnMut(&UiRequest) -> UiResponse + Send,
{
    fn handle<'a>(&'a mut self, request: &'a UiRequest) -> UiFuture<'a> {
        let response = (self.0)(request);
        Box::pin(async move { response })
    }
}

/// Answers every dialog as cancelled. Used when the approval gate is off, so an
/// unexpected UI request can never block an ungated run.
pub struct CancelHandler;

impl UiHandler for CancelHandler {
    fn handle<'a>(&'a mut self, _request: &'a UiRequest) -> UiFuture<'a> {
        Box::pin(async { UiResponse::Cancelled })
    }
}

/// One completed rpc turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    /// Authoritative final assistant text (`get_last_assistant_text`).
    pub text: String,
    /// Advisory count of streamed delta characters, for diagnostics/tests.
    pub delta_chars: usize,
    /// Extension-UI requests the host answered during the turn.
    pub ui_requests: Vec<UiRequest>,
}

/// A live `pi --mode rpc` child with its protocol streams.
pub struct Session {
    child: Child,
    stdin: ChildStdin,
    events: Lines<BufReader<ChildStdout>>,
    /// Rolling stderr tail, drained by a background task so a full pipe can
    /// never block the child. Only consulted to classify a fatal error.
    stderr_tail: Arc<Mutex<String>>,
    session_id: Option<String>,
}

impl Session {
    /// Spawns a child from an explicit argv tail (everything after the binary).
    /// The caller owns the flags (`--mode rpc`, `--session`, `-e`, …) and the
    /// environment (`GATE_TOOLS`, …), so this module stays protocol-only.
    pub fn spawn(
        bin: &str,
        args: &[String],
        env: &[(String, String)],
        work_dir: &str,
    ) -> Result<Self> {
        let mut command = Command::new(bin);
        command.args(args);
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command
            .current_dir(work_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawn {bin} --mode rpc"))?;
        let stdin = child.stdin.take().context("pi rpc stdin unavailable")?;
        let stdout = child.stdout.take().context("pi rpc stdout unavailable")?;
        let stderr = child.stderr.take().context("pi rpc stderr unavailable")?;
        let stderr_tail = Arc::new(Mutex::new(String::new()));
        let sink = stderr_tail.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut tail = sink.lock().unwrap();
                if tail.len() < STDERR_TAIL_LIMIT {
                    tail.push_str(&line);
                    tail.push('\n');
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            events: BufReader::new(stdout).lines(),
            stderr_tail,
            session_id: None,
        })
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Best-effort stderr seen so far, for classifying a fatal spawn/init error.
    pub fn stderr_tail(&self) -> String {
        self.stderr_tail.lock().unwrap().clone()
    }

    async fn send(&mut self, command: &Value) -> Result<()> {
        let mut line = command.to_string();
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .await
            .context("write pi rpc command")?;
        self.stdin.flush().await.context("flush pi rpc command")
    }

    /// Reads one JSONL event. `Ok(None)` is EOF; a malformed line is reported
    /// to the caller as `Ok(Event::Malformed)` so the run can keep going.
    async fn read_event(&mut self) -> Result<Event> {
        match self
            .events
            .next_line()
            .await
            .context("read pi rpc stdout")?
        {
            None => Ok(Event::Eof),
            Some(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return Ok(Event::Malformed(trimmed.to_string()));
                }
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(value) => Ok(Event::Value(value)),
                    Err(_) => Ok(Event::Malformed(trimmed.to_string())),
                }
            }
        }
    }

    /// Sends `get_state` and records the session id. Returns the id.
    pub async fn init(&mut self, timeout: Duration) -> Result<String> {
        self.send(&json!({"type": "get_state", "id": "push-state"}))
            .await?;
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let event = tokio::time::timeout_at(deadline, self.read_event())
                .await
                .map_err(|_| anyhow::anyhow!("pi rpc get_state timed out"))??;
            match event {
                Event::Value(value)
                    if value.get("command").and_then(Value::as_str) == Some("get_state") =>
                {
                    let id = value
                        .pointer("/data/sessionId")
                        .and_then(Value::as_str)
                        .context("pi rpc get_state returned no sessionId")?
                        .to_string();
                    self.session_id = Some(id.clone());
                    return Ok(id);
                }
                Event::Value(_) | Event::Malformed(_) => continue,
                Event::Eof => anyhow::bail!("pi rpc closed before get_state answered"),
            }
        }
    }

    /// Runs one prompt to completion, answering extension-UI requests through
    /// `handler`. The final text comes from `get_last_assistant_text` after
    /// `agent_settled`; an empty tool-call turn is never mistaken for an answer.
    pub async fn turn(
        &mut self,
        prompt: &str,
        timeout: Duration,
        handler: &mut dyn UiHandler,
    ) -> Result<TurnOutcome> {
        self.send(&json!({"type": "prompt", "message": prompt}))
            .await?;
        let mut skipped = 0usize;
        let mut delta_chars = 0usize;
        let mut ui_requests = Vec::new();
        let mut last_message_end: Option<String> = None;
        let mut asked_for_final = false;
        let deadline = tokio::time::Instant::now() + timeout;

        let result = loop {
            let event = match tokio::time::timeout_at(deadline, self.read_event()).await {
                Err(_) => break Err(anyhow::anyhow!("pi rpc turn timed out")),
                Ok(Err(error)) => break Err(error),
                Ok(Ok(event)) => event,
            };
            let value = match event {
                Event::Eof => break Err(anyhow::anyhow!("pi rpc closed during the run")),
                Event::Malformed(line) => {
                    skipped += 1;
                    if skipped > MAX_SKIPPED_LINES {
                        break Err(anyhow::anyhow!(
                            "pi rpc stream produced more than {MAX_SKIPPED_LINES} malformed lines"
                        ));
                    }
                    tracing::warn!("pi rpc skipped a malformed line: {:.200}", line);
                    continue;
                }
                Event::Value(value) => value,
            };
            match value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "message_update" => {
                    if value
                        .pointer("/assistantMessageEvent/type")
                        .and_then(Value::as_str)
                        == Some("text_delta")
                    {
                        if let Some(delta) = value
                            .pointer("/assistantMessageEvent/delta")
                            .and_then(Value::as_str)
                        {
                            delta_chars += delta.chars().count();
                        }
                    }
                }
                "message_end" => {
                    if value.pointer("/message/role").and_then(Value::as_str) == Some("assistant") {
                        last_message_end = Some(message_text(&value));
                    }
                }
                "extension_ui_request" => {
                    let id = value
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let method = value
                        .get("method")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let title = value
                        .get("title")
                        .or_else(|| value.get("message"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let request = match method.as_str() {
                        "select" => UiRequest::Select {
                            id,
                            title,
                            options: value
                                .get("options")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .filter_map(|option| option.as_str().map(str::to_string))
                                .collect(),
                        },
                        "confirm" => UiRequest::Confirm { id, title },
                        _ => {
                            // notify/setStatus/setWidget are fire-and-forget: the
                            // spike confirms replying to them is protocol noise.
                            UiRequest::Other { id, method }
                        }
                    };
                    let response = handler.handle(&request).await;
                    ui_requests.push(request.clone());
                    self.send(&response.wire(request.id())).await?;
                }
                "agent_settled" => {
                    if !asked_for_final {
                        asked_for_final = true;
                        self.send(&json!({"type": "get_last_assistant_text", "id": "push-final"}))
                            .await?;
                    }
                }
                "response"
                    if value.get("command").and_then(Value::as_str)
                        == Some("get_last_assistant_text") =>
                {
                    let text = value
                        .pointer("/data/text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or(last_message_end.take())
                        .unwrap_or_default();
                    break Ok(TurnOutcome {
                        text,
                        delta_chars,
                        ui_requests,
                    });
                }
                "extension_error" => {
                    tracing::warn!("pi rpc extension_error: {:.200}", value.to_string());
                }
                _ => {}
            }
        };
        result
    }

    /// Asks the child to abort the current run. An outstanding extension dialog
    /// must be answered with [`UiResponse::Cancelled`] *first*: abort alone does
    /// not resolve it (verified in the spike).
    pub async fn abort(&mut self) -> Result<()> {
        self.send(&json!({"type": "abort", "id": "push-abort"}))
            .await
    }

    /// Graceful shutdown: drop stdin (pi treats EOF as a shutdown request),
    /// then kill after a short grace if it does not exit.
    pub async fn shutdown(mut self) {
        let _ = self.stdin.shutdown().await;
        drop(self.stdin);
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
        let _ = self.child.kill().await;
    }
}

enum Event {
    Value(Value),
    Malformed(String),
    Eof,
}

/// Extracts the concatenated `text` parts of an assistant `message_end` event.
fn message_text(value: &Value) -> String {
    value
        .pointer("/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

/// A small FIFO of pending UI answers, useful for tests and simple callers.
#[derive(Default)]
pub struct UiQueue {
    answers: VecDeque<UiResponse>,
}

impl UiQueue {
    pub fn push(&mut self, response: UiResponse) {
        self.answers.push_back(response);
    }

    pub fn pop(&mut self, request: &UiRequest) -> UiResponse {
        self.answers.pop_front().unwrap_or(match request {
            UiRequest::Confirm { .. } => UiResponse::Confirm(false),
            _ => UiResponse::Cancelled,
        })
    }
}

/// A [`UiHandler`] over a prepared [`UiQueue`], for tests that stage a fixed
/// sequence of answers.
pub struct QueueHandler(pub UiQueue);

impl UiHandler for QueueHandler {
    fn handle<'a>(&'a mut self, request: &'a UiRequest) -> UiFuture<'a> {
        let response = self.0.pop(request);
        Box::pin(async move { response })
    }
}

/// Best-effort child termination without a graceful attempt; used on drop paths.
pub fn terminate(child: &mut Child) -> io::Result<()> {
    child.start_kill()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeCli;
    use std::sync::{Arc, Mutex};

    /// A fake `pi --mode rpc`: reads commands line by line and reacts by
    /// substring, so the test needs no JSON tooling.
    const FAKE: &str = r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *"get_state"*)
      printf '%s\n' '{"type":"response","command":"get_state","id":"push-state","data":{"sessionId":"sess-1","model":{"id":"m"},"thinkingLevel":"high"}}'
      ;;
    *"get_last_assistant_text"*)
      printf '%s\n' '{"type":"response","command":"get_last_assistant_text","id":"push-final","data":{"text":"hello from gate"}}'
      ;;
    *"extension_ui_response"*)
      printf '%s\n' '{"type":"tool_execution_end","toolName":"bash","isError":false,"result":{"content":[{"type":"text","text":"ran"}]}}'
      printf '%s\n' '{"type":"message_end","message":{"role":"assistant","stopReason":"stop","content":[{"type":"text","text":"hello from gate"}]}}'
      printf '%s\n' '{"type":"agent_settled"}'
      ;;
    *"get_state_ok"*)
      ;;
    *'"type":"prompt"'*)
      printf '%s\n' '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"hello "}}'
      printf '%s\n' 'this is not json'
      printf '%s\n' '{"type":"extension_ui_request","id":"ui-1","method":"select","title":"Approve bash?","options":["1. Allow","2. Deny"]}'
      ;;
    *)
      ;;
  esac
done
"#;

    fn spam() -> String {
        FAKE.to_string()
    }

    #[tokio::test]
    async fn init_reads_the_session_id_from_get_state() {
        let cli = FakeCli::new("rpc-init", &spam());
        let mut session = Session::spawn(
            &cli.bin(),
            &["--mode".to_string(), "rpc".to_string()],
            &[],
            ".",
        )
        .unwrap();
        let id = session.init(Duration::from_secs(5)).await.unwrap();
        assert_eq!(id, "sess-1");
        assert_eq!(session.session_id(), Some("sess-1"));
        session.shutdown().await;
    }

    #[tokio::test]
    async fn turn_answers_a_select_and_returns_the_final_text() {
        let cli = FakeCli::new("rpc-turn", &spam());
        let mut session = Session::spawn(
            &cli.bin(),
            &["--mode".to_string(), "rpc".to_string()],
            &[],
            ".",
        )
        .unwrap();
        session.init(Duration::from_secs(5)).await.unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_handle = seen.clone();
        let mut handler = FnHandler(move |request: &UiRequest| {
            seen_handle.lock().unwrap().push(request.clone());
            match request {
                UiRequest::Select { options, .. } => {
                    UiResponse::Select(options.first().cloned().unwrap_or_default())
                }
                _ => UiResponse::Cancelled,
            }
        });
        let outcome = session
            .turn("do the thing", Duration::from_secs(5), &mut handler)
            .await
            .unwrap();

        assert_eq!(outcome.text, "hello from gate");
        assert_eq!(outcome.delta_chars, "hello ".chars().count());
        assert_eq!(outcome.ui_requests.len(), 1);
        assert!(matches!(
            seen.lock().unwrap().first(),
            Some(UiRequest::Select { options, .. }) if options.len() == 2
        ));
        session.shutdown().await;
    }

    #[tokio::test]
    async fn a_malformed_line_is_skipped_not_fatal() {
        // The fake emits `this is not json` between the delta and the prompt
        // request; the turn above already proved it does not abort. Here we
        // assert directly on the reader contract.
        let cli = FakeCli::new("rpc-skip", &spam());
        let mut session = Session::spawn(
            &cli.bin(),
            &["--mode".to_string(), "rpc".to_string()],
            &[],
            ".",
        )
        .unwrap();
        session.init(Duration::from_secs(5)).await.unwrap();
        let mut handler = CancelHandler;
        let outcome = session
            .turn("go", Duration::from_secs(5), &mut handler)
            .await
            .unwrap();
        assert_eq!(outcome.text, "hello from gate");
        session.shutdown().await;
    }

    #[tokio::test]
    async fn ui_queue_defaults_to_the_safe_answer() {
        let mut queue = UiQueue::default();
        queue.push(UiResponse::Select("1. Allow".to_string()));
        let select = UiRequest::Select {
            id: "a".to_string(),
            title: "t".to_string(),
            options: vec!["1. Allow".to_string(), "2. Deny".to_string()],
        };
        assert_eq!(
            queue.pop(&select),
            UiResponse::Select("1. Allow".to_string())
        );
        // Exhausted queue: select/other cancel, confirm denies.
        assert_eq!(queue.pop(&select), UiResponse::Cancelled);
        let confirm = UiRequest::Confirm {
            id: "b".to_string(),
            title: "t".to_string(),
        };
        assert_eq!(queue.pop(&confirm), UiResponse::Confirm(false));
    }

    #[tokio::test]
    async fn queue_handler_adapts_the_queue_to_the_trait() {
        let mut handler = QueueHandler(UiQueue::default());
        let select = UiRequest::Select {
            id: "a".to_string(),
            title: "t".to_string(),
            options: vec!["1. Allow".to_string(), "2. Deny".to_string()],
        };
        assert_eq!(
            handler.handle(&select).await,
            UiResponse::Cancelled,
            "an exhausted queue cancels a select"
        );
    }
}
