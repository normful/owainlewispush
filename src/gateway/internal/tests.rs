//! Conformance tests for the voice-bridge endpoint, following
//! `spikes/voice-bridge/BRIDGE-PUSH-CONTRACT.md` §10. They drive the real
//! WebSocket server with a fake `pi --mode rpc` child, so no real `pi` binary is
//! required.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::audit::AuditLog;
use crate::config::{Config, InternalConfig};
use crate::history::History;
use crate::paths::PushPaths;
use crate::pi::Pool;
use crate::store::Store;
use crate::test_support::{sh_arg, temp_dir, FakeCli};

use super::{InternalEndpoint, TOKEN_HEADER};

type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// A running endpoint plus the state a test needs to talk to it.
struct Harness {
    cfg: Config,
    addr: SocketAddr,
    token: String,
    store: Arc<Mutex<Store>>,
    shutdown: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
    _cli: FakeCli,
}

impl Harness {
    async fn stop(self) {
        let _ = self.shutdown.send(true);
        let _ = self.task.await;
    }
}

/// A fake `pi --mode rpc` that streams two sentences and reports them as the
/// final text. `ui` makes it also raise a confirm dialog, to prove the endpoint
/// answers without creating an approval question. `delay` holds the turn open
/// long enough for a second frame (busy) or a cancel to arrive.
fn fake_rpc(args: &Path, env: &Path, session: &str, delay: &str, ui: bool) -> String {
    let completion = if ui {
        r#"printf '%s\n' '{"type":"extension_ui_request","id":"ui-1","method":"confirm","title":"Approve bash?"}'"#
    } else {
        r#"printf '%s\n' '{"type":"agent_settled"}'"#
    };
    format!(
        r#"#!/bin/sh
printf '%s\n' "$@" > {args}
printenv GATE_TOOLS > {env}
while IFS= read -r line; do
  case "$line" in
    *"get_state"*)
      printf '%s\n' '{{"type":"response","command":"get_state","id":"push-state","data":{{"sessionId":"{session}","model":{{"id":"m"}},"thinkingLevel":"high"}}}}'
      ;;
    *"get_last_assistant_text"*)
      printf '%s\n' '{{"type":"response","command":"get_last_assistant_text","id":"push-final","data":{{"text":"One. Two."}}}}'
      ;;
    *"extension_ui_response"*)
      printf '%s\n' '{{"type":"agent_settled"}}'
      ;;
    *'"type":"prompt"'*)
      sleep {delay}
      printf '%s\n' '{{"type":"message_update","assistantMessageEvent":{{"type":"text_delta","delta":"One. "}}}}'
      printf '%s\n' '{{"type":"message_update","assistantMessageEvent":{{"type":"text_delta","delta":"Two."}}}}'
      {completion}
      ;;
  esac
done
"#,
        args = sh_arg(args),
        env = sh_arg(env),
    )
}

async fn start(script: &str) -> Harness {
    let root = temp_dir("internal-endpoint");
    let assistant = root.join("assistant");
    std::fs::create_dir_all(&assistant).unwrap();
    std::fs::write(assistant.join("SOUL.md"), "Be useful.").unwrap();
    let voice = root.join("voice");
    let cli = FakeCli::new("pi", script);

    let mut cfg = crate::test_support::test_config();
    cfg.assistant_root = assistant.to_string_lossy().to_string();
    cfg.assistant_dir = assistant.to_string_lossy().to_string();
    cfg.jobs_dir = assistant.join("jobs").to_string_lossy().to_string();
    cfg.pi_mode = "rpc".to_string();
    cfg.run_timeout = "10s".to_string();
    cfg.agent_commands.pi = cli.bin();
    cfg.paths = PushPaths {
        root: root.join("home"),
        config: root.join("home/config.toml"),
        database: root.join("home/push.db"),
        state: root.join("home/state.json"),
        audit: root.join("home/audit.jsonl"),
        jobs_run: root.join("home/run"),
        inbox: root.join("home/state.slack-inbox.db"),
        cache: root.join("home/cache"),
        internal_token: root.join("home/internal.token"),
    };
    cfg.internal = Some(InternalConfig {
        listen: Some("127.0.0.1:0".to_string()),
        work_dir: Some(voice.to_string_lossy().to_string()),
    });
    std::fs::create_dir_all(&cfg.paths.root).unwrap();

    let store = Arc::new(Mutex::new(Store::open(&cfg.paths).unwrap()));
    // Create the approval_questions schema so test 6 can assert on it.
    let history = History::open(&cfg.paths.database).unwrap();
    drop(history);
    let pool = Arc::new(Pool::new(cfg.rpc_pool_config().unwrap()));
    let audit = Arc::new(AuditLog::new(cfg.paths.audit.clone(), false, "voice"));

    let endpoint = InternalEndpoint::new(&cfg, store.clone(), pool, audit)
        .unwrap()
        .expect("listen is configured");
    let listener = endpoint.bind().await.unwrap();
    let addr = listener.listener.local_addr().unwrap();
    let token = std::fs::read_to_string(&cfg.paths.internal_token)
        .unwrap()
        .trim()
        .to_string();
    let (shutdown, rx) = watch::channel(false);
    let task = tokio::spawn(async move {
        listener.serve(rx).await.unwrap();
    });

    Harness {
        cfg,
        addr,
        token,
        store,
        shutdown,
        task,
        _cli: cli,
    }
}

async fn connect_with(addr: SocketAddr, token: Option<&str>) -> Result<Client, String> {
    let mut request = format!("ws://{addr}")
        .into_client_request()
        .map_err(|error| error.to_string())?;
    if let Some(token) = token {
        request.headers_mut().insert(
            TOKEN_HEADER,
            HeaderValue::from_str(token).map_err(|error| error.to_string())?,
        );
    }
    connect_async(request)
        .await
        .map(|(socket, _)| socket)
        .map_err(|error| error.to_string())
}

async fn write_json(socket: &mut Client, value: Value) {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .unwrap();
}

async fn next_json(socket: &mut Client) -> Value {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match socket
                .next()
                .await
                .expect("socket ended")
                .expect("socket error")
            {
                Message::Text(text) => {
                    return serde_json::from_str(text.as_str()).expect("frame is JSON")
                }
                Message::Close(_) => panic!("server closed the connection"),
                _ => continue,
            }
        }
    })
    .await
    .expect("timed out waiting for a frame")
}

async fn hello(socket: &mut Client) {
    write_json(socket, json!({"type": "hello", "v": 1, "client": "test"})).await;
    let ready = next_json(socket).await;
    assert_eq!(ready["type"], "ready", "frame was {ready}");
}

async fn open(h: &Harness) -> Client {
    let mut socket = connect_with(h.addr, Some(&h.token))
        .await
        .expect("upgrade with the token succeeds");
    hello(&mut socket).await;
    socket
}

/// 1. `hello` -> `ready`; an unknown `v` -> `error` class `protocol`.
#[tokio::test]
async fn hello_reads_ready_and_a_bad_version_is_a_protocol_error() {
    let h = start(&fake_rpc(
        &temp_dir("internal-t1-args").join("args"),
        &temp_dir("internal-t1-env").join("env"),
        "sess-1",
        "0",
        false,
    ))
    .await;
    let mut socket = connect_with(h.addr, Some(&h.token)).await.unwrap();

    write_json(&mut socket, json!({"type": "hello", "v": 2})).await;
    let error = next_json(&mut socket).await;
    assert_eq!(error["type"], "error", "frame was {error}");
    assert_eq!(error["class"], "protocol");

    hello(&mut socket).await;
    h.stop().await;
}

/// 2. Upgrade without a token and with a wrong token are both rejected. A
/// non-loopback listen address is refused outright, and the peer predicate that
/// guards the accept loop rejects anything outside loopback.
#[tokio::test]
async fn upgrade_requires_the_token_and_loopback() {
    let h = start(&fake_rpc(
        &temp_dir("internal-t2-args").join("args"),
        &temp_dir("internal-t2-env").join("env"),
        "sess-1",
        "0",
        false,
    ))
    .await;

    assert!(
        connect_with(h.addr, None).await.is_err(),
        "an upgrade with no token must be rejected"
    );
    assert!(
        connect_with(h.addr, Some("not-the-token")).await.is_err(),
        "an upgrade with a wrong token must be rejected"
    );

    // A non-loopback bind is refused before the listener exists.
    let mut cfg = crate::test_support::test_config();
    cfg.internal = Some(InternalConfig {
        listen: Some("0.0.0.0:7317".to_string()),
        work_dir: None,
    });
    assert!(cfg.internal_listen_addr().is_err());

    // The accept loop's decision is a pure predicate, so it is testable without
    // a second network interface.
    for (peer, allowed) in [
        ("127.0.0.1:5000", true),
        ("127.9.9.9:5000", true),
        ("[::1]:5000", true),
        ("192.168.1.5:5000", false),
        ("10.0.0.1:5000", false),
    ] {
        assert_eq!(
            super::peer_is_allowed(peer.parse().unwrap()),
            allowed,
            "peer {peer}"
        );
    }
    h.stop().await;
}

/// 3. `turn` -> `accepted`, at least one non-empty `delta`, then `done` whose
/// text contains every delta (the lossy-stream guarantee).
#[tokio::test]
async fn turn_streams_deltas_then_an_authoritative_done() {
    let h = start(&fake_rpc(
        &temp_dir("internal-t3-args").join("args"),
        &temp_dir("internal-t3-env").join("env"),
        "sess-1",
        "0",
        false,
    ))
    .await;
    let mut socket = open(&h).await;
    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "t1", "thread": "voice:desk", "text": "hi"}),
    )
    .await;

    let accepted = next_json(&mut socket).await;
    assert_eq!(accepted["type"], "accepted", "frame was {accepted}");
    assert_eq!(accepted["id"], "t1");
    assert_eq!(accepted["is_new"], true);
    // A fresh turn has no session id until the child reports one, so `accepted`
    // carries the store's placeholder (empty). Test 7 covers the resumed id.
    assert_eq!(accepted["session"], "");

    let mut deltas = Vec::new();
    let mut saw_ui = false;
    let done = loop {
        let frame = next_json(&mut socket).await;
        match frame["type"].as_str().unwrap_or_default() {
            "delta" => {
                let text = frame["text"].as_str().unwrap_or_default().to_string();
                assert!(!text.trim().is_empty(), "delta text must be non-empty");
                deltas.push(text);
            }
            "ui" => saw_ui = true,
            "done" => break frame,
            other => panic!("unexpected frame {other}: {frame}"),
        }
    };
    assert!(!saw_ui, "an ungated turn raises no UI request");
    assert!(
        !deltas.is_empty(),
        "the turn must stream at least one delta"
    );
    let text = done["text"].as_str().unwrap();
    for delta in &deltas {
        assert!(
            text.contains(delta.as_str()),
            "{text:?} is missing {delta:?}"
        );
    }
    h.stop().await;
}

/// 4. A second `turn` before `done` gets `error` class `busy`.
#[tokio::test]
async fn a_second_turn_while_busy_is_rejected() {
    let h = start(&fake_rpc(
        &temp_dir("internal-t4-args").join("args"),
        &temp_dir("internal-t4-env").join("env"),
        "sess-1",
        "1",
        false,
    ))
    .await;
    let mut socket = open(&h).await;
    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "a", "thread": "voice:desk", "text": "first"}),
    )
    .await;
    assert_eq!(next_json(&mut socket).await["type"], "accepted");
    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "b", "thread": "voice:desk", "text": "second"}),
    )
    .await;

    let busy = loop {
        let frame = next_json(&mut socket).await;
        if frame["type"] == "error" && frame["class"] == "busy" {
            break frame;
        }
    };
    assert_eq!(busy["id"], "b");
    h.stop().await;
}

/// 5. `cancel` mid-turn -> `cancelled`; the run future is dropped, so the next
/// turn on the same thread succeeds (proves `RunGuard` recovery).
#[tokio::test]
async fn cancel_stops_the_turn_and_the_thread_recovers() {
    let h = start(&fake_rpc(
        &temp_dir("internal-t5-args").join("args"),
        &temp_dir("internal-t5-env").join("env"),
        "sess-1",
        "1",
        false,
    ))
    .await;
    let mut socket = open(&h).await;
    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "c1", "thread": "voice:desk", "text": "long"}),
    )
    .await;
    assert_eq!(next_json(&mut socket).await["type"], "accepted");

    write_json(&mut socket, json!({"type": "cancel", "v": 1, "id": "c1"})).await;
    let cancelled = loop {
        let frame = next_json(&mut socket).await;
        if frame["type"] == "cancelled" {
            break frame;
        }
    };
    assert_eq!(cancelled["id"], "c1");

    // The next turn respawns a child and resumes the stored session.
    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "c2", "thread": "voice:desk", "text": "again"}),
    )
    .await;
    let done = loop {
        let frame = next_json(&mut socket).await;
        if frame["type"] == "done" {
            break frame;
        }
    };
    assert_eq!(done["id"], "c2");
    assert!(!done["text"].as_str().unwrap().is_empty());
    h.stop().await;
}

/// 6. With the gate configured, a voice turn that raises a confirm dialog
/// completes without creating an `approval_questions` row, and the child was
/// spawned ungated (no `-e`, no `GATE_TOOLS`).
#[tokio::test]
async fn voice_turns_run_ungated_and_create_no_questions() {
    let args = temp_dir("internal-t6-args").join("args");
    let env = temp_dir("internal-t6-env").join("env");
    let h = start(&fake_rpc(&args, &env, "sess-1", "0", true)).await;
    let mut socket = connect_with(h.addr, Some(&h.token)).await.unwrap();
    write_json(&mut socket, json!({"type": "hello", "v": 1})).await;
    assert_eq!(next_json(&mut socket).await["type"], "ready");

    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "g1", "thread": "voice:gate", "text": "run bash"}),
    )
    .await;
    let outcome = loop {
        let frame = next_json(&mut socket).await;
        if frame["type"] == "done" || frame["type"] == "error" {
            break frame;
        }
    };
    assert_eq!(outcome["type"], "done", "frame was {outcome}");

    let spawned = std::fs::read_to_string(&args).unwrap();
    assert!(
        !spawned.lines().any(|line| line.trim() == "-e"),
        "the gate extension must not be loaded: {spawned}"
    );
    assert_eq!(
        std::fs::read_to_string(&env).unwrap().trim(),
        "",
        "GATE_TOOLS must not be set for voice"
    );

    let connection = rusqlite::Connection::open(&h.cfg.paths.database).unwrap();
    let questions: i64 = connection
        .query_row("SELECT COUNT(*) FROM approval_questions", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(questions, 0, "voice must never create an approval question");
    h.stop().await;
}

/// 7. A `voice:<x>` turn writes a `backend_sessions` row, and a later turn
/// resumes it with `is_new: false`.
#[tokio::test]
async fn voice_threads_persist_and_resume_a_session() {
    let h = start(&fake_rpc(
        &temp_dir("internal-t7-args").join("args"),
        &temp_dir("internal-t7-env").join("env"),
        "sess-7",
        "0",
        false,
    ))
    .await;
    let mut socket = open(&h).await;

    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "s1", "thread": "voice:phone", "text": "one"}),
    )
    .await;
    let first = next_json(&mut socket).await;
    assert_eq!(first["type"], "accepted");
    assert_eq!(first["is_new"], true);
    loop {
        if next_json(&mut socket).await["type"] == "done" {
            break;
        }
    }

    let stored = h
        .store
        .lock()
        .unwrap()
        .session_for("voice:phone", "pi", String::new())
        .unwrap();
    assert_eq!(stored, ("sess-7".to_string(), false));

    write_json(
        &mut socket,
        json!({"type": "turn", "v": 1, "id": "s2", "thread": "voice:phone", "text": "two"}),
    )
    .await;
    let second = next_json(&mut socket).await;
    assert_eq!(second["type"], "accepted");
    assert_eq!(second["is_new"], false);
    assert_eq!(second["session"], "sess-7");
    h.stop().await;
}

/// 8. With no `[internal]` table nothing listens: no address, no endpoint, and
/// no token file.
#[tokio::test]
async fn no_internal_table_means_no_endpoint() {
    let cfg = crate::test_support::test_config();
    assert!(cfg.internal_listen_addr().unwrap().is_none());

    let root = temp_dir("internal-t8");
    let store = Arc::new(Mutex::new(
        Store::open(&PushPaths::from_root(root.clone()).unwrap()).unwrap(),
    ));
    let pool = Arc::new(Pool::new(cfg.rpc_pool_config().unwrap()));
    let audit = Arc::new(AuditLog::new(root.join("audit.jsonl"), false, "voice"));
    let endpoint = InternalEndpoint::new(&cfg, store, pool, audit).unwrap();
    assert!(endpoint.is_none(), "the endpoint must be disabled");
    assert!(
        !root.join("internal.token").exists(),
        "a disabled endpoint writes no token"
    );
}

/// The configured voice work_dir defaults beside jobs_dir and is not the
/// Discord work_dir.
#[test]
fn default_voice_work_dir_is_beside_jobs() {
    let root = temp_dir("internal-default-work");
    let mut cfg = crate::test_support::test_config();
    cfg.jobs_dir = root.join("assistant/jobs").to_string_lossy().to_string();
    assert_eq!(cfg.internal_work_dir(), root.join("assistant/voice"));
}
