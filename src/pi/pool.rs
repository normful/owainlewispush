//! Per-thread pool of long-lived `pi --mode rpc` children (DESIGN §1, §7 step 2).
//!
//! push's unit of serialization and its backend-session identity are both the
//! channel-qualified thread, so one thread owns one child. The pool spawns that
//! child lazily on the thread's first rpc run, reuses it for every later turn,
//! and evicts it once it has been idle past `rpc_idle_timeout`. An idle child is
//! ~126 MiB, so an unbounded pool is the default failure mode.
//!
//! The run owns its [`Session`] for the duration of a turn: it is *moved out* of
//! the map rather than borrowed behind a lock. Two consequences fall out of
//! that:
//!
//! * a run can never be cancelled while it holds the map lock, so the worker's
//!   interrupt `select!` stays responsive;
//! * when the interrupt arm drops the run future, the [`Session`] goes with it,
//!   and `kill_on_drop` reaps the child. [`RunGuard`] clears the map entry so
//!   the next turn on that thread spawns a fresh child and resumes it with
//!   `--session <id>`, exactly the crash contract in DESIGN §4.
//!
//! An entry with `session: None` is a thread whose run is in flight; idle
//! eviction and the cap skip it, so neither can pull the child out from under a
//! parked approval or a running turn.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::agent::{final_reply, RunError, RunOutput};

use super::rpc::{Session, UiRequest, UiResponse};

/// Static limits for a pool, resolved once from `[agent]` config.
#[derive(Debug, Clone, Copy)]
pub struct PoolConfig {
    /// Maximum concurrent children; over the cap the longest-idle child is
    /// evicted rather than the run refused.
    pub max_sessions: usize,
    /// How long a child may sit unused before the next run/sweep reaps it.
    pub idle_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_sessions: 12,
            idle_timeout: Duration::from_secs(600),
        }
    }
}

/// Everything the pool needs to spawn a thread's child lazily. Built by the
/// worker from the resolved job and config; the per-turn prompt is passed
/// separately to [`Pool::run`].
#[derive(Debug, Clone)]
pub struct PiRpcSpec {
    pub bin: String,
    /// The stored session id, used only when a child must be spawned to resume.
    pub session_id: String,
    /// True when the store has not yet activated a session for this thread. A
    /// fresh child is spawned without `--session`; the id discovered from
    /// `get_state` is returned so the worker can persist it.
    pub is_new: bool,
    pub work_dir: String,
    pub instructions: String,
    pub model: Option<String>,
    pub thinking: Option<String>,
    pub images: Vec<PathBuf>,
    /// Extra environment for the child (gate tool list, …).
    pub env: Vec<(String, String)>,
}

struct Entry {
    /// `None` while the owning run holds the child.
    session: Option<Session>,
    session_id: String,
    last_used: Instant,
}

pub struct Pool {
    config: PoolConfig,
    sessions: Mutex<HashMap<String, Entry>>,
}

impl Pool {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// The configured idle timeout, used to pace the janitor.
    pub fn idle_timeout(&self) -> Duration {
        self.config.idle_timeout
    }

    /// Runs one turn on `key`'s thread, spawning and initializing a child on
    /// first use. `key` is the channel-qualified thread push already uses for
    /// serialization (`job.thread`).
    pub async fn run<F>(
        &self,
        key: &str,
        spec: PiRpcSpec,
        prompt: &str,
        timeout: Duration,
        on_ui: F,
    ) -> Result<RunOutput, RunError>
    where
        F: FnMut(&UiRequest) -> UiResponse,
    {
        let (mut session, session_id) = self.acquire(key, &spec, timeout).await?;
        // The guard forgets this thread's entry if the run future is dropped
        // (interrupt/task abort) before the turn settles.
        let mut guard = RunGuard {
            pool: self,
            key: key.to_string(),
            armed: true,
        };

        let outcome = match session.turn(prompt, timeout, on_ui).await {
            Ok(outcome) => outcome,
            Err(error) => {
                guard.armed = false;
                self.discard(key);
                drop(session);
                return Err(classify(error));
            }
        };
        let reply = match final_reply("pi", &outcome.text) {
            Ok(reply) => reply,
            Err(error) => {
                guard.armed = false;
                self.discard(key);
                drop(session);
                return Err(error);
            }
        };

        guard.armed = false;
        self.release(key, session, session_id.clone());
        // Match the one-shot runner: only a freshly created session reports its
        // id. A resumed turn leaves the stored row untouched.
        Ok(RunOutput {
            reply,
            session_id: spec.is_new.then_some(session_id),
        })
    }

    /// Reaps every child idle past `idle_timeout`. Called before each spawn and
    /// available to a janitor; never touches a thread with a run in flight.
    pub async fn sweep_idle(&self) {
        for session in self.take_idle(Instant::now()) {
            session.shutdown().await;
        }
    }

    /// Drops a thread's child without a run, e.g. on `/clear`, so the next turn
    /// starts a fresh conversation instead of reusing the rotated session.
    pub fn evict(&self, key: &str) {
        if let Some(entry) = self.sessions.lock().unwrap().remove(key) {
            drop(entry.session);
        }
    }

    /// Live children currently held (diagnostics/tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn live_sessions(&self) -> usize {
        self.sessions
            .lock()
            .unwrap()
            .values()
            .filter(|entry| entry.session.is_some())
            .count()
    }

    async fn acquire(
        &self,
        key: &str,
        spec: &PiRpcSpec,
        timeout: Duration,
    ) -> Result<(Session, String), RunError> {
        // Idle reaping happens on the run path because a returning thread is
        // exactly when memory is worth reclaiming; a busy thread's child is
        // still in the map as `None` and is skipped.
        for session in self.take_idle(Instant::now()) {
            session.shutdown().await;
        }
        {
            let mut map = self.sessions.lock().unwrap();
            if let Some(entry) = map.get_mut(key) {
                if let Some(session) = entry.session.take() {
                    entry.last_used = Instant::now();
                    return Ok((session, entry.session_id.clone()));
                }
                // The per-thread worker serializes a thread's jobs, so this is
                // only reachable if the pool is misused; fail loudly rather
                // than multiplex two runs onto one child.
                return Err(RunError::Failed(
                    "pi rpc pool: a run is already in flight for this thread".to_string(),
                ));
            }
        }
        for session in self.make_room() {
            session.shutdown().await;
        }

        let mut session = self.spawn(spec)?;
        match session.init(timeout).await {
            Ok(id) => Ok((session, id)),
            Err(error) => {
                // pi reports a missing `--session` on stderr as it exits; the
                // drain is a background task, so wait briefly for it.
                let stderr = drain_stderr(&session).await;
                drop(session);
                Err(classify_init(error, spec, &stderr))
            }
        }
    }

    fn spawn(&self, spec: &PiRpcSpec) -> Result<Session, RunError> {
        let mut args = vec!["--mode".to_string(), "rpc".to_string()];
        if !spec.instructions.trim().is_empty() {
            args.push("--append-system-prompt".to_string());
            args.push(spec.instructions.trim().to_string());
        }
        if let Some(model) = spec
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            args.push("--model".to_string());
            args.push(model.to_string());
        }
        if let Some(thinking) = spec
            .thinking
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            args.push("--thinking".to_string());
            args.push(thinking.to_string());
        }
        if !spec.is_new {
            let session_id = spec.session_id.trim();
            if !session_id.is_empty() {
                args.push("--session".to_string());
                args.push(session_id.to_string());
            }
        }
        for image in &spec.images {
            args.push(format!("@{}", image.display()));
        }
        Session::spawn(&spec.bin, &args, &spec.env, &spec.work_dir)
            .map_err(|error| RunError::Failed(format!("spawn pi rpc: {error:#}")))
    }

    /// Returns the child to the pool after a completed turn.
    fn release(&self, key: &str, session: Session, session_id: String) {
        self.sessions.lock().unwrap().insert(
            key.to_string(),
            Entry {
                session: Some(session),
                session_id,
                last_used: Instant::now(),
            },
        );
    }

    /// Forgets a thread's entry; the owning [`Session`] is dropped separately.
    fn discard(&self, key: &str) {
        self.sessions.lock().unwrap().remove(key);
    }

    fn take_idle(&self, now: Instant) -> Vec<Session> {
        let mut map = self.sessions.lock().unwrap();
        let victims = map
            .iter()
            .filter(|(_, entry)| entry.session.is_some())
            .filter(|(_, entry)| {
                now.saturating_duration_since(entry.last_used) >= self.config.idle_timeout
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        victims
            .into_iter()
            .filter_map(|key| map.remove(&key).and_then(|entry| entry.session))
            .collect()
    }

    /// Frees room for a new child by evicting the longest-idle session. Only
    /// live, non-busy sessions are candidates; if every child is in flight the
    /// cap is exceeded rather than refusing the run.
    fn make_room(&self) -> Vec<Session> {
        let mut map = self.sessions.lock().unwrap();
        let mut evicted = Vec::new();
        while map.len() >= self.config.max_sessions {
            let victim = map
                .iter()
                .filter(|(_, entry)| entry.session.is_some())
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else {
                break;
            };
            if let Some(session) = map.remove(&victim).and_then(|entry| entry.session) {
                evicted.push(session);
            }
        }
        evicted
    }

    #[cfg(test)]
    fn held_keys(&self) -> Vec<String> {
        let mut keys = self
            .sessions
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, entry)| entry.session.is_some())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        keys.sort();
        keys
    }
}

/// Clears the map entry when a run future is dropped before it settles, so the
/// thread respawns instead of reusing a child that was killed with the future.
struct RunGuard<'a> {
    pool: &'a Pool,
    key: String,
    armed: bool,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.pool.discard(&self.key);
        }
    }
}

/// Waits briefly for the background stderr drain to see anything the child
/// wrote before dying, then returns what it has. Bounded so a silent crash
/// cannot stall a run.
async fn drain_stderr(session: &Session) -> String {
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        let tail = session.stderr_tail();
        if !tail.is_empty() || Instant::now() >= deadline {
            return tail;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Maps a spawn/init failure. Only a resume attempt can be a missing session,
/// and pi reports that on stderr rather than the protocol stream.
fn classify_init(error: anyhow::Error, spec: &PiRpcSpec, stderr_tail: &str) -> RunError {
    if !spec.is_new && super::missing_resume_error(stderr_tail) {
        return RunError::SessionMissing(
            "Pi could not find the saved session; Push will rebuild it from conversation history"
                .to_string(),
        );
    }
    classify(error)
}

/// Maps a turn failure onto the existing `RunError` variants.
fn classify(error: anyhow::Error) -> RunError {
    if error.to_string().contains("timed out") {
        RunError::Timeout
    } else {
        RunError::Failed(format!("run pi rpc: {error:#}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{sh_arg, temp_path, FakeCli};
    use std::sync::Arc;

    /// A fake `pi --mode rpc` that reacts to commands by substring. `delay` is
    /// the seconds slept before settling a prompt, so a test can hold a turn in
    /// flight. The spawn marker is appended once per child.
    fn fake_rpc(marker: &std::path::Path, session: &str, delay: f64) -> String {
        format!(
            r#"#!/bin/sh
printf 'x' >> {marker}
while IFS= read -r line; do
  case "$line" in
    *"get_state"*)
      printf '%s\n' '{{"type":"response","command":"get_state","id":"push-state","data":{{"sessionId":"{session}","model":{{"id":"m"}},"thinkingLevel":"high"}}}}'
      ;;
    *"get_last_assistant_text"*)
      printf '%s\n' '{{"type":"response","command":"get_last_assistant_text","id":"push-final","data":{{"text":"reply"}}}}'
      ;;
    *'"type":"prompt"'*)
      sleep {delay}
      printf '%s\n' '{{"type":"agent_settled"}}'
      ;;
  esac
done
"#,
            marker = sh_arg(marker),
        )
    }

    fn spec(bin: &str, is_new: bool, session_id: &str, work_dir: &str) -> PiRpcSpec {
        PiRpcSpec {
            bin: bin.to_string(),
            session_id: session_id.to_string(),
            is_new,
            work_dir: work_dir.to_string(),
            instructions: String::new(),
            model: None,
            thinking: None,
            images: Vec::new(),
            env: Vec::new(),
        }
    }

    fn default_pool(config: PoolConfig, cli: &FakeCli) -> Arc<Pool> {
        let _ = cli;
        Arc::new(Pool::new(config))
    }

    #[tokio::test]
    async fn lazy_spawn_reuses_one_child_for_two_turns() {
        let marker = temp_path("pool-spawn-marker");
        let cli = FakeCli::new("pi", &fake_rpc(&marker, "sess-1", 0.0));
        let work_dir = crate::test_support::temp_dir("pool-reuse-work");
        let pool = default_pool(PoolConfig::default(), &cli);
        let dir = work_dir.to_str().unwrap();

        let first = pool
            .run(
                "thread-a",
                spec(&cli.bin(), true, "", dir),
                "one",
                Duration::from_secs(5),
                |_| UiResponse::Cancelled,
            )
            .await
            .unwrap();
        let second = pool
            .run(
                "thread-a",
                spec(&cli.bin(), false, "sess-1", dir),
                "two",
                Duration::from_secs(5),
                |_| UiResponse::Cancelled,
            )
            .await
            .unwrap();

        assert_eq!(first.reply, "reply");
        assert_eq!(first.session_id.as_deref(), Some("sess-1"));
        assert_eq!(second.reply, "reply");
        assert_eq!(second.session_id, None);
        // One child served both turns.
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "x");
        assert_eq!(pool.live_sessions(), 1);
    }

    #[tokio::test]
    async fn idle_eviction_reaps_a_quiet_child() {
        let marker = temp_path("pool-idle-marker");
        let cli = FakeCli::new("pi", &fake_rpc(&marker, "sess-1", 0.0));
        let work_dir = crate::test_support::temp_dir("pool-idle-work");
        let pool = Pool::new(PoolConfig {
            max_sessions: 4,
            idle_timeout: Duration::from_millis(30),
        });
        let dir = work_dir.to_str().unwrap();

        pool.run(
            "thread-a",
            spec(&cli.bin(), true, "", dir),
            "one",
            Duration::from_secs(5),
            |_| UiResponse::Cancelled,
        )
        .await
        .unwrap();
        assert_eq!(pool.live_sessions(), 1);

        tokio::time::sleep(Duration::from_millis(60)).await;
        pool.sweep_idle().await;
        assert_eq!(pool.live_sessions(), 0);
    }

    #[tokio::test]
    async fn eviction_never_fires_while_a_run_is_in_flight() {
        let marker = temp_path("pool-busy-marker");
        let cli = FakeCli::new("pi", &fake_rpc(&marker, "sess-1", 0.3));
        let work_dir = crate::test_support::temp_dir("pool-busy-work");
        let pool = Arc::new(Pool::new(PoolConfig {
            max_sessions: 4,
            idle_timeout: Duration::from_millis(100),
        }));
        let dir = work_dir.to_str().unwrap().to_string();
        let bin = cli.bin();

        // First turn spawns and stores the child.
        pool.run(
            "thread-a",
            spec(&bin, true, "", &dir),
            "one",
            Duration::from_secs(5),
            |_| UiResponse::Cancelled,
        )
        .await
        .unwrap();
        assert_eq!(pool.live_sessions(), 1);

        // Second turn moves the child out of the map for the duration.
        let running = {
            let pool = pool.clone();
            let bin = bin.clone();
            let dir = dir.clone();
            tokio::spawn(async move {
                pool.run(
                    "thread-a",
                    spec(&bin, false, "sess-1", &dir),
                    "two",
                    Duration::from_secs(5),
                    |_| UiResponse::Cancelled,
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(150)).await;
        pool.sweep_idle().await;
        let output = running.await.unwrap().unwrap();
        assert_eq!(output.reply, "reply");
        // The child survived the sweep and is still the thread's.
        assert_eq!(pool.live_sessions(), 1);
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "x");
    }

    #[tokio::test]
    async fn cap_evicts_the_longest_idle_session() {
        let marker = temp_path("pool-cap-marker");
        let cli = FakeCli::new("pi", &fake_rpc(&marker, "sess-1", 0.0));
        let work_dir = crate::test_support::temp_dir("pool-cap-work");
        let pool = Pool::new(PoolConfig {
            max_sessions: 2,
            idle_timeout: Duration::from_secs(3600),
        });
        let dir = work_dir.to_str().unwrap();

        for key in ["a", "b"] {
            pool.run(
                key,
                spec(&cli.bin(), true, "", dir),
                "hello",
                Duration::from_secs(5),
                |_| UiResponse::Cancelled,
            )
            .await
            .unwrap();
            // Order `last_used` so `a` is unambiguously the oldest.
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        pool.run(
            "c",
            spec(&cli.bin(), true, "", dir),
            "hello",
            Duration::from_secs(5),
            |_| UiResponse::Cancelled,
        )
        .await
        .unwrap();

        assert_eq!(pool.live_sessions(), 2);
        assert_eq!(pool.held_keys(), vec!["b".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn a_dead_resumed_child_is_reported_as_session_missing() {
        let work_dir = crate::test_support::temp_dir("pool-missing-work");
        // No get_state answer: pi exits at once, exactly like `--session` for a
        // session it cannot find.
        let cli = FakeCli::new(
            "pi",
            "#!/bin/sh\nprintf '%s\\n' \"No session found matching 'gone'\" >&2\nexit 1\n",
        );
        let pool = Pool::new(PoolConfig::default());
        let error = pool
            .run(
                "thread-a",
                spec(&cli.bin(), false, "gone", work_dir.to_str().unwrap()),
                "one",
                Duration::from_secs(5),
                |_| UiResponse::Cancelled,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, RunError::SessionMissing(_)));
    }
}
