//! Gateway configuration loaded from a TOML file.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::paths::PushPaths;
use crate::util::expand_home;

pub const TELEGRAM_BOT_TOKEN_ENV: &str = "TELEGRAM_BOT_TOKEN";
pub const SLACK_APP_TOKEN_ENV: &str = "SLACK_APP_TOKEN";
pub const SLACK_BOT_TOKEN_ENV: &str = "SLACK_BOT_TOKEN";
pub const DEFAULT_VOICE_NAME: &str = "cedar";
/// Pi's valid `--thinking` levels (kept in sync with the pi CLI).
const VALID_THINKING_LEVELS: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const SUPPORTED_VOICE_NAMES: &[&str] = &[
    "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
    "marin", "cedar",
];

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct AgentCommands {
    pub claude: String,
    pub codex: String,
    pub pi: String,
}

#[cfg(test)]
impl Default for AgentCommands {
    fn default() -> Self {
        Self {
            claude: "claude".to_string(),
            codex: "codex".to_string(),
            pi: "pi".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_channel")]
    pub channel: String,
    #[serde(default)]
    pub channels: Vec<String>,
    #[serde(default)]
    pub primary_delivery: Option<PrimaryDeliveryConfig>,
    #[serde(default = "default_db_path")]
    pub db_path: String,
    /// Adaptive poll floor while the channel is active (PLAN §3).
    #[serde(default = "default_poll_interval")]
    pub poll_interval: String,
    /// Poll interval once the channel goes idle; doubles up to
    /// `poll_max_interval` while it stays quiet (PLAN §3).
    #[serde(default = "default_poll_idle_interval")]
    pub poll_idle_interval: String,
    /// Ceiling for the idle backoff (PLAN §3).
    #[serde(default = "default_poll_max_interval")]
    pub poll_max_interval: String,
    #[serde(default = "default_run_timeout")]
    pub run_timeout: String,
    #[serde(default)]
    pub self_handles: Vec<String>,
    #[serde(default)]
    pub allow_from: Vec<String>,
    #[serde(default)]
    pub telegram_bot_token: Option<String>,
    #[serde(default)]
    pub telegram_allow_user_ids: Vec<i64>,
    #[serde(default)]
    pub telegram_allow_chat_ids: Vec<i64>,
    #[serde(default)]
    pub slack_app_token: Option<String>,
    #[serde(default)]
    pub slack_bot_token: Option<String>,
    #[serde(default)]
    pub slack_allow_user_ids: Vec<String>,
    /// Integration hostname for the Discord Bot integration, for example
    /// `https://discord.int.exe.xyz`. **Not a secret**: the integration injects
    /// the bot token server-side, so no token exists on this VM (PLAN §2).
    #[serde(default)]
    pub discord_base_url: Option<String>,
    /// The single guild channel this assistant answers in, also selectable in
    /// `channel` / `channels` as `discord:<guild-channel-id>`.
    #[serde(default)]
    pub discord_channel_id: Option<String>,
    /// Allowlisted guild channel ids, the Discord equivalent of
    /// `telegram_allow_chat_ids`.
    #[serde(default)]
    pub discord_allow_channel_ids: Vec<String>,
    #[serde(default)]
    pub voice_openai_api_key: Option<String>,
    #[serde(default = "default_voice_name")]
    pub voice_name: String,
    #[serde(default = "default_agent")]
    pub agent: String,
    #[serde(default)]
    pub routes: Vec<RouteRule>,
    /// Canonical root of the single user-owned assistant repository.
    #[serde(default)]
    pub assistant_root: String,
    /// Derived from `assistant_root`. Parsed only for legacy migration.
    #[serde(default = "default_jobs_dir")]
    pub jobs_dir: String,
    #[serde(default)]
    pub jobs_agent: Option<String>,
    #[serde(default = "default_jobs_max_timeout")]
    pub jobs_max_timeout: String,
    #[serde(default, rename = "jobs_run_dir")]
    pub(crate) jobs_run_dir_override: Option<String>,
    #[serde(default = "default_jobs_max_workers")]
    pub jobs_max_workers: usize,
    #[serde(default, rename = "state_path")]
    pub(crate) state_path_override: Option<String>,
    #[serde(default, rename = "audit_log_path")]
    pub(crate) audit_log_path_override: Option<String>,
    #[serde(default, rename = "database_path")]
    pub(crate) database_path_override: Option<String>,
    #[serde(default)]
    pub audit_log_content: bool,
    /// Derived from `assistant_root`. Parsed only for legacy migration.
    #[serde(default = "default_assistant_dir")]
    pub assistant_dir: String,
    /// Canonical path of the loaded config file. Set by `load`, never parsed.
    #[serde(skip)]
    pub config_path: String,
    /// Authoritative resolved locations for all Push-owned runtime data.
    #[serde(skip)]
    pub paths: PushPaths,
    /// Test-only command injection point. Never parsed from user configuration.
    #[cfg(test)]
    #[serde(skip)]
    pub(crate) agent_commands: AgentCommands,
}

impl Config {
    /// Load, expand `~` in path fields, and validate the config at `path`.
    pub fn load(path: &str) -> Result<Config> {
        Self::load_with_paths(path, PushPaths::discover()?)
    }

    pub(crate) fn load_with_paths(path: &str, default_paths: PushPaths) -> Result<Config> {
        let expanded_path = expand_home(path);
        let raw = std::fs::read_to_string(&expanded_path)
            .with_context(|| format!("read config {expanded_path}"))?;
        let mut value: toml::Value = toml::from_str(&raw).context("parse TOML config")?;
        let root = value
            .as_table_mut()
            .context("config must be a TOML table")?;
        if root.contains_key("assistant") {
            bail!(
                "structured [assistant] settings are no longer supported; move assistant identity into assistant_root/SOUL.md"
            );
        }
        for removed in ["permission_profile", "permission_profiles"] {
            if root.contains_key(removed) {
                bail!(
                    "{removed} is no longer supported; configure permissions in the selected agent and remove this key from Push config"
                );
            }
        }
        if root
            .get("routes")
            .and_then(toml::Value::as_array)
            .is_some_and(|routes| {
                routes.iter().any(|route| {
                    route
                        .as_table()
                        .is_some_and(|route| route.contains_key("permission_profile"))
                })
            })
        {
            bail!(
                "route permission_profile is no longer supported; configure permissions in the selected agent and remove this key from Push config"
            );
        }
        for legacy in [
            "permission_mode",
            "tools",
            "allowed_tools",
            "disallowed_tools",
            "claude_permission_mode",
            "claude_tools",
            "claude_allowed_tools",
            "claude_disallowed_tools",
            "codex_sandbox",
            "codex_approval_policy",
        ] {
            if root.contains_key(legacy) {
                bail!(
                    "legacy permission setting {legacy:?} is no longer supported; configure permissions in the selected agent and remove this key from Push config"
                );
            }
        }
        let has_removed_telegram_token_env = root.contains_key("telegram_bot_token_env")
            || root
                .get("telegram")
                .and_then(toml::Value::as_table)
                .is_some_and(|telegram| telegram.contains_key("bot_token_env"));
        if has_removed_telegram_token_env {
            bail!(
                "telegram_bot_token_env / telegram.bot_token_env is no longer configurable; set TELEGRAM_BOT_TOKEN or remove this key from Push config"
            );
        }
        for (removed, replacement) in [
            ("claude_bin", "put claude on the service PATH instead"),
            ("codex_bin", "put codex on the service PATH instead"),
            ("pi_bin", "put pi on the service PATH instead"),
            ("codex_model", "configure the model in Codex instead"),
            ("sessions_dir", "remove this key from Push config"),
            ("reply_marker", "remove this key from Push config"),
        ] {
            if root.contains_key(removed) {
                bail!("{removed} is no longer configurable; {replacement}");
            }
        }
        if root.contains_key("job_permission_profiles") {
            bail!(
                "job_permission_profiles is no longer supported; jobs run unattended, so remove this key"
            );
        }
        let has_assistant_root = root.contains_key("assistant_root");
        let has_legacy_assistant_dir = root.contains_key("assistant_dir");
        let has_legacy_jobs_dir = root.contains_key("jobs_dir");
        if has_assistant_root && (has_legacy_assistant_dir || has_legacy_jobs_dir) {
            bail!(
                "assistant_root replaces legacy assistant_dir and jobs_dir; remove the legacy keys after moving SOUL.md, context, and jobs under assistant_root"
            );
        }
        flatten_provider_section(
            root,
            "imessage",
            &[
                ("db_path", "db_path"),
                ("self_handles", "self_handles"),
                ("allow_from", "allow_from"),
            ],
        )?;
        flatten_provider_section(
            root,
            "telegram",
            &[
                ("bot_token", "telegram_bot_token"),
                ("allow_user_ids", "telegram_allow_user_ids"),
                ("allow_chat_ids", "telegram_allow_chat_ids"),
            ],
        )?;
        flatten_provider_section(
            root,
            "slack",
            &[
                ("app_token", "slack_app_token"),
                ("bot_token", "slack_bot_token"),
                ("allow_user_ids", "slack_allow_user_ids"),
            ],
        )?;
        flatten_provider_section(
            root,
            "discord",
            &[
                ("base_url", "discord_base_url"),
                ("channel_id", "discord_channel_id"),
                ("allow_channel_ids", "discord_allow_channel_ids"),
            ],
        )?;
        flatten_provider_section(
            root,
            "voice",
            &[
                ("openai_api_key", "voice_openai_api_key"),
                ("name", "voice_name"),
            ],
        )?;
        let mut c: Config = value.try_into().context("parse TOML config")?;
        apply_discord_channel_selector(&mut c)?;
        let config_path = std::fs::canonicalize(&expanded_path)
            .with_context(|| format!("resolve config {expanded_path}"))?;
        c.db_path = expand_home(&c.db_path);
        c.paths = default_paths.with_overrides(
            c.state_path_override.as_deref(),
            c.database_path_override.as_deref(),
            c.audit_log_path_override.as_deref(),
            c.jobs_run_dir_override.as_deref(),
        )?;
        let assistant_root = if has_assistant_root {
            resolve_assistant_root(&c.assistant_root, &config_path)?
        } else {
            let legacy_root = resolve_assistant_root(&c.assistant_dir, &config_path)?;
            let legacy_jobs = resolved_absolute("jobs_dir", Path::new(&expand_home(&c.jobs_dir)))?;
            let derived_jobs = legacy_root.join("jobs");
            if legacy_jobs != derived_jobs {
                bail!(
                    "legacy assistant_dir ({}) and jobs_dir ({}) do not form one assistant repository. Move SOUL.md, context, and jobs under one directory, then replace both settings with assistant_root = \"/path/to/assistant\"",
                    legacy_root.display(),
                    legacy_jobs.display()
                );
            }
            legacy_root
        };
        if has_assistant_root {
            validate_inline_token_location(
                &config_path,
                &assistant_root,
                c.telegram_bot_token.as_deref(),
            )?;
            validate_inline_slack_token_location(
                &config_path,
                &assistant_root,
                c.slack_app_token.as_deref(),
                c.slack_bot_token.as_deref(),
            )?;
            validate_inline_voice_key_location(
                &config_path,
                &assistant_root,
                c.voice_openai_api_key.as_deref(),
            )?;
        }
        c.assistant_root = assistant_root.to_string_lossy().to_string();
        c.assistant_dir = c.assistant_root.clone();
        c.jobs_dir = assistant_root.join("jobs").to_string_lossy().to_string();
        validate_runtime_outside_assistant(&c)?;
        c.validate()?;
        c.config_path = config_path.to_string_lossy().to_string();
        Ok(c)
    }

    /// Returns the assistant context directory only when it is a real
    /// directory directly beneath the canonical assistant root. This keeps a
    /// repository symlink from widening an agent backend's writable boundary.
    pub fn backend_context_dir(&self) -> Result<Option<PathBuf>> {
        let root = std::fs::canonicalize(&self.assistant_root)
            .with_context(|| format!("resolve assistant_root {}", self.assistant_root))?;
        let context = root.join("context");
        let metadata = match std::fs::symlink_metadata(&context) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect assistant context {}", context.display()));
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!(
                "assistant context {} must be a real directory, not a symlink or file",
                context.display()
            );
        }
        let resolved = std::fs::canonicalize(&context)
            .with_context(|| format!("resolve assistant context {}", context.display()))?;
        if resolved.parent() != Some(root.as_path()) {
            bail!(
                "assistant context {} must stay directly beneath assistant_root {}",
                resolved.display(),
                root.display()
            );
        }
        Ok(Some(resolved))
    }

    pub fn poll_interval_dur(&self) -> Result<Duration> {
        humantime::parse_duration(&self.poll_interval)
            .with_context(|| format!("invalid poll_interval {}", self.poll_interval))
    }

    pub fn poll_idle_interval_dur(&self) -> Result<Duration> {
        humantime::parse_duration(&self.poll_idle_interval)
            .with_context(|| format!("invalid poll_idle_interval {}", self.poll_idle_interval))
    }

    pub fn poll_max_interval_dur(&self) -> Result<Duration> {
        humantime::parse_duration(&self.poll_max_interval)
            .with_context(|| format!("invalid poll_max_interval {}", self.poll_max_interval))
    }

    pub fn run_timeout_dur(&self) -> Result<Duration> {
        humantime::parse_duration(&self.run_timeout)
            .with_context(|| format!("invalid run_timeout {}", self.run_timeout))
    }

    pub fn agent_backend(&self) -> Result<AgentBackend> {
        AgentBackend::parse(&self.agent)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn channel_kind(&self) -> Result<ChannelKind> {
        ChannelKind::parse(&self.channel)
    }

    /// Every enabled channel as a concrete [`ChannelSpec`], one per registered
    /// push channel. Dedup is by `store_key()`, not by kind, so several Discord
    /// guild channels can coexist (PLAN §3 / Phase 4).
    pub fn enabled_channels(&self) -> Result<Vec<ChannelSpec>> {
        let selectors: Vec<&str> = if self.channels.is_empty() {
            vec![self.channel.as_str()]
        } else {
            self.channels.iter().map(String::as_str).collect()
        };
        let mut seen = HashSet::new();
        let mut enabled = Vec::with_capacity(selectors.len());
        for selector in selectors {
            let spec = self.resolve_channel_spec(selector)?;
            if !seen.insert(spec.store_key()) {
                bail!("duplicate enabled channel {selector:?}");
            }
            enabled.push(spec);
        }
        Ok(enabled)
    }

    /// Projection of [`enabled_channels`] onto kinds, for callers that only need
    /// the provider (route matching, job backend discovery).
    pub fn enabled_channel_kinds(&self) -> Result<Vec<ChannelKind>> {
        Ok(self
            .enabled_channels()?
            .into_iter()
            .map(|spec| spec.kind)
            .collect())
    }

    /// Resolves a selector such as `discord:900` or bare `discord` into a
    /// concrete channel. A bare `discord` falls back to `discord.channel_id`.
    pub(crate) fn resolve_channel_spec(&self, selector: &str) -> Result<ChannelSpec> {
        let kind = ChannelKind::parse(selector)?;
        if kind != ChannelKind::Discord {
            return Ok(ChannelSpec {
                kind,
                instance: None,
            });
        }
        match selector.strip_prefix("discord:") {
            Some(id) => {
                let id = id.trim();
                if !is_discord_snowflake(id) {
                    bail!(
                        "discord channel selector {selector:?} must carry a numeric guild channel id"
                    );
                }
                Ok(ChannelSpec {
                    kind,
                    instance: Some(id.to_string()),
                })
            }
            None => Ok(ChannelSpec {
                kind,
                instance: Some(self.discord_channel_id()?),
            }),
        }
    }

    pub fn telegram_token(&self) -> Option<String> {
        self.telegram_bot_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .or_else(|| {
                std::env::var(TELEGRAM_BOT_TOKEN_ENV)
                    .ok()
                    .map(|token| token.trim().to_string())
                    .filter(|token| !token.is_empty())
            })
    }

    pub fn slack_app_token(&self) -> Option<String> {
        configured_secret(self.slack_app_token.as_deref(), SLACK_APP_TOKEN_ENV)
    }

    pub fn slack_bot_token(&self) -> Option<String> {
        configured_secret(self.slack_bot_token.as_deref(), SLACK_BOT_TOKEN_ENV)
    }

    /// The Discord integration hostname, normalized to an `https` URL.
    ///
    /// There is no Discord token setting anywhere in Push: the exe.dev
    /// integration injects the bot token server-side, so this hostname is the
    /// entire credential surface (PLAN §2).
    pub fn discord_base_url(&self) -> Result<String> {
        let configured = self
            .discord_base_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .with_context(|| {
                format!(
                    "set discord.base_url to the integration hostname, for example {:?}",
                    crate::discord::DEFAULT_BASE_URL
                )
            })?;
        let normalized = if configured.contains("://") {
            configured.to_string()
        } else {
            format!("https://{configured}")
        };
        let normalized = normalized.trim_end_matches('/').to_string();
        let url = reqwest::Url::parse(&normalized)
            .with_context(|| format!("invalid discord.base_url {configured:?}"))?;
        if url.scheme() != "https" {
            bail!(
                "discord.base_url must use https; the integration injects the bot token, so it must not be reached in plaintext"
            );
        }
        if url.host_str().is_none_or(str::is_empty) {
            bail!("discord.base_url {configured:?} has no host");
        }
        if !matches!(url.path(), "" | "/") {
            bail!("discord.base_url must be a bare hostname without a path");
        }
        Ok(normalized)
    }

    /// The single guild channel this assistant answers in.
    pub fn discord_channel_id(&self) -> Result<String> {
        let id = self
            .discord_channel_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .context(
                "set discord.channel_id, or select the guild channel with channel = \"discord:<guild-channel-id>\"",
            )?;
        if !is_discord_snowflake(id) {
            bail!("discord.channel_id {id:?} must be a numeric Discord channel id");
        }
        Ok(id.to_string())
    }

    pub fn route_for_message(
        &self,
        channel: &str,
        channel_instance: &str,
        route_thread_groups: &[Vec<String>],
    ) -> Result<RouteSelection> {
        for aliases in route_thread_groups {
            for route in self.routes.iter().filter(|route| route.thread.is_some()) {
                if route.matches_threads(channel, channel_instance, aliases) {
                    return self.resolve_route(route);
                }
            }
        }
        for route in self.routes.iter().filter(|route| route.thread.is_none()) {
            if route.matches(channel, channel_instance) {
                return self.resolve_route(route);
            }
        }
        Ok(RouteSelection {
            backend: self.agent_backend()?,
            model: None,
            thinking: None,
            prompt_file: None,
        })
    }

    pub fn jobs_backend(&self) -> Result<AgentBackend> {
        AgentBackend::parse(self.jobs_agent.as_deref().unwrap_or(&self.agent))
            .context("invalid jobs_agent")
    }

    pub fn jobs_max_timeout_dur(&self) -> Result<Duration> {
        humantime::parse_duration(&self.jobs_max_timeout)
            .with_context(|| format!("invalid jobs_max_timeout {}", self.jobs_max_timeout))
    }

    // Jobs run unattended with write access. The assistant repository is an
    // allowed workdir so jobs can use its context, skills, and runbooks.
    // Push-owned runtime state and configuration stay protected.
    pub fn validate_job_workdir(&self, workdir: &Path) -> Result<()> {
        let workdir = resolved_absolute("job workdir", workdir)?;
        let protected_paths = [
            ("Push home", self.paths.root.as_path()),
            ("jobs_run_dir", self.paths.jobs_run.as_path()),
            ("state_path", self.paths.state.as_path()),
            ("database_path", self.paths.database.as_path()),
            ("audit_log_path", self.paths.audit.as_path()),
            ("Slack inbox", self.paths.inbox.as_path()),
            ("cache directory", self.paths.cache.as_path()),
        ];
        for (label, protected) in protected_paths
            .into_iter()
            .filter(|(_, protected)| !protected.as_os_str().is_empty())
        {
            let protected = resolved_absolute(label, protected)?;
            if paths_overlap(&workdir, &protected) {
                bail!(
                    "job workdir {} overlaps Push-owned {label} {}",
                    workdir.display(),
                    protected.display()
                );
            }
        }
        if !self.config_path.is_empty() {
            let config = resolved_absolute("config file", Path::new(&self.config_path))?;
            let assistant = resolved_absolute("assistant_root", Path::new(&self.assistant_root))?;
            if !config.starts_with(&assistant) && paths_overlap(&workdir, &config) {
                bail!(
                    "job workdir {} overlaps config file {}",
                    workdir.display(),
                    config.display()
                );
            }
        }
        Ok(())
    }

    fn resolve_route(&self, route: &RouteRule) -> Result<RouteSelection> {
        Ok(RouteSelection {
            backend: AgentBackend::parse(&route.agent)?,
            model: route.model.clone(),
            thinking: route.thinking.clone(),
            prompt_file: route.prompt_file.clone(),
        })
    }

    pub fn required_agent_bins(&self) -> Result<Vec<&str>> {
        let mut backends = HashSet::new();
        backends.insert(self.agent_backend()?);
        let enabled = self.enabled_channel_kinds()?;
        for route in self.routes.iter().filter(|route| {
            enabled
                .iter()
                .any(|kind| route.can_match_channel(kind.as_str()))
        }) {
            backends.insert(AgentBackend::parse(&route.agent)?);
        }
        if let Some(jobs_agent) = self.jobs_agent.as_deref() {
            backends.insert(AgentBackend::parse(jobs_agent).context("invalid jobs_agent")?);
        }

        let mut bins = Vec::new();
        for backend in backends {
            bins.push(self.agent_bin(backend));
        }
        Ok(bins)
    }

    pub fn agent_bin(&self, backend: AgentBackend) -> &str {
        #[cfg(test)]
        {
            match backend {
                AgentBackend::Claude => self.agent_commands.claude.as_str(),
                AgentBackend::Codex => self.agent_commands.codex.as_str(),
                AgentBackend::Pi => self.agent_commands.pi.as_str(),
            }
        }
        #[cfg(not(test))]
        {
            backend.as_str()
        }
    }

    fn validate(&self) -> Result<()> {
        if self
            .voice_openai_api_key
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            bail!("voice.openai_api_key cannot be empty");
        }
        if !SUPPORTED_VOICE_NAMES.contains(&self.voice_name.as_str()) {
            bail!(
                "invalid voice.name {:?}; expected one of: {}",
                self.voice_name,
                SUPPORTED_VOICE_NAMES.join(", ")
            );
        }
        for spec in self.enabled_channels()? {
            match spec.kind {
                ChannelKind::IMessage => {
                    if self.self_handles.is_empty() && self.allow_from.is_empty() {
                        bail!("set imessage.self_handles or imessage.allow_from for iMessage");
                    }
                }
                ChannelKind::Telegram => {
                    if self.telegram_allow_user_ids.is_empty()
                        && self.telegram_allow_chat_ids.is_empty()
                    {
                        bail!(
                            "set telegram.allow_user_ids or telegram.allow_chat_ids for Telegram"
                        );
                    }
                    if self
                        .telegram_bot_token
                        .as_deref()
                        .is_some_and(|v| v.trim().is_empty())
                    {
                        bail!("telegram.bot_token cannot be empty");
                    }
                }
                ChannelKind::Slack => {
                    if self.slack_allow_user_ids.is_empty()
                        || self
                            .slack_allow_user_ids
                            .iter()
                            .any(|user| user.trim().is_empty())
                    {
                        bail!("set slack.allow_user_ids to explicit Slack user IDs");
                    }
                    if self
                        .slack_app_token
                        .as_deref()
                        .is_some_and(|v| v.trim().is_empty())
                    {
                        bail!("slack.app_token cannot be empty");
                    }
                    if self
                        .slack_bot_token
                        .as_deref()
                        .is_some_and(|v| v.trim().is_empty())
                    {
                        bail!("slack.bot_token cannot be empty");
                    }
                }
                ChannelKind::Discord => {
                    // Discord is allowlisted by guild channel, not by sender:
                    // the integration exposes a bot identity, and the channel
                    // is the operator boundary, exactly as it is for Telegram
                    // chats.
                    self.discord_base_url()?;
                    let channel_id = spec
                        .discord_channel_id()
                        .context("discord channel is missing its guild channel id")?;
                    if self.discord_allow_channel_ids.is_empty() {
                        bail!(
                            "set discord.allow_channel_ids to the guild channel id Discord answers in"
                        );
                    }
                    for allowed in &self.discord_allow_channel_ids {
                        if !is_discord_snowflake(allowed.trim()) {
                            bail!(
                                "discord.allow_channel_ids must contain numeric Discord channel ids, found {allowed:?}"
                            );
                        }
                    }
                    if !self
                        .discord_allow_channel_ids
                        .iter()
                        .any(|allowed| allowed.trim() == channel_id)
                    {
                        bail!("discord channel {channel_id} is not in discord.allow_channel_ids");
                    }
                }
            }
        }
        self.agent_backend()?;
        self.jobs_backend()?;
        if self.jobs_max_timeout_dur()?.is_zero() {
            bail!("jobs_max_timeout must be positive");
        }
        if self.jobs_dir.trim().is_empty() || self.paths.jobs_run.as_os_str().is_empty() {
            bail!("jobs_dir and jobs_run_dir cannot be empty");
        }
        if self.jobs_max_workers == 0 {
            bail!("jobs_max_workers must be positive");
        }
        for route in &self.routes {
            let backend = AgentBackend::parse(&route.agent)
                .with_context(|| format!("invalid route agent for {route:?}"))?;
            if (route.model.is_some() || route.thinking.is_some()) && backend != AgentBackend::Pi {
                bail!("route model/thinking are only supported for the pi backend");
            }
            if route.thread.is_none() && route.channel.is_none() {
                bail!("route must set thread or channel");
            }
            if route.thread.as_deref().is_some_and(|v| v.trim().is_empty()) {
                bail!("route thread cannot be empty");
            }
            if let Some(channel) = &route.channel {
                ChannelKind::parse(channel).context("invalid route channel")?;
            }
            if let Some(thinking) = route.thinking.as_deref() {
                if !VALID_THINKING_LEVELS.contains(&thinking.trim()) {
                    bail!(
                        "route thinking {thinking:?} must be one of: {}",
                        VALID_THINKING_LEVELS.join(", ")
                    );
                }
            }
            if route.model.as_deref().is_some_and(|m| m.trim().is_empty()) {
                bail!("route model cannot be empty");
            }
            if let Some(prompt_file) = route.prompt_file.as_deref() {
                let prompt_file = prompt_file.trim();
                if prompt_file.is_empty() {
                    bail!("route prompt_file cannot be empty");
                }
                let path = Path::new(prompt_file);
                if path.is_absolute()
                    || path
                        .components()
                        .any(|component| !matches!(component, Component::Normal(_)))
                {
                    bail!(
                        "route prompt_file {prompt_file:?} must be a plain relative path inside assistant_root"
                    );
                }
                let root = resolved_absolute("assistant_root", Path::new(&self.assistant_root))?;
                let resolved = root.join(path);
                if !resolved.is_file() {
                    bail!(
                        "route prompt_file {prompt_file:?} does not exist under assistant_root ({})",
                        resolved.display()
                    );
                }
            }
        }
        let poll_active = self.poll_interval_dur()?;
        let poll_idle = self.poll_idle_interval_dur()?;
        let poll_max = self.poll_max_interval_dur()?;
        if poll_active > poll_idle || poll_idle > poll_max {
            bail!(
                "poll intervals must satisfy poll_interval <= poll_idle_interval <= poll_max_interval"
            );
        }
        self.run_timeout_dur()?;
        Ok(())
    }
}

fn validate_runtime_outside_assistant(cfg: &Config) -> Result<()> {
    let assistant = resolved_absolute("assistant_root", Path::new(&cfg.assistant_root))?;
    let push_home = resolved_absolute("PUSH_HOME", &cfg.paths.root)?;
    if paths_overlap(&assistant, &push_home) {
        bail!(
            "assistant_root {} must stay outside Push home {}; choose a separate assistant repository or set PUSH_HOME to a separate runtime directory",
            assistant.display(),
            push_home.display()
        );
    }
    let jobs_run = resolved_absolute("jobs_run_dir", &cfg.paths.jobs_run)?;
    if paths_overlap(&assistant, &jobs_run) {
        bail!("jobs_run_dir must stay outside assistant_root");
    }
    for (label, value) in [
        ("state_path", cfg.paths.state.as_path()),
        ("database_path", cfg.paths.database.as_path()),
        ("audit_log_path", cfg.paths.audit.as_path()),
        ("Slack inbox", cfg.paths.inbox.as_path()),
        ("cache directory", cfg.paths.cache.as_path()),
    ] {
        let path = resolved_absolute(label, value)?;
        if path.starts_with(&assistant) {
            bail!("{label} must stay outside assistant_root");
        }
    }
    Ok(())
}

pub(crate) fn validate_inline_token_location(
    config_path: &Path,
    assistant_root: &Path,
    token: Option<&str>,
) -> Result<()> {
    if token.is_some_and(|token| !token.trim().is_empty())
        && config_path.starts_with(assistant_root)
    {
        bail!(
            "config {} contains an inline Telegram token inside the Git-versioned assistant repository. Move the config outside the assistant or use TELEGRAM_BOT_TOKEN.",
            config_path.display()
        );
    }
    Ok(())
}

pub(crate) fn validate_inline_voice_key_location(
    config_path: &Path,
    assistant_root: &Path,
    key: Option<&str>,
) -> Result<()> {
    if key.is_some_and(|key| !key.trim().is_empty()) && config_path.starts_with(assistant_root) {
        bail!(
            "config {} contains an inline OpenAI API key inside the Git-versioned assistant repository. Move the config outside the assistant or use OPENAI_API_KEY.",
            config_path.display()
        );
    }
    Ok(())
}

pub(crate) fn validate_inline_slack_token_location(
    config_path: &Path,
    assistant_root: &Path,
    app_token: Option<&str>,
    bot_token: Option<&str>,
) -> Result<()> {
    if [app_token, bot_token]
        .into_iter()
        .flatten()
        .any(|token| !token.trim().is_empty())
        && config_path.starts_with(assistant_root)
    {
        bail!(
            "config {} contains inline Slack tokens inside the Git-versioned assistant repository. Move the config outside the assistant or use SLACK_APP_TOKEN and SLACK_BOT_TOKEN.",
            config_path.display()
        );
    }
    Ok(())
}

fn resolve_assistant_root(value: &str, config_path: &Path) -> Result<PathBuf> {
    let expanded = expand_home(value);
    if expanded.trim().is_empty() {
        bail!("assistant_root cannot be empty");
    }
    let configured = Path::new(&expanded);
    let candidate = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        config_path
            .parent()
            .context("config path has no parent directory")?
            .join(configured)
    };
    resolved_absolute("assistant_root", &candidate)
}

fn normalized_absolute(label: &str, path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("{label} must be an absolute path");
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => bail!("{label} cannot contain '..'"),
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn resolved_absolute(label: &str, path: &Path) -> Result<PathBuf> {
    let normalized = normalized_absolute(label, path)?;
    let mut existing = normalized.as_path();
    let mut missing = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .with_context(|| format!("{label} has no existing ancestor"))?;
        missing.push(name.to_os_string());
        existing = existing
            .parent()
            .with_context(|| format!("{label} has no existing ancestor"))?;
    }
    let mut resolved = std::fs::canonicalize(existing)
        .with_context(|| format!("resolve existing ancestor for {label}"))?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn flatten_provider_section(
    root: &mut toml::Table,
    section: &str,
    fields: &[(&str, &str)],
) -> Result<()> {
    let Some(value) = root.remove(section) else {
        return Ok(());
    };
    let table = value
        .as_table()
        .with_context(|| format!("[{section}] must be a TOML table"))?;

    for (key, value) in table {
        let Some((_, destination)) = fields.iter().find(|(source, _)| source == key) else {
            let expected = fields
                .iter()
                .map(|(source, _)| *source)
                .collect::<Vec<_>>()
                .join(", ");
            bail!("unknown [{section}] setting {key:?}; expected one of: {expected}");
        };
        if root.contains_key(*destination) {
            bail!("set {destination} either at the top level or as [{section}].{key}, not both");
        }
        root.insert((*destination).to_string(), value.clone());
    }
    Ok(())
}

#[derive(Debug, Deserialize, Clone)]
pub struct RouteRule {
    #[serde(default)]
    pub thread: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    pub agent: String,
    /// Per-channel Pi model pattern (PLAN §3). `None` uses the CLI default.
    #[serde(default)]
    pub model: Option<String>,
    /// Per-channel Pi thinking level (PLAN §3). `None` uses the CLI default.
    #[serde(default)]
    pub thinking: Option<String>,
    /// Per-channel system-prompt file, relative to `assistant_root`, appended to
    /// the system instructions for this channel's runs (PLAN §3).
    #[serde(default)]
    pub prompt_file: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
pub struct PrimaryDeliveryConfig {
    pub channel: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteSelection {
    pub backend: AgentBackend,
    pub model: Option<String>,
    pub thinking: Option<String>,
    pub prompt_file: Option<String>,
}

impl RouteRule {
    fn can_match_channel(&self, channel: &str) -> bool {
        self.channel.as_deref().is_none_or(|value| value == channel)
    }

    /// A route matches by provider kind (`"discord"`) or by the concrete
    /// channel instance (`"discord:900"`), so a per-guild-channel route can
    /// carry its own model/thinking (PLAN §3).
    fn matches_channel_or_instance(&self, channel: &str, instance: &str) -> bool {
        self.can_match_channel(channel) || self.can_match_channel(instance)
    }

    fn matches_threads(&self, channel: &str, instance: &str, threads: &[String]) -> bool {
        if !self.matches_channel_or_instance(channel, instance) {
            return false;
        }
        self.thread
            .as_deref()
            .is_some_and(|value| threads.iter().any(|thread| value == thread))
    }

    fn matches(&self, channel: &str, instance: &str) -> bool {
        self.matches_channel_or_instance(channel, instance) && self.thread.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelKind {
    IMessage,
    Telegram,
    Slack,
    Discord,
}

impl ChannelKind {
    /// Parses a channel selector. Discord additionally accepts
    /// `discord:<guild-channel-id>`, because each guild channel is registered
    /// as its own Push channel (PLAN §3). The id is recorded on the config by
    /// [`apply_discord_channel_selector`]; the kind itself stays data-less until
    /// multi-channel Discord arrives.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "imessage" => Ok(Self::IMessage),
            "telegram" => Ok(Self::Telegram),
            "slack" => Ok(Self::Slack),
            "discord" => Ok(Self::Discord),
            other if other.starts_with("discord:") => {
                let id = &other["discord:".len()..];
                if !is_discord_snowflake(id) {
                    bail!(
                        "invalid discord channel selector {other:?}; expected \"discord\" or \"discord:<guild-channel-id>\""
                    );
                }
                Ok(Self::Discord)
            }
            other => bail!(
                "invalid channel {other:?}; expected \"imessage\", \"telegram\", \"slack\", \"discord\", or \"discord:<guild-channel-id>\""
            ),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::IMessage => "imessage",
            Self::Telegram => "telegram",
            Self::Slack => "slack",
            Self::Discord => "discord",
        }
    }
}

/// A concrete enabled channel: its provider kind plus the instance id for kinds
/// that support more than one instance (Discord's guild channel id).
///
/// `ChannelKind` stays data-less and `Copy` so route matching and every `match`
/// arm over kinds is unchanged; the instance lives here, where the channel
/// registry needs it (PLAN §3 / Phase 4).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChannelSpec {
    pub kind: ChannelKind,
    /// `Some(guild-channel-id)` for Discord, `None` for single-instance kinds.
    pub instance: Option<String>,
}

impl ChannelSpec {
    /// The durable identity used for cursors, history, jobs, and the audit log.
    /// Unique per instance, unlike the bare kind, so two Discord guild channels
    /// cannot collide.
    pub fn store_key(&self) -> String {
        match (self.kind, self.instance.as_deref()) {
            (ChannelKind::Discord, Some(id)) => format!("discord:{id}"),
            _ => self.kind.as_str().to_string(),
        }
    }

    pub fn discord_channel_id(&self) -> Option<&str> {
        if self.kind == ChannelKind::Discord {
            self.instance.as_deref()
        } else {
            None
        }
    }
}

/// A Discord snowflake: all digits and small enough to stay an exact `i64`,
/// which is what push uses as a channel row id.
fn is_discord_snowflake(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && value.len() <= 19
        && value.chars().all(|character| character.is_ascii_digit())
        && value.parse::<i64>().is_ok()
}

/// Records the guild channel id carried by a `discord:<id>` selector on the
/// **primary** `channel` setting, so `channel = "discord:123"` is enough to
/// configure a single Discord channel (and a bare `discord` can fall back to
/// `discord.channel_id`). The `channels` list is deliberately *not* collapsed
/// here: several `discord:<id>` selectors each resolve their own id through
/// [`Config::enabled_channels`] (PLAN §3 / Phase 4). An explicit
/// `discord.channel_id` that disagrees with the primary selector is still a
/// configuration error rather than a silent winner.
fn apply_discord_channel_selector(cfg: &mut Config) -> Result<()> {
    let selector = cfg.channel.as_str();
    let Some(id) = selector.strip_prefix("discord:") else {
        return Ok(());
    };
    let id = id.trim();
    if !is_discord_snowflake(id) {
        bail!("discord channel selector {selector:?} must carry a numeric guild channel id");
    }
    match cfg.discord_channel_id.as_deref().map(str::trim) {
        Some(existing) if !existing.is_empty() && existing != id => {
            bail!(
                "discord channel selector {selector:?} conflicts with discord.channel_id {existing:?}"
            )
        }
        _ => cfg.discord_channel_id = Some(id.to_string()),
    }
    Ok(())
}

fn configured_secret(inline: Option<&str>, environment: &str) -> Option<String> {
    inline
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(environment)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentBackend {
    Claude,
    Codex,
    Pi,
}

impl AgentBackend {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "claude" => Ok(AgentBackend::Claude),
            "codex" => Ok(AgentBackend::Codex),
            "pi" => Ok(AgentBackend::Pi),
            other => bail!("invalid agent {other:?}; expected \"claude\", \"codex\", or \"pi\""),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AgentBackend::Claude => "claude",
            AgentBackend::Codex => "codex",
            AgentBackend::Pi => "pi",
        }
    }
}

fn default_db_path() -> String {
    "~/Library/Messages/chat.db".to_string()
}
fn default_channel() -> String {
    "imessage".to_string()
}
fn default_poll_interval() -> String {
    "2s".to_string()
}
fn default_poll_idle_interval() -> String {
    "30s".to_string()
}
fn default_poll_max_interval() -> String {
    "60s".to_string()
}
fn default_run_timeout() -> String {
    "10m".to_string()
}
fn default_agent() -> String {
    "claude".to_string()
}
fn default_voice_name() -> String {
    DEFAULT_VOICE_NAME.to_string()
}
fn default_jobs_dir() -> String {
    "~/.push/jobs".to_string()
}

fn default_jobs_max_timeout() -> String {
    "30m".to_string()
}
fn default_jobs_max_workers() -> usize {
    2
}
fn default_assistant_dir() -> String {
    "~/.push".to_string()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_dir;

    fn config() -> Config {
        let root = temp_dir("config-draft-boundary");
        Config {
            channel: "imessage".to_string(),
            channels: Vec::new(),
            primary_delivery: None,
            db_path: root.join("chat.db").to_string_lossy().to_string(),
            poll_interval: "1s".to_string(),
            poll_idle_interval: "30s".to_string(),
            poll_max_interval: "60s".to_string(),
            run_timeout: "1s".to_string(),
            self_handles: vec!["me@example.com".to_string()],
            allow_from: Vec::new(),
            telegram_bot_token: None,
            telegram_allow_user_ids: Vec::new(),
            telegram_allow_chat_ids: Vec::new(),
            slack_app_token: None,
            slack_bot_token: None,
            slack_allow_user_ids: Vec::new(),
            discord_base_url: None,
            discord_channel_id: None,
            discord_allow_channel_ids: Vec::new(),
            voice_openai_api_key: None,
            voice_name: DEFAULT_VOICE_NAME.to_string(),
            agent: "codex".to_string(),
            routes: Vec::new(),
            assistant_root: root.to_string_lossy().to_string(),
            jobs_dir: root.join("jobs").to_string_lossy().to_string(),
            jobs_agent: None,
            jobs_max_timeout: "30m".to_string(),
            jobs_run_dir_override: None,
            jobs_max_workers: 2,
            state_path_override: None,
            audit_log_path_override: None,
            database_path_override: None,
            audit_log_content: false,
            config_path: String::new(),
            paths: PushPaths::from_root(root.join("runtime")).unwrap(),
            agent_commands: AgentCommands::default(),
            assistant_dir: root.to_string_lossy().to_string(),
        }
    }

    #[test]
    fn discord_channel_selector_loads_from_toml() {
        let root = temp_dir("config-discord-load");
        let assistant = root.join("assistant");
        std::fs::create_dir_all(&assistant).unwrap();
        let path = root.join("config.toml");
        std::fs::write(
            &path,
            format!(
                "channel = \"discord:900\"\nagent = \"pi\"\nassistant_root = \"{}\"\n\n[discord]\nbase_url = \"discord.int.exe.xyz\"\nallow_channel_ids = [\"900\"]\n",
                assistant.to_string_lossy()
            ),
        )
        .unwrap();

        let cfg = Config::load_with_paths(
            path.to_str().unwrap(),
            PushPaths::from_root(root.join("runtime")).unwrap(),
        )
        .unwrap();

        assert_eq!(
            cfg.enabled_channel_kinds().unwrap(),
            vec![ChannelKind::Discord]
        );
        assert_eq!(cfg.channel_kind().unwrap(), ChannelKind::Discord);
        assert_eq!(cfg.discord_channel_id().unwrap(), "900");
        assert_eq!(
            cfg.discord_base_url().unwrap(),
            "https://discord.int.exe.xyz"
        );

        // Phase 4: several guild channels coexist, each its own push channel.
        // Dedup is by `store_key`, not by kind.
        std::fs::write(
            &path,
            format!(
                "channels = [\"discord:900\", \"discord:901\"]\nagent = \"pi\"\nassistant_root = \"{}\"\n\n[discord]\nbase_url = \"discord.int.exe.xyz\"\nallow_channel_ids = [\"900\", \"901\"]\n",
                assistant.to_string_lossy()
            ),
        )
        .unwrap();
        let multi = Config::load_with_paths(
            path.to_str().unwrap(),
            PushPaths::from_root(root.join("runtime")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            multi.enabled_channel_kinds().unwrap(),
            vec![ChannelKind::Discord, ChannelKind::Discord]
        );
        assert_eq!(
            multi
                .enabled_channels()
                .unwrap()
                .iter()
                .map(ChannelSpec::store_key)
                .collect::<Vec<_>>(),
            vec!["discord:900".to_string(), "discord:901".to_string()]
        );
        // A repeated selector is still rejected.
        std::fs::write(
            &path,
            format!(
                "channels = [\"discord:900\", \"discord:900\"]\nagent = \"pi\"\nassistant_root = \"{}\"\n\n[discord]\nbase_url = \"discord.int.exe.xyz\"\nallow_channel_ids = [\"900\"]\n",
                assistant.to_string_lossy()
            ),
        )
        .unwrap();
        let duplicate = Config::load_with_paths(
            path.to_str().unwrap(),
            PushPaths::from_root(root.join("runtime")).unwrap(),
        )
        .unwrap_err();
        assert!(
            duplicate.to_string().contains("duplicate enabled channel"),
            "{duplicate}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn discord_selectors_carry_the_guild_channel_and_are_validated() {
        assert_eq!(ChannelKind::parse("discord").unwrap(), ChannelKind::Discord);
        assert_eq!(
            ChannelKind::parse("discord:900").unwrap(),
            ChannelKind::Discord
        );
        assert!(ChannelKind::parse("discord").unwrap().as_str() == "discord");
        assert!(ChannelKind::parse("discord:").is_err());
        assert!(ChannelKind::parse("discord:main").is_err());
        assert!(ChannelKind::parse("discordd").is_err());

        let mut cfg = config();
        cfg.channel = "discord:900".to_string();
        cfg.discord_base_url = Some("discord.int.exe.xyz".to_string());
        cfg.discord_allow_channel_ids = vec!["900".to_string()];
        apply_discord_channel_selector(&mut cfg).unwrap();

        assert_eq!(cfg.discord_channel_id.as_deref(), Some("900"));
        assert_eq!(cfg.channel_kind().unwrap(), ChannelKind::Discord);
        assert_eq!(
            cfg.discord_base_url().unwrap(),
            "https://discord.int.exe.xyz"
        );
        cfg.validate().unwrap();

        // A conflicting explicit channel id is rejected rather than guessed.
        let mut conflict = cfg.clone();
        conflict.discord_channel_id = Some("901".to_string());
        assert!(apply_discord_channel_selector(&mut conflict).is_err());

        // The channel id must be allowlisted...
        let mut unlisted = config();
        unlisted.channel = "discord:900".to_string();
        unlisted.discord_base_url = Some("discord.int.exe.xyz".to_string());
        unlisted.discord_allow_channel_ids = vec!["901".to_string()];
        apply_discord_channel_selector(&mut unlisted).unwrap();
        assert!(unlisted
            .validate()
            .unwrap_err()
            .to_string()
            .contains("not in discord.allow_channel_ids"));

        // ...the integration hostname must be present and https...
        let mut insecure = config();
        insecure.channel = "discord".to_string();
        insecure.discord_base_url = Some("http://discord.int.exe.xyz".to_string());
        insecure.discord_channel_id = Some("900".to_string());
        insecure.discord_allow_channel_ids = vec!["900".to_string()];
        assert!(insecure
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must use https"));

        // ...and a channel id is required one way or the other.
        let mut missing = config();
        missing.channel = "discord".to_string();
        missing.discord_base_url = Some("discord.int.exe.xyz".to_string());
        missing.discord_allow_channel_ids = vec!["900".to_string()];
        assert!(missing
            .validate()
            .unwrap_err()
            .to_string()
            .contains("discord.channel_id"));
    }

    #[test]
    fn pi_parses_and_is_selectable_for_default_routes_and_jobs() {
        let mut cfg = config();
        cfg.agent = "pi".to_string();
        cfg.jobs_agent = Some("pi".to_string());
        cfg.routes = vec![RouteRule {
            prompt_file: None,
            model: None,
            thinking: None,
            thread: Some("imessage:chat:pi".to_string()),
            channel: Some("imessage".to_string()),
            agent: "pi".to_string(),
        }];

        assert_eq!(AgentBackend::parse("pi").unwrap(), AgentBackend::Pi);
        assert_eq!(AgentBackend::Pi.as_str(), "pi");
        assert_eq!(cfg.agent_backend().unwrap(), AgentBackend::Pi);
        assert_eq!(cfg.jobs_backend().unwrap(), AgentBackend::Pi);
        assert_eq!(
            cfg.route_for_message(
                "imessage",
                "imessage",
                &[vec!["imessage:chat:pi".to_string()]]
            )
            .unwrap()
            .backend,
            AgentBackend::Pi
        );
        assert_eq!(cfg.required_agent_bins().unwrap(), vec!["pi"]);
    }

    #[test]
    fn pi_binary_defaults_to_pi_when_loading_toml() {
        let cfg: Config = toml::from_str("agent = 'pi'").unwrap();

        assert_eq!(cfg.agent_backend().unwrap(), AgentBackend::Pi);
        assert_eq!(cfg.agent_bin(AgentBackend::Pi), "pi");
    }

    #[test]
    fn chat_run_timeout_defaults_to_ten_minutes_and_accepts_an_override() {
        let default: Config = toml::from_str("agent = 'codex'").unwrap();
        let overridden: Config = toml::from_str("agent = 'codex'\nrun_timeout = '45s'").unwrap();

        assert_eq!(default.run_timeout, "10m");
        assert_eq!(default.run_timeout_dur().unwrap(), Duration::from_secs(600));
        assert_eq!(
            overridden.run_timeout_dur().unwrap(),
            Duration::from_secs(45)
        );
    }

    #[test]
    fn pi_binary_is_only_required_when_selected() {
        let mut cfg = config();
        assert_eq!(cfg.required_agent_bins().unwrap(), vec!["codex"]);

        cfg.routes.push(RouteRule {
            prompt_file: None,
            model: None,
            thinking: None,
            thread: None,
            channel: Some("telegram".to_string()),
            agent: "pi".to_string(),
        });
        assert_eq!(cfg.required_agent_bins().unwrap(), vec!["codex"]);

        cfg.jobs_agent = Some("pi".to_string());
        let mut bins = cfg.required_agent_bins().unwrap();
        bins.sort_unstable();
        assert_eq!(bins, vec!["codex", "pi"]);
    }

    #[test]
    fn job_workdir_must_not_contain_the_loaded_config_file() {
        let mut cfg = config();
        let workdir = crate::test_support::temp_dir("config-shield-workdir");
        cfg.config_path = workdir.join("config.toml").to_string_lossy().to_string();

        let error = cfg.validate_job_workdir(&workdir).unwrap_err();
        assert!(error.to_string().contains("config file"));

        let sibling = crate::test_support::temp_dir("config-shield-sibling");
        assert!(cfg.validate_job_workdir(&sibling).is_ok());
        let _ = std::fs::remove_dir_all(workdir);
        let _ = std::fs::remove_dir_all(sibling);
    }

    #[test]
    fn assistant_root_is_a_valid_job_workdir() {
        let mut cfg = config();
        let assistant = crate::test_support::temp_dir("job-assistant-workdir");
        cfg.assistant_root = assistant.to_string_lossy().to_string();
        cfg.assistant_dir = cfg.assistant_root.clone();
        cfg.jobs_dir = assistant.join("jobs").to_string_lossy().to_string();
        cfg.config_path = assistant.join("config.toml").to_string_lossy().to_string();

        assert!(cfg.validate_job_workdir(&assistant).is_ok());

        let _ = std::fs::remove_dir_all(assistant);
    }

    #[cfg(unix)]
    #[test]
    fn backend_context_rejects_a_symlink_outside_the_assistant() {
        use std::os::unix::fs::symlink;

        let mut cfg = config();
        let assistant = crate::test_support::temp_dir("config-context-assistant");
        let outside = crate::test_support::temp_dir("config-context-outside");
        symlink(&outside, assistant.join("context")).unwrap();
        cfg.assistant_root = assistant.to_string_lossy().to_string();
        cfg.assistant_dir = cfg.assistant_root.clone();

        let error = cfg.backend_context_dir().unwrap_err();

        assert!(error.to_string().contains("not a symlink"));
        let _ = std::fs::remove_dir_all(assistant);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn poll_intervals_must_be_ordered() {
        let mut cfg = config();
        cfg.agent = "pi".to_string();
        cfg.jobs_agent = Some("pi".to_string());

        cfg.poll_interval = "30s".to_string();
        cfg.poll_idle_interval = "2s".to_string();
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("poll_interval <= poll_idle_interval <= poll_max_interval"));

        cfg.poll_interval = "2s".to_string();
        cfg.poll_idle_interval = "30s".to_string();
        cfg.poll_max_interval = "10s".to_string();
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("poll_interval <= poll_idle_interval <= poll_max_interval"));
    }

    #[test]
    fn route_can_carry_model_and_thinking_for_a_guild_channel() {
        let mut cfg = config();
        cfg.agent = "pi".to_string();
        cfg.jobs_agent = Some("pi".to_string());
        cfg.channels = vec!["discord:900".to_string(), "discord:901".to_string()];
        cfg.discord_base_url = Some("discord.int.exe.xyz".to_string());
        cfg.discord_allow_channel_ids = vec!["900".to_string(), "901".to_string()];
        cfg.routes = vec![RouteRule {
            prompt_file: None,
            thread: None,
            channel: Some("discord:901".to_string()),
            agent: "pi".to_string(),
            model: Some("gemini-3.8-live-extended-thinking".to_string()),
            thinking: Some("high".to_string()),
        }];
        cfg.validate().unwrap();

        // The route matches the concrete guild channel, not just the kind.
        let selected = cfg
            .route_for_message("discord", "discord:901", &[vec!["discord:901".to_string()]])
            .unwrap();
        assert_eq!(
            selected.model.as_deref(),
            Some("gemini-3.8-live-extended-thinking")
        );
        assert_eq!(selected.thinking.as_deref(), Some("high"));

        // A different guild channel falls through to the CLI defaults.
        let other = cfg
            .route_for_message("discord", "discord:900", &[vec!["discord:900".to_string()]])
            .unwrap();
        assert_eq!(other.model, None);
        assert_eq!(other.thinking, None);

        // Thinking must be a valid pi level.
        let mut bad = cfg.clone();
        bad.routes[0].thinking = Some("very-high".to_string());
        assert!(bad.validate().unwrap_err().to_string().contains("thinking"));

        // model/thinking belong to the pi backend only.
        let mut non_pi = cfg.clone();
        non_pi.routes[0].agent = "codex".to_string();
        assert!(non_pi
            .validate()
            .unwrap_err()
            .to_string()
            .contains("only supported for the pi backend"));
    }

    #[test]
    fn route_prompt_file_must_be_relative_and_exist_under_assistant_root() {
        let mut cfg = config();
        cfg.agent = "pi".to_string();
        cfg.jobs_agent = Some("pi".to_string());
        cfg.discord_base_url = Some("discord.int.exe.xyz".to_string());
        cfg.discord_channel_id = Some("900".to_string());
        cfg.discord_allow_channel_ids = vec!["900".to_string()];

        let root = std::path::PathBuf::from(&cfg.assistant_root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("main.md"), "Be terse.\n").unwrap();

        cfg.routes = vec![RouteRule {
            thread: None,
            channel: Some("discord:900".to_string()),
            agent: "pi".to_string(),
            model: None,
            thinking: None,
            prompt_file: Some("main.md".to_string()),
        }];
        cfg.validate().unwrap();

        cfg.routes[0].prompt_file = Some("missing.md".to_string());
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("does not exist under assistant_root"));

        for escape in ["../secret.md", "/etc/passwd", "sub/../x.md"] {
            cfg.routes[0].prompt_file = Some(escape.to_string());
            assert!(
                cfg.validate()
                    .unwrap_err()
                    .to_string()
                    .contains("plain relative path"),
                "{escape}"
            );
        }
    }
}
