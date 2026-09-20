//! Discord channel backed by the exe.dev Discord Bot integration.
//!
//! `https://discord.int.exe.xyz` is a verbatim REST proxy for `/api/v10` that
//! injects the bot token server-side, so **no token exists on this VM** — this
//! module holds only the integration hostname. The integration cannot deliver
//! Gateway events, so inbound messages are read by REST polling.
//!
//! Everything provider-specific lives here: request pacing, both error shapes,
//! message mapping, the outbound splitter, and the two-step attachment upload.
//! `crate::channel` holds the `ChannelContract` plumbing that keeps the shared
//! gateway provider-neutral.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::channel::{InboundImage, InboundVoice, RawMessage};
use crate::image::{DownloadedImage, MAX_IMAGE_BYTES};
use crate::voice::{AudioClip, MAX_AUDIO_BYTES};

/// Default integration hostname. Not a secret: the token is injected by the
/// proxy, so the VM only ever holds this name (PLAN §2).
pub const DEFAULT_BASE_URL: &str = "https://discord.int.exe.xyz";
/// Discord's REST version prefix, appended to the integration base URL.
pub const API_PREFIX: &str = "/api/v10";

/// Discord's verified per-message ceiling, in Unicode **code points**:
/// `spikes/discord-limits/FINDINGS.md` §1-2 reports 2000 accepted and 2001
/// rejected with `BASE_TYPE_MAX_LENGTH`, 2000 astral emoji accepted although
/// they are 4000 UTF-16 units, and 2000 graphemes of `e`+U+0301 rejected as
/// 4000 code points. `chars().count()` measures exactly that quantity;
/// `str::len()` and `encode_utf16().count()` over-count, and a grapheme count
/// under-counts and would ship rejected messages.
pub const TEXT_LIMIT: usize = 2000;

/// Working budget for prose. 2000 is a ceiling, not a budget: the verified trap
/// is a chunk of exactly 2000 code points that can no longer take a closing
/// `\n``` (2004 -> 400). FINDINGS §6.2 suggests ~1990 for plain text.
pub const TEXT_BUDGET: usize = 1990;

/// Room for the `\n``` that closes a chunk ending inside a fenced code block —
/// exactly four code points, which is why fenced content gets
/// `TEXT_LIMIT - FENCE_RESERVE` (FINDINGS §6.2/§6.3).
const FENCE_RESERVE: usize = 4;

/// Zero-width joiner, the one character that binds two emoji into one cluster.
const ZWJ: char = '\u{200d}';

/// Discord's `IS_VOICE_MESSAGE` message flag.
pub const VOICE_MESSAGE_FLAG: u64 = 1 << 13;

/// Sent on every request. Discord's CDN returns 403 for default client
/// User-Agents (`Python-urllib`, and absent ones), and reqwest sends **no**
/// User-Agent unless one is set explicitly — see the test at the bottom of
/// this file, which asserts both halves of that claim against a local socket.
pub const USER_AGENT: &str =
    "Mozilla/5.0 (compatible; push/0.10.0; +https://github.com/owainlewis/push)";

/// A page of `GET /channels/{id}/messages`. The API caps this at 100.
const POLL_LIMIT: usize = 100;

const HTTP_TIMEOUT: Duration = Duration::from_secs(30);
/// Bounded 429 retries inside one request. push already treats a returned
/// error as "log and retry next interval" for polls and as a retryable
/// delivery failure for sends, so exhausting these is never fatal.
const MAX_RATE_LIMIT_RETRIES: usize = 5;
/// Fallback back-off when a 429 carries no usable `retry-after`.
const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(1);

/// ~5 requests/second across every Discord channel loop. These are the
/// integration's per-user headroom (PLAN §2, FINDINGS 5 and 6) and they are
/// shared with outbound sends, so the budget belongs to the process, not to a
/// single poll loop.
pub const DEFAULT_MIN_REQUEST_INTERVAL: Duration = Duration::from_millis(200);

// ---------------------------------------------------------------------------
// Shared request limiter
// ---------------------------------------------------------------------------

static GLOBAL_LIMITER: OnceLock<RequestLimiter> = OnceLock::new();

/// Process-wide request pacing plus a per-route 429 back-off.
///
/// push runs one poll loop per channel (ARCHITECTURE §9), so a limiter owned
/// by a single loop cannot bound the aggregate request rate — every
/// `DiscordChannel` therefore takes the same process-wide instance from
/// [`RequestLimiter::shared`]. The other channels have no equivalent: Telegram
/// uses long polling and Slack sleeps `retry-after` inline inside its own HTTP
/// helper.
///
/// Two limits are modelled separately because they are separate on the wire:
///
/// * one global minimum interval between *any* two Discord requests, which is
///   the integration's per-user budget;
/// * one back-off deadline per route, because a 429 applies to a route bucket
///   (Discord's own per-channel message bucket, enforced locally by exe.dev)
///   and must not stall unrelated routes.
#[derive(Clone)]
pub struct RequestLimiter {
    inner: Arc<LimiterState>,
}

struct LimiterState {
    min_interval: Duration,
    next_slot: Mutex<Instant>,
    backoff: Mutex<HashMap<String, Instant>>,
}

impl RequestLimiter {
    pub fn new(min_interval: Duration) -> Self {
        Self {
            inner: Arc::new(LimiterState {
                min_interval,
                next_slot: Mutex::new(Instant::now()),
                backoff: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The one limiter every Discord channel loop shares.
    pub fn shared() -> Self {
        GLOBAL_LIMITER
            .get_or_init(|| Self::new(DEFAULT_MIN_REQUEST_INTERVAL))
            .clone()
    }

    /// Waits until `route` may issue a request.
    pub async fn acquire(&self, route: &str) {
        loop {
            let now = Instant::now();
            let waiting = {
                let backoff = self.inner.backoff.lock().unwrap();
                backoff.get(route).filter(|until| **until > now).copied()
            };
            if let Some(until) = waiting {
                tokio::time::sleep(until - now).await;
                continue;
            }
            let slot = {
                let mut next_slot = self.inner.next_slot.lock().unwrap();
                let slot = (*next_slot).max(now);
                *next_slot = slot + self.inner.min_interval;
                slot
            };
            tokio::time::sleep_until(tokio::time::Instant::from_std(slot)).await;
            return;
        }
    }

    /// Records a server-directed back-off for one route. A later, shorter
    /// penalty never shortens an earlier, longer one.
    pub fn penalize(&self, route: &str, retry_after: Duration) {
        let until = Instant::now() + retry_after;
        let mut backoff = self.inner.backoff.lock().unwrap();
        backoff
            .entry(route.to_string())
            .and_modify(|existing| *existing = (*existing).max(until))
            .or_insert(until);
    }

    #[cfg(test)]
    fn backoff_until(&self, route: &str) -> Option<Instant> {
        self.inner.backoff.lock().unwrap().get(route).copied()
    }
}

// ---------------------------------------------------------------------------
// Error shapes
// ---------------------------------------------------------------------------

/// A failed response, decoded **without** assuming which error shape it uses.
///
/// exe.dev's proxy answers its own failures with `{"error": "...", "message":
/// "..."}` — `body_too_large` on a 413, `rate_limited` on a 429 — while
/// Discord's own errors use `{"message": "...", "code": N}`. Both are decoded
/// independently and neither is required to be present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    /// `error` from exe.dev's proxy shape, for example `body_too_large`.
    pub proxy_error: Option<String>,
    /// `message`, present in both shapes.
    pub message: Option<String>,
    /// `code` from Discord's own shape.
    pub discord_code: Option<u64>,
    /// The nested message Discord uses for field validation failures, for
    /// example `Must be 2000 or fewer in length.` under
    /// `errors.content._errors[0]` of an oversize 50035 rejection.
    pub validation_message: Option<String>,
}

const MAX_ERROR_TEXT: usize = 300;

impl ApiError {
    pub fn decode(status: u16, body: &[u8]) -> Self {
        let value: Option<Value> = serde_json::from_slice(body).ok();
        let text = |key: &str| {
            value
                .as_ref()
                .and_then(|value| value.get(key))
                .and_then(Value::as_str)
                .map(|value| value.chars().take(MAX_ERROR_TEXT).collect::<String>())
        };
        Self {
            status,
            proxy_error: text("error"),
            message: text("message"),
            discord_code: value
                .as_ref()
                .and_then(|value| value.get("code"))
                .and_then(Value::as_u64),
            validation_message: value.as_ref().and_then(validation_message),
        }
    }

    pub fn is_rate_limited(&self) -> bool {
        self.status == 429 || self.proxy_error.as_deref() == Some("rate_limited")
    }

    pub fn describe(&self) -> String {
        let mut parts = vec![format!("HTTP {}", self.status)];
        if let Some(error) = &self.proxy_error {
            parts.push(format!("proxy error {error:?}"));
        }
        if let Some(code) = self.discord_code {
            parts.push(format!("Discord code {code}"));
        }
        if let Some(message) = &self.message {
            parts.push(message.clone());
        }
        if let Some(message) = &self.validation_message {
            parts.push(message.clone());
        }
        parts.join(": ")
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.describe())
    }
}

/// Pulls the first nested field-validation message out of Discord's
/// `{"errors": {"<field>": {"_errors": [{"message": ...}]}}}` shape, which is
/// what an oversize message returns.
fn validation_message(value: &Value) -> Option<String> {
    let errors = value.get("errors")?.as_object()?;
    for details in errors.values() {
        for entry in details.get("_errors")?.as_array()? {
            if let Some(message) = entry.get("message").and_then(Value::as_str) {
                return Some(message.chars().take(MAX_ERROR_TEXT).collect());
            }
        }
    }
    None
}

/// Reads `retry-after` as an integer second count, which is what exe.dev
/// sends. A fractional value (legal HTTP, not observed here) is rounded up,
/// and anything unusable falls back to one second.
pub fn retry_after_duration(header: Option<&str>) -> Duration {
    let Some(value) = header.map(str::trim).filter(|value| !value.is_empty()) else {
        return DEFAULT_RETRY_AFTER;
    };
    if let Ok(seconds) = value.parse::<u64>() {
        return Duration::from_secs(seconds);
    }
    match value.parse::<f64>() {
        Ok(seconds) if seconds.is_finite() && seconds > 0.0 => Duration::from_secs_f64(seconds),
        _ => DEFAULT_RETRY_AFTER,
    }
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct TransportResponse {
    pub status: u16,
    /// Raw `retry-after` header value, kept undecoded so the shape stays
    /// testable without a live server.
    pub retry_after: Option<String>,
    pub body: Vec<u8>,
}

pub(crate) type TransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<TransportResponse>> + Send + 'a>>;

/// The four wire operations this channel needs. A trait, like Telegram's
/// `Transport`, so contract tests exercise request construction against a
/// local fake instead of the live integration.
pub(crate) trait Transport: Send + Sync {
    fn get<'a>(&'a self, url: &'a str) -> TransportFuture<'a>;

    fn post_json<'a>(&'a self, url: &'a str, body: &'a Value) -> TransportFuture<'a>;

    fn put_bytes<'a>(
        &'a self,
        url: &'a str,
        content_type: &'a str,
        bytes: &'a [u8],
    ) -> TransportFuture<'a>;

    /// Fetches a pre-signed or CDN URL. The bytes are not JSON, so this
    /// returns the response body verbatim.
    fn get_bytes<'a>(&'a self, url: &'a str) -> TransportFuture<'a>;
}

struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    fn new() -> Result<Self> {
        // `reqwest` sends no User-Agent at all unless one is configured
        // (`ClientBuilder::user_agent` defaults to unset), and Discord's CDN
        // 403s clients that look like scripts. Setting it on the client covers
        // every request, including the PUT to Google Cloud Storage.
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(HTTP_TIMEOUT)
            .build()
            .context("build Discord HTTP client")?;
        Ok(Self { client })
    }
}

async fn decode_response(response: reqwest::Response) -> Result<TransportResponse> {
    let status = response.status().as_u16();
    let retry_after = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = response
        .bytes()
        .await
        .context("read Discord response body")?
        .to_vec();
    Ok(TransportResponse {
        status,
        retry_after,
        body,
    })
}

impl Transport for ReqwestTransport {
    fn get<'a>(&'a self, url: &'a str) -> TransportFuture<'a> {
        Box::pin(async move {
            let response = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|error| anyhow::anyhow!("Discord request to {url} failed: {error}"))?;
            decode_response(response).await
        })
    }

    fn post_json<'a>(&'a self, url: &'a str, body: &'a Value) -> TransportFuture<'a> {
        Box::pin(async move {
            let response = self
                .client
                .post(url)
                .json(body)
                .send()
                .await
                .map_err(|error| anyhow::anyhow!("Discord request to {url} failed: {error}"))?;
            decode_response(response).await
        })
    }

    #[allow(dead_code)]
    #[allow(dead_code)]
    fn put_bytes<'a>(
        &'a self,
        url: &'a str,
        content_type: &'a str,
        bytes: &'a [u8],
    ) -> TransportFuture<'a> {
        Box::pin(async move {
            let response = self
                .client
                .put(url)
                .header(reqwest::header::CONTENT_TYPE, content_type)
                .body(bytes.to_vec())
                .send()
                .await
                .map_err(|error| anyhow::anyhow!("Discord upload to {url} failed: {error}"))?;
            decode_response(response).await
        })
    }

    fn get_bytes<'a>(&'a self, url: &'a str) -> TransportFuture<'a> {
        Box::pin(async move {
            let response =
                self.client.get(url).send().await.map_err(|error| {
                    anyhow::anyhow!("Discord download from {url} failed: {error}")
                })?;
            decode_response(response).await
        })
    }
}

// ---------------------------------------------------------------------------
// HTTP helper: pacing, 429 back-off, error classification
// ---------------------------------------------------------------------------

/// One request, described without borrowing the channel.
enum Request<'a> {
    Get(&'a str),
    PostJson(&'a str, &'a Value),
    #[allow(dead_code)]
    PutBytes(&'a str, &'a str, &'a [u8]),
    GetBytes(&'a str),
}

#[derive(Clone)]
struct HttpClient {
    transport: Arc<dyn Transport>,
    limiter: RequestLimiter,
}

impl HttpClient {
    async fn send(&self, route: &str, request: Request<'_>) -> Result<TransportResponse> {
        let mut attempt = 0;
        loop {
            // Pacing first: the limiter is the only thing standing between a
            // burst of polls and a 429.
            self.limiter.acquire(route).await;
            let response = match &request {
                Request::Get(url) => self.transport.get(url).await?,
                Request::PostJson(url, body) => self.transport.post_json(url, body).await?,
                Request::PutBytes(url, content_type, bytes) => {
                    self.transport.put_bytes(url, content_type, bytes).await?
                }
                Request::GetBytes(url) => self.transport.get_bytes(url).await?,
            };
            if !(200..300).contains(&response.status) {
                let error = ApiError::decode(response.status, &response.body);
                if error.is_rate_limited() && attempt < MAX_RATE_LIMIT_RETRIES {
                    let retry_after = retry_after_duration(response.retry_after.as_deref());
                    // A 429 is routine on this API and is a per-route back-off,
                    // never a fatal error: record it, wait, and try again.
                    self.limiter.penalize(route, retry_after);
                    attempt += 1;
                    continue;
                }
            }
            return Ok(response);
        }
    }

    async fn send_value(&self, route: &str, request: Request<'_>) -> Result<Value> {
        let response = self.send(route, request).await?;
        if !(200..300).contains(&response.status) {
            bail!(
                "Discord request failed: {}",
                ApiError::decode(response.status, &response.body)
            );
        }
        if response.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&response.body).with_context(|| {
            format!(
                "Discord returned HTTP {} with invalid JSON",
                response.status
            )
        })
    }

    async fn get_value(&self, url: &str, route: &str) -> Result<Value> {
        self.send_value(route, Request::Get(url)).await
    }

    async fn post_value(&self, url: &str, route: &str, body: &Value) -> Result<Value> {
        self.send_value(route, Request::PostJson(url, body)).await
    }

    async fn put_bytes(
        &self,
        url: &str,
        route: &str,
        content_type: &str,
        bytes: &[u8],
    ) -> Result<()> {
        let response = self
            .send(route, Request::PutBytes(url, content_type, bytes))
            .await?;
        if !(200..300).contains(&response.status) {
            bail!(
                "Discord upload failed: {}",
                ApiError::decode(response.status, &response.body)
            );
        }
        Ok(())
    }

    async fn get_bytes(
        &self,
        url: &str,
        route: &str,
        max_bytes: usize,
        kind: &'static str,
    ) -> Result<Vec<u8>> {
        let response = self.send(route, Request::GetBytes(url)).await?;
        if !(200..300).contains(&response.status) {
            bail!(
                "{kind} download failed: {}",
                ApiError::decode(response.status, &response.body)
            );
        }
        if response.body.len() > max_bytes {
            bail!("{kind} exceeds the configured size limit");
        }
        Ok(response.body)
    }
}

// ---------------------------------------------------------------------------
// Message payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DiscordMessage {
    pub id: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub author: Option<DiscordAuthor>,
    #[serde(default)]
    pub attachments: Vec<DiscordAttachment>,
    #[serde(default)]
    pub flags: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DiscordAuthor {
    pub id: String,
    #[serde(default)]
    pub bot: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DiscordAttachment {
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default)]
    pub size: Option<usize>,
    #[serde(default)]
    pub url: String,
}

/// The two-step upload seam below is **not called by Phase 3 delivery** — that
/// is text-only — but it is the verified wire format Phase 5 and Phase 6 build
/// on, so it ships with its own tests rather than a comment.
///
/// `POST /channels/{id}/attachments` identifies the prepared upload with an
/// integer, but older responses and the spike's mock used a string; both are
/// accepted (FINDINGS, "Discord attachment-response shape").
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UploadId {
    Integer(u64),
    Text(String),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PreparedUpload {
    pub id: UploadId,
    pub upload_url: String,
    pub upload_filename: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AttachmentUploadResponse {
    #[serde(default)]
    attachments: Vec<PreparedUpload>,
}

/// Voice-message metadata. Phase 6 computes both values from the OGG/Opus
/// bytes; the upload helper takes them as input so the wire format is already
/// settled.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VoiceMetadata {
    pub duration_secs: f64,
    /// base64 of Discord's 256-sample waveform.
    pub waveform: String,
}

#[allow(dead_code)]
/// One attachment to upload, in the order the two-step flow needs.
pub(crate) struct AttachmentUpload<'a> {
    pub filename: &'a str,
    pub bytes: &'a [u8],
    pub content_type: &'a str,
    /// `Some` turns the message into a native voice message: `flags: 8192`
    /// plus `duration_secs` and `waveform`, and an `audio/ogg` PUT.
    pub voice: Option<VoiceMetadata>,
}

/// The channel-visible halves of an inbound Discord message.
pub(crate) fn discord_message_to_raw(
    channel_id: &str,
    bot_user_id: Option<&str>,
    message: DiscordMessage,
) -> Result<RawMessage> {
    let row_id = message
        .id
        .parse::<i64>()
        .with_context(|| format!("Discord message id {:?} is not a snowflake", message.id))?;
    let author_id = message
        .author
        .as_ref()
        .map(|author| author.id.clone())
        .unwrap_or_default();
    // The bot's own messages are marked `is_from_me`, exactly as Slack does,
    // so the ignore is decided by `accept` and shows up in the audit log
    // rather than being dropped silently here. Any bot author counts as self:
    // the alternative is two bots answering each other forever.
    let is_from_me = message.author.as_ref().is_some_and(|author| author.bot)
        || bot_user_id.is_some_and(|bot| bot == author_id);
    let mut voice = None;
    let mut images = Vec::new();
    for attachment in &message.attachments {
        let content_type = attachment.content_type.as_deref().unwrap_or_default();
        // A voice note is an audio attachment, or a flagged voice message whose
        // attachment carries no content type (Discord omits it sometimes).
        // FINDINGS: the MESSAGE_CONTENT intent must be on, or `attachments` is
        // empty here and inbound voice notes are invisible.
        let is_audio = content_type.starts_with("audio/");
        let is_flagged_voice = message.flags & VOICE_MESSAGE_FLAG != 0;
        if voice.is_none() && (is_audio || (is_flagged_voice && content_type.is_empty())) {
            voice = Some(InboundVoice {
                locator: attachment.url.clone(),
                file_size: attachment.size,
                mime_type: if content_type.is_empty() {
                    "audio/ogg".to_string()
                } else {
                    content_type.to_string()
                },
                filename: if attachment.filename.is_empty() {
                    "voice.ogg".to_string()
                } else {
                    attachment.filename.clone()
                },
                data: None,
            });
        } else if content_type.starts_with("image/") {
            images.push(InboundImage {
                locator: attachment.url.clone(),
                file_size: attachment.size,
                mime_type: Some(content_type.to_string()),
                data: None,
            });
        }
    }
    Ok(RawMessage {
        row_id,
        provider_event_id: Some(message.id),
        channel: "discord",
        handle: author_id,
        chat_identifier: channel_id.to_string(),
        is_group: false,
        text: message.content,
        voice,
        images,
        is_from_me,
        is_supported: true,
        thread_id: None,
    })
}

// ---------------------------------------------------------------------------
// Trigger rule and outbound splitting
// ---------------------------------------------------------------------------

/// The Phase 3 trigger rule: a mention or a `!` prefix, plus the voice-note
/// exception.
///
/// A voice note carries no `content` at all — it can never contain a mention
/// or a `!` — so an inbound voice-note attachment is treated as
/// self-triggering (PLAN §3, `Trigger`). Mention detection reads the literal
/// `<@id>` / `<@!id>` markup Discord puts in `content`, which is also why the
/// `MESSAGE_CONTENT` intent matters (FINDINGS, "MESSAGE_CONTENT").
pub(crate) fn is_triggered(message: &RawMessage, bot_user_id: Option<&str>) -> bool {
    if message.voice.is_some() {
        return true;
    }
    if message.text.trim_start().starts_with('!') {
        return true;
    }
    bot_user_id.is_some_and(|bot| mentions_user(&message.text, bot))
}

pub(crate) fn mentions_user(text: &str, user_id: &str) -> bool {
    text.contains(&format!("<@{user_id}>")) || text.contains(&format!("<@!{user_id}>"))
}

/// Splits one agent answer into Discord message payloads.
///
/// The signature is deliberately narrow — `text -> chunks` — so every rule
/// lives in one place. The behaviour is the measured contract from
/// `spikes/discord-limits/FINDINGS.md`:
///
/// * at most [`TEXT_LIMIT`] Unicode code points per chunk, counted with
///   `chars().count()`, and never relying on the server's trailing-whitespace
///   strip to make a chunk fit (§2, §4);
/// * a chunk that would end inside a ``` / ~~~ fence is closed with `\n````
///   and the next chunk reopens it with the **same language tag**, so no chunk
///   ever carries an odd number of fence markers (§6.3);
/// * boundaries are preferred at a paragraph, then a line, then a word, and
///   only then a hard cut at a code-point boundary (§6.4);
/// * links, inline code spans, table rows, list items, and quoted lines are
///   never cut through — they move whole to the next chunk, and are cut only
///   when one of them cannot fit in a chunk of its own (§5d, §6.5);
/// * whitespace-only chunks are dropped, because the API answers them with
///   `50006 "Cannot send an empty message"` (§4, §6.7);
/// * Unicode is never normalised, matching Discord, which stores the scalar
///   sequence it is given (§3).
pub fn discord_chunks(text: &str) -> Vec<String> {
    let mut packer = Packer::new();
    for atom in atomize(text) {
        packer.place(&atom);
    }
    packer.finish()
}

/// The smallest thing the packer will move as a unit.
#[derive(Debug, Clone)]
struct Atom {
    /// Text to append, carrying its own separators (trailing space or newline).
    text: String,
    /// Preferred line-level break points, used only when the atom does not fit
    /// in a chunk of its own, so one oversized table cannot break the channel.
    parts: Vec<String>,
    /// The opening fence marker of the block this atom belongs to, when it is
    /// fenced content. Its presence selects the code budget.
    fence: Option<String>,
    /// True for the closing fence marker of an open block.
    closes_fence: bool,
}

impl Atom {
    fn plain(text: String) -> Self {
        Self {
            text,
            parts: Vec::new(),
            fence: None,
            closes_fence: false,
        }
    }
}

fn is_fence_line(trimmed: &str) -> bool {
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

fn is_list_item(trimmed: &str) -> bool {
    if ["- ", "* ", "+ "]
        .iter()
        .any(|marker| trimmed.starts_with(marker))
    {
        return true;
    }
    let digits = trimmed.chars().take_while(char::is_ascii_digit).count();
    digits > 0 && trimmed[digits..].starts_with(". ")
}

/// A group of lines that must stay together when they fit: one table, one
/// quoted block, or one list item with its continuation lines. The parts are
/// the fallback break points if the group is larger than a whole chunk.
fn group_atom(rows: &[&str]) -> Atom {
    let mut text = String::new();
    let mut parts = Vec::with_capacity(rows.len());
    for row in rows {
        text.push_str(row);
        text.push('\n');
        parts.push(format!("{row}\n"));
    }
    Atom {
        text,
        parts,
        fence: None,
        closes_fence: false,
    }
}

/// One prose line, split at word boundaries but never inside a link or an
/// inline code span. The newline stays attached to the line's last token so the
/// atoms concatenate back to the source.
fn prose_line_atoms(line: &str) -> Vec<Atom> {
    let tokens = prose_tokens(line);
    let last = tokens.len().saturating_sub(1);
    tokens
        .into_iter()
        .enumerate()
        .map(|(index, mut text)| {
            if index == last {
                text.push('\n');
            }
            Atom::plain(text)
        })
        .collect()
}

/// Splits a line at whitespace, attaching each whitespace run to the token
/// before it, while treating `[text](url)` and `` `code span` `` as single
/// tokens so a chunk boundary can never land inside them.
fn prose_tokens(line: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut index = 0;
    while index < chars.len() {
        let (byte_index, character) = chars[index];
        if character == '[' {
            if let Some(end) = link_end(line, byte_index) {
                current.push_str(&line[byte_index..end]);
                index = chars.partition_point(|(offset, _)| *offset < end);
                continue;
            }
        }
        if character == '`' {
            if let Some(end) = code_span_end(line, byte_index) {
                current.push_str(&line[byte_index..end]);
                index = chars.partition_point(|(offset, _)| *offset < end);
                continue;
            }
        }
        if character.is_whitespace() {
            let mut next = index;
            while next < chars.len() && chars[next].1.is_whitespace() {
                next += 1;
            }
            let end = chars.get(next).map_or(line.len(), |(offset, _)| *offset);
            current.push_str(&line[byte_index..end]);
            tokens.push(std::mem::take(&mut current));
            index = next;
            continue;
        }
        current.push(character);
        index += 1;
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Byte offset just past `[label](destination)`, or `None` when this `[` does
/// not open a Markdown link.
fn link_end(line: &str, start: usize) -> Option<usize> {
    let rest = line.get(start..)?;
    let label = rest.find(']')?;
    if rest.as_bytes().get(label + 1) != Some(&b'(') {
        return None;
    }
    let destination = rest[label + 1..].find(')')?;
    Some(start + label + 1 + destination + 1)
}

/// Byte offset just past a run of backticks closed by an equal-length run, or
/// `None` when this backtick does not open an inline code span.
fn code_span_end(line: &str, start: usize) -> Option<usize> {
    let rest = line.get(start..)?;
    let opened = rest
        .chars()
        .take_while(|character| *character == '`')
        .count();
    let closing = "`".repeat(opened);
    let body_start = start + opened;
    let body = line.get(body_start..)?;
    let end = body.find(&closing)?;
    Some(body_start + end + opened)
}

/// Breaks the source into atoms, tracking fence state so code lines keep the
/// marker that must be reopened after a split.
fn atomize(text: &str) -> Vec<Atom> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut atoms = Vec::new();
    let mut fence: Option<String> = None;
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim_start();
        if let Some(marker) = fence.clone() {
            if is_fence_line(trimmed) {
                atoms.push(Atom {
                    text: format!("{}\n", trimmed.trim_end()),
                    parts: Vec::new(),
                    fence: Some(marker),
                    closes_fence: true,
                });
                fence = None;
            } else {
                atoms.push(Atom {
                    text: format!("{line}\n"),
                    parts: Vec::new(),
                    fence: Some(marker),
                    closes_fence: false,
                });
            }
            index += 1;
            continue;
        }
        if is_fence_line(trimmed) {
            let marker = trimmed.trim_end().to_string();
            atoms.push(Atom {
                text: format!("{marker}\n"),
                parts: Vec::new(),
                fence: Some(marker.clone()),
                closes_fence: false,
            });
            fence = Some(marker);
            index += 1;
            continue;
        }
        if trimmed.is_empty() {
            atoms.push(Atom::plain("\n".to_string()));
            index += 1;
            continue;
        }
        if trimmed.starts_with('|') {
            let mut rows = Vec::new();
            while index < lines.len() && lines[index].trim_start().starts_with('|') {
                rows.push(lines[index]);
                index += 1;
            }
            atoms.push(group_atom(&rows));
            continue;
        }
        if trimmed.starts_with('>') {
            let mut rows = Vec::new();
            while index < lines.len() && lines[index].trim_start().starts_with('>') {
                rows.push(lines[index]);
                index += 1;
            }
            atoms.push(group_atom(&rows));
            continue;
        }
        if is_list_item(trimmed) {
            let mut rows = vec![line];
            index += 1;
            while index < lines.len()
                && !lines[index].trim().is_empty()
                && lines[index].starts_with([' ', '\t'])
                && !is_list_item(lines[index].trim_start())
            {
                rows.push(lines[index]);
                index += 1;
            }
            atoms.push(group_atom(&rows));
            continue;
        }
        atoms.extend(prose_line_atoms(line));
        index += 1;
    }
    atoms
}

struct Packer {
    chunks: Vec<String>,
    current: String,
    /// The fence the current chunk is inside, if any.
    current_fence: Option<String>,
    /// Marker to reopen with when the next fenced atom is placed.
    pending_fence: Option<String>,
}

impl Packer {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            current: String::new(),
            current_fence: None,
            pending_fence: None,
        }
    }

    fn code_budget() -> usize {
        TEXT_LIMIT - FENCE_RESERVE
    }

    fn place(&mut self, atom: &Atom) {
        let fenced =
            atom.fence.is_some() || self.current_fence.is_some() || self.pending_fence.is_some();
        let budget = if fenced {
            Self::code_budget()
        } else {
            TEXT_BUDGET
        };
        if self.fits(atom, budget) {
            self.append(atom);
            return;
        }
        // A chunk ending inside a fence must take its closing "\n```". The code
        // budget already reserved exactly those four code points, so the marker
        // is appended to an otherwise full chunk instead of opening a new,
        // empty fenced block.
        if atom.closes_fence
            && self.current.chars().count() + atom.text.chars().count() <= TEXT_LIMIT
        {
            self.append(atom);
            return;
        }
        // Break whenever anything is buffered. A whitespace-only buffer is
        // dropped by `break_chunk` (it would be rejected as an empty message),
        // and clearing it is what guarantees the retry below sees an empty
        // chunk and makes progress.
        if !self.current.is_empty() {
            self.break_chunk();
            self.place(atom);
            return;
        }
        // The chunk is empty and the atom still does not fit: fall back to its
        // own line-level break points, then to a hard code-point cut.
        if !atom.parts.is_empty() {
            for part in atom.parts.clone() {
                self.place(&Atom {
                    text: part,
                    parts: Vec::new(),
                    fence: atom.fence.clone(),
                    closes_fence: false,
                });
            }
            return;
        }
        let pending = self
            .pending_fence
            .as_ref()
            .map_or(0, |marker| marker.chars().count() + 1);
        let piece_budget = budget.saturating_sub(pending).max(1);
        for piece in hard_pieces(&atom.text, piece_budget) {
            self.place(&Atom {
                text: piece,
                parts: Vec::new(),
                fence: atom.fence.clone(),
                closes_fence: false,
            });
        }
    }

    /// Conservative on purpose: `current`'s trailing whitespace is counted even
    /// though the server would strip it, so a chunk never depends on the
    /// trailing-whitespace strip to fit (FINDINGS §6.1).
    fn fits(&self, atom: &Atom, budget: usize) -> bool {
        let mut length = self.current.chars().count() + atom.text.chars().count();
        if atom.fence.is_some() {
            if let Some(marker) = &self.pending_fence {
                length += marker.chars().count() + 1;
            }
        }
        length <= budget
    }

    fn append(&mut self, atom: &Atom) {
        if atom.fence.is_some() {
            if let Some(marker) = self.pending_fence.take() {
                self.current.push_str(&marker);
                self.current.push('\n');
                self.current_fence = Some(marker);
            }
        }
        self.current.push_str(&atom.text);
        if atom.closes_fence {
            self.current_fence = None;
        } else if let Some(marker) = &atom.fence {
            self.current_fence = Some(marker.clone());
        }
    }

    fn break_chunk(&mut self) {
        let mut finished = self.current.trim_end().to_string();
        if let Some(marker) = self.current_fence.take() {
            finished.push_str("\n```");
            self.pending_fence = Some(marker);
        }
        if !finished.trim().is_empty() {
            self.chunks.push(finished);
        }
        self.current.clear();
    }

    fn finish(mut self) -> Vec<String> {
        let mut finished = self.current.trim_end().to_string();
        if self.current_fence.is_some() {
            // A source fence that is never closed is closed here: a chunk with
            // an odd number of markers renders as broken code, not as an error.
            finished.push_str("\n```");
        }
        if !finished.trim().is_empty() {
            self.chunks.push(finished);
        }
        self.chunks
    }
}

/// Cuts one oversized atom at code-point boundaries, nudged so a cut never
/// lands inside a grapheme cluster.
fn hard_pieces(text: &str, budget: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let budget = budget.max(1);
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = (start + budget).min(chars.len());
        if end < chars.len() {
            end = avoid_grapheme_cut(&chars, start, end);
        }
        pieces.push(chars[start..end].iter().collect());
        start = end;
    }
    pieces
}

/// A rendering nicety rather than a limit rule (FINDINGS §6.8): a cut never
/// follows a zero-width joiner and never separates a base character from a
/// combining mark. Without Unicode segmentation tables this is a heuristic, but
/// it only ever moves a cut earlier, so it cannot make a chunk too long.
fn avoid_grapheme_cut(chars: &[char], start: usize, cut: usize) -> usize {
    let mut cut = cut;
    while cut > start + 1 && cut < chars.len() {
        if chars[cut - 1] == ZWJ || is_combining_or_variation(chars[cut]) {
            cut -= 1;
        } else {
            break;
        }
    }
    cut
}

fn is_combining_or_variation(character: char) -> bool {
    matches!(character,
        '\u{0300}'..='\u{036f}'
        | '\u{1ab0}'..='\u{1aff}'
        | '\u{1dc0}'..='\u{1dff}'
        | '\u{20d0}'..='\u{20ff}'
        | '\u{fe00}'..='\u{fe0f}'
        | '\u{fe20}'..='\u{fe2f}'
        | '\u{200d}')
}

/// The attachment URLs Discord hands out are CDN URLs. Refusing anything else
/// keeps a message payload from turning into a request to an arbitrary host.
fn validated_attachment_url(url: &str) -> Result<&str> {
    let parsed = reqwest::Url::parse(url).with_context(|| "Discord attachment URL is invalid")?;
    let allowed = parsed.scheme() == "https"
        && parsed
            .host_str()
            .is_some_and(|host| matches!(host, "cdn.discordapp.com" | "media.discordapp.net"));
    if !allowed {
        bail!("Discord attachment URL is outside the trusted CDN origin");
    }
    Ok(url)
}

// ---------------------------------------------------------------------------
// The channel
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DiscordChannel {
    base: Arc<str>,
    channel_id: Arc<str>,
    allow_channel_ids: Arc<HashSet<String>>,
    /// Resolved once from `GET /users/@me`. Needed to tell "mentioned me" from
    /// "mentioned somebody else"; the bot token is injected by the proxy, so
    /// there is no local source of the bot's user id.
    bot_user_id: Arc<Mutex<Option<String>>>,
    /// Durable cursor key: `discord:<channel-id>` (PLAN §3, one push channel
    /// per guild channel). The `ChannelContract::store_key` hook carries it.
    store_key: Arc<str>,
    messages_route: Arc<str>,
    send_route: Arc<str>,
    #[allow(dead_code)]
    upload_route: Arc<str>,
    client: HttpClient,
}

impl DiscordChannel {
    pub fn new(base_url: &str, channel_id: String, allow_channel_ids: Vec<String>) -> Result<Self> {
        let transport = Arc::new(ReqwestTransport::new()?);
        Ok(Self::with_transport(
            base_url,
            channel_id,
            allow_channel_ids,
            transport,
            RequestLimiter::shared(),
        ))
    }

    fn with_transport(
        base_url: &str,
        channel_id: String,
        allow_channel_ids: Vec<String>,
        transport: Arc<dyn Transport>,
        limiter: RequestLimiter,
    ) -> Self {
        let base = base_url.trim_end_matches('/').to_string();
        Self {
            base: Arc::from(base),
            allow_channel_ids: Arc::new(allow_channel_ids.into_iter().collect()),
            messages_route: Arc::from(format!("messages:{channel_id}")),
            send_route: Arc::from(format!("send:{channel_id}")),
            upload_route: Arc::from(format!("upload:{channel_id}")),
            store_key: Arc::from(format!("discord:{channel_id}")),
            channel_id: Arc::from(channel_id),
            bot_user_id: Arc::new(Mutex::new(None)),
            client: HttpClient { transport, limiter },
        }
    }

    pub fn channel_id(&self) -> &str {
        &self.channel_id
    }

    pub fn store_key(&self) -> &str {
        &self.store_key
    }

    pub fn bot_user_id(&self) -> Option<String> {
        self.bot_user_id.lock().unwrap().clone()
    }

    pub fn is_allowed(&self, message: &RawMessage) -> bool {
        self.allow_channel_ids.contains(&message.chat_identifier)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{API_PREFIX}{path}", self.base)
    }

    /// Resolves and caches the bot's own user id. Failing here fails the poll
    /// rather than guessing: without it a mention cannot be told apart from
    /// mentioning somebody else, and push logs and retries a failed poll.
    async fn ensure_bot_user_id(&self) -> Result<String> {
        if let Some(id) = self.bot_user_id() {
            return Ok(id);
        }
        let value = self
            .client
            .get_value(&self.url("/users/@me"), "users:@me")
            .await?;
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .context("Discord /users/@me did not return a user id")?
            .to_string();
        *self.bot_user_id.lock().unwrap() = Some(id.clone());
        Ok(id)
    }

    pub async fn poll(&self, since: i64) -> Result<Vec<RawMessage>> {
        let bot_user_id = self.ensure_bot_user_id().await?;
        let url = self.url(&format!(
            "/channels/{}/messages?after={since}&limit={POLL_LIMIT}",
            self.channel_id
        ));
        let value = self.client.get_value(&url, &self.messages_route).await?;
        let mut messages: Vec<DiscordMessage> = serde_json::from_value(value)
            .context("Discord messages endpoint returned an unexpected payload")?;
        // The endpoint is newest-first and `after` is exclusive; push processes
        // oldest-first. Sorting by snowflake is exact, unlike string order
        // across lengths.
        messages.sort_by_key(|message| message.id.parse::<i64>().unwrap_or_default());
        messages
            .into_iter()
            .map(|message| discord_message_to_raw(&self.channel_id, Some(&bot_user_id), message))
            .collect()
    }

    pub async fn latest_cursor(&self) -> Result<i64> {
        let url = self.url(&format!("/channels/{}/messages?limit=1", self.channel_id));
        let value = self.client.get_value(&url, &self.messages_route).await?;
        let messages: Vec<DiscordMessage> = serde_json::from_value(value)
            .context("Discord messages endpoint returned an unexpected payload")?;
        match messages.first() {
            Some(message) => message
                .id
                .parse::<i64>()
                .with_context(|| format!("Discord message id {:?} is not a snowflake", message.id)),
            None => Ok(0),
        }
    }

    pub async fn send_text(&self, target: &str, text: &str) -> Result<()> {
        // The hard limit is 2000 code points; `outbound_chunks` already keeps
        // every chunk at or below `TEXT_BUDGET`, so this only catches a caller
        // that built its own chunk.
        if text.chars().count() > TEXT_LIMIT {
            bail!("Discord message exceeds the {TEXT_LIMIT} code point limit");
        }
        let url = self.url(&format!("/channels/{target}/messages"));
        self.client
            .post_value(&url, &self.send_route, &json!({ "content": text }))
            .await?;
        Ok(())
    }

    /// The two-step upload: prepare on the integration, `PUT` the bytes
    /// straight at the pre-signed Google Cloud Storage URL (which is what
    /// bypasses the 256 KiB request cap), then post the message referencing
    /// the prepared upload.
    ///
    /// The prepare response does **not** echo `filename`, so it is carried from
    /// the request; `upload_filename` carries a UUID prefix and is sent back as
    /// `uploaded_filename`. One attachment per call — multiple attachments need
    /// distinct `id` indices in the `files` array.
    #[allow(dead_code)]
    pub async fn upload_attachment(
        &self,
        attachment: AttachmentUpload<'_>,
        content: &str,
    ) -> Result<String> {
        let prepare_url = self.url(&format!("/channels/{}/attachments", self.channel_id));
        let prepared = self
            .client
            .post_value(
                &prepare_url,
                &self.upload_route,
                &json!({
                    "files": [{
                        "filename": attachment.filename,
                        "file_size": attachment.bytes.len(),
                        "id": 0
                    }]
                }),
            )
            .await?;
        let response: AttachmentUploadResponse = serde_json::from_value(prepared)
            .context("Discord attachments endpoint returned an unexpected payload")?;
        let upload = response
            .attachments
            .into_iter()
            .next()
            .context("Discord attachments endpoint returned no prepared upload")?;

        self.client
            .put_bytes(
                &upload.upload_url,
                "upload",
                attachment.content_type,
                attachment.bytes,
            )
            .await?;

        let mut body = serde_json::Map::new();
        if !content.is_empty() {
            body.insert("content".to_string(), json!(content));
        }
        body.insert(
            "attachments".to_string(),
            json!([{
                "id": upload.id,
                "filename": attachment.filename,
                "uploaded_filename": upload.upload_filename
            }]),
        );
        if let Some(voice) = &attachment.voice {
            body.insert("flags".to_string(), json!(VOICE_MESSAGE_FLAG));
            body.insert("duration_secs".to_string(), json!(voice.duration_secs));
            body.insert("waveform".to_string(), json!(voice.waveform));
        }
        let message_url = self.url(&format!("/channels/{}/messages", self.channel_id));
        let posted = self
            .client
            .post_value(&message_url, &self.send_route, &Value::Object(body))
            .await?;
        Ok(posted
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string())
    }

    pub async fn download_voice(&self, voice: &InboundVoice) -> Result<AudioClip> {
        if let Some(bytes) = &voice.data {
            return Ok(AudioClip {
                bytes: bytes.clone(),
                filename: voice.filename.clone(),
                mime_type: voice.mime_type.clone(),
            });
        }
        if voice.file_size.is_some_and(|size| size > MAX_AUDIO_BYTES) {
            bail!("Discord voice message exceeds the 20 MB limit");
        }
        let url = validated_attachment_url(&voice.locator)?;
        let bytes = self
            .client
            .get_bytes(url, "cdn", MAX_AUDIO_BYTES, "voice message")
            .await?;
        Ok(AudioClip {
            bytes,
            filename: voice.filename.clone(),
            mime_type: voice.mime_type.clone(),
        })
    }

    pub async fn download_image(&self, image: &InboundImage) -> Result<DownloadedImage> {
        if let Some(bytes) = &image.data {
            return Ok(DownloadedImage {
                bytes: bytes.clone(),
            });
        }
        if image.file_size.is_some_and(|size| size > MAX_IMAGE_BYTES) {
            bail!("Discord image exceeds the 6 MiB limit");
        }
        if !matches!(
            image.mime_type.as_deref(),
            Some("image/jpeg" | "image/png" | "image/webp")
        ) {
            bail!("Discord image is not a supported JPEG, PNG, or WebP file");
        }
        let url = validated_attachment_url(&image.locator)?;
        let bytes = self
            .client
            .get_bytes(url, "cdn", MAX_IMAGE_BYTES, "image")
            .await?;
        Ok(DownloadedImage { bytes })
    }
}

/// A transport for the `ChannelContract` tests in `crate::channel`, which
/// exercise `accept`, the thread key, and chunking — all local decisions.
#[cfg(test)]
pub(crate) struct NullTransport;

#[cfg(test)]
impl Transport for NullTransport {
    fn get<'a>(&'a self, _url: &'a str) -> TransportFuture<'a> {
        Box::pin(async { bail!("NullTransport performs no requests") })
    }

    fn post_json<'a>(&'a self, _url: &'a str, _body: &'a Value) -> TransportFuture<'a> {
        Box::pin(async { bail!("NullTransport performs no requests") })
    }

    #[allow(dead_code)]
    fn put_bytes<'a>(
        &'a self,
        _url: &'a str,
        _content_type: &'a str,
        _bytes: &'a [u8],
    ) -> TransportFuture<'a> {
        Box::pin(async { bail!("NullTransport performs no requests") })
    }

    fn get_bytes<'a>(&'a self, _url: &'a str) -> TransportFuture<'a> {
        Box::pin(async { bail!("NullTransport performs no requests") })
    }
}

#[cfg(test)]
impl DiscordChannel {
    /// A channel with no network, a known bot identity, and pacing disabled.
    pub(crate) fn for_contract_tests(
        channel_id: &str,
        allow_channel_ids: &[&str],
        bot_user_id: Option<&str>,
    ) -> Self {
        let channel = Self::with_transport(
            DEFAULT_BASE_URL,
            channel_id.to_string(),
            allow_channel_ids.iter().map(|id| id.to_string()).collect(),
            Arc::new(NullTransport),
            RequestLimiter::new(Duration::ZERO),
        );
        if let Some(id) = bot_user_id {
            *channel.bot_user_id.lock().unwrap() = Some(id.to_string());
        }
        channel
    }
}

#[cfg(test)]
mod tests {

    /// Independent invariant check written by the reviewer, not the author: the
    /// measured contract must hold for adversarial input, not only the documented
    /// cases. Every chunk must fit 2000 code points, must never be whitespace-only
    /// (the API rejects those with 50006), and must carry an even number of fence
    /// markers so no chunk can render as an unterminated code block.
    #[test]
    fn every_chunk_respects_the_measured_contract() {
        let cases: Vec<(&str, String)> = vec![
            ("emoji", "\u{1f600}".repeat(5000)),
            ("combining", "e\u{301}".repeat(3000)),
            (
                "fenced",
                format!("intro\n\n```python\n{}\n```\n", "x = 1\n".repeat(600)),
            ),
            ("unbreakable", format!("see {} end", "a".repeat(2500))),
            (
                "table",
                (0..400)
                    .map(|i| format!("| row {i} | value {i} |\n"))
                    .collect(),
            ),
            ("whitespace_only", "\n\n   \n\t\n".to_string()),
            (
                "mixed",
                format!(
                    "{}\n```\n{}\n```\n{}",
                    "prose ".repeat(200),
                    "code line\n".repeat(300),
                    "\u{1f600}".repeat(1200)
                ),
            ),
        ];
        for (name, input) in cases {
            let chunks = discord_chunks(&input);
            for (index, chunk) in chunks.iter().enumerate() {
                let points = chunk.chars().count();
                assert!(
                    points <= TEXT_LIMIT,
                    "{name} chunk {index}: {points} code points exceeds {TEXT_LIMIT}"
                );
                assert!(
                    !chunk.trim().is_empty(),
                    "{name} chunk {index} is whitespace-only (API would answer 50006)"
                );
                let fences = chunk.matches("```").count();
                assert_eq!(
                    fences % 2,
                    0,
                    "{name} chunk {index} has {fences} fence markers (odd renders broken)"
                );
            }
        }
    }
    use super::*;
    use std::collections::VecDeque;

    // -- fake transport ----------------------------------------------------

    #[derive(Debug, Clone, PartialEq)]
    enum Call {
        Get(String),
        PostJson(String, Value),
        PutBytes(String, String, Vec<u8>),
        GetBytes(String),
    }

    #[derive(Default)]
    struct FakeTransport {
        calls: Mutex<Vec<Call>>,
        responses: Mutex<VecDeque<TransportResponse>>,
    }

    impl FakeTransport {
        fn with_json(responses: Vec<Value>) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                responses: Mutex::new(
                    responses
                        .into_iter()
                        .map(|body| TransportResponse {
                            status: 200,
                            retry_after: None,
                            body: body.to_string().into_bytes(),
                        })
                        .collect(),
                ),
            }
        }

        fn with_raw(responses: Vec<TransportResponse>) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                responses: Mutex::new(responses.into_iter().collect()),
            }
        }

        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }

        fn next(&self) -> Result<TransportResponse> {
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("missing fake response"))
        }
    }

    impl Transport for FakeTransport {
        fn get<'a>(&'a self, url: &'a str) -> TransportFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(Call::Get(url.to_string()));
                self.next()
            })
        }

        fn post_json<'a>(&'a self, url: &'a str, body: &'a Value) -> TransportFuture<'a> {
            Box::pin(async move {
                self.calls
                    .lock()
                    .unwrap()
                    .push(Call::PostJson(url.to_string(), body.clone()));
                self.next()
            })
        }

        fn put_bytes<'a>(
            &'a self,
            url: &'a str,
            content_type: &'a str,
            bytes: &'a [u8],
        ) -> TransportFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(Call::PutBytes(
                    url.to_string(),
                    content_type.to_string(),
                    bytes.to_vec(),
                ));
                self.next()
            })
        }

        fn get_bytes<'a>(&'a self, url: &'a str) -> TransportFuture<'a> {
            Box::pin(async move {
                self.calls
                    .lock()
                    .unwrap()
                    .push(Call::GetBytes(url.to_string()));
                self.next()
            })
        }
    }

    fn json_response(status: u16, body: Value) -> TransportResponse {
        TransportResponse {
            status,
            retry_after: None,
            body: body.to_string().into_bytes(),
        }
    }

    fn channel(transport: Arc<FakeTransport>) -> DiscordChannel {
        DiscordChannel::with_transport(
            DEFAULT_BASE_URL,
            "900".to_string(),
            vec!["900".to_string()],
            transport,
            // Pacing is disabled for tests; the limiter itself is exercised
            // separately.
            RequestLimiter::new(Duration::ZERO),
        )
    }

    fn user_message(text: &str) -> RawMessage {
        RawMessage {
            row_id: 1,
            provider_event_id: Some("1".to_string()),
            channel: "discord",
            handle: "7".to_string(),
            chat_identifier: "900".to_string(),
            is_group: false,
            text: text.to_string(),
            voice: None,
            images: Vec::new(),
            is_from_me: false,
            is_supported: true,
            thread_id: None,
        }
    }

    // -- limiter -----------------------------------------------------------

    #[test]
    fn the_shared_limiter_is_one_instance_across_channel_loops() {
        assert!(Arc::ptr_eq(
            &RequestLimiter::shared().inner,
            &RequestLimiter::shared().inner
        ));
    }

    #[tokio::test]
    async fn penalized_routes_wait_and_do_not_block_other_routes() {
        let limiter = RequestLimiter::new(Duration::ZERO);
        limiter.penalize("messages:900", Duration::from_millis(120));

        let started = Instant::now();
        limiter.acquire("messages:900").await;
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "route should have waited out its back-off, waited {:?}",
            started.elapsed()
        );
        assert!(limiter.backoff_until("messages:900").is_some());

        let started = Instant::now();
        limiter.acquire("send:900").await;
        assert!(started.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn a_later_shorter_penalty_never_shortens_an_earlier_one() {
        let limiter = RequestLimiter::new(Duration::ZERO);
        limiter.penalize("messages:900", Duration::from_secs(60));
        let long = limiter.backoff_until("messages:900").unwrap();
        limiter.penalize("messages:900", Duration::from_secs(1));
        assert_eq!(limiter.backoff_until("messages:900"), Some(long));
    }

    // -- error shapes ------------------------------------------------------

    #[test]
    fn decodes_the_proxy_error_shape_without_discord_fields() {
        let error = ApiError::decode(
            429,
            br#"{"error":"rate_limited","message":"Discord route rate limit active, retry later"}"#,
        );

        assert_eq!(error.proxy_error.as_deref(), Some("rate_limited"));
        assert_eq!(
            error.message.as_deref(),
            Some("Discord route rate limit active, retry later")
        );
        assert_eq!(error.discord_code, None);
        assert!(error.is_rate_limited());
        assert!(error.describe().contains("rate_limited"));
    }

    #[test]
    fn decodes_the_discord_error_shape_without_proxy_fields() {
        let error = ApiError::decode(404, br#"{"message":"Unknown Channel","code":10003}"#);

        assert_eq!(error.proxy_error, None);
        assert_eq!(error.message.as_deref(), Some("Unknown Channel"));
        assert_eq!(error.discord_code, Some(10003));
        assert!(!error.is_rate_limited());
    }

    #[test]
    fn decodes_a_body_too_large_rejection_and_a_non_json_body() {
        let error = ApiError::decode(413, br#"{"error":"body_too_large"}"#);
        assert_eq!(error.proxy_error.as_deref(), Some("body_too_large"));
        assert_eq!(error.message, None);

        let error = ApiError::decode(502, b"<html>bad gateway</html>");
        assert_eq!(error.proxy_error, None);
        assert_eq!(error.message, None);
        assert_eq!(error.describe(), "HTTP 502");
    }

    #[test]
    fn retry_after_is_an_integer_second_count_with_a_sane_fallback() {
        assert_eq!(retry_after_duration(Some("5")), Duration::from_secs(5));
        assert_eq!(retry_after_duration(Some("0")), Duration::ZERO);
        assert_eq!(
            retry_after_duration(Some("1.5")),
            Duration::from_secs_f64(1.5)
        );
        assert_eq!(retry_after_duration(Some("later")), DEFAULT_RETRY_AFTER);
        assert_eq!(retry_after_duration(None), DEFAULT_RETRY_AFTER);
    }

    // -- request shaping ---------------------------------------------------

    #[tokio::test]
    async fn a_rate_limited_request_backs_off_then_succeeds() {
        let mut limited = json_response(429, json!({"error": "rate_limited"}));
        limited.retry_after = Some("0".to_string());
        let fake = Arc::new(FakeTransport::with_raw(vec![
            json_response(200, json!({"id": "42"})),
            limited,
            json_response(200, json!([])),
        ]));
        let channel = channel(fake.clone());

        assert!(channel.poll(10).await.unwrap().is_empty());

        assert_eq!(fake.calls().len(), 3);
        assert!(channel
            .client
            .limiter
            .backoff_until(&channel.messages_route)
            .is_some());
    }

    #[tokio::test]
    async fn an_exhausted_rate_limit_is_reported_not_panicked() {
        let mut limited = json_response(429, json!({"error": "rate_limited"}));
        limited.retry_after = Some("0".to_string());
        let fake = Arc::new(FakeTransport::with_raw(vec![
            limited;
            MAX_RATE_LIMIT_RETRIES + 2
        ]));
        let channel = channel(fake.clone());

        let error = channel.poll(10).await.unwrap_err();

        assert!(error.to_string().contains("rate_limited"));
        assert_eq!(fake.calls().len(), MAX_RATE_LIMIT_RETRIES + 1);
    }

    #[tokio::test]
    async fn poll_asks_for_new_oldest_first_and_resolves_the_bot_identity_once() {
        let fake = Arc::new(FakeTransport::with_json(vec![
            json!({"id": "42"}),
            json!([
                {"id": "30", "content": "newer", "author": {"id": "7"}},
                {"id": "20", "content": "older", "author": {"id": "7"}}
            ]),
        ]));
        let channel = channel(fake.clone());

        let messages = channel.poll(10).await.unwrap();

        assert_eq!(
            messages.iter().map(|m| m.row_id).collect::<Vec<_>>(),
            vec![20, 30]
        );
        assert_eq!(messages[0].text, "older");
        assert_eq!(messages[0].event_id(), "discord:20");
        assert_eq!(messages[0].chat_identifier, "900");
        assert!(messages.iter().all(|m| !m.is_from_me));

        // The identity is cached, so a second poll costs one request.
        fake.responses
            .lock()
            .unwrap()
            .push_back(json_response(200, json!([])));
        channel.poll(30).await.unwrap();

        assert_eq!(
            fake.calls(),
            vec![
                Call::Get(format!("{DEFAULT_BASE_URL}{API_PREFIX}/users/@me")),
                Call::Get(format!(
                    "{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/messages?after=10&limit=100"
                )),
                Call::Get(format!(
                    "{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/messages?after=30&limit=100"
                )),
            ]
        );
    }

    #[tokio::test]
    async fn latest_cursor_reads_the_newest_message() {
        let fake = Arc::new(FakeTransport::with_json(vec![
            json!([{"id": "1551217339952013343"}]),
        ]));
        let channel = channel(fake.clone());

        assert_eq!(channel.latest_cursor().await.unwrap(), 1551217339952013343);
        assert_eq!(
            fake.calls(),
            vec![Call::Get(format!(
                "{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/messages?limit=1"
            ))]
        );
    }

    #[tokio::test]
    async fn send_posts_content_and_rejects_an_oversized_chunk() {
        let fake = Arc::new(FakeTransport::with_json(vec![json!({"id": "1"})]));
        let channel = channel(fake.clone());

        channel.send_text("900", "reply").await.unwrap();
        assert_eq!(
            fake.calls(),
            vec![Call::PostJson(
                format!("{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/messages"),
                json!({"content": "reply"})
            )]
        );

        let error = channel
            .send_text("900", &"x".repeat(TEXT_LIMIT + 1))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("code point limit"));
    }

    #[tokio::test]
    async fn a_proxy_413_is_reported_with_its_own_shape() {
        let fake = Arc::new(FakeTransport::with_raw(vec![json_response(
            413,
            json!({"error": "body_too_large"}),
        )]));
        let channel = channel(fake.clone());

        let error = channel.send_text("900", "reply").await.unwrap_err();

        assert!(error.to_string().contains("body_too_large"), "{error}");
    }

    // -- mapping -----------------------------------------------------------

    #[test]
    fn maps_content_attachments_and_the_bots_own_messages() {
        let message: DiscordMessage = serde_json::from_value(json!({
            "id": "30",
            "content": "hello <@42>",
            "author": {"id": "7"},
            "attachments": [
                {"filename": "voice-message.ogg", "content_type": "audio/ogg",
                 "size": 4242, "url": "https://cdn.discordapp.com/attachments/900/1/voice-message.ogg"},
                {"filename": "chart.png", "content_type": "image/png",
                 "size": 12, "url": "https://cdn.discordapp.com/attachments/900/2/chart.png"}
            ]
        }))
        .unwrap();

        let raw = discord_message_to_raw("900", Some("42"), message).unwrap();

        assert_eq!(raw.row_id, 30);
        assert_eq!(raw.handle, "7");
        assert!(!raw.is_from_me);
        let voice = raw.voice.unwrap();
        assert_eq!(voice.filename, "voice-message.ogg");
        assert_eq!(voice.mime_type, "audio/ogg");
        assert_eq!(voice.file_size, Some(4242));
        assert!(voice.data.is_none());
        assert_eq!(raw.images.len(), 1);
        assert_eq!(
            raw.images[0].locator,
            "https://cdn.discordapp.com/attachments/900/2/chart.png"
        );

        let own: DiscordMessage = serde_json::from_value(json!({
            "id": "31",
            "content": "",
            "author": {"id": "42", "bot": true},
            "flags": VOICE_MESSAGE_FLAG
        }))
        .unwrap();
        let raw = discord_message_to_raw("900", Some("42"), own).unwrap();
        assert!(raw.is_from_me);
        assert!(raw.voice.is_none());
    }

    // -- trigger rule ------------------------------------------------------

    #[test]
    fn trigger_is_a_mention_a_bang_prefix_or_a_voice_note() {
        let bot = Some("42");

        assert!(is_triggered(
            &user_message("hey <@42> what time is it"),
            bot
        ));
        assert!(is_triggered(&user_message("hey <@!42> ping"), bot));
        assert!(is_triggered(&user_message("!summarise this"), bot));
        assert!(!is_triggered(&user_message("just chatting"), bot));
        assert!(!is_triggered(&user_message("hello <@43>"), bot));
        assert!(!is_triggered(&user_message("hey <@420> not me"), bot));
        assert!(is_triggered(&user_message("!x"), None));
        assert!(!is_triggered(&user_message("mention <@42> here"), None));

        let mut voice = user_message("");
        voice.voice = Some(InboundVoice {
            locator: "https://cdn.discordapp.com/a.ogg".to_string(),
            file_size: Some(10),
            mime_type: "audio/ogg".to_string(),
            filename: "voice-message.ogg".to_string(),
            data: None,
        });
        assert!(is_triggered(&voice, bot));
        assert!(is_triggered(&voice, None));
    }

    // -- two-step upload ---------------------------------------------------

    #[tokio::test]
    async fn the_two_step_upload_carries_the_filename_the_response_omits() {
        let fake = Arc::new(FakeTransport::with_raw(vec![
            json_response(
                200,
                json!({"attachments": [{
                    "id": 0,
                    "upload_filename": "9f0c9c1e/note.txt",
                    "upload_url": "https://discord-attachments-uploads-prd.storage.googleapis.com/9f0c9c1e/note.txt?upload_id=token"
                }]}),
            ),
            TransportResponse {
                status: 200,
                retry_after: None,
                body: Vec::new(),
            },
            json_response(200, json!({"id": "1551217339952013343"})),
        ]));
        let channel = channel(fake.clone());

        let id = channel
            .upload_attachment(
                AttachmentUpload {
                    filename: "note.txt",
                    bytes: b"hello",
                    content_type: "text/plain",
                    voice: None,
                },
                "here you go",
            )
            .await
            .unwrap();

        assert_eq!(id, "1551217339952013343");
        assert_eq!(
            fake.calls(),
            vec![
                Call::PostJson(
                    format!("{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/attachments"),
                    json!({"files": [{"filename": "note.txt", "file_size": 5, "id": 0}]})
                ),
                Call::PutBytes(
                    "https://discord-attachments-uploads-prd.storage.googleapis.com/9f0c9c1e/note.txt?upload_id=token"
                        .to_string(),
                    "text/plain".to_string(),
                    b"hello".to_vec()
                ),
                Call::PostJson(
                    format!("{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/messages"),
                    json!({
                        "content": "here you go",
                        "attachments": [{
                            "id": 0,
                            "filename": "note.txt",
                            "uploaded_filename": "9f0c9c1e/note.txt"
                        }]
                    })
                ),
            ]
        );
    }

    #[tokio::test]
    async fn a_voice_upload_puts_audio_ogg_and_sends_voice_metadata() {
        let fake = Arc::new(FakeTransport::with_raw(vec![
            json_response(
                200,
                json!({"attachments": [{
                    "id": "7",
                    "upload_filename": "uuid/reply.ogg",
                    "upload_url": "https://discord-attachments-uploads-prd.storage.googleapis.com/uuid/reply.ogg?upload_id=token"
                }]}),
            ),
            TransportResponse {
                status: 200,
                retry_after: None,
                body: Vec::new(),
            },
            json_response(200, json!({"id": "5"})),
        ]));
        let channel = channel(fake.clone());

        channel
            .upload_attachment(
                AttachmentUpload {
                    filename: "reply.ogg",
                    bytes: b"OggS",
                    content_type: "audio/ogg",
                    voice: Some(VoiceMetadata {
                        duration_secs: 3.5,
                        waveform: "AAAA".to_string(),
                    }),
                },
                "",
            )
            .await
            .unwrap();

        let calls = fake.calls();
        assert_eq!(
            calls[1],
            Call::PutBytes(
                "https://discord-attachments-uploads-prd.storage.googleapis.com/uuid/reply.ogg?upload_id=token"
                    .to_string(),
                "audio/ogg".to_string(),
                b"OggS".to_vec()
            )
        );
        assert_eq!(
            calls[2],
            Call::PostJson(
                format!("{DEFAULT_BASE_URL}{API_PREFIX}/channels/900/messages"),
                json!({
                    "attachments": [{
                        "id": "7",
                        "filename": "reply.ogg",
                        "uploaded_filename": "uuid/reply.ogg"
                    }],
                    "flags": 8192,
                    "duration_secs": 3.5,
                    "waveform": "AAAA"
                })
            )
        );
    }

    #[tokio::test]
    async fn downloads_voice_and_image_bytes_from_the_cdn() {
        let fake = Arc::new(FakeTransport::with_raw(vec![
            TransportResponse {
                status: 200,
                retry_after: None,
                body: vec![1, 2, 3],
            },
            TransportResponse {
                status: 200,
                retry_after: None,
                body: b"\x89PNG\r\n\x1a\nbody".to_vec(),
            },
        ]));
        let channel = channel(fake.clone());

        let clip = channel
            .download_voice(&InboundVoice {
                locator: "https://cdn.discordapp.com/attachments/900/1/voice-message.ogg"
                    .to_string(),
                file_size: Some(3),
                mime_type: "audio/ogg".to_string(),
                filename: "voice-message.ogg".to_string(),
                data: None,
            })
            .await
            .unwrap();
        assert_eq!(clip.bytes, vec![1, 2, 3]);

        let image = channel
            .download_image(&InboundImage {
                locator: "https://cdn.discordapp.com/attachments/900/2/chart.png".to_string(),
                file_size: Some(12),
                mime_type: Some("image/png".to_string()),
                data: None,
            })
            .await
            .unwrap();
        assert!(image.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));

        assert_eq!(
            fake.calls(),
            vec![
                Call::GetBytes(
                    "https://cdn.discordapp.com/attachments/900/1/voice-message.ogg".to_string()
                ),
                Call::GetBytes(
                    "https://cdn.discordapp.com/attachments/900/2/chart.png".to_string()
                ),
            ]
        );
    }

    #[tokio::test]
    async fn rejects_an_attachment_url_outside_the_cdn() {
        let fake = Arc::new(FakeTransport::default());
        let channel = channel(fake.clone());

        let error = channel
            .download_voice(&InboundVoice {
                locator: "https://example.com/steal".to_string(),
                file_size: Some(3),
                mime_type: "audio/ogg".to_string(),
                filename: "voice-message.ogg".to_string(),
                data: None,
            })
            .await
            .unwrap_err();

        assert!(error.to_string().contains("trusted CDN origin"));
        assert!(fake.calls().is_empty());
    }

    // -- splitter ----------------------------------------------------------
    //
    // The rules under test are the measured ones in
    // `spikes/discord-limits/FINDINGS.md`; each test names the section it pins.

    /// Every non-whitespace character survives, in order, and no chunk is
    /// whitespace-only. True even when a token had to be hard-split.
    fn assert_characters_preserved(chunks: &[String], text: &str) {
        assert_eq!(
            chunks
                .concat()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
            text.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        );
        assert!(chunks
            .iter()
            .all(|chunk| !chunk.trim().is_empty() && chunk.trim_end() == chunk));
    }

    /// `chunks` are right-stripped, so the source survives modulo whitespace at
    /// a chunk boundary. Word order and every non-whitespace character must.
    fn assert_words_preserved(chunks: &[String], text: &str) {
        assert_eq!(
            chunks
                .iter()
                .flat_map(|chunk| chunk.split_whitespace())
                .collect::<Vec<_>>(),
            text.split_whitespace().collect::<Vec<_>>()
        );
        assert!(chunks
            .iter()
            .all(|chunk| !chunk.trim().is_empty() && chunk.trim_end() == chunk));
    }

    /// §1-2: the unit is code points, and the ceiling is 2000.
    #[test]
    fn counts_code_points_not_bytes_utf16_units_or_graphemes() {
        let text = "😀".repeat(TEXT_LIMIT);
        let chunks = discord_chunks(&text);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.chars().count() <= TEXT_LIMIT));
        assert_eq!(chunks.concat(), text);
        // 2000 code points do not fit the 1990 working budget, but they are one
        // legal message, so they become exactly two.
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chars().count(), TEXT_BUDGET);

        // 1500 astral characters are 3000 UTF-16 units and 6000 bytes and still
        // fit in one message: a count in either unit would have split them.
        let astral = "😀".repeat(1500);
        let chunks = discord_chunks(&astral);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chars().count(), 1500);

        // 2000 `e` + U+0301 is 2000 graphemes and 4000 code points, which
        // Discord rejects; counting code points splits it.
        let combining = "e\u{301}".repeat(2000);
        let chunks = discord_chunks(&combining);
        assert!(chunks.len() > 1);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.chars().count() <= TEXT_LIMIT));
        assert_eq!(chunks.concat(), combining);
    }

    /// §6.4/§6.5: links and inline code move whole across a boundary; §5d.
    #[test]
    fn never_cuts_through_a_link_or_an_inline_code_span() {
        let link = "[the exe.dev docs](https://exe.dev/docs.md)";
        let text = format!("{} {}", "a".repeat(TEXT_BUDGET - 20), link);
        let chunks = discord_chunks(&text);
        assert!(chunks.len() >= 2);
        assert_eq!(
            chunks.iter().filter(|chunk| chunk.contains(link)).count(),
            1,
            "{chunks:?}"
        );
        assert!(!chunks
            .iter()
            .any(|chunk| chunk.contains("[the exe.dev docs](") && !chunk.contains(link)));

        let span = "`a long inline code span with spaces`";
        let text = format!("{} {}", "b".repeat(TEXT_BUDGET - 20), span);
        let chunks = discord_chunks(&text);
        assert_eq!(
            chunks.iter().filter(|chunk| chunk.contains(span)).count(),
            1,
            "{chunks:?}"
        );
    }

    /// §6.5: a table header, separator, and rows stay together "where they
    /// fit", and a row is never split in half.
    #[test]
    fn keeps_a_table_whole_across_a_boundary_without_splitting_a_row() {
        let table = "| name | value |\n| - | - |\n| alpha | 1 |\n| beta | 2 |";
        let text = format!(
            "{}\n\n{table}\n\n{}",
            "x".repeat(TEXT_BUDGET - 8),
            "y".repeat(40)
        );
        let chunks = discord_chunks(&text);

        assert!(chunks.len() >= 2);
        assert_eq!(
            chunks.iter().filter(|chunk| chunk.contains(table)).count(),
            1,
            "{chunks:?}"
        );
        for chunk in &chunks {
            for line in chunk
                .lines()
                .filter(|line| line.trim_start().starts_with('|'))
            {
                assert!(
                    line.trim_end().ends_with('|'),
                    "a table row was cut: {line:?} in {chunk:?}"
                );
            }
        }
        assert_words_preserved(&chunks, &text);
    }

    /// §6.5: a continued quote keeps its `> ` prefix, and a list item is whole.
    #[test]
    fn keeps_quote_prefixes_and_list_items_whole() {
        let quote = "> first line of the quote\n> second line of the quote";
        let text = format!(
            "{}\n\n{quote}\n\n- item one\n- item two",
            "q".repeat(TEXT_BUDGET - 4)
        );
        let chunks = discord_chunks(&text);

        assert!(chunks.len() >= 2);
        let quoted: Vec<&String> = chunks
            .iter()
            .filter(|chunk| chunk.contains("of the quote"))
            .collect();
        for chunk in quoted {
            for line in chunk
                .lines()
                .filter(|line| line.trim().contains("of the quote"))
            {
                assert!(line.trim_start().starts_with('>'), "prefix lost: {line:?}");
            }
        }
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| chunk.contains("- item two"))
                .count(),
            1
        );
        assert_words_preserved(&chunks, &text);
    }

    /// §6.3 and §6.2: a fenced block split across the boundary is closed and
    /// reopened with the same language tag, and every chunk stays under the
    /// ceiling with an even number of fence markers.
    #[test]
    fn splits_a_fenced_code_block_with_balanced_fences_and_a_preserved_tag() {
        let body = (0..300)
            .map(|index| {
                format!("    line{index:04} = {index}  # filler to push the fence past the budget")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let text = format!("Here is the fix:\n\n```rust\n{body}\n```\n\nThat is all.");
        let chunks = discord_chunks(&text);

        assert!(chunks.len() >= 3, "{chunks:?}");
        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= TEXT_LIMIT,
                "chunk of {} code points",
                chunk.chars().count()
            );
            assert_eq!(
                chunk.matches("```").count() % 2,
                0,
                "an odd number of fence markers in {chunk:?}"
            );
        }
        // Every chunk that carries code reopens the block with the same tag.
        for chunk in &chunks[1..chunks.len() - 1] {
            assert!(chunk.starts_with("```rust\n"), "{chunk:?}");
        }
        assert!(chunks[0].contains("```rust"));
        assert!(chunks.last().unwrap().contains("That is all."));
        // Close/reopen markers are added, so compare the code and prose words
        // with the fence markers removed from both sides.
        let strip = |value: &str| {
            value
                .replace("```rust", "")
                .replace("```", "")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(strip(&chunks.join("\n")), strip(&text.replace('\n', " ")));
    }

    /// §5b/§6.3: a source fence that is never closed is closed rather than
    /// shipped with an odd marker.
    #[test]
    fn closes_an_unbalanced_source_fence() {
        let chunks = discord_chunks("probe\n```python\ndef f():\n    return 1");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].matches("```").count(), 2, "{chunks:?}");
        assert!(chunks[0].ends_with("```"));
    }

    /// §6.6: a 3034-code-point token is hard-split rather than refused.
    #[test]
    fn hard_splits_a_single_token_longer_than_the_budget() {
        let token = format!("https://example.com/{}", "q".repeat(3000));
        let chunks = discord_chunks(&format!("probe {token}"));

        assert!(chunks.len() >= 2);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.chars().count() <= TEXT_BUDGET));
        // Every character survives, in order, and none of the pieces is a lone
        // whitespace chunk.
        assert_eq!(chunks.concat().matches('q').count(), 3000);
        assert_characters_preserved(&chunks, &format!("probe {token}"));
    }

    /// §4/§6.7: a chunk that is empty after the right-strip is dropped.
    #[test]
    fn drops_whitespace_only_chunks() {
        assert!(discord_chunks("").is_empty());
        assert!(discord_chunks("\n\n\n").is_empty());
        assert!(discord_chunks("   \t \n \n").is_empty());

        // A boundary that falls on blank lines produces an empty middle chunk;
        // it must not be handed to the API (50006 "Cannot send an empty
        // message").
        let text = format!("{}\n\n{}", "a".repeat(TEXT_BUDGET), "b".repeat(TEXT_BUDGET));
        let chunks = discord_chunks(&text);

        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|chunk| !chunk.trim().is_empty()));
        assert_eq!(chunks[0], "a".repeat(TEXT_BUDGET));
    }

    /// §3: the splitter never normalises, and prose survives the round trip.
    #[test]
    fn preserves_prose_and_never_normalises() {
        let text = format!("{}\n\n{}\n", "a".repeat(TEXT_BUDGET), "b".repeat(50));
        let chunks = discord_chunks(&text);
        assert_words_preserved(&chunks, &text);

        let decomposed = "e\u{301}x".repeat(10);
        let chunks = discord_chunks(&decomposed);
        assert_eq!(chunks.concat(), decomposed);
        assert_eq!(chunks.concat(), "e\u{301}x".repeat(10));
    }

    // -- default User-Agent -------------------------------------------------

    /// Binds a local listener that answers exactly one request and returns the
    /// raw request head it saw.
    async fn serve_one_request_head() -> (std::net::SocketAddr, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            while !head.windows(4).any(|window| window == b"\r\n\r\n") {
                let mut buffer = [0u8; 1024];
                let read = socket.read(&mut buffer).await.unwrap();
                if read == 0 {
                    break;
                }
                head.extend_from_slice(&buffer[..read]);
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}")
                .await
                .unwrap();
            let _ = socket.shutdown().await;
            String::from_utf8_lossy(&head).to_string()
        });
        (address, server)
    }

    /// `reqwest::Client::new()` — what every other channel in push uses — sends
    /// **no** User-Agent, and Discord's CDN 403s clients that look like
    /// scripts. This pins both halves of that claim against a local socket
    /// rather than against the live CDN.
    #[tokio::test]
    async fn reqwest_sends_no_default_user_agent_and_discord_sets_one() {
        let (address, server) = serve_one_request_head().await;
        reqwest::Client::new()
            .get(format!("http://{address}/probe"))
            .send()
            .await
            .unwrap();
        let bare_head = server.await.unwrap();

        assert!(
            !bare_head.to_ascii_lowercase().contains("user-agent:"),
            "reqwest::Client::new() sent a User-Agent after all: {bare_head}"
        );

        let (address, server) = serve_one_request_head().await;
        ReqwestTransport::new()
            .unwrap()
            .get(&format!("http://{address}/probe"))
            .await
            .unwrap();
        let discord_head = server.await.unwrap();

        assert!(
            discord_head
                .to_ascii_lowercase()
                .contains(&format!("user-agent: {}", USER_AGENT.to_ascii_lowercase())),
            "the Discord client did not send the explicit User-Agent: {discord_head}"
        );
    }
}
