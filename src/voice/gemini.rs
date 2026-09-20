//! Gemini voice provider (PLAN §6).
//!
//! The exe.dev Gemini integration (`https://gemini.int.exe.xyz`) injects
//! credentials server-side, so — like the Discord integration — no API key
//! exists on this VM. Three calls make up the provider:
//!
//! * **Transcription** — `gemini-3.8-flash` via `generateContent` with the
//!   voice note as inline audio. The `*-transcribe-live` models are
//!   WebSocket-only and reject `generateContent`.
//! * **Speech (primary)** — `gemini-3.8-live-extended-thinking` over the raw
//!   Live WebSocket. The reply is sent as a *framed reading instruction*,
//!   because a bare reply as a user turn is answered rather than read. The
//!   model's own `outputTranscription` is compared against the intended text
//!   and divergence retries once, then falls back.
//! * **Speech (fallback)** — `gemini-3.1-flash-tts-preview` via the
//!   Interactions API. It also returns raw PCM, so the transcode is identical.
//!
//! Both speech paths return raw 16-bit little-endian PCM at 24 kHz mono, so
//! `ffmpeg` runs on both and produces the OGG/Opus Discord voice message.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

use super::{AudioClip, VoiceFuture, VoiceProvider, MAX_AUDIO_BYTES};

/// Default integration hostname. Not a secret: the proxy injects the
/// credential, so the VM only holds this name (PLAN §2).
pub const DEFAULT_BASE_URL: &str = "https://gemini.int.exe.xyz";
/// Transcription model (PLAN §3). WebSocket-only `*-transcribe-live` models
/// cannot be used here.
pub const TRANSCRIPTION_MODEL: &str = "gemini-3.8-flash";
/// Live speech model. The extended-thinking variant requires a thinking level.
pub const LIVE_MODEL: &str = "gemini-3.8-live-extended-thinking";
pub const LIVE_THINKING_LEVEL: &str = "high";
/// HTTP fallback when the Live transcript diverges from the intended text.
pub const FALLBACK_TTS_MODEL: &str = "gemini-3.1-flash-tts-preview";
/// Minimum similarity between the intended text and the Live transcript.
pub const FIDELITY_THRESHOLD: f64 = 0.85;
/// Both speech paths return raw s16le PCM at this rate.
pub const OUTPUT_SAMPLE_RATE: u32 = 24000;
/// Discord's waveform is a fixed 256-sample byte array.
pub const WAVEFORM_SAMPLES: usize = 256;

const WS_PATH: &str =
    "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(90);

pub struct GeminiVoice {
    base_url: String,
    voice_name: String,
    client: reqwest::Client,
}

impl GeminiVoice {
    pub fn new(base_url: String, voice_name: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            voice_name,
            client: reqwest::Client::new(),
        }
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}/v1beta{path}", self.base_url)
    }

    fn websocket_url(&self) -> Result<String> {
        let rest = self
            .base_url
            .strip_prefix("https://")
            .map(|host| format!("wss://{host}"))
            .or_else(|| {
                self.base_url
                    .strip_prefix("http://")
                    .map(|host| format!("ws://{host}"))
            })
            .context("Gemini base URL must be http(s)://")?;
        Ok(format!("{rest}{WS_PATH}"))
    }

    /// `generateContent` transcription. Returns the concatenated candidate
    /// text; the caller trims and rejects an empty result.
    pub async fn transcribe_audio(&self, clip: &AudioClip) -> Result<String> {
        let body = json!({
            "contents": [{
                "parts": [
                    {"text": "Transcribe this voice note verbatim. Output only the transcript."},
                    {"inlineData": {
                        "mimeType": clip.mime_type,
                        "data": base64::engine::general_purpose::STANDARD.encode(&clip.bytes),
                    }}
                ]
            }]
        });
        let value: Value = self
            .post_json(
                &format!("/models/{TRANSCRIPTION_MODEL}:generateContent"),
                &body,
            )
            .await
            .context("Gemini transcription failed")?;
        let text = candidate_text(&value);
        if text.trim().is_empty() {
            bail!("Gemini transcription returned no text");
        }
        Ok(text)
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let response = self
            .client
            .post(self.api_url(path))
            .timeout(REQUEST_TIMEOUT)
            .json(body)
            .send()
            .await
            .context("Gemini request failed")?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .context("read Gemini response body")?;
        if !status.is_success() {
            bail!(
                "Gemini returned HTTP {}: {}",
                status.as_u16(),
                short_error(&bytes)
            );
        }
        serde_json::from_slice(&bytes).context("Gemini returned invalid JSON")
    }

    /// HTTP fallback synthesis via the Interactions API. Returns raw PCM.
    pub async fn synthesize_http(&self, text: &str) -> Result<Vec<u8>> {
        let body = json!({
            "model": FALLBACK_TTS_MODEL,
            "input": text,
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": self.voice_name}]},
        });
        let value = self
            .post_json("/interactions", &body)
            .await
            .context("Gemini fallback speech failed")?;
        let mut data = None;
        for step in value
            .get("steps")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            for entry in step
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if entry.get("type").and_then(Value::as_str) == Some("audio") {
                    data = entry
                        .get("data")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
            }
        }
        let data = data.context("Gemini fallback speech returned no audio")?;
        let pcm = base64::engine::general_purpose::STANDARD
            .decode(data)
            .context("Gemini fallback speech returned invalid base64")?;
        if pcm.is_empty() {
            bail!("Gemini fallback speech returned empty audio");
        }
        Ok(pcm)
    }

    /// One-shot Live synthesis. Returns the PCM and the model's own output
    /// transcript so the caller can check fidelity.
    pub async fn synthesize_live(&self, framed_text: &str) -> Result<(Vec<u8>, String)> {
        let url = self.websocket_url()?;
        let (mut socket, _) = tokio_tungstenite::connect_async(url.as_str())
            .await
            .context("connect to Gemini Live")?;

        let setup = json!({"setup": {
            "model": format!("models/{LIVE_MODEL}"),
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "thinkingConfig": {"thinkingLevel": LIVE_THINKING_LEVEL},
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice_name}}
                }
            },
            "outputAudioTranscription": {},
            "inputAudioTranscription": {}
        }});
        socket
            .send(Message::Text(setup.to_string().into()))
            .await
            .context("send Gemini Live setup")?;

        let run = async {
            // Wait for setupComplete (or a setup error).
            loop {
                let value = next_message(&mut socket).await?;
                if value.get("setupComplete").is_some() {
                    break;
                }
                if let Some(error) = value.get("error") {
                    bail!("Gemini Live setup error: {}", short_value(error));
                }
            }
            let content = json!({"clientContent": {
                "turns": [{"role": "user", "parts": [{"text": framed_text}]}],
                "turnComplete": true
            }});
            socket
                .send(Message::Text(content.to_string().into()))
                .await
                .context("send Gemini Live content")?;

            let mut pcm = Vec::new();
            let mut transcript = String::new();
            loop {
                let value = next_message(&mut socket).await?;
                let content = value.get("serverContent").cloned().unwrap_or(Value::Null);
                if let Some(parts) = content
                    .pointer("/modelTurn/parts")
                    .and_then(Value::as_array)
                {
                    for part in parts {
                        if let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str)
                        {
                            let chunk = base64::engine::general_purpose::STANDARD
                                .decode(data)
                                .context("Gemini Live returned invalid audio base64")?;
                            if pcm.len().saturating_add(chunk.len()) > MAX_AUDIO_BYTES {
                                bail!("Gemini Live audio exceeds the 20 MB limit");
                            }
                            pcm.extend_from_slice(&chunk);
                        }
                    }
                }
                if let Some(text) = content
                    .pointer("/outputTranscription/text")
                    .and_then(Value::as_str)
                {
                    transcript.push_str(text);
                }
                if content
                    .get("turnComplete")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    break;
                }
                if let Some(error) = value.get("error") {
                    bail!("Gemini Live error: {}", short_value(error));
                }
            }
            Ok::<_, anyhow::Error>((pcm, transcript))
        };

        let result = tokio::time::timeout(REQUEST_TIMEOUT, run)
            .await
            .map_err(|_| anyhow::anyhow!("Gemini Live timed out"))?;
        let _ = socket.close(None).await;
        let (pcm, transcript) = result?;
        if pcm.is_empty() {
            bail!("Gemini Live returned no audio");
        }
        Ok((pcm, transcript))
    }

    /// Chooses speech audio for `text`: Live first, retrying once on a
    /// divergent transcript, then the HTTP fallback.
    pub async fn synthesize_pcm(&self, speakable: &str) -> Result<Vec<u8>> {
        if speakable.trim().is_empty() {
            bail!("nothing to speak");
        }
        let framed = frame_reading_instruction(speakable);
        for attempt in 0..2 {
            match self.synthesize_live(&framed).await {
                Ok((pcm, said)) => {
                    let score = fidelity(speakable, &said);
                    if score >= FIDELITY_THRESHOLD {
                        return Ok(pcm);
                    }
                    eprintln!(
                        "gemini live transcript diverged (fidelity {score:.3}, attempt {}); {}",
                        attempt + 1,
                        if attempt == 0 {
                            "retrying"
                        } else {
                            "falling back"
                        }
                    );
                }
                Err(error) => {
                    eprintln!("gemini live synthesis failed: {error:#}");
                    break;
                }
            }
        }
        self.synthesize_http(speakable).await
    }
}

impl VoiceProvider for GeminiVoice {
    fn transcribe<'a>(&'a self, clip: AudioClip) -> VoiceFuture<'a, String> {
        Box::pin(async move { self.transcribe_audio(&clip).await })
    }

    fn synthesize<'a>(&'a self, text: &'a str) -> VoiceFuture<'a, AudioClip> {
        Box::pin(async move {
            let speakable = speakable(text);
            let pcm = self.synthesize_pcm(&speakable).await?;
            let duration_secs = pcm.len() as f64 / 2.0 / OUTPUT_SAMPLE_RATE as f64;
            let waveform = base64::engine::general_purpose::STANDARD.encode(waveform(&pcm));
            let bytes = transcode_pcm_to_ogg(&pcm).await?;
            Ok(AudioClip {
                bytes,
                filename: "reply.ogg".to_string(),
                mime_type: "audio/ogg".to_string(),
                duration_secs: Some(duration_secs),
                waveform: Some(waveform),
            })
        })
    }
}

/// Reads the next text/binary JSON frame. Gemini may answer on either opcode.
async fn next_message<S>(socket: &mut S) -> Result<Value>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        let message = socket
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("Gemini Live closed the connection"))?
            .context("Gemini Live read failed")?;
        let payload = match message {
            Message::Text(text) => text.to_string(),
            Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
            Message::Close(frame) => {
                bail!("Gemini Live closed: {frame:?}");
            }
        };
        if payload.trim().is_empty() {
            continue;
        }
        return serde_json::from_str(&payload).context("Gemini Live returned invalid JSON");
    }
}

/// A bare reply sent as a user turn is answered, not read; framing it as a
/// finished reply to read aloud produces an exact match (PLAN §3, Phase 2).
pub(crate) fn frame_reading_instruction(text: &str) -> String {
    format!(
        "Repeat the following text aloud exactly as written, adding nothing before or after:\n\n\"\"\"\n{text}\n\"\"\""
    )
}

/// Strips markdown so the model does not read syntax aloud. Used for both the
/// spoken text and the 800-character modality budget (PLAN §3).
pub(crate) fn speakable(text: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    for line in text.split('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let mut line = trimmed.to_string();
        while let Some(rest) = line.strip_prefix("> ") {
            line = rest.to_string();
        }
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if hashes > 0 && line.chars().nth(hashes) == Some(' ') {
            line = line[hashes..].trim_start().to_string();
        }
        for marker in ["- ", "* ", "+ "] {
            if let Some(rest) = line.strip_prefix(marker) {
                line = rest.to_string();
                break;
            }
        }
        let digits = line.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 && line[digits..].starts_with(". ") {
            line = line[digits + 2..].to_string();
        }
        out.push_str(&line);
        out.push('\n');
    }
    let stripped = strip_links(&out);
    let stripped: String = stripped
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`'))
        .collect();
    collapse_whitespace(&stripped)
}

/// Replaces `[label](url)` with `label`, leaving non-link brackets alone.
fn strip_links(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '[' {
            if let Some(close) = chars[i + 1..].iter().position(|c| *c == ']') {
                let close = i + 1 + close;
                if chars.get(close + 1) == Some(&'(') {
                    if let Some(end) = chars[close + 1..].iter().position(|c| *c == ')') {
                        for c in &chars[i + 1..close] {
                            out.push(*c);
                        }
                        i = close + 1 + end + 1;
                        continue;
                    }
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Similarity of the intended text and the model's transcript, normalized to
/// lower-case collapsed whitespace. `1.0` is identical. Uses a normalized
/// Levenshtein ratio; the Phase 2 spike used Python's `difflib`, but on a
/// verbatim read both measure ~1.0 and on a rephrased answer both collapse, so
/// the threshold transfers.
pub(crate) fn fidelity(intended: &str, said: &str) -> f64 {
    let a = collapse_whitespace(&intended.to_lowercase());
    let b = collapse_whitespace(&said.to_lowercase());
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let longest = a.chars().count().max(b.chars().count());
    if longest == 0 {
        return 1.0;
    }
    let distance = levenshtein(&a, &b);
    1.0 - distance as f64 / longest as f64
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for (i, ac) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, bc) in b.iter().enumerate() {
            let cost = usize::from(ac != bc);
            current[j + 1] = (previous[j] + cost)
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// Discord's 256-sample waveform: per-bucket peak amplitude scaled to 0-255.
pub(crate) fn waveform(pcm: &[u8]) -> Vec<u8> {
    let mut samples: Vec<i16> = Vec::with_capacity(pcm.len() / 2);
    let mut index = 0;
    while index + 1 < pcm.len() {
        samples.push(i16::from_le_bytes([pcm[index], pcm[index + 1]]));
        index += 2;
    }
    if samples.is_empty() {
        return vec![0; WAVEFORM_SAMPLES];
    }
    let per = samples.len().div_ceil(WAVEFORM_SAMPLES).max(1);
    let mut out = Vec::with_capacity(WAVEFORM_SAMPLES);
    for bucket in 0..WAVEFORM_SAMPLES {
        let start = bucket * per;
        let end = (start + per).min(samples.len());
        let peak = samples[start.min(samples.len())..end]
            .iter()
            .map(|s| s.unsigned_abs() as u32)
            .max()
            .unwrap_or(0);
        out.push(((peak * 255) / 32768).min(255) as u8);
    }
    out
}

/// Decodes raw 24 kHz s16le mono PCM into an OGG/Opus stream via `ffmpeg`.
/// Opus always runs at 48 kHz internally, so ffmpeg resamples.
pub(crate) async fn transcode_pcm_to_ogg(pcm: &[u8]) -> Result<Vec<u8>> {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-i",
            "-",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-application",
            "voip",
            "-f",
            "ogg",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("spawn ffmpeg")?;
    let mut stdin = child.stdin.take().context("ffmpeg stdin unavailable")?;
    let pcm = pcm.to_vec();
    let write = tokio::spawn(async move {
        let _ = stdin.write_all(&pcm).await;
        let _ = stdin.shutdown().await;
    });
    let output = child.wait_with_output().await.context("run ffmpeg")?;
    let _ = write.await;
    if !output.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    if output.stdout.is_empty() {
        bail!("ffmpeg produced no audio");
    }
    Ok(output.stdout)
}

/// Concatenates the `text` parts of a `generateContent` response.
fn candidate_text(value: &Value) -> String {
    let mut text = String::new();
    for candidate in value
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for part in candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(part) = part.get("text").and_then(Value::as_str) {
                text.push_str(part);
            }
        }
    }
    text
}

/// A bounded, non-secret error snippet from a JSON error body.
fn short_error(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    text.chars().take(300).collect()
}

fn short_value(value: &Value) -> String {
    value.to_string().chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_the_reply_as_a_reading_instruction() {
        let framed = frame_reading_instruction("hello");
        assert!(framed.contains("exactly as written"));
        assert!(framed.contains("hello"));
    }

    #[test]
    fn speakable_strips_markdown_syntax() {
        let text = "# Title\n\nSome **bold** and `code` and a [link](https://x).\n\n```rust\nfn main() {}\n```\n> quote\n- item\n1. ordered";
        let spoken = speakable(text);
        assert!(spoken.contains("Title"));
        assert!(spoken.contains("bold"));
        assert!(!spoken.contains('*'));
        assert!(!spoken.contains('`'));
        assert!(!spoken.contains("fn main"));
        assert!(spoken.contains("link"));
        assert!(!spoken.contains("https://x"));
        assert!(spoken.contains("quote"));
        assert!(!spoken.contains('>'));
        assert!(spoken.contains("item"));
        assert!(spoken.contains("ordered"));
    }

    #[test]
    fn fidelity_scores_identical_high_and_rephrased_low() {
        assert!(fidelity("Hello there friend", "  hello   THERE friend ") >= 0.99);
        assert!(fidelity("The build passed.", "Thanks for letting me know!") < FIDELITY_THRESHOLD);
    }

    #[test]
    fn waveform_scales_peaks_into_256_buckets() {
        let pcm: Vec<u8> = (0..2000)
            .flat_map(|i| ((i as i16 % 100) * 300).to_le_bytes())
            .collect();
        let wave = waveform(&pcm);
        assert_eq!(wave.len(), WAVEFORM_SAMPLES);
        assert!(wave.iter().any(|value| *value > 0));
        assert_eq!(waveform(&[]), vec![0; WAVEFORM_SAMPLES]);
    }

    #[test]
    fn candidate_text_concatenates_parts() {
        let value: Value = serde_json::from_str(
            r#"{"candidates":[{"content":{"parts":[{"text":"hello "},{"text":"world"}]}}]}"#,
        )
        .unwrap();
        assert_eq!(candidate_text(&value), "hello world");
    }

    #[test]
    fn websocket_url_swaps_scheme_and_keeps_host() {
        let voice = GeminiVoice::new("https://gemini.int.exe.xyz".to_string(), "Kore".to_string());
        assert_eq!(
            voice.websocket_url().unwrap(),
            format!("wss://gemini.int.exe.xyz{WS_PATH}")
        );
    }

    #[tokio::test]
    async fn transcodes_raw_pcm_to_ogg_opus_with_ffmpeg() {
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| !status.success())
            .unwrap_or(true)
        {
            eprintln!("ffmpeg unavailable; skipping transcode test");
            return;
        }
        // ~0.25 s of a quiet tone at 24 kHz s16le mono.
        let pcm: Vec<u8> = (0..6000)
            .flat_map(|i| (((i % 100) as i16 - 50) * 100).to_le_bytes())
            .collect();
        let ogg = transcode_pcm_to_ogg(&pcm).await.unwrap();
        assert!(ogg.starts_with(b"OggS"), "expected an Ogg container");
    }
}
