//! Provider-neutral voice transcription and speech synthesis.
//!
//! The production provider is Gemini (`crate::voice::gemini`); the
//! `VoiceProvider` boundary keeps the gateway free of provider specifics and
//! makes the delivery decisions unit-testable.
#![cfg_attr(test, allow(dead_code))]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::config::Config;

mod gemini;

pub use gemini::{DEFAULT_BASE_URL as GEMINI_VOICE_BASE_URL, TRANSCRIPTION_MODEL};

pub const MAX_AUDIO_BYTES: usize = 20 * 1024 * 1024;
const MAX_SPEECH_CHARS: usize = 4096;

/// Speakable budget for a voice reply. At or above this the answer is delivered
/// as text instead (PLAN §3, Reply modality).
pub const VOICE_REPLY_MAX_CHARS: usize = 800;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioClip {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
    /// Voice-message metadata, present on synthesized replies (PLAN §6).
    pub duration_secs: Option<f64>,
    /// base64 of Discord's 256-sample waveform, present on synthesized replies.
    pub waveform: Option<String>,
}

pub(crate) type VoiceFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

pub trait VoiceProvider: Send + Sync {
    fn transcribe<'a>(&'a self, clip: AudioClip) -> VoiceFuture<'a, String>;
    fn synthesize<'a>(&'a self, text: &'a str) -> VoiceFuture<'a, AudioClip>;
}

#[derive(Clone)]
pub struct Voice {
    provider: Arc<dyn VoiceProvider>,
}

impl Voice {
    /// The Gemini integration needs no key on the VM (PLAN §2), so voice is
    /// always available; only a per-call failure falls back to text.
    pub fn from_config(config: &Config) -> Option<Self> {
        let base_url = config
            .voice_gemini_base_url
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| gemini::DEFAULT_BASE_URL.to_string());
        Some(Self {
            provider: Arc::new(gemini::GeminiVoice::new(
                base_url,
                config.voice_name.clone(),
            )),
        })
    }

    #[cfg(test)]
    pub fn with_provider(provider: Arc<dyn VoiceProvider>) -> Self {
        Self { provider }
    }

    pub async fn transcribe(&self, clip: AudioClip) -> Result<String> {
        if clip.bytes.is_empty() {
            bail!("voice message is empty");
        }
        if clip.bytes.len() > MAX_AUDIO_BYTES {
            bail!("voice message exceeds the 20 MB limit");
        }
        let text = self.provider.transcribe(clip).await?;
        let text = text.trim();
        if text.is_empty() {
            bail!("voice message did not contain recognizable speech");
        }
        Ok(text.to_string())
    }

    pub async fn synthesize(&self, text: &str) -> Result<AudioClip> {
        if text.trim().is_empty() {
            bail!("speech reply is empty");
        }
        if text.chars().count() > MAX_SPEECH_CHARS {
            bail!("speech reply exceeds the 4096 character limit");
        }
        let clip = self.provider.synthesize(text).await?;
        if clip.bytes.is_empty() {
            bail!("speech reply is empty");
        }
        if clip.bytes.len() > MAX_AUDIO_BYTES {
            bail!("speech reply exceeds the 20 MB limit");
        }
        Ok(clip)
    }
}

/// How a reply is delivered: exactly one copy, chosen deterministically
/// (PLAN §3, Reply modality).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyModality {
    Text,
    Voice,
}

/// A typed message always gets text. A voice note gets voice only when the
/// answer is short enough to listen to and has no construct that renders badly
/// as speech; otherwise text. Pure, so the rule is unit-testable.
pub fn select_modality(input_was_voice: bool, answer: &str) -> ReplyModality {
    if !input_was_voice {
        return ReplyModality::Text;
    }
    if answer.contains("```") || answer.contains("~~~") {
        return ReplyModality::Text;
    }
    if answer
        .lines()
        .any(|line| line.trim_start().starts_with('|'))
    {
        return ReplyModality::Text;
    }
    if answer.contains("http://") || answer.contains("https://") {
        return ReplyModality::Text;
    }
    if answer.contains("[send-file:") {
        return ReplyModality::Text;
    }
    if gemini::speakable(answer).chars().count() >= VOICE_REPLY_MAX_CHARS {
        return ReplyModality::Text;
    }
    ReplyModality::Voice
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeProvider;

    impl VoiceProvider for FakeProvider {
        fn transcribe<'a>(&'a self, _clip: AudioClip) -> VoiceFuture<'a, String> {
            Box::pin(async { Ok("  remind me tomorrow  ".to_string()) })
        }

        fn synthesize<'a>(&'a self, _text: &'a str) -> VoiceFuture<'a, AudioClip> {
            Box::pin(async {
                Ok(AudioClip {
                    bytes: vec![1, 2, 3],
                    filename: "reply.ogg".to_string(),
                    mime_type: "audio/ogg".to_string(),
                    duration_secs: Some(1.0),
                    waveform: Some("AAAA".to_string()),
                })
            })
        }
    }

    fn clip(bytes: Vec<u8>) -> AudioClip {
        AudioClip {
            bytes,
            filename: "voice.ogg".to_string(),
            mime_type: "audio/ogg".to_string(),
            duration_secs: None,
            waveform: None,
        }
    }

    #[tokio::test]
    async fn provider_boundary_trims_transcripts_and_returns_ogg() {
        let voice = Voice::with_provider(Arc::new(FakeProvider));
        assert_eq!(
            voice.transcribe(clip(vec![1])).await.unwrap(),
            "remind me tomorrow"
        );
        let out = voice.synthesize("done").await.unwrap();
        assert_eq!(out.mime_type, "audio/ogg");
        assert_eq!(out.duration_secs, Some(1.0));
    }

    #[tokio::test]
    async fn provider_boundary_rejects_empty_and_oversized_input() {
        let voice = Voice::with_provider(Arc::new(FakeProvider));
        assert!(voice.transcribe(clip(Vec::new())).await.is_err());
        assert!(voice
            .transcribe(clip(vec![0; MAX_AUDIO_BYTES + 1]))
            .await
            .is_err());
        assert!(voice
            .synthesize(&"x".repeat(MAX_SPEECH_CHARS + 1))
            .await
            .is_err());
    }

    #[test]
    fn modality_rule_is_deterministic() {
        // Typed input never gets voice.
        assert_eq!(select_modality(false, "short"), ReplyModality::Text);
        // A short voice answer gets voice.
        assert_eq!(select_modality(true, "yes, done"), ReplyModality::Voice);
        // At/over the speakable budget it becomes text, measured on the
        // speakable text not the raw length.
        assert_eq!(
            select_modality(true, &"a".repeat(VOICE_REPLY_MAX_CHARS)),
            ReplyModality::Text
        );
        assert_eq!(
            select_modality(true, &"a".repeat(VOICE_REPLY_MAX_CHARS - 1)),
            ReplyModality::Voice
        );
        // Hard overrides to text.
        assert_eq!(
            select_modality(true, "```rust\nlet x = 1;\n```"),
            ReplyModality::Text
        );
        assert_eq!(
            select_modality(true, "| a | b |\n| - | - |"),
            ReplyModality::Text
        );
        assert_eq!(
            select_modality(true, "see https://example.com"),
            ReplyModality::Text
        );
        assert_eq!(
            select_modality(true, "here\n[send-file: report.pdf]"),
            ReplyModality::Text
        );
    }
}
