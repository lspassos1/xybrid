//! Live (rolling-window) ASR bindings for Flutter.
//!
//! Wraps the SDK's [`xybrid_sdk::XybridStream`] so a Dart caller can feed
//! microphone PCM and receive partial transcripts as a `Stream`. A session is
//! created from an already-loaded model via `FfiModel::stream` — the model's
//! on-disk location is resolved by the SDK, so the same registry / Hugging Face
//! / bundle / directory model you loaded for batch inference streams with no
//! extra path wrangling. Audio flows *in* via [`FfiStreamSession::feed`];
//! partial transcripts flow *out* via the sink from [`FfiStreamSession::subscribe`].
//!
//! Backend (Whisper / Wav2Vec2) is auto-detected from the model metadata.
//!
//! # Threading model
//!
//! A single worker thread owns the `XybridStream` for its whole lifetime.
//! Commands reach it over a [`tokio::sync::mpsc`] channel, so they are applied
//! in submission order — audio fed in order is transcribed in order. [`feed`]
//! is a cheap, non-blocking channel send (`#[frb(sync)]`); the heavy inference
//! runs on the worker, never on the Dart isolate, so the UI never stalls.
//! (Rolling-window chunking itself lives entirely in `xybrid-core`; this
//! binding is a transport — see the module docs of `xybrid_core::streaming`.)
//!
//! # Audio contract
//!
//! Samples are PCM **f32, mono, 16 kHz**. Conversion from the platform mic
//! format is the caller's responsibility (kept out of FFI deliberately).
//!
//! [`feed`]: FfiStreamSession::feed
//! [`subscribe`]: FfiStreamSession::subscribe

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use flutter_rust_bridge::frb;
use xybrid_ffi_facade as facade;

use crate::frb_generated::StreamSink;

/// 16 kHz mono — the only sample rate the ASR backends accept.
pub const REQUIRED_SAMPLE_RATE: u32 = 16_000;

static LOG_INIT: Once = Once::new();

/// Install a `log` backend and a panic hook on first use. Idempotent.
///
/// Without this the binding registers no logger, so on Android every `log::*`
/// line across the whole Rust stack is dropped and a panic on a worker thread
/// dies silently. This makes both visible in `logcat` (tag `xybrid`).
fn ensure_logging() {
    LOG_INIT.call_once(|| {
        #[cfg(target_os = "android")]
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Debug)
                .with_tag("xybrid"),
        );
        // Surface panics that would otherwise vanish on a detached worker thread.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            log::error!("xybrid RUST PANIC: {info}");
            prev(info);
        }));
    });
}

/// How voice-activity detection (VAD) chunking is resolved for a session.
///
/// Models one decision in one type instead of a `bool` + `Option<String>`
/// pair where illegal states (disabled, yet a path is set) are representable.
#[derive(Debug, Clone)]
pub enum FfiVadMode {
    /// Fixed time-window chunking; no voice-activity detection.
    Off,
    /// VAD on, using the bundled default Silero model.
    Default,
    /// VAD on, using a Silero model from this directory.
    Custom {
        /// Directory containing the VAD model.
        model_dir: String,
    },
}

/// Configuration for a live ASR session.
///
/// The model itself is not named here — it comes from the loaded `FfiModel`
/// you call `stream` on. This only configures *how* the audio is chunked.
#[derive(Debug, Clone)]
pub struct FfiStreamingConfig {
    /// PCM sample rate of the audio you will feed. Must be 16 kHz; validated
    /// (and asserted as the contract) rather than forwarded — the backends are
    /// fixed at 16 kHz.
    pub sample_rate: u32,
    /// Voice-activity-detection mode.
    pub vad: FfiVadMode,
    /// Optional language hint (e.g. `"en"`); `None` uses the model default.
    pub language: Option<String>,
    /// Optional Whisper encoder context in mel frames; `None` uses the model default.
    pub audio_ctx: Option<u32>,
}

impl FfiStreamingConfig {
    /// Validate and convert to the shared facade's streaming config.
    ///
    /// `pub(crate)` so `FfiModel::stream` can build the SDK request without
    /// re-exposing the SDK type at the Dart boundary.
    pub(crate) fn to_facade(&self) -> facade::AsrStreamConfig {
        let (enable_vad, vad_model_dir) = match &self.vad {
            FfiVadMode::Off => (false, None),
            FfiVadMode::Default => (true, None),
            FfiVadMode::Custom { model_dir } => (true, Some(model_dir.clone())),
        };
        facade::AsrStreamConfig {
            sample_rate: self.sample_rate,
            enable_vad,
            vad_threshold: 0.5,
            vad_model_dir,
            language: self.language.clone(),
            audio_ctx: self.audio_ctx,
        }
    }
}

/// A partial transcript emitted while audio is streaming.
#[derive(Debug, Clone)]
pub struct FfiPartialResult {
    /// Best-effort transcript text so far.
    pub text: String,
    /// `true` once this span is committed and will not change.
    pub is_stable: bool,
    /// Monotonic chunk sequence number this result corresponds to.
    pub chunk_sequence: u64,
    /// Audio covered so far, in milliseconds.
    pub audio_duration_ms: u64,
}

impl From<facade::AsrPartialResult> for FfiPartialResult {
    fn from(p: facade::AsrPartialResult) -> Self {
        Self {
            text: p.text,
            is_stable: p.is_stable,
            chunk_sequence: p.chunk_index,
            audio_duration_ms: p.audio_duration_ms,
        }
    }
}

/// A live ASR session usable from Dart.
///
/// The shared facade owns the ordered worker and pull queue. This wrapper only
/// adapts its pull-based partials to Dart's `StreamSink`, so Flutter, Swift,
/// Kotlin, and Unity all use the same warm-up and deduplication behavior.
#[frb(opaque)]
pub struct FfiStreamSession {
    inner: Arc<facade::AsrStreamingSession>,
    subscribed: AtomicBool,
}

impl FfiStreamSession {
    /// Wrap a shared facade session.
    ///
    /// `pub(crate)`: not an FFI entry point. Callers reach this through
    /// `FfiModel::stream`, which resolves the model directory for us.
    pub(crate) fn new(inner: Arc<facade::AsrStreamingSession>) -> Self {
        ensure_logging();
        Self {
            inner,
            subscribed: AtomicBool::new(false),
        }
    }

    /// Subscribe to partial transcripts. Call this once per session.
    ///
    /// Partials already queued before subscription are delivered in order, so
    /// there is no feed/subscribe race. Repeated subscriptions are ignored.
    ///
    /// [`feed`]: Self::feed
    pub fn subscribe(&self, sink: StreamSink<FfiPartialResult>) {
        if self.subscribed.swap(true, Ordering::AcqRel) {
            return;
        }
        let session = Arc::clone(&self.inner);
        let spawn_result = std::thread::Builder::new()
            .name("xybrid-asr-dart".into())
            .spawn(move || loop {
                match session.next() {
                    Ok(Some(partial)) => {
                        if sink.add(FfiPartialResult::from(partial)).is_err() {
                            let _ = session.close();
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        log::warn!("live ASR partial stream stopped: {error}");
                        break;
                    }
                }
            });
        if let Err(error) = spawn_result {
            log::warn!("failed to spawn Dart ASR delivery thread: {error}");
        }
    }

    /// Feed PCM f32 mono 16 kHz samples.
    ///
    /// A cheap, ordered, non-blocking channel send — inference happens on the
    /// worker thread. Takes an owned `Vec` because the samples are moved across
    /// the thread boundary; frb hands us an owned buffer already, so this adds
    /// no copy beyond the unavoidable cross-thread handoff.
    ///
    /// # Errors
    ///
    /// If the session has been finalized (after [`flush`]) or otherwise torn
    /// down, so the worker is no longer accepting audio.
    ///
    /// [`flush`]: Self::flush
    #[frb(sync)]
    pub fn feed(&self, samples: Vec<f32>) -> Result<(), String> {
        self.inner.feed(samples).map_err(|error| error.to_string())
    }

    /// Finalize: drain buffered audio and return the complete transcript.
    ///
    /// After this the session is finalized; further [`feed`] calls error.
    ///
    /// # Errors
    ///
    /// If finalization fails in the core, or the worker is already gone.
    ///
    /// [`feed`]: Self::feed
    pub async fn flush(&self) -> Result<String, String> {
        let session = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || session.flush())
            .await
            .map_err(|error| format!("ASR flush worker failed: {error}"))?
            .map(|result| result.text)
            .map_err(|error| error.to_string())
    }

    /// Reset the session to transcribe fresh audio without reloading the model.
    ///
    /// # Errors
    ///
    /// If the reset fails in the core, or the worker is already gone.
    pub async fn reset(&self) -> Result<(), String> {
        let session = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || session.reset())
            .await
            .map_err(|error| format!("ASR reset worker failed: {error}"))?
            .map_err(|error| error.to_string())
    }
}

impl Drop for FfiStreamSession {
    fn drop(&mut self) {
        let _ = self.inner.close();
    }
}
