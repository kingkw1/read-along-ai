"""Gradio application for the Read-Along AI hackathon submission.

The UI routes through wrapper functions so cloud and local inference paths can
share the same frontend event wiring.
"""

from __future__ import annotations

import base64
import html
import io
import inspect
import json
import logging
import os
import re
import tempfile
import threading
import warnings
import wave
from array import array
from math import sqrt
from pathlib import Path
from typing import Optional

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    category=Warning,
)

import gradio as gr
import modal

from local_inference import _load_whisper_model, local_ask_minicpm_judge, local_synthesize_speech, local_transcribe_audio

LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODAL_APP_NAME = "read-along-ai-inference"
TURBO_ENGINE = "⚡ Turbo Mode (Modal)"
LOCAL_ENGINE = "🏕️ Off the Grid Mode (Local)"
INFERENCE_ENGINES = [TURBO_ENGINE, LOCAL_ENGINE]
DEFAULT_INFERENCE_ENGINE = LOCAL_ENGINE

CURRICULUM = ["The cat sat.", "The dog ran fast.", "She had a red hat.", "I love to play outside."]
TTS_MEMORY_CACHE: dict[tuple[str, str], bytes] = {}
TTS_PREWARM_STATUS: dict[str, object] = {
    "sentence": "",
    "total": 0,
    "ready": 0,
    "failed": 0,
    "running": False,
    "ready_words": [],
    "clip_method": "",
    "fallback_reason": "",
}
TTS_CACHE_LOCK = threading.Lock()

SAMPLE_RATE = 16_000
DUMMY_AUDIO_SECONDS = 1
WORD_CLIP_PADDING_SECONDS = 0.06
CURRICULUM_AUDIO_MANIFEST_PATH = Path("data/curriculum_audio/manifest.json")
LOCAL_CURRICULUM_AUDIO_VARIANT = os.environ.get("LOCAL_CURRICULUM_AUDIO_VARIANT", "comma")
LOCAL_LIVE_TTS_ENABLED = os.environ.get("LOCAL_LIVE_TTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _write_silent_wav(label: str = "speech") -> str:
    """Create a short silent WAV file and return its local path."""
    safe_label = "".join(ch for ch in label.lower() if ch.isalnum() or ch in ("-", "_"))[:24] or "speech"
    output_path = Path(tempfile.gettempdir()) / f"read_along_{safe_label}.wav"
    frame_count = SAMPLE_RATE * DUMMY_AUDIO_SECONDS

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * frame_count)

    return str(output_path)


def _modal_function(function_name: str):
    lookup = getattr(modal.Function, "lookup", None)
    if lookup is not None:
        return lookup(MODAL_APP_NAME, function_name)
    return modal.Function.from_name(MODAL_APP_NAME, function_name)


def modal_credentials_available() -> bool:
    """Return whether the Space has enough Modal credentials to call endpoints."""
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


def run_cohere_asr(audio_bytes: bytes) -> dict[str, str]:
    """Invoke the deployed Modal ASR endpoint."""
    return _modal_function("run_cohere_asr").remote(audio_bytes)


def run_voxcpm_tts(text: str) -> bytes:
    """Invoke the deployed Modal TTS endpoint."""
    return _modal_function("run_voxcpm_tts").remote(text)


def run_minicpm_evaluator(target_text: str, transcript: str) -> str:
    """Invoke the deployed Modal MiniCPM phonetic evaluator endpoint."""
    return _modal_function("run_minicpm_evaluator").remote(target_text, transcript)


# ---------------------------------------------------------------------------
# Backend abstraction layer required by docs/API_CONTRACT_SPEC.md.
# ---------------------------------------------------------------------------
def transcribe_audio(audio_filepath: str, inference_engine: str = TURBO_ENGINE) -> str:
    """Return a clean transcription for a local microphone recording."""
    try:
        if inference_engine == LOCAL_ENGINE:
            return local_transcribe_audio(audio_filepath)

        audio_bytes = Path(audio_filepath).read_bytes()
        result = run_cohere_asr(audio_bytes)
        if result.get("status") != "success":
            return "[ASR_ERROR]"
        return normalize_text(result.get("text", ""))
    except Exception:
        return "[ASR_ERROR]"


def synthesize_speech(target_text: str, inference_engine: str = TURBO_ENGINE) -> Optional[str]:
    """Return a local WAV path for generated speech."""
    try:
        audio_bytes = synthesize_speech_bytes(target_text, inference_engine)
        return write_tts_audio_file(target_text, audio_bytes)
    except Exception:
        return None


def safe_tts_label(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in ("-", "_"))[:24] or "speech"


def write_tts_audio_file(label: str, audio_bytes: bytes) -> str:
    output_path = Path(tempfile.gettempdir()) / f"read_along_{safe_tts_label(label)}.wav"
    output_path.write_bytes(audio_bytes)
    return str(output_path)


def synthesize_speech_bytes(target_text: str, inference_engine: str = TURBO_ENGINE) -> bytes:
    """Return generated speech as WAV bytes for in-memory caching."""
    tts_text = format_text_for_tts(target_text)
    if inference_engine == LOCAL_ENGINE:
        if not LOCAL_LIVE_TTS_ENABLED:
            raise RuntimeError("Local live VoxCPM TTS is disabled; set LOCAL_LIVE_TTS=1 to enable fallback generation.")
        return Path(local_synthesize_speech(tts_text)).read_bytes()
    return run_voxcpm_tts(tts_text)


def normalize_text(text: str) -> str:
    """Normalize spoken/target text for tolerant reading evaluation."""
    return re.sub(r"\s+", " ", "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())).strip()


def clean_tts_word(word: str) -> str:
    """Normalize a single TTS helper word for cache lookup."""
    return normalize_text(word)


def format_text_for_tts(text: str) -> str:
    """Capitalize and punctuate isolated words before sending them to VoxCPM."""
    stripped_text = text.strip()
    if stripped_text and not any(char.isspace() for char in stripped_text):
        return f"{stripped_text[:1].upper()}{stripped_text[1:]}."
    return text


def sentence_tts_words(sentence: str) -> list[str]:
    """Return unique cleaned words in reading order for TTS prewarming."""
    words: list[str] = []
    seen: set[str] = set()
    for raw_word in sentence.split():
        word = clean_tts_word(raw_word)
        if word and word not in seen:
            words.append(word)
            seen.add(word)
    return words


def tts_cache_key(sentence: str, word: str) -> tuple[str, str]:
    return normalize_text(sentence), clean_tts_word(word)


def _curriculum_audio_entries() -> list[dict[str, object]]:
    try:
        return json.loads(CURRICULUM_AUDIO_MANIFEST_PATH.read_text())
    except Exception as exc:
        LOGGER.warning("Could not read local curriculum audio manifest: %s", exc)
        return []


def local_curriculum_audio_entry(sentence: str, variant: str = LOCAL_CURRICULUM_AUDIO_VARIANT) -> Optional[dict[str, object]]:
    """Return the committed local audio manifest entry for a curriculum sentence."""
    sentence_key = normalize_text(sentence)
    fallback_entry: Optional[dict[str, object]] = None
    for entry in _curriculum_audio_entries():
        if normalize_text(str(entry.get("sentence", ""))) != sentence_key:
            continue
        if entry.get("variant") == variant:
            return entry
        fallback_entry = fallback_entry or entry
    return fallback_entry


def local_curriculum_audio_path(sentence: str) -> Optional[Path]:
    entry = local_curriculum_audio_entry(sentence)
    if not entry:
        return None
    wav_path = Path(str(entry.get("wav", "")))
    if wav_path.exists():
        return wav_path
    return None


def local_curriculum_label_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_labels.txt")


def read_curriculum_word_timestamps(label_path: Path) -> dict[str, tuple[float, float]]:
    """Read Audacity-style start/end/word labels for committed local curriculum audio."""
    timestamps: dict[str, tuple[float, float]] = {}
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        start_text, end_text, *word_parts = parts
        word = clean_tts_word(" ".join(word_parts))
        if not word:
            continue
        start = float(start_text)
        end = float(end_text)
        if end <= start:
            raise ValueError(f"invalid label timestamp for {word!r}: {start}-{end}")
        timestamps.setdefault(word, (start, end))
    return timestamps


def slice_local_curriculum_word_clips(sentence: str) -> tuple[dict[str, bytes], str]:
    """Slice committed local curriculum WAVs with checked-in label timings."""
    audio_path = local_curriculum_audio_path(sentence)
    if audio_path is None:
        return {}, "Committed curriculum WAV is missing."
    label_path = local_curriculum_label_path(audio_path)
    if not label_path.exists():
        return {}, "Committed curriculum word label file is missing."
    try:
        timestamps = read_curriculum_word_timestamps(label_path)
        clips = slice_sentence_audio_by_timestamps(sentence, audio_path.read_bytes(), timestamps)
    except Exception as exc:
        LOGGER.warning("Could not slice committed curriculum audio for %r: %s", sentence, exc)
        return {}, str(exc)
    return clips, ""

def wav_duration_seconds(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def _write_wav_clip(params: wave._wave_params, frames: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(frames)
    return output.getvalue()


def _slice_wav_by_seconds(audio_bytes: bytes, start_seconds: float, end_seconds: float) -> Optional[bytes]:
    """Return a WAV clip for a timestamp range, padded and clamped to the source audio."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        params = wav_file.getparams()
        total_frames = wav_file.getnframes()
        all_frames = wav_file.readframes(total_frames)

    if total_frames <= 0 or end_seconds <= start_seconds:
        return None

    padding_frames = int(params.framerate * WORD_CLIP_PADDING_SECONDS)
    start_frame = max(0, int(start_seconds * params.framerate) - padding_frames)
    end_frame = min(total_frames, int(end_seconds * params.framerate) + padding_frames)
    if end_frame <= start_frame:
        return None

    bytes_per_frame = params.nchannels * params.sampwidth
    return _write_wav_clip(params, all_frames[start_frame * bytes_per_frame : end_frame * bytes_per_frame])


def _read_mono_pcm16_samples(audio_bytes: bytes) -> tuple[int, list[int]]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"signal alignment expects 16-bit PCM WAV, got {sample_width * 8}-bit")

    pcm = array("h")
    pcm.frombytes(frames)
    if channels > 1:
        pcm = array("h", (pcm[index] for index in range(0, len(pcm), channels)))
    return frame_rate, list(pcm)


def _short_time_rms(samples: list[int], frame_rate: int, frame_seconds: float = 0.005) -> tuple[list[float], int]:
    frame_size = max(1, int(frame_rate * frame_seconds))
    rms_values: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms_values.append(sqrt(sum(sample * sample for sample in frame) / len(frame)))
    return rms_values, frame_size


def _local_rms_minimum(rms_values: list[float], frame_size: int, frame_rate: int, target_seconds: float, window_seconds: float) -> float:
    if not rms_values:
        return target_seconds
    target_index = int(target_seconds * frame_rate / frame_size)
    radius = max(1, int(window_seconds * frame_rate / frame_size))
    start_index = max(0, target_index - radius)
    end_index = min(len(rms_values) - 1, target_index + radius)
    best_index = min(range(start_index, end_index + 1), key=lambda index: (rms_values[index], abs(index - target_index)))
    return best_index * frame_size / frame_rate


_SIGNAL_ALIGNMENT_WORD_WEIGHTS = {
    "a": 0.38,
    "i": 0.78,
    "the": 0.42,
    "to": 0.52,
    "cat": 0.42,
    "dog": 0.95,
    "ran": 1.09,
    "fast": 1.26,
    "sat": 1.29,
    "she": 1.34,
    "had": 1.04,
    "red": 0.53,
    "hat": 0.94,
    "love": 0.94,
    "play": 1.24,
    "outside": 1.79,
}


def signal_word_timestamps(sentence: str, audio_bytes: bytes) -> dict[str, tuple[float, float]]:
    """Estimate word timestamps with local waveform valleys and lightweight duration priors."""
    words = sentence_tts_words(sentence)
    if not words:
        return {}
    if len(words) == 1:
        return {words[0]: (0.0, wav_duration_seconds(audio_bytes))}

    frame_rate, samples = _read_mono_pcm16_samples(audio_bytes)
    if not samples:
        return {}
    duration = len(samples) / frame_rate
    rms_values, frame_size = _short_time_rms(samples, frame_rate)
    if not rms_values or max(rms_values) < 1.0:
        raise ValueError("signal alignment requires non-silent audio")

    weights = [_SIGNAL_ALIGNMENT_WORD_WEIGHTS.get(word, max(0.55, len(word) * 0.26)) for word in words]
    total_weight = sum(weights)
    elapsed = 0.0
    boundaries: list[float] = []
    for previous_weight, next_weight in zip(weights, weights[1:]):
        elapsed += previous_weight
        prior_boundary = duration * elapsed / total_weight
        # Use a narrow search so the duration prior remains stable for
        # coarticulated words while still snapping to a nearby low-energy frame.
        search_window = 0.005
        boundary = _local_rms_minimum(rms_values, frame_size, frame_rate, prior_boundary, search_window)
        if boundaries and boundary <= boundaries[-1]:
            boundary = min(duration, boundaries[-1] + 0.001)
        boundaries.append(boundary)

    timestamps: dict[str, tuple[float, float]] = {}
    starts = [0.0, *boundaries]
    ends = [*boundaries, duration]
    for word, start, end in zip(words, starts, ends):
        if end <= start:
            raise ValueError(f"invalid signal timestamp for {word!r}: {start}-{end}")
        timestamps.setdefault(word, (start, end))
    return timestamps

def align_sentence_audio_words(sentence: str, audio_bytes: bytes) -> dict[str, tuple[float, float]]:
    """Align generated sentence audio to target words using local faster-whisper word timestamps.

    The generated sentence is known, so this accepts only sequential timestamp
    matches for every unique target word. Any mismatch, missing timestamp, or
    obviously unusable boundary raises ``ValueError`` so callers can use the
    proportional slicer fallback.
    """
    target_words = sentence_tts_words(sentence)
    if not target_words:
        return {}

    LOGGER.info("Starting word alignment for %d target words", len(target_words))
    audio_path = write_tts_audio_file("alignment_sentence", audio_bytes)
    model = _load_whisper_model()
    segments, _info = model.transcribe(
        audio_path,
        language="en",
        beam_size=1,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    recognized_words: list[tuple[str, float, float]] = []
    for segment in segments:
        for word_info in getattr(segment, "words", []) or []:
            word = clean_tts_word(getattr(word_info, "word", ""))
            start = getattr(word_info, "start", None)
            end = getattr(word_info, "end", None)
            probability = getattr(word_info, "probability", None)
            if not word or start is None or end is None:
                continue
            if probability is not None and float(probability) < 0.2:
                continue
            recognized_words.append((word, float(start), float(end)))

    audio_duration = wav_duration_seconds(audio_bytes)
    timestamps: dict[str, tuple[float, float]] = {}
    search_index = 0
    for target_word in target_words:
        while search_index < len(recognized_words) and recognized_words[search_index][0] != target_word:
            search_index += 1
        if search_index >= len(recognized_words):
            raise ValueError(f"missing aligned word timestamp for {target_word!r}")

        _word, start, end = recognized_words[search_index]
        search_index += 1
        if start < 0 or end <= start or end > audio_duration + WORD_CLIP_PADDING_SECONDS:
            raise ValueError(f"unusable aligned timestamp for {target_word!r}: {start}-{end}")
        timestamps.setdefault(target_word, (start, min(end, audio_duration)))

    LOGGER.info("Word alignment succeeded for %d/%d target words", len(timestamps), len(target_words))
    return timestamps


def slice_sentence_audio_by_timestamps(
    sentence: str, audio_bytes: bytes, word_timestamps: dict[str, tuple[float, float]]
) -> dict[str, bytes]:
    """Slice sentence WAV bytes into word clips from exact timestamp boundaries."""
    clips: dict[str, bytes] = {}
    for word in sentence_tts_words(sentence):
        timestamp = word_timestamps.get(word)
        if timestamp is None:
            raise ValueError(f"missing timestamp for {word!r}")
        clip = _slice_wav_by_seconds(audio_bytes, timestamp[0], timestamp[1])
        if clip is None:
            raise ValueError(f"invalid timestamp for {word!r}")
        clips.setdefault(word, clip)
    return clips


def slice_sentence_audio_with_alignment_or_fallback(
    sentence: str, audio_bytes: bytes, method_report: Optional[dict[str, str]] = None
) -> dict[str, bytes]:
    """Prefer local word alignment for clips, falling back to proportional slicing on failure."""
    try:
        timestamps = signal_word_timestamps(sentence, audio_bytes)
        clips = slice_sentence_audio_by_timestamps(sentence, audio_bytes, timestamps)
        if method_report is not None:
            method_report.update({"method": "signal_alignment", "fallback_reason": ""})
        LOGGER.info("Using signal-aligned word clips for sentence %r", sentence)
        return clips
    except Exception as signal_exc:
        LOGGER.warning("Signal word alignment failed for sentence %r: %s", sentence, signal_exc)

    try:
        timestamps = align_sentence_audio_words(sentence, audio_bytes)
        clips = slice_sentence_audio_by_timestamps(sentence, audio_bytes, timestamps)
        if method_report is not None:
            method_report.update({"method": "alignment", "fallback_reason": ""})
        LOGGER.info("Using whisper-aligned word clips for sentence %r", sentence)
        return clips
    except Exception as exc:
        if method_report is not None:
            method_report.update({"method": "proportional_fallback", "fallback_reason": str(exc)})
        LOGGER.warning("Using proportional word-clip fallback for sentence %r: %s", sentence, exc)
        return slice_sentence_audio_by_words(sentence, audio_bytes)


def slice_sentence_audio_by_words(sentence: str, audio_bytes: bytes) -> dict[str, bytes]:
    """Approximate word clips by splitting sentence audio by word character weight."""
    timestamps = proportional_word_timestamps(sentence, audio_bytes)
    return slice_sentence_audio_by_timestamps(sentence, audio_bytes, timestamps)


def proportional_word_timestamps(sentence: str, audio_bytes: bytes) -> dict[str, tuple[float, float]]:
    """Approximate word timestamps by splitting sentence audio by word character weight."""
    words = sentence_tts_words(sentence)
    if not words:
        return {}

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        total_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

    if total_frames <= 0:
        return {}

    weights = [max(len(word), 1) for word in words]
    total_weight = sum(weights)

    timestamps: dict[str, tuple[float, float]] = {}
    elapsed_weight = 0
    for word, weight in zip(words, weights):
        start_frame = int(total_frames * elapsed_weight / total_weight)
        elapsed_weight += weight
        end_frame = int(total_frames * elapsed_weight / total_weight)
        if end_frame > start_frame:
            timestamps.setdefault(word, (start_frame / frame_rate, end_frame / frame_rate))

    return timestamps


def _initialize_prewarm_status(sentence: str, words: list[str]) -> None:
    with TTS_CACHE_LOCK:
        TTS_MEMORY_CACHE.clear()
        sentence_key = normalize_text(sentence)
        ready_words = [word for word in words if (sentence_key, word) in TTS_MEMORY_CACHE]
        TTS_PREWARM_STATUS.update(
            {
                "sentence": sentence,
                "total": len(words),
                "ready": len(ready_words),
                "failed": 0,
                "running": len(ready_words) < len(words),
                "ready_words": ready_words,
                "clip_method": "cache" if ready_words else "",
                "fallback_reason": "",
            }
        )


def prewarm_level_words(sentence: str, engine_mode: str) -> None:
    """Generate sentence TTS once, slice approximate per-word clips, and cache them."""
    words = sentence_tts_words(sentence)
    _initialize_prewarm_status(sentence, words)

    with TTS_CACHE_LOCK:
        sentence_key = normalize_text(sentence)
        missing_words = [word for word in words if (sentence_key, word) not in TTS_MEMORY_CACHE]
    if not missing_words:
        with TTS_CACHE_LOCK:
            if TTS_PREWARM_STATUS.get("sentence") == sentence:
                TTS_PREWARM_STATUS["running"] = False
        return

    if engine_mode == TURBO_ENGINE and not modal_credentials_available():
        with TTS_CACHE_LOCK:
            if TTS_PREWARM_STATUS.get("sentence") == sentence:
                TTS_PREWARM_STATUS["failed"] = int(TTS_PREWARM_STATUS.get("failed", 0)) + len(missing_words)
                TTS_PREWARM_STATUS["running"] = False
                TTS_PREWARM_STATUS["fallback_reason"] = "Modal credentials are not configured in the Space."
        LOGGER.info("Skipping Modal word voice prewarm because Modal credentials are not configured")
        return

    if engine_mode == LOCAL_ENGINE:
        word_clips, fallback_reason = slice_local_curriculum_word_clips(sentence)
        if word_clips:
            with TTS_CACHE_LOCK:
                if TTS_PREWARM_STATUS.get("sentence") == sentence:
                    TTS_PREWARM_STATUS["clip_method"] = "curriculum_labels"
                    TTS_PREWARM_STATUS["fallback_reason"] = ""
        elif not LOCAL_LIVE_TTS_ENABLED:
            with TTS_CACHE_LOCK:
                if TTS_PREWARM_STATUS.get("sentence") == sentence:
                    TTS_PREWARM_STATUS["failed"] = int(TTS_PREWARM_STATUS.get("failed", 0)) + len(missing_words)
                    TTS_PREWARM_STATUS["running"] = False
                    TTS_PREWARM_STATUS["fallback_reason"] = fallback_reason
            LOGGER.info("Local curriculum word clips unavailable; browser word help will be used: %s", fallback_reason)
            return
        else:
            try:
                sentence_audio = synthesize_speech_bytes(sentence, engine_mode)
                method_report: dict[str, str] = {}
                word_clips = slice_sentence_audio_with_alignment_or_fallback(sentence, sentence_audio, method_report)
                with TTS_CACHE_LOCK:
                    if TTS_PREWARM_STATUS.get("sentence") == sentence:
                        TTS_PREWARM_STATUS["clip_method"] = method_report.get("method", "")
                        TTS_PREWARM_STATUS["fallback_reason"] = method_report.get("fallback_reason", "")
            except Exception as exc:
                LOGGER.exception("Word voice prewarm failed for sentence %r", sentence)
                with TTS_CACHE_LOCK:
                    if TTS_PREWARM_STATUS.get("sentence") == sentence:
                        TTS_PREWARM_STATUS["failed"] = int(TTS_PREWARM_STATUS.get("failed", 0)) + len(missing_words)
                        TTS_PREWARM_STATUS["running"] = False
                        TTS_PREWARM_STATUS["fallback_reason"] = str(exc)
                return
    else:
        try:
            sentence_audio = synthesize_speech_bytes(sentence, engine_mode)
            method_report: dict[str, str] = {}
            word_clips = slice_sentence_audio_with_alignment_or_fallback(sentence, sentence_audio, method_report)
            with TTS_CACHE_LOCK:
                if TTS_PREWARM_STATUS.get("sentence") == sentence:
                    TTS_PREWARM_STATUS["clip_method"] = method_report.get("method", "")
                    TTS_PREWARM_STATUS["fallback_reason"] = method_report.get("fallback_reason", "")
        except Exception as exc:
            LOGGER.exception("Word voice prewarm failed for sentence %r", sentence)
            with TTS_CACHE_LOCK:
                if TTS_PREWARM_STATUS.get("sentence") == sentence:
                    TTS_PREWARM_STATUS["failed"] = int(TTS_PREWARM_STATUS.get("failed", 0)) + len(missing_words)
                    TTS_PREWARM_STATUS["running"] = False
                    TTS_PREWARM_STATUS["fallback_reason"] = str(exc)
            return

    for word in missing_words:
        audio_bytes = word_clips.get(word)
        if audio_bytes is None:
            with TTS_CACHE_LOCK:
                if TTS_PREWARM_STATUS.get("sentence") == sentence:
                    TTS_PREWARM_STATUS["failed"] = int(TTS_PREWARM_STATUS.get("failed", 0)) + 1
            continue
        with TTS_CACHE_LOCK:
            if TTS_PREWARM_STATUS.get("sentence") == sentence:
                TTS_MEMORY_CACHE.setdefault(tts_cache_key(sentence, word), audio_bytes)
                ready_words = list(TTS_PREWARM_STATUS.get("ready_words", []))
                if word not in ready_words:
                    ready_words.append(word)
                TTS_PREWARM_STATUS["ready_words"] = ready_words
                TTS_PREWARM_STATUS["ready"] = len(ready_words)

    with TTS_CACHE_LOCK:
        if TTS_PREWARM_STATUS.get("sentence") == sentence:
            TTS_PREWARM_STATUS["running"] = False


def start_prewarm_level_words(sentence: str, engine_mode: str) -> None:
    threading.Thread(target=prewarm_level_words, args=(sentence, engine_mode), daemon=True).start()


def start_word_voice_prewarm(sentence: str, engine_mode: str = DEFAULT_INFERENCE_ENGINE) -> tuple[str, str]:
    words = sentence_tts_words(sentence)
    _initialize_prewarm_status(sentence, words)
    if words:
        start_prewarm_level_words(sentence, engine_mode)
    return current_tts_status()


def current_tts_status() -> tuple[str, str]:
    with TTS_CACHE_LOCK:
        status = dict(TTS_PREWARM_STATUS)
        sentence = str(status.get("sentence", ""))
        ready_audio = {
            word: f"data:audio/wav;base64,{base64.b64encode(TTS_MEMORY_CACHE[tts_cache_key(sentence, word)]).decode('ascii')}"
            for word in status.get("ready_words", [])
            if tts_cache_key(sentence, word) in TTS_MEMORY_CACHE
        }

    return render_tts_status(status), json.dumps(ready_audio)


def ensure_current_level_prewarm(
    current_index: int, inference_engine: str = DEFAULT_INFERENCE_ENGINE, prewarm_started: bool = False
) -> tuple[bool, str, str]:
    """Start word voice prewarm once after the UI has had a chance to render."""
    if prewarm_started:
        tts_status, ready_words = current_tts_status()
        return True, tts_status, ready_words

    sentence = CURRICULUM[int(current_index) % len(CURRICULUM)]
    tts_status, ready_words = start_word_voice_prewarm(sentence, inference_engine)
    return True, tts_status, ready_words


def engine_button_classes(engine_mode: str, button_engine: str) -> list[str]:
    classes = ["engine-choice", "engine-choice-turbo" if button_engine == TURBO_ENGINE else "engine-choice-local"]
    classes.append("engine-choice-selected" if engine_mode == button_engine else "engine-choice-muted")
    return classes


def select_inference_engine(engine_mode: str, current_index: int) -> tuple[str, dict[str, object], dict[str, object], str, str]:
    """Update the active inference engine without doing heavy work during the click."""
    del current_index
    tts_status, ready_words = current_tts_status()
    return (
        engine_mode,
        gr.update(elem_classes=engine_button_classes(engine_mode, TURBO_ENGINE)),
        gr.update(elem_classes=engine_button_classes(engine_mode, LOCAL_ENGINE)),
        tts_status,
        ready_words,
    )


def select_turbo_engine(current_index: int) -> tuple[str, dict[str, object], dict[str, object], str, str]:
    return select_inference_engine(TURBO_ENGINE, current_index)


def select_local_engine(current_index: int) -> tuple[str, dict[str, object], dict[str, object], str, str]:
    return select_inference_engine(LOCAL_ENGINE, current_index)


def render_tts_status(status: dict[str, object]) -> str:
    total = int(status.get("total", 0))
    if total == 0:
        return '<div class="voice-status voice-status-hidden"></div>'

    ready = int(status.get("ready", 0))
    failed = int(status.get("failed", 0))
    running = bool(status.get("running", False))
    clip_method = str(status.get("clip_method", ""))
    fallback_reason = html.escape(str(status.get("fallback_reason", "")), quote=True)
    method_label = ""
    if clip_method == "alignment":
        method_label = "fast word help"
    elif clip_method == "proportional_fallback":
        method_label = "slower word help"
    elif clip_method == "signal_alignment":
        method_label = "fast word help"
    elif clip_method == "curriculum_labels":
        method_label = "instant local word help"

    if ready >= total:
        title = f' title="{fallback_reason}"' if fallback_reason else ""
        return (
            f'<div class="voice-status voice-status-ready"{title}>'
            f'<span class="voice-status-legacy">Word voices ready{" (proportional fallback)" if clip_method == "proportional_fallback" else ""}</span>'
            f'✨ {method_label or "word help ready"}</div>'
        )
    if running:
        return (
            f'<div class="voice-status voice-status-loading">'
            f'<span class="voice-status-legacy">Getting word voices ready... {ready}/{total}</span>'
            f'✨ preparing word help {ready}/{total}</div>'
        )
    if failed:
        title = f' title="{fallback_reason}"' if fallback_reason else ""
        return f'<div class="voice-status voice-status-loading"{title}>✨ browser word help ready</div>'
    return (
        f'<div class="voice-status voice-status-loading">'
        f'<span class="voice-status-legacy">Getting word voices ready... {ready}/{total}</span>'
        f'✨ preparing word help {ready}/{total}</div>'
    )


def ask_minicpm_judge(target_text: str, transcript: str, inference_engine: str = TURBO_ENGINE) -> bool:
    """Ask the fine-tuned MiniCPM evaluator whether the reading is acceptable."""
    try:
        if inference_engine == LOCAL_ENGINE:
            return local_ask_minicpm_judge(target_text, transcript)

        verdict = str(run_minicpm_evaluator(target_text, transcript)).strip().casefold()
    except Exception:
        return False
    return verdict == "true"


def render_reading_canvas(sentence: str) -> str:
    """Render target text as clickable HTML spans, not a Gradio textbox."""
    spans: list[str] = []
    for raw_word in sentence.split():
        clean_word = raw_word.strip(".,!?;:\"'")
        escaped_display = html.escape(raw_word)
        escaped_word = html.escape(clean_word, quote=True)
        spans.append(
            f'<span class="clickable-word" role="button" tabindex="0" '
            f'aria-label="Hear the word {escaped_word}" '
            f'onclick="readAlongSpeakWord(\'{escaped_word}\')" '
            f'onkeydown="if(event.key === \'Enter\' || event.key === \' \') {{ event.preventDefault(); readAlongSpeakWord(\'{escaped_word}\'); }}">'
            f"{escaped_display}</span>"
        )

    return f"""
    <section class="reading-card" aria-label="Reading sentence">
        <div class="reading-helper">Tap a word if you need help ✨</div>
        <div class="reading-sentence">{' '.join(spans)}</div>
    </section>
    """


def loading_feedback() -> str:
    return """
    <div class="feedback-card feedback-loading" aria-live="polite">
        <div class="spinner-star">⭐</div>
        <div>Listening to your reading...</div>
    </div>
    """


def hidden_feedback() -> str:
    return '<div class="feedback-card feedback-hidden" aria-live="polite"></div>'


def success_feedback() -> str:
    return """
    <div class="feedback-card feedback-success" aria-live="polite">
        <div class="star-row"><span>🌟</span><span>🎉</span><span>🌟</span></div>
        <div class="feedback-title">Amazing reading!</div>
        <div class="feedback-subtitle">You read the sentence perfectly.</div>
    </div>
    """


def retry_feedback() -> str:
    return """
    <div class="feedback-card feedback-retry" aria-live="polite">
        <div class="feedback-title">Nice try!</div>
        <div class="feedback-subtitle">Try pressing record again!</div>
    </div>
    """


def _call_with_engine(function, *args, inference_engine: str):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, inference_engine)

    accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
    if accepts_varargs or len(signature.parameters) > len(args):
        return function(*args, inference_engine)
    return function(*args)


def evaluate_reading(
    audio_filepath: Optional[str], current_index: int, inference_engine: str = TURBO_ENGINE
) -> tuple[str, Optional[str], str]:
    """Evaluate one read attempt against the active curriculum sentence."""
    if not audio_filepath:
        return hidden_feedback(), None, ""

    transcript = _call_with_engine(transcribe_audio, audio_filepath, inference_engine=inference_engine)
    target_sentence = CURRICULUM[int(current_index) % len(CURRICULUM)]
    print(
        f"[read-along] engine={inference_engine!r} target={target_sentence!r} transcript={transcript!r}",
        flush=True,
    )

    if transcript == "[ASR_ERROR]":
        return retry_feedback(), None, ""

    exact_match = normalize_text(transcript) == normalize_text(target_sentence)
    if exact_match or _call_with_engine(ask_minicpm_judge, target_sentence, transcript, inference_engine=inference_engine):
        return success_feedback(), None, "SUCCESS"

    return retry_feedback(), None, ""


def prewarm_current_level(current_index: int, inference_engine: str = TURBO_ENGINE) -> tuple[str, str]:
    sentence = CURRICULUM[int(current_index) % len(CURRICULUM)]
    return start_word_voice_prewarm(sentence, inference_engine)


def next_sentence(idx: int, inference_engine: str = TURBO_ENGINE) -> tuple[int, str, None, str, None, None, str, str, str]:
    """Advance to the next curriculum sentence and clear transient outputs."""
    next_index = (int(idx) + 1) % len(CURRICULUM)
    next_level_sentence = CURRICULUM[next_index]
    tts_status, ready_words = start_word_voice_prewarm(next_level_sentence, inference_engine)
    return next_index, render_reading_canvas(next_level_sentence), None, hidden_feedback(), None, None, tts_status, ready_words, ""


def listen_to_sentence(current_index: int, inference_engine: str = TURBO_ENGINE) -> Optional[str]:
    sentence = CURRICULUM[int(current_index) % len(CURRICULUM)]
    if inference_engine == LOCAL_ENGINE:
        audio_path = local_curriculum_audio_path(sentence)
        if audio_path is not None:
            return str(audio_path)
        if not LOCAL_LIVE_TTS_ENABLED:
            return None
    return synthesize_speech(sentence, inference_engine)


def update_audio_help(
    clicked_word: str, inference_engine: str = TURBO_ENGINE, sentence: Optional[str] = None
) -> Optional[bytes]:
    """Return cached realistic word audio if pre-generation has finished.

    Word clicks must not block on local VoxCPM generation. If the audio is not
    cached yet, the browser speech-synthesis fallback handles the click.
    """
    word = clean_tts_word(clicked_word or "")
    if not word:
        return None

    with TTS_CACHE_LOCK:
        active_sentence = sentence if sentence is not None else str(TTS_PREWARM_STATUS.get("sentence", ""))
        cached_audio = TTS_MEMORY_CACHE.get(tts_cache_key(active_sentence, word))
    if cached_audio is not None:
        return cached_audio

    return None


def finish_word_click(
    clicked_word: str, inference_engine: str = TURBO_ENGINE, sentence: Optional[str] = None
) -> tuple[Optional[str], gr.update]:
    word = clean_tts_word(clicked_word or "")
    audio_bytes = update_audio_help(word, inference_engine, sentence)
    audio_path = write_tts_audio_file(word, audio_bytes) if audio_bytes is not None else None
    return audio_path, gr.update(value=word or "Word helper")


def listen_to_word(word: str, inference_engine: str = TURBO_ENGINE, sentence: Optional[str] = None) -> Optional[bytes]:
    """Backward-compatible alias for word-level audio help."""
    return update_audio_help(word, inference_engine, sentence)


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
CUSTOM_CSS = (ASSETS_DIR / "read_along.css").read_text()
CONFETTI_SCRIPT = "<script src='https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js'></script>"
FRONTEND_JS = f"<script>{(ASSETS_DIR / 'read_along.js').read_text()}</script>"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Read-Along AI") as demo:
        current_index = gr.State(0)
        prewarm_started = gr.State(False)
        prewarm_timer = gr.Timer(1)

        with gr.Column(elem_classes="main-container"):
            gr.HTML('<h1 class="app-title">Read-Along AI</h1>')
            with gr.Column(elem_classes="hero-panel"):
                with gr.Row(elem_classes="top-toolbar"):
                    with gr.Column(elem_classes="engine-panel"):
                        gr.HTML('<div class="engine-title">Inference Engine</div>')
                        with gr.Row(elem_classes="engine-buttons"):
                            turbo_engine_button = gr.Button(
                                "⚡ Turbo Mode (Modal)",
                                elem_classes=engine_button_classes(DEFAULT_INFERENCE_ENGINE, TURBO_ENGINE),
                                elem_id="turbo-engine-button",
                            )
                            local_engine_button = gr.Button(
                                "🏕️ Off the Grid Mode (Local)",
                                elem_classes=engine_button_classes(DEFAULT_INFERENCE_ENGINE, LOCAL_ENGINE),
                                elem_id="local-engine-button",
                            )
                    inference_engine = gr.Radio(
                        choices=INFERENCE_ENGINES,
                        value=DEFAULT_INFERENCE_ENGINE,
                        label="Inference Engine",
                        visible=False,
                        elem_id="inference-engine-state",
                    )
                    tts_status_display = gr.HTML(render_tts_status(dict(TTS_PREWARM_STATUS)), elem_id="tts-status-display")
                reading_canvas = gr.HTML(render_reading_canvas(CURRICULUM[0]))

            with gr.Column(elem_classes="interaction-zone"):
                with gr.Row(elem_classes="action-row"):
                    next_button = gr.Button(
                        "Next Level ➡️",
                        elem_classes="control-button",
                        elem_id="next-word-button",
                        variant="secondary",
                    )
                    listen_button = gr.Button("🔊 Listen to Sentence", elem_classes="control-button", variant="primary")
                microphone = gr.Audio(
                    label="🎙️ Press and read out loud",
                    sources=["microphone"],
                    type="filepath",
                    elem_id="mic-capture",
                )

            feedback_display = gr.HTML(hidden_feedback(), elem_id="feedback-display")
            speech_output = gr.Audio(
                label="Read-Along voice",
                autoplay=True,
                visible="hidden",
                elem_id="speech-output",
            )
            word_help_output = gr.Audio(
                label="Word helper voice",
                autoplay=True,
                visible="hidden",
                elem_id="word-help-output",
            )

            tts_ready_audio = gr.Textbox(value="{}", visible="hidden", elem_id="tts-ready-audio")
            success_trigger = gr.Textbox(value="", visible=False, elem_id="success-trigger")

        microphone.stop_recording(
            fn=loading_feedback,
            inputs=None,
            outputs=feedback_display,
            show_progress="hidden",
        ).then(
            fn=evaluate_reading,
            inputs=[microphone, current_index, inference_engine],
            outputs=[feedback_display, speech_output, success_trigger],
        )

        turbo_engine_button.click(
            fn=select_turbo_engine,
            inputs=current_index,
            outputs=[inference_engine, turbo_engine_button, local_engine_button, tts_status_display, tts_ready_audio],
            show_progress="hidden",
        )

        local_engine_button.click(
            fn=select_local_engine,
            inputs=current_index,
            outputs=[inference_engine, turbo_engine_button, local_engine_button, tts_status_display, tts_ready_audio],
            show_progress="hidden",
        )

        next_button.click(
            fn=next_sentence,
            inputs=[current_index, inference_engine],
            outputs=[
                current_index,
                reading_canvas,
                microphone,
                feedback_display,
                speech_output,
                word_help_output,
                tts_status_display,
                tts_ready_audio,
                success_trigger,
            ],
        )

        success_trigger.change(
            fn=None,
            inputs=[success_trigger],
            outputs=None,
            js="""(val) => {
                if (val === 'SUCCESS' && typeof confetti === 'function') {
                    confetti({ particleCount: 200, spread: 90, origin: { y: 0.6 } });
                }
            }""",
        )

        listen_button.click(
            fn=listen_to_sentence,
            inputs=[current_index, inference_engine],
            outputs=speech_output,
        )

        demo.load(
            fn=current_tts_status,
            inputs=None,
            outputs=[tts_status_display, tts_ready_audio],
            show_progress="hidden",
        )

        prewarm_timer.tick(
            fn=ensure_current_level_prewarm,
            inputs=[current_index, inference_engine, prewarm_started],
            outputs=[prewarm_started, tts_status_display, tts_ready_audio],
            show_progress="hidden",
        )
    return demo


if __name__ == "__main__":
    build_app().launch(css=CUSTOM_CSS, head=CONFETTI_SCRIPT + FRONTEND_JS, footer_links=[])
