"""Lightweight word-boundary detection for comma-paused curriculum audio."""

from __future__ import annotations

import io
import re
import wave
from array import array


def normalize_word_text(text: str) -> str:
    """Normalize text to the same word keys used by the TTS cache."""
    return re.sub(r"\s+", " ", "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())).strip()


def sentence_tts_words(sentence: str) -> list[str]:
    """Return unique cleaned words in reading order for TTS prewarming."""
    words: list[str] = []
    seen: set[str] = set()
    for raw_word in sentence.split():
        word = normalize_word_text(raw_word)
        if word and word not in seen:
            words.append(word)
            seen.add(word)
    return words


def pcm16_mono_samples(frames: bytes, channels: int, sample_width: int) -> array:
    """Return mono 16-bit-ish samples from a PCM WAV frame buffer."""
    if sample_width != 2:
        return array("h")

    samples = array("h")
    samples.frombytes(frames)
    if channels <= 1:
        return samples

    mono = array("h")
    for index in range(0, len(samples), channels):
        mono.append(int(sum(samples[index : index + channels]) / channels))
    return mono


def manual_gap_boundaries_from_audio(
    frames: bytes,
    *,
    channels: int,
    sample_width: int,
    frame_rate: int,
    needed_boundaries: int,
) -> list[int]:
    """Find word-boundary frame offsets by placing cuts inside detected silence gaps.

    VoxCPM reads comma-separated curriculum prompts with short pauses at commas.
    This lightweight detector measures local RMS energy, extracts sufficiently long
    low-energy islands, and returns the midpoint of the strongest internal gaps.
    """
    if needed_boundaries <= 0 or frame_rate <= 0:
        return []

    samples = pcm16_mono_samples(frames, channels, sample_width)
    if not samples:
        return []

    window = max(1, int(frame_rate * 0.01))
    energies: list[float] = []
    positions: list[int] = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            continue
        energies.append(sum(abs(sample) for sample in chunk) / len(chunk))
        positions.append(start)

    if not energies:
        return []

    peak = max(energies)
    if peak <= 0:
        return []
    threshold = max(peak * 0.08, 120.0)
    min_gap_windows = max(2, int(0.035 / 0.01))
    edge_margin = int(frame_rate * 0.08)

    gaps: list[tuple[float, int, int]] = []
    gap_start: int | None = None
    for index, energy in enumerate(energies + [peak + 1]):
        silent = index < len(energies) and energy <= threshold
        if silent and gap_start is None:
            gap_start = index
        elif not silent and gap_start is not None:
            gap_end = index
            if gap_end - gap_start >= min_gap_windows:
                start_frame = positions[gap_start]
                end_frame = min(len(samples), positions[gap_end - 1] + window)
                if start_frame > edge_margin and end_frame < len(samples) - edge_margin:
                    mean_energy = sum(energies[gap_start:gap_end]) / (gap_end - gap_start)
                    score = (end_frame - start_frame) / max(mean_energy, 1.0)
                    gaps.append((score, start_frame, end_frame))
            gap_start = None

    chosen = sorted(sorted(gaps, reverse=True)[:needed_boundaries], key=lambda gap: gap[1])
    return [(start + end) // 2 for _score, start, end in chosen]


def word_boundary_timestamps(sentence: str, audio_bytes: bytes) -> list[tuple[str, float, float]]:
    """Return per-word timestamps, preferring comma-pause silence boundaries."""
    words = sentence_tts_words(sentence)
    if not words:
        return []

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        params = wav_file.getparams()
        total_frames = wav_file.getnframes()
        frames = wav_file.readframes(total_frames)

    if total_frames <= 0:
        return []

    boundaries = manual_gap_boundaries_from_audio(
        frames,
        channels=params.nchannels,
        sample_width=params.sampwidth,
        frame_rate=params.framerate,
        needed_boundaries=len(words) - 1,
    )
    if len(boundaries) != len(words) - 1:
        weights = [max(len(word), 1) for word in words]
        total_weight = sum(weights)
        elapsed_weight = 0
        boundaries = []
        for weight in weights[:-1]:
            elapsed_weight += weight
            boundaries.append(int(total_frames * elapsed_weight / total_weight))

    edges = [0, *boundaries, total_frames]
    return [
        (word, edges[index] / params.framerate, edges[index + 1] / params.framerate)
        for index, word in enumerate(words)
    ]
