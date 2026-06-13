"""Benchmarks for local word-boundary detection on comma-separated VoxCPM-style audio."""

from __future__ import annotations

import io
import math
import wave
from array import array
from dataclasses import dataclass

import pytest

import word_boundaries


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class BoundaryMetrics:
    boundaries: list[float]
    hit_count: int
    total: int
    hit_rate: float
    mean_abs_error_to_gap_midpoint: float
    max_abs_error_to_gap_midpoint: float


@dataclass(frozen=True)
class SyntheticBoundaryCase:
    name: str
    sentence: str
    word_durations: tuple[float, ...]
    gap_durations: tuple[float, ...]

    @property
    def manual_gaps(self) -> list[tuple[float, float]]:
        elapsed = 0.0
        gaps: list[tuple[float, float]] = []
        for index, duration in enumerate(self.word_durations):
            elapsed += duration
            if index < len(self.gap_durations):
                gap_start = elapsed
                elapsed += self.gap_durations[index]
                gaps.append((gap_start, elapsed))
        return gaps

    @property
    def total_duration(self) -> float:
        return sum(self.word_durations) + sum(self.gap_durations)


def _tone(duration: float, amplitude: int = 3000, frequency: float = 220.0) -> array:
    samples = array("h")
    for index in range(int(SAMPLE_RATE * duration)):
        samples.append(int(amplitude * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)))
    return samples


def _silence(duration: float) -> array:
    return array("h", [0] * int(SAMPLE_RATE * duration))


def _wav_bytes(segments: list[array]) -> bytes:
    pcm = array("h")
    for segment in segments:
        pcm.extend(segment)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _case_audio(case: SyntheticBoundaryCase) -> bytes:
    segments: list[array] = []
    for index, duration in enumerate(case.word_durations):
        segments.append(_tone(duration, frequency=220.0 + 40.0 * index))
        if index < len(case.gap_durations):
            segments.append(_silence(case.gap_durations[index]))
    return _wav_bytes(segments)


def _proportional_boundaries(sentence: str, total_duration: float) -> list[float]:
    words = word_boundaries.sentence_tts_words(sentence)
    weights = [max(len(word), 1) for word in words]
    total_weight = sum(weights)
    elapsed_weight = 0
    boundaries: list[float] = []
    for weight in weights[:-1]:
        elapsed_weight += weight
        boundaries.append(total_duration * elapsed_weight / total_weight)
    return boundaries


def _metrics(boundaries: list[float], manual_gaps: list[tuple[float, float]]) -> BoundaryMetrics:
    assert len(boundaries) == len(manual_gaps)
    errors: list[float] = []
    hit_count = 0
    for boundary, (gap_start, gap_end) in zip(boundaries, manual_gaps):
        midpoint = (gap_start + gap_end) / 2
        errors.append(abs(boundary - midpoint))
        if gap_start <= boundary <= gap_end:
            hit_count += 1

    return BoundaryMetrics(
        boundaries=boundaries,
        hit_count=hit_count,
        total=len(manual_gaps),
        hit_rate=hit_count / len(manual_gaps),
        mean_abs_error_to_gap_midpoint=sum(errors) / len(errors),
        max_abs_error_to_gap_midpoint=max(errors),
    )


CASES = (
    SyntheticBoundaryCase(
        name="equal_text_equal_audio",
        sentence="cat, dog, red, hat",
        word_durations=(0.22, 0.24, 0.24, 0.24),
        gap_durations=(0.12, 0.14, 0.14),
    ),
    SyntheticBoundaryCase(
        name="uneven_text_equal_audio",
        sentence="a, elephant, ox, rhinoceros",
        word_durations=(0.22, 0.24, 0.24, 0.24),
        gap_durations=(0.12, 0.14, 0.14),
    ),
    SyntheticBoundaryCase(
        name="uneven_audio_even_text",
        sentence="blue, green, white, black",
        word_durations=(0.14, 0.33, 0.18, 0.29),
        gap_durations=(0.10, 0.16, 0.12),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_candidate_splitter_lands_every_internal_boundary_in_manual_gap(
    case: SyntheticBoundaryCase,
) -> None:
    audio_bytes = _case_audio(case)

    stamps = word_boundaries.word_boundary_timestamps(case.sentence, audio_bytes)
    internal_boundaries = [end for _word, _start, end in stamps[:-1]]
    current_metrics = _metrics(internal_boundaries, case.manual_gaps)

    assert current_metrics.hit_count == current_metrics.total


def test_silence_detector_outperforms_proportional_splitter_on_benchmark_suite() -> None:
    current_totals = BoundaryMetrics([], 0, 0, 0.0, 0.0, 0.0)
    proportional_totals = BoundaryMetrics([], 0, 0, 0.0, 0.0, 0.0)
    current_errors: list[float] = []
    proportional_errors: list[float] = []

    for case in CASES:
        audio_bytes = _case_audio(case)
        stamps = word_boundaries.word_boundary_timestamps(case.sentence, audio_bytes)
        current_boundaries = [end for _word, _start, end in stamps[:-1]]
        proportional_boundaries = _proportional_boundaries(case.sentence, case.total_duration)

        current = _metrics(current_boundaries, case.manual_gaps)
        proportional = _metrics(proportional_boundaries, case.manual_gaps)

        current_totals = BoundaryMetrics(
            [],
            current_totals.hit_count + current.hit_count,
            current_totals.total + current.total,
            0.0,
            0.0,
            0.0,
        )
        proportional_totals = BoundaryMetrics(
            [],
            proportional_totals.hit_count + proportional.hit_count,
            proportional_totals.total + proportional.total,
            0.0,
            0.0,
            0.0,
        )

        for boundary, gap in zip(current_boundaries, case.manual_gaps):
            current_errors.append(abs(boundary - ((gap[0] + gap[1]) / 2)))
        for boundary, gap in zip(proportional_boundaries, case.manual_gaps):
            proportional_errors.append(abs(boundary - ((gap[0] + gap[1]) / 2)))

    current_hit_rate = current_totals.hit_count / current_totals.total
    proportional_hit_rate = proportional_totals.hit_count / proportional_totals.total
    current_mean_error = sum(current_errors) / len(current_errors)
    proportional_mean_error = sum(proportional_errors) / len(proportional_errors)

    assert current_totals.hit_count == 9
    assert current_hit_rate == pytest.approx(1.0)
    assert current_mean_error == pytest.approx(0.0, abs=0.005)

    assert proportional_totals.hit_count == 5
    assert proportional_hit_rate == pytest.approx(5 / 9)
    assert proportional_mean_error > current_mean_error
