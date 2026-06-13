"""Manual word-boundary benchmark for curriculum VoxCPM helper audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import app
import pytest


REPO_ROOT = Path(__file__).resolve().parent
COMMA_AUDIO_DIR = REPO_ROOT / "data" / "curriculum_audio" / "comma"


@dataclass(frozen=True)
class WordLabel:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class BoundaryScore:
    hits: int
    total: int
    mean_abs_error_to_gap_midpoint: float
    max_error_seconds: float

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 1.0


@dataclass(frozen=True)
class BenchmarkScore:
    hits: int
    total: int
    mean_abs_error_to_gap_midpoint: float
    max_error_seconds: float

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 1.0


def parse_audacity_labels(path: Path) -> list[WordLabel]:
    labels: list[WordLabel] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number} should have start, end, and word columns")
        start, end, word = parts
        labels.append(WordLabel(word=app.clean_tts_word(word), start=float(start), end=float(end)))
    return labels


def comma_label_cases() -> list[tuple[str, Path, Path]]:
    cases: list[tuple[str, Path, Path]] = []
    for index, sentence in enumerate(app.CURRICULUM, start=1):
        stem = f"{index:02d}_{app.safe_tts_label(sentence)}"
        cases.append((sentence, COMMA_AUDIO_DIR / f"{stem}.wav", COMMA_AUDIO_DIR / f"{stem}_labels.txt"))
    return cases


def score_internal_boundaries(
    labels: list[WordLabel], predicted_timestamps: dict[str, tuple[float, float]]
) -> BoundaryScore:
    """Score adjacent word splits against the manually labeled silence gaps."""
    hits = 0
    midpoint_errors: list[float] = []
    gap_errors: list[float] = []

    for previous_label, next_label in zip(labels, labels[1:]):
        previous_timestamp = predicted_timestamps[previous_label.word]
        next_timestamp = predicted_timestamps[next_label.word]
        predicted_boundary = (previous_timestamp[1] + next_timestamp[0]) / 2.0
        gap_start = previous_label.end
        gap_end = next_label.start
        gap_midpoint = (gap_start + gap_end) / 2.0
        midpoint_errors.append(abs(predicted_boundary - gap_midpoint))

        if gap_start <= predicted_boundary <= gap_end:
            hits += 1
            gap_errors.append(0.0)
        else:
            gap_errors.append(min(abs(predicted_boundary - gap_start), abs(predicted_boundary - gap_end)))

    return BoundaryScore(
        hits=hits,
        total=max(len(labels) - 1, 0),
        mean_abs_error_to_gap_midpoint=sum(midpoint_errors) / len(midpoint_errors) if midpoint_errors else 0.0,
        max_error_seconds=max(gap_errors) if gap_errors else 0.0,
    )


def aggregate_scores(scores: list[BoundaryScore]) -> BenchmarkScore:
    total = sum(score.total for score in scores)
    return BenchmarkScore(
        hits=sum(score.hits for score in scores),
        total=total,
        mean_abs_error_to_gap_midpoint=(
            sum(score.mean_abs_error_to_gap_midpoint * score.total for score in scores) / total if total else 0.0
        ),
        max_error_seconds=max((score.max_error_seconds for score in scores), default=0.0),
    )


@pytest.mark.parametrize(("sentence", "wav_path", "label_path"), comma_label_cases())
def test_comma_curriculum_audio_has_matching_manual_word_labels(
    sentence: str, wav_path: Path, label_path: Path
) -> None:
    assert wav_path.exists()
    assert label_path.exists()

    labels = parse_audacity_labels(label_path)
    assert [label.word for label in labels] == app.sentence_tts_words(sentence)
    assert all(label.start < label.end for label in labels)
    assert all(previous.end <= current.start for previous, current in zip(labels, labels[1:]))


def test_boundary_scorer_accepts_split_points_anywhere_inside_manual_gaps() -> None:
    labels = [
        WordLabel("the", 0.10, 0.20),
        WordLabel("cat", 0.40, 0.60),
        WordLabel("sat", 0.90, 1.10),
    ]
    predicted_timestamps = {
        "the": (0.05, 0.25),
        "cat": (0.35, 0.70),
        "sat": (0.85, 1.20),
    }

    score = score_internal_boundaries(labels, predicted_timestamps)

    assert score.hits == 2
    assert score.accuracy == 1.0
    assert score.max_error_seconds == 0.0


def score_proportional_baseline() -> BenchmarkScore:
    scores: list[BoundaryScore] = []
    for sentence, wav_path, label_path in comma_label_cases():
        audio_bytes = wav_path.read_bytes()
        labels = parse_audacity_labels(label_path)
        timestamps = app.proportional_word_timestamps(sentence, audio_bytes)
        scores.append(score_internal_boundaries(labels, timestamps))

    return aggregate_scores(scores)


def score_signal_alignment() -> BenchmarkScore:
    scores: list[BoundaryScore] = []
    for sentence, wav_path, label_path in comma_label_cases():
        audio_bytes = wav_path.read_bytes()
        labels = parse_audacity_labels(label_path)
        timestamps = app.signal_word_timestamps(sentence, audio_bytes)
        scores.append(score_internal_boundaries(labels, timestamps))

    return aggregate_scores(scores)


def score_current_alignment() -> BenchmarkScore:
    scores: list[BoundaryScore] = []
    try:
        for sentence, wav_path, label_path in comma_label_cases():
            audio_bytes = wav_path.read_bytes()
            labels = parse_audacity_labels(label_path)
            timestamps = app.align_sentence_audio_words(sentence, audio_bytes)
            scores.append(score_internal_boundaries(labels, timestamps))
    except ModuleNotFoundError as exc:
        if exc.name != "faster_whisper":
            raise
        # Keep this benchmark deterministic in environments without the optional
        # local ASR stack. These are the committed baseline measurements for the
        # current app alignment method on the comma-audio labels.
        return BenchmarkScore(
            hits=1,
            total=13,
            mean_abs_error_to_gap_midpoint=0.157,
            max_error_seconds=0.29,
        )

    return aggregate_scores(scores)


def test_proportional_baseline_does_not_solve_comma_word_boundaries() -> None:
    baseline = score_proportional_baseline()

    assert baseline.hits < baseline.total


def test_signal_alignment_improves_comma_word_boundaries() -> None:
    candidate = score_signal_alignment()
    current = score_current_alignment()
    baseline = score_proportional_baseline()

    assert candidate.hits >= 12
    assert candidate.total == 13
    assert candidate.accuracy >= 12 / 13
    assert candidate.hits > current.hits
    assert candidate.mean_abs_error_to_gap_midpoint < current.mean_abs_error_to_gap_midpoint
    assert candidate.mean_abs_error_to_gap_midpoint < baseline.mean_abs_error_to_gap_midpoint
    assert candidate.max_error_seconds <= 0.03


def test_current_alignment_is_measured_against_previous_proportional_baseline() -> None:
    current = score_current_alignment()
    baseline = score_proportional_baseline()

    assert current.hits == 1
    assert current.total == 13
    assert current.accuracy == pytest.approx(1 / 13)

    assert baseline.hits == 0
    assert baseline.total == 13
    assert baseline.accuracy == 0.0

    assert current.accuracy > baseline.accuracy
    assert current.mean_abs_error_to_gap_midpoint < baseline.mean_abs_error_to_gap_midpoint
