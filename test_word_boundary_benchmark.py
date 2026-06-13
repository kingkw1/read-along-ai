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
    max_error = 0.0

    for previous_label, next_label in zip(labels, labels[1:]):
        previous_timestamp = predicted_timestamps[previous_label.word]
        next_timestamp = predicted_timestamps[next_label.word]
        predicted_boundary = (previous_timestamp[1] + next_timestamp[0]) / 2.0
        gap_start = previous_label.end
        gap_end = next_label.start

        if gap_start <= predicted_boundary <= gap_end:
            hits += 1
            continue

        error = min(abs(predicted_boundary - gap_start), abs(predicted_boundary - gap_end))
        max_error = max(max_error, error)

    return BoundaryScore(hits=hits, total=max(len(labels) - 1, 0), max_error_seconds=max_error)


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


def test_proportional_baseline_does_not_solve_comma_word_boundaries() -> None:
    scores: list[BoundaryScore] = []
    for sentence, wav_path, label_path in comma_label_cases():
        audio_bytes = wav_path.read_bytes()
        labels = parse_audacity_labels(label_path)
        timestamps = app.proportional_word_timestamps(sentence, audio_bytes)
        scores.append(score_internal_boundaries(labels, timestamps))

    hits = sum(score.hits for score in scores)
    total = sum(score.total for score in scores)

    assert hits < total


@pytest.mark.xfail(reason="Replace proportional timestamps with the candidate edge detector under test.")
def test_candidate_splitter_lands_every_internal_boundary_in_manual_gap() -> None:
    scores: list[BoundaryScore] = []
    for sentence, wav_path, label_path in comma_label_cases():
        audio_bytes = wav_path.read_bytes()
        labels = parse_audacity_labels(label_path)
        candidate_timestamps = app.proportional_word_timestamps(sentence, audio_bytes)
        scores.append(score_internal_boundaries(labels, candidate_timestamps))

    assert all(score.accuracy == 1.0 for score in scores)
