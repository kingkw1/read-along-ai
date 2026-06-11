"""Unit tests for MiniCPM-backed reading evaluation."""

from __future__ import annotations

import app


def test_evaluate_reading_accepts_exact_match_without_judge(monkeypatch) -> None:
    judge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "the dog ran fast")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: "/tmp/praise.wav")
    monkeypatch.setattr(
        app,
        "ask_minicpm_judge",
        lambda target, transcript: judge_calls.append((target, transcript)) or False,
    )

    feedback, praise_audio = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Amazing reading!" in feedback
    assert praise_audio == "/tmp/praise.wav"
    assert judge_calls == []


def test_evaluate_reading_accepts_minicpm_true_verdict(monkeypatch) -> None:
    judge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "dog ran fass")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: "/tmp/praise.wav")

    def fake_judge(target: str, transcript: str) -> bool:
        judge_calls.append((target, transcript))
        return True

    monkeypatch.setattr(app, "ask_minicpm_judge", fake_judge)

    feedback, praise_audio = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Amazing reading!" in feedback
    assert praise_audio == "/tmp/praise.wav"
    assert judge_calls == [("The dog ran fast.", "dog ran fass")]


def test_evaluate_reading_retries_minicpm_false_verdict(monkeypatch) -> None:
    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "banana")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: "/tmp/praise.wav")
    monkeypatch.setattr(app, "ask_minicpm_judge", lambda _target, _transcript: False)

    feedback, praise_audio = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Nice try!" in feedback
    assert praise_audio is None


def test_ask_minicpm_judge_parses_true_verdict(monkeypatch) -> None:
    monkeypatch.setattr(app, "run_minicpm_evaluator", lambda _target, _transcript: "True")

    assert app.ask_minicpm_judge("cat", "kat")


def test_ask_minicpm_judge_rejects_errors(monkeypatch) -> None:
    def failing_evaluator(_target: str, _transcript: str) -> str:
        raise RuntimeError("modal unavailable")

    monkeypatch.setattr(app, "run_minicpm_evaluator", failing_evaluator)

    assert not app.ask_minicpm_judge("cat", "banana")
