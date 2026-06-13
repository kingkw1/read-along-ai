"""Unit tests for MiniCPM-backed reading evaluation."""

from __future__ import annotations

from pathlib import Path

import app
import pytest


@pytest.fixture(autouse=True)
def clear_tts_memory_cache() -> None:
    app.TTS_MEMORY_CACHE.clear()


def test_evaluate_reading_accepts_exact_match_without_judge(monkeypatch) -> None:
    judge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "the cat sat")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: pytest.fail("successful reading should not trigger TTS"))
    monkeypatch.setattr(
        app,
        "ask_minicpm_judge",
        lambda target, transcript: judge_calls.append((target, transcript)) or False,
    )

    feedback, praise_audio = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Amazing reading!" in feedback
    assert praise_audio is None
    assert judge_calls == []


def test_evaluate_reading_accepts_minicpm_true_verdict(monkeypatch) -> None:
    judge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "dog ran fass")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: pytest.fail("successful reading should not trigger TTS"))

    def fake_judge(target: str, transcript: str) -> bool:
        judge_calls.append((target, transcript))
        return True

    monkeypatch.setattr(app, "ask_minicpm_judge", fake_judge)

    feedback, praise_audio = app.evaluate_reading("/tmp/audio.wav", 1)

    assert "Amazing reading!" in feedback
    assert praise_audio is None
    assert judge_calls == [("The dog ran fast.", "dog ran fass")]


def test_evaluate_reading_retries_minicpm_false_verdict(monkeypatch) -> None:
    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "banana")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: "/tmp/praise.wav")
    monkeypatch.setattr(app, "ask_minicpm_judge", lambda _target, _transcript: False)

    feedback, praise_audio = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Nice try!" in feedback
    assert praise_audio is None


def test_next_sentence_cycles_curriculum_and_clears_outputs() -> None:
    next_index, sentence_html, microphone, feedback, speech_output, word_help_output = app.next_sentence(3)

    assert next_index == 0
    assert "The" in sentence_html
    assert "cat" in sentence_html
    assert "feedback-hidden" in feedback
    assert microphone is None
    assert speech_output is None
    assert word_help_output is None


def test_reading_canvas_uses_browser_tts_for_word_clicks() -> None:
    sentence_html = app.render_reading_canvas("The cat sat.")

    assert "readAlongSpeakWord('The')" in sentence_html
    assert "readAlongSpeakWord('cat')" in sentence_html
    assert "readAlongSpeakWord('sat')" in sentence_html
    assert "readAlongSendWord" not in sentence_html


def test_tts_outputs_are_hidden_but_mounted_for_autoplay() -> None:
    demo = app.build_app()
    audio_outputs = {
        block.label: block
        for block in demo.blocks.values()
        if block.__class__.__name__ == "Audio"
    }

    speech_output = audio_outputs["Read-Along voice"]
    word_help_output = audio_outputs["Word helper voice"]

    assert speech_output.visible == "hidden"
    assert speech_output.autoplay is True
    assert word_help_output.visible == "hidden"
    assert word_help_output.autoplay is True


def test_prewarm_level_words_caches_unique_clean_words_and_suppresses_errors(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_synthesize_bytes(word: str, engine: str) -> bytes:
        calls.append((word, engine))
        if word == "sat":
            raise RuntimeError("tts unavailable")
        return f"audio:{word}".encode()

    monkeypatch.setattr(app, "synthesize_speech_bytes", fake_synthesize_bytes)

    app.prewarm_level_words("The cat sat. The cat!", app.LOCAL_ENGINE)

    assert calls == [("the", app.LOCAL_ENGINE), ("cat", app.LOCAL_ENGINE), ("sat", app.LOCAL_ENGINE)]
    assert app.TTS_MEMORY_CACHE == {
        "the": b"audio:the",
        "cat": b"audio:cat",
    }


def test_update_audio_help_returns_cached_bytes_without_generating(monkeypatch) -> None:
    app.TTS_MEMORY_CACHE["cat"] = b"cached-cat"
    monkeypatch.setattr(
        app,
        "synthesize_speech_bytes",
        lambda _word, _engine: pytest.fail("cache hit should not generate TTS"),
    )

    assert app.update_audio_help("Cat!", app.TURBO_ENGINE) == b"cached-cat"


def test_update_audio_help_generates_and_caches_on_miss(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_synthesize_bytes(word: str, engine: str) -> bytes:
        calls.append((word, engine))
        return b"new-dog-audio"

    monkeypatch.setattr(app, "synthesize_speech_bytes", fake_synthesize_bytes)

    assert app.update_audio_help("Dog.", app.LOCAL_ENGINE) == b"new-dog-audio"
    assert app.TTS_MEMORY_CACHE["dog"] == b"new-dog-audio"
    assert calls == [("dog", app.LOCAL_ENGINE)]


def test_update_audio_help_fails_open_on_generation_error(monkeypatch) -> None:
    def failing_synthesize_bytes(_word: str, _engine: str) -> bytes:
        raise RuntimeError("tts unavailable")

    monkeypatch.setattr(app, "synthesize_speech_bytes", failing_synthesize_bytes)

    assert app.update_audio_help("Dog.", app.LOCAL_ENGINE) is None
    assert app.TTS_MEMORY_CACHE == {}


def test_finish_word_click_returns_playable_wav_path_from_cached_bytes() -> None:
    app.TTS_MEMORY_CACHE["cat"] = b"cached-cat-audio"

    audio_path, button_update = app.finish_word_click("Cat!", app.TURBO_ENGINE)

    assert audio_path is not None
    assert audio_path.endswith(".wav")
    assert Path(audio_path).read_bytes() == b"cached-cat-audio"
    assert button_update["value"] == "cat"


def test_ask_minicpm_judge_parses_true_verdict(monkeypatch) -> None:
    monkeypatch.setattr(app, "run_minicpm_evaluator", lambda _target, _transcript: "True")

    assert app.ask_minicpm_judge("cat", "kat")


def test_ask_minicpm_judge_rejects_errors(monkeypatch) -> None:
    def failing_evaluator(_target: str, _transcript: str) -> str:
        raise RuntimeError("modal unavailable")

    monkeypatch.setattr(app, "run_minicpm_evaluator", failing_evaluator)

    assert not app.ask_minicpm_judge("cat", "banana")
