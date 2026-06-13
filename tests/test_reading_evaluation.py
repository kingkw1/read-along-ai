"""Unit tests for MiniCPM-backed reading evaluation."""

from __future__ import annotations

import io
import json
from pathlib import Path
import wave

import app
import pytest


@pytest.fixture(autouse=True)
def clear_tts_memory_cache() -> None:
    app.TTS_MEMORY_CACHE.clear()
    app.TTS_PREWARM_STATUS.update(
        {
            "sentence": "",
            "total": 0,
            "ready": 0,
            "failed": 0,
            "running": False,
            "ready_words": [],
            "clip_method": "",
            "fallback_reason": "",
        }
    )


def test_evaluate_reading_accepts_exact_match_without_judge(monkeypatch) -> None:
    judge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "the cat sat")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: pytest.fail("successful reading should not trigger TTS"))
    monkeypatch.setattr(
        app,
        "ask_minicpm_judge",
        lambda target, transcript: judge_calls.append((target, transcript)) or False,
    )

    feedback, praise_audio, success_trigger = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Amazing reading!" in feedback
    assert praise_audio is None
    assert success_trigger == "SUCCESS"
    assert judge_calls == []


def test_evaluate_reading_accepts_minicpm_true_verdict(monkeypatch) -> None:
    judge_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "dog ran fass")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: pytest.fail("successful reading should not trigger TTS"))

    def fake_judge(target: str, transcript: str) -> bool:
        judge_calls.append((target, transcript))
        return True

    monkeypatch.setattr(app, "ask_minicpm_judge", fake_judge)

    feedback, praise_audio, success_trigger = app.evaluate_reading("/tmp/audio.wav", 1)

    assert "Amazing reading!" in feedback
    assert praise_audio is None
    assert success_trigger == "SUCCESS"
    assert judge_calls == [("The dog ran fast.", "dog ran fass")]


def test_evaluate_reading_retries_minicpm_false_verdict(monkeypatch) -> None:
    monkeypatch.setattr(app, "transcribe_audio", lambda _path: "banana")
    monkeypatch.setattr(app, "synthesize_speech", lambda _text: "/tmp/praise.wav")
    monkeypatch.setattr(app, "ask_minicpm_judge", lambda _target, _transcript: False)

    feedback, praise_audio, success_trigger = app.evaluate_reading("/tmp/audio.wav", 0)

    assert "Nice try!" in feedback
    assert praise_audio is None
    assert success_trigger == ""


def test_next_sentence_cycles_curriculum_clears_outputs_and_starts_word_prewarm(monkeypatch) -> None:
    monkeypatch.setattr(app, "start_prewarm_level_words", lambda _sentence, _engine: None)

    (
        next_index,
        sentence_html,
        microphone,
        feedback,
        speech_output,
        word_help_output,
        tts_status,
        ready_audio,
        success_trigger,
    ) = app.next_sentence(3)

    assert next_index == 0
    assert "The" in sentence_html
    assert "cat" in sentence_html
    assert "feedback-hidden" in feedback
    assert microphone is None
    assert speech_output is None
    assert word_help_output is None
    assert "Getting word voices ready... 0/3" in tts_status
    assert json.loads(ready_audio) == {}
    assert success_trigger == ""


def test_reading_canvas_uses_browser_tts_for_word_clicks() -> None:
    sentence_html = app.render_reading_canvas("The cat sat.")

    assert "readAlongSpeakWord('The')" in sentence_html
    assert "readAlongSpeakWord('cat')" in sentence_html
    assert "readAlongSpeakWord('sat')" in sentence_html
    assert "readAlongSendWord" not in sentence_html


def test_frontend_cached_voice_uses_audio_data_urls_with_browser_fallback() -> None:
    assert "#tts-ready-audio" in app.FRONTEND_JS
    assert "new Audio(audioUrl)" in app.FRONTEND_JS
    assert "window.speechSynthesis.cancel()" in app.FRONTEND_JS
    assert "audio.play().catch(() => window.readAlongSpeakWithBrowser" in app.FRONTEND_JS
    assert "word-click-submit" not in app.FRONTEND_JS


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


def test_app_defaults_to_turbo_mode() -> None:
    demo = app.build_app()
    engine_radios = [
        block
        for block in demo.blocks.values()
        if block.__class__.__name__ == "Radio" and block.label == "Inference Engine"
    ]

    assert len(engine_radios) == 1
    assert engine_radios[0].value == app.TURBO_ENGINE


def test_format_text_for_tts_pads_single_words_only() -> None:
    assert app.format_text_for_tts("cat") == "Cat."
    assert app.format_text_for_tts("fast") == "Fast."
    assert app.format_text_for_tts("Cat") == "Cat."
    assert app.format_text_for_tts("The dog ran fast.") == "The dog ran fast."


def test_synthesize_speech_bytes_formats_text_before_voxcpm(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(app, "run_voxcpm_tts", lambda text: calls.append(text) or b"audio")

    assert app.synthesize_speech_bytes("cat", app.TURBO_ENGINE) == b"audio"
    assert calls == ["Cat."]


def silent_wav_bytes(seconds: float = 1.2) -> bytes:
    buffer = io.BytesIO()
    frame_count = int(app.SAMPLE_RATE * seconds)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(app.SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def test_slice_sentence_audio_by_words_returns_valid_weighted_word_clips() -> None:
    clips = app.slice_sentence_audio_by_words("The cat sat.", silent_wav_bytes(seconds=1.2))

    assert list(clips) == ["the", "cat", "sat"]
    for audio_bytes in clips.values():
        assert audio_bytes.startswith(b"RIFF")
        assert app.wav_duration_seconds(audio_bytes) > 0


def test_slice_sentence_audio_by_timestamps_returns_valid_wav_clips() -> None:
    clips = app.slice_sentence_audio_by_timestamps(
        "The cat sat.",
        silent_wav_bytes(seconds=1.2),
        {"the": (0.0, 0.25), "cat": (0.35, 0.65), "sat": (0.75, 1.1)},
    )

    assert list(clips) == ["the", "cat", "sat"]
    assert app.wav_duration_seconds(clips["the"]) > 0.25
    assert app.wav_duration_seconds(clips["cat"]) > 0.30
    assert all(audio_bytes.startswith(b"RIFF") for audio_bytes in clips.values())


def test_alignment_slicer_falls_back_to_proportional_when_alignment_fails(monkeypatch, caplog) -> None:
    audio_bytes = silent_wav_bytes(seconds=1.2)
    fallback_clips = {"the": b"fallback-the"}
    method_report: dict[str, str] = {}

    monkeypatch.setattr(
        app,
        "align_sentence_audio_words",
        lambda _sentence, _audio_bytes: (_ for _ in ()).throw(ValueError("alignment failed")),
    )
    monkeypatch.setattr(app, "slice_sentence_audio_by_words", lambda _sentence, _audio_bytes: fallback_clips)

    with caplog.at_level("WARNING", logger=app.LOGGER.name):
        clips = app.slice_sentence_audio_with_alignment_or_fallback("The cat sat.", audio_bytes, method_report)

    assert clips == fallback_clips
    assert method_report == {"method": "proportional_fallback", "fallback_reason": "alignment failed"}
    assert "Using proportional word-clip fallback" in caplog.text


def test_slice_sentence_audio_by_timestamps_rejects_missing_partial_timestamps() -> None:
    with pytest.raises(ValueError, match="missing timestamp"):
        app.slice_sentence_audio_by_timestamps(
            "The cat sat.",
            silent_wav_bytes(seconds=1.2),
            {"the": (0.0, 0.25), "cat": (0.35, 0.65)},
        )


def test_prewarm_level_words_generates_sentence_once_and_caches_word_clips(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_synthesize_bytes(word: str, engine: str) -> bytes:
        calls.append((word, engine))
        return silent_wav_bytes(seconds=1.2)

    monkeypatch.setattr(app, "synthesize_speech_bytes", fake_synthesize_bytes)
    monkeypatch.setattr(
        app,
        "align_sentence_audio_words",
        lambda _sentence, _audio_bytes: {"the": (0.0, 0.25), "cat": (0.35, 0.65), "sat": (0.75, 1.1)},
    )

    app.prewarm_level_words("The cat sat. The cat!", app.LOCAL_ENGINE)

    assert calls == [("The cat sat. The cat!", app.LOCAL_ENGINE)]
    assert set(app.TTS_MEMORY_CACHE) == {
        app.tts_cache_key("The cat sat. The cat!", "the"),
        app.tts_cache_key("The cat sat. The cat!", "cat"),
        app.tts_cache_key("The cat sat. The cat!", "sat"),
    }
    assert all(audio_bytes.startswith(b"RIFF") for audio_bytes in app.TTS_MEMORY_CACHE.values())
    assert app.TTS_PREWARM_STATUS["ready_words"] == ["the", "cat", "sat"]
    assert app.TTS_PREWARM_STATUS["ready"] == 3
    assert app.TTS_PREWARM_STATUS["failed"] == 0
    assert app.TTS_PREWARM_STATUS["running"] is False
    assert app.TTS_PREWARM_STATUS["clip_method"] == "alignment"
    assert app.TTS_PREWARM_STATUS["fallback_reason"] == ""


def test_prewarm_level_words_does_not_reuse_word_clips_from_previous_sentence(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    old_key = app.tts_cache_key("The cat sat.", "the")
    new_key = app.tts_cache_key("The dog ran fast.", "the")
    app.TTS_MEMORY_CACHE[old_key] = b"old-the-from-cat"

    def fake_synthesize_bytes(sentence: str, engine: str) -> bytes:
        calls.append((sentence, engine))
        return silent_wav_bytes(seconds=1.4)

    monkeypatch.setattr(app, "synthesize_speech_bytes", fake_synthesize_bytes)
    monkeypatch.setattr(
        app,
        "align_sentence_audio_words",
        lambda _sentence, _audio_bytes: {
            "the": (0.0, 0.2),
            "dog": (0.3, 0.55),
            "ran": (0.65, 0.9),
            "fast": (1.0, 1.3),
        },
    )

    app.prewarm_level_words("The dog ran fast.", app.LOCAL_ENGINE)

    assert calls == [("The dog ran fast.", app.LOCAL_ENGINE)]
    assert old_key not in app.TTS_MEMORY_CACHE
    assert new_key in app.TTS_MEMORY_CACHE
    assert app.TTS_MEMORY_CACHE[new_key] != b"old-the-from-cat"


def test_current_tts_status_reports_proportional_fallback_method() -> None:
    app.TTS_PREWARM_STATUS.update(
        {
            "sentence": "The cat sat.",
            "total": 3,
            "ready": 3,
            "failed": 0,
            "running": False,
            "ready_words": ["the", "cat", "sat"],
            "clip_method": "proportional_fallback",
            "fallback_reason": "alignment failed",
        }
    )

    status_html = app.render_tts_status(app.TTS_PREWARM_STATUS)

    assert "Word voices ready" in status_html
    assert "proportional fallback" in status_html
    assert 'title="alignment failed"' in status_html


def test_update_audio_help_returns_cached_bytes_without_generating(monkeypatch) -> None:
    app.TTS_MEMORY_CACHE[app.tts_cache_key("The cat sat.", "cat")] = b"cached-cat"
    app.TTS_PREWARM_STATUS["sentence"] = "The cat sat."
    monkeypatch.setattr(
        app,
        "synthesize_speech_bytes",
        lambda _word, _engine: pytest.fail("cache hit should not generate TTS"),
    )

    assert app.update_audio_help("Cat!", app.TURBO_ENGINE) == b"cached-cat"


def test_update_audio_help_does_not_generate_on_cache_miss(monkeypatch) -> None:
    monkeypatch.setattr(
        app,
        "synthesize_speech_bytes",
        lambda _word, _engine: pytest.fail("word clicks should not generate synchronously"),
    )

    assert app.update_audio_help("Dog.", app.LOCAL_ENGINE) is None
    assert app.TTS_MEMORY_CACHE == {}


def test_update_audio_help_does_not_reuse_same_word_from_previous_sentence() -> None:
    app.TTS_MEMORY_CACHE[app.tts_cache_key("The cat sat.", "the")] = b"old-the-from-cat"
    app.TTS_MEMORY_CACHE[app.tts_cache_key("The dog ran fast.", "the")] = b"new-the-from-dog"
    app.TTS_PREWARM_STATUS["sentence"] = "The dog ran fast."

    assert app.update_audio_help("The", app.LOCAL_ENGINE) == b"new-the-from-dog"
    assert app.update_audio_help("The", app.LOCAL_ENGINE, "The cat sat.") == b"old-the-from-cat"


def test_finish_word_click_returns_playable_wav_path_from_cached_bytes() -> None:
    app.TTS_MEMORY_CACHE[app.tts_cache_key("The cat sat.", "cat")] = b"cached-cat-audio"
    app.TTS_PREWARM_STATUS["sentence"] = "The cat sat."

    audio_path, button_update = app.finish_word_click("Cat!", app.TURBO_ENGINE)

    assert audio_path is not None
    assert audio_path.endswith(".wav")
    assert Path(audio_path).read_bytes() == b"cached-cat-audio"
    assert button_update["value"] == "cat"


def test_current_tts_status_reports_ready_words() -> None:
    app.TTS_MEMORY_CACHE[app.tts_cache_key("The cat sat.", "the")] = b"wav-the"
    app.TTS_MEMORY_CACHE[app.tts_cache_key("The cat sat.", "cat")] = b"wav-cat"
    app.TTS_PREWARM_STATUS.update(
        {
            "sentence": "The cat sat.",
            "total": 3,
            "ready": 2,
            "failed": 0,
            "running": True,
            "ready_words": ["the", "cat"],
        }
    )

    status_html, ready_audio = app.current_tts_status()

    assert "Getting word voices ready... 2/3" in status_html
    parsed_audio = json.loads(ready_audio)
    assert parsed_audio["the"] == "data:audio/wav;base64,d2F2LXRoZQ=="
    assert parsed_audio["cat"] == "data:audio/wav;base64,d2F2LWNhdA=="


def test_ask_minicpm_judge_parses_true_verdict(monkeypatch) -> None:
    monkeypatch.setattr(app, "run_minicpm_evaluator", lambda _target, _transcript: "True")

    assert app.ask_minicpm_judge("cat", "kat")


def test_ask_minicpm_judge_rejects_errors(monkeypatch) -> None:
    def failing_evaluator(_target: str, _transcript: str) -> str:
        raise RuntimeError("modal unavailable")

    monkeypatch.setattr(app, "run_minicpm_evaluator", failing_evaluator)

    assert not app.ask_minicpm_judge("cat", "banana")
