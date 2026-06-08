"""Backend API contract tests for the deployed Modal endpoints."""

from __future__ import annotations

import io
import wave

import modal


MODAL_APP_NAME = "read-along-ai-inference"
SAMPLE_RATE = 16_000
DUMMY_AUDIO_SECONDS = 1


def _modal_function(function_name: str):
    lookup = getattr(modal.Function, "lookup", None)
    if lookup is not None:
        return lookup(MODAL_APP_NAME, function_name)
    return modal.Function.from_name(MODAL_APP_NAME, function_name)


def _silent_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    frame_count = SAMPLE_RATE * DUMMY_AUDIO_SECONDS

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * frame_count)

    return buffer.getvalue()


def test_cohere_asr() -> None:
    audio_bytes = _silent_wav_bytes()

    result = _modal_function("run_cohere_asr").remote(audio_bytes)

    assert isinstance(result, dict)
    assert "text" in result


def test_voxcpm_tts() -> None:
    audio_bytes = _modal_function("run_voxcpm_tts").remote("Hello")

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0
