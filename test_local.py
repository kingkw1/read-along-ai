"""Manual smoke checks for heavyweight local model backends.

These checks download and load model weights, so they are opt-in instead of
running during the normal pytest suite. Set RUN_LOCAL_MODEL_SMOKE=1 to execute.
"""

from __future__ import annotations

import os
import time

import pytest

from local_inference import (
    local_ask_minicpm_judge,
    local_synthesize_speech,
    local_transcribe_audio,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_MODEL_SMOKE") != "1",
    reason="set RUN_LOCAL_MODEL_SMOKE=1 to run heavyweight local model smoke checks",
)

TEST_AUDIO = "data/processed_audio/01_scientist.wav"
TEST_TARGET = "scientist"


def test_local_model_smoke() -> None:
    print("--- 1. Testing Faster-Whisper (ASR) ---")
    start = time.time()
    transcript = local_transcribe_audio(TEST_AUDIO)
    print(f"Transcript: '{transcript}' (Took {time.time() - start:.2f}s)\n")

    print("--- 2. Testing llama.cpp (MiniCPM Judge) ---")
    start = time.time()
    verdict = local_ask_minicpm_judge(TEST_TARGET, "scientists")
    print(
        f"Verdict for '{TEST_TARGET}' vs 'scientists': "
        f"{verdict} (Took {time.time() - start:.2f}s)\n"
    )

    print("--- 3. Testing VoxCPM (TTS) ---")
    start = time.time()
    audio_path = local_synthesize_speech("You did a great job!")
    print(f"Generated audio at {audio_path}. (Took {time.time() - start:.2f}s)\n")
