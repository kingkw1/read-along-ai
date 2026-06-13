"""Manual smoke test for the local-only inference path.

Run from the repository root after local model assets are available:
    python scripts/manual/local_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from local_inference import (
    local_ask_minicpm_judge,
    local_synthesize_speech,
    local_transcribe_audio,
)

# Replace this with the path to one of your real audio files
TEST_AUDIO = "data/processed_audio/01_scientist.wav" 
TEST_TARGET = "scientist"

print("--- 1. Testing Faster-Whisper (ASR) ---")
start = time.time()
transcript = local_transcribe_audio(TEST_AUDIO)
print(f"Transcript: '{transcript}' (Took {time.time() - start:.2f}s)\n")

print("--- 2. Testing llama.cpp (MiniCPM Judge) ---")
# Let's test a deliberate failure scenario to see if the GGUF model reasons correctly
start = time.time()
verdict = local_ask_minicpm_judge(TEST_TARGET, "scientists") 
print(f"Verdict for '{TEST_TARGET}' vs 'scientists': {verdict} (Took {time.time() - start:.2f}s)\n")

print("--- 3. Testing VoxCPM (TTS) ---")
start = time.time()
audio_path = local_synthesize_speech("You did a great job!")
print(f"Generated audio at {audio_path!r}. (Took {time.time() - start:.2f}s)\n")

print("All local models loaded and executed successfully.")
