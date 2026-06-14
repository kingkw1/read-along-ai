"""Smoke test the local-only inference path used by Off the Grid mode.

Run from the repository root after local dependencies and model assets exist:
    python scripts/manual/local_smoke.py

This intentionally calls local_inference.py directly and does not require
Modal credentials. It is a verification helper for the Hugging Face Space
local mode, not a unit test.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from local_inference import (  # noqa: E402
    _resolve_minicpm_gguf_path,
    local_ask_minicpm_judge,
    local_synthesize_speech,
    local_transcribe_audio,
)


DEFAULT_AUDIO = REPO_ROOT / "data" / "curriculum_audio" / "comma" / "01_thecatsat.wav"
DEFAULT_TARGET = "The cat sat."


def timed(label: str, action):
    print(f"--- {label} ---", flush=True)
    start = time.time()
    result = action()
    print(f"Took {time.time() - start:.2f}s\n", flush=True)
    return result


def assert_wav(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected WAV output does not exist: {path}")
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnframes() <= 0:
            raise RuntimeError(f"WAV output has no frames: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local Read-Along AI inference.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Audio file for faster-whisper ASR.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target text for the MiniCPM judge.")
    parser.add_argument("--tts-text", default=DEFAULT_TARGET, help="Text to synthesize with local VoxCPM.")
    parser.add_argument("--skip-tts", action="store_true", help="Skip local VoxCPM generation.")
    args = parser.parse_args()

    print("Local smoke test for Off the Grid mode")
    print(f"Repository: {REPO_ROOT}")
    print(f"Modal credentials present: {bool(os.environ.get('MODAL_TOKEN_ID') and os.environ.get('MODAL_TOKEN_SECRET'))}")

    gguf_path = _resolve_minicpm_gguf_path()
    if not gguf_path.exists():
        raise FileNotFoundError(f"MiniCPM Q4 GGUF not found at {gguf_path}")
    print(f"MiniCPM GGUF: {gguf_path} ({gguf_path.stat().st_size / (1024 ** 3):.2f} GiB)\n")

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"ASR test audio not found: {audio_path}")

    transcript = timed(
        "1. faster-whisper local ASR",
        lambda: local_transcribe_audio(str(audio_path)),
    )
    print(f"Transcript: {transcript!r}\n")

    verdict = timed(
        "2. llama.cpp MiniCPM local judge",
        lambda: local_ask_minicpm_judge(args.target, transcript),
    )
    print(f"Verdict for target={args.target!r} transcript={transcript!r}: {verdict}\n")

    if not args.skip_tts:
        tts_path = timed(
            "3. VoxCPM local TTS",
            lambda: Path(local_synthesize_speech(args.tts_text)),
        )
        assert_wav(tts_path)
        print(f"Generated audio: {tts_path}\n")

    print("Local smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
