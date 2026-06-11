#!/usr/bin/env python3
"""Evaluate baseline ASR accuracy on cleaned word audio slices."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import modal


MODAL_APP_NAME = "read-along-ai-inference"
MODAL_FUNCTION_NAME = "run_cohere_asr"
DEFAULT_AUDIO_DIR = Path("data/processed_audio/jane_cleaned")
DEFAULT_OUTPUT_CSV = Path("data/baseline_results.csv")
CSV_HEADERS = ["File", "Target Word", "ASR Transcript", "Strict Match"]


def modal_function(app_name: str, function_name: str):
    lookup = getattr(modal.Function, "lookup", None)
    if lookup is not None:
        return lookup(app_name, function_name)
    return modal.Function.from_name(app_name, function_name)


def target_word_from_filename(path: Path) -> str:
    """Extract the target word from names like 01_scientist.wav."""
    stem = path.stem
    match = re.match(r"^\d+_(.+)$", stem)
    return match.group(1) if match else stem


def transcript_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def strict_match(transcript: str, target_word: str) -> bool:
    return transcript.strip().casefold() == target_word.strip().casefold()


def iter_wav_files(audio_dir: Path) -> list[Path]:
    return sorted(path for path in audio_dir.rglob("*.wav") if path.is_file())


def resolve_audio_dir(audio_dir: Path) -> Path:
    if audio_dir.exists():
        return audio_dir
    typo_compat = {
        Path("data/processed_audio/janed_cleaned"): Path("data/processed_audio/jane_cleaned"),
    }
    replacement = typo_compat.get(audio_dir)
    if replacement and replacement.exists():
        print(f"Audio directory {audio_dir} not found; using {replacement} instead.", flush=True)
        return replacement
    return audio_dir


def completed_files(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        return {
            row["File"]
            for row in csv.DictReader(csv_file)
            if row.get("File")
        }


def write_row(csv_file, writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow(row)
    csv_file.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N files")
    parser.add_argument("--resume", action="store_true", help="Append to an existing CSV and skip files already present")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and targets without calling the Modal ASR endpoint",
    )
    args = parser.parse_args()

    audio_dir = resolve_audio_dir(args.audio_dir)
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}", file=sys.stderr)
        return 1

    wav_files = iter_wav_files(audio_dir)
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    if not wav_files:
        print(f"No .wav files found in {audio_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Audio directory: {audio_dir}")
        for audio_path in wav_files:
            print(f"{audio_path} -> {target_word_from_filename(audio_path)}")
        print(f"Found {len(wav_files)} .wav file(s).")
        return 0

    asr = modal_function(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_files = completed_files(args.output_csv) if args.resume else set()
    if skipped_files:
        wav_files = [path for path in wav_files if str(path) not in skipped_files]

    matches = 0
    total = 0
    mode = "a" if args.resume and args.output_csv.exists() else "w"
    with args.output_csv.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        if mode == "w":
            writer.writeheader()
            csv_file.flush()

        for audio_path in wav_files:
            target_word = target_word_from_filename(audio_path)
            audio_bytes = audio_path.read_bytes()
            print(f"[{total + 1}/{len(wav_files)}] Calling ASR for {audio_path.name}...", flush=True)
            result = asr.remote(audio_bytes)
            transcript = transcript_text(result)
            is_match = strict_match(transcript, target_word)

            total += 1
            matches += int(is_match)
            write_row(
                csv_file,
                writer,
                {
                    "File": str(audio_path),
                    "Target Word": target_word,
                    "ASR Transcript": transcript,
                    "Strict Match": is_match,
                },
            )
            print(
                f"[{total}/{len(wav_files)}] {audio_path.name}: "
                f"target={target_word!r} transcript={transcript!r} match={is_match}",
                flush=True,
            )

    accuracy = (matches / total) * 100 if total else 0.0
    print(f"Baseline accuracy: {accuracy:.2f}% ({matches}/{total})")
    print(f"Results written to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
