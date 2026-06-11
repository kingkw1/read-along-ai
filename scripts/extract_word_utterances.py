#!/usr/bin/env python3
"""Extract a labeled single-word dataset from a continuous reading recording.

The script is intentionally conservative:
1. Convert the source audio to mono 16 kHz WAV with ffmpeg.
2. Find speech-like candidate regions with adaptive RMS thresholding.
3. Optionally transcribe each candidate with Whisper/faster-whisper.
4. Select one candidate per target word in list order and export WAV files.

When no transcriber is installed, it falls back to sequential candidate export.
That mode is useful for a first pass, but Whisper mode is much better when the
recording includes corrections, wrong words, or between-word conversation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TargetWord:
    index: int
    word: str
    slug: str


@dataclass(frozen=True)
class Segment:
    index: int
    start_sec: float
    end_sec: float
    transcript: str = ""

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class Match:
    target: TargetWord
    segment: Segment | None
    score: float
    mode: str
    warning: str


def parse_word_list(path: Path) -> list[TargetWord]:
    words: list[TargetWord] = []
    pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        number = int(match.group(1))
        word = match.group(2).strip()
        slug = slugify(word)
        words.append(TargetWord(index=number, word=word, slug=slug))
    if not words:
        raise ValueError(f"No numbered words found in {path}")
    return words


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "word"


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def convert_to_work_wav(input_audio: Path, wav_path: Path, sample_rate: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    subprocess.run(command, check=True)


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected 16-bit mono WAV after conversion")
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def detect_segments(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float | None,
    frame_ms: int,
    hop_ms: int,
    min_duration_sec: float,
    merge_gap_sec: float,
    pad_sec: float,
) -> list[Segment]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    hop_size = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < frame_size:
        return []

    starts = np.arange(0, len(samples) - frame_size + 1, hop_size)
    rms = np.empty(len(starts), dtype=np.float32)
    for idx, start in enumerate(starts):
        frame = samples[start : start + frame_size]
        rms[idx] = math.sqrt(float(np.mean(frame * frame)) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)

    if threshold_db is None:
        noise_db = float(np.percentile(db, 20))
        mid_db = float(np.percentile(db, 60))
        high_db = float(np.percentile(db, 92))
        threshold_db = max(noise_db + 8.0, mid_db + 3.0)
        threshold_db = min(threshold_db, high_db - 3.0)

    active = db > threshold_db
    raw: list[tuple[int, int]] = []
    start_frame: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_frame is None:
            start_frame = idx
        is_last = idx == len(active) - 1
        if start_frame is not None and ((not is_active) or is_last):
            end_frame = idx + 1 if is_last and is_active else idx
            start_sample = int(starts[start_frame])
            end_sample = int(starts[end_frame - 1] + frame_size)
            if (end_sample - start_sample) / sample_rate >= min_duration_sec:
                raw.append((start_sample, end_sample))
            start_frame = None

    merge_gap_samples = int(merge_gap_sec * sample_rate)
    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    pad_samples = int(pad_sec * sample_rate)
    segments = []
    for idx, (start, end) in enumerate(merged, start=1):
        padded_start = max(0, start - pad_samples)
        padded_end = min(len(samples), end + pad_samples)
        segments.append(
            Segment(
                index=idx,
                start_sec=padded_start / sample_rate,
                end_sec=padded_end / sample_rate,
            )
        )
    return segments


def export_segment(source_wav: Path, segment: Segment, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{segment.start_sec:.3f}",
        "-to",
        f"{segment.end_sec:.3f}",
        "-i",
        str(source_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def get_transcriber(name: str, model_name: str, device: str, compute_type: str):
    if name in {"auto", "faster-whisper"}:
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel(model_name, device=device, compute_type=compute_type)

            def transcribe(path: Path) -> str:
                segments, _ = model.transcribe(
                    str(path),
                    beam_size=5,
                    language="en",
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                return " ".join(part.text.strip() for part in segments).strip()

            return transcribe, "faster-whisper"
        except Exception as exc:
            if name == "faster-whisper":
                raise RuntimeError(f"Could not load faster-whisper: {exc}") from exc

    if name in {"auto", "openai-whisper"}:
        try:
            import whisper  # type: ignore

            model = whisper.load_model(model_name)

            def transcribe(path: Path) -> str:
                result = model.transcribe(
                    str(path),
                    language="en",
                    fp16=False,
                    condition_on_previous_text=False,
                )
                return str(result.get("text", "")).strip()

            return transcribe, "openai-whisper"
        except Exception as exc:
            if name == "openai-whisper":
                raise RuntimeError(f"Could not load openai-whisper: {exc}") from exc

    return None, "none"


def score_candidate(target: str, transcript: str) -> float:
    target_norm = normalize_text(target)
    transcript_norm = normalize_text(transcript)
    if not transcript_norm:
        return 0.0
    tokens = transcript_norm.split()
    exact = 1.0 if target_norm in tokens else 0.0
    token_score = max(
        (SequenceMatcher(None, target_norm, token).ratio() for token in tokens),
        default=0.0,
    )
    phrase_score = SequenceMatcher(None, target_norm, transcript_norm).ratio()
    substring = 0.92 if target_norm in transcript_norm else 0.0
    return max(exact, token_score, phrase_score, substring)


def choose_matches(
    targets: list[TargetWord],
    segments: list[Segment],
    use_transcripts: bool,
    min_match_score: float,
    lookahead: int,
    fallback_unmatched: bool,
) -> list[Match]:
    matches: list[Match] = []
    cursor = 0

    for target in targets:
        if not use_transcripts:
            segment = segments[cursor] if cursor < len(segments) else None
            cursor += 1
            warning = "sequential fallback; verify by ear"
            if segment is None:
                warning = "missing segment"
            matches.append(Match(target, segment, 0.0, "sequential", warning))
            continue

        search_end = min(len(segments), cursor + lookahead)
        best_idx: int | None = None
        best_score = -1.0
        for idx in range(cursor, search_end):
            score = score_candidate(target.word, segments[idx].transcript)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None or best_score < min_match_score:
            warning = (
                f"low/no transcript match in candidates {cursor + 1}-{search_end}; "
                f"best_score={max(best_score, 0.0):.3f}"
            )
            if fallback_unmatched and cursor < len(segments):
                segment = segments[cursor]
                cursor += 1
                matches.append(
                    Match(
                        target,
                        segment,
                        max(best_score, 0.0),
                        "hybrid-fallback",
                        warning + "; exported next chronological candidate",
                    )
                )
            else:
                matches.append(Match(target, None, max(best_score, 0.0), "transcript", warning))
            continue

        skipped = best_idx - cursor
        segment = segments[best_idx]
        cursor = best_idx + 1
        warning = ""
        if skipped:
            warning = f"skipped {skipped} earlier candidate(s)"
        if best_score < 0.82:
            warning = (warning + "; " if warning else "") + "weak transcript match"
        matches.append(Match(target, segment, best_score, "transcript", warning))

    return matches


def write_manifest(path: Path, matches: list[Match], output_dir: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "word_index",
                "word",
                "output_file",
                "segment_index",
                "start_sec",
                "end_sec",
                "duration_sec",
                "transcript",
                "score",
                "mode",
                "warning",
            ],
        )
        writer.writeheader()
        for match in matches:
            segment = match.segment
            filename = f"{match.target.index:02d}_{match.target.slug}.wav"
            writer.writerow(
                {
                    "word_index": match.target.index,
                    "word": match.target.word,
                    "output_file": filename if segment else "",
                    "segment_index": segment.index if segment else "",
                    "start_sec": f"{segment.start_sec:.3f}" if segment else "",
                    "end_sec": f"{segment.end_sec:.3f}" if segment else "",
                    "duration_sec": f"{segment.duration_sec:.3f}" if segment else "",
                    "transcript": segment.transcript if segment else "",
                    "score": f"{match.score:.3f}",
                    "mode": match.mode,
                    "warning": match.warning,
                }
            )


def write_candidates_manifest(path: Path, segments: list[Segment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["segment_index", "start_sec", "end_sec", "duration_sec", "transcript"],
        )
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "segment_index": segment.index,
                    "start_sec": f"{segment.start_sec:.3f}",
                    "end_sec": f"{segment.end_sec:.3f}",
                    "duration_sec": f"{segment.duration_sec:.3f}",
                    "transcript": segment.transcript,
                }
            )


def remove_previous_outputs(output_dir: Path, targets: list[TargetWord]) -> None:
    for target in targets:
        path = output_dir / f"{target.index:02d}_{target.slug}.wav"
        if path.exists():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw_audio/jane.ogg"))
    parser.add_argument("--word-list", type=Path, default=Path("data/word_list.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_audio/jane_words"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--threshold-db", type=float, default=None)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--hop-ms", type=int, default=10)
    parser.add_argument("--min-duration-sec", type=float, default=0.10)
    parser.add_argument("--merge-gap-sec", type=float, default=0.55)
    parser.add_argument("--pad-sec", type=float, default=0.18)
    parser.add_argument("--transcriber", choices=["auto", "none", "faster-whisper", "openai-whisper"], default="auto")
    parser.add_argument("--model", default="base.en", help="Whisper model name, e.g. tiny.en, base.en, small.en")
    parser.add_argument("--device", default="cpu", help="faster-whisper device: cpu, cuda, or auto")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type, e.g. int8, float32, float16")
    parser.add_argument("--min-match-score", type=float, default=0.72)
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove previous NN_word.wav outputs before exporting")
    parser.add_argument(
        "--no-fallback-unmatched",
        action="store_true",
        help="Leave low-confidence transcript matches unexported instead of using the next chronological segment",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = parse_word_list(args.word_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean and not args.dry_run:
        remove_previous_outputs(args.output_dir, targets)

    with tempfile.TemporaryDirectory(prefix="word_extract_") as tmp:
        tmp_dir = Path(tmp)
        work_wav = tmp_dir / "source_16k_mono.wav"
        convert_to_work_wav(args.input, work_wav, args.sample_rate)
        samples, sample_rate = read_wav_mono(work_wav)
        segments = detect_segments(
            samples=samples,
            sample_rate=sample_rate,
            threshold_db=args.threshold_db,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            min_duration_sec=args.min_duration_sec,
            merge_gap_sec=args.merge_gap_sec,
            pad_sec=args.pad_sec,
        )
        if not segments:
            raise RuntimeError("No speech candidates found. Try lowering --threshold-db.")

        candidate_dir = args.output_dir / "_candidates"
        if args.keep_candidates:
            candidate_dir.mkdir(parents=True, exist_ok=True)

        transcribe = None
        transcriber_mode = "none"
        if args.transcriber != "none":
            transcribe, transcriber_mode = get_transcriber(
                args.transcriber,
                args.model,
                args.device,
                args.compute_type,
            )
            if transcribe is None:
                print(
                    "No Whisper transcriber found; using sequential fallback. "
                    "Install faster-whisper or openai-whisper for robust correction handling.",
                    file=sys.stderr,
                )

        updated_segments: list[Segment] = []
        for segment in segments:
            candidate_path = candidate_dir / f"candidate_{segment.index:03d}.wav"
            temp_candidate = candidate_path if args.keep_candidates else tmp_dir / f"candidate_{segment.index:03d}.wav"
            export_segment(work_wav, segment, temp_candidate)
            transcript = transcribe(temp_candidate) if transcribe else ""
            updated_segments.append(
                Segment(
                    index=segment.index,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    transcript=transcript,
                )
            )

        matches = choose_matches(
            targets=targets,
            segments=updated_segments,
            use_transcripts=transcribe is not None,
            min_match_score=args.min_match_score,
            lookahead=args.lookahead,
            fallback_unmatched=not args.no_fallback_unmatched,
        )

        if not args.dry_run:
            for match in matches:
                if match.segment is None:
                    continue
                output_path = args.output_dir / f"{match.target.index:02d}_{match.target.slug}.wav"
                export_segment(work_wav, match.segment, output_path)

        write_manifest(args.output_dir / "manifest.csv", matches, args.output_dir)
        write_candidates_manifest(args.output_dir / "candidates_manifest.csv", updated_segments)

    exported = sum(1 for match in matches if match.segment is not None)
    warnings = sum(1 for match in matches if match.warning)
    print(f"Parsed {len(targets)} target words.")
    print(f"Detected {len(segments)} speech candidates.")
    print(f"Transcriber: {transcriber_mode}.")
    print(f"Exported {exported}/{len(targets)} labeled WAV files to {args.output_dir}.")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    if warnings:
        print(f"{warnings} row(s) have warnings; review manifest.csv and listen manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
