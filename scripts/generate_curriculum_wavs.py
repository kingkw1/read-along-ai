"""Generate curriculum WAV variants for manual word-boundary annotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app


OUTPUT_DIR = REPO_ROOT / "data" / "curriculum_audio"


def words_as_sentence(sentence: str, separator: str, ending: str = ".") -> str:
    words = [word[:1].upper() + word[1:] if index == 0 else word for index, word in enumerate(app.sentence_tts_words(sentence))]
    return separator.join(words) + ending


def curriculum_variants(sentence: str) -> dict[str, str]:
    return {
        "natural": sentence,
        "comma": words_as_sentence(sentence, ", ", "."),
        "period": words_as_sentence(sentence, ". ", "."),
        "spaced_comma": words_as_sentence(sentence, " , ", "."),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for index, sentence in enumerate(app.CURRICULUM):
        words = app.sentence_tts_words(sentence)
        for variant_name, tts_text in curriculum_variants(sentence).items():
            variant_dir = OUTPUT_DIR / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{index + 1:02d}_{app.safe_tts_label(sentence)}.wav"
            output_path = variant_dir / filename
            audio_bytes = app.synthesize_speech_bytes(tts_text, app.LOCAL_ENGINE)
            output_path.write_bytes(audio_bytes)
            manifest.append(
                {
                    "index": index,
                    "variant": variant_name,
                    "sentence": sentence,
                    "tts_text": tts_text,
                    "wav": str(output_path.relative_to(REPO_ROOT)),
                    "words": words,
                    "manual_timestamps": {},
                }
            )
            print(f"Wrote {output_path}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
