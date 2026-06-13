"""Generate full-sentence curriculum WAVs for manual word-boundary annotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app


OUTPUT_DIR = REPO_ROOT / "data" / "curriculum_audio"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for index, sentence in enumerate(app.CURRICULUM):
        filename = f"{index + 1:02d}_{app.safe_tts_label(sentence)}.wav"
        output_path = OUTPUT_DIR / filename
        audio_bytes = app.synthesize_speech_bytes(sentence, app.LOCAL_ENGINE)
        output_path.write_bytes(audio_bytes)
        manifest.append(
            {
                "index": index,
                "sentence": sentence,
                "wav": str(output_path.relative_to(REPO_ROOT)),
                "words": app.sentence_tts_words(sentence),
                "manual_timestamps": {},
            }
        )
        print(f"Wrote {output_path}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
