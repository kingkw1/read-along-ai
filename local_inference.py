"""Local inference backend for Read-Along AI.

These functions mirror the app-level Modal wrappers:
- audio path in, normalized transcript out
- target text in, local WAV path out
- target/transcript in, boolean phonetic verdict out
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
CONDA_ENV_KEYS = {
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "_CE_CONDA",
    "_CE_M",
}
VOXCPM_MODEL_ID = "openbmb/VoxCPM-0.5B"
WHISPER_MODEL_ID = "tiny.en"
SAMPLE_RATE = 16_000
MINICPM_INSTRUCTION = (
    "Determine if the ASR transcript is a valid phonetic match for the target word. "
    "Output only True or False."
)
DEFAULT_MINICPM_GGUF_PATH = REPO_ROOT / "models" / "minicpm-phonetic-evaluator-Q4_K_M.gguf"
CONVERTED_MINICPM_GGUF_PATH = REPO_ROOT / "models" / "gguf" / "minicpm-phonetic-evaluator-q4_k_m.gguf"
DEFAULT_MINICPM_GGUF_REPO_ID = "kingkw1/minicpm-phonetic-evaluator"
DEFAULT_MINICPM_GGUF_FILENAME = "minicpm-phonetic-evaluator-q4_k_m.gguf"

_whisper_model: Any | None = None
_voxcpm_model: Any | None = None
_minicpm_llm: Any | None = None

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _sanitize_runtime_env_before_torch_import() -> None:
    """Restart once without Conda linker paths that break venv Torch imports."""
    if os.environ.get("LOCAL_INFERENCE_ENV_SANITIZED") == "1":
        return

    sanitized = os.environ.copy()
    changed = False
    for key in CONDA_ENV_KEYS | {"LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"}:
        if key in sanitized:
            sanitized.pop(key, None)
            changed = True

    if not changed:
        sanitized["LOCAL_INFERENCE_ENV_SANITIZED"] = "1"
        os.environ["LOCAL_INFERENCE_ENV_SANITIZED"] = "1"
        return

    sanitized["LOCAL_INFERENCE_ENV_SANITIZED"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], sanitized)


_sanitize_runtime_env_before_torch_import()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())).strip()


def _safe_audio_label(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in ("-", "_"))[:24] or "speech"


def _resolve_audio_filepath(audio_filepath: str) -> str:
    audio_path = Path(audio_filepath).expanduser()
    if audio_path.exists():
        return str(audio_path)

    if not audio_path.is_absolute():
        repo_relative = REPO_ROOT / audio_path
        if repo_relative.exists():
            return str(repo_relative)

    processed_audio_dir = REPO_ROOT / "data" / "processed_audio"
    if processed_audio_dir.exists():
        matches = sorted(processed_audio_dir.glob(f"**/{audio_path.name}"))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(f"Audio file not found: {audio_filepath}")


def _load_whisper_model() -> Any:
    global _whisper_model

    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            WHISPER_MODEL_ID,
            device=os.environ.get("LOCAL_WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("LOCAL_WHISPER_COMPUTE_TYPE", "int8"),
        )

    return _whisper_model


def _load_voxcpm_model() -> Any:
    global _voxcpm_model

    if _voxcpm_model is None:
        from voxcpm import VoxCPM

        cache_dir = os.environ.get("LOCAL_MODEL_CACHE_DIR")
        kwargs: dict[str, Any] = {
            "load_denoiser": os.environ.get("LOCAL_VOXCPM_LOAD_DENOISER", "0") == "1",
            "optimize": os.environ.get("LOCAL_VOXCPM_OPTIMIZE", "0") == "1",
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if os.environ.get("LOCAL_VOXCPM_DEVICE"):
            kwargs["device"] = os.environ["LOCAL_VOXCPM_DEVICE"]
        if os.environ.get("LOCAL_VOXCPM_LOCAL_FILES_ONLY", "0") == "1":
            kwargs["local_files_only"] = True
        _voxcpm_model = VoxCPM.from_pretrained(VOXCPM_MODEL_ID, **kwargs)

    return _voxcpm_model


def _resolve_minicpm_gguf_path() -> Path:
    configured = os.environ.get("LOCAL_MINICPM_GGUF_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    if DEFAULT_MINICPM_GGUF_PATH.exists():
        return DEFAULT_MINICPM_GGUF_PATH
    if CONVERTED_MINICPM_GGUF_PATH.exists():
        return CONVERTED_MINICPM_GGUF_PATH

    if os.environ.get("LOCAL_MINICPM_GGUF_LOCAL_ONLY", "0") == "1":
        return CONVERTED_MINICPM_GGUF_PATH

    return _download_minicpm_gguf()


def _download_minicpm_gguf() -> Path:
    """Download the Q4 GGUF from the Hub into the Space cache when absent."""
    from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("LOCAL_MINICPM_GGUF_REPO_ID", DEFAULT_MINICPM_GGUF_REPO_ID)
    filename = os.environ.get("LOCAL_MINICPM_GGUF_FILENAME", DEFAULT_MINICPM_GGUF_FILENAME)
    cache_dir = os.environ.get("LOCAL_MODEL_CACHE_DIR")
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=os.environ.get("LOCAL_MINICPM_GGUF_REPO_TYPE", "model"),
        cache_dir=cache_dir,
        token=token,
        local_files_only=os.environ.get("LOCAL_MINICPM_GGUF_HF_LOCAL_ONLY", "0") == "1",
    )
    return Path(downloaded_path).resolve()


def _load_minicpm_llm() -> Any:
    global _minicpm_llm

    if _minicpm_llm is None:
        from llama_cpp import Llama

        model_path = _resolve_minicpm_gguf_path()
        if not model_path.exists():
            raise FileNotFoundError(
                "MiniCPM GGUF model not found. Expected "
                f"{DEFAULT_MINICPM_GGUF_PATH}, {CONVERTED_MINICPM_GGUF_PATH}, "
                "or a Hub file configured by LOCAL_MINICPM_GGUF_REPO_ID/"
                "LOCAL_MINICPM_GGUF_FILENAME."
            )

        _minicpm_llm = Llama(
            model_path=str(model_path),
            n_ctx=int(os.environ.get("LOCAL_MINICPM_N_CTX", "2048")),
            n_threads=int(os.environ.get("LOCAL_MINICPM_THREADS", str(os.cpu_count() or 4))),
            verbose=os.environ.get("LOCAL_MINICPM_VERBOSE", "0") == "1",
        )

    return _minicpm_llm


def _format_minicpm_prompt(target_text: str, transcript: str) -> str:
    return (
        "### Instruction:\n"
        f"{MINICPM_INSTRUCTION}\n\n"
        "### Input:\n"
        f"Target: {target_text} | ASR: {transcript}\n\n"
        "### Output:\n"
    )


def _parse_boolean_response(text: str) -> bool:
    normalized = text.strip()
    if normalized.startswith("True"):
        return True
    if normalized.startswith("False"):
        return False

    for token in normalized.replace("\n", " ").split():
        clean_token = token.strip(" .,:;\"'`[]{}()").casefold()
        if clean_token == "true":
            return True
        if clean_token == "false":
            return False

    return False


def local_transcribe_audio(audio_filepath: str) -> str:
    """Transcribe a local audio file with faster-whisper tiny.en."""
    model = _load_whisper_model()
    resolved_audio_filepath = _resolve_audio_filepath(audio_filepath)
    segments, _info = model.transcribe(
        resolved_audio_filepath,
        language="en",
        beam_size=1,
        vad_filter=True,
        word_timestamps=False,
    )
    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return _normalize_text(transcript)


def local_synthesize_speech(target_text: str) -> str:
    """Synthesize target text locally with VoxCPM and return a WAV filepath."""
    import soundfile as sf

    model = _load_voxcpm_model()
    wav = model.generate(
        text=target_text,
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=float(os.environ.get("LOCAL_VOXCPM_CFG_VALUE", "2.0")),
        inference_timesteps=int(os.environ.get("LOCAL_VOXCPM_INFERENCE_TIMESTEPS", "10")),
        normalize=os.environ.get("LOCAL_VOXCPM_NORMALIZE", "1") != "0",
        denoise=os.environ.get("LOCAL_VOXCPM_DENOISE", "1") != "0",
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )

    output_path = Path(tempfile.gettempdir()) / f"read_along_{_safe_audio_label(target_text)}.wav"
    sf.write(str(output_path), wav, SAMPLE_RATE)
    return str(output_path)


def local_ask_minicpm_judge(target_text: str, transcript: str) -> bool:
    """Ask the local GGUF MiniCPM evaluator whether the reading is acceptable."""
    try:
        llm = _load_minicpm_llm()
        response = llm(
            _format_minicpm_prompt(target_text, transcript),
            max_tokens=8,
            temperature=0.0,
            stop=["###", "\n\n"],
        )
        text = str(response["choices"][0]["text"])
        return _parse_boolean_response(text)
    except Exception:
        return False
