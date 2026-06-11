"""Modal inference endpoints for Read-Along AI.

Deploy with:
    modal deploy modal_inference.py
"""

from __future__ import annotations

import io
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import modal


APP_NAME = "read-along-ai-inference"
CACHE_DIR = "/model-cache"
COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
VOXCPM_MODEL_ID = "openbmb/VoxCPM-0.5B"
MINICPM_MODEL_ID = "kingkw1/minicpm-phonetic-evaluator"
SAMPLE_RATE = 16_000
MINICPM_INSTRUCTION = (
    "Determine if the ASR transcript is a valid phonetic match for the target word. "
    "Output only True or False."
)


app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("read-along-ai-model-cache", create_if_missing=True)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "accelerate",
        "huggingface_hub",
        "librosa",
        "numpy",
        "protobuf",
        "scipy",
        "sentencepiece",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers>=5.4.0",
        "voxcpm",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/transformers",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "MODELSCOPE_CACHE": f"{CACHE_DIR}/modelscope",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

minicpm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.40.2",
        "accelerate==0.29.3",
        "huggingface_hub",
        "sentencepiece",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/transformers",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


_cohere_processor: Any | None = None
_cohere_model: Any | None = None
_voxcpm_model: Any | None = None
_minicpm_tokenizer: Any | None = None
_minicpm_model: Any | None = None


def _ensure_cache_dirs() -> None:
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _is_silent_wav(audio_bytes: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error:
        return False

    return bool(frames) and all(byte == 0 for byte in frames)


def _load_cohere_asr() -> tuple[Any, Any]:
    global _cohere_processor, _cohere_model

    if _cohere_processor is None or _cohere_model is None:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        _ensure_cache_dirs()
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _cohere_processor = AutoProcessor.from_pretrained(
            COHERE_MODEL_ID,
            cache_dir=CACHE_DIR,
            token=_hf_token(),
        )
        _cohere_model = CohereAsrForConditionalGeneration.from_pretrained(
            COHERE_MODEL_ID,
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=dtype,
            token=_hf_token(),
        )
        _cohere_model.eval()
        model_cache.commit()

    return _cohere_processor, _cohere_model


def _load_voxcpm_tts() -> Any:
    global _voxcpm_model

    if _voxcpm_model is None:
        from voxcpm import VoxCPM

        _ensure_cache_dirs()
        _voxcpm_model = VoxCPM.from_pretrained(
            VOXCPM_MODEL_ID,
            cache_dir=CACHE_DIR,
        )
        model_cache.commit()

    return _voxcpm_model


def _load_minicpm_evaluator() -> tuple[Any, Any]:
    global _minicpm_tokenizer, _minicpm_model

    if _minicpm_tokenizer is None or _minicpm_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _ensure_cache_dirs()
        _minicpm_tokenizer = AutoTokenizer.from_pretrained(
            MINICPM_MODEL_ID,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            token=_hf_token(),
        )
        if _minicpm_tokenizer.pad_token is None:
            _minicpm_tokenizer.pad_token = _minicpm_tokenizer.eos_token

        _minicpm_model = AutoModelForCausalLM.from_pretrained(
            MINICPM_MODEL_ID,
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=_hf_token(),
        )
        _minicpm_model.eval()
        model_cache.commit()

    return _minicpm_tokenizer, _minicpm_model


def _format_minicpm_prompt(target_word: str, transcript: str) -> str:
    return (
        "### Instruction:\n"
        f"{MINICPM_INSTRUCTION}\n\n"
        "### Input:\n"
        f"Target: {target_word} | ASR: {transcript}\n\n"
        "### Output:\n"
    )


def _parse_boolean_response(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith("True"):
        return "True"
    if normalized.startswith("False"):
        return "False"

    for token in normalized.replace("\n", " ").split():
        clean_token = token.strip(" .,:;\"'`[]{}()").casefold()
        if clean_token == "true":
            return "True"
        if clean_token == "false":
            return "False"

    return "False"


@app.function(
    image=inference_image,
    gpu="A10G",
    timeout=600,
    min_containers=0,
    scaledown_window=300,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_cohere_asr(audio_bytes: bytes) -> dict[str, str]:
    """Transcribe WAV/audio bytes with Cohere Transcribe."""
    if _is_silent_wav(audio_bytes):
        return {"text": "", "status": "success"}

    import torch
    from transformers.audio_utils import load_audio

    processor, model = _load_cohere_asr()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = audio_file.name

    try:
        audio = load_audio(audio_path, sampling_rate=SAMPLE_RATE)
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language="en",
            punctuation=False,
        ).to(model.device)
        inputs = inputs.to(model.device, dtype=model.dtype)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256)

        audio_chunk_index = inputs.get("audio_chunk_index")
        decode_kwargs: dict[str, Any] = {"skip_special_tokens": True}
        if audio_chunk_index is not None:
            decode_kwargs.update(
                {
                    "audio_chunk_index": audio_chunk_index,
                    "language": "en",
                }
            )
        decoded = processor.decode(outputs, **decode_kwargs)
        if isinstance(decoded, list):
            text = " ".join(str(chunk).strip() for chunk in decoded if str(chunk).strip())
        else:
            text = str(decoded).strip()
        return {"text": text, "status": "success"}
    finally:
        try:
            os.unlink(audio_path)
        except FileNotFoundError:
            pass


@app.function(
    image=inference_image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_voxcpm_tts(text: str) -> bytes:
    """Generate raw WAV audio bytes with OpenBMB VoxCPM."""
    import soundfile as sf

    model = _load_voxcpm_tts()
    wav = model.generate(
        text=text,
        prompt_wav_path=None,
        prompt_text=None,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=True,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )

    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


@app.function(
    image=minicpm_image,
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_minicpm_evaluator(target_word: str, transcript: str) -> str:
    """Classify whether an ASR transcript is an acceptable phonetic match."""
    import torch

    tokenizer, model = _load_minicpm_evaluator()
    prompt = _format_minicpm_prompt(target_word=target_word, transcript=transcript)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _parse_boolean_response(generated_text)
