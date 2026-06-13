#!/usr/bin/env python3
"""Download a Hugging Face model and quantize it to GGUF with llama.cpp.

This script intentionally delegates the model conversion and quantization
steps to llama.cpp's own tools. It uses huggingface_hub only for the model
download/snapshot step.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "kingkw1/minicpm-phonetic-evaluator"
DEFAULT_LLAMA_CPP_DIR = REPO_ROOT / "third_party" / "llama.cpp"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "hf" / "kingkw1" / "minicpm-phonetic-evaluator"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "gguf"
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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    display = " ".join(cmd)
    if cwd is not None:
        print(f"\n$ (cd {cwd} && {display})", flush=True)
    else:
        print(f"\n$ {display}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clean_python_subprocess_env(extra_pythonpath: list[Path] | None = None) -> dict[str, str]:
    """Return an env that keeps the active venv but removes Conda/Python leakage."""
    env = os.environ.copy()
    for key in CONDA_ENV_KEYS:
        env.pop(key, None)

    # These can make a venv interpreter import/link against Conda packages.
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    if extra_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in extra_pythonpath)

    return env


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download kingkw1/minicpm-phonetic-evaluator from Hugging Face, "
            "convert it to FP16 GGUF, then quantize it to Q4_K_M."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision/branch/tag/SHA.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llama-cpp-dir", type=Path, default=DEFAULT_LLAMA_CPP_DIR)
    parser.add_argument(
        "--clone-llama-cpp",
        action="store_true",
        help="Clone ggml-org/llama.cpp into --llama-cpp-dir if it does not exist.",
    )
    parser.add_argument(
        "--build-llama-cpp",
        action="store_true",
        help="Build llama.cpp with CMake before quantizing.",
    )
    parser.add_argument(
        "--install-conversion-requirements",
        action="store_true",
        help="Install llama.cpp's Python conversion requirements into the current Python environment.",
    )
    parser.add_argument("--jobs", type=positive_int, default=os.cpu_count() or 4)
    parser.add_argument("--outtype", default="f16", choices=["f16", "f32", "bf16", "q8_0", "auto"])
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_llama_cpp(args: argparse.Namespace) -> Path:
    llama_dir = args.llama_cpp_dir.resolve()
    if not llama_dir.exists():
        if not args.clone_llama_cpp:
            raise SystemExit(
                f"llama.cpp was not found at {llama_dir}.\n"
                "Re-run with --clone-llama-cpp, or pass --llama-cpp-dir /path/to/llama.cpp."
            )
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(llama_dir)])

    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise SystemExit(f"Expected llama.cpp conversion script not found: {convert_script}")

    if args.install_conversion_requirements:
        req_file = llama_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
        if not req_file.exists():
            raise SystemExit(f"Expected llama.cpp conversion requirements not found: {req_file}")
        run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            env=clean_python_subprocess_env(),
        )

    if args.build_llama_cpp:
        build_dir = llama_dir / "build"
        run(["cmake", "-S", str(llama_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(args.jobs)])

    return llama_dir


def find_quantize_binary(llama_dir: Path) -> Path:
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "bin" / "quantize",
        llama_dir / "llama-quantize",
        llama_dir / "quantize",
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    from_path = shutil.which("llama-quantize") or shutil.which("quantize")
    if from_path:
        return Path(from_path)

    searched = "\n  ".join(str(path) for path in candidates)
    raise SystemExit(
        "Could not find a llama.cpp quantize binary. Build llama.cpp with "
        "--build-llama-cpp or point --llama-cpp-dir at a built checkout.\n"
        f"Searched:\n  {searched}\n  PATH: llama-quantize, quantize"
    )


def download_model(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - user-facing dependency guard
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  python -m pip install huggingface_hub"
        ) from exc

    model_dir = args.model_dir.resolve()
    if args.skip_download:
        if not model_dir.exists():
            raise SystemExit(f"--skip-download was set, but model dir does not exist: {model_dir}")
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} to {model_dir}", flush=True)
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(model_dir),
        token=args.hf_token or os.environ.get("HF_TOKEN"),
    )
    return model_dir


def convert_to_fp_gguf(args: argparse.Namespace, llama_dir: Path, model_dir: Path) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp_gguf = (args.output_dir / "minicpm-phonetic-evaluator-f16.gguf").resolve()

    if args.skip_convert:
        if not fp_gguf.exists():
            raise SystemExit(f"--skip-convert was set, but FP GGUF does not exist: {fp_gguf}")
        return fp_gguf

    if fp_gguf.exists() and not args.overwrite:
        print(f"Using existing FP GGUF: {fp_gguf}", flush=True)
        return fp_gguf

    gguf_py = llama_dir / "gguf-py"
    env = clean_python_subprocess_env([gguf_py])

    run(
        [
            sys.executable,
            str(llama_dir / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile",
            str(fp_gguf),
            "--outtype",
            args.outtype,
        ],
        cwd=llama_dir,
        env=env,
    )
    return fp_gguf


def quantize_gguf(args: argparse.Namespace, llama_dir: Path, fp_gguf: Path) -> Path:
    quantized = (args.output_dir / "minicpm-phonetic-evaluator-q4_k_m.gguf").resolve()
    if args.skip_quantize:
        return quantized

    if quantized.exists() and not args.overwrite:
        print(f"Using existing quantized GGUF: {quantized}", flush=True)
        return quantized

    quantize = find_quantize_binary(llama_dir)
    run([str(quantize), str(fp_gguf), str(quantized), args.quantization])
    return quantized


def main() -> None:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.llama_cpp_dir = args.llama_cpp_dir.resolve()

    llama_dir = ensure_llama_cpp(args)
    model_dir = download_model(args)
    fp_gguf = convert_to_fp_gguf(args, llama_dir, model_dir)
    quantized = quantize_gguf(args, llama_dir, fp_gguf)

    print("\nDone.", flush=True)
    print(f"FP GGUF:        {fp_gguf}", flush=True)
    print(f"Quantized GGUF: {quantized}", flush=True)


if __name__ == "__main__":
    main()
