# Read-Along AI: Deployment & Environment Specification

## 1. Objective
This document outlines the strict deployment architecture for hosting Read-Along AI as Hugging Face Spaces **inside the official `build-small-hackathon` organization**. The main submission Space supports both Modal-powered Turbo Mode and local Off the Grid Mode; a second lightweight UI Space is used for fast CSS/layout iteration.

## 2. File Structure
Codex must generate and organize the repository using the following flat structure to comply with Hugging Face Spaces requirements:

* `/app.py` - The main Gradio application and UI logic.
* `/modal_inference.py` - The Modal backend definitions (`@app.function()` for Cohere, OpenBMB, and the fine-tuned MiniCPM evaluator).
* `/requirements.txt` - The full local-inference Python dependencies for the main Hugging Face Space.
* `/packages.txt` - System packages needed by the local inference stack.
* `/README.md` - The hackathon submission document containing the required Codex and Modal attributions.
* `/.gitignore` - Must explicitly ignore any local `.wav` files, `.env` files, and `__pycache__`.

## 3. Environment Variables & Security
**CRITICAL RULE FOR CODEX:** Under no circumstances should any authentication tokens, API keys, or secrets be hardcoded into `app.py` or `modal_inference.py`. 

* **Modal Authentication:** The Hugging Face Space will authenticate with Modal using environment variables. 
* Codex must write the Python code under the assumption that `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` will be injected at runtime by the Hugging Face Spaces Secrets manager.
* Use standard `modal.App.lookup()` or direct Modal Python Client invocation methods to trigger the endpoints defined in `modal_inference.py` securely.

## 4. Dependencies (`requirements.txt`)
The deployment has two possible runtime profiles:

* **UI iteration Space:** Keep dependencies light (`gradio`, `modal`, `rich`) and use `build-small-hackathon/read-along-ai-ui` for fast visual iteration.
* **Off the Grid submission Space:** Include the local inference stack needed by `local_inference.py`, including `faster-whisper`, CPU Torch/VoxCPM dependencies, and `llama-cpp-python`.

The main Space uses Python 3.11 and CPU-specific dependency pins to avoid the free-tier builder pulling large CUDA wheels or compiling `llama-cpp-python` from source. The Q4 GGUF is **not** uploaded to the Space repo because the Space has a 1 GB repository storage limit. Instead, `local_inference.py` resolves `minicpm-phonetic-evaluator-q4_k_m.gguf` from `LOCAL_MINICPM_GGUF_PATH`, a local checked-out model path, or the Hugging Face model cache after downloading it from `kingkw1/minicpm-phonetic-evaluator`.

Training-only dependencies (`peft`, `trl`, `bitsandbytes`, notebooks, and data-prep tooling) are useful for development but should be removed from the submitted Space unless they are required at runtime.

Development and training dependencies are tracked separately in `requirements-dev.txt`.

Heavy Modal-side dependencies remain defined in `modal_inference.py` images:
* Cohere ASR and VoxCPM dependencies live in the shared inference image.
* MiniCPM evaluator dependencies (`torch`, `transformers==4.40.2`, `accelerate`, `sentencepiece`) live in the dedicated evaluator image.

## 5. Deployment Execution Flow
When the developer pushes to the main branch, the expected execution flow is:
1. **Backend Deploy:** The developer runs `modal deploy modal_inference.py` locally to push the ASR, TTS, and MiniCPM evaluator endpoints to the cloud.
2. **Frontend Sync:** The developer deploys the vetted public Space payload with `scripts/deploy_space.sh` rather than pushing the whole repository to the Space remote.
3. **Runtime:** When a user visits the Space, `app.py` boots up the Gradio UI. The inference-engine toggle routes read attempts through either Modal endpoints or local inference, depending on the selected mode and deployed assets.
4. **Submission Verification:** Before claiming Off the Grid or Llama Champion, manually verify the submitted Space can complete a read attempt in local mode without calling Modal. The current deployed local path has completed a short recorded sentence in roughly 10 seconds end-to-end.

### Space Upload Shortcut
The Hugging Face Space is treated as a deployment target, while GitHub remains the source-of-truth repository. Deploy the main submission Space with:

```bash
COMMIT_MESSAGE="Enable local Off the Grid inference" ./scripts/deploy_space.sh --target main --yes
```

Deploy the lightweight UI iteration Space with:

```bash
COMMIT_MESSAGE="UI polish" ./scripts/deploy_space.sh --target ui --yes
```

The script stages only public runtime files, docs, and minimal data in `/tmp/read-along-ai-space-upload`, then uses `hf upload` to publish them. It deliberately excludes local raw audio, processed audio, notebook outputs, model artifacts, caches, `.hackathon`, `.venv`, and `.git`.
