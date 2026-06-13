# Read-Along AI: Deployment & Environment Specification

## 1. Objective
This document outlines the strict deployment architecture for hosting the Read-Along AI frontend as a Hugging Face Space **inside the official `build-small-hackathon` organization**, while securely routing inference calls to Modal. Codex must structure the repository to support an automated, error-free deployment into this specific org.

## 2. File Structure
Codex must generate and organize the repository using the following flat structure to comply with Hugging Face Spaces requirements:

* `/app.py` - The main Gradio application and UI logic.
* `/modal_inference.py` - The Modal backend definitions (`@app.function()` for Cohere, OpenBMB, and the fine-tuned MiniCPM evaluator).
* `/requirements.txt` - The Python dependencies for the Hugging Face Space.
* `/README.md` - The hackathon submission document containing the required Codex and Modal attributions.
* `/.gitignore` - Must explicitly ignore any local `.wav` files, `.env` files, and `__pycache__`.

## 3. Environment Variables & Security
**CRITICAL RULE FOR CODEX:** Under no circumstances should any authentication tokens, API keys, or secrets be hardcoded into `app.py` or `modal_inference.py`. 

* **Modal Authentication:** The Hugging Face Space will authenticate with Modal using environment variables. 
* Codex must write the Python code under the assumption that `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` will be injected at runtime by the Hugging Face Spaces Secrets manager.
* Use standard `modal.App.lookup()` or direct Modal Python Client invocation methods to trigger the endpoints defined in `modal_inference.py` securely.

## 4. Dependencies (`requirements.txt`)
The deployment has two possible runtime profiles:

* **Modal-first Space:** Keep Space dependencies light (`gradio`, `modal`) and route heavy inference to `modal_inference.py`.
* **Off the Grid Space:** Include the local inference stack needed by `local_inference.py`, including `faster-whisper`, VoxCPM/Torch dependencies, and `llama-cpp-python` plus the GGUF MiniCPM evaluator artifact.

The final `requirements.txt` should match the selected Space strategy before upload. Training-only dependencies (`peft`, `trl`, `bitsandbytes`, notebooks, and data-prep tooling) are useful for development but should be removed from the submitted Space unless they are required at runtime.

Development and training dependencies are tracked separately in `requirements-dev.txt`.

Heavy Modal-side dependencies remain defined in `modal_inference.py` images:
* Cohere ASR and VoxCPM dependencies live in the shared inference image.
* MiniCPM evaluator dependencies (`torch`, `transformers==4.40.2`, `accelerate`, `sentencepiece`) live in the dedicated evaluator image.

## 5. Deployment Execution Flow
When the developer pushes to the main branch, the expected execution flow is:
1. **Backend Deploy:** The developer runs `modal deploy modal_inference.py` locally to push the ASR, TTS, and MiniCPM evaluator endpoints to the cloud.
2. **Frontend Sync:** The developer deploys the vetted public Space payload with `scripts/deploy_space.sh` rather than pushing the whole repository to the Space remote.
3. **Runtime:** When a user visits the Space, `app.py` boots up the Gradio UI. The inference-engine toggle routes read attempts through either Modal endpoints or local inference, depending on the selected mode and deployed assets.
4. **Submission Verification:** Before claiming Off the Grid or Llama Champion, manually verify the submitted Space can complete a read attempt in local mode without calling Modal.

### Space Upload Shortcut
The Hugging Face Space is treated as a deployment target, while GitHub remains the source-of-truth repository. Deploy the Space with:

```bash
./scripts/deploy_space.sh
```

For non-interactive deploys:

```bash
COMMIT_MESSAGE="Describe this deploy" ./scripts/deploy_space.sh --yes
```

The script stages only public runtime files, docs, and minimal data in `/tmp/read-along-ai-space-upload`, then uses `hf upload` to publish them. It deliberately excludes local raw audio, processed audio, notebook outputs, model artifacts, caches, `.hackathon`, `.venv`, and `.git`.
