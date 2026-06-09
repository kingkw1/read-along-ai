# Read-Along AI: Deployment & Environment Specification

## 1. Objective
This document outlines the strict deployment architecture for hosting the Read-Along AI frontend as a Hugging Face Space **inside the official `build-small-hackathon` organization**, while securely routing inference calls to Modal. Codex must structure the repository to support an automated, error-free deployment into this specific org.

## 2. File Structure
Codex must generate and organize the repository using the following flat structure to comply with Hugging Face Spaces requirements:

* `/app.py` - The main Gradio application and UI logic.
* `/modal_inference.py` - The Modal backend definitions (`@app.function()` for Cohere and OpenBMB).
* `/requirements.txt` - The Python dependencies for the Hugging Face Space.
* `/README.md` - The hackathon submission document containing the required Codex and Modal attributions.
* `/.gitignore` - Must explicitly ignore any local `.wav` files, `.env` files, and `__pycache__`.

## 3. Environment Variables & Security
**CRITICAL RULE FOR CODEX:** Under no circumstances should any authentication tokens, API keys, or secrets be hardcoded into `app.py` or `modal_inference.py`. 

* **Modal Authentication:** The Hugging Face Space will authenticate with Modal using environment variables. 
* Codex must write the Python code under the assumption that `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` will be injected at runtime by the Hugging Face Spaces Secrets manager.
* Use standard `modal.App.lookup()` or direct Modal Python Client invocation methods to trigger the endpoints defined in `modal_inference.py` securely.

## 4. Dependencies (`requirements.txt`)
The `requirements.txt` file must be kept as lightweight as possible to ensure fast boot times on the Hugging Face Space. Codex should include:
* `gradio` (Specify a stable, recent version, e.g., `gradio>=4.0.0`)
* `modal` (For the client-side RPC calls)
* `python-Levenshtein` or `thefuzz` (For the fuzzy matching logic in Level 2 & 3)
* *Note: Heavy ML dependencies like `torch`, `transformers`, or the `cohere` SDK must NOT be in this file. They belong strictly in the image definition of the Modal functions in `modal_inference.py`.*

## 5. Deployment Execution Flow
When the developer pushes to the main branch, the expected execution flow is:
1. **Backend Deploy:** The developer runs `modal deploy modal_inference.py` locally to push the inference endpoints to the cloud.
2. **Frontend Sync:** The Hugging Face Space automatically rebuilds using the GitHub repository integration.
3. **Runtime:** When a user visits the Space, `app.py` boots up the Gradio UI, and any audio events trigger the authenticated Modal client to route the payloads to the pre-deployed serverless functions.