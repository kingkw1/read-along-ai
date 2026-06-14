---
title: Read-Along AI
emoji: 🦉
colorFrom: blue
colorTo: yellow
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
tags:
  - build-small-hackathon
  - backyard-ai
  - off-brand
  - off-the-grid
  - llama-champion
  - tiny-titan
  - sharing-is-caring
  - field-notes
  - well-tuned
---

# 🦉 Read-Along AI: The Offline Reading Teacher

**Live App:** [read-along-ai](https://huggingface.co/spaces/build-small-hackathon/read-along-ai)

**Organization:** Hosted under the official [`build-small-hackathon`](https://huggingface.co/build-small-hackathon) HF Org.

**A Build Small Hackathon Submission**
*Track: Backyard AI*

## 📖 The Vision
Learning to read is a monumental milestone for young children, but practice can be intimidating when feedback is inconsistent, delayed, or too harsh for developing speech. **Read-Along AI** was built to solve a very specific problem: *[Insert personalization: e.g., Detail how you homeschool your children, and this tool was built specifically to solve the friction of your daily reading curriculum]* providing a patient, distraction-free reading assistant that listens to a child read short sentences and gives instant, gentle feedback.

**Field Testing Note:** *[Insert brief anecdote or reaction from the children actually using the app here to prove real-world usage for the Backyard AI track]*

Crucially, because this tool is for kids, it requires absolute data privacy. It leverages localized, small-parameter models to ensure a child's voice data never enters a corporate data lake. 

## 🛠️ The Tech Stack & Architecture
This application strictly adheres to the < 32B parameter constraint, utilizing highly optimized small models for a real-time, fluid user experience.

### Development Documentation
For a deep dive into the architecture and development plan, please review our spec documents:
* [Product Specification](docs/PRODUCT_SPEC.md)
* [UI/UX & Frontend Specification](docs/UI_UX_SPEC.md)
* [API & Backend Contract](docs/API_CONTRACT_SPEC.md)
* [Deployment Strategy](docs/DEPLOYMENT_SPEC.md)
* [Hackathon Roadmap](docs/ROADMAP.md)

### Components
* **Frontend:** A custom, gamified Gradio interface ("Off-Brand" UI) built for legibility and young readers.
* **ASR (Speech-to-Text):** **Cohere Transcribe** (2B parameters) in Turbo Mode and `faster-whisper` `tiny.en` in Off the Grid Mode.
* **Reading Evaluator:** A fine-tuned **MiniCPM phonetic evaluator** (`kingkw1/minicpm-phonetic-evaluator`) judges close or ambiguous ASR transcripts after exact normalized matching.
* **TTS (Text-to-Speech):** **OpenBMB VoxCPM** (0.5B parameters). This acts as the central interactive component for sentence read-back and on-demand word assistance.
* **Compute / Inference:** Utilizes a **Dual-Mode Hybrid Architecture**. The app includes **Turbo Mode** for Modal serverless endpoints and **Off the Grid Mode** for local Hugging Face Space resources.

### Dual-Mode Inference Engine
The app deliberately ships with both inference paths:

* **🏕️ Off the Grid Mode (Local):** Runs inside the Hugging Face Space without Modal. Local ASR uses `faster-whisper`, the phonetic evaluator loads the Q4 MiniCPM GGUF through `llama-cpp-python`, and local TTS uses VoxCPM.
* **⚡ Turbo Mode (Modal):** Routes the same Gradio UI through Modal endpoints for low-latency Cohere ASR, VoxCPM TTS, and the hosted MiniCPM evaluator.

For final judging, Off the Grid Mode should be verified on upgraded Hugging Face Space hardware first. The Q4 GGUF is included in the Space payload at `models/gguf/minicpm-phonetic-evaluator-q4_k_m.gguf`; `LOCAL_MINICPM_GGUF_PATH` only needs to be set if the model is mounted elsewhere.

## 🏆 Hackathon Eligibility & Attributions

### OpenAI Codex Track ($10,000)
This entire application, including the Gradio UI and backend Modal logic, was orchestrated using OpenAI's **Codex (GPT-5.5)**. Codex acted as the lead developer in a truly holistic manner:
* Generated the custom CSS overrides for the Off-Brand gamification.
* Wrote the Modal serverless stub functions and the Gradio abstraction wrappers.
* Managed the repository structure and environment variable integration.

* **GitHub Repository:** https://github.com/kingkw1/read-along-ai
* *Note to Judges: Please see the commit history for automated Codex attributions.*

### Modal Compute Awards
The high-speed inference endpoints powering the primary "Turbo Mode" are hosted entirely on **Modal**. This provides the necessary sub-second response times required to keep a young child focused. We also utilized Modal A100s for a rapid fine-tuning job to train the phonetic evaluator model.

### Local Verification
The repository includes a local-only smoke script for the Space path:

```bash
python scripts/manual/local_smoke.py
```

This script imports `local_inference.py`, resolves the Q4 GGUF, transcribes a committed curriculum audio file with `faster-whisper`, calls the MiniCPM judge through `llama-cpp-python`, and generates a local VoxCPM audio clip. It does not require Modal credentials.

### Badges Claimed (Bonus Quest Champion Strategy)
* 🏅 **Off-Brand:** The default Gradio UI has been completely overhauled with custom CSS to create a distraction-free, gamified experience for early learners.
* 🏅 **Well-Tuned:** [`kingkw1/minicpm-phonetic-evaluator`](https://huggingface.co/kingkw1/minicpm-phonetic-evaluator)
* 🏅 **Tiny Titan:** Every individual model used in this pipeline (and their combined footprint) is strictly under the 4B parameter threshold.
  * *Parameter Math:* Cohere Transcribe/faster-whisper (2B / 0.04B) + OpenBMB VoxCPM (0.5B) + MiniCPM Evaluator (2.4B) = ***2.9B Total Parameters***.
* 🏅 **Off the Grid:** The app includes a UI toggle that disconnects from Modal and runs `faster-whisper`, `llama.cpp`, and VoxCPM entirely locally inside the Hugging Face Space.
* 🏅 **Llama Champion:** The local phonetic evaluator runs exclusively through `llama-cpp-python`.
* 🏅 **Sharing is Caring:** *[Insert Hugging Face Dataset Link to Codex Agent Traces]*
* 🏅 **Field Notes:** *[Insert Link to Blog Post / Medium Article]*

## 🚀 How It Works
The hackathon MVP is focused on a stable sentence-reading loop:
1. **Choose a sentence:** The app displays one short curriculum sentence in large, clickable text.
2. **Get help when needed:** The child can tap a word to hear it, or press "Listen to Sentence" for a full VoxCPM read-back.
3. **Read aloud:** The child records an attempt through the microphone.
4. **Evaluate gently:** The app accepts exact normalized matches immediately, then asks the fine-tuned MiniCPM evaluator to judge close child-speech or ASR variants.
5. **Celebrate or retry:** Correct readings trigger confetti and advance to the next sentence; rejected readings get a simple, encouraging retry prompt.

Earlier phonics and CVC word levels were intentionally deferred. The current submission prioritizes reliability, privacy, and demo clarity around short-sentence reading practice.

## 📹 Submission Links
* **Demo Video:** [Insert Video Link]
* **Social Post:** [Insert X/LinkedIn Link]

## 👥 Team
* Hugging Face Username: `kingkw1`

### About the Developer
**Kevin King** is a Senior Machine Learning Engineer and AI Research Scientist with an M.S. in Neural and Electrical Engineering. With over seven years of enterprise MLOps experience, he specializes in multimodal signal fusion, neuro-symbolic architectures, and the deployment of local-first, air-gapped AI systems for resource-constrained environments.

Kevin has a proven track record of engineering highly optimized, award-winning agentic workflows. His previous hackathon builds include **AffectLink** (a real-time multimodal emotion recognition pipeline that won 1st Place in the 2025 HP & NVIDIA Developer Challenge) and **The Lung Listener** (Winner of the 2026 Google Gemini API Hackathon). *Read-Along AI* represents a continuation of his focus on privacy-preserving, localized AI that solves tangible, real-world problems.

## 📜 Open Source License
This project is licensed under the MIT License.

## Future Roadmap: Real-Time Streaming
The long-term V2 architecture will move from batch-style audio processing to real-time WebSocket streaming so young readers can receive instant visual feedback as they speak. The goal is to dynamically highlight each word on the page using word-level ASR timestamps, creating a tighter read-aloud loop for 4, 6, and 7-year-old learners.

For the 9-day hackathon MVP, real-time streaming was intentionally deferred to protect stability, privacy, and demo reliability. The planned post-hackathon architecture will refactor Gradio audio capture to use `streaming=True`, convert Modal inference endpoints into continuous generators, and extract word-level timestamps from the ASR model for live word highlighting.
