# Read-Along AI: 9-Day Development Roadmap & Architecture Strategy

## 1. Development Philosophy
This hackathon sprint is strictly time-boxed. The primary directive is to secure a fully functional, high-speed application and record the demo video *before* attempting advanced local integrations. 

To achieve this, the application relies on a **Decoupled Two-Phase Architecture**:
* **Phase 1 (Speed & Stability):** The Gradio UI routes all inference through fast, serverless Modal endpoints. This guarantees the app is ready for the demo video without local compute bottlenecks.
* **Phase 2 (The Llama Swap):** Once the core app is secure, the backend is swapped to a local `llama.cpp` runtime to capture the "Llama Champion" and "Off the Grid" merit badges.

## 2. Phase 1: The Fast Build (Days 1–7)

### Days 1–2: UI Scaffolding & Codex Integration
* **Objective:** Establish the "Off-Brand" Gradio frontend and secure OpenAI Codex eligibility.
* **Tasks:**
  * Use Codex to generate the base `app.py` utilizing the `UI_UX_SPEC.md`.
  * Ensure all Gradio components are styled with custom CSS (no default textboxes).
  * Push initial commits to the public GitHub repository (ensuring Codex attribution).

### Days 3–4: Backend Plumbing (Modal)
* **Objective:** Wire the Gradio UI to the heavy ML models without locking up the main thread.
* **Tasks:**
  * Deploy `modal_inference.py` containing the `run_cohere_asr` and `run_voxcpm_tts` endpoints.
  * Connect the Gradio audio recording events to the Modal endpoints via the abstraction wrappers defined in `API_CONTRACT_SPEC.md`.
  * Test round-trip latency to ensure it is suitable for a child's attention span.

### Days 5–6: Core Logic & Gamification
* **Objective:** Build the brain of the reading teacher.
* **Tasks:**
  * Implement the progression states (Phonics -> CVC -> Sentences).
  * Write the fuzzy-matching and Levenshtein distance logic for the ASR evaluation.
  * Trigger the hidden HTML CSS animations (stars/confetti) upon successful reads.

### Day 7: The Safety Net & Demo Recording
* **Objective:** Lock in the hackathon submission requirements.
* **Tasks:**
  * Freeze feature development.
  * Field-test the Phase 1 app with the target users (ages 4, 6, and 7).
  * **Crucial:** Record the final demo video and draft the social media post using this stable, Modal-backed version.

## 3. Phase 2: The Integration Stretch (Days 8–9)

### Day 8: The `llama.cpp` Swap
* **Objective:** Secure the "Llama Champion" and "Off the Grid" badges.
* **Tasks:**
  * Isolate the Modal wrapper functions in `app.py`.
  * Pull quantized GGUF versions of the reasoning and TTS models into the Hugging Face Space.
  * Use the `llama-cpp-python` bindings to redirect the wrapper functions to the local models.
  * *Constraint:* If C++ variable consolidation or local runtime latency breaks the UX, immediately revert the wrappers back to the Modal endpoints.

### Day 9: Polish & Final Submission
* **Objective:** Submit a winning entry.
* **Tasks:**
  * Finalize the `README.md` to ensure all rules (Codex commits, Model parameters, Org submission) are strictly documented.
  * Publish the social media post.
  * Submit the Hugging Face Space link to the Build Small Hackathon portal before midnight UTC.