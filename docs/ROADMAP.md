# Read-Along AI: 9-Day Development Roadmap & Architecture Strategy

## 1. Development Philosophy
This hackathon sprint is strictly time-boxed. The primary directive is to secure a fully functional Minimum Viable Product (MVP) as fast as possible, test the highest-risk technical hurdles immediately after, and save the UI gamification for the final polish.

## 2. Phase 1: The Barebones MVP (Days 1–4)
*Goal: Establish a bulletproof, end-to-end event loop using "safe" cloud endpoints before attempting local deployment.*

### Days 1–2: UI Scaffolding & Mock Inference
* **Objective:** Establish the Gradio frontend and secure OpenAI Codex eligibility.
* **Tasks:**
  * Use Codex to generate the base `app.py` utilizing the `UI_UX_SPEC.md`. 
  * **Fail-Point Test:** Wire the UI to *mock* backend functions (e.g., functions that instantly return dummy text/audio) to verify the Gradio event loops, state management, and microphone capture work flawlessly without waiting for ML models to load.
  * Push initial commits to the public GitHub repository (ensuring Codex attribution).
* **Verification Checkpoint 1:** **Run the Gradio app locally. Manually verify that clicking "Record" correctly captures audio and triggers the mock success state (confetti animation) without any stack trace errors in the terminal.**

### Days 3–4: Modal Plumbing & Core Logic
* **Objective:** Connect the real brains and build the progression logic.
* **Tasks:**
  * Deploy `modal_inference.py` containing the `run_cohere_asr` and `run_voxcpm_tts` endpoints.
  * Connect the Gradio UI to the Modal endpoints via abstraction wrappers.
  * Implement the progression states (Phonics -> CVC -> Sentences) and fuzzy-matching logic. 
  * *Checkpoint:* The app is now fully functional. If everything else fails, this is your submission.
* **Verification Checkpoint 2:** **Run `pytest test_backend.py` to verify Modal ASR and TTS endpoints return the correct JSON/bytes. Then, use `gradio.Client` to pass a test audio file through the full UI pipeline to confirm end-to-end integration.**

## 3. Phase 2: The High-Risk Pivot (Days 5–6)
*Goal: Chase the "Well-Tuned", "Off the Grid", and "Llama Champion" badges by moving compute to the edge.*

### Day 5: The Modal Fine-Tuning Job
* **Objective:** Secure the "Well-Tuned" badge and prepare for local deployment.
* **Tasks:**
  * Utilize Modal's A100 infrastructure to run a rapid fine-tuning job (e.g., fine-tuning a tiny LLM for better phonetic error correction or a TTS voice).
  * Export the resulting model weights to the quantized GGUF format.

### Day 6: The `llama.cpp` Edge Swap (Early Risk Test)
* **Objective:** Test local runtime latency on the Hugging Face ZeroGPU.
* **Tasks:**
  * Isolate the Modal wrapper functions in `app.py`.
  * Pull the custom GGUF models into the Hugging Face Space.
  * Use `llama-cpp-python` bindings to run the models locally.
  * **Crucial Decision Point:** If the local latency ruins the child's UX, immediately abandon this phase and revert to the Phase 1 Modal endpoints. 
* **Verification Checkpoint 3:** **Deploy to the Hugging Face Space. Measure the round-trip latency of a single phonetic read attempt. If it takes longer than 2.5 seconds, trigger the rollback to Modal.**

## 4. Phase 3: Bells, Whistles & Submission (Days 7–9)
*Goal: Polish the UI, gather real-world proof, and stack the final badges.*

### Day 7: Gamification & UI Polish
* **Objective:** Secure the "Off-Brand" badge.
* **Tasks:**
  * Inject the custom CSS rules to overhaul the Gradio interface.
  * Trigger the hidden HTML CSS animations (stars/confetti) upon successful reads.
  * Ensure the text sizing and colors are optimized for a 4-to-7-year-old's accessibility.

### Day 8: Field Testing & Demo Recording
* **Objective:** Fulfill the *Backyard AI* criteria and target the "Best Demo" award.
* **Tasks:**
  * Put the app in front of the target users (ages 4, 6, and 7).
  * Record the final demo video showing real, authentic usage. Focus heavily on narrative: clearly film the *problem* (e.g., frustration of reading practice) followed by the *solution* (the joy of the interactive confetti animation). Storytelling counts as much as the build!
  * Draft the required social media post.

### Day 9: Final Submission & The Cleanup Badges
* **Objective:** Submit the winning entry and capture the final Bonus Quests.
* **Tasks:**
  * Publish your dataset/agent traces to the Hub (Secures "Sharing is Caring").
  * Publish a short blog post detailing your architecture choices (Secures "Field Notes").
  * Finalize the `README.md` to ensure all rules are strictly documented.
  * Submit the Hugging Face Space link to the portal.

## 5. Phase 4: Post-Hackathon / V2 Architecture
*Goal: Transition from batch processing to real-time streaming inference for instant visual feedback.*

* **Objective:** Allow words to highlight dynamically as the child speaks.
* **Architecture Shift:**
  * Refactor Gradio `gr.Audio` to utilize `streaming=True` via WebSockets.
  * Rewrite Modal endpoints as continuous generators.
  * Implement word-level timestamp extraction from the ASR model.
* **Product Impact:** Give early readers immediate visual confirmation by synchronizing spoken words with on-screen highlights, reducing the delay between effort and feedback.
* **Engineering Rationale:** The hackathon MVP deliberately favors stable batch-style inference over streaming complexity so the core reading loop remains reliable during the 9-day sprint. V2 can introduce real-time streaming once the baseline experience is proven.
