# Read-Along AI: Hackathon Roadmap & Architecture Strategy

## 1. Development Philosophy
This hackathon sprint is strictly time-boxed. The primary directive is to ship a reliable sentence-reading MVP, prove the highest-risk model integrations, and keep the final surface simple enough for a young reader to use without instruction.

The original multi-level reading plan has been deferred. The submission MVP is sentence-first: one short sentence at a time, clickable word help, full-sentence read-back, ASR transcription, MiniCPM phonetic judging, and immediate visual feedback.

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
* **Objective:** Connect the real brains and build the sentence-reading loop.
* **Tasks:**
  * Deploy `modal_inference.py` containing the `run_cohere_asr` and `run_voxcpm_tts` endpoints.
  * Connect the Gradio UI to the Modal endpoints via abstraction wrappers.
  * Implement the sentence curriculum loop and an initial tolerant matcher.
  * *Checkpoint:* The app is now fully functional. If everything else fails, this is your submission.
* **Verification Checkpoint 2:** **Run `RUN_MODAL_INTEGRATION=1 pytest tests/test_modal_integration.py` to verify Modal ASR and TTS endpoints return the correct JSON/bytes. Then, use `gradio.Client` or the live UI to pass a test audio file through the full app pipeline.**

## 3. Phase 2: The Fine-Tuned Evaluator (Days 5–6)
*Goal: Secure the "Well-Tuned" badge and improve child-speech scoring with a published small model.*

### Day 5: The Modal Fine-Tuning Job
* **Objective:** Secure the "Well-Tuned" badge and make reading evaluation more robust than strict ASR matching.
* **Tasks:**
  * Extract and manually verify a 50-word child-speech dataset.
  * Run Cohere ASR over the cleaned utterances and record strict-match failures.
  * Convert the results into binary phonetic-evaluator training data.
  * Utilize Modal's A100 infrastructure to fine-tune MiniCPM with LoRA.
  * Publish the resulting model to Hugging Face as `kingkw1/minicpm-phonetic-evaluator`.
  * Document the result in `notebooks/01_dataset_preparation.ipynb` and `notebooks/02_post_tuning_evaluation.ipynb`.
  * Expose `run_minicpm_evaluator` in `modal_inference.py`.
  * Route non-exact app matches through a MiniCPM wrapper such as `ask_minicpm_judge`.
  * Test deliberate mistakes such as "the dogs ran fast" against "the dog ran fast" to ensure the evaluator does not over-accept semantically different readings.
  * Remove broad edit-distance acceptance from the final app path so wrong readings are not accepted too easily.
* **Verification Checkpoint 3:** **Run the post-tuning notebook and manually test close-but-wrong readings in the Gradio app. The demo should show both recovered child-speech variants and rejected wrong readings.**

### Day 6: The Dual-Mode Hybrid Engine (Edge Integration)
* **Objective:** Secure the "Off the Grid" and "Llama Champion" badges while protecting the demo UX.
* **Tasks:**
  * Implement a Dual-Mode Toggle in `app.py` (`⚡ Turbo Mode` vs `🏕️ Off the Grid Mode`).
  * Create `local_inference.py` to house the local execution logic.
  * Integrate `faster-whisper` (tiny.en) for local ASR.
  * Use `llama-cpp-python` bindings to load the quantized MiniCPM evaluator locally.
  * Use committed curriculum WAVs and label timings for responsive local audio help, with live local VoxCPM kept as an opt-in fallback.
  * Wire the toggle to route to either `modal_inference.py` or `local_inference.py`.
* **Verification Checkpoint 4:** **Deploy to the Hugging Face Space. Flip the toggle to "Off the Grid Mode" and manually verify that a read attempt executes completely locally without calling Modal APIs.**

## 4. Phase 3: Bells, Whistles & Submission (Days 7–9)
*Goal: Polish the UI, gather real-world proof, and stack the final badges.*

### Day 7: Gamification & UI Polish
* **Objective:** Secure the "Off-Brand" badge.
* **Tasks:**
  * Inject the custom CSS rules to overhaul the Gradio interface.
  * Trigger the hidden HTML CSS animations (stars/confetti) upon successful reads.
  * Ensure the sentence text, microphone control, retry state, and word-help state are optimized for a young reader's accessibility.
  * Keep the visible workflow focused on sentence practice; do not reintroduce separate reading-level controls for the hackathon submission.

### Day 8: Field Testing & Demo Recording
* **Objective:** Fulfill the *Backyard AI* criteria and target the "Best Demo" award.
* **Tasks:**
  * Put the app in front of the target user(s).
  * Record the final demo video showing real, authentic usage. Focus heavily on narrative: clearly film the reading-practice friction followed by the sentence-reading loop, word help, and success feedback. Storytelling counts as much as the build!
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

## 6. Deferred Product Ideas
The earlier phonics and CVC levels remain useful future ideas, but they are not part of the current submission target. They should be revisited only after the sentence-reading MVP is deployed, tested, and documented.
