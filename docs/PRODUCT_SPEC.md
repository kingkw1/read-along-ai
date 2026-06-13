# Read-Along AI: Product Specification

## 1. Core User Flow
The hackathon MVP provides a guided, distraction-free sentence reading experience. The central loop is:
1. **Display:** The UI presents one short target sentence from the fixed curriculum.
2. **Assist:** The child can tap any word to hear help before attempting the sentence.
3. **Listen:** The child records a read-aloud attempt for the full sentence.
4. **Evaluate (ASR):** An ASR model (Cohere or faster-whisper) converts the recording to text.
5. **Judge:** The system first checks for an exact normalized sentence match, then uses the fine-tuned MiniCPM phonetic evaluator for close or ambiguous cases.
6. **Feedback:** The system triggers a visual reward and advances on success, or gives a gentle retry prompt on failure.

*Note: The user can toggle between "Turbo Mode" (Modal Cloud APIs) and "Off the Grid Mode" (Local Inference) to control where these models execute.*

## 2. MVP Curriculum State
The current application intentionally avoids multiple reading levels. The shipped curriculum is a compact set of short, decodable sentence prompts:

* "The cat sat."
* "The dog ran fast."
* "She had a red hat."
* "I love to play outside."

The goal is to prove the end-to-end read-aloud loop, phonetic tolerance, local/cloud routing, and child-friendly UI without adding separate phonics or CVC workflows during the final submission window.

### Sentence Reading Mode
* **Objective:** Practice confidence, fluency, and sight-word tracking on short sentences.
* **UI State:** One large sentence, rendered as clickable word spans.
* **Behavior:** The system evaluates the full recorded sentence against the displayed target.
* **Assistance:** Clicking an individual word plays a cached local word clip when available, with browser speech synthesis as a fallback. Clicking "Listen to Sentence" plays the full sentence using VoxCPM.

## 3. Phonetic Evaluation Logic
Children have developing articulation. Standard ASR will often misinterpret their speech, and strict string matching can turn an acceptable read attempt into a discouraging failure. The target evaluation stack is therefore:

* **Exact Normalized Match:** Strip punctuation and casing. If the transcript exactly matches the target text after normalization, accept immediately.
* **Fine-Tuned MiniCPM Judge:** If the exact match fails, call the published MiniCPM evaluator (`kingkw1/minicpm-phonetic-evaluator`) with the target text and ASR transcript. The model returns `True` when the transcript is an acceptable phonetic match and `False` when the child likely said the wrong thing.
* **Fail-Closed Judge:** If the MiniCPM evaluator fails, the attempt is rejected rather than being accidentally accepted.
* **Decision Boundary:** The judge should accept child-speech variants such as lisps, slow articulation, or harmless ASR phrasing, while rejecting unrelated speech, skipped target content, or semantically different words.

## 4. Reward & Feedback Mechanism
The app must provide immediate, positive reinforcement without overwhelming the UI.

* **Success State:**
    * *Visual:* A hidden HTML `<div>` toggles to display animated stars and a confetti burst.
    * *Action:* Automatically loads the next sentence after a short delay.
* **Retry State:**
    * If the evaluator rejects an attempt, the UI asks the child to try recording again without exposing technical errors.
* **On-Demand Assistance:**
    * Triggered by the child clicking a word or the "Listen to Sentence" button.

## 5. Deferred Scope
The original plan included separate phonics and CVC word levels. Those are now post-hackathon candidates, not part of the submission MVP. The current submission should be judged as a sentence-first reading assistant with strong privacy, small-model, and speech-evaluation architecture.
