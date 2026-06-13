# Read-Along AI: Product Specification

## 1. Core User Flow
The application provides a guided, distraction-free reading experience that scales in complexity. The central loop is:
1. **Display:** The UI presents a target text (a letter, word, or sentence).
2. **Listen:** The child presses "Record" and attempts to read the target text.
3. **Evaluate (ASR):** An ASR model (Cohere or faster-whisper) converts the audio to text.
4. **Judge:** The system first checks for an exact normalized match, then uses the fine-tuned MiniCPM phonetic evaluator for close or ambiguous cases.
5. **Feedback:** The system triggers a visual reward and proceeds, or uses VoxCPM to provide gentle verbal assistance.

*Note: The user can toggle between "Turbo Mode" (Modal Cloud APIs) and "Off the Grid Mode" (Local Inference) to control where these models execute.*

## 2. Progression States
The application manages three distinct difficulty tiers tailored to specific cognitive milestones.

### Level 1: Foundational Phonics (Age 4 Target)
* **Objective:** Phoneme isolation and letter recognition.
* **UI State:** A single massive character (e.g., "M").
* **Behavior:** The system is listening for the *sound* of the letter, not the name. 
* **Assistance:** If the child clicks the letter, VoxCPM plays the phonetic sound (e.g., /m/ not "em").

### Level 2: CVC Words (Age 6 Target)
* **Objective:** Blending Consonant-Vowel-Consonant words.
* **UI State:** A single 3-4 letter word (e.g., "C A T").
* **Behavior:** The system listens for the full blended word. 
* **Assistance:** Clicking the word triggers VoxCPM to slowly sound out the phonemes (/k/ ... /a/ ... /t/ -> "cat").

### Level 3: Sentences & Sight Words (Age 7 Target)
* **Objective:** Reading fluency and tracking.
* **UI State:** A short sentence (e.g., "The dog ran fast.").
* **Behavior:** The system evaluates word-by-word tracking.
* **Assistance:** Clicking any individual word triggers VoxCPM to read just that word. Clicking a "Read to Me" button triggers VoxCPM to read the entire sentence with natural prosody.

## 3. Phonetic Evaluation Logic
Children have developing articulation. Standard ASR will often misinterpret their speech, and strict string matching can turn an acceptable read attempt into a discouraging failure. The target evaluation stack is therefore:

* **Exact Normalized Match:** Strip punctuation and casing. If the transcript exactly matches the target text after normalization, accept immediately.
* **Fine-Tuned MiniCPM Judge:** If the exact match fails, call the published MiniCPM evaluator (`kingkw1/minicpm-phonetic-evaluator`) with the target text and ASR transcript. The model returns `True` when the transcript is an acceptable phonetic match and `False` when the child likely said the wrong thing.
* **Legacy Fallback During Integration:** The current feature branch still contains earlier Levenshtein/alias matching in `app.py`. This is useful for local experimentation, but the product direction is to replace broad edit-distance acceptance with the tuned MiniCPM judge because examples like "the dogs ran fast" can otherwise be accepted too easily.
* **Decision Boundary:** The judge should accept child-speech variants such as lisps, slow articulation, or harmless ASR phrasing, while rejecting unrelated speech, skipped target content, or semantically different words.

## 4. Reward & Feedback Mechanism
The app must provide immediate, positive reinforcement without overwhelming the UI.

* **Success State:** * *Visual:* A hidden HTML `<div>` toggles to display a bright, animated star or confetti burst.
    * *Audio:* VoxCPM triggers a short, randomized positive affirmation ("Great job!", "You got it!", "Awesome!").
    * *Action:* Automatically loads the next target text after a 2-second delay.
* **Struggle State (3 Failed Attempts):**
    * If the evaluator rejects three consecutive attempts on the same target, the system gently intervenes to prevent frustration.
    * *Audio:* VoxCPM automatically reads the word out loud ("Let's try this together...").
* **On-Demand Assistance:**
    * Triggered exclusively by the child clicking the text block.
