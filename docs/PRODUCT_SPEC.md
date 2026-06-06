# Read-Along AI: Product Specification

## 1. Core User Flow
The application provides a guided, distraction-free reading experience that scales in complexity. The central loop is:
1. **Display:** The UI presents a target text (a letter, word, or sentence).
2. **Listen:** The child presses "Record" and attempts to read the target text.
3. **Evaluate (ASR):** The Cohere Transcribe model converts the audio to text.
4. **Match:** The system applies fuzzy-matching logic to compare the ASR output against the target text.
5. **Feedback:** The system triggers a visual reward and proceeds, or uses OpenBMB VoxCPM to provide gentle verbal assistance.

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

## 3. Fuzzy Matching & Evaluation Logic
Children have developing articulation. Standard ASR will often misinterpret their speech. The evaluation logic must be highly tolerant to prevent frustration.

* **Phonetic Aliasing (Level 1 & 2):** Create a dictionary of acceptable ASR misinterpretations for early sounds.
    * *Example:* Target "C". Acceptable ASR returns: "see", "sea", "si", "c".
    * *Example:* Target "M". Acceptable ASR returns: "em", "am", "um", "m".
* **Levenshtein Distance (Level 2 & 3):** Apply a string-matching algorithm with a tolerance threshold.
    * For a 3-letter target word, accept an ASR output with a Levenshtein distance of 1 (e.g., Target: "cat", ASR output: "cap" -> Accept as correct to maintain flow).
* **Filler Word Stripping:** Automatically strip common hesitations from the ASR string before evaluation (remove "um", "uh", "like", "the").
* **Substring Matching (Level 3):** If the target sentence is "The dog ran fast," and the ASR outputs "dog ran fast," mark the attempt as successful. Do not penalize skipped sight words if the core nouns/verbs are captured.

## 4. Reward & Feedback Mechanism
The app must provide immediate, positive reinforcement without overwhelming the UI.

* **Success State:** * *Visual:* A hidden HTML `<div>` toggles to display a bright, animated star or confetti burst.
    * *Audio:* VoxCPM triggers a short, randomized positive affirmation ("Great job!", "You got it!", "Awesome!").
    * *Action:* Automatically loads the next target text after a 2-second delay.
* **Struggle State (3 Failed Attempts):**
    * If the fuzzy-match fails three consecutive times on the same target, the system gently intervenes to prevent frustration.
    * *Audio:* VoxCPM automatically reads the word out loud ("Let's try this together...").
* **On-Demand Assistance:**
    * Triggered exclusively by the child clicking the text block.