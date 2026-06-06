# 🦉 Read-Along AI: The Offline Reading Teacher

**Live App:** [Insert Hugging Face Space Link]
**Organization:** Hosted under the official [`build-small-hackathon`](https://huggingface.co/build-small-hackathon) HF Org.

**A Build Small Hackathon Submission**
*Track: Backyard AI*

## 📖 The Vision
Learning to read is a monumental milestone for young children, but it can be an intimidating process. **Read-Along AI** was built to solve a very specific problem for a household with children aged 4, 6, and 7: providing a patient, distraction-free, and interactive reading assistant that gives instant, gentle vocal feedback. 

Crucially, because this tool is for kids, it requires absolute data privacy. It leverages localized, small-parameter models to ensure a child's voice data never enters a corporate data lake. 

## 🛠️ The Tech Stack & Architecture
This application strictly adheres to the < 32B parameter constraint, utilizing highly optimized small models for a real-time, fluid user experience.

* **Frontend:** A custom, gamified Gradio interface ("Off-Brand" UI) built for legibility and young readers.
* **ASR (Speech-to-Text):** **Cohere Transcribe** (2B parameters). Optimized for low-latency voice capture to process the child's reading attempts instantly.
* **TTS (Text-to-Speech):** **OpenBMB VoxCPM**. Serves as the central component for the app's interactive feedback, providing full-sentence read-backs and on-demand word pronunciation.
* **Compute / Inference:** Deployed via **Modal** serverless endpoints to guarantee zero-latency responses during the frontend interaction.

## 🏆 Hackathon Eligibility & Attributions

### OpenAI Codex Track ($10,000)
This entire application, including the Gradio UI and backend Modal logic, was orchestrated using OpenAI's **Codex (GPT-5.5)**.
* **GitHub Repository:** https://github.com/kingkw1/read-along-ai
* *Note to Judges: Please see the commit history for automated Codex attributions.*

### Modal Compute Awards
The high-speed inference endpoints powering the Cohere and OpenBMB models are hosted entirely on **Modal**. This provides the necessary sub-second response times required to keep a young child focused.

### Badges Claimed
* 🏅 **Off-Brand:** The default Gradio UI has been completely overhauled with custom CSS to create a distraction-free, gamified experience for early learners.
* 🏅 **Tiny Titan:** The core speech-recognition engine relies on Cohere Transcribe, well under the 4B parameter threshold.

## 🚀 How It Works
The app scales with the developmental stages of early readers:
1. **Level 1 (Phonics):** Focuses on single letters and their phonetic sounds.
2. **Level 2 (CVC Words):** Short Consonant-Vowel-Consonant words with fuzzy-matching for developing speech patterns.
3. **Level 3 (Sentences):** Full sentence reading with interactive word-click pronunciation and a final TTS read-back.

## 📹 Submission Links
* **Demo Video:** [Insert Video Link]
* **Social Post:** [Insert X/LinkedIn Link]

## 👥 Team
* Hugging Face Username: `kingkw1`
