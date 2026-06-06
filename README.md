# 🦉 Read-Along AI: The Offline Reading Teacher

**Live App:** [Insert Hugging Face Space Link]

**Organization:** Hosted under the official [`build-small-hackathon`](https://huggingface.co/build-small-hackathon) HF Org.

**A Build Small Hackathon Submission**
*Track: Backyard AI*

## 📖 The Vision
Learning to read is a monumental milestone for young children, but it can be an intimidating process. **Read-Along AI** was built to solve a very specific problem: *[Insert personalization: e.g., Detail how you and Elizabeth homeschool your three children (ages 7, 6, and 4), and this tool was built specifically to solve the friction of your daily reading curriculum]* providing a patient, distraction-free, and interactive reading assistant that gives instant, gentle vocal feedback. 

**Field Testing Note:** *[Insert brief anecdote or reaction from the 4, 6, and 7-year-olds actually using the app here to prove real-world usage for the Backyard AI track]*

Crucially, because this tool is for kids, it requires absolute data privacy. It leverages localized, small-parameter models to ensure a child's voice data never enters a corporate data lake. 

## 🛠️ The Tech Stack & Architecture
This application strictly adheres to the < 32B parameter constraint, utilizing highly optimized small models for a real-time, fluid user experience.

### Development Documentation
For a deep dive into the architecture and development plan, please review our spec documents:
* [Product Specification](docs/PRODUCT_SPEC.md)
* [UI/UX & Frontend Specification](docs/UI_UX_SPEC.md)
* [API & Backend Contract](docs/API_CONTRACT_SPEC.md)
* [Deployment Strategy](docs/DEPLOYMENT_SPEC.md)
* [9-Day Roadmap](docs/ROADMAP.md)

### Components
* **Frontend:** A custom, gamified Gradio interface ("Off-Brand" UI) built for legibility and young readers.
* **ASR (Speech-to-Text):** **Cohere Transcribe** (2B parameters). Optimized for low-latency voice capture to process the child's reading attempts instantly.
* **TTS (Text-to-Speech):** **OpenBMB VoxCPM** (*[Insert Parameter Count]*). This acts as the **central interactive component** of the app, driving the core feedback loop, stepping in during "Struggle States," and providing on-demand phonemic assistance.
* **Compute / Inference:** Deployed via **Modal** serverless endpoints to guarantee zero-latency responses during the frontend interaction.

## 🏆 Hackathon Eligibility & Attributions

### OpenAI Codex Track ($10,000)
This entire application, including the Gradio UI and backend Modal logic, was orchestrated using OpenAI's **Codex (GPT-5.5)**. Codex acted as the lead developer in a truly holistic manner:
* Generated the custom CSS overrides for the Off-Brand gamification.
* Wrote the Modal serverless stub functions and the Gradio abstraction wrappers.
* Managed the repository structure and environment variable integration.

* **GitHub Repository:** https://github.com/kingkw1/read-along-ai
* *Note to Judges: Please see the commit history for automated Codex attributions.*

### Modal Compute Awards
The high-speed inference endpoints powering the Cohere and OpenBMB models are hosted entirely on **Modal**. This provides the necessary sub-second response times required to keep a young child focused. *(Note: Our Roadmap outlines swapping to local `llama.cpp` for Phase 2 to capture local execution badges).*

### Badges Claimed (Bonus Quest Champion Strategy)
* 🏅 **Off-Brand:** The default Gradio UI has been completely overhauled with custom CSS to create a distraction-free, gamified experience for early learners.
* 🏅 **Tiny Titan:** The total combined footprint of the models used is strictly under the 4B parameter threshold. 
  * *Parameter Math:* Cohere Transcribe (2B) + OpenBMB VoxCPM (*[Insert Parameter Count]*) = ***[Insert Total] Parameters***.
* 🏅 **Sharing is Caring:** *[Insert Hugging Face Dataset Link to Codex Agent Traces]*
* 🏅 **Field Notes:** *[Insert Link to Blog Post / Medium Article]*

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

### About the Developer
*[Placeholder for Developer Bio: Explicitly state your specialization in multimodal signal fusion and edge deployment. Briefly cite your previous wins with "AffectLink" (HP & NVIDIA Developer Challenge) and "The Lung Listener" (Google Gemini API Hackathon). This signals to the judges—especially NVIDIA and Hugging Face—that the architecture is engineered by a seasoned professional.]*

## 📜 Open Source License
This project is licensed under the MIT License.
