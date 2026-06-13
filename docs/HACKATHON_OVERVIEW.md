# Hugging Face Build Small Hackathon: Consolidated Guide

## Overview
The **Build Small Hackathon** is a Hugging Face and Gradio community event centered on building practical or playful AI applications with **small models** (≤ 32 billion parameters). The hackathon emphasizes returning to tinkerable, fun, and offline-friendly AI. Submissions are hosted as Hugging Face Spaces under the official `build-small-hackathon` organization.

## Timeline
- **May 7 – June 3, 2026:** Registration / sign-up window.
- **June 5 – 15, 2026:** Hack window.
- **June 9, 10, 11, 2026:** AMAs and Live Sessions on Discord.
- **June 15, 2026:** Submissions close.
- **Date TBD:** Winners announced.

## Core Rules & Constraints
1. **Small Models Only:** Every model your project depends on must individually be under **32 billion** total parameters. You can combine multiple models (e.g., a 14B LLM and a 7B ASR model) as long as *each individual model* stays under the 32B cap.
2. **Built on Gradio:** The submission must be a Gradio app hosted as a Hugging Face Space under the hackathon organization.
3. **Show, Don’t Tell:** A short **demo video** and a **social-media post** are required. Links should be added to your Space's README.
4. **README Requirements:** Your Space `README.md` MUST include frontmatter tags for your chosen track and target badges, along with a short technical write-up detailing model sizes and sponsor-tool usage notes.

### Eligibility Rules
- Join the org on Hugging Face & Register on the official app. Both are required.
- Build your Gradio app as a Space under the org — only Spaces inside the org count.
- Solo or team — both welcome. Every teammate must register and join the org separately, and list all HF usernames in your Space README.

## Main Tracks & Judging Criteria
There are two main tracks. You can participate in either to build your application:

1. **Backyard AI:** Build small AI tools that solve a real problem for someone you actually know (e.g., a neighbor, parent, or small-business owner). 
   - **Judging Criteria:** Emphasizes addressing a specific real problem, actual use by the intended person, fit with the small-model constraint, and the polish of the Gradio app.
2. **Adventure in Thousand Token Wood:** Build creative, playful, strange, or whimsical AI experiences that would not exist without AI. 
   - **Judging Criteria:** Emphasizes delight, AI as a load-bearing part of the experience, originality, and the polish of the Gradio app.

## Bonus Quests (Merit Badges)
You can collect "badges" to add extra points to your submission and qualify for specific awards:
- **Off the Grid:** No cloud APIs; the app runs locally on the model.
- **Well-Tuned:** The app uses a fine-tuned model published on Hugging Face.
- **Off-Brand:** A custom frontend that pushes beyond the default Gradio look.
- **Llama Champion:** The model runs through `llama.cpp`.
- **Sharing is Caring:** The agent trace is shared on the Hub.
- **Field Notes:** A blog post or report explains what was built and learned.

## Sponsors, Credits & Recommended Models

Participants get credits to help them build:
- **Hugging Face:** As a member of the hackathon organization, participants receive 40 minutes per day of ZeroGPU access for free without needing a PRO subscription. Past the daily free quota, ZeroGPU runs on a pay-as-you-go model at $1 per 10 minutes, which deducts from the provided $20 in Hugging Face credits. Participants can build up to 10 ZeroGPU apps. Also introduced the new **Gradio Workflow** (no-code canvas).
- **Modal:** $250 in credits for all participants to run inference, fine-tuning, or coding agents.
- **OpenAI:** $100 credits for the first 1,000 registrations.

### Sponsor Models
- **Black Forest Labs (BFL):** Sponsored $5k. Recommends **Flux 2** (4B and 9B) for image generation/editing.
- **OpenBMB:** Recommends the **MiniCPM** family: MiniCPM 51B, MiniCPM 4.1 8B (text/reasoning), MiniCPM-V 4.6 (vision), MiniCPM-o 4.5 (omni), and VoxCPM (TTS).
- **OpenAI:** Supplying **Codex** (GPT-5.5) for orchestrating agents and coding tools. Goal Mode can help you build end-to-end. OpenAI will be evaluating submissions based on Codex-attributed commits.
- **NVIDIA:** Showcased the **Nemotron** models, including Nemotron 3 Nano (30B total, 3B active MoE), Nano Omni, Cascade (math/code), Speech models, Parse (document extraction), and Embedding models.
- **JetBrains:** Sponsored $5k. Recommends **Milum 2**, a 12B parameter MoE model optimized for code and high throughput. 
- **Cohere Labs:** Sponsored $5k. Recommends **Cohere Transcribe** (2B parameter ASR) and their 3.3B multilingual LLMs (Base, Global, Earth, Fire, Water versions) optimized for low latency and language translation.

## Prizes & Awards
The hackathon offers ~$48,000 in cash prizes, $20,000 in Modal credits, 2 RTX 5080 GPUs, and 1 year of GPT Pro. Submissions can win in multiple categories (up to 29 ways to win).

### Main Track Awards ($22,000 Total across both tracks)
*These prizes are awarded independently in EACH track (Backyard AI & Thousand Token Wood):*
- **1st Place:** $4,000
- **2nd Place:** $2,500
- **3rd Place:** $1,500
- **4th Place:** $1,000
- **Community Choice:** $2,000

### Sponsor Awards & Specific Rules
- **OpenBMB Awards ($10,000):** Top MiniCPM apps per track (1st: $2,500, 2nd: $1,500, 3rd: $1,000). This pool is split $5,000 to Backyard AI and $5,000 to Thousand Token Wood. Vision and Omni variants also qualify.
  - **Rule:** The MiniCPM model must be used as a central part of the application.
- **OpenAI Track ($10,000 total across all submissions):** A dedicated prize track with $10,000 in cash for the top 3 builds (1st: $5,000, 2nd: $3,000, 3rd: $1,000). 
  - **Rule:** You must mention a public GitHub repository in your Space's README containing Codex-attributed commits. Using Codex holistically (fine-tuning, complex agents) ranks higher than light use.
- **NVIDIA Nemotron Quest:** 2x **RTX 5080 GPUs** for standout Nemotron builds. 
  - **Rule:** Must build using the Nemotron models. One GPU is awarded for "Best space" (judged by NVIDIA); the second is awarded for "Community engagement" (judged by likes & interactions).
- **Modal Awards ($20,000 in credits):** Top Modal-powered apps (1st: $10,000 credits, 2nd: $7,000 credits, 3rd: $3,000 credits). 
  - **Rule:** You must specifically mention your use of Modal in your Space's README to be eligible.

### Special Awards ($8,000 Total)
- **Bonus Quest Champion ($2,000):** The most bonus criteria met. Ties go to the most ambitious, high-quality submission despite constraints.
- **Off-Brand Award ($1,500):** The best custom UI pushing past default Gradio (use `gr.Server` to go beyond stock components).
- **Tiny Titan ($1,500):** Best app built on a genuinely tiny model (each model used is ≤ 4B parameters). Biggest impact from smallest weights wins.
- **Best Demo ($1,000):** The full package: great app, demo video, and social post. Storytelling counts just as much as the build.
- **Best Agent ($1,000):** The best agentic app under the 32B cap, emphasizing multi-step tool use and planning.
- **Judges’ Wildcard ($1,000):** For an amazing entry that fits no specific category.
