# Read-Along AI: UI/UX & Frontend Specification

## 1. Design Philosophy & Objective
The primary objective of this frontend is to provide a zero-distraction, highly accessible interface for children ages 4 to 7. Standard Gradio interfaces are built for data scientists and are too cluttered for early learners. 

To qualify for the **Off-Brand Award**, this application must aggressively overwrite default Gradio styling using the `css` parameter in `gr.Blocks` and custom `gr.HTML` components. 

## 2. Global Styling Rules (Custom CSS)
*Note: As per the Roadmap, aggressive CSS gamification and styling polish should be deferred to Phase 3 (Day 7) after the core Phase 1 backend event loop is proven.*

Codex must inject the following CSS rules to override the default Gradio theme:
* **Typography:** Use a highly legible, rounded sans-serif font (e.g., `'Nunito', 'Quicksand', 'Comic Sans MS', sans-serif`). 
* **Sizing:** Text must be massive. Base font size for the target reading block should be at least `4rem` to `6rem`.
* **Colors:** Use high-contrast, soft, friendly colors. Avoid stark white backgrounds (use a soft cream or pastel blue: e.g., `#F8F9FA` or `#E3F2FD`). 
* **Chrome Removal:** Hide the default Gradio footer, "Built with Gradio" badges, and unnecessary padding around the main container.

## 3. Component Layout & Structure
The app should utilize a single-column, centered layout (`gr.Column(elem_classes="main-container")`). 

### Header: The Architecture Toggle
* **Mode Switch:** A small, unobtrusive `gr.Radio` or `gr.Dropdown` at the top allowing the user to select between `⚡ Turbo Mode (Modal)` and `🏕️ Off the Grid Mode (Local)`. This controls whether the backend endpoints execute locally or in the cloud.

### Top: The Reading Canvas (`gr.HTML`)
* **Do NOT use `gr.Textbox`** for the target words/sentences. 
* To allow the child to click individual words for audio assistance, the text must be rendered dynamically via `gr.HTML`.
* Every word in a sentence must be wrapped in a clickable `<span>` tag with a specific CSS class (e.g., `<span class="clickable-word" onclick="...">Word</span>`).
* Hovering over a word should highlight it (e.g., change background to soft yellow) to indicate it is interactive.

### Middle: The Interaction Zone
* **Record Button:** Use `gr.Audio(sources=["microphone"], type="filepath")`. 
* **CSS Override:** The default Gradio audio waveform UI is too complex. Use CSS to hide the waveform and editing tools. Style the wrapper to look like a single, massive, colorful "Microphone/Record" button with heavy border-radius (pill or circle shape) and a box-shadow.

### Bottom: The Reward & Control Container
* **Feedback Display (`gr.HTML`):** A dedicated, hidden container used exclusively for rendering success animations (e.g., CSS keyframe bouncing stars, confetti emojis, or a smiling mascot).
* **Navigation Controls:** Two simple, oversized buttons: "Next Word" and "Listen to Sentence" (triggers the full sentence TTS). 

## 4. Frontend-to-Backend Event Mapping
Codex must wire the Gradio events as follows:
* **Event 1 (The Read Attempt):** When the `gr.Audio` component finishes recording (`.stop_recording()` or `.change()`), it immediately triggers the Python evaluation function (which calls the Modal ASR endpoint). The UI should display a simple, child-friendly loading state (like a spinning star) during inference.
* **Event 2 (Word Click Assist):** Clicking a `<span>` in the Reading Canvas triggers a Gradio custom JS event that passes the specific word string back to the Python backend to call the Modal TTS endpoint.
* **Event 3 (Success State):** If the evaluation function returns `True`, update the Feedback Display `gr.HTML` to show the success animation, play the TTS praise, and auto-load the next level payload after 2.5 seconds.