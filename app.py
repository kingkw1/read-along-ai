"""Phase 1 Gradio scaffold for Read-Along AI.

This file intentionally uses mock speech wrappers. The wrappers are the only
functions the UI calls for ASR/TTS so the implementation can be swapped for
Modal RPCs in a later phase without changing frontend event wiring.
"""

from __future__ import annotations

import html
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import gradio as gr

TARGET_SENTENCES = [
    "The dog ran fast.",
    "A cat sat on a mat.",
    "We can read and play.",
]

SAMPLE_RATE = 16_000
DUMMY_AUDIO_SECONDS = 1


def _write_silent_wav(label: str = "speech") -> str:
    """Create a short silent WAV file and return its local path."""
    safe_label = "".join(ch for ch in label.lower() if ch.isalnum() or ch in ("-", "_"))[:24] or "speech"
    output_path = Path(tempfile.gettempdir()) / f"read_along_{safe_label}.wav"
    frame_count = SAMPLE_RATE * DUMMY_AUDIO_SECONDS

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * frame_count)

    return str(output_path)


# ---------------------------------------------------------------------------
# Phase 1 mock Modal endpoint stubs.
# ---------------------------------------------------------------------------
def run_cohere_asr(audio_bytes: bytes) -> dict[str, str]:
    """Mock Modal ASR RPC stub for Phase 1."""
    del audio_bytes
    time.sleep(1)
    return {"text": "The dog ran fast.", "status": "success"}


def run_voxcpm_tts(text: str) -> bytes:
    """Mock Modal TTS RPC stub for Phase 1."""
    del text
    time.sleep(1)
    dummy_path = _write_silent_wav("tts")
    return Path(dummy_path).read_bytes()


# ---------------------------------------------------------------------------
# Backend abstraction layer required by docs/API_CONTRACT_SPEC.md.
# ---------------------------------------------------------------------------
def transcribe_audio(audio_filepath: str) -> str:
    """Return a clean transcription for a local microphone recording.

    Phase 1 is mocked: this waits one second and returns the fixed phrase
    required by the implementation brief.
    """
    del audio_filepath
    try:
        time.sleep(1)
        return "The dog ran fast."
    except Exception:
        return "[ASR_ERROR]"


def synthesize_speech(target_text: str) -> Optional[str]:
    """Return a local WAV path for generated speech.

    Phase 1 is mocked: this waits one second and returns a generated silent WAV.
    """
    try:
        time.sleep(1)
        return _write_silent_wav(target_text)
    except Exception:
        return None


def normalize_text(text: str) -> str:
    """Normalize spoken/target text for the simple Phase 1 evaluator."""
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).strip()


def render_reading_canvas(sentence: str) -> str:
    """Render target text as clickable HTML spans, not a Gradio textbox."""
    spans: list[str] = []
    for raw_word in sentence.split():
        clean_word = raw_word.strip(".,!?;:\"'")
        escaped_display = html.escape(raw_word)
        escaped_word = html.escape(clean_word, quote=True)
        spans.append(
            f'<span class="clickable-word" role="button" tabindex="0" '
            f'aria-label="Hear the word {escaped_word}" '
            f'onclick="readAlongSendWord(\'{escaped_word}\')" '
            f'onkeydown="if(event.key === \'Enter\' || event.key === \' \') readAlongSendWord(\'{escaped_word}\')">'
            f"{escaped_display}</span>"
        )

    return f"""
    <section class="reading-card" aria-label="Reading sentence">
        <div class="reading-helper">Tap a word if you need help ✨</div>
        <div class="reading-sentence">{' '.join(spans)}</div>
    </section>
    """


def loading_feedback() -> str:
    return """
    <div class="feedback-card feedback-loading" aria-live="polite">
        <div class="spinner-star">⭐</div>
        <div>Listening to your reading...</div>
    </div>
    """


def hidden_feedback() -> str:
    return '<div class="feedback-card feedback-hidden" aria-live="polite"></div>'


def success_feedback() -> str:
    return """
    <div class="feedback-card feedback-success" aria-live="polite">
        <div class="star-row"><span>🌟</span><span>🎉</span><span>🌟</span></div>
        <div class="feedback-title">Amazing reading!</div>
        <div class="feedback-subtitle">You read the sentence perfectly.</div>
    </div>
    """


def retry_feedback() -> str:
    return """
    <div class="feedback-card feedback-retry" aria-live="polite">
        <div class="feedback-title">Nice try!</div>
        <div class="feedback-subtitle">Try pressing record again!</div>
    </div>
    """


def evaluate_reading(audio_filepath: str, current_index: int) -> tuple[str, Optional[str]]:
    """Evaluate one read attempt and return feedback HTML plus optional praise audio."""
    transcript = transcribe_audio(audio_filepath)
    target_sentence = TARGET_SENTENCES[current_index]

    if transcript == "[ASR_ERROR]":
        return retry_feedback(), None

    if normalize_text(transcript) == normalize_text(target_sentence):
        praise_audio = synthesize_speech("Amazing reading!")
        return success_feedback(), praise_audio

    return retry_feedback(), None


def next_sentence(current_index: int) -> tuple[int, str, str]:
    """Advance to the next sentence and clear feedback."""
    next_index = (current_index + 1) % len(TARGET_SENTENCES)
    return next_index, render_reading_canvas(TARGET_SENTENCES[next_index]), hidden_feedback()


def listen_to_sentence(current_index: int) -> Optional[str]:
    return synthesize_speech(TARGET_SENTENCES[current_index])


def listen_to_word(word: str) -> Optional[str]:
    return synthesize_speech(word or "word")


CUSTOM_CSS = """
:root {
  --readalong-cream: #fff7df;
  --readalong-blue: #dff3ff;
  --readalong-navy: #12355b;
  --readalong-coral: #ff7a70;
  --readalong-yellow: #ffe873;
  --readalong-green: #58c98f;
}

footer, .api-docs, .built-with, .show-api, .gradio-container > .footer {
  display: none !important;
}

.gradio-container {
  background: radial-gradient(circle at top left, #fff1b8 0, transparent 30%),
    linear-gradient(135deg, var(--readalong-cream), var(--readalong-blue)) !important;
  color: var(--readalong-navy) !important;
  font-family: 'Nunito', 'Quicksand', 'Comic Sans MS', system-ui, sans-serif !important;
  min-height: 100vh;
  padding: 0 !important;
}

.main-container {
  max-width: 980px;
  margin: 0 auto !important;
  min-height: 100vh;
  padding: 2rem 1.25rem 3rem !important;
  gap: 1.4rem !important;
}

.app-title {
  text-align: center;
  font-size: clamp(2.2rem, 6vw, 4.5rem);
  font-weight: 900;
  letter-spacing: 0.02em;
  margin: 0.5rem 0 0;
  color: var(--readalong-navy);
  text-shadow: 0 4px 0 rgba(255, 255, 255, 0.9);
}

.reading-card {
  background: rgba(255, 255, 255, 0.78);
  border: 6px solid rgba(18, 53, 91, 0.14);
  border-radius: 42px;
  box-shadow: 0 20px 50px rgba(18, 53, 91, 0.16);
  padding: clamp(1.5rem, 4vw, 3rem);
  text-align: center;
}

.reading-helper {
  font-size: clamp(1.2rem, 3vw, 1.8rem);
  font-weight: 800;
  margin-bottom: 1rem;
  color: #38618c;
}

.reading-sentence {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  font-size: clamp(4rem, 11vw, 6rem);
  font-weight: 900;
  gap: 0.18em;
  justify-content: center;
  line-height: 1.15;
}

.clickable-word {
  border-radius: 0.35em;
  cursor: pointer;
  display: inline-block;
  padding: 0.02em 0.12em;
  transition: background-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}

.clickable-word:hover, .clickable-word:focus {
  background: var(--readalong-yellow);
  box-shadow: 0 0.12em 0 rgba(18, 53, 91, 0.2);
  outline: none;
  transform: translateY(-0.04em) rotate(-1deg);
}

.interaction-zone {
  background: rgba(255,255,255,0.45);
  border-radius: 36px;
  padding: 1rem;
}

#mic-capture {
  border: none !important;
  box-shadow: none !important;
}

#mic-capture .waveform,
#mic-capture canvas,
#mic-capture button[aria-label*='Edit'],
#mic-capture button[aria-label*='Trim'],
#mic-capture button[aria-label*='Download'] {
  display: none !important;
}

#mic-capture button,
#mic-capture .record-button {
  background: linear-gradient(135deg, var(--readalong-coral), #ffb067) !important;
  border: 7px solid white !important;
  border-radius: 999px !important;
  box-shadow: 0 14px 0 #c84d4b, 0 24px 40px rgba(18, 53, 91, 0.24) !important;
  color: white !important;
  font-size: clamp(1.7rem, 5vw, 3rem) !important;
  font-weight: 900 !important;
  min-height: 96px !important;
}

#mic-capture label span {
  font-size: clamp(1.5rem, 4vw, 2.4rem) !important;
  font-weight: 900 !important;
}

.feedback-card {
  border-radius: 34px;
  min-height: 122px;
  padding: 1.2rem;
  text-align: center;
}

.feedback-hidden { display: none; }
.feedback-loading { background: #fff5c7; font-size: 1.8rem; font-weight: 900; }
.feedback-success { background: #d9ffe9; border: 5px solid var(--readalong-green); }
.feedback-retry { background: #ffe4df; border: 5px solid var(--readalong-coral); }
.feedback-title { font-size: clamp(2rem, 5vw, 3rem); font-weight: 900; }
.feedback-subtitle { font-size: clamp(1.2rem, 3vw, 1.7rem); font-weight: 800; }
.star-row { font-size: clamp(2.2rem, 6vw, 4rem); }
.star-row span, .spinner-star { display: inline-block; animation: bounce-star 780ms infinite alternate ease-in-out; }
.star-row span:nth-child(2) { animation-delay: 120ms; }
.star-row span:nth-child(3) { animation-delay: 240ms; }
.spinner-star { font-size: 3rem; animation: spin-star 1s infinite linear; }

.control-button button {
  border-radius: 999px !important;
  font-size: clamp(1.3rem, 3vw, 2rem) !important;
  font-weight: 900 !important;
  min-height: 76px !important;
  box-shadow: 0 10px 0 rgba(18, 53, 91, 0.18) !important;
}

@keyframes bounce-star {
  from { transform: translateY(0) scale(1); }
  to { transform: translateY(-0.22em) scale(1.12); }
}

@keyframes spin-star {
  from { transform: rotate(0deg) scale(1); }
  to { transform: rotate(360deg) scale(1.05); }
}
"""


FRONTEND_JS = """
<script>
  window.readAlongSendWord = function(word) {
    const target = document.querySelector('#word-click-target textarea');
    if (target) {
      target.value = word;
      target.dispatchEvent(new Event('input', { bubbles: true }));
    }
    const button = document.querySelector('#word-click-submit button');
    if (button) button.click();
  };

  window.addEventListener('load', () => {
    const armSuccessAdvance = () => {
      const feedback = document.querySelector('#feedback-display');
      if (!feedback || feedback.dataset.readAlongObserved === 'true') return;
      feedback.dataset.readAlongObserved = 'true';
      let timer = null;
      const observer = new MutationObserver(() => {
        if (feedback.querySelector('.feedback-success')) {
          window.clearTimeout(timer);
          timer = window.setTimeout(() => {
            document.querySelector('#next-word-button button')?.click();
          }, 2500);
        }
      });
      observer.observe(feedback, { childList: true, subtree: true });
    };
    armSuccessAdvance();
    window.setTimeout(armSuccessAdvance, 1000);
  });
</script>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Read-Along AI", head=FRONTEND_JS) as demo:
        sentence_index = gr.State(0)

        gr.HTML('<h1 class="app-title">Read-Along AI</h1>')
        with gr.Column(elem_classes="main-container"):
            reading_canvas = gr.HTML(render_reading_canvas(TARGET_SENTENCES[0]))

            with gr.Column(elem_classes="interaction-zone"):
                microphone = gr.Audio(
                    label="🎙️ Press and read out loud",
                    sources=["microphone"],
                    type="filepath",
                    elem_id="mic-capture",
                )

            feedback_display = gr.HTML(hidden_feedback(), elem_id="feedback-display")
            speech_output = gr.Audio(label="Read-Along voice", autoplay=True, visible=False)

            with gr.Row():
                next_button = gr.Button("Next Word ➜", elem_classes="control-button", elem_id="next-word-button", variant="secondary")
                listen_button = gr.Button("🔊 Listen to Sentence", elem_classes="control-button", variant="primary")

            word_click_target = gr.Textbox(visible=False, elem_id="word-click-target")
            word_click_submit = gr.Button(visible=False, elem_id="word-click-submit")

        microphone.change(
            fn=loading_feedback,
            inputs=None,
            outputs=feedback_display,
            show_progress="hidden",
        ).then(
            fn=evaluate_reading,
            inputs=[microphone, sentence_index],
            outputs=[feedback_display, speech_output],
        )

        next_button.click(
            fn=next_sentence,
            inputs=sentence_index,
            outputs=[sentence_index, reading_canvas, feedback_display],
        )

        listen_button.click(
            fn=listen_to_sentence,
            inputs=sentence_index,
            outputs=speech_output,
        )

        word_click_submit.click(
            fn=listen_to_word,
            inputs=word_click_target,
            outputs=speech_output,
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
