"""Phase 1 Gradio scaffold for Read-Along AI.

The wrappers are the only functions the UI calls for ASR/TTS so the
implementation can be swapped without changing frontend event wiring.
"""

from __future__ import annotations

import html
import inspect
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import gradio as gr
import modal

from local_inference import local_ask_minicpm_judge, local_synthesize_speech, local_transcribe_audio

MODAL_APP_NAME = "read-along-ai-inference"
TURBO_ENGINE = "⚡ Turbo Mode (Modal)"
LOCAL_ENGINE = "🏕️ Off the Grid Mode (Local)"
INFERENCE_ENGINES = [TURBO_ENGINE, LOCAL_ENGINE]

CURRICULUM = ["The cat sat.", "The dog ran fast.", "She had a red hat.", "I love to play outside."]
TTS_MEMORY_CACHE: dict[str, bytes] = {}
TTS_CACHE_LOCK = threading.Lock()

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


def _modal_function(function_name: str):
    lookup = getattr(modal.Function, "lookup", None)
    if lookup is not None:
        return lookup(MODAL_APP_NAME, function_name)
    return modal.Function.from_name(MODAL_APP_NAME, function_name)


def run_cohere_asr(audio_bytes: bytes) -> dict[str, str]:
    """Invoke the deployed Modal ASR endpoint."""
    return _modal_function("run_cohere_asr").remote(audio_bytes)


def run_voxcpm_tts(text: str) -> bytes:
    """Invoke the deployed Modal TTS endpoint."""
    return _modal_function("run_voxcpm_tts").remote(text)


def run_minicpm_evaluator(target_text: str, transcript: str) -> str:
    """Invoke the deployed Modal MiniCPM phonetic evaluator endpoint."""
    return _modal_function("run_minicpm_evaluator").remote(target_text, transcript)


# ---------------------------------------------------------------------------
# Backend abstraction layer required by docs/API_CONTRACT_SPEC.md.
# ---------------------------------------------------------------------------
def transcribe_audio(audio_filepath: str, inference_engine: str = TURBO_ENGINE) -> str:
    """Return a clean transcription for a local microphone recording."""
    try:
        if inference_engine == LOCAL_ENGINE:
            return local_transcribe_audio(audio_filepath)

        audio_bytes = Path(audio_filepath).read_bytes()
        result = run_cohere_asr(audio_bytes)
        if result.get("status") != "success":
            return "[ASR_ERROR]"
        return normalize_text(result.get("text", ""))
    except Exception:
        return "[ASR_ERROR]"


def synthesize_speech(target_text: str, inference_engine: str = TURBO_ENGINE) -> Optional[str]:
    """Return a local WAV path for generated speech."""
    try:
        audio_bytes = synthesize_speech_bytes(target_text, inference_engine)
        return write_tts_audio_file(target_text, audio_bytes)
    except Exception:
        return None


def safe_tts_label(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in ("-", "_"))[:24] or "speech"


def write_tts_audio_file(label: str, audio_bytes: bytes) -> str:
    output_path = Path(tempfile.gettempdir()) / f"read_along_{safe_tts_label(label)}.wav"
    output_path.write_bytes(audio_bytes)
    return str(output_path)


def synthesize_speech_bytes(target_text: str, inference_engine: str = TURBO_ENGINE) -> bytes:
    """Return generated speech as WAV bytes for in-memory caching."""
    if inference_engine == LOCAL_ENGINE:
        return Path(local_synthesize_speech(target_text)).read_bytes()
    return run_voxcpm_tts(target_text)


def normalize_text(text: str) -> str:
    """Normalize spoken/target text for tolerant reading evaluation."""
    return re.sub(r"\s+", " ", "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())).strip()


def clean_tts_word(word: str) -> str:
    """Normalize a single TTS helper word for cache lookup."""
    return normalize_text(word)


def sentence_tts_words(sentence: str) -> list[str]:
    """Return unique cleaned words in reading order for TTS prewarming."""
    words: list[str] = []
    seen: set[str] = set()
    for raw_word in sentence.split():
        word = clean_tts_word(raw_word)
        if word and word not in seen:
            words.append(word)
            seen.add(word)
    return words


def prewarm_level_words(sentence: str, engine_mode: str) -> None:
    """Generate per-word TTS audio in the background and keep it in memory."""
    for word in sentence_tts_words(sentence):
        with TTS_CACHE_LOCK:
            if word in TTS_MEMORY_CACHE:
                continue

        try:
            audio_bytes = synthesize_speech_bytes(word, engine_mode)
        except Exception:
            continue

        with TTS_CACHE_LOCK:
            TTS_MEMORY_CACHE.setdefault(word, audio_bytes)


def start_prewarm_level_words(sentence: str, engine_mode: str) -> None:
    threading.Thread(target=prewarm_level_words, args=(sentence, engine_mode), daemon=True).start()


def ask_minicpm_judge(target_text: str, transcript: str, inference_engine: str = TURBO_ENGINE) -> bool:
    """Ask the fine-tuned MiniCPM evaluator whether the reading is acceptable."""
    try:
        if inference_engine == LOCAL_ENGINE:
            return local_ask_minicpm_judge(target_text, transcript)

        verdict = str(run_minicpm_evaluator(target_text, transcript)).strip().casefold()
    except Exception:
        return False
    return verdict == "true"


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
            f'onclick="readAlongSpeakWord(\'{escaped_word}\')" '
            f'onkeydown="if(event.key === \'Enter\' || event.key === \' \') {{ event.preventDefault(); readAlongSpeakWord(\'{escaped_word}\'); }}">'
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


def _call_with_engine(function, *args, inference_engine: str):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, inference_engine)

    accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
    if accepts_varargs or len(signature.parameters) > len(args):
        return function(*args, inference_engine)
    return function(*args)


def evaluate_reading(audio_filepath: str, current_index: int, inference_engine: str = TURBO_ENGINE) -> tuple[str, Optional[str]]:
    """Evaluate one read attempt against the active curriculum sentence."""
    transcript = _call_with_engine(transcribe_audio, audio_filepath, inference_engine=inference_engine)
    target_sentence = CURRICULUM[int(current_index) % len(CURRICULUM)]
    print(
        f"[read-along] engine={inference_engine!r} target={target_sentence!r} transcript={transcript!r}",
        flush=True,
    )

    if transcript == "[ASR_ERROR]":
        return retry_feedback(), None

    exact_match = normalize_text(transcript) == normalize_text(target_sentence)
    if exact_match or _call_with_engine(ask_minicpm_judge, target_sentence, transcript, inference_engine=inference_engine):
        return success_feedback(), None

    return retry_feedback(), None


def prewarm_current_level(current_index: int, inference_engine: str = TURBO_ENGINE) -> None:
    sentence = CURRICULUM[int(current_index) % len(CURRICULUM)]
    start_prewarm_level_words(sentence, inference_engine)


def next_sentence(idx: int, inference_engine: str = TURBO_ENGINE) -> tuple[int, str, None, str, None, None]:
    """Advance to the next curriculum sentence and clear transient outputs."""
    next_index = (int(idx) + 1) % len(CURRICULUM)
    next_level_sentence = CURRICULUM[next_index]
    return next_index, render_reading_canvas(next_level_sentence), None, hidden_feedback(), None, None


def listen_to_sentence(current_index: int, inference_engine: str = TURBO_ENGINE) -> Optional[str]:
    return synthesize_speech(CURRICULUM[int(current_index) % len(CURRICULUM)], inference_engine)


def update_audio_help(clicked_word: str, inference_engine: str = TURBO_ENGINE) -> Optional[bytes]:
    """Generate audio help for a clicked reading word.

    The reading canvas sends the cleaned word text through a hidden Gradio
    textbox/button bridge because the words are rendered as accessible HTML
    controls. Returning cached WAV bytes to the autoplay Audio component lets
    the browser play the helper audio without waiting for TTS generation.
    """
    word = clean_tts_word(clicked_word or "")
    if not word:
        return None

    with TTS_CACHE_LOCK:
        cached_audio = TTS_MEMORY_CACHE.get(word)
    if cached_audio is not None:
        return cached_audio

    try:
        audio_bytes = synthesize_speech_bytes(word, inference_engine)
    except Exception:
        return None

    with TTS_CACHE_LOCK:
        TTS_MEMORY_CACHE[word] = audio_bytes
    return audio_bytes


def word_click_loading_state(clicked_word: str) -> gr.update:
    word = clean_tts_word(clicked_word or "")
    if not word:
        return gr.update(value="Word helper")

    with TTS_CACHE_LOCK:
        is_cached = word in TTS_MEMORY_CACHE
    return gr.update(value=word if is_cached else "🪄...")


def finish_word_click(clicked_word: str, inference_engine: str = TURBO_ENGINE) -> tuple[Optional[str], gr.update]:
    word = clean_tts_word(clicked_word or "")
    audio_bytes = update_audio_help(word, inference_engine)
    audio_path = write_tts_audio_file(word, audio_bytes) if audio_bytes is not None else None
    return audio_path, gr.update(value=word or "Word helper")


def listen_to_word(word: str, inference_engine: str = TURBO_ENGINE) -> Optional[bytes]:
    """Backward-compatible alias for word-level audio help."""
    return update_audio_help(word, inference_engine)


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
  window.readAlongSpeakWord = function(word) {
    const text = (word || '').trim();
    if (!text || !('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 0.82;
    utterance.pitch = 1.08;
    window.speechSynthesis.speak(utterance);
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
    with gr.Blocks(title="Read-Along AI") as demo:
        current_index = gr.State(0)

        gr.HTML('<h1 class="app-title">Read-Along AI</h1>')
        with gr.Column(elem_classes="main-container"):
            inference_engine = gr.Radio(
                choices=INFERENCE_ENGINES,
                value=TURBO_ENGINE,
                label="Inference Engine",
                elem_classes="engine-toggle",
            )
            reading_canvas = gr.HTML(render_reading_canvas(CURRICULUM[0]))

            with gr.Column(elem_classes="interaction-zone"):
                microphone = gr.Audio(
                    label="🎙️ Press and read out loud",
                    sources=["microphone"],
                    type="filepath",
                    elem_id="mic-capture",
                )

            feedback_display = gr.HTML(hidden_feedback(), elem_id="feedback-display")
            speech_output = gr.Audio(
                label="Read-Along voice",
                autoplay=True,
                visible="hidden",
                elem_id="speech-output",
            )
            word_help_output = gr.Audio(
                label="Word helper voice",
                autoplay=True,
                visible="hidden",
                elem_id="word-help-output",
            )

            with gr.Row():
                next_button = gr.Button("Next Level ➡️", elem_classes="control-button", elem_id="next-word-button", variant="secondary")
                listen_button = gr.Button("🔊 Listen to Sentence", elem_classes="control-button", variant="primary")

        microphone.change(
            fn=loading_feedback,
            inputs=None,
            outputs=feedback_display,
            show_progress="hidden",
        ).then(
            fn=evaluate_reading,
            inputs=[microphone, current_index, inference_engine],
            outputs=[feedback_display, speech_output],
        )

        next_button.click(
            fn=next_sentence,
            inputs=[current_index, inference_engine],
            outputs=[current_index, reading_canvas, microphone, feedback_display, speech_output, word_help_output],
        )

        listen_button.click(
            fn=listen_to_sentence,
            inputs=[current_index, inference_engine],
            outputs=speech_output,
        )
    return demo


if __name__ == "__main__":
    build_app().launch(css=CUSTOM_CSS, head=FRONTEND_JS)
