  window.readAlongCleanWord = function(word) {
    return (word || '').toLowerCase().replace(/[^a-z0-9\\s]/g, '').replace(/\\s+/g, ' ').trim();
  };

  window.readAlongReadyAudio = function() {
    const target = document.querySelector('#tts-ready-audio textarea, #tts-ready-audio input');
    if (!target || !target.value) return {};
    try {
      return JSON.parse(target.value);
    } catch (_error) {
      return {};
    }
  };

  window.readAlongPlayCachedWord = function(word, fallbackText) {
    const readyAudio = window.readAlongReadyAudio();
    const audioUrl = readyAudio[word];
    if (!audioUrl) return false;

    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    if (window.readAlongWordAudio) {
      window.readAlongWordAudio.pause();
      window.readAlongWordAudio.src = '';
      window.readAlongWordAudio.currentTime = 0;
    }

    const audio = new Audio(audioUrl);
    window.readAlongWordAudio = audio;
    audio.play().catch(() => window.readAlongSpeakWithBrowser(fallbackText || word));
    return true;
  };

  window.readAlongSpeakWithBrowser = function(word) {
    const text = (word || '').trim();
    if (!text || !('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 0.82;
    utterance.pitch = 1.08;
    window.speechSynthesis.speak(utterance);
  };

  window.readAlongSpeakWord = function(word) {
    const text = (word || '').trim();
    if (!text) return;

    const cleanWord = window.readAlongCleanWord(text);
    if (window.readAlongPlayCachedWord(cleanWord, text)) {
      return;
    }

    window.readAlongSpeakWithBrowser(text);
  };

  window.addEventListener('load', () => {
    const armSuccessAdvance = () => {
      const feedback = document.querySelector('#feedback-display');
      if (!feedback || feedback.dataset.readAlongObserved === 'true') return;
      feedback.dataset.readAlongObserved = 'true';
      let timer = null;
      const syncFeedbackLayout = () => {
        feedback.classList.toggle('feedback-wrapper-hidden', Boolean(feedback.querySelector('.feedback-hidden')));
      };
      const observer = new MutationObserver(() => {
        syncFeedbackLayout();
        if (feedback.querySelector('.feedback-success')) {
          window.clearTimeout(timer);
          timer = window.setTimeout(() => {
            document.querySelector('#next-word-button button')?.click();
          }, 2500);
        }
      });
      observer.observe(feedback, { childList: true, subtree: true });
      syncFeedbackLayout();
    };
    armSuccessAdvance();
    window.setTimeout(armSuccessAdvance, 1000);
  });
