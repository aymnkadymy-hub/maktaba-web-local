// ─── Text-to-Speech (TTS) ──────────────────────────────────────────
let _ttsEnabled = false;
let _ttsUtter   = null;
let _ttsLang    = 'ar';   // resolved at runtime — ar-IQ if supported, else ar

function _resolveTTSLang() {
  if (!window.speechSynthesis) return;
  const voices = speechSynthesis.getVoices();
  if (!voices.length) return;   // not loaded yet — will retry via onvoiceschanged
  if (voices.some(v => v.lang === 'ar-IQ')) {
    _ttsLang = 'ar-IQ';
  } else {
    const arVoice = voices.find(v => v.lang.startsWith('ar'));
    if (arVoice) _ttsLang = arVoice.lang;
  }
}

// Voices load async in Chrome — resolve on change, also try immediately (Firefox)
if (window.speechSynthesis) {
  speechSynthesis.onvoiceschanged = _resolveTTSLang;
  _resolveTTSLang();
}

function toggleTTS() {
  _ttsEnabled = !_ttsEnabled;
  const btn = document.getElementById('tts-btn');
  btn.textContent = _ttsEnabled ? '🔊' : '🔇';
  btn.title       = _ttsEnabled ? 'إيقاف صوت البوت' : 'تشغيل صوت البوت';
  if (!_ttsEnabled && speechSynthesis.speaking) speechSynthesis.cancel();
}

function speakText(text) {
  if (!_ttsEnabled || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const clean = text
    .replace(/```[\s\S]*?```/g, 'كود برمجي')
    .replace(/[*_#`>\[\]]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 500);
  if (!clean) return;
  _ttsUtter      = new SpeechSynthesisUtterance(clean);
  _ttsUtter.lang = _ttsLang;
  _ttsUtter.rate = 1.0;
  speechSynthesis.speak(_ttsUtter);
}

// Stop TTS when user starts typing a new message
document.addEventListener('DOMContentLoaded', () => {
  const inp = document.getElementById('msg-input');
  if (inp) {
    inp.addEventListener('input', () => {
      if (speechSynthesis.speaking) speechSynthesis.cancel();
    });
  }
});

// ─── Voice Input (Web Speech API) ──────────────────────────────────
const _SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let _recog          = null;
let _listening      = false;
let _userStoppedMic = false;
let _priorText      = '';

function _initMic() {
  const btn = document.getElementById('mic-btn');
  if (!_SR) { btn.classList.add('hidden'); return; }

  _recog = new _SR();
  _recog.continuous      = true;
  _recog.interimResults  = true;
  _recog.maxAlternatives = 1;

  _recog.onstart = () => {
    _listening = true;
    btn.classList.add('recording');
    btn.title = 'إيقاف التسجيل';
    btn.textContent = '⏹';
  };

  _recog.onresult = (e) => {
    const input = document.getElementById('msg-input');
    let final = '', interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      e.results[i].isFinal ? (final += t) : (interim += t);
    }
    input.value = _priorText + (final || interim);
    autoResize(input);
  };

  _recog.onend = () => {
    _listening = false;
    btn.classList.remove('recording');
    btn.title = 'تحدث';
    btn.textContent = '🎤';
    if (_userStoppedMic) {
      _userStoppedMic = false;
      _priorText = '';
      const input = document.getElementById('msg-input');
      if (input.value.trim()) sendMessage();
    } else {
      // Browser ended recognition early (silence timeout, network hiccup, etc.)
      // Delay 250ms before restarting — Chrome throws a silent InvalidStateError
      // if start() is called immediately after the session ends.
      const input = document.getElementById('msg-input');
      const cur   = input.value;
      _priorText  = cur ? cur + ' ' : '';
      setTimeout(() => {
        if (!_userStoppedMic) {
          try { _recog.start(); } catch (_) {}
        }
      }, 250);
    }
  };

  _recog.onerror = (e) => {
    // Permanent failures — hide the button entirely
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      _listening = false;
      _userStoppedMic = false;
      _priorText = '';
      btn.classList.remove('recording');
      btn.classList.add('hidden');
      btn.title = 'تحدث';
      btn.textContent = '🎤';
      return;
    }
    // Transient errors (no-speech, network, audio-capture, aborted):
    // restore the input field but keep _priorText intact so onend can
    // re-capture it and restart with the accumulated transcript.
    if (e.error !== 'aborted') {
      document.getElementById('msg-input').value = _priorText;
    }
    // audio-capture = hardware gone; clear state so restart gets a clean slate
    if (e.error === 'audio-capture') {
      _priorText = '';
    }
  };
}

function toggleMic() {
  if (!_recog) return;
  if (_listening) {
    _userStoppedMic = true;
    _recog.stop();
    return;
  }
  _userStoppedMic = false;
  const txt       = document.getElementById('msg-input').value;
  const hasArabic = /[؀-ۿ]/.test(txt);
  const hasLatin  = /[a-zA-Z]{3,}/.test(txt);
  _recog.lang = (hasLatin && !hasArabic) ? 'en-US' : 'ar-IQ';

  _priorText = txt ? txt + ' ' : '';
  try { _recog.start(); } catch (_) { /* already started */ }
}
