// ─── Active stream controller (aborted on tab switch) ─────────────
let _chatAbortCtrl = null;

function cancelChatStream() {
  if (_chatAbortCtrl) { _chatAbortCtrl.abort(); _chatAbortCtrl = null; }
  removeTyping();
  streaming = false;
  const btn = document.getElementById('send-btn');
  if (btn) btn.disabled = false;
}

// ─── Chat bubble helpers ───────────────────────────────────────────
function addBubble(text, isUser) {
  removeEmpty();
  const m   = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'bubble ' + (isUser ? 'user-bubble' : 'bot-bubble');
  if (isUser) {
    div.textContent = text;
  } else {
    div.innerHTML = text ? renderMarkdown(text) : '';
    const actions = document.createElement('div');
    actions.className = 'bubble-actions';
    const pin = document.createElement('button');
    pin.className = 'bubble-save-btn';
    pin.textContent = '📌 حفظ ملاحظة';
    pin.onclick = () => saveNoteFromBubble(text, pin);
    actions.appendChild(pin);
    div.appendChild(actions);
  }
  m.appendChild(div);
  scrollDown();
  return div;
}

function addTyping() {
  removeEmpty();
  const m   = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'bubble bot-bubble typing';
  div.id = 'typing-indicator';
  div.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  m.appendChild(div);
  scrollDown();
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

// ─── Send message ──────────────────────────────────────────────────
async function sendMessage() {
  if (streaming || viewingHistory) return;
  const input = document.getElementById('msg-input');
  const text  = input.value.trim();
  if (!text && !_attachedImage) return;

  if (_attachedImage) { await sendImageMessage(text); return; }

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;
  streaming = true;

  addBubble(text, true);
  persistMessage('user', text);

  let botDiv   = null;
  let fullText = '';

  _chatAbortCtrl = new AbortController();

  try {
    const res = await fetch(getAPI() + '/chat/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
      body:    JSON.stringify({ message: text, user_id: _authUser, session_id: sessionId }),
      signal:  _chatAbortCtrl.signal,
    });

    if (res.status === 401) { handle401(); streaming = false; document.getElementById('send-btn').disabled = false; return; }
    if (!res.ok) throw new Error('HTTP ' + res.status);
    addTyping();

    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let buffer   = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;
        try {
          const d = JSON.parse(raw);
          if (d.token) {
            fullText += d.token;
            if (!botDiv) { removeTyping(); botDiv = addBubble('', false); }
            botDiv.textContent = fullText;   // plain during streaming (fast)
            scrollDown();
          }
          if (d.replace !== undefined) {
            // Dialectized full text — replace raw streaming text before markdown render
            fullText = d.replace;
            if (botDiv) botDiv.textContent = fullText;
          }
          if (d.error) { removeTyping(); addBubble('❌ ' + d.error, false); }
        } catch (_) {}
      }
    }

    if (fullText) {
      persistMessage('assistant', fullText);
      if (botDiv) {
        botDiv.innerHTML = renderMarkdown(fullText);
        const actions = document.createElement('div');
        actions.className = 'bubble-actions';
        const pin = document.createElement('button');
        pin.className   = 'bubble-save-btn';
        pin.textContent = '📌 حفظ ملاحظة';
        pin.onclick = () => saveNoteFromBubble(fullText, pin);
        actions.appendChild(pin);
        botDiv.appendChild(actions);
        scrollDown();
      }
      if (_ttsEnabled) speakText(fullText);
    }

  } catch (err) {
    if (err.name !== 'AbortError') {
      removeTyping();
      if (!botDiv) addBubble(`⚠️ تعذر الاتصال بالسيرفر — تأكد أنه شغال على ${getAPI()}`, false);
    }
    // AbortError: stream was cancelled by tab switch — silently ignore
  }

  _chatAbortCtrl = null;
  streaming = false;
  document.getElementById('send-btn').disabled = false;
  input.focus();
}

// ─── Image Attachment ──────────────────────────────────────────────
function attachImage(e) {
  const file = e.target.files[0];
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = ev => {
    _attachedImage = { file, dataURL: ev.target.result };
    document.getElementById('img-preview').src    = ev.target.result;
    document.getElementById('img-preview-name').textContent = file.name;
    document.getElementById('img-preview-area').classList.add('show');
    document.getElementById('img-btn').classList.add('has-img');
  };
  reader.readAsDataURL(file);
  e.target.value = ''; // allow re-selecting the same file
}

function clearImage() {
  _attachedImage = null;
  document.getElementById('img-preview-area').classList.remove('show');
  document.getElementById('img-btn').classList.remove('has-img');
}

async function sendImageMessage(caption) {
  const img = _attachedImage;
  clearImage();

  const input = document.getElementById('msg-input');
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;
  streaming = true;

  removeEmpty();
  const m       = document.getElementById('messages');
  const userDiv = document.createElement('div');
  userDiv.className = 'bubble user-bubble';
  userDiv.innerHTML =
    `<img src="${img.dataURL}" style="max-width:200px;max-height:150px;border-radius:8px;display:block;${caption ? 'margin-bottom:6px' : ''}">` +
    (caption ? `<span>${escHtml(caption)}</span>` : '');
  m.appendChild(userDiv);
  scrollDown();
  persistMessage('user', `[صورة] ${caption || img.file.name}`);

  addTyping();

  try {
    const fd = new FormData();
    fd.append('file',       img.file);
    fd.append('user_input', caption);
    fd.append('user_id',    _authUser);
    fd.append('session_id', sessionId);

    const r = await fetch(getAPI() + '/chat/image', { method: 'POST', headers: getAuthHeaders(), body: fd });
    if (r.status === 401) { removeTyping(); handle401(); streaming = false; document.getElementById('send-btn').disabled = false; return; }
    removeTyping();

    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data   = await r.json();
    const botDiv = addBubble(data.response, false);
    persistMessage('assistant', data.response);
    if (botDiv) { botDiv.innerHTML = renderMarkdown(data.response); scrollDown(); }
  } catch (err) {
    removeTyping();
    addBubble('⚠️ فشل تحليل الصورة — تأكد أن Ollama شغال ونموذج llava:7b متاح (ollama pull llava:7b)', false);
  }

  streaming = false;
  document.getElementById('send-btn').disabled = false;
  input.focus();
}
