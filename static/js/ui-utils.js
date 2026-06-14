// ─── Storage helpers ───────────────────────────────────────────────
function loadAll() {
  try { return JSON.parse(localStorage.getItem(getStorageKey())) || {}; }
  catch { return {}; }
}
function saveAll(data) {
  try { localStorage.setItem(getStorageKey(), JSON.stringify(data)); } catch {}
}
function todayKey() {
  return new Date().toISOString().slice(0, 10); // "YYYY-MM-DD"
}

function persistMessage(role, content) {
  sessionMessages.push({ role, content, ts: Date.now() });
  const all = loadAll();
  const day = todayKey();
  if (!all[day]) all[day] = {};
  all[day][sessionId] = {
    id:       sessionId,
    date:     day,
    preview:  sessionMessages.find(m => m.role === 'user')?.content?.slice(0, 55) || 'محادثة',
    messages: sessionMessages,
  };
  saveAll(all);
  renderSidebar();
}

// ─── Date formatting (Arabic) ──────────────────────────────────────
const AR_MONTHS = ['يناير','فبراير','مارس','أبريل','مايو','يونيو',
                   'يوليو','أغسطس','سبتمبر','أكتوبر','نوفمبر','ديسمبر'];
const AR_DAYS   = ['الأحد','الاثنين','الثلاثاء','الأربعاء','الخميس','الجمعة','السبت'];

function formatDateLabel(dateKey) {
  const now   = new Date();
  const d     = new Date(dateKey + 'T00:00:00');
  const diffD = Math.floor((new Date(now.toISOString().slice(0,10)) - d) / 86400000);
  if (diffD === 0) return 'اليوم';
  if (diffD === 1) return 'أمس';
  if (diffD <= 6)  return AR_DAYS[d.getDay()];
  return `${d.getDate()} ${AR_MONTHS[d.getMonth()]} ${d.getFullYear()}`;
}

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString('ar', { hour: '2-digit', minute: '2-digit' });
}

// ─── Sidebar rendering ─────────────────────────────────────────────
function renderSidebar() {
  const all  = loadAll();
  const list = document.getElementById('history-list');
  list.innerHTML = '';

  const dayMap   = {};
  const localIds = new Set();

  // 1. localStorage sessions
  for (const [day, daySessions] of Object.entries(all)) {
    if (!dayMap[day]) dayMap[day] = [];
    for (const s of Object.values(daySessions)) {
      localIds.add(s.id);
      dayMap[day].push({ ...s, _ts: s.messages[0]?.ts || 0 });
    }
  }

  // 2. Server-only sessions (not in localStorage)
  for (const ss of _serverSessions) {
    if (localIds.has(ss.id)) continue;
    const day = (ss.created_at || '').slice(0, 10) || todayKey();
    if (!dayMap[day]) dayMap[day] = [];
    dayMap[day].push({
      id:       ss.id,
      date:     day,
      preview:  ss.topic || 'محادثة',
      messages: [],
      _ts:      ss.created_at ? new Date(ss.created_at).getTime() : 0,
      _server:  true,
    });
  }

  const days = Object.keys(dayMap).sort().reverse();
  if (days.length === 0) {
    list.innerHTML = '<div class="no-history">لا توجد محادثات سابقة</div>';
    return;
  }

  for (const day of days) {
    const sessions = dayMap[day].sort((a, b) => b._ts - a._ts);

    const group = document.createElement('div');
    group.className = 'date-group';

    const label = document.createElement('div');
    label.className = 'date-label';
    label.textContent = formatDateLabel(day);
    group.appendChild(label);

    for (const s of sessions) {
      const item = document.createElement('div');
      item.className = 'session-item' + (s.id === sessionId && !viewingHistory ? ' active' : '');
      const timeEl = s._server
        ? `<span class="session-time">☁️ سيرفر</span>`
        : `<span class="session-time">${formatTime(s._ts)}</span>`;
      item.innerHTML =
        `<div class="session-text">` +
          `<span>${escHtml(s.preview || 'محادثة')}</span>${timeEl}` +
        `</div>` +
        `<button class="session-del-btn" title="حذف هذه المحادثة">🗑</button>`;
      item.querySelector('.session-text').onclick = () => openHistoricalSession(s);
      item.querySelector('.session-del-btn').onclick = (e) => {
        e.stopPropagation();
        deleteSession(s);
      };
      group.appendChild(item);
    }

    list.appendChild(group);
  }
}

// ─── Open a historical session (read-only view) ────────────────────
async function openHistoricalSession(s) {
  viewingHistory  = true;
  sessionId       = s.id;
  _viewingSession = s;

  const messagesDiv = document.getElementById('messages');

  // Server-only session — messages not yet loaded
  if (s._server && s.messages.length === 0) {
    messagesDiv.innerHTML =
      '<div style="text-align:center;padding:40px;color:#9e9e9e">⏳ جاري التحميل…</div>';
    try {
      const r = await fetch(
        `${getAPI()}/sessions/${s.id}/messages`,
        { headers: getAuthHeaders(), signal: AbortSignal.timeout(30000) }
      );
      if (r.status === 401) { handle401(); return; }
      if (r.ok) {
        const fetched = await r.json();
        s.messages = fetched
          .filter(m => m.role !== 'summary')
          .map(m => ({ role: m.role, content: m.content, ts: 0 }));
      }
    } catch (_) {
      messagesDiv.innerHTML =
        '<div style="text-align:center;padding:30px;color:#f44336">⚠️ تعذر تحميل المحادثة</div>';
      return;
    }
  }

  messagesDiv.innerHTML = '';
  for (const m of (s.messages || [])) {
    const div    = document.createElement('div');
    const isUser = m.role === 'user';
    div.className = 'bubble ' + (isUser ? 'user-bubble' : 'bot-bubble');
    if (isUser) {
      div.textContent = m.content;
    } else {
      div.innerHTML = renderMarkdown(m.content);
      const actions = document.createElement('div');
      actions.className = 'bubble-actions';
      const pin = document.createElement('button');
      pin.className = 'bubble-save-btn';
      pin.textContent = '📌 حفظ ملاحظة';
      pin.onclick = () => saveNoteFromBubble(m.content, pin);
      actions.appendChild(pin);
      div.appendChild(actions);
    }
    messagesDiv.appendChild(div);
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  document.getElementById('view-banner').classList.add('show');
  document.getElementById('input-area').classList.add('disabled');
  renderSidebar();
  document.getElementById('sidebar').classList.remove('open');
}

// ─── Continue the currently viewed historical session ──────────────
function continueSession() {
  viewingHistory = false;
  const all = loadAll();
  const dayData = Object.values(all).find(d => d[sessionId]);
  if (dayData?.[sessionId]) {
    sessionMessages = [...dayData[sessionId].messages];
  } else if (_viewingSession?.messages?.length) {
    sessionMessages = [..._viewingSession.messages];
  }
  _viewingSession = null;
  document.getElementById('view-banner').classList.remove('show');
  document.getElementById('input-area').classList.remove('disabled');
  document.getElementById('msg-input').focus();
  renderSidebar();
}

// ─── New session ───────────────────────────────────────────────────
function startNewSession() {
  sessionId       = crypto.randomUUID();
  sessionMessages = [];
  viewingHistory  = false;

  const messagesDiv = document.getElementById('messages');
  messagesDiv.innerHTML =
    '<div id="empty"><div class="icon">📖</div><p>مرحباً!</p>' +
    '<small>اسأل عن أي كتاب في مكتبتك أو أي سؤال هندسي</small></div>';

  document.getElementById('view-banner').classList.remove('show');
  document.getElementById('input-area').classList.remove('disabled');
  document.getElementById('msg-input').focus();
  document.getElementById('sidebar').classList.remove('open');
  renderSidebar();
}

// ─── Delete session helpers ────────────────────────────────────────
function deleteSession(s) {
  const all = loadAll();
  for (const day of Object.keys(all)) {
    if (all[day][s.id]) {
      delete all[day][s.id];
      if (Object.keys(all[day]).length === 0) delete all[day];
    }
  }
  saveAll(all);

  _serverSessions = _serverSessions.filter(ss => ss.id !== s.id);

  if (s._server) {
    try {
      const bl = JSON.parse(localStorage.getItem('maktaba_deleted_sessions') || '[]');
      if (!bl.includes(s.id)) { bl.push(s.id); localStorage.setItem('maktaba_deleted_sessions', JSON.stringify(bl)); }
    } catch {}
  }

  if (s.id === sessionId) {
    startNewSession();
  } else {
    renderSidebar();
  }
}

// ─── Server sessions (PostgreSQL) ──────────────────────────────────
async function loadServerHistory() {
  if (!_authUser) return;
  try {
    const r = await fetch(
      `${getAPI()}/sessions/${_authUser}`,
      { headers: getAuthHeaders(), signal: AbortSignal.timeout(4000) }
    );
    if (r.status === 401) { handle401(); return; }
    if (!r.ok) return;
    const all     = await r.json();
    const deleted = JSON.parse(localStorage.getItem('maktaba_deleted_sessions') || '[]');
    _serverSessions = all.filter(s => !deleted.includes(s.id));
    renderSidebar();
  } catch (_) {
    // Server unreachable — sidebar shows localStorage only
  }
}

// ─── Confirm modal (clear all) ────────────────────────────────────
function openConfirm() {
  document.getElementById('confirm-overlay').classList.add('show');
}
function closeConfirm() {
  document.getElementById('confirm-overlay').classList.remove('show');
}

function confirmClearAll() {
  localStorage.removeItem(getStorageKey());
  _serverSessions = [];
  localStorage.removeItem('maktaba_deleted_sessions');
  closeConfirm();
  startNewSession();
}

// ─── Settings modal ───────────────────────────────────────────────
function openSettings() {
  document.getElementById('server-url-input').value = getAPI();
  document.getElementById('conn-status').textContent = '';
  document.getElementById('conn-status').className = '';
  document.getElementById('settings-overlay').classList.add('show');
}

function closeSettings() {
  document.getElementById('settings-overlay').classList.remove('show');
}

async function saveSettings() {
  const input  = document.getElementById('server-url-input').value.trim();
  const status = document.getElementById('conn-status');
  if (!input) return;

  status.textContent = 'جاري الفحص…';
  status.className = '';

  try {
    const r = await fetch(input.replace(/\/$/, '') + '/health', { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      localStorage.setItem(API_KEY, input.replace(/\/$/, ''));
      status.textContent = '✅ متصل بنجاح — سيتم تطبيق التغيير';
      status.className = 'ok';
      setTimeout(closeSettings, 900);
    } else {
      status.textContent = `⚠️ السيرفر رد بـ HTTP ${r.status}`;
      status.className = 'err';
    }
  } catch {
    localStorage.setItem(API_KEY, input.replace(/\/$/, ''));
    status.textContent = '💾 تم الحفظ (لم يتحقق الاتصال)';
    status.className = 'ok';
    setTimeout(closeSettings, 1000);
  }
}

// ─── Theme ────────────────────────────────────────────────────────
function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.getElementById('theme-btn').textContent = isDark ? '🌙' : '☀️';
  localStorage.setItem('maktaba_theme', isDark ? 'light' : 'dark');
}

// Restore saved theme on load
(function() {
  const saved = localStorage.getItem('maktaba_theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = '☀️';
  }
})();

// ─── Basic UI helpers ──────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}
function scrollDown() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}
function removeEmpty() {
  const e = document.getElementById('empty');
  if (e) e.remove();
}
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// ─── Tab switching ─────────────────────────────────────────────────
let _activeTab = 'chat';

function switchTab(tab) {
  // ── Leave current tab cleanly ──────────────────────────────────
  if (_activeTab === 'chat' && tab !== 'chat' && streaming) {
    // Cancel any in-flight SSE stream so the connection isn't left dangling
    cancelChatStream();
  }

  _activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');

  const chatEls = ['view-banner', 'messages', 'img-preview-area', 'input-area'];
  chatEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = (tab === 'chat') ? '' : 'none';
  });

  const panelLoaders = {
    quiz:       () => loadQuizBooks(),
    flashcards: () => { if (typeof loadFlashcards === 'function') loadFlashcards(); },
    notes:      () => loadNotes(),
    search:     () => { if (typeof loadSearchHistory === 'function') loadSearchHistory(); },
    library:    () => loadLibraryBooks(),
  };
  ['quiz-panel', 'flashcards-panel', 'notes-panel', 'search-panel', 'library-panel'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  if (panelLoaders[tab]) {
    document.getElementById(tab + '-panel').classList.add('active');
    panelLoaders[tab]();
  }
}

// ─── Server ping ───────────────────────────────────────────────────
async function pingServer() {
  try {
    const r = await fetch(getAPI() + '/health', { signal: AbortSignal.timeout(3000) });
    document.getElementById('status-dot').style.background = r.ok ? '#4caf50' : '#ff9800';
  } catch {
    document.getElementById('status-dot').style.background = '#f44336';
  }
}
