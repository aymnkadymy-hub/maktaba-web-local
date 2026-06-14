// ─── Arabic + English normalization ──────────────────────────────
function _norm(t) {
  if (!t) return '';
  return String(t)
    .replace(/[أإآٱ]/g, 'ا') // أإآٱ → ا
    .replace(/[يى]/g, 'ي')              // يى → ي
    .replace(/ة/g, 'ه')                      // ة → ه
    .replace(/[\u064B-\u065F\u0610-\u061A\u0670]/g, '') // tashkeel
    .toLowerCase()
    .trim();
}

// Score how well a text chunk matches the query (0..1)
function _scoreText(qNorm, text) {
  if (!text || !qNorm) return 0;
  const tNorm = _norm(text);
  if (tNorm.includes(qNorm)) return 1.0;
  const words = qNorm.split(/\s+/).filter(w => w.length >= 2);
  if (!words.length) return 0;
  const hits = words.filter(w => tNorm.includes(w)).length;
  return (hits / words.length) * 0.8;
}

// Search ALL maktaba_history_* keys in localStorage
// (covers every user key and the guest key)
function _searchLocal(q) {
  const qNorm = _norm(q);
  const out   = [];

  // Collect all history keys regardless of which user is active
  const historyKeys = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith('maktaba_history_')) historyKeys.push(k);
    }
  } catch (e) {
    console.warn('[search] localStorage scan failed:', e);
  }

  if (!historyKeys.length) return out;

  const seenIds = new Set();

  for (const key of historyKeys) {
    let all = {};
    try { all = JSON.parse(localStorage.getItem(key)) || {}; } catch (_) { continue; }

    for (const daySessions of Object.values(all)) {
      if (typeof daySessions !== 'object' || !daySessions) continue;

      for (const s of Object.values(daySessions)) {
        if (!s || !s.id || seenIds.has(s.id)) continue;
        seenIds.add(s.id);

        const msgs = Array.isArray(s.messages) ? s.messages : [];
        let bestScore = 0;
        let bestUser  = s.preview || '';
        let bestBot   = '';

        if (!q) {
          // No query — show last user + bot pair
          for (const m of msgs.slice(-30)) {
            if (!m || !m.role) continue;
            if (m.role === 'user')
              bestUser = m.content || '';
            if (m.role === 'assistant' || m.role === 'bot')
              bestBot  = m.content || '';
          }
          bestScore = 0.01;
        } else {
          // Find the highest-scoring message and pair it with its neighbor
          for (let i = 0; i < msgs.length; i++) {
            const m  = msgs[i];
            if (!m || !m.content) continue;
            const sc = _scoreText(qNorm, m.content);
            if (sc > bestScore) {
              bestScore = sc;
              if (m.role === 'user') {
                bestUser = m.content;
                const nxt = msgs[i + 1];
                if (nxt) bestBot = nxt.content || '';
              } else {
                bestBot = m.content;
                const prv = msgs[i - 1];
                if (prv && prv.role === 'user') bestUser = prv.content || '';
              }
            }
          }
        }

        if (bestScore > 0 || !q) {
          out.push({
            session_id:   s.id,
            user_message: bestUser,
            bot_message:  bestBot,
            score:        bestScore,
            created_at:   (s.date || '') + 'T00:00:00',
            book_title:   null,
            topic:        s.preview || 'محادثة',
            _local:       true,
          });
        }
      }
    }
  }

  return out;
}

// ─── Highlight matching words ─────────────────────────────────────
function _hiText(text, query) {
  if (!text) return '';
  const safe  = escHtml(text);
  if (!query) return safe;
  const words = query.trim().split(/\s+/).filter(w => w.length >= 2);
  if (!words.length) return safe;
  let out = safe;
  words.forEach(w => {
    try {
      const re = new RegExp(w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
      out = out.replace(re, m => `<mark class="srch-hi">${m}</mark>`);
    } catch (_) {}
  });
  return out;
}

// Relevance dot (green / orange / grey)
function _dot(score) {
  const pct = Math.round((score || 0) * 100);
  if (pct < 5) return '';
  const col = pct >= 70 ? '#4caf50' : pct >= 35 ? '#ff9800' : '#bdbdbd';
  return `<span class="srch-dot" style="background:${col}" title="تطابق ${pct}%"></span>`;
}

// Single result card
function _card(item, query) {
  const userHtml = item.user_message
    ? `<div class="search-result-q">❓ ${_hiText(item.user_message, query)}</div>`
    : '';
  const bot     = item.bot_message || '';
  const preview = bot.length > 280 ? bot.substring(0, 280) + '…' : bot;
  const botHtml = preview
    ? `<div class="search-result-a">🤖 ${_hiText(preview, query)}</div>`
    : '';
  const date     = (item.created_at || '').substring(0, 10);
  const book     = item.book_title
    ? `<span>📚 ${escHtml(item.book_title)}</span>`
    : '';
  const topicRaw = item.topic || '';
  const topic    = (topicRaw && topicRaw !== 'محادثة')
    ? `<span title="${escHtml(topicRaw)}">💬 ${escHtml(topicRaw.substring(0, 28))}${topicRaw.length > 28 ? '…' : ''}</span>`
    : '';
  return `
    <div class="search-result-item">
      <div class="search-result-meta">
        ${_dot(item.score)}
        <span>📅 ${date}</span>
        ${book}${topic}
      </div>
      ${userHtml}
      ${botHtml}
    </div>`;
}

// Loading spinner
function _srchLoading() {
  const el = document.getElementById('search-results');
  if (el) el.innerHTML = '<div style="text-align:center;padding:40px;color:#9e9e9e">⏳ جارٍ البحث…</div>';
}

// ─── Main search function ─────────────────────────────────────────
async function doConvSearch() {
  const inputEl = document.getElementById('conv-search-input');
  const q       = inputEl ? inputEl.value.trim() : '';
  const results = document.getElementById('search-results');
  if (!results) return;

  _srchLoading();

  try {
    // 1. Search localStorage — covers ALL local conversations
    const localItems = _searchLocal(q);
    console.log('[search] localStorage sessions found:', localItems.length, '| query:', q || '(none)');

    // 2. Try server search — covers server-only sessions (may fail if not logged in)
    let serverItems      = [];
    let serverHasHistory = false;
    let serverAnyMatch   = false;

    try {
      const url = getAPI() + '/search/conversations'
        + (q ? '?q=' + encodeURIComponent(q) + '&limit=50' : '?limit=50');
      const r = await fetch(url, { headers: getAuthHeaders(), credentials: 'include' });
      if (r.ok) {
        const data       = await r.json();
        serverItems      = data.results    || [];
        serverHasHistory = !!data.has_history;
        serverAnyMatch   = !!data.any_match;
        console.log('[search] server sessions:', serverItems.length, '| has_history:', serverHasHistory);
      } else {
        console.log('[search] server returned', r.status);
      }
    } catch (fetchErr) {
      console.warn('[search] server unreachable:', fetchErr.message);
    }

    // 3. Merge: server items take priority (richer metadata), local fills in the rest
    const serverIds = new Set(serverItems.map(i => i.session_id));
    const localOnly = localItems.filter(i => !serverIds.has(i.session_id));
    let items = [...serverItems, ...localOnly];

    // 4. Sort
    if (q) {
      items.sort((a, b) => (b.score || 0) - (a.score || 0));
    } else {
      items.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    }
    items = items.slice(0, 100);

    const hasHistory = serverHasHistory || localItems.length > 0;
    const anyMatch   = serverAnyMatch   || items.some(i => (i.score || 0) >= 0.12);

    // 5. Empty states
    if (!hasHistory) {
      results.innerHTML = `
        <div style="text-align:center;padding:48px 20px;color:#9e9e9e">
          <div style="font-size:36px;margin-bottom:14px">📭</div>
          <div style="font-size:15px;font-weight:600;margin-bottom:8px">ما عندك محادثات محفوظة بعد</div>
          <div style="font-size:12px;line-height:1.7">
            اذهب لتبويب <b>الدردشة</b> وابدأ محادثة<br>
            ستظهر هنا تلقائياً بعد أول رسالة
          </div>
        </div>`;
      return;
    }

    if (!items.length) {
      results.innerHTML = `
        <div style="text-align:center;padding:40px;color:#9e9e9e">
          <div style="font-size:32px;margin-bottom:10px">🗂️</div>
          <div>لا توجد نتائج مطابقة</div>
        </div>`;
      return;
    }

    // 6. Header
    let headerHtml = '';
    if (!q) {
      headerHtml = `<div class="srch-header">🕐 ${items.length} محادثة محفوظة</div>`;
    } else if (anyMatch) {
      const good = items.filter(i => (i.score || 0) >= 0.12).length;
      headerHtml = `<div class="srch-header">🎯 ${good} نتيجة مطابقة — الأقرب أولاً</div>`;
    } else {
      headerHtml = `<div class="srch-header srch-header-weak">🔎 لم نجد تطابقاً دقيقاً لـ «${escHtml(q)}» — إليك آخر المحادثات</div>`;
    }

    results.innerHTML = headerHtml + items.map(item => _card(item, q)).join('');

  } catch (err) {
    console.error('[search] fatal error:', err);
    const el = document.getElementById('search-results');
    if (el) el.innerHTML = `<div style="text-align:center;padding:40px;color:#e53935">⚠️ خطأ: ${escHtml(String(err.message || err))}</div>`;
  }
}

// Auto-load when search tab opens
function loadSearchHistory() {
  const inputEl = document.getElementById('conv-search-input');
  if (!inputEl || !inputEl.value.trim()) doConvSearch();
}
