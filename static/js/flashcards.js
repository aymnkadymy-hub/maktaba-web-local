// ─── Flashcards + spaced repetition (SM-2) ──────────────────────────
// Generate cards from a book, then review them on an SM-2 schedule.
// Grades map to the SM-2 0-5 scale: نسيت=1 (lapse), صعب=3, جيد=4, سهل=5.
let _fcBooksLoaded = false;
let _fcQueue       = [];   // due cards currently being reviewed
let _fcIdx         = 0;
let _fcRevealed    = false;

// Called on tab entry — refresh everything (book list cached, stats/list cheap)
async function loadFlashcards() {
  showFlashView('setup');
  await Promise.all([loadFlashBooks(), refreshFlashStats(), loadFlashList()]);
}

async function loadFlashBooks() {
  if (_fcBooksLoaded) return;
  const sel = document.getElementById('fc-book-select');
  sel.innerHTML = '<option value="">جارٍ تحميل…</option>';
  try {
    const r = await fetch(getAPI() + '/books', { headers: getAuthHeaders() });
    if (!r.ok) { sel.innerHTML = '<option value="">فشل تحميل الكتب</option>'; return; }
    const data  = await r.json();
    const books = Array.isArray(data) ? data : (data.books || []);
    if (!books.length) {
      sel.innerHTML = '<option value="">لا توجد كتب مرفوعة</option>';
      _fcBooksLoaded = true;
      return;
    }
    sel.innerHTML = '<option value="">— اختر كتاباً —</option>' +
      books.map(b => {
        const t = typeof b === 'string' ? b : (b.title || b.book_title || String(b));
        const e = escHtml(t);
        return `<option value="${e}">${e}</option>`;
      }).join('');
    _fcBooksLoaded = true;
  } catch (e) {
    sel.innerHTML = '<option value="">خطأ في الاتصال</option>';
  }
}

async function refreshFlashStats() {
  try {
    const r = await fetch(getAPI() + '/flashcards/stats', { headers: getAuthHeaders() });
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('fc-stat-total').textContent = d.total || 0;
    document.getElementById('fc-stat-due').textContent   = d.due   || 0;
    const rb = document.getElementById('fc-review-btn');
    const has = (d.due || 0) > 0;
    rb.disabled      = !has;
    rb.style.opacity = has ? '1' : '.5';
    rb.textContent   = has ? `▶ ابدأ المراجعة (${d.due})` : 'لا بطاقات مستحقّة الآن';
  } catch (e) { /* non-fatal */ }
}

async function loadFlashList() {
  const wrap = document.getElementById('fc-list');
  try {
    const r = await fetch(getAPI() + '/flashcards', { headers: getAuthHeaders() });
    const d = await r.json();
    const cards = d.cards || [];
    if (!cards.length) {
      wrap.innerHTML = '<div style="text-align:center;padding:30px;color:#9e9e9e">لا توجد بطاقات بعد — ولّد بعضها من كتاب</div>';
      return;
    }
    wrap.innerHTML = cards.map(c => `
      <div style="display:flex;gap:8px;align-items:flex-start;background:var(--bg-input);border:1px solid var(--border-input);border-radius:10px;padding:10px 12px;margin-bottom:8px">
        <div style="flex:1;min-width:0">
          <div style="font-weight:600;font-size:13px;color:var(--text-main)">${escHtml(c.front)}</div>
          <div style="font-size:12px;color:var(--text-main);opacity:.7;margin-top:3px">${escHtml(c.back)}</div>
          <div style="font-size:11px;color:#9e9e9e;margin-top:5px">📖 ${escHtml(c.book_title || '')} · مراجعات ${c.reviews || 0}</div>
        </div>
        <button onclick="deleteFlash('${escHtml(c.id)}')" title="حذف"
          style="background:none;border:none;cursor:pointer;font-size:15px;opacity:.6;flex-shrink:0">🗑️</button>
      </div>`).join('');
  } catch (e) {
    wrap.innerHTML = '<div style="text-align:center;padding:30px;color:#9e9e9e">خطأ في التحميل</div>';
  }
}

async function generateFlashcards() {
  const book = document.getElementById('fc-book-select').value;
  if (!book) { alert('اختر كتاباً أولاً'); return; }
  const n = parseInt(document.getElementById('fc-n-range').value);
  showFlashView('loading');
  document.getElementById('fc-loading-msg').textContent = `جارٍ توليد ${n} بطاقة…`;
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), Math.min(60000 + n * 9000, 300000));
  try {
    const r = await fetch(getAPI() + '/flashcards/generate', {
      method:  'POST',
      headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
      body:    JSON.stringify({ book_title: book, n_cards: n }),
      signal:  ctrl.signal,
    });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `خطأ ${r.status}`); }
    const d = await r.json();
    showFlashView('setup');
    await loadFlashcards();
    alert(`✅ تم توليد ${d.created} بطاقة — اضغط "ابدأ المراجعة"`);
  } catch (e) {
    showFlashView('setup');
    alert('⚠️ ' + (e.name === 'AbortError' ? 'انتهت المهلة — جرب عدداً أقل' : (e.message || 'خطأ غير معروف')));
  } finally {
    clearTimeout(timer);
  }
}

async function startFlashReview() {
  try {
    const r = await fetch(getAPI() + '/flashcards/due?limit=50', { headers: getAuthHeaders() });
    const d = await r.json();
    _fcQueue = d.cards || [];
    if (!_fcQueue.length) { alert('لا توجد بطاقات مستحقّة الآن 🎉'); return; }
    _fcIdx = 0;
    showFlashView('review');
    renderFlashCard();
  } catch (e) {
    alert('⚠️ تعذّر تحميل البطاقات المستحقّة');
  }
}

function renderFlashCard() {
  const c = _fcQueue[_fcIdx];
  _fcRevealed = false;
  document.getElementById('fc-review-book').textContent      = '📖 ' + (c.book_title || '');
  document.getElementById('fc-review-remaining').textContent = _fcQueue.length - _fcIdx;
  document.getElementById('fc-card-front').textContent = c.front;
  const back = document.getElementById('fc-card-back');
  back.textContent     = c.back;
  back.style.display   = 'none';
  document.getElementById('fc-card-hint').style.display = 'block';
  document.getElementById('fc-grade-row').style.display = 'none';
}

function revealFlashBack() {
  if (_fcRevealed) return;
  _fcRevealed = true;
  document.getElementById('fc-card-back').style.display = 'block';
  document.getElementById('fc-card-hint').style.display = 'none';
  document.getElementById('fc-grade-row').style.display = 'flex';
}

async function gradeFlash(grade) {
  if (!_fcRevealed) return;
  const c = _fcQueue[_fcIdx];
  try {
    await fetch(getAPI() + '/flashcards/' + encodeURIComponent(c.id) + '/review', {
      method:  'POST',
      headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
      body:    JSON.stringify({ grade }),
    });
  } catch (e) { /* keep going — review is best-effort */ }
  _fcIdx++;
  if (_fcIdx >= _fcQueue.length) {
    showFlashView('done');
    document.getElementById('fc-done-msg').textContent = `راجعت ${_fcQueue.length} بطاقة`;
    refreshFlashStats();
  } else {
    renderFlashCard();
  }
}

async function deleteFlash(id) {
  if (!confirm('حذف هذه البطاقة؟')) return;
  try {
    await fetch(getAPI() + '/flashcards/' + encodeURIComponent(id), {
      method: 'DELETE', headers: getAuthHeaders(),
    });
    loadFlashList();
    refreshFlashStats();
  } catch (e) {
    alert('⚠️ تعذّر حذف البطاقة');
  }
}

function resetFlashcards() {
  showFlashView('setup');
  refreshFlashStats();
  loadFlashList();
}

function showFlashView(v) {
  ['setup', 'loading', 'review', 'done'].forEach(x => {
    const el = document.getElementById('fc-' + x);
    if (el) el.style.display = (x === v) ? 'block' : 'none';
  });
}
