// ─── Quiz logic ────────────────────────────────────────────────────
let _quizDiff        = 'medium';
let _quizQuestions   = [];
let _quizCurrent     = 0;
let _quizScore       = 0;
let _quizAnswered    = false;
let _quizBooksLoaded = false;

function setQuizDiff(d) {
  _quizDiff = d;
  document.querySelectorAll('.quiz-diff-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.diff === d);
  });
}

async function loadQuizBooks() {
  if (_quizBooksLoaded) return;
  const sel = document.getElementById('quiz-book-select');
  sel.innerHTML = '<option value="">جارٍ تحميل…</option>';
  try {
    const r = await fetch(getAPI() + '/books', { headers: getAuthHeaders() });
    if (!r.ok) {
      sel.innerHTML = '<option value="">فشل تحميل الكتب — أعد المحاولة</option>';
      return;
    }
    const data  = await r.json();
    const books = Array.isArray(data) ? data : (data.books || []);
    if (!books.length) {
      sel.innerHTML = '<option value="">لا توجد كتب مرفوعة</option>';
      _quizBooksLoaded = true;
      return;
    }
    sel.innerHTML = '<option value="">— اختر كتاباً —</option>' +
      books.map(b => {
        const t = typeof b === 'string' ? b : (b.title || b.book_title || String(b));
        const escaped = escHtml(t);
        return `<option value="${escaped}">${escaped}</option>`;
      }).join('');
    _quizBooksLoaded = true;
  } catch(e) {
    sel.innerHTML = '<option value="">خطأ في الاتصال — أعد المحاولة</option>';
  }
}

async function generateQuiz() {
  const book = document.getElementById('quiz-book-select').value;
  if (!book) { alert('اختر كتاباً أولاً'); return; }
  const n = document.getElementById('quiz-n-range').value;
  document.getElementById('quiz-setup').style.display         = 'none';
  document.getElementById('quiz-loading').style.display       = 'block';
  document.getElementById('quiz-question-view').style.display = 'none';
  document.getElementById('quiz-result').style.display        = 'none';
  // Timeout: batches now run in parallel → faster than before
  // base 60s + 15s/question (was 90s + 25s) — max 8 min
  const nInt      = parseInt(n);
  const timeoutMs = Math.min(60000 + nInt * 15000, 480000);
  const loadingEl = document.getElementById('quiz-loading-msg');
  if (loadingEl) {
    const estSec = Math.round((60000 + nInt * 12000) / 1000);  // optimistic estimate
    loadingEl.textContent = nInt > 7
      ? `جارٍ توليد ${nInt} سؤال بشكل متوازٍ… تقديراً ${estSec} ثانية`
      : `جارٍ توليد ${nInt} سؤال…`;
  }
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(getAPI() + '/quiz/generate', {
      method: 'POST',
      headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ book_title: book, n_questions: nInt, difficulty: _quizDiff }),
      signal: ctrl.signal,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `خطأ ${r.status}`);
    }
    const data = await r.json();
    _quizQuestions = data.questions || [];
    if (!_quizQuestions.length) throw new Error('لم تُولَّد أسئلة — حاول مجدداً');
    _quizCurrent = 0; _quizScore = 0; _quizAnswered = false;
    renderQuizQuestion();
    document.getElementById('quiz-loading').style.display       = 'none';
    document.getElementById('quiz-question-view').style.display = 'block';
  } catch(e) {
    document.getElementById('quiz-loading').style.display = 'none';
    document.getElementById('quiz-setup').style.display   = 'block';
    const msg = e.name === 'AbortError'
      ? `انتهت المهلة (${Math.round(timeoutMs/60000)} دقيقة) — جرب عدداً أقل من الأسئلة`
      : (e.message || 'خطأ غير معروف');
    alert('⚠️ ' + msg);
  } finally {
    clearTimeout(timer);
  }
}

function renderQuizQuestion() {
  const q     = _quizQuestions[_quizCurrent];
  const total = _quizQuestions.length;
  const pct   = _quizCurrent / total;
  document.getElementById('quiz-score-fill').style.width    = (pct * 100) + '%';
  document.getElementById('quiz-score-display').textContent = _quizScore;
  document.getElementById('quiz-q-num').textContent         = _quizCurrent + 1;
  document.getElementById('quiz-q-total').textContent       = total;
  document.getElementById('quiz-question-text').textContent = q.question || '';
  document.getElementById('quiz-explain-box').style.display = 'none';
  document.getElementById('quiz-next-btn').style.display    = 'none';
  _quizAnswered = false;

  const letters = ['أ', 'ب', 'ج', 'د'];
  const opts    = q.options || [];
  const cont    = document.getElementById('quiz-options-container');
  cont.innerHTML = opts.map((opt, i) => `
    <div class="quiz-option" onclick="chooseQuizAnswer(${i})" id="quiz-opt-${i}">
      <div class="quiz-letter">${letters[i] || i + 1}</div>
      <div style="flex:1;direction:rtl">${escHtml(opt)}</div>
    </div>`).join('');
}

function chooseQuizAnswer(idx) {
  if (_quizAnswered) return;
  _quizAnswered = true;
  const q       = _quizQuestions[_quizCurrent];
  const correct = q.correct ?? 0;
  if (idx === correct) _quizScore++;
  for (let i = 0; i < (q.options || []).length; i++) {
    const el = document.getElementById('quiz-opt-' + i);
    if (!el) continue;
    if (i === correct)               el.classList.add('correct');
    else if (i === idx && i !== correct) el.classList.add('wrong');
  }
  if (q.explanation) {
    const box = document.getElementById('quiz-explain-box');
    box.textContent   = q.explanation;
    box.style.display = 'block';
  }
  const nb = document.getElementById('quiz-next-btn');
  nb.style.display = 'block';
  nb.textContent   = _quizCurrent < _quizQuestions.length - 1 ? 'التالي ▶' : 'عرض النتيجة';
}

function quizNext() {
  if (_quizCurrent < _quizQuestions.length - 1) {
    _quizCurrent++;
    renderQuizQuestion();
  } else {
    showQuizResult();
  }
}

function showQuizResult() {
  document.getElementById('quiz-question-view').style.display = 'none';
  document.getElementById('quiz-result').style.display        = 'block';
  const total = _quizQuestions.length;
  const pct   = Math.round(_quizScore / total * 100);
  document.getElementById('quiz-result-emoji').textContent = pct >= 80 ? '🏆' : pct >= 60 ? '👍' : '📚';
  document.getElementById('quiz-result-score').textContent = `${_quizScore} / ${total}`;
  document.getElementById('quiz-result-pct').textContent   = pct + '%';
  document.getElementById('quiz-result-msg').textContent   = pct >= 80 ? 'ممتاز!' : pct >= 60 ? 'جيد، استمر!' : 'راجع الكتاب مجدداً';
}

function resetQuiz() {
  const prevBook = document.getElementById('quiz-book-select').value;
  document.getElementById('quiz-result').style.display        = 'none';
  document.getElementById('quiz-question-view').style.display = 'none';
  document.getElementById('quiz-setup').style.display         = 'block';
  _quizBooksLoaded = false;
  loadQuizBooks().then(() => {
    const sel = document.getElementById('quiz-book-select');
    if (prevBook && sel) sel.value = prevBook;
  });
}
