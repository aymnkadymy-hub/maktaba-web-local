// ─── Library tab ───────────────────────────────────────────────────
let _libPollTimer    = null;
let _libBooksLoaded  = false;
let _libPollStart    = 0;
const _LIB_POLL_TIMEOUT_MS = 20 * 60 * 1000; // 20 minutes max

async function uploadBook(evt) {
  const file = evt.target.files[0];
  evt.target.value = '';
  if (!file) return;

  const maxMB = 300;
  if (file.size > maxMB * 1024 * 1024) {
    alert(`حجم الملف كبير جداً (${(file.size/1024/1024).toFixed(0)} ميغابايت). الحد الأقصى ${maxMB} ميغابايت.`);
    return;
  }

  const statusDiv = document.getElementById('lib-upload-status');
  const msgDiv    = document.getElementById('lib-upload-msg');
  const barDiv    = document.getElementById('lib-progress-bar');
  const txtDiv    = document.getElementById('lib-progress-text');

  statusDiv.style.display = 'block';
  barDiv.style.background = '';
  barDiv.style.width      = '5%';
  msgDiv.textContent      = `⏫ جارٍ رفع "${file.name}"…`;
  txtDiv.textContent      = `${(file.size/1024/1024).toFixed(1)} ميغابايت`;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const r = await fetch(getAPI() + '/ingest', {
      method: 'POST',
      headers: getAuthHeaders(),
      credentials: 'include',
      body: fd,
    });
    const data = await r.json().catch(() => ({}));

    if (r.status === 202 || r.status === 200) {
      const bookName = data.book || file.name.replace(/\.pdf$/i, '');
      msgDiv.textContent = `⏳ جارٍ المعالجة: "${bookName}"`;
      barDiv.style.width = '10%';
      txtDiv.textContent = 'جارٍ تحليل الصفحات وبناء قاعدة البيانات…';
      _libStartPoll(bookName);
    } else {
      const detail = data.detail || r.statusText || '';
      if (typeof detail === 'string' && detail.startsWith('book_limit_reached:')) {
        const limit = detail.split(':')[1] || '15';
        openBookLimitModal(limit);
        statusDiv.style.display = 'none';
      } else {
        msgDiv.textContent = `❌ خطأ: ${detail}`;
        barDiv.style.width = '0%';
      }
    }
  } catch(e) {
    msgDiv.textContent = `❌ تعذّر الاتصال بالسيرفر`;
  }
}

function _libStartPoll(bookName) {
  clearInterval(_libPollTimer);
  _libPollStart = Date.now();
  _libPollTimer = setInterval(() => _libPollProgress(bookName), 2500);
}

async function _libPollProgress(bookName) {
  try {
    const r = await fetch(
      getAPI() + '/ingest/progress/' + encodeURIComponent(bookName),
      { headers: getAuthHeaders() }
    );
    if (!r.ok) return;
    const d = await r.json();

    const msgDiv = document.getElementById('lib-upload-msg');
    const barDiv = document.getElementById('lib-progress-bar');
    const txtDiv = document.getElementById('lib-progress-text');
    if (!msgDiv) return;

    if (d.status === 'ingesting') {
      const done  = d.pages_done   || 0;
      const total = d.total_pages  || 1;
      const cDone = d.chunks_done  || 0;
      const cTot  = d.total_chunks || 0;

      if (cTot > 0) {
        // Vector store insertion phase
        const pct = Math.min(83 + Math.round((cDone / cTot) * 14), 97);
        barDiv.style.width = pct + '%';
        txtDiv.textContent = `جاري إضافة المقاطع: ${cDone} / ${cTot}`;
      } else {
        // Page extraction phase
        const pct = Math.min(10 + Math.round((done / total) * 72), 82);
        barDiv.style.width = pct + '%';
        txtDiv.textContent = `الصفحات: ${done} / ${total}`;
      }

      // Safety timeout
      if (Date.now() - _libPollStart > _LIB_POLL_TIMEOUT_MS) {
        clearInterval(_libPollTimer);
        msgDiv.textContent = '⚠️ انتهت مدة الانتظار — تحقق من السيرفر أو أعد رفع الكتاب';
        txtDiv.textContent = '';
      }
    } else if (d.status === 'done') {
      clearInterval(_libPollTimer);
      barDiv.style.width = '100%';
      msgDiv.textContent = `✅ اكتمل! (${d.chunks || '?'} مقطع · ${d.total_pages || '?'} صفحة)`;
      txtDiv.textContent = 'الكتاب جاهز للاستخدام في الدردشة';
      _quizBooksLoaded = false; // force quiz dropdown to refresh
      setTimeout(loadLibraryBooks, 1200);
    } else if (d.status === 'failed') {
      clearInterval(_libPollTimer);
      barDiv.style.background = '#e53935';
      barDiv.style.width      = '100%';
      msgDiv.textContent      = `❌ فشل: ${d.error || 'خطأ غير معروف'}`;
      txtDiv.textContent      = '';
    }
  } catch(_) { /* network hiccup — keep polling */ }
}

const _MAX_BOOKS = 20;

function _updateUploadZoneState(count) {
  const warning   = document.getElementById('lib-limit-warning');
  const label     = document.getElementById('lib-upload-label');
  const usedSpan  = document.getElementById('lib-limit-used');
  const fileInput = document.getElementById('lib-file-input');
  if (!warning || !label) return;
  const atLimit = count >= _MAX_BOOKS;
  warning.style.display = atLimit ? 'block' : 'none';
  if (usedSpan) usedSpan.textContent = count;
  label.classList.toggle('disabled', atLimit);
  if (fileInput) fileInput.disabled = atLimit;
}

async function loadLibraryBooks() {
  const listDiv = document.getElementById('lib-book-list');
  if (!listDiv) return;
  listDiv.innerHTML = '<div style="text-align:center;padding:24px;color:#9e9e9e">جارٍ التحميل…</div>';
  try {
    const r = await fetch(getAPI() + '/books', {
      headers: getAuthHeaders(),
      credentials: 'include',
    });
    if (r.status === 401) {
      _libBooksLoaded = false;
      listDiv.innerHTML = '<div style="text-align:center;padding:24px;color:#9e9e9e">سجّل دخولك لعرض المكتبة</div>';
      if (typeof handle401 === 'function') handle401();
      return;
    }
    if (!r.ok) {
      _libBooksLoaded = false;
      listDiv.innerHTML = '<div style="text-align:center;padding:24px;color:#e53935">تعذّر تحميل المكتبة</div>';
      return;
    }
    const data  = await r.json();
    const books = Array.isArray(data) ? data : (data.books || []);
    _updateUploadZoneState(books.length);
    if (!books.length) {
      _libBooksLoaded = true;
      listDiv.innerHTML = '<div style="text-align:center;padding:40px;color:#9e9e9e">المكتبة فارغة — ارفع كتاباً للبدء</div>';
      return;
    }
    _libBooksLoaded = true;
    listDiv.innerHTML = books.map(b => {
      const title        = typeof b === 'string' ? b : (b.title || b.book_title || String(b));
      const chunks       = b.chunks        ? `${b.chunks} مقطع` : '';
      const raptorChunks = b.raptor_chunks ? ` · ${b.raptor_chunks} ملخص` : '';
      const safeTitle    = escHtml(title);
      return `<div class="lib-book-card">
        <div class="lib-book-icon">📖</div>
        <div style="flex:1">
          <div class="lib-book-title">${safeTitle}</div>
          <div class="lib-book-meta">${chunks}${raptorChunks}</div>
        </div>
        <button class="lib-delete-btn" data-title="${safeTitle}" title="حذف الكتاب">🗑 حذف</button>
      </div>`;
    }).join('');
    // Attach listeners after rendering — inline onclick breaks on titles
    // containing quotes (same pattern as notes.js)
    listDiv.querySelectorAll('.lib-delete-btn').forEach(btn => {
      btn.addEventListener('click', () => deleteBook(btn.dataset.title));
    });
  } catch(e) {
    _libBooksLoaded = false;
    listDiv.innerHTML = `<div style="text-align:center;padding:24px;color:#e53935">خطأ: ${escHtml(String(e))}</div>`;
  }
}

function openBookLimitModal(limit) {
  document.getElementById('book-limit-count').textContent = limit;
  document.getElementById('book-limit-overlay').style.display = 'flex';
}
function closeBookLimitModal() {
  document.getElementById('book-limit-overlay').style.display = 'none';
}

async function deleteBook(title) {
  if (!confirm(`حذف كتاب "${title}" نهائياً؟`)) return;
  try {
    const r = await fetch(getAPI() + '/books/' + encodeURIComponent(title), {
      method: 'DELETE',
      headers: getAuthHeaders(),
      credentials: 'include',
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok) {
      _quizBooksLoaded = false;
      loadLibraryBooks();
    } else {
      alert(data.detail || 'تعذّر الحذف');
    }
  } catch(e) {
    alert('تعذّر الاتصال بالسيرفر');
  }
}
