// ─── Notes logic ───────────────────────────────────────────────────
let _allNotes = [];

async function saveNoteFromBubble(text, btn) {
  if (!_authUser) { alert('سجّل دخولك أولاً لحفظ الملاحظات'); return; }
  const plain = text.replace(/<[^>]+>/g, '').trim();
  if (!plain) return;
  try {
    const r = await fetch(getAPI() + '/notes/', {
      method: 'POST',
      headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: plain, note: '', session_id: sessionId, book_title: '' }),
    });
    if (r.ok) {
      btn.textContent = '✅ تم الحفظ';
      setTimeout(() => { btn.textContent = '📌 حفظ ملاحظة'; }, 2000);
    } else {
      alert('فشل الحفظ');
    }
  } catch { alert('فشل الحفظ'); }
}

async function loadNotes() {
  const list = document.getElementById('notes-list');
  list.innerHTML = '<div style="text-align:center;padding:40px;color:#9e9e9e">جارٍ التحميل…</div>';
  try {
    const r = await fetch(getAPI() + '/notes/', { headers: getAuthHeaders() });
    if (!r.ok) throw new Error('auth');
    _allNotes = await r.json();
    renderNotes(_allNotes);
  } catch {
    list.innerHTML = '<div style="text-align:center;padding:40px;color:#9e9e9e">فشل تحميل الملاحظات — تأكد من تسجيل الدخول</div>';
  }
}

function renderNotes(notes) {
  const list = document.getElementById('notes-list');
  if (!notes.length) {
    list.innerHTML = '<div style="text-align:center;padding:40px;color:#9e9e9e">لا توجد ملاحظات بعد<br><small>اضغط طويلاً على رد البوت لحفظه</small></div>';
    return;
  }
  list.innerHTML = notes.map(n => {
    const content  = (n.content    || '').substring(0, 200);
    const personal = n.note        || '';
    const book     = n.book_title  || '';
    const date     = n.created_at  ? n.created_at.substring(0, 10) : '';
    const noteData = JSON.stringify(n.note || '');   // safe JS string — handles apostrophes
    return `<div class="note-card" data-id="${n.id}" data-note="${escHtml(n.note||'')}">
      <button class="note-del-btn" data-id="${n.id}" onclick="event.stopPropagation();deleteNote('${n.id}')">✕</button>
      ${book ? `<div class="note-book-tag">${escHtml(book)}</div>` : ''}
      <div class="note-content-preview">${escHtml(content)}${n.content?.length > 200 ? '…' : ''}</div>
      ${personal ? `<div class="note-personal">✏️ ${escHtml(personal)}</div>` : ''}
      <div class="note-date">${date}</div>
    </div>`;
  }).join('');
  // Attach click listeners after rendering (avoids inline onclick JS-injection risk)
  list.querySelectorAll('.note-card').forEach(card => {
    card.addEventListener('click', () => editNote(card.dataset.id, card.dataset.note));
  });
}

function filterNotes(q) {
  const lq = q.toLowerCase();
  const filtered = lq ? _allNotes.filter(n =>
    (n.content    || '').toLowerCase().includes(lq) ||
    (n.note       || '').toLowerCase().includes(lq) ||
    (n.book_title || '').toLowerCase().includes(lq)
  ) : _allNotes;
  renderNotes(filtered);
}

async function deleteNote(id) {
  if (!confirm('حذف هذه الملاحظة؟')) return;
  await fetch(getAPI() + '/notes/' + id, { method: 'DELETE', headers: getAuthHeaders() });
  loadNotes();
}

async function editNote(id, currentText) {
  const newText = prompt('تعديل الملاحظة:', currentText);
  if (newText === null) return;
  await fetch(getAPI() + '/notes/' + id, {
    method: 'PATCH',
    headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ note: newText }),
  });
  loadNotes();
}
