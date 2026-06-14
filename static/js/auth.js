// ─── Auth UI helpers ───────────────────────────────────────────────
function showAuthOverlay() {
  document.getElementById('auth-overlay').style.display = 'flex';
}
function hideAuthOverlay() {
  document.getElementById('auth-overlay').style.display = 'none';
}

function showAuthTab(tab) {
  document.getElementById('login-form').style.display    = tab === 'login'    ? 'flex' : 'none';
  document.getElementById('register-form').style.display = tab === 'register' ? 'flex' : 'none';
  document.querySelectorAll('.auth-tab').forEach((b, i) =>
    b.classList.toggle('active', (i === 0) === (tab === 'login')));
}

function handle401() {
  _authUser = null;
  localStorage.removeItem('maktaba_auth_user');
  document.getElementById('username-display').textContent = '';
  document.getElementById('logout-btn').style.display = 'none';
  showAuthOverlay();
}

function onLoginSuccess() {
  hideAuthOverlay();
  document.getElementById('username-display').textContent = '👤 ' + _authUser;
  document.getElementById('logout-btn').style.display = '';
  startNewSession();
  loadServerHistory();
  pingServer();
}

// ─── Login ────────────────────────────────────────────────────────
async function doLogin(e) {
  e.preventDefault();
  const btn = document.getElementById('login-btn');
  const err = document.getElementById('login-error');
  btn.disabled = true; err.textContent = '';
  const user = document.getElementById('login-user').value.trim();
  const pass = document.getElementById('login-pass').value;
  try {
    const r = await fetch(getAPI() + '/auth/login', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password: pass }),
    });
    const data = await r.json();
    if (!r.ok) { err.textContent = data.detail || 'خطأ في تسجيل الدخول'; btn.disabled = false; return; }
    // Token is in HttpOnly cookie — only store username for display
    _authUser = data.username;
    localStorage.setItem('maktaba_auth_user', _authUser);
    onLoginSuccess();
  } catch {
    err.textContent = 'تعذر الاتصال بالسيرفر';
    btn.disabled = false;
  }
}

// ─── Register ─────────────────────────────────────────────────────
async function doRegister(e) {
  e.preventDefault();
  const btn  = document.getElementById('reg-btn');
  const err  = document.getElementById('reg-error');
  btn.disabled = true; err.textContent = '';
  const user  = document.getElementById('reg-user').value.trim();
  const pass  = document.getElementById('reg-pass').value;
  const pass2 = document.getElementById('reg-pass2').value;
  if (pass !== pass2) { err.textContent = 'كلمتا المرور غير متطابقتين'; btn.disabled = false; return; }
  try {
    const r = await fetch(getAPI() + '/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password: pass }),
    });
    const data = await r.json();
    if (!r.ok) { err.textContent = data.detail || 'خطأ في إنشاء الحساب'; btn.disabled = false; return; }
    // Auto-login after successful registration
    document.getElementById('login-user').value = user;
    document.getElementById('login-pass').value = pass;
    showAuthTab('login');
    document.getElementById('login-btn').disabled = false;
    document.getElementById('login-form').dispatchEvent(new Event('submit', { cancelable: true }));
  } catch {
    err.textContent = 'تعذر الاتصال بالسيرفر';
    btn.disabled = false;
  }
}

// ─── Logout ───────────────────────────────────────────────────────
async function doLogout() {
  try {
    // Cookie is sent automatically (same-origin); server revokes it and clears Set-Cookie
    await fetch(getAPI() + '/auth/logout', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    });
  } catch {}
  handle401();
}

// ─── Check auth on page load ───────────────────────────────────────
async function checkAuth() {
  // No client-side token to check — always verify via server (cookie is sent
  // automatically). Two attempts: a short pause between them covers a server
  // that is still starting, without visibly freezing startup.
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const r = await fetch(getAPI() + '/auth/me', {
        credentials: 'include',
        signal: AbortSignal.timeout(4000),
      });
      if (r.ok) {
        const data = await r.json();
        _authUser = data.username;
        localStorage.setItem('maktaba_auth_user', _authUser);
        document.getElementById('username-display').textContent = '👤 ' + _authUser;
        document.getElementById('logout-btn').style.display = '';
        return;
      }
      break; // server answered (401/…) — no point retrying
    } catch {
      if (attempt === 0) await new Promise(res => setTimeout(res, 600));
    }
  }
  // Failed or unauthorized — show login overlay (don't trust localStorage alone)
  handle401();
}
