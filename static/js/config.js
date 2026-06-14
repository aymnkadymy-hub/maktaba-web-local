// ─── Config ────────────────────────────────────────────────────────
const API_KEY     = 'maktaba_api_url';
const DEFAULT_API = 'http://localhost:8000';

function getAPI() {
  const stored = localStorage.getItem(API_KEY);
  if (stored) return stored.replace(/\/$/, '');
  // Auto-detect: use the same origin that served this page (works for tunnel URLs too)
  const origin = window.location.origin;
  if (origin && origin !== 'null' && !origin.startsWith('file')) return origin;
  return DEFAULT_API;
}

// ─── Auth state ────────────────────────────────────────────────────
// Auth uses HttpOnly cookies — no JS-accessible token.
// _authUser is kept in localStorage for display purposes only.
let _authUser  = localStorage.getItem('maktaba_auth_user') || null;

function getStorageKey() { return `maktaba_history_${_authUser || 'guest'}`; }
// Cookie sent automatically for same-origin requests — no Authorization header needed.
function getAuthHeaders() { return {}; }

// ─── Session state ─────────────────────────────────────────────────
let sessionId       = crypto.randomUUID();
let sessionMessages = [];   // {role, content, ts}
let streaming       = false;
let viewingHistory  = false; // true when browsing a past session
let _serverSessions = [];   // [{id, topic, created_at}] from server
let _viewingSession = null; // session object currently being viewed
let _attachedImage  = null; // {file, dataURL} when image is attached
