// ─── Init ──────────────────────────────────────────────────────────
_initMic();
renderSidebar();
pingServer();
checkAuth().then(() => {
  if (_authUser) loadServerHistory();
});

window.addEventListener('beforeunload', () => {
  if (_libPollTimer) clearInterval(_libPollTimer);
});
