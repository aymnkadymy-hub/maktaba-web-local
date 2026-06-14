// XSS-escaping invariants for static/js/markdown.js
// Run with:  node --test tests/js/
'use strict';
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// markdown.js is DOM-free — load its globals into a bare context
const src = fs.readFileSync(
  path.join(__dirname, '..', '..', 'static', 'js', 'markdown.js'), 'utf8');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(src, ctx);
const { escHtml, renderMarkdown } = ctx;

test('escHtml escapes all HTML-significant characters', () => {
  assert.strictEqual(escHtml('<script>"&'), '&lt;script&gt;&quot;&amp;');
  assert.strictEqual(escHtml(''), '');
  assert.strictEqual(escHtml(123), '123', 'non-strings coerced safely');
});

test('renderMarkdown neutralizes script tags', () => {
  const html = renderMarkdown('<script>alert(1)</script>');
  assert.ok(!html.includes('<script>'), 'raw <script> must never pass through');
  assert.ok(html.includes('&lt;script&gt;'));
});

test('renderMarkdown neutralizes event-handler injection', () => {
  const html = renderMarkdown('<img src=x onerror=alert(1)>');
  assert.ok(!html.includes('<img'), 'raw tags must be escaped');
});

test('payloads inside markdown structures are still escaped', () => {
  for (const md of [
    '**<svg onload=alert(1)>**',
    '# <iframe src=evil>',
    '- <b onclick=x>item',
    '> quoted <script>x</script>',
    '`<script>inline</script>`',
    '```\n<script>fenced</script>\n```',
  ]) {
    const html = renderMarkdown(md);
    assert.ok(!/<(script|svg|iframe|b onclick|img)\b/.test(html),
      `unescaped tag leaked from: ${md}\n→ ${html}`);
  }
});

test('legitimate markdown still renders', () => {
  assert.ok(renderMarkdown('**غامق**').includes('<strong>غامق</strong>'));
  assert.ok(renderMarkdown('# عنوان').includes('<h1>عنوان</h1>'));
  assert.ok(renderMarkdown('- بند').includes('<li>بند</li>'));
  assert.ok(renderMarkdown('`كود`').includes('<code>كود</code>'));
  assert.strictEqual(renderMarkdown(''), '');
});
