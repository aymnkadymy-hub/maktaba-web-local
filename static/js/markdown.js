// ─── HTML escaping (used by all modules) ──────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─── Markdown renderer ─────────────────────────────────────────────
function inlineFmt(text) {
  text = escHtml(text);   // must escape before injecting any HTML tags
  text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  text = text.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
  text = text.replace(/\*(.+?)\*/g,         '<em>$1</em>');
  return text;
}

function renderMarkdown(md) {
  if (!md) return '';

  // Stash fenced code blocks
  const blocks = [];
  md = md.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${escHtml(code.trim())}</code></pre>`);
    return `\x00BLK${blocks.length - 1}\x00`;
  });

  // Inline code — stash like fenced blocks; substituting the <code> tag
  // directly would get re-escaped by inlineFmt() and render as literal text
  md = md.replace(/`([^`\n]+)`/g, (_, c) => {
    blocks.push(`<code>${escHtml(c)}</code>`);
    return `\x00BLK${blocks.length - 1}\x00`;
  });

  const lines = md.split('\n');
  let html = '', listTag = '', inList = false;

  const closeList = () => {
    if (inList) { html += `</${listTag}>`; inList = false; listTag = ''; }
  };

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];

    // Heading
    const hm = raw.match(/^(#{1,6})\s+(.*)/);
    if (hm) {
      closeList();
      const lv = hm[1].length;
      html += `<h${lv}>${inlineFmt(hm[2])}</h${lv}>`;
      continue;
    }

    // Unordered list
    const ul = raw.match(/^[-*+]\s+(.*)/);
    if (ul) {
      if (listTag !== 'ul') { closeList(); html += '<ul>'; inList = true; listTag = 'ul'; }
      html += `<li>${inlineFmt(ul[1])}</li>`;
      continue;
    }

    // Ordered list
    const ol = raw.match(/^\d+\.\s+(.*)/);
    if (ol) {
      if (listTag !== 'ol') { closeList(); html += '<ol>'; inList = true; listTag = 'ol'; }
      html += `<li>${inlineFmt(ol[1])}</li>`;
      continue;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(raw.trim())) { closeList(); html += '<hr>'; continue; }

    // Blockquote
    const bq = raw.match(/^>\s?(.*)/);
    if (bq) { closeList(); html += `<blockquote>${inlineFmt(bq[1])}</blockquote>`; continue; }

    // Blank line → paragraph break
    if (!raw.trim()) { closeList(); html += '</p><p>'; continue; }

    // Normal line
    closeList();
    html += inlineFmt(raw) + '<br>';
  }

  closeList();

  // Wrap in paragraph and clean up empty ones
  html = `<p>${html}</p>`.replace(/<p><\/p>/g, '').replace(/<p>(<[huo])/g,'$1').replace(/(<\/[huo][^>]*>)<\/p>/g,'$1');

  // Restore code blocks
  blocks.forEach((b, i) => { html = html.replace(`\x00BLK${i}\x00`, b); });

  return html;
}
