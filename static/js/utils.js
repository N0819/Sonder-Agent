// utils.js — shared helpers. Browser globals, no modules (engine convention:
// script order in index.html is the dependency graph; never rename a shared
// function without grepping every file).
'use strict';

// esc() is safe in ELEMENT TEXT and nowhere else. textContent->innerHTML is
// the DOM's own serialisation, which escapes & < > and deliberately leaves
// quotes alone — correct for text, wrong the moment the result lands inside
// href="...". Quotes are escaped here too so one helper is safe in both.
function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s == null ? '' : s);
  return d.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Escaping is not enough for a URL: `javascript:alert(1)` contains nothing
// that needs escaping and is still a script the user runs by clicking. The
// urls rendered here come from search results and from model-chosen fetches
// — untrusted by this project's own doctrine — and research.canonical_url
// passes any non-http scheme through untouched by design, because it also
// carries internal refs like `experiment:<digest>`. So the scheme check
// belongs at the render seam, which is here.
function escUrl(u) {
  const s = String(u == null ? '' : u).trim();
  return /^https?:\/\//i.test(s) ? esc(s) : '';
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({
    headers: { 'Content-Type': 'application/json' },
  }, opts || {}));
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).error || ''; } catch (e) { /* opaque */ }
    throw new Error(detail || (res.status + ' ' + res.statusText));
  }
  return res.json();
}

function el(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstChild;
}
