// app.js — panel switching and the read-only inspection panels (memory,
// research, beliefs). Loads last; wires everything.
'use strict';

function memoryCard(m) {
  const tags = ['<span class="tag">' + esc(m.kind) + '</span>',
                '<span class="tag">' + esc(m.provenance) + '</span>'];
  if (m.disputed) tags.push('<span class="tag disputed">re-read</span>');
  if (m.archived) tags.push('<span class="tag">archived</span>');
  const card = el('<div class="card">' + tags.join('') +
    '<span class="when">turn ' + esc(m.turn_idx == null ? '—' : m.turn_idx) +
    ' · salience ' + Number(m.salience).toFixed(2) +
    (m.importance_revised
      ? ' · importance ' + Number(m.importance).toFixed(2) : '') +
    ' · confidence ' + Number(m.confidence).toFixed(2) + '</span>' +
    '<div>' + esc(m.gist || m.content) + '</div>' +
    (escUrl(m.source_url)
      ? '<div class="when"><a href="' + escUrl(m.source_url) +
        '" target="_blank" rel="noopener">' + esc(m.source_url) + '</a></div>'
      : m.source_url ? '<div class="when">' + esc(m.source_url) + '</div>'
      : '') +
    (m.disputed
      ? '<div class="reading">now read as: ' + esc(m.disputed.reading) +
        '</div>'
      : '') + '</div>');
  return card;
}

async function loadMemories(query) {
  const box = document.getElementById('memories');
  box.textContent = 'loading…';
  const rows = await api('/api/memories?query=' +
    encodeURIComponent(query || ''));
  box.textContent = '';
  if (!rows.length) box.textContent = 'no memories yet';
  rows.slice().reverse().forEach(m => box.appendChild(memoryCard(m)));
}

async function loadResearch() {
  const box = document.getElementById('hypotheses');
  box.textContent = 'loading…';
  const rows = await api('/api/hypotheses');
  box.textContent = '';
  if (!rows.length) box.textContent = 'no research yet';
  for (const h of rows) {
    const card = el('<div class="card">' +
      '<span class="tag' + (h.status === 'disputed' ? ' disputed' : '') +
      '">' + esc(h.status) + '</span>' +
      '<span class="when">confidence ' +
      Number(h.confidence).toFixed(2) + '</span>' +
      '<div><strong>' + esc(h.question) + '</strong></div>' +
      (h.statement ? '<div>' + esc(h.statement) + '</div>' : '') +
      '<div class="evidence"></div></div>');
    box.appendChild(card);
    api('/api/hypotheses/' + h.id + '/evidence').then(evs => {
      const holder = card.querySelector('.evidence');
      for (const e of evs) {
        holder.appendChild(el('<div class="when">' +
          '<span class="tag ' + esc(e.stance) + '">' + esc(e.stance) +
          '</span>[' + esc(e.ref) + '] ' +
          (escUrl(e.url)
            ? '<a href="' + escUrl(e.url) + '" target="_blank" ' +
              'rel="noopener">' + esc(e.title || e.url) + '</a>'
            : esc(e.title || e.url)) +
          ' — ' + esc(e.excerpt) + '</div>'));
      }
    });
  }
}

async function loadBeliefs() {
  const box = document.getElementById('beliefs');
  box.textContent = 'loading…';
  const data = await api('/api/beliefs');
  box.textContent = '';
  const sheet = data.active_hypotheses || [];
  if (sheet.length) {
    const card = el('<div class="card"><strong>Actively wondering</strong>' +
      '</div>');
    for (const s of sheet) {
      card.appendChild(el('<div class="when">' + esc(s.about) + ' · ' +
        esc(s.kind) + ' · ' + Number(s.confidence).toFixed(2) +
        ' — <em>I suspect: ' + esc(s.i_suspect) + '</em></div>'));
    }
    box.appendChild(card);
  }
  const model = data.user_model || {};
  for (const about of Object.keys(model)) {
    const card = el('<div class="card"><strong>' + esc(about) +
      '</strong></div>');
    for (const kind of Object.keys(model[about])) {
      const entry = model[about][kind];
      card.appendChild(el('<div class="when">' + esc(kind) + ': ' +
        esc(entry.leading.claim) + ' (' +
        Number(entry.leading.confidence).toFixed(2) + ')' +
        (entry.competitors.length
          ? ' — still weighing: ' + entry.competitors.map(c =>
              esc(c.claim) + ' (' + Number(c.confidence).toFixed(2) + ')')
              .join('; ')
          : '') + '</div>'));
    }
    box.appendChild(card);
  }
  if (!sheet.length && !Object.keys(model).length) {
    box.textContent = 'no beliefs formed yet';
  }
}

document.addEventListener('DOMContentLoaded', function () {
  initChat();
  document.querySelectorAll('nav button').forEach(btn => {
    btn.addEventListener('click', function () {
      document.querySelectorAll('nav button, .panel')
        .forEach(n => n.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('panel-' + btn.dataset.panel)
        .classList.add('active');
      if (btn.dataset.panel === 'history') loadHistory();
      if (btn.dataset.panel === 'memory') loadMemories('');
      if (btn.dataset.panel === 'research') loadResearch();
      if (btn.dataset.panel === 'beliefs') loadBeliefs();
    });
  });
  document.getElementById('memsearch').addEventListener('submit',
    function (ev) {
      ev.preventDefault();
      loadMemories(document.getElementById('memquery').value);
    });
});
