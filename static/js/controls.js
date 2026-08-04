// controls.js — the panels that CHANGE something: files, subagent
// permissions, provider settings, persona. app.js owns the read-only
// inspection panels; this file owns the ones with consequences, and the
// separation is deliberate — a panel that can spend money or grant autonomy
// should not be sitting in the same file as a memory viewer.
//
// Browser globals, no bundler; loaded after app.js (engine convention: script
// order in index.html is the dependency graph).
'use strict';

function bytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

// ---- Files ----

function renderFiles(payload) {
  const box = document.getElementById('files');
  box.innerHTML = '';
  const files = (payload && payload.files) || [];
  if (!files.length) {
    box.appendChild(el('<p class="muted">Nothing uploaded yet.</p>'));
    return;
  }
  for (const f of files) {
    const row = el('<div class="card"><div class="when">' + bytes(f.bytes) +
      '</div><div class="path"></div><div class="row"></div></div>');
    // textContent, not interpolation: a filename is user-supplied text and
    // this is the one place it becomes markup.
    row.querySelector('.path').textContent = f.path;
    const actions = row.querySelector('.row');
    if (f.archive) {
      const ex = el('<button>Extract</button>');
      ex.addEventListener('click', () => fileAction('extract', f.path));
      actions.appendChild(ex);
    }
    const del = el('<button class="danger">Delete</button>');
    del.addEventListener('click', () => fileAction('delete', f.path));
    actions.appendChild(del);
    box.appendChild(row);
  }
}

async function loadFiles() {
  const sid = currentSessionId();
  if (!sid) {
    const box = document.getElementById('files');
    box.innerHTML = '';
    box.appendChild(el('<p class="muted"></p>')).textContent =
      'Say something in Chat first — files attach to a session.';
    return;
  }
  try {
    renderFiles(await api('/api/files?session_id=' + sid));
  } catch (e) {
    setFileStatus('could not list files: ' + e.message);
  }
}

function setFileStatus(text) {
  document.getElementById('filestatus').textContent = text || '';
}

async function fileAction(action, path) {
  const sid = currentSessionId();
  if (!sid) return;
  setFileStatus(action === 'extract' ? 'extracting…' : 'deleting…');
  try {
    const out = await api('/api/files/' + action, {
      method: 'POST',
      body: JSON.stringify({ session_id: sid, path: path }),
    });
    // A refusal is a REASON, not a failure: workspace.py returns "that
    // archive expands 1029x" rather than throwing, and the user needs to
    // read it.
    if (out.ok === false) setFileStatus(out.error || 'refused');
    else if (action === 'extract') {
      setFileStatus('extracted ' + out.written + ' file(s) into ' + out.into +
        (out.refused && out.refused.length
          ? ' — refused ' + out.refused.length + ' unsafe member(s)' : ''));
    } else setFileStatus('');
    renderFiles(out);
  } catch (e) {
    setFileStatus(e.message);
  }
}

async function uploadFiles(fileList) {
  const sid = currentSessionId();
  if (!sid) {
    setFileStatus('say something in Chat first — files attach to a session.');
    return;
  }
  if (!fileList || !fileList.length) return;
  const form = new FormData();
  form.append('session_id', sid);
  for (const f of fileList) form.append('files', f, f.name);
  setFileStatus('uploading ' + fileList.length + ' file(s)…');
  try {
    // Not `api()`: FormData must set its own multipart boundary, and the
    // JSON Content-Type header api() adds would break the parse server-side.
    const res = await fetch('/api/files', { method: 'POST', body: form });
    const out = await res.json();
    const refused = (out.results || []).filter(r => r.ok === false);
    setFileStatus(refused.length
      ? refused.map(r => r.filename + ': ' + r.error).join('; ')
      : 'uploaded ' + (out.results || []).length + ' file(s)');
    renderFiles(out);
  } catch (e) {
    setFileStatus(e.message);
  }
}

function initFiles() {
  const zone = document.getElementById('dropzone');
  // dragover MUST be prevented or the browser navigates to the file, which
  // silently discards whatever the user was doing.
  ['dragenter', 'dragover'].forEach(name =>
    zone.addEventListener(name, ev => {
      ev.preventDefault();
      zone.classList.add('dragging');
    }));
  ['dragleave', 'drop'].forEach(name =>
    zone.addEventListener(name, ev => {
      ev.preventDefault();
      zone.classList.remove('dragging');
    }));
  zone.addEventListener('drop', ev => uploadFiles(ev.dataTransfer.files));
  document.getElementById('filepick').addEventListener('change', function () {
    uploadFiles(this.files);
    this.value = '';
  });
  // A file dropped anywhere else must not navigate away either.
  window.addEventListener('dragover', ev => ev.preventDefault());
  window.addEventListener('drop', ev => ev.preventDefault());
}

// ---- Retired memories ----

async function loadRetired() {
  let out;
  try {
    out = await api('/api/memories/retired');
  } catch (e) { return; }
  const box = document.getElementById('retired');
  box.innerHTML = '';
  const rows = out.retired || [];
  if (!rows.length) {
    box.appendChild(el('<p class="muted">Nothing set aside.</p>'));
    return;
  }
  // Grouped by the batch that retired them: a retirement is one judgement
  // about one topic, and reviewing it row by row misrepresents the decision.
  const batches = new Map();
  for (const r of rows) {
    if (!batches.has(r.batch)) batches.set(r.batch, []);
    batches.get(r.batch).push(r);
  }
  for (const [batch, items] of batches) {
    const card = el('<div class="card retired"><div class="when"></div>' +
      '<div class="why"></div><ul class="items"></ul>' +
      '<div class="row"></div></div>');
    card.querySelector('.when').textContent =
      items.length + ' memories, set aside at turn ' +
      (items[0].retired_at_turn ?? '?');
    card.querySelector('.why').textContent = 'Reason: ' + (items[0].reason || '(none)');
    const list = card.querySelector('.items');
    for (const item of items.slice(0, 12)) {
      const li = el('<li></li>');
      li.textContent = item.gist;
      list.appendChild(li);
    }
    const restore = el('<button>Put these back</button>');
    restore.addEventListener('click', () => restoreBatch(batch));
    card.querySelector('.row').appendChild(restore);
    const purge = el('<button class="danger">Delete for good</button>');
    purge.addEventListener('click', () => purgeBatch(batch, items.length));
    card.querySelector('.row').appendChild(purge);
    box.appendChild(card);
  }
}

async function restoreBatch(batch) {
  const out = await api('/api/memories/restore', {
    method: 'POST', body: JSON.stringify({ batch: batch }),
  });
  document.getElementById('retired-status').textContent =
    'restored ' + out.restored + ' memories';
  loadRetired();
}

async function purgeBatch(batch, count) {
  // Irreversible, so it asks. Retiring is the assistant's call; destroying
  // the record is yours.
  if (!window.confirm('Permanently delete ' + count + ' memories? This ' +
      'cannot be undone. "Put these back" is the reversible option.')) return;
  const out = await api('/api/memories/purge', {
    method: 'POST', body: JSON.stringify({ batch: batch, confirm: true }),
  });
  document.getElementById('retired-status').textContent =
    'deleted ' + out.purged + ' memories' +
    (out.kept_because_cited_by_a_summary
      ? '; kept ' + out.kept_because_cited_by_a_summary +
        ' still cited by a summary' : '');
  loadRetired();
}

// ---- Subagents ----

async function loadAgents() {
  let out;
  try {
    out = await api('/api/subagents');
  } catch (e) { return; }
  const req = document.getElementById('agent-requests');
  req.innerHTML = '';
  for (const r of out.requests || []) {
    const card = el('<div class="card request"><div class="when"></div>' +
      '<div class="why"></div><div class="row"></div></div>');
    card.querySelector('.when').textContent =
      'The assistant asked for ' + r.count + ' ' + r.kind + ' subagent(s)';
    card.querySelector('.why').textContent = r.why;
    const ok = el('<button>Allow ' + Number(r.count) + '</button>');
    ok.addEventListener('click', () => grantAgents(r.kind, r.count));
    card.querySelector('.row').appendChild(ok);
    req.appendChild(card);
  }
  const box = document.getElementById('agent-grants');
  box.innerHTML = '';
  for (const kind of out.kinds) {
    const a = out.allowance[kind] || {};
    const card = el('<div class="card"><div class="when"></div>' +
      '<div class="row"></div></div>');
    card.querySelector('.when').textContent =
      kind + ' — ' + a.remaining + ' remaining (' + a.used + ' used of ' +
      a.granted + ' granted)';
    const row = card.querySelector('.row');
    const input = el('<input type="number" min="0" value="1" style="width:5em">');
    input.max = out.max_grant[kind];
    const give = el('<button>Allow</button>');
    give.addEventListener('click', () => grantAgents(kind, input.value));
    row.appendChild(input);
    row.appendChild(give);
    box.appendChild(card);
  }
  const badge = document.getElementById('agents-badge');
  badge.hidden = !(out.requests || []).length;
  loadAgentArchives();
}

// What a deep subagent actually did, after the fact.
//
// The endpoint has existed since subagents were built and nothing ever
// called it: the tarballs were on disk and unreachable from the interface,
// which for the person trying to audit a run is the same as not existing.
// Metadata only, deliberately — subagents.py argues that opening one should
// be an act, not something a listing does for you.
async function loadAgentArchives() {
  const box = document.getElementById('agent-archives');
  if (!box) return;
  let out;
  try {
    out = await api('/api/subagents/archives');
  } catch (e) { return; }
  box.innerHTML = '';
  const runs = out.archives || [];
  if (!runs.length) {
    const empty = el('<div class="muted"></div>');
    empty.textContent = 'No deep runs archived yet. They appear here after a '
      + 'deep subagent finishes, and the newest ' + out.keep_runs + ' are kept.';
    box.appendChild(empty);
    return;
  }
  for (const r of runs) {
    const card = el('<div class="card"><div class="when"></div>' +
                    '<div class="why"></div><div class="muted"></div></div>');
    card.querySelector('.when').textContent =
      (r.kind || 'deep') + ' — ' + new Date((r.archived || 0) * 1000)
        .toLocaleString();
    card.querySelector('.why').textContent = r.task || '(no task recorded)';
    card.querySelector('.muted').textContent =
      r.id + '  ·  ' + Math.round((r.bytes || 0) / 1024) + ' KB  ·  '
      + out.root + '/' + r.id + '.tar.gz';
    box.appendChild(card);
  }
}

async function grantAgents(kind, count) {
  try {
    await api('/api/subagents/grant', {
      method: 'POST',
      body: JSON.stringify({ kind: kind, count: Number(count) }),
    });
  } catch (e) { /* the reload below shows the real state either way */ }
  loadAgents();
}

// ---- Settings ----

const SETTING_FIELDS = [
  ['chat_provider', 'Provider', 'select'],
  ['chat_base', 'Base URL (OpenAI-compatible)', 'text'],
  ['chat_model', 'Model', 'text'],
  ['chat_key', 'Chat API key (stored in assistant.db)', 'password'],
  ['chat_key_env', '…or the name of an env var holding it', 'text'],
  ['claude_binary', 'Claude Code binary', 'text'],
  ['claude_model', 'Claude Code model (blank = its default)', 'text'],
  ['claude_timeout', 'Claude Code timeout (seconds)', 'number'],
  ['embed_base', 'Embeddings base URL', 'text'],
  ['embed_model', 'Embeddings model', 'text'],
  ['embed_key', 'Embeddings API key (stored in assistant.db)', 'password'],
  ['embed_key_env', '…or the name of an env var holding it', 'text'],
];

// A stored key is never sent back to the page, so its input is always drawn
// blank — which makes "blank" mean "unchanged", not "delete". Clearing is a
// separate, deliberate act. `config.CLEAR_SECRET`.
const CLEAR_SECRET = '__clear__';
const KEY_VALUE_FIELDS = { chat_key: 'chat_key_env', embed_key: 'embed_key_env' };

async function applyPreset(role, preset) {
  if (!preset) return;
  await api('/api/settings/preset', {
    method: 'POST', body: JSON.stringify({ role: role, preset: preset }),
  });
  loadSettings();
}

function presetRow(label, role, presets, current) {
  const field = el('<label class="field"><span></span></label>');
  field.querySelector('span').textContent = label;
  const select = el('<select></select>');
  select.appendChild(el('<option value="">— pick a known endpoint —</option>'));
  for (const [id, preset] of Object.entries(presets)) {
    const opt = el('<option></option>');
    opt.value = id;
    opt.textContent = preset.label;
    if (preset.model && preset.model === current) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => applyPreset(role, select.value));
  field.appendChild(select);
  const note = el('<em class="muted"></em>');
  const chosen = presets[select.value];
  note.textContent = chosen && chosen.note ? chosen.note : '';
  field.appendChild(note);
  return field;
}

async function rebuildEmbeddings() {
  const status = document.getElementById('settings-status');
  status.textContent = 're-embedding stranded memories…';
  try {
    const out = await api('/api/settings/rebuild-embeddings',
                          { method: 'POST' });
    // Name the two tables separately. "rebuilt 40 memories" while every
    // summary window stayed stranded is the reading to make impossible.
    const what = out.memories_rebuilt + ' memories and ' +
                 out.summaries_rebuilt + ' summary windows';
    status.textContent = out.ok
      ? 'rebuilt ' + what + ' onto ' + out.model +
        (out.remaining ? '; ' + out.remaining + ' still to go — run again'
                       : '; the whole bank is now comparable')
      : 'stopped after ' + what + ': ' + out.error;
  } catch (e) {
    status.textContent = 'failed: ' + e.message;
  }
}

// `prefetched` lets a save redraw from ITS OWN response rather than a second
// GET. The save is the only call that can report a REFUSED field, and that
// warning exists nowhere else — re-fetching threw away the one message the
// user needed and redrew the page looking untouched.
async function loadSettings(prefetched) {
  let out = prefetched;
  if (!out || typeof out !== 'object' || !out.config) {
    try {
      out = await api('/api/settings');
    } catch (e) { return; }
  }
  const box = document.getElementById('settings-form');
  box.innerHTML = '';
  box.appendChild(presetRow('Chat endpoint preset', 'chat',
                            out.chat_presets, out.config.chat_model));
  box.appendChild(presetRow('Embeddings preset (its own provider — the ' +
                            'Claude Code CLI has no embeddings endpoint)',
                            'embed', out.embed_presets,
                            out.config.embed_model));
  for (const [key, label, type] of SETTING_FIELDS) {
    const field = el('<label class="field"><span></span></label>');
    field.querySelector('span').textContent = label;
    let input;
    if (type === 'select') {
      input = el('<select></select>');
      for (const p of out.providers) {
        const opt = el('<option></option>');
        opt.value = p;
        opt.textContent = p === 'claude-code'
          ? 'Claude Code CLI' : 'OpenAI-compatible HTTP';
        input.appendChild(opt);
      }
    } else {
      input = el('<input>');
      input.type = type;
    }
    input.value = out.config[key];
    input.dataset.key = key;
    field.appendChild(input);

    // A stored-key field. The server redacts it, so this is always blank —
    // say what blank MEANS, or a stored key reads as a lost one.
    const pairedName = KEY_VALUE_FIELDS[key];
    if (pairedName) {
      const secret = out.secrets[pairedName] || {};
      const stored = secret.source === 'stored';
      input.placeholder = stored
        ? '•••••••• stored — type to replace, or leave blank to keep'
        : 'paste the key here and press Save';
      const mark = el('<em class="muted"></em>');
      mark.textContent = stored
        ? '● stored in assistant.db and in use'
        : (secret.source === 'environment'
           ? '● using the environment variable below'
           : '○ no key — requests will 401');
      field.appendChild(mark);
      if (stored) {
        // A credential you cannot delete through the interface that stored it
        // is a worse position than never having stored it.
        const forget = el('<button type="button">Forget stored key</button>');
        forget.addEventListener('click', async function () {
          const patch = {};
          patch[key] = CLEAR_SECRET;
          await api('/api/settings',
                    { method: 'PUT', body: JSON.stringify({ settings: patch }) });
          loadSettings();
        });
        field.appendChild(forget);
      }
    }

    // A variable-NAME field. It is the alternative to storing the key, not a
    // second place to paste one.
    const secret = out.secrets[key];
    if (secret) {
      const mark = el('<em class="muted"></em>');
      mark.textContent = secret.source === 'environment'
        ? '● set in this process'
        : (secret.source === 'stored'
           ? '○ not needed — a stored key is in use'
           : '○ not found in this process’s environment');
      field.appendChild(mark);
    }
    box.appendChild(field);
  }
  const status = document.getElementById('settings-status');
  status.innerHTML = '';
  for (const w of out.warnings || []) {
    const line = el('<div class="warn"></div>');
    line.textContent = w;
    status.appendChild(line);
  }
}

async function saveSettings() {
  const values = {};
  document.querySelectorAll('#settings-form [data-key]')
    .forEach(i => { values[i.dataset.key] = i.value; });
  const status = document.getElementById('settings-status');
  status.textContent = 'saving…';
  let out;
  try {
    out = await api('/api/settings', {
      method: 'PUT', body: JSON.stringify({ settings: values }),
    });
  } catch (e) {
    status.textContent = 'failed: ' + e.message;
    return;
  }
  // Redraw from the save's OWN response. A refused field reverts visibly and
  // its warning is rendered, instead of the page reloading unchanged and
  // reading as "I pressed save and nothing happened".
  await loadSettings(out);
}

async function testSettings() {
  const status = document.getElementById('settings-status');
  status.textContent = 'calling the provider…';
  try {
    const out = await api('/api/settings/test', { method: 'POST' });
    status.textContent = out.ok
      ? 'OK (' + out.provider + '): ' + out.reply
      : 'failed: ' + out.error;
  } catch (e) {
    status.textContent = 'failed: ' + e.message;
  }
}

// ---- Persona ----

const PERSONA_FIELDS = ['drive', 'identity', 'expertise', 'working_style',
                        'standing_commitments', 'preferences'];

async function loadPersona() {
  let out;
  try {
    out = await api('/api/persona');
  } catch (e) { return; }
  const box = document.getElementById('persona-form');
  box.innerHTML = '';
  for (const key of PERSONA_FIELDS) {
    const field = el('<label class="field"><span></span>' +
      '<textarea rows="3"></textarea></label>');
    field.querySelector('span').textContent = key.replace(/_/g, ' ');
    const area = field.querySelector('textarea');
    const value = out.persona[key];
    // Lists are edited one item per line — the shape the author thinks in.
    area.value = Array.isArray(value) ? value.join('\n') : (value || '');
    area.dataset.key = key;
    area.dataset.list = Array.isArray(value) ? '1' : '';
    box.appendChild(field);
  }
  for (const w of out.warnings || []) {
    const line = el('<div class="warn"></div>');
    line.textContent = w;
    box.appendChild(line);
  }
}

async function savePersona() {
  const sheet = {};
  document.querySelectorAll('#persona-form [data-key]').forEach(a => {
    sheet[a.dataset.key] = a.dataset.list
      ? a.value.split('\n').map(s => s.trim()).filter(Boolean)
      : a.value;
  });
  await api('/api/persona', {
    method: 'PUT', body: JSON.stringify({ persona: sheet }),
  });
  loadPersona();
}

document.addEventListener('DOMContentLoaded', function () {
  initFiles();
  document.getElementById('settings-save')
    .addEventListener('click', saveSettings);
  document.getElementById('settings-test')
    .addEventListener('click', testSettings);
  document.getElementById('settings-rebuild')
    .addEventListener('click', rebuildEmbeddings);
  document.getElementById('persona-save')
    .addEventListener('click', savePersona);
  document.getElementById('agent-revoke').addEventListener('click',
    async function () {
      await api('/api/subagents/revoke', { method: 'POST' });
      loadAgents();
    });
  document.querySelectorAll('nav button').forEach(btn =>
    btn.addEventListener('click', function () {
      if (btn.dataset.panel === 'memory') loadRetired();
      if (btn.dataset.panel === 'files') loadFiles();
      if (btn.dataset.panel === 'agents') loadAgents();
      if (btn.dataset.panel === 'settings') { loadSettings(); loadPersona(); }
    }));
  // The badge is the whole point of the request channel: an assistant that
  // asked for permission and got no answer is stuck, and it has no way to
  // tell you except through this.
  loadAgents();
  setInterval(loadAgents, 15000);
});
