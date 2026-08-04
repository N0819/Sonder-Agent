// chat.js — the conversation panel. Owns the log, the composer, and the
// in-flight state; app.js owns panel switching and the read-only panels.
'use strict';

let sessionId = null;

// The Files panel uploads into a SESSION workspace, so it needs the id the
// chat loop is holding. Exposed through a getter rather than by making the
// variable global: one owner, one writer.
function currentSessionId() { return sessionId; }
let sending = false;

// The small subset of Markdown a reply actually uses, rendered.
//
// ESCAPE FIRST, THEN MARK UP, and never the other way round. This is model
// output: it is untrusted by the same rule that governs everything else here
// — model output is judged, never trusted — and it routinely quotes the
// user's own uploaded files, so a source file containing a <script> tag would
// otherwise execute in the page that displays it. Escaping first means the
// only tags that can reach the DOM are the four this function writes.
//
// Inline only. Blocks (lists, headings, tables) are deliberately absent: the
// bubble is `white-space: pre-wrap`, so their plain-text form already reads
// correctly, and half-implemented block rendering looks worse than none.
function renderInlineMarkdown(text) {
  const escaped = String(text === null || text === undefined ? '' : text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return escaped
    // Code first: whatever is inside a span of backticks is literal, so it
    // must be claimed before the emphasis rules can chew on its asterisks.
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^\n]+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^\n]+?)__/g, '<strong>$1</strong>')
    // Single-mark emphasis runs AFTER the double-mark rules, or `**x**` would
    // be read as an empty italic wrapping `x`.
    //
    // The content may not START or END with whitespace, which is what stops
    // `2 * 3 * 4` from rendering as `2 <em> 3 </em> 4`. An assistant that
    // writes about code emits bare asterisks constantly — multiplication,
    // globs, footnote marks — so the arithmetic case is the common one, not
    // the corner one.
    .replace(/(^|[^*\w])\*(\S|\S[^*\n]*?\S)\*(?![*\w])/g, '$1<em>$2</em>')
    .replace(/(^|[^_\w])_(\S|\S[^_\n]*?\S)_(?![_\w])/g, '$1<em>$2</em>');
}

function addMsg(cls, text, warnings) {
  const log = document.getElementById('log');
  const node = el('<div class="msg ' + cls + '"></div>');
  const body = el('<span class="body"></span>');
  body.innerHTML = renderInlineMarkdown(text);
  node.appendChild(body);
  if (warnings && warnings.length) {
    // Engine rule: a warning is the system WORKING (a dropped ungrounded
    // citation means nothing crossed). Show them small, never as errors.
    const w = el('<div class="warnings"></div>');
    w.textContent = '⚠ ' + warnings.join(' · ');
    node.appendChild(w);
  }
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
  return node;
}

// Offer a retry on the message that failed, carrying the text with it.
//
// A turn whose respond stage died still COMMITS: memory records the exchange
// and the reply says so. Retyping a long message to get past a transient
// provider failure is the wrong tax, and re-reading it off the screen is
// worse when the failure was caused by its size in the first place.
//
// `respond_ok` comes from the server. Deciding here by searching warning text
// for "failed" would stop working the day a warning is reworded, and it would
// stop working silently.
function addRetry(node, text) {
  const bar = el('<div class="retry-bar"></div>');
  const btn = el('<button type="button">Retry this turn</button>');
  btn.addEventListener('click', function () {
    btn.disabled = true;
    bar.remove();
    sendMessage(text, true);
  });
  bar.appendChild(btn);
  const note = el('<em class="muted"></em>');
  // Say the cost. The retry is a NEW turn, so the exchange is recorded twice
  // — visible in the memory panel, and better stated than discovered there.
  note.textContent = 'Sends the same message again as a new turn.';
  bar.appendChild(note);
  node.appendChild(bar);
}

// One line of the reasoning trail. Rendered from the stage events the server
// emits, which is why the wording lives here and the FACTS live server-side —
// the panel must never be able to imply a step happened that did not.
function reasoningLine(ev) {
  if (ev.stage === 'recall') {
    const bits = [ev.returned + ' memories recalled'];
    if (ev.pondered) bits.push(ev.pondered + ' pondered');
    if (ev.ponder_query) bits.push('pondering: "' + ev.ponder_query + '"');
    let text = bits.join(' · ');
    if (ev.gists && ev.gists.length) {
      text += '\n' + ev.gists.map(g => '   · ' + g).join('\n');
    }
    return text;
  }
  if (ev.stage === 'respond') {
    let text = 'respond: ' + ev.state;
    if (ev.ponder) text += '\n   wants to ponder: "' + ev.ponder + '"';
    if (ev.asks_research) text += '\n   asked to research';
    return text;
  }
  if (ev.stage === 'research') {
    if (ev.state === 'opened') return 'research opened: ' + ev.question;
    const d = ev.detail || {};
    const what = d.query || d.url || d.answer || d.statement || '';
    return 'research round (' + ev.rounds_left + ' left) → ' + ev.action
      + (what ? '\n   ' + what : '');
  }
  if (ev.stage === 'commit') return 'committing — past the point of no halt';
  if (ev.stage === 'halted') return 'halted; nothing was committed';
  if (ev.stage === 'failed') return 'failed: ' + ev.error;
  if (ev.stage === 'start' || ev.stage === 'done') return null;
  return ev.stage;
}

function makeReasoningPanel() {
  const wrap = el('<div class="reasoning"></div>');
  const toggle = el('<button type="button" class="reasoning-toggle">'
                    + 'Show reasoning ▸</button>');
  const body = el('<div class="reasoning-body"></div>');
  body.style.display = 'none';
  toggle.addEventListener('click', function () {
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    toggle.textContent = open ? 'Show reasoning ▸' : 'Hide reasoning ▾';
  });
  wrap.appendChild(toggle);
  wrap.appendChild(body);
  return { wrap: wrap, body: body, toggle: toggle };
}

async function sendMessage(text, isRetry) {
  if (sending || !text.trim()) return;
  sending = true;
  const sendBtn = document.getElementById('send');
  const haltBtn = document.getElementById('halt');
  sendBtn.disabled = true;
  addMsg('user', isRetry ? text + '  (retry)' : text);

  const pending = addMsg('meta', 'thinking…');
  const panel = makeReasoningPanel();
  pending.appendChild(panel.wrap);
  let runId = null;
  let steps = 0;

  const finish = () => {
    sending = false;
    sendBtn.disabled = false;
    haltBtn.style.display = 'none';
    haltBtn.disabled = false;
    haltBtn.textContent = 'Halt';
  };

  try {
    const started = await api('/api/chat/start', {
      method: 'POST',
      body: JSON.stringify({ text: text, session_id: sessionId }),
    });
    runId = started.turn_run_id;
    haltBtn.style.display = '';
    haltBtn.onclick = async function () {
      haltBtn.disabled = true;
      haltBtn.textContent = 'halting…';
      try {
        const out = await api('/api/chat/' + runId, { method: 'DELETE' });
        // Say which of the three things happened. "too_late" is a real
        // outcome, not a failure, and hiding it would make the button look
        // broken on exactly the turns where it behaved correctly.
        if (out.outcome === 'too_late') {
          haltBtn.textContent = 'too late — committing';
        }
      } catch (e) { haltBtn.textContent = 'Halt'; haltBtn.disabled = false; }
    };

    await new Promise(function (resolve) {
      const src = new EventSource('/api/chat/' + runId + '/events');
      src.onmessage = function (msg) {
        const ev = JSON.parse(msg.data);
        if (ev.stage === 'end') {
          src.close();
          pending.remove();
          const out = ev.result || {};
          if (ev.status === 'halted') {
            const node = addMsg('meta', 'Halted. Nothing was committed — the '
                                + 'turn was stopped before its write.');
            if (steps) node.appendChild(panel.wrap);
            addRetry(node, text);
          } else if (ev.status === 'failed') {
            const node = addMsg('meta', 'error: ' + ev.error);
            if (steps) node.appendChild(panel.wrap);
            addRetry(node, text);
          } else {
            sessionId = out.session_id || sessionId;
            const node = addMsg('assistant', out.reply || '(no reply)',
                                out.warnings);
            if (steps) node.appendChild(panel.wrap);
            if (out.respond_ok === false) addRetry(node, text);
          }
          resolve();
          return;
        }
        const line = reasoningLine(ev);
        if (line === null) return;
        steps += 1;
        const row = el('<div class="reasoning-step"></div>');
        row.textContent = '[' + ev.t.toFixed(1) + 's] ' + line;
        panel.body.appendChild(row);
        // The live view is the point: show the newest step in the collapsed
        // state too, so a running turn is legible without opening anything.
        panel.toggle.textContent =
          (panel.body.style.display === 'none' ? 'Show reasoning ▸ '
                                               : 'Hide reasoning ▾ ')
          + '(' + steps + ' steps)';
      };
      src.onerror = function () {
        src.close();
        pending.remove();
        addRetry(addMsg('meta', 'lost the connection to this turn'), text);
        resolve();
      };
    });
  } catch (err) {
    pending.remove();
    // A transport failure never reached the pipeline, so nothing was
    // committed and a retry is unambiguously the right offer.
    addRetry(addMsg('meta', 'error: ' + err.message), text);
  }
  finish();
}

function initChat() {
  const form = document.getElementById('composer');
  const input = document.getElementById('input');
  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    const text = input.value;
    input.value = '';
    sendMessage(text);
  });
  input.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      form.requestSubmit();
    }
  });
}
