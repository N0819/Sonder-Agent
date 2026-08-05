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

function addMsg(cls, text, warnings, askedText) {
  const log = document.getElementById('log');
  const node = el('<div class="msg ' + cls + '"></div>');
  const body = el('<span class="body"></span>');
  body.innerHTML = renderInlineMarkdown(text);
  node.appendChild(body);
  // A CORNER AFFORDANCE, NOT A FAILURE FIX. Retry used to appear only when a
  // turn had failed, which made it a repair tool and made asking again after
  // an ANSWER — the ordinary reason anyone re-asks — a matter of retyping the
  // message. It belongs on any message that came from a question, whether or
  // not that question went well.
  if (askedText) addRetry(node, askedText);
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
  const btn = el('<button type="button" class="retry-corner" '
                 + 'title="Ask this again as a new turn">⟳</button>');
  btn.addEventListener('click', function () {
    if (sending) return;
    btn.disabled = true;
    sendMessage(text, true);
  });
  node.appendChild(btn);
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
  if (ev.stage === 'subagent') {
    // A subagent used to be a hole in this panel: the turn went quiet for up
    // to DEEP_TIMEOUT with no way to tell work from a hang.
    let text = (ev.kind || 'subagent') + ' — ' + ev.state;
    if (ev.task) text += '\n   task: ' + ev.task;
    if (ev.detail) text += '\n   ' + ev.detail;
    if (ev.questions_left !== undefined) {
      text += '\n   (' + ev.questions_left + ' questions left)';
    }
    if (ev.rounds_left !== undefined) {
      text += '\n   (' + ev.rounds_left + ' rounds left)';
    }
    if (ev.state === 'reported') {
      const bits = [];
      if (ev.claims !== undefined) bits.push(ev.claims + ' claims');
      if (ev.evidence !== undefined) bits.push(ev.evidence + ' evidence');
      if (bits.length) text += '\n   ' + bits.join(', ');
      if (ev.summary) text += '\n   ' + ev.summary;
    }
    return text;
  }
  if (ev.stage === 'stream') return null;   // rendered live, not as a step
  if (ev.stage === 'edit') {
    return 'edited ' + ev.path + (ev.created ? ' (new file)' : '')
      + '\n   re-chunked into ' + ev.rechunked + ' pieces';
  }
  if (ev.stage === 'commit') return 'committing — past the point of no halt';
  if (ev.stage === 'halted') return 'halted; nothing was committed';
  if (ev.stage === 'failed') return 'failed: ' + ev.error;
  if (ev.stage === 'start' || ev.stage === 'done') return null;
  return ev.stage;
}

function reasoningStep(ev, line) {
  // EVERY STAGE CARRIES MORE THAN ITS HEADLINE, and dumping all of it made
  // the panel a wall nobody read — the gists behind a recall, the query
  // behind a search, the task behind a subagent. First line is the summary,
  // the rest opens on click, so scanning and reading are different acts.
  const parts = line.split('\n');
  const summary = parts[0];
  const detail = parts.slice(1).join('\n');
  const row = el('<div class="reasoning-step"></div>');
  const head = el('<div class="step-head"></div>');
  head.textContent = '[' + ev.t.toFixed(1) + 's] ' + summary;
  row.appendChild(head);
  if (!detail) return row;
  head.classList.add('expandable');
  head.textContent += '  ▸';
  const body = el('<div class="step-detail"></div>');
  body.textContent = detail;
  body.style.display = 'none';
  head.addEventListener('click', function () {
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    head.textContent = '[' + ev.t.toFixed(1) + 's] ' + summary
      + (open ? '  ▸' : '  ▾');
  });
  row.appendChild(body);
  return row;
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
  document.getElementById('send').disabled = true;
  addMsg('user', isRetry ? text + '  (retry)' : text);
  let runId = null;
  try {
    const started = await api('/api/chat/start', {
      method: 'POST',
      body: JSON.stringify({ text: text, session_id: sessionId }),
    });
    runId = started.turn_run_id;
    // Remembered BEFORE the stream opens. A page that closes between the
    // POST and the first event has still started a turn, and the id is the
    // only way back to it.
    rememberRun(runId, text);
  } catch (err) {
    releaseComposer();
    addMsg('meta', 'error: ' + err.message, null, text);
    return;
  }
  await watchExisting(runId, text);
}

function releaseComposer() {
  sending = false;
  document.getElementById('send').disabled = false;
  const haltBtn = document.getElementById('halt');
  haltBtn.style.display = 'none';
  haltBtn.disabled = false;
  haltBtn.textContent = 'Halt';
}

// Watch a run that already exists. Shared by the first send and by a resume,
// deliberately: a rejoined turn that rendered through a second code path
// would drift from the live one, and the drift would only ever show up in the
// case nobody tests.
async function watchExisting(runId, text) {
  sending = true;
  document.getElementById('send').disabled = true;
  const haltBtn = document.getElementById('halt');

  const pending = addMsg('meta', 'thinking\u2026');
  const dropped = el('<div class="dropped"></div>');
  dropped.style.display = 'none';
  pending.appendChild(dropped);
  const live = el('<div class="live"></div>');
  live.style.display = 'none';
  live.think = el('<div class="live-stream thinking"></div>');
  live.answer = el('<div class="live-stream"></div>');
  live.think.style.display = 'none';
  live.answer.style.display = 'none';
  live.appendChild(live.think);
  live.appendChild(live.answer);
  pending.appendChild(live);
  const panel = makeReasoningPanel();
  pending.appendChild(panel.wrap);
  let steps = 0;

  haltBtn.style.display = '';
  haltBtn.onclick = async function () {
    haltBtn.disabled = true;
    haltBtn.textContent = 'halting\u2026';
    try {
      const out = await api('/api/chat/' + runId, { method: 'DELETE' });
      // Say which of the three things happened. "too_late" is a real
      // outcome, not a failure, and hiding it would make the button look
      // broken on exactly the turns where it behaved correctly.
      if (out.outcome === 'too_late') {
        haltBtn.textContent = 'too late \u2014 committing';
      }
    } catch (e) { haltBtn.textContent = 'Halt'; haltBtn.disabled = false; }
  };

  await new Promise(function (resolve) {
    const src = new EventSource('/api/chat/' + runId + '/events');
    let ended = false;
    // CLEARED ON RECONNECT, NOT ON THE NEXT MESSAGE. The banner was only
    // taken down by an arriving event, and the gap between events is minutes
    // during a deep subagent — so a stream that had already recovered went on
    // saying "connection lost" for as long as the subagent took. Observed
    // live: the browser held an ESTABLISHED connection the whole time the
    // page claimed to be reconnecting. A status line that lies about the
    // thing it exists to report is worse than no status line.
    src.onopen = function () { dropped.style.display = 'none'; };
    src.onmessage = function (msg) {
      const ev = JSON.parse(msg.data);
      if (ev.stage === 'end') {
        ended = true;
        src.close();
        forgetRun();
        pending.remove();
        const out = ev.result || {};
        if (ev.status === 'halted') {
          const node = addMsg('meta', 'Halted. Nothing was committed \u2014 '
                              + 'the turn was stopped before its write.',
                              null, text);
          if (steps) node.appendChild(panel.wrap);
        } else if (ev.status === 'failed') {
          const node = addMsg('meta', 'error: ' + ev.error, null, text);
          if (steps) node.appendChild(panel.wrap);
        } else {
          sessionId = out.session_id || sessionId;
          const node = addMsg('assistant', out.reply || '(no reply)',
                              out.warnings, text);
          if (steps) node.appendChild(panel.wrap);
        }
        resolve();
        return;
      }
      // Anything arriving means the pipe is healthy again.
      dropped.style.display = 'none';
      // The model's own words, as they are written. Before streaming, a turn
      // was a spinner for as long as the answer took and there was no way to
      // tell a slow answer from a dead one \u2014 the same ambiguity the idle
      // timeout fixes on the server, seen from the outside.
      if (ev.stage === 'stream') {
        // Reasoning is shown SEPARATELY from the answer. It is what the model
        // was working through, not what it decided to say, and the two in one
        // pane read as a single confident statement.
        const target = ev.kind === 'thinking' ? live.think : live.answer;
        if (ev.truncated) { target.classList.add('capped'); return; }
        target.textContent += ev.delta || '';
        target.style.display = 'block';
        live.style.display = 'block';
        target.scrollTop = target.scrollHeight;
        return;
      }
      const line = reasoningLine(ev);
      if (line === null) return;
      steps += 1;
      panel.body.appendChild(reasoningStep(ev, line));
      // The live view is the point: show the newest step in the collapsed
      // state too, so a running turn is legible without opening anything.
      panel.toggle.textContent =
        (panel.body.style.display === 'none' ? 'Show reasoning \u25b8 '
                                             : 'Hide reasoning \u25be ')
        + '(' + steps + ' steps)';
    };
    // HOLD, DO NOT GIVE UP. This used to close the stream and offer a retry
    // \u2014 which threw away a turn that was still running on the server and,
    // if the user took the offer, ran the whole thing a second time. The
    // connection dropping says nothing about the turn.
    //
    // EventSource reconnects by itself and sends `Last-Event-ID`, so the
    // resumed stream picks up after the last step already drawn. The only
    // decision here is whether a reconnect is even possible: CONNECTING means
    // the browser is already retrying, CLOSED means the server refused (a 404
    // for a run the registry has dropped) and waiting will not help.
    src.onerror = function () {
      if (ended) return;
      if (src.readyState === EventSource.CONNECTING) {
        dropped.textContent = 'connection lost \u2014 reconnecting\u2026 '
          + '(the turn is still running)';
        dropped.style.display = 'block';
        return;
      }
      src.close();
      forgetRun();
      pending.remove();
      addMsg('meta', 'lost this turn: the server no longer has it. Anything '
             + 'it committed is in your memory panel.', null, text);
      resolve();
    };
  });
  releaseComposer();
}

// The run id outlives the page.
//
// A reload mid-turn used to lose the turn: the worker kept running, wrote its
// memories and finished, and the user was looking at a page that had no idea
// any of it had happened. Remembering the id is what makes the answer
// collectable afterwards — `sessionStorage`, not `localStorage`, because a
// run does not survive a server restart and a stale id in a new browser
// session is a promise this cannot keep.
const RUN_KEY = 'sonder.activeRun';

function rememberRun(runId, text) {
  try {
    sessionStorage.setItem(RUN_KEY, JSON.stringify({ id: runId, text: text }));
  } catch (e) { /* private mode: resume is a nicety, not a requirement */ }
}

function forgetRun() {
  try { sessionStorage.removeItem(RUN_KEY); } catch (e) { /* as above */ }
}

async function resumeRun() {
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem(RUN_KEY) || 'null'); }
  catch (e) { saved = null; }
  if (!saved || !saved.id) return;
  let status = null;
  try {
    status = await api('/api/chat/' + saved.id);
  } catch (e) {
    forgetRun();          // the server has dropped it; nothing to rejoin
    return;
  }
  addMsg('user', saved.text);
  if (status.status === 'running') {
    addMsg('meta', 'rejoining a turn that was still running when the page '
                   + 'reloaded…');
    await watchExisting(saved.id, saved.text);
    return;
  }
  // It finished while the page was away. The result is still in the registry,
  // so the answer is collectable rather than merely lost politely.
  forgetRun();
  const out = status.result || {};
  if (status.status === 'done') {
    sessionId = out.session_id || sessionId;
    addMsg('assistant', out.reply || '(no reply)', out.warnings, saved.text);
  } else {
    addMsg('meta', 'that turn ended as "' + status.status + '"'
           + (status.error ? ': ' + status.error : ''), null, saved.text);
  }
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
  // Last, and not awaited: a turn left running by a reload is rejoined in the
  // background so the page is usable while it finishes.
  resumeRun();
}
