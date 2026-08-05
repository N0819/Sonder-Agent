// history.js — prior sessions, their turns, and the trail each one left.
//
// EVERY RELOAD STARTED A NEW SESSION AND ABANDONED THE OLD ONE. `sessionId`
// begins null in the page and is only ever set from the reply of a turn this
// page ran, so closing the tab ended the conversation as far as the browser
// was concerned. The turns were all on disk — `/api/sessions` and
// `/api/sessions/{id}/turns` have existed since the routes were written — and
// nothing in the interface ever called either one.
//
// That is a bad fit for a chat client and a disqualifying one for a research
// tool. The durable artefact of this project is not the reply; it is the
// TRACE: which memories surfaced, what was retrieved, which hypotheses moved,
// what the subagents reported, what the payload cost. All of it is written to
// `turns.trace` on every commit, and until now the only way to read a trace
// was to open the database by hand.
//
// So this panel shows what happened, and it also lets a session be PICKED UP
// again — the transcript back in the log, `sessionId` adopted, the next
// message continuing the same thread rather than founding a new one.
'use strict';

// The trace keys worth leading with, in the order a reader wants them. The
// rest still render — this only decides what is read first, and an unknown
// key must never be dropped silently, since a new stage would then be
// invisible in exactly the panel built to see stages.
const TRACE_ORDER = ['warnings', 'payload_cost', 'retrieval_health',
                     'subagents', 'experiments', 'deliberation',
                     'closed_threads', 'minted'];

function historyWhen(seconds) {
  if (!seconds) return '';
  return new Date(seconds * 1000).toLocaleString();
}

async function loadHistory() {
  const box = document.getElementById('sessions');
  const pane = document.getElementById('session-turns');
  pane.textContent = '';
  box.textContent = 'loading…';
  let rows;
  try {
    rows = await api('/api/sessions');
  } catch (err) {
    box.textContent = 'could not load sessions: ' + err.message;
    return;
  }
  box.textContent = '';
  if (!rows.length) {
    box.textContent = 'no sessions yet';
    return;
  }
  rows.forEach(function (s) {
    const btn = el('<button type="button" class="session-row"></button>');
    // Nothing has ever written `title`, so it is the fallback and the opening
    // line is the label. A session with no committed turns still gets a row —
    // it is the trace of a reload that started a thread and abandoned it, and
    // that is worth being able to see rather than worth hiding.
    const label = s.title || s.opened_with || '(nothing was said)';
    btn.textContent = '#' + s.id + '  ' + label;
    const sub = el('<span class="session-sub"></span>');
    sub.textContent = s.turns + (s.turns === 1 ? ' turn' : ' turns')
      + (s.created ? ' · ' + historyWhen(s.created) : '');
    btn.appendChild(sub);
    btn.addEventListener('click', function () {
      document.querySelectorAll('.session-row')
        .forEach(function (n) { n.classList.remove('active'); });
      btn.classList.add('active');
      openSession(s.id);
    });
    box.appendChild(btn);
  });
}

async function openSession(sid) {
  const pane = document.getElementById('session-turns');
  pane.textContent = 'loading…';
  let turns;
  try {
    turns = await api('/api/sessions/' + sid + '/turns');
  } catch (err) {
    pane.textContent = 'could not load turns: ' + err.message;
    return;
  }
  pane.textContent = '';
  const bar = el('<div class="session-bar"></div>');
  const resume = el('<button type="button" class="resume">'
                    + 'Continue this session</button>');
  // The whole point of being able to READ an old session is being able to go
  // on with it. Adopting the id is what makes the next turn land in the same
  // thread — recall, beliefs and the episode chain all key off it.
  resume.addEventListener('click', function () { adoptSession(sid, turns); });
  bar.appendChild(resume);
  const count = el('<span class="session-sub"></span>');
  count.textContent = turns.length + (turns.length === 1 ? ' turn' : ' turns');
  bar.appendChild(count);
  pane.appendChild(bar);
  if (!turns.length) {
    pane.appendChild(el('<div class="session-sub">this session has no '
                        + 'committed turns</div>'));
    return;
  }
  turns.forEach(function (t) { pane.appendChild(turnCard(t)); });
}

function turnCard(t) {
  const card = el('<div class="turn-card"></div>');
  const head = el('<div class="turn-head"></div>');
  head.textContent = 'turn ' + t.turn_idx
    + (t.created ? ' · ' + historyWhen(t.created) : '');
  card.appendChild(head);
  if (t.user_text) {
    const asked = el('<div class="turn-said user"></div>');
    asked.textContent = t.user_text;
    card.appendChild(asked);
  }
  const said = el('<div class="turn-said"></div>');
  said.innerHTML = renderInlineMarkdown(t.reply_text || '(no reply)');
  card.appendChild(said);
  const trace = t.trace || {};
  const warnings = trace.warnings || [];
  if (warnings.length) {
    const w = el('<div class="warnings"></div>');
    w.textContent = '⚠ ' + warnings.join(' · ');
    card.appendChild(w);
  }
  // The trace, as READABLE TEXT rather than as JSON. Same rule the live panel
  // follows: a blob of braces is not a record anyone consults twice.
  const keys = TRACE_ORDER.filter(function (k) { return k in trace; })
    .concat(Object.keys(trace).filter(function (k) {
      return TRACE_ORDER.indexOf(k) === -1;
    }));
  keys.forEach(function (k) {
    if (k === 'warnings') return;             // already shown, in full
    card.appendChild(traceSection(k, trace[k]));
  });
  if (!keys.length) {
    card.appendChild(el('<div class="session-sub">no trace recorded for '
                        + 'this turn</div>'));
  }
  return card;
}

function traceSection(key, value) {
  const row = el('<div class="trace-section"></div>');
  const head = el('<div class="step-head expandable"></div>');
  const body = el('<div class="step-detail"></div>');
  // `readable` returns '' for an empty list or object, and an expandable that
  // opens onto nothing reads as a rendering bug. Say EMPTY instead: "this
  // turn ran no experiments" is a finding about the turn, and this whole
  // project's first rule about measurement is that zero is a result.
  body.textContent = readable(value, 0);
  if (!body.textContent.trim()) {
    const flat = el('<div class="step-head"></div>');
    flat.textContent = key + ': (empty)';
    row.appendChild(flat);
    return row;
  }
  body.style.display = 'none';
  function label(open) {
    head.textContent = key + (open ? '  ▾' : '  ▸');
  }
  label(false);
  head.addEventListener('click', function () {
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    label(!open);
  });
  row.appendChild(head);
  row.appendChild(body);
  return row;
}
