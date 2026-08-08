"""Read-only query access to databases too large to move.

WHY THIS EXISTS. The engine's `engine.db` is 1,118,785,536 bytes — 2.1x the
whole workspace ceiling and 550x the per-file snapshot limit. Every existing
lane into data assumes the data can be COPIED: `snapshot_for_sandbox` builds a
{path: text} payload, `read_file` reads a workspace file, the chunk index
stores bodies. None of them can carry a gigabyte, and raising the limits would
mean writing that gigabyte into a fresh temp directory on every single run.

The size ceilings bind on what a SANDBOX must see, because a run gets a copy.
A fetch verb runs in the assistant's own process and never copies anything, so
it is not bound at all. Same shape as the `read_file` repair: the fix for data
that will not fit through a door is another door, not a wider one.

WHAT IT DELIBERATELY IS NOT. Not a write path — the connection is opened
read-only at the URI level, so a write is refused by SQLite itself rather than
by a string that has to be maintained. Not a workspace: nothing here is
editable, chunked, or delivered to a sandbox. A reference database is
evidence, and evidence is read.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time

MAX_ROWS = 200
MAX_CHARS = 20000
MAX_CELL_CHARS = 2000
TIME_LIMIT_S = 5.0

# A REFERENCE DATABASE IS SOMEBODY ELSE'S DATABASE, and it holds their
# credentials. `SELECT name, api_key FROM providers` against the engine
# returned two live keys verbatim — measured, not supposed — and `SELECT * FROM
# settings` returned the host password hash and its salt. Read-only stops a
# write; it does nothing about a read, and a read is the whole problem here.
#
# What made it worth fixing rather than documenting: everything this lane
# returns is written down. It lands in an evidence excerpt, in the turn trace,
# in `assistant.db` — and this repository is public. A key does not have to be
# published to be burned; it only has to be recorded somewhere it was never
# meant to be, by a process nobody remembers reading.
#
# So it is folded on the way OUT, at the one point every cell passes through,
# rather than written as a rule about which tables not to ask for. A rule that
# must be remembered will be forgotten — and here it would have to be
# remembered by a model that has every reason to go looking at settings when it
# is debugging a settings-shaped defect.

# The COLUMN NAME says it is a credential. `_hash`/`_salt` are deliberately
# only matched next to a password or secret: a bare `_hash` column is usually a
# content digest, and digests are load-bearing evidence in this repository.
_SECRET_NAME = re.compile(
    r"(^|_)(api_?key|apikey|secret|password|passwd|token|credential|"
    r"private_key|access_key|refresh_token)(_|$)"
    r"|(^|_)(pw|password|passwd|secret)_(hash|salt)(_|$)", re.I)

# The ROW KEY says it, in a key/value table like `settings`, where the column
# is called `value` and carries no information at all. `_key$` lives here and
# not above on purpose: as a row key it catches `freesound_key`, whereas as a
# column name it would catch the `key` column itself — which holds names.
_SECRET_ROW_KEY = re.compile(
    r"(^|_)(api_?key|apikey|secret|password|passwd|token|credential)(_|$)"
    r"|_key$|_hash$|_salt$", re.I)

# The VALUE announces itself. The backstop for a credential in a column nobody
# thought to name suggestively — a key pasted into a notes field is still a key.
_SECRET_VALUE = re.compile(
    r"^(sk-|sk_live|pk_live|ghp_|gho_|ghu_|ghs_|github_pat_|xox[baprs]-|"
    r"AKIA|ASIA|AIza|glpat-|hf_|-----BEGIN)")

# THE SAME ANNOUNCEMENT, ANYWHERE IN THE CELL. The three checks above all ask
# what a cell IS — its column, its row key, what it starts with. None of them
# sees a credential nested inside a document, and that is the shape the live
# data actually takes: `assistant.db` holds one settings row, keyed
# `providers`, whose value is a JSON blob carrying live provider keys. The
# columns are `key` and `value`, `providers` matches no secret-key pattern,
# and the anchor on `_SECRET_VALUE` never reaches past the opening brace.
# Verified against the real row before this existed: all three returned False
# with a credential-shaped token present. `4144d63` stopped this lane handing
# back the engine's credentials; this is the same defect in a shape that fix
# could not see.
#
# A TOKEN BODY IS REQUIRED, or an unanchored scan eats prose — and a false
# positive here costs real data silently. The near-misses are ordinary
# sentences: "sk- on its own", "AIza as a string", a column named `task_key`.
# Eight token characters after the prefix separates a credential from a word.
#
# The WHOLE cell goes, not the matched span. Excising the key from a document
# means parsing the document, and a parser that fails open on malformed JSON
# leaks exactly the row most worth leaking.
_SECRET_IN_TEXT = re.compile(
    r"(sk-|sk_live|pk_live|ghp_|gho_|ghu_|ghs_|github_pat_|xox[baprs]-|"
    r"AKIA|ASIA|AIza|glpat-|hf_)[A-Za-z0-9_\-]{8,}"
    r"|-----BEGIN[A-Z ]*PRIVATE KEY")

# Which column holds the row key, when the table is key/value shaped.
_KEY_COLUMNS = ("key", "name", "setting", "option")

# One careless scan of a 699 MiB table holds the turn open for minutes, and the
# assistant cannot know a table's size before it asks. The handler fires every
# _PROGRESS_OPS virtual-machine instructions and aborts on wall clock, so a bad
# query costs seconds instead of the turn.
_PROGRESS_OPS = 50000

# A statement gate on top of `mode=ro`, which already makes writes impossible.
# It exists for the ERROR MESSAGE: "read-only database" tells the author that
# SQLite refused something, not which of their statements was the problem.
_ALLOWED = ("select", "with", "explain", "pragma", "values")

_DATABASES: dict[str, str] = {}


def configure(mapping=None):
    """Name the databases that may be read. Tests call this with tmp paths.

    Also reads `ASSISTANT_REFDB` when no mapping is given: a comma-separated
    list of `name=/absolute/path` pairs. NAMES, not paths, are what the
    assistant passes — a path parameter would be a second workspace escape to
    guard, and there is no reason for the caller to choose the file.
    """
    global _DATABASES
    if mapping is None:
        # STORED BEATS EXPORTED, the same precedence the credential fields
        # follow and for the same reason: an export belongs to whoever started
        # the process. This lane was environment-only, and a restart that did
        # not carry `ASSISTANT_REFDB` left the engine silently unreachable —
        # `reference_databases` empty, `query_db` refusing a name that had
        # worked an hour earlier, and nothing saying the lane had been
        # configured and then lost. Imported here rather than at module level:
        # `config` needs a database, and this module is imported by tests that
        # have none.
        raw = os.environ.get("ASSISTANT_REFDB") or ""
        try:
            import config
            raw = str(config.get_config().get("reference_databases") or raw)
        except Exception:  # noqa: BLE001 - no settings row is not an error
            pass
        mapping = {}
        for pair in raw.split(","):
            name, _, path = pair.partition("=")
            if name.strip() and path.strip():
                mapping[name.strip()] = path.strip()
    _DATABASES = {str(k): str(v) for k, v in dict(mapping).items()}
    return dict(_DATABASES)


def databases():
    """What can be queried, with the size and reachability of each.

    AN ABSENT DATABASE MUST LOOK ABSENT. A registry that lists a name whose
    file has been moved reads exactly like one whose file is there, and the
    failure surfaces later as an empty result set that looks like a finding.
    """
    if not _DATABASES:
        configure()
    out = []
    for name, path in sorted(_DATABASES.items()):
        try:
            size = os.path.getsize(path)
            out.append({"name": name, "bytes": size, "ok": True})
        except OSError as exc:
            out.append({"name": name, "ok": False,
                        "error": f"unreadable: {exc.strerror}"})
    return out


def _blank_strings(sql):
    """Replace every quoted literal with an empty one, for analysis only.

    THE GATE WAS READING PUNCTUATION INSIDE DATA. A story named
    "O'Brien; the sequel" composes a perfectly good single SELECT, and the
    checks below saw the semicolon in the literal and refused it as a batch —
    with a message about batching, which is not what happened. The same
    mistake in the other direction is worse: `--` inside a literal made the
    comment stripper eat the rest of the statement, so what got analysed was
    not what would run.

    The executed SQL is untouched; only the copy the checks read is stripped.
    """
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"":
            out.append(ch + ch)
            i += 1
            while i < n:
                if sql[i] == ch:
                    if i + 1 < n and sql[i + 1] == ch:  # doubled = escaped
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _statement_problem(sql):
    """Why this is not a single read-only statement, or "" if it is."""
    text = re.sub(r"--[^\n]*", " ", _blank_strings(str(sql or "")))
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S).strip()
    if not text:
        return "no query was given"
    # A trailing semicolon is ordinary; a second statement after it is not.
    body = text[:-1] if text.endswith(";") else text
    if ";" in body:
        return ("one statement per query — a batch cannot be graded, because "
                "only the last result would come back")
    head = body.split(None, 1)[0].lower() if body.split() else ""
    if head not in _ALLOWED:
        return (f"{head or 'that'!r} is not a read: this lane runs "
                f"{', '.join(_ALLOWED)} and nothing else")
    return ""


def _render(value, limit=MAX_CELL_CHARS):
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes blob>"
    text = str(value)
    if len(text) > limit:
        return text[:limit] + f"… [cell truncated, {len(text)} chars total]"
    return text


def _redact_row(columns, rendered):
    """Blank the credentials in one already-rendered row.

    Returns `(row, redacted_columns)`. THE REPLACEMENT NAMES WHAT IT ATE. A
    silently emptied cell reads as a NULL in the source table, which is a
    finding — and a wrong one. `<redacted: api_key>` cannot be misread as data,
    and it still tells the reader the column exists and is populated, which is
    usually all the debugging actually needed.
    """
    hit = []
    row_key = ""
    for i, col in enumerate(columns):
        if str(col).lower() in _KEY_COLUMNS and i < len(rendered):
            row_key = rendered[i]
            break
    for i, col in enumerate(columns):
        if i >= len(rendered):
            break
        value = rendered[i]
        if not value:
            # Nothing to leak, and blanking an empty cell would claim a secret
            # is there. `host_secret` is empty in the engine right now.
            continue
        why = ""
        if _SECRET_NAME.search(str(col)):
            why = str(col)
        elif row_key and str(col).lower() not in _KEY_COLUMNS \
                and _SECRET_ROW_KEY.search(row_key):
            why = row_key
        elif _SECRET_VALUE.match(value):
            why = "credential-shaped value"
        # Last, so a cell that any of the named checks already caught is
        # reported by its NAME rather than by this one — the column or row key
        # is the more useful thing to tell the reader.
        elif _SECRET_IN_TEXT.search(value):
            why = "credential inside the value"
        if why:
            rendered[i] = f"<redacted: {why[:60]}, {len(value)} chars>"
            hit.append(str(col))
    return rendered, hit


def query(name, sql, max_rows=MAX_ROWS, max_chars=MAX_CHARS,
          time_limit=TIME_LIMIT_S, max_cell=MAX_CELL_CHARS):
    """Run one read-only statement and return rows, or an explicit failure.

    Returns {"ok", "columns", "rows", "row_count", "truncated", "why"} — or
    {"ok": False, "error"}. A failure is a RESULT, not an exception, for the
    same reason `read_file` returns one: the caller is a fetch verb whose job
    is to hand the reason back to the assistant, and a traceback loses it.

    THE CAP SAYS WHEN IT BIT. A result set silently cut at 200 rows is a
    census that reads as complete, and this repository has already paid for
    that once: a `skipped` list capped at 40 beside a true `skipped_count`.
    `truncated` and `why` are populated whenever anything was dropped, so a
    partial answer can never be mistaken for the whole one.
    """
    if not _DATABASES:
        # A registry that is empty because nobody loaded it looks exactly like
        # one that is empty because nothing is configured. Re-read the
        # environment before reporting an absence.
        configure()
    path = _DATABASES.get(str(name))
    if path is None:
        known = ", ".join(sorted(_DATABASES)) or "none are configured"
        return {"ok": False, "error": f"no reference database named "
                                      f"{name!r} — known: {known}"}
    if not os.path.isfile(path):
        return {"ok": False, "error": f"the file behind {name!r} is not "
                                      f"there: {path}"}
    problem = _statement_problem(sql)
    if problem:
        return {"ok": False, "error": problem}

    deadline = time.monotonic() + float(time_limit)
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, _PROGRESS_OPS)
        cur = conn.execute(str(sql))
        columns = [d[0] for d in (cur.description or [])]
        rows, chars, truncated, why = [], 0, False, ""
        redacted = set()
        for raw in cur:
            if len(rows) >= max_rows:
                truncated = True
                why = f"stopped at the {max_rows}-row cap"
                break
            rendered = [_render(v, max_cell) for v in raw]
            rendered, hit = _redact_row(columns, rendered)
            redacted.update(hit)
            size = sum(len(c) for c in rendered)
            if chars + size > max_chars and rows:
                truncated = True
                why = f"stopped at the {max_chars}-character cap"
                break
            rows.append(rendered)
            chars += size
        return {"ok": True, "columns": columns, "rows": rows,
                "row_count": len(rows), "truncated": truncated, "why": why,
                # SAID, not silent — for the same reason `truncated` is said.
                # An assistant reading a redacted key needs to know the lane
                # took it, or it will conclude the column is empty and go on
                # to explain a defect with that as a premise.
                "redacted": sorted(redacted)}
    except sqlite3.OperationalError as exc:
        # The progress handler aborts with "interrupted"; say what that MEANT,
        # because "interrupted" reads as a harness fault rather than as the
        # query having been too expensive to finish.
        if "interrupt" in str(exc).lower():
            return {"ok": False,
                    "error": f"the query ran past {time_limit:g}s and was "
                             f"stopped — narrow it with a WHERE or a LIMIT, "
                             f"and remember COUNT(*) over a large table is a "
                             f"full scan"}
        return {"ok": False, "error": f"sqlite refused it: {exc}"}
    except sqlite3.Error as exc:
        return {"ok": False, "error": f"sqlite refused it: {exc}"}
    finally:
        if conn is not None:
            conn.close()
