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
        mapping = {}
        for pair in (os.environ.get("ASSISTANT_REFDB") or "").split(","):
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


def _statement_problem(sql):
    """Why this is not a single read-only statement, or "" if it is."""
    text = re.sub(r"--[^\n]*", " ", str(sql or ""))
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


def _render(value):
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes blob>"
    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return (text[:MAX_CELL_CHARS]
                + f"… [cell truncated, {len(text)} chars total]")
    return text


def query(name, sql, max_rows=MAX_ROWS, max_chars=MAX_CHARS,
          time_limit=TIME_LIMIT_S):
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
        for raw in cur:
            if len(rows) >= max_rows:
                truncated = True
                why = f"stopped at the {max_rows}-row cap"
                break
            rendered = [_render(v) for v in raw]
            size = sum(len(c) for c in rendered)
            if chars + size > max_chars and rows:
                truncated = True
                why = f"stopped at the {max_chars}-character cap"
                break
            rows.append(rendered)
            chars += size
        return {"ok": True, "columns": columns, "rows": rows,
                "row_count": len(rows), "truncated": truncated, "why": why}
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
