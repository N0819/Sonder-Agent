# chunks.py — large material as a navigable list, not a wall of text.
#
# The problem this solves is the one that produced `[Errno 7] Argument list
# too long`: a 115-file upload became 97 KB of codemap in a single turn
# payload. Raising the ceiling only moves the wall — the model's context is
# the real limit, and a payload that spends all of it describing a codebase
# has none left for thinking about it.
#
# THE SHAPE: a DIGEST is a pinned summary, then one line per chunk (an id and
# a gist), and then only those chunks that were explicitly expanded. The
# assistant reads the summary, scans the gists, and asks for the three entries
# that matter. It works off relevance instead of off everything.
#
# Four decisions, each with a reason:
#
# 1. GISTS ARE DERIVED, NEVER MODEL-WRITTEN. A model-written gist costs a call
#    per chunk and can misdescribe what it summarises, and a wrong gist is
#    worse than none: it is the thing the assistant navigates by, so an
#    invented one sends it confidently to the wrong chunk. A signature plus
#    the first docstring line is not elegant, but it cannot lie.
#
# 2. CHUNKS SPLIT AT STRUCTURAL BOUNDARIES, never at a character count. A
#    function cut in half is worse than absent — absent is visible, and half a
#    function reads as a whole one. Code splits at symbol starts, prose at
#    blank lines.
#
# 3. A DIGEST ALWAYS STATES total VERSUS showing. `codemap.for_prompt` already
#    established this: the difference between an agent that asks for more and
#    one that assumes it has everything is knowing it is looking at a sample.
#
# 4. THE BUDGET IS IN CHARACTERS, NOT ENTRIES. `codemap_for` bounded by file
#    count (80 files) and produced 97 KB, because a bound on the number of
#    things says nothing about their size. Every limit here is a character
#    budget, and what it drops it says it dropped.

import hashlib
import json
import re
import time

import codemap
from db import q, qi, transaction

# One digest's share of a turn payload. Deliberately far below the argv wall
# that started this: the wall is a transport limit and this is a thinking
# limit, and the thinking limit is the smaller of the two.
DIGEST_CHAR_BUDGET = 12_000
GIST_CHARS = 150
MAX_EXPAND_CHARS = 24_000
# Below this a "chunk" is noise — a one-line helper on its own row buys an
# entry in the list and tells the reader nothing.
MIN_CHUNK_CHARS = 40


# The chunk map shares the WORKSPACE's lifetime, not a session's.
#
# The files are persistent now, so a map keyed per session would be rebuilt
# from scratch on every new conversation while describing exactly the same
# tree — and worse, a session that never uploaded anything would see an empty
# map and conclude there was no code. One workspace, one map.
#
# The column stays `session_id` because renaming it would be a migration that
# buys nothing; what it holds is this constant.
WORKSPACE = 0


def _scope(_ignored=None):
    """Always the workspace. The argument is accepted and discarded.

    A DEFAULT WAS NOT ENOUGH, and the failure is the reason this function
    exists. Making `session_id` default to WORKSPACE left four call sites
    still passing a live session id, so the map was written at scope 0 and
    read at scope N: `digest` returned "0 chunks across 0 sources" while 777
    rows sat in the table, and `expand` could not resolve an id even if one
    had been offered. Silent, total, and invisible from either end.

    That is the identity lesson from AGENTS.md, arrived at from the other
    direction — two spellings of one scope existed, so they had to be folded
    where the data enters rather than at each caller who must remember. A
    guard that must be remembered will be forgotten; this one cannot be."""
    return WORKSPACE


def _key(session_id, source_ref, ordinal):
    """A short, stable handle the model can name back at us.

    Content-independent on purpose: a chunk keeps its id when its body
    changes, so an expand request issued one round ago still resolves to the
    thing the assistant meant rather than silently to nothing."""
    raw = f"{session_id}:{source_ref}:{ordinal}"
    return "c" + hashlib.blake2s(raw.encode(), digest_size=4).hexdigest()


def _first_sentence(text, limit=GIST_CHARS):
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    cut = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return (cut if len(cut) <= limit else cut[:limit - 1].rstrip() + "…")


def _code_gist(body, name, kind):
    """Signature plus the first line of the docstring, when there is one.

    This is the whole navigational surface for a code chunk, so it is built
    from the two things a reader would look at first and from nothing that
    requires understanding the body."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    signature = lines[0].strip() if lines else f"{kind} {name}"
    doc = ""
    match = re.search(r'"""(.*?)(?:"""|$)', body, re.S)
    if match:
        doc = _first_sentence(match.group(1), 90)
    out = signature[:110] + (f" — {doc}" if doc else "")
    return out[:GIST_CHARS]


def split_code(text, language):
    """Split source at top-level symbol boundaries.

    Falls back to one chunk for the whole file when nothing parses, rather
    than to arbitrary slices: an unparsed file the assistant can read whole is
    more useful than one cut at column 2000."""
    symbols = (codemap._python_symbols(text)[0] if language == "python"
               else codemap._regex_symbols(text, language)[0])
    lines = text.splitlines()
    tops = [(ln, nm, kind) for kind, nm, ln in symbols if ln and ln <= len(lines)]
    tops.sort()
    if not tops:
        return [{"title": "(whole file)", "start": 1, "end": len(lines),
                 "body": text}]
    out = []
    # A preamble — imports, module docstring, constants — is a real chunk and
    # is frequently the one that answers "what is this file".
    if tops[0][0] > 1:
        body = "\n".join(lines[:tops[0][0] - 1])
        if len(body.strip()) >= MIN_CHUNK_CHARS:
            out.append({"title": "(module preamble)", "start": 1,
                        "end": tops[0][0] - 1, "body": body})
    for i, (start, name, kind) in enumerate(tops):
        end = tops[i + 1][0] - 1 if i + 1 < len(tops) else len(lines)
        body = "\n".join(lines[start - 1:end])
        if len(body.strip()) < MIN_CHUNK_CHARS and out:
            # Too small to navigate to: fold it into the previous entry rather
            # than making the list longer and less informative.
            out[-1]["body"] += "\n" + body
            out[-1]["end"] = end
            continue
        out.append({"title": f"{kind} {name}", "start": start, "end": end,
                    "body": body})
    return out


def split_prose(text):
    """Split prose at blank lines, merging runs too small to navigate to."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", str(text or ""))]
    out = []
    for block in blocks:
        if not block:
            continue
        if out and len(block) < MIN_CHUNK_CHARS * 4:
            out[-1] += "\n\n" + block
        else:
            out.append(block)
    return [{"title": _first_sentence(b, 60), "start": None, "end": None,
             "body": b} for b in out]


def put(session_id=WORKSPACE, kind='code', source_ref='', pieces=()):
    """Store one source's chunks, replacing any previous set for that source.

    Replace rather than append: re-chunking a file the assistant has just
    edited must not leave the old version's chunks alongside the new ones,
    which is the failure mode where an expand returns code that no longer
    exists."""
    session_id = _scope(session_id)
    rows = []
    for ordinal, piece in enumerate(pieces):
        body = str(piece.get("body") or "")
        gist = (piece.get("gist")
                or (_code_gist(body, piece.get("title") or "", kind)
                    if kind == "code" else _first_sentence(body)))
        rows.append((_key(session_id, source_ref, ordinal), session_id, kind,
                     source_ref, str(piece.get("title") or "")[:200],
                     gist[:GIST_CHARS], body, piece.get("start"),
                     piece.get("end"), len(body), time.time()))
    with transaction():
        qi("DELETE FROM chunks WHERE session_id=? AND source_ref=?",
           (session_id, source_ref))
        for row in rows:
            qi("""INSERT INTO chunks(chunk_key,session_id,kind,source_ref,
                title,gist,body,start_line,end_line,chars,created)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id,chunk_key) DO UPDATE SET
                title=excluded.title, gist=excluded.gist, body=excluded.body,
                start_line=excluded.start_line, end_line=excluded.end_line,
                chars=excluded.chars""", row)
    return [r[0] for r in rows]


def ingest_workspace(session_id=WORKSPACE, max_files=60):
    """Chunk every source file in a session's workspace. Returns a count."""
    import workspace
    import os
    root = workspace.session_root(session_id)
    total = 0
    for entry in workspace.list_files(session_id)[:max_files]:
        path = entry["path"]
        language = codemap.language_of(os.path.basename(path))
        if not language:
            continue
        try:
            with open(os.path.join(root, path), "r", encoding="utf-8") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not source.strip():
            continue
        total += len(put(session_id, "code", path, split_code(source, language)))
    return total


def expand(session_id=WORKSPACE, ids=(), budget=MAX_EXPAND_CHARS):
    """Bodies for the ids asked for, in the order asked, within a budget.

    An id that does not resolve comes back as an explicit `unknown` entry
    rather than being dropped: silence would read to the assistant as "that
    chunk was empty", and it would carry on as though it had looked."""
    session_id = _scope(session_id)
    wanted = [str(i).strip() for i in (ids or []) if str(i).strip()][:24]
    if not wanted:
        return []
    rows = {r["chunk_key"]: r for r in q(
        "SELECT chunk_key,kind,source_ref,title,gist,body,start_line,end_line "
        "FROM chunks WHERE session_id=? AND chunk_key IN (%s)"
        % ",".join("?" * len(wanted)), (session_id, *wanted))}
    out, spent = [], 0
    for key in wanted:
        row = rows.get(key)
        if row is None:
            out.append({"id": key, "unknown": True})
            continue
        body = row["body"]
        if spent + len(body) > budget:
            body = body[:max(0, budget - spent)]
            out.append({"id": key, "source": row["source_ref"],
                        "title": row["title"], "text": body,
                        "truncated_by_budget": True})
            break
        spent += len(body)
        out.append({"id": key, "source": row["source_ref"],
                    "title": row["title"],
                    "lines": [row["start_line"], row["end_line"]],
                    "text": body})
    return out


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Words that appear in every question and discriminate nothing. Without this
# list, "how does the embeddings rebuild work" ranked `test_workspace.py`
# first — on "work", "does" and "how" — and the one term that mattered was
# outvoted by four that did not.
_STOP = frozenset("""
the and for with that this from what how does did are was were you your има
not but its it has have can could will would should about into over under
only just very more most some any all one two now then than they them their
there here when where which who whom whose why get set use used using make
""".split())


def _tokens(text):
    """Words AND their identifier parts, so `rebuild_embeddings` is findable
    by "rebuild" and by "embeddings" alike. Splitting snake_case and camelCase
    is what makes a question in English match a name in code."""
    out = set()
    for raw in _TOKEN.findall(str(text or "")):
        out.add(raw.lower())
        for part in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", raw):
            if len(part) > 2:
                out.add(part.lower())
    return out - _STOP


def _relevance(query, row):
    """How much this chunk looks like an answer to `query`.

    Deterministic set overlap over tokens — not substring containment, which
    is what made "work" match "workspace" and put the archive tests at the top
    of a question about embeddings. Not a model call and not an embedding
    either: this runs over every chunk on every turn, so it has to be free,
    and where code can decide, code decides.

    Path matches count double. Asking about `config` should surface
    `config.py` above a passing mention of the word in another file's
    docstring, and the filename is the strongest cheap evidence of that."""
    terms = _tokens(query)
    if not terms:
        return 0.0
    gist = _tokens(str(row["gist"] or "") + " " + str(row["title"] or ""))
    path = _tokens(str(row["source_ref"] or "").replace("/", " "))
    return (len(terms & gist) + 2 * len(terms & path)) / (len(terms) + 2.0)


def digest(session_id=WORKSPACE, *, kind=None, expand_ids=(), budget=DIGEST_CHAR_BUDGET,
           summary="", query=""):
    """The navigable view: a pinned summary, a gist per chunk, and only the
    chunks explicitly expanded.

    This is the payload shape the whole module exists for. Everything in it is
    bounded, and every bound that bites is stated in the payload rather than
    applied silently."""
    session_id = _scope(session_id)
    args = [session_id]
    sql = "SELECT chunk_key,kind,source_ref,title,gist,chars FROM chunks " \
          "WHERE session_id=?"
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    rows = q(sql + " ORDER BY source_ref, start_line, id", tuple(args))
    # WHICH chunks get shown is the whole feature. Source order shows the
    # alphabetically-first files and calls it a sample; ranked order shows
    # what the turn is actually about. The list stays in source order for
    # anything the query does not discriminate on, so an empty query behaves
    # exactly as before rather than shuffling.
    ranked = rows
    if query:
        scored = [(_relevance(query, r), -i, r) for i, r in enumerate(rows)]
        scored.sort(key=lambda s: (-s[0], -s[1]))
        ranked = [r for score, _i, r in scored if score > 0] + \
                 [r for score, _i, r in scored if score <= 0]
    entries, spent, shown = [], 0, 0
    for row in ranked:
        line = {"id": row["chunk_key"], "source": row["source_ref"],
                "gist": row["gist"] or row["title"], "chars": row["chars"]}
        cost = len(json.dumps(line, ensure_ascii=False))
        if spent + cost > budget:
            break
        spent += cost
        shown += 1
        entries.append(line)
    sources = sorted({r["source_ref"] for r in rows})
    return {
        # Pinned: what the whole thing IS, before any of the parts.
        "summary": summary or (
            f"{len(rows)} chunks across {len(sources)} sources"
            + (": " + ", ".join(sources[:8]) if sources else "")
            + ("…" if len(sources) > 8 else "")),
        "total_chunks": len(rows),
        "showing": shown,
        "entries": entries,
        **({"expanded": expand(session_id, expand_ids)} if expand_ids else {}),
        "how_to_use_this": (
            "A LIST, not the material. Each entry is a gist and an id; the "
            "bodies are not here. Pick the ids that look relevant and put "
            "them in `expand_chunks` to see them next round. `showing` below "
            "`total_chunks` means you are looking at part of the list and "
            "there is more to ask for — say so rather than concluding from "
            "the sample."),
    }
