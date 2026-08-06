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
from db import q, qi, state_get, state_put, transaction

# One digest's share of a turn payload. Deliberately far below the argv wall
# that started this: the wall is a transport limit and this is a thinking
# limit, and the thinking limit is the smaller of the two.
DIGEST_CHAR_BUDGET = 12_000
# What the digest spends on everything that is not an entry: the pinned
# summary, the selection note, the not-indexed list and the usage note.
# Charged against the budget before the entries are filled, so the bound the
# caller is told is the bound that ships.
_WRAPPER_RESERVE = 3_000
GIST_CHARS = 150
MAX_EXPAND_CHARS = 24_000
# Below this a "chunk" is noise — a one-line helper on its own row buys an
# entry in the list and tells the reader nothing.
MIN_CHUNK_CHARS = 40
# The largest a single chunk may be. `MAX_EXPAND_CHARS` is what an expand may
# return, so a chunk above it is one no reader can ever open whole — visible in
# an outline and unreachable through it, which is worse than absent.
MAX_CHUNK_CHARS = 20_000

# What one pass over a workspace is allowed to read off disk. Characters, not
# files, for the reason stated above — and generous, because this bound is not
# about a model's context. Nothing here reaches a payload; the digest's own
# budget does that. This exists so a workspace containing a 900 MB dependency
# tree cannot turn one upload into a ten-minute read.
#
# Raised from 8M. The workspace now holds a 453-file project alongside the
# assistant's own tree: 8.57M characters of source in the engine plus ~1M of
# its own, so the old ceiling would have indexed the newest files and silently
# dropped the assistant's own codebase out of its own index — the drop is
# recorded, but recording it does not make an assistant that cannot find its
# own modules any less blind. Chunk rows are cheap; `DIGEST_CHAR_BUDGET` is
# what bounds the payload, and it is unchanged at 12k.
INGEST_CHAR_BUDGET = 24_000_000
# Where the last pass says what it did and what it left out. The record has to
# live somewhere `digest` can read without walking the filesystem, because
# `digest` runs on every turn and the walk does not.
INGEST_STATE_KEY = "chunk_index"

# The largest single file worth chunking. Above this it is data, not source.
#
# A workspace held two demo story blobs — 18.2 MB and 11.7 MB of JSON — which
# `language_of` recognises as a language and which therefore consumed 30 MB of
# a 24 MB budget between them. They chunk into thousands of entries no one can
# navigate by gist, and they pushed 27 real source files out of the index
# entirely: the assistant asked for the engine's `memory.py` and was told no
# indexed file matched, because two fixtures had spent the budget first.
#
# Set above the largest genuine source in either project (`commit.py` at 292 KB,
# `CHANGELOG.md` at 354 KB) so nothing hand-written is ever caught by it. A file
# over this is named and its size given, never dropped in silence — the same
# rule the rest of this module follows.
MAX_SOURCE_CHARS = 500_000


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


def split_markdown(text):
    """Split a document at its headings.

    MARKDOWN HAD NO SYMBOLS, SO IT BECAME ONE CHUNK PER FILE. `split_code`
    asks the symbol scanner for boundaries and falls back to the whole file
    when it finds none, which is right for an unparsed source and catastrophic
    for prose: the four documents carrying the most design intent in the
    ingested project came out atomic — `CHANGELOG.md` as a single chunk of
    352,149 characters, `UNBUILT.md` 166,980, `Design.md` 124,525 — against a
    24,000-character expand ceiling. The only way to read any part was to take
    the whole thing, which is the failure that killed a turn outright.

    Headings are what a document is navigated by, so they are what it is cut
    at. Falls through to `split_prose` for a document with no headings at all,
    rather than back to one chunk."""
    lines = str(text or "").splitlines()
    marks = [i for i, ln in enumerate(lines)
             if re.match(r"^#{1,3} +\S", ln)]
    if not marks:
        return split_prose(text)
    if marks[0] > 0:
        marks.insert(0, 0)
    out = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        head = lines[start].lstrip("# ").strip() if lines[start].startswith("#") \
            else "(preamble)"
        if out and len(body) < MIN_CHUNK_CHARS:
            out[-1]["body"] += "\n\n" + body
            continue
        # A HEADING IS NOT A LENGTH BOUND. Splitting on headings alone left a
        # section with no sub-headings whole — `Design.md` came out with an
        # 87,448-character chunk, still far over the 24,000 expand ceiling, so
        # the document was navigable but that part of it was still unreadable.
        # Long sections fall through to prose splitting, keeping the heading as
        # the title so the pieces still say where they came from.
        if len(body) > MAX_CHUNK_CHARS:
            for n_part, part in enumerate(split_prose(body), 1):
                out.append({"title": f"{head[:100]} ({n_part})",
                            "start": start + 1, "end": end,
                            "body": part["body"]})
            continue
        out.append({"title": head[:120] or "(section)",
                    "start": start + 1, "end": end, "body": body})
    return out or split_prose(text)


def _bound_pieces(pieces):
    """Break any piece longer than MAX_CHUNK_CHARS at line boundaries."""
    out = []
    for piece in pieces or []:
        body = str(piece.get("body") or "")
        if len(body) <= MAX_CHUNK_CHARS:
            out.append(piece)
            continue
        # A LINE CAN BE LONGER THAN THE BOUND. Line boundaries are the cut
        # that keeps a chunk readable, but a minified file or a single-line
        # data blob has none — 90,000 characters on one line came back as one
        # part and defeated the guard entirely. Hard-cut those; an unreadable
        # cut is still better than an unreachable chunk.
        lines = []
        for line in body.splitlines(True):
            while len(line) > MAX_CHUNK_CHARS:
                lines.append(line[:MAX_CHUNK_CHARS])
                line = line[MAX_CHUNK_CHARS:]
            lines.append(line)
        part, size = [], 0
        parts = []
        for line in lines:
            if part and size + len(line) > MAX_CHUNK_CHARS:
                parts.append("".join(part))
                part, size = [], 0
            part.append(line)
            size += len(line)
        if part:
            parts.append("".join(part))
        title = str(piece.get("title") or "")
        for n, chunk in enumerate(parts, 1):
            out.append({**piece, "body": chunk,
                        "title": f"{title[:100]} [part {n}/{len(parts)}]"})
    return out


def put(session_id=WORKSPACE, kind='code', source_ref='', pieces=()):
    """Store one source's chunks, replacing any previous set for that source.

    Replace rather than append: re-chunking a file the assistant has just
    edited must not leave the old version's chunks alongside the new ones,
    which is the failure mode where an expand returns code that no longer
    exists."""
    session_id = _scope(session_id)
    # ENFORCED HERE BECAUSE A SPLITTER CANNOT PROMISE IT. Every splitter cuts
    # at a structural boundary — symbols, headings, blank lines — and a file
    # that simply has none between two points produces a chunk as long as the
    # gap: a 276,241-character chunk from the code path, an 87,336 section of
    # `Design.md` with no sub-headings, both far over `MAX_EXPAND_CHARS`. A
    # chunk no expand can return is visible in an outline and unreachable
    # through it, which is worse than absent.
    #
    # So the bound lives where every chunk passes, not in each splitter that
    # must remember it. Cut at line boundaries, and the piece says it is one
    # part of several rather than pretending to be whole.
    pieces = _bound_pieces(pieces)
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


def ingest_workspace(session_id=WORKSPACE, budget=INGEST_CHAR_BUDGET):
    """Chunk every source file in the workspace. Returns the chunk count, and
    records what it did NOT index where `digest` will read it.

    THE SCAR IS THE SAME ONE, POINTED THE OTHER WAY. This bounded by
    `max_files=60` over a newest-modified-first listing, so the 61st file and
    everything older simply was not there — no error, no warning, nothing in
    the payload. The index still announced "N chunks across M sources" in the
    same confident tone, and an assistant reading it would conclude a symbol
    was absent from the codebase when it was only absent from the index.

    A silent truncation is worse than a crash. A crash is a fact; a corpus
    quietly missing its older half produces confident answers about code
    nobody looked at, and there is nothing in the output to tell them apart.
    So: the bound is characters (a count of things says nothing about their
    size — the rule this module opens with), and every file that falls out of
    it is named, with the reason, in a record the digest carries.

    Newest-modified-first is kept as the order deliberately. It is arbitrary
    with respect to importance, but recency is the one cheap signal that
    correlates with what the user is working on, and no ordering is defensible
    when the drop is silent — which is the half that is fixed here."""
    import workspace
    import os
    root = workspace.session_root(session_id)
    total, indexed, spent = 0, 0, 0
    skipped = []
    live = set()
    for entry in workspace.list_files(session_id):
        path = entry["path"]
        # A RUN'S OWN OUTPUT IS NOT THE PROJECT'S SOURCE. `_runs/` holds what
        # an experiment asked to collect, and a run may collect a `.py` or a
        # `.json` as readily as a `.txt` — which would then be indexed as if
        # the project contained it, and answer a later "where is this defined"
        # with a file the project never had. Recorded rather than dropped in
        # silence, for the same reason as everything else in this loop.
        if path.replace("\\", "/").split("/")[0] == workspace.RUN_OUTPUT_DIR:
            skipped.append({"path": path,
                            "why": "a previous run's collected output, not "
                                   "project source — read it directly"})
            continue
        language = codemap.language_of(os.path.basename(path))
        if not language:
            # Recorded, not skipped in silence. A reader comparing "56 files"
            # against "52 sources" and finding four unaccounted for cannot
            # tell a file this index does not handle from one it lost, and
            # spent a scout asking. The walked count minus the skipped count
            # is the indexed count, or the numbers are not checkable.
            skipped.append({"path": path, "why": "no recognised language"})
            continue
        # Checked BEFORE the read, so the first file is always indexed however
        # large it is: a workspace holding one enormous file must not index
        # nothing at all.
        if spent >= budget:
            skipped.append({"path": path, "why": "index budget spent"})
            continue
        # AN ARCHIVE IS NOT A LIVE CODEBASE, AND ITS PROSE IS THE USEFUL HALF.
        # `demo/` and `demos/` hold recorded stories and captured runs — 18.2 MB
        # and 11.7 MB of JSON in the project that prompted this — which chunk
        # into thousands of entries nobody can navigate by gist and which spent
        # the budget twenty-seven real modules needed. Their notes and write-ups
        # are worth having; their transcripts are not.
        #
        # Indexed, not hidden: the files stay in `list_files` and stay readable,
        # so an assistant that decides it needs the transcript can open it
        # deliberately. What it cannot do is have it arrive by default.
        if workspace.in_archive_dir(path) and language != "markdown":
            skipped.append({"path": path,
                            "why": "recorded run — only prose is indexed from "
                                   "an archive folder; read it directly to go "
                                   "deeper"})
            continue
        if entry.get("bytes", 0) > MAX_SOURCE_CHARS:
            skipped.append({
                "path": path,
                "why": f"{entry['bytes']:,} bytes — data, not source; over the "
                       f"{MAX_SOURCE_CHARS:,} ceiling"})
            continue
        try:
            with open(os.path.join(root, path), "r", encoding="utf-8") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({"path": path, "why": type(exc).__name__})
            continue
        if not source.strip():
            skipped.append({"path": path, "why": "empty"})
            continue
        spent += len(source)
        indexed += 1
        live.add(path)
        pieces = (split_markdown(source) if language == "markdown"
                  else split_code(source, language))
        total += len(put(session_id, "code", path, pieces))
    # ORPHANS OUTLIVE THE RULE THAT EXCLUDED THEM. `put` replaces per source,
    # so a file that STOPS being indexed is never revisited and its chunks sit
    # in the table forever — still returned by `outline` and `expand`, still
    # describing material the index no longer claims to cover. Excluding a
    # directory without this leaves exactly the stale-map failure `reingest_path`
    # exists to prevent, arriving from the other side.
    stale = [r["source_ref"] for r in q(
        "SELECT DISTINCT source_ref FROM chunks WHERE session_id=?",
        (_scope(session_id),)) if r["source_ref"] not in live]
    for ref in stale:
        qi("DELETE FROM chunks WHERE session_id=? AND source_ref=?",
           (_scope(session_id), ref))
    state_put(INGEST_STATE_KEY, {
        "indexed_sources": indexed, "chunks": total,
        "pruned_sources": len(stale),
        "read_chars": spent, "budget_chars": budget,
        "skipped_count": len(skipped), "skipped": skipped[:40]})
    return total


def reingest_path(relative, session_id=WORKSPACE):
    """Re-chunk one file after it changed on disk. Returns the chunk count.

    A MAP THAT OUTLIVES THE CODE IT DESCRIBES IS WORSE THAN NO MAP. Once the
    assistant can edit a file, every chunk of that file is a claim about what
    is there — and `expand` returning the version from before the edit is the
    failure `put` already guards against for a re-upload, arriving by the new
    route. Called by the write path rather than left to the caller, because
    the caller that forgets is the one whose edit silently desynchronises the
    index."""
    import os

    import workspace
    language = codemap.language_of(os.path.basename(str(relative or "")))
    if not language:
        return 0
    got = workspace.read_file(relative, session_id)
    if not got.get("ok"):
        # The file is gone or unreadable: drop its chunks rather than leave
        # them claiming it exists.
        qi("DELETE FROM chunks WHERE session_id=? AND source_ref=?",
           (_scope(session_id), str(relative)))
        return 0
    return len(put(session_id, "code", str(relative),
                   split_code(got["text"], language)))


def outline(path, session_id=WORKSPACE, limit=200):
    """Every chunk of ONE named file: id, title, gist, line range.

    THE MISSING STEP BETWEEN KNOWING A FILENAME AND READING IT. `digest` ranks
    the whole workspace against the turn's message, and `expand` needs ids —
    so an assistant told "fix coding.py" had no route to coding.py's ids
    unless the user's own words happened to rank it into the sample. Measured
    on a real turn: the message was "Go for it", nothing in it ranked, the
    digest showed 67 of 987 chunks and stopped at `beliefs.py`, and both the
    assistant and the deep subagent it delegated to reported they could not
    obtain an anchor for the file they had been asked to edit. Neither was
    confused; the lookup did not exist.

    Suffix match, so `coding.py` finds it wherever the tree puts it — an
    uploaded project sits under its archive's own directory, and requiring
    the full path would mean knowing the layout to ask about the file.
    Ambiguity is REPORTED rather than resolved by picking: two files with one
    basename is exactly when guessing is worst."""
    session_id = _scope(session_id)
    wanted = str(path or "").strip().strip("/")
    if not wanted:
        return {"path": "", "error": "no path given", "entries": []}
    rows = q("SELECT chunk_key,source_ref,title,gist,chars,start_line,end_line "
             "FROM chunks WHERE session_id=? ORDER BY source_ref, start_line, id",
             (session_id,))
    sources = sorted({r["source_ref"] for r in rows})
    matches = [s for s in sources
               if s == wanted or s.endswith("/" + wanted)]
    if not matches:
        near = [s for s in sources if wanted.lower() in s.lower()][:8]
        return {"path": wanted, "entries": [],
                "error": f"no indexed file matches {wanted!r}",
                **({"did_you_mean": near} if near else {}),
                "note": "`not_indexed` in the code digest lists files this "
                        "index does not cover."}
    if len(matches) > 1:
        return {"path": wanted, "entries": [],
                "error": f"{len(matches)} files match {wanted!r}; name one",
                "candidates": matches[:8]}
    source = matches[0]
    entries = [{"id": r["chunk_key"], "title": r["title"],
                "gist": r["gist"], "lines": [r["start_line"], r["end_line"]],
                "chars": r["chars"]}
               for r in rows if r["source_ref"] == source][:limit]
    return {"path": source, "chunks": len(entries), "entries": entries,
            "how_to_use_this": (
                "The whole file as an ordered list of pieces. Put the ids you "
                "need in `expand_chunks` to read their actual lines — an "
                "anchored edit needs text copied from an expansion, never "
                "from a gist.")}


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
            # A PATH WHERE AN ID WAS EXPECTED IS A REASONABLE MISTAKE, and
            # answering it with a bare `unknown` teaches nothing. The
            # assistant knows filenames long before it knows chunk ids, so
            # asking to expand one is the obvious wrong guess — met here with
            # the file's outline, which is what it needed to ask for.
            guess = outline(key, session_id)
            if guess.get("entries"):
                out.append({"id": key, "not_a_chunk_id": True,
                            "resolved_as_file": guess["path"],
                            "entries": guess["entries"],
                            "note": "that is a path, not a chunk id — here "
                                    "are its chunks; expand the ids you want"})
                continue
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


def _fit_chars(rows, budget):
    """As many rows as fit in `budget` characters, with the rest counted."""
    out, spent = [], 0
    for row in rows:
        size = len(json.dumps(row, ensure_ascii=False, default=str))
        if out and spent + size > budget:
            out.append({"path": f"…and {len(rows) - len(out)} more",
                        "why": "not listed: this report has a size budget too"})
            break
        out.append(row)
        spent += size
    return out


def _unwalked_sources(session_id, limit=8):
    """Source files present in the workspace that the index has never seen.

    Distinct from `skipped`, which is what an ingest walked and declined. This
    is what no ingest ever looked at — the failure that has no record anywhere,
    because the code that would have written the record did not run."""
    import os
    import workspace
    try:
        present = {e["path"] for e in workspace.list_files(session_id)
                   if codemap.language_of(os.path.basename(e["path"]))}
    except Exception:
        return []
    if not present:
        return []
    known = {r["source_ref"] for r in q(
        "SELECT DISTINCT source_ref FROM chunks WHERE session_id=?",
        (_scope(session_id),))}
    # A DELIBERATE EXCLUSION IS NOT A GAP, AND THEY NEED OPPOSITE RESPONSES.
    # This compared "present with a language" against "indexed" and called
    # every difference unwalked — so the archive transcripts, which an ingest
    # walked and skipped for a stated reason, were reported as files no ingest
    # had ever seen, advising a re-ingest that would change nothing. The rule
    # that excluded them has to be applied here too, or the report describes a
    # different workspace from the one the indexer saw.
    present = {p for p in present
               if not (workspace.in_archive_dir(p)
                       and codemap.language_of(os.path.basename(p))
                       != "markdown")}
    missing = sorted(present - known)
    if not missing:
        return []
    # NAMED, BUT NOT AT ANY PRICE. The rest of this module names what it drops
    # rather than counting it, and that is right — but this list is computed
    # from the filesystem and can be arbitrarily long, and it lands inside a
    # payload with a 12k budget. Naming forty files pushed a digest to 8.6k on
    # its own. So: name a few, and say how many there are.
    out = [{"path": p, "why": "never indexed — no ingest has walked it"}
           for p in missing[:limit]]
    if len(missing) > limit:
        out.append({"path": f"…and {len(missing) - limit} more",
                    "why": "never indexed — run an ingest to map them"})
    return out


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
    ranked, matched = rows, 0
    if query:
        scored = [(_relevance(query, r), -i, r) for i, r in enumerate(rows)]
        scored.sort(key=lambda s: (-s[0], -s[1]))
        matched = sum(1 for score, _i, _r in scored if score > 0)
        ranked = [r for score, _i, r in scored if score > 0] + \
                 [r for score, _i, r in scored if score <= 0]
    # THE BUDGET IS THE PAYLOAD'S, NOT THE ENTRY LIST'S. Entries were filled
    # to exactly `budget` and then the summary, the selection note, the
    # not-indexed list and the usage note were added on top — a digest that
    # reported a 12,000-character bound and shipped 18,815. Everything around
    # the entries is charged first, so the number the caller is given is the
    # number that arrives.
    #
    # Reserved rather than measured exactly: the wrapper cannot be built until
    # `shown` is known, and `shown` cannot be known until the wrapper is
    # charged. A fixed reserve breaks that circle in the safe direction.
    budget = max(1000, budget - _WRAPPER_RESERVE)
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
    index = state_get(INGEST_STATE_KEY) or {}
    dropped = index.get("skipped") or []
    # THE RECORD KNOWS ONLY WHAT THE LAST INGEST WALKED, WHICH IS NOT THE SAME
    # AS WHAT IS THERE. `ingest_workspace` runs from the upload and extract
    # routes — "map on the way in" — so a tree that arrives by any other door
    # is never walked, never indexed, and never skipped either: it is absent
    # from the record entirely, and the record still announces "N chunks across
    # M sources" in the same confident tone.
    #
    # That happened. A 453-file project was unpacked into the workspace by hand
    # and stayed invisible; `outline` answered "no indexed file matches" and an
    # assistant reading it would conclude the code was not there. This is the
    # module's own opening scar — a corpus quietly missing part of itself
    # produces confident answers about code nobody looked at — arriving through
    # a door the fix did not cover.
    #
    # So the drift is measured against the FILESYSTEM, not against memory. The
    # walk was assumed too expensive to run per turn; measured, it is 20ms for
    # 590 files, against a turn that costs a minute of model time.
    dropped = list(dropped) + _unwalked_sources(session_id)
    return {
        # Pinned: what the whole thing IS, before any of the parts.
        "summary": summary or (
            f"{len(rows)} chunks across {len(sources)} sources"
            + (": " + ", ".join(sources[:8]) if sources else "")
            + ("…" if len(sources) > 8 else "")
            + (f" — at least {len(dropped)} file(s) in the workspace are NOT "
               "in this index, listed under `not_indexed`"
               if dropped else "")),
        "total_chunks": len(rows),
        "showing": shown,
        # HOW the shown entries were chosen, not just how many. A ranked list
        # degrades gracefully and an arbitrary slice hides whole modules, and
        # from inside the payload the two are indistinguishable — a reader who
        # cannot tell them apart cannot know whether "not in the list" means
        # "not relevant" or "not looked at".
        "selection": (
            f"ranked by relevance to this turn; {matched} of {len(rows)} "
            "chunks matched at least one term, the rest follow in source order"
            if query else "source order — nothing was given to rank against"),
        "entries": entries,
        # Named, not counted. "3 files were skipped" tells a reader something
        # is missing and not whether it is the one they are about to
        # conclude does not exist.
        #
        # And the LIST is capped where the count is not, so past 40 the two
        # disagree — a reader counting entries in this array comes up short.
        # That is the module's own defect reappearing inside the fix for it,
        # which is why it is stated rather than left to be noticed.
        # BOUNDED IN CHARACTERS LIKE EVERYTHING ELSE HERE. This list was
        # capped at 40 ENTRIES and sat outside the budget entirely, so a
        # digest that had trimmed its entries to exactly 12,032 characters
        # then shipped 5,392 more — 45% on top of a bound the caller was told
        # was the bound. The module opens by saying a limit on the number of
        # things says nothing about their size, and then this.
        #
        # A share rather than the whole budget: naming what is missing matters,
        # and it matters less than the map itself.
        **({"not_indexed": _fit_chars(dropped, budget // 8)}
           if dropped else {}),
        **({"not_indexed_is_partial":
            f"{len(dropped)} of {index.get('skipped_count')} listed; "
            "read the count, not the length of the list"}
           if len(dropped) < int(index.get("skipped_count") or 0) else {}),
        **({"expanded": expand(session_id, expand_ids)} if expand_ids else {}),
        "how_to_use_this": (
            "A LIST, not the material. Each entry is a gist and an id; the "
            "bodies are not here. Pick the ids that look relevant and put "
            "them in `expand_chunks` to see them next round. `showing` below "
            "`total_chunks` means you are looking at part of the list and "
            "there is more to ask for — say so rather than concluding from "
            "the sample. Anything under `not_indexed` is a file in the "
            "workspace that this index does NOT cover: absent from the list "
            "is not evidence of absent from the code."),
    }
