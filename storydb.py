"""Reading a story out of a live engine database, by the name a person uses.

WHY THIS EXISTS. A bug report about interactive fiction arrives as "the Blizzard
story went wrong around turn 40" — a name a person remembers and a number they
counted off the screen. Everything needed to investigate that is in `engine.db`
and reachable through `refdb` today, but only as SQL, and the SQL is not the
hard part: `chats` is keyed by integer, turns are numbered per chat, the agent
outputs are one join away in `variants`, and none of that is guessable from the
report. Every investigation therefore began by rediscovering the schema, and
the rediscovery cost more rounds than the defect did.

So this is the resolution layer, not a new door. Every read still goes through
`refdb.query`, which means the row caps, the wall-clock abort and the credential
redaction all apply here for free and cannot be forgotten — a second connection
would have been a second place to remember them.

WHAT IT REFUSES TO DO. It does not pick a story. Branches in this engine are
named by suffixing the parent — 'Elyndra — Hinami', 'Elyndra — Hinami ⎇16',
'Elyndra — Hinami ⎇16 ⎇1' — so the name a person says is routinely a prefix of
three real chats, and the one they mean is usually not the one a LIKE would
rank first. Choosing silently would aim the whole investigation at the wrong
transcript, and nothing downstream could tell: every turn would be real, every
quotation checkable, the theory internally consistent and about another story.
An ambiguous name comes back as a list, with the turn counts that let a person
say which one they meant.

TURN NUMBERS ARE `turns.idx`, per chat, zero-based — the same number the engine
shows. `turns.id` is a global row id and the two are wildly different (turn 0 of
'Run!' is row 1388), so every function here takes and reports `idx`, and says so
in what it returns.
"""

from __future__ import annotations

import json

import refdb

DEFAULT_DATABASE = "engine"

# A step's active variant is the agent's actual output, and a narrator variant
# runs to several thousand characters. The default cell cap is 2000, which cuts
# prose mid-sentence — fine for a census, useless for reading what the model
# said. These lanes raise it deliberately and per call.
DETAIL_CELL_CHARS = 12000
DETAIL_TOTAL_CHARS = 60000
INPUT_PREVIEW = 400


def _rows(sql, database, **kw):
    """Run one read and hand back dicts, or the failure unchanged."""
    result = refdb.query(database, sql, **kw)
    if not result.get("ok"):
        return None, result
    columns = result.get("columns") or []
    out = [dict(zip(columns, row)) for row in result.get("rows") or []]
    return out, result


def _int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _note(result):
    """The cap message, if one bit, for passing through to the caller."""
    if result.get("truncated"):
        return result.get("why") or "truncated"
    return ""


def find_story(name, database=DEFAULT_DATABASE, limit=20):
    """Which chats could 'the Blizzard story' mean, and how long is each.

    Matched case-insensitively on any substring, because people quote the part
    of a title they remember rather than the whole of it.

    NEVER RETURNS ONE WHEN SEVERAL MATCH — see the module docstring. The reply
    always carries `matches` as a list and `ambiguous`, so a caller that wants
    to proceed has to have looked at the count.
    """
    text = str(name or "").strip()
    if not text:
        return {"ok": False, "error": "no story name was given"}
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    # Parameters would be better and this lane has none — `refdb.query` takes a
    # statement and nothing else, deliberately, so that what runs is what was
    # written. The quoting below is the price: doubled single quotes, and the
    # statement gate upstream refuses anything that is not a single SELECT.
    literal = escaped.replace("'", "''")
    rows, result = _rows(
        "SELECT c.id AS chat_id, c.name AS name, c.created AS created, "
        "c.branched_from AS branched_from, "
        "COUNT(t.id) AS turns, MIN(t.idx) AS first_idx, MAX(t.idx) AS last_idx "
        "FROM chats c LEFT JOIN turns t ON t.chat_id = c.id "
        f"WHERE c.name LIKE '%{literal}%' ESCAPE '\\' "
        "GROUP BY c.id ORDER BY c.id DESC "
        f"LIMIT {_int(limit, 20)}", database)
    if rows is None:
        return result
    if not rows:
        # AN EMPTY RESULT IS NOT AN ANSWER HERE. "No story called that" and "I
        # spelled it differently from the person who named it" look identical,
        # and the second is far more common. Hand back what does exist so the
        # next round is a choice rather than another guess.
        recent, _ = _rows(
            "SELECT c.id AS chat_id, c.name AS name, COUNT(t.id) AS turns "
            "FROM chats c LEFT JOIN turns t ON t.chat_id = c.id "
            "GROUP BY c.id ORDER BY c.id DESC LIMIT 15", database)
        return {"ok": True, "query": text, "matches": [], "ambiguous": False,
                "nothing_matched": True,
                "note": "no chat name contains that. These exist — the story "
                        "may be recorded under a different name than the one "
                        "it is called out loud.",
                "recent_stories": recent or []}
    return {"ok": True, "query": text, "matches": rows,
            "match_count": len(rows),
            "ambiguous": len(rows) > 1,
            "turn_numbering": "turns.idx, per chat, zero-based",
            **({"note": "several stories match. Branches are named by "
                        "suffixing the parent, so these are probably one "
                        "story and its forks — the turn counts and ids "
                        "distinguish them. Do not investigate until it is "
                        "settled which one the report is about."}
               if len(rows) > 1 else {}),
            **({"cap": _note(result)} if _note(result) else {})}


def story_turns(chat_id, lo=None, hi=None, database=DEFAULT_DATABASE,
                limit=60):
    """The turns of one story in a range: what the player typed, which agents
    ran, and which of their outputs the engine marked stale.

    This is the census before the reading. A turn where a step is missing
    entirely, or where `stale` is set, is a different defect from one where an
    agent ran and said something wrong — and the two are indistinguishable from
    the prose alone, which is the only thing the person reporting the bug saw.
    """
    cid = _int(chat_id)
    if cid is None:
        return {"ok": False, "error": f"chat_id must be an integer, got "
                                      f"{chat_id!r} — resolve the story name "
                                      f"with find_story first"}
    where = [f"t.chat_id = {cid}"]
    if _int(lo) is not None:
        where.append(f"t.idx >= {_int(lo)}")
    if _int(hi) is not None:
        where.append(f"t.idx <= {_int(hi)}")
    rows, result = _rows(
        "SELECT t.idx AS idx, t.id AS turn_row_id, t.player_input AS input, "
        "t.created AS created, "
        "(SELECT COUNT(*) FROM steps s WHERE s.turn_id = t.id) AS steps, "
        "(SELECT COUNT(*) FROM steps s WHERE s.turn_id = t.id AND s.stale = 1)"
        " AS stale_steps, "
        "(SELECT GROUP_CONCAT(s.key, ' ') FROM (SELECT key, ord FROM steps "
        "WHERE turn_id = t.id ORDER BY ord) s) AS step_keys "
        f"FROM turns t WHERE {' AND '.join(where)} "
        f"ORDER BY t.idx LIMIT {_int(limit, 60)}", database,
        max_cell=1200)
    if rows is None:
        return result
    for row in rows:
        text = str(row.get("input") or "")
        row["input"] = text[:INPUT_PREVIEW] + ("…" if len(text) > INPUT_PREVIEW
                                               else "")
        row["steps"] = _int(row.get("steps"), 0)
        row["stale_steps"] = _int(row.get("stale_steps"), 0)
    story, _ = _rows(f"SELECT name, scenario FROM chats WHERE id = {cid}",
                     database, max_cell=600)
    if not rows:
        # Say which range was empty and what the story actually spans. An empty
        # list otherwise reads as "those turns went wrong in a way that left no
        # record", which is a far more alarming finding than "off by one".
        span, _ = _rows("SELECT MIN(idx) AS first_idx, MAX(idx) AS last_idx, "
                        f"COUNT(*) AS turns FROM turns WHERE chat_id = {cid}",
                        database)
        return {"ok": True, "chat_id": cid, "turns": [],
                "story": (story or [{}])[0].get("name", ""),
                "nothing_in_range": True,
                "story_span": (span or [{}])[0],
                "note": "no turn in that range. The span above is what this "
                        "story actually has."}
    return {"ok": True, "chat_id": cid,
            "story": (story or [{}])[0].get("name", ""),
            "scenario": (story or [{}])[0].get("scenario", ""),
            "turn_numbering": "turns.idx, per chat, zero-based",
            "returned": len(rows), "turns": rows,
            **({"cap": _note(result) + " — narrow the range"}
               if _note(result) else {})}


def turn_detail(chat_id, idx, step=None, database=DEFAULT_DATABASE,
                chars=DETAIL_TOTAL_CHARS):
    """What every agent actually produced on one turn of one story.

    THIS IS THE EVIDENCE. The engine runs a turn as a chain of agents —
    director, mapping, perception, the character loop, narrator, commit — and
    each one's output is stored as the active variant of its step. A defect the
    player saw in the prose was almost always introduced upstream of the
    narrator, and reading only the prose is reading the last stage where it
    became visible rather than the earliest where it became wrong.

    Pass `step` to read one agent's output at full length once the census has
    narrowed it; the whole chain at once is bounded and will be cut.
    """
    cid, i = _int(chat_id), _int(idx)
    if cid is None or i is None:
        return {"ok": False, "error": "chat_id and idx must both be integers "
                                      "— idx is the per-chat turn number"}
    where = f"t.chat_id = {cid} AND t.idx = {i}"
    if step:
        key = str(step).replace("'", "''")
        where += f" AND s.key = '{key}'"
    rows, result = _rows(
        "SELECT s.key AS step, s.label AS label, s.ord AS ord, "
        "s.stale AS stale, v.content AS content, v.reasoning AS reasoning, "
        # ASKED IN SQL, because a NULL does not survive the trip. Every cell
        # comes back rendered as text, so a missing variant arrived as the
        # four-character string "None" and read as content — an agent that
        # stored nothing was indistinguishable from one that stored the word.
        "(v.content IS NULL) AS missing_variant, "
        "(SELECT COUNT(*) FROM variants v2 WHERE v2.step_id = s.id) AS variants "
        "FROM turns t JOIN steps s ON s.turn_id = t.id "
        "LEFT JOIN variants v ON v.step_id = s.id AND v.active = 1 "
        f"WHERE {where} ORDER BY s.ord", database,
        max_cell=DETAIL_CELL_CHARS, max_chars=_int(chars, DETAIL_TOTAL_CHARS))
    if rows is None:
        return result
    if not rows:
        return {"ok": True, "chat_id": cid, "idx": i, "steps": [],
                "nothing_found": True,
                "note": "that turn has no steps recorded — either the idx is "
                        "outside the story, or the turn died before its first "
                        "agent wrote anything. story_turns tells you which."}
    turn, _ = _rows("SELECT player_input FROM turns "
                    f"WHERE chat_id = {cid} AND idx = {i}", database,
                    max_cell=4000)
    for row in rows:
        row["stale"] = bool(_int(row.get("stale"), 0))
        if _int(row.pop("missing_variant", 0), 0):
            # A step with no active variant ran and stored nothing, which is a
            # different failure from a step that never ran at all. Neither is
            # visible in the prose, and they need different fixes.
            row["no_active_variant"] = True
            row["content"] = None
    return {"ok": True, "chat_id": cid, "idx": i,
            "player_input": (turn or [{}])[0].get("player_input", ""),
            "steps": rows, "returned": len(rows),
            **({"cap": _note(result) + " — read one step at a time with "
                                       "`step`"} if _note(result) else {})}


def story_memories(chat_id, lo=None, hi=None, database=DEFAULT_DATABASE,
                   limit=60):
    """What the story committed to memory across a range of turns.

    Aimed at the defect class where the prose is fine and the story goes wrong
    two beats later: a memory written with the wrong provenance, the wrong
    salience, or attached to the wrong character reads as nothing at all on the
    turn it was written and as a continuity break afterwards. Embeddings are
    excluded — they are large, and they are not what anyone reads.
    """
    cid = _int(chat_id)
    if cid is None:
        return {"ok": False, "error": "chat_id must be an integer"}
    where = [f"m.chat_id = {cid}"]
    if _int(lo) is not None:
        where.append(f"m.turn_idx >= {_int(lo)}")
    if _int(hi) is not None:
        where.append(f"m.turn_idx <= {_int(hi)}")
    rows, result = _rows(
        "SELECT m.turn_idx AS turn_idx, m.kind AS kind, m.category AS category,"
        " m.provenance AS provenance, m.salience AS salience, "
        "m.importance AS importance, m.confidence AS confidence, "
        "m.disputed AS disputed, m.archived AS archived, "
        "c.name AS character, m.gist AS gist, m.content AS content "
        "FROM memories m LEFT JOIN characters c ON c.id = m.char_id "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY m.turn_idx, m.id LIMIT {_int(limit, 60)}", database,
        max_cell=1500)
    if rows is None:
        return result
    return {"ok": True, "chat_id": cid, "returned": len(rows),
            "memories": rows,
            **({"cap": _note(result)} if _note(result) else {})}


def story_lorebooks(chat_id, database=DEFAULT_DATABASE):
    """Every lorebook a story can actually retrieve from, and by which route.

    THE TRAP THIS EXISTS FOR. A chat reaches its lore by TWO independent
    paths — `chats.lorebook_id`, a single column, and `chat_lorebooks`, a link
    table — and they routinely disagree. Live: chat 63's `lorebook_id` is 184,
    a seven-entry book of odds and ends the engine minted during play, while
    the authored shrine book its whole story rests on is 187 and reaches it
    only through `chat_lorebooks`. Reading the column alone answers "this
    story has no layout lore", which is false and looks like a finding.

    So both routes are resolved and each book says how it got here. `enabled`
    is carried because a linked book can be switched off, and a disabled book
    that still appears in a join is the next version of this same mistake.
    """
    cid = _int(chat_id)
    if cid is None:
        return {"ok": False, "error": "chat_id must be an integer"}
    rows, result = _rows(
        "SELECT b.id AS book_id, b.name AS name, b.book_type AS book_type, "
        "'chat_lorebooks' AS route, cl.enabled AS enabled, "
        "(SELECT COUNT(*) FROM lore_entries e WHERE e.lorebook_id = b.id) "
        "AS entries, "
        "(SELECT COUNT(*) FROM lore_entries e WHERE e.lorebook_id = b.id "
        " AND e.canon_locked = 1) AS canon_locked, "
        "(SELECT COUNT(*) FROM lore_entries e WHERE e.lorebook_id = b.id "
        " AND e.turn_added IS NOT NULL) AS written_in_play "
        "FROM chat_lorebooks cl JOIN lorebooks b ON b.id = cl.lorebook_id "
        f"WHERE cl.chat_id = {cid} "
        "UNION "
        "SELECT b.id, b.name, b.book_type, 'chats.lorebook_id', 1, "
        "(SELECT COUNT(*) FROM lore_entries e WHERE e.lorebook_id = b.id), "
        "(SELECT COUNT(*) FROM lore_entries e WHERE e.lorebook_id = b.id "
        " AND e.canon_locked = 1), "
        "(SELECT COUNT(*) FROM lore_entries e WHERE e.lorebook_id = b.id "
        " AND e.turn_added IS NOT NULL) "
        "FROM chats c JOIN lorebooks b ON b.id = c.lorebook_id "
        f"WHERE c.id = {cid} "
        "ORDER BY entries DESC", database)
    if rows is None:
        return result
    return {"ok": True, "chat_id": cid, "returned": len(rows), "books": rows,
            "note": "a book reached only by 'chats.lorebook_id' and a book "
                    "reached only by 'chat_lorebooks' are both live; reading "
                    "one route alone is how a story looks lore-less",
            **({"cap": _note(result)} if _note(result) else {})}


def lore_entries(book_id=None, chat_id=None, text=None,
                 database=DEFAULT_DATABASE, limit=40):
    """Lore entries, by book, by story, or by what they say.

    Entries are listed WITHOUT their content — a book runs to thousands of
    characters an entry and the question is almost always which entry, not
    what it says. `lore_entry` returns one whole. Embeddings never come back:
    they are large and nobody reads them.

    `turn_added` is the column worth reading first and the reason this returns
    it beside the title. An entry with a turn number was written by the ENGINE
    during play; one without was authored. Live, that distinction was the whole
    diagnosis of a layout bug: entry 2734, `turn_added: 130`, titled "Shrine
    Interior — Main Hall and Upstairs Resting Area", was the engine's own
    invention sitting in the same book as twelve authored entries describing a
    different building, with nothing ranking one over the other.
    """
    where = []
    if _int(book_id) is not None:
        where.append(f"e.lorebook_id = {_int(book_id)}")
    if _int(chat_id) is not None:
        where.append(
            "e.lorebook_id IN ("
            f"SELECT lorebook_id FROM chat_lorebooks WHERE chat_id = {_int(chat_id)}"
            f" UNION SELECT lorebook_id FROM chats WHERE id = {_int(chat_id)})")
    if text:
        safe = str(text).replace("'", "''")
        where.append("(e.keys LIKE '%%%s%%' OR e.title LIKE '%%%s%%' "
                     "OR e.content LIKE '%%%s%%')" % (safe, safe, safe))
    if not where:
        return {"ok": False, "error": "give a book_id, a chat_id or text"}
    rows, result = _rows(
        "SELECT e.id AS id, e.lorebook_id AS book_id, e.category AS category, "
        "e.title AS title, e.keys AS keys, e.canon_locked AS canon_locked, "
        "e.importance AS importance, e.turn_added AS turn_added, "
        "LENGTH(e.content) AS content_chars, e.embedding_model AS embedded_by "
        f"FROM lore_entries e WHERE {' AND '.join(where)} "
        f"ORDER BY e.lorebook_id, e.id LIMIT {_int(limit, 40)}", database)
    if rows is None:
        return result
    return {"ok": True, "returned": len(rows), "entries": rows,
            "note": "turn_added set = written by the engine during play; "
                    "null = authored. canon_locked scores +0.1 in retrieval.",
            **({"cap": _note(result)} if _note(result) else {})}


def lore_entry(entry_id, database=DEFAULT_DATABASE):
    """One lore entry whole, content included."""
    eid = _int(entry_id)
    if eid is None:
        return {"ok": False, "error": "entry_id must be an integer"}
    rows, result = _rows(
        "SELECT e.id AS id, e.lorebook_id AS book_id, b.name AS book, "
        "e.category AS category, e.title AS title, e.keys AS keys, "
        "e.aliases AS aliases, e.canon_locked AS canon_locked, "
        "e.importance AS importance, e.turn_added AS turn_added, "
        "e.scope AS scope, e.relations AS relations, "
        "e.knowledge_tag AS knowledge_tag, e.content AS content "
        "FROM lore_entries e LEFT JOIN lorebooks b ON b.id = e.lorebook_id "
        f"WHERE e.id = {eid}", database, max_cell=12000)
    if rows is None:
        return result
    if not rows:
        return {"ok": False, "error": f"no lore entry {eid}"}
    return {"ok": True, "entry": rows[0],
            **({"cap": _note(result)} if _note(result) else {})}


def story_overview(chat_id, database=DEFAULT_DATABASE):
    """The shape of one story: how long, which characters, where it stalled.

    Cheap enough to run before any theory, and it answers the questions that
    otherwise get answered by assumption — whether the story is 6 turns or 200,
    whether it has been branched, and whether stale steps cluster anywhere.
    """
    cid = _int(chat_id)
    if cid is None:
        return {"ok": False, "error": "chat_id must be an integer"}
    head, _ = _rows("SELECT c.id AS chat_id, c.name AS name, "
                    "c.branched_from AS branched_from, c.scenario AS scenario, "
                    "COUNT(t.id) AS turns, MIN(t.idx) AS first_idx, "
                    "MAX(t.idx) AS last_idx FROM chats c "
                    "LEFT JOIN turns t ON t.chat_id = c.id "
                    f"WHERE c.id = {cid} GROUP BY c.id", database,
                    max_cell=1200)
    if not head:
        return {"ok": True, "chat_id": cid, "exists": False,
                "note": "no chat with that id — resolve the name with "
                        "find_story rather than guessing the integer"}
    cast, _ = _rows("SELECT ch.name AS name, cc.status AS status "
                    "FROM chat_chars cc JOIN characters ch ON ch.id = cc.char_id "
                    f"WHERE cc.chat_id = {cid} ORDER BY ch.name", database,
                    max_cell=200)
    stale, _ = _rows("SELECT t.idx AS idx, s.key AS step FROM turns t "
                     "JOIN steps s ON s.turn_id = t.id "
                     f"WHERE t.chat_id = {cid} AND s.stale = 1 "
                     "ORDER BY t.idx LIMIT 40", database, max_cell=200)
    mem, _ = _rows("SELECT COUNT(*) AS memories FROM memories "
                   f"WHERE chat_id = {cid}", database)
    # `branched_from` IS A JSON LIST, not an integer — '[57, 56]' for a branch
    # of a branch. Comparing it to an id matched nothing at all and reported
    # every story as unbranched, which is the reading that sends a person to
    # test a fix against a different chat from the one it was made in.
    branches, _ = _rows("SELECT id AS chat_id, name FROM chats c WHERE EXISTS "
                        "(SELECT 1 FROM json_each(c.branched_from) "
                        f"WHERE value = {cid}) ORDER BY id", database,
                        max_cell=200)
    return {"ok": True, **(head[0]), "exists": True,
            "cast": cast or [], "memories": _int(
                (mem or [{}])[0].get("memories"), 0),
            "stale_steps": stale or [],
            # `branched_from` is the whole ANCESTRY, not the parent — '[57, 56]'
            # for a branch of a branch — so this is every descendant at any
            # depth, which is what "am I about to fix this in the copy nobody
            # is reading" actually needs.
            "descendants": branches or [],
            "turn_numbering": "turns.idx, per chat, zero-based"}


def schema(table=None, database=DEFAULT_DATABASE):
    """The columns of the engine's tables, so a bespoke query can be written.

    The lanes above cover the questions asked so far. They will not cover the
    next defect class, and the answer to that is a documented schema rather
    than a new function per question — `refdb.query` is already there for
    anything these do not reach.
    """
    if table:
        name = str(table).replace("'", "''")
        rows, result = _rows(f"PRAGMA table_info('{name}')", database)
        if rows is None:
            return result
        if not rows:
            return {"ok": True, "table": table, "exists": False,
                    "note": "no such table — call schema() with no argument "
                            "for the list"}
        return {"ok": True, "table": table, "exists": True,
                "columns": [{"name": r.get("name"), "type": r.get("type")}
                            for r in rows]}
    rows, result = _rows("SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name",
                         database, max_rows=200)
    if rows is None:
        return result
    return {"ok": True, "tables": [r.get("name") for r in rows],
            "note": "the story lanes cover chats, turns, steps, variants and "
                    "memories. For anything else, read the columns with "
                    "schema('<table>') and query it with query_db."}


def summarise(result, limit=2400):
    """Render a lane result for a fetch step without losing what it said.

    JSON straight into the payload is what the other verbs do and it is right
    here too; this exists only for the places that want a line rather than a
    blob, and it keeps the caps and notes because those ARE the content when
    they fire.
    """
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(text)} chars total]"
