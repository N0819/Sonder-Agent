# memory.py — the assistant's long-term memory. Ported from Sonder Engine's
# memory.py with the fiction and the body removed; the retrieval and revision
# mechanics are kept exactly, because every constant in them was measured
# there, not guessed. Where a number appears below with a justification, the
# measurement behind it is the engine's (docs/MEMORY.md in the Sonder_Engine
# repo); nothing here re-derives them, and nothing here changed them.
#
# The shape of a memory:
#   kind        episodic / dialogue / inference / semantic / commitment
#   provenance  witnessed  — it happened in conversation, both of us there
#               told       — the user told me
#               read       — I read it on a page I fetched (evidence rows)
#               inferred   — I concluded it
#               remembered — my own act ("I chose to ...")
#   salience    how much it mattered when formed. Never revised.
#   importance  how central it BECAME. NULL = never revised, reads as salience.
#   confidence  how much I credit it NOW. Moves every turn for inferences.
#   disputed    my own later re-reading, recorded BESIDE the row, never over it.
#
# Provenance routes: three summary scopes (what happened / what I was told or
# read / what I concluded), because the engine found that one melted summary
# let an inference come back indistinguishable from something witnessed —
# belief laundered into knowledge inside one mind. Separate rows, not tags in
# prose, because a tag inside model-written prose is a convention the model
# can drop; a separate row cannot be.

import json
import re
import time
from collections import defaultdict

import numpy as np

from db import q, qi, transaction
from providers import embed_texts_meta

MEMORY_KINDS = ["episodic", "dialogue", "inference", "semantic", "commitment"]
MEMORY_PROVENANCE = ["witnessed", "told", "read", "inferred", "remembered"]

SCOPE_FIRSTHAND = "autobiographical"
SCOPE_RECEIVED = "received"
SCOPE_SURMISE = "surmise"

_PROVENANCE_SCOPE = {
    "witnessed": SCOPE_FIRSTHAND, "remembered": SCOPE_FIRSTHAND,
    "told": SCOPE_RECEIVED, "read": SCOPE_RECEIVED,
    "inferred": SCOPE_SURMISE,
}

_SCOPE_LABELS = {
    SCOPE_FIRSTHAND: "what_happened_between_us",
    SCOPE_RECEIVED: "what_i_was_told_or_read",
    SCOPE_SURMISE: "what_i_concluded",
}

# Epistemic-origin labels stamped on every projected row. The claim, not the
# container: a vivid episode of the user ASSERTING something is still received
# information, and a conclusion formed mid-conversation is still inferred.
_ORIGIN_LABELS = {
    "witnessed": "what_i_experienced", "remembered": "what_i_experienced",
    "told": "what_i_was_told", "read": "what_i_read",
    "inferred": "what_i_concluded",
}


def provenance_scope(provenance):
    # An unrecognised provenance falls to RECEIVED, not first-hand. Both maps
    # used to default to the most-trusted class, so a corrupt string, a
    # direct DB write, or a future provenance added to one map and forgotten
    # in the other would be classified as something the assistant personally
    # experienced — sources laundering into experience, which is the exact
    # direction the three scopes exist to prevent, arriving through a `.get`
    # default. `prepare_memory` gates the enum today, so this is a fail-safe
    # rather than a live bug; the point is which way it fails. "I was told
    # this" is the safest wrong answer available: it claims the least.
    return _PROVENANCE_SCOPE.get(str(provenance or ""), SCOPE_RECEIVED)


def scope_label(scope):
    return _SCOPE_LABELS.get(scope, scope)


# ---- Text plumbing ----

_STOPWORDS = {
    "about", "after", "again", "against", "because", "before", "being",
    "could", "does", "from", "have", "into", "itself", "might", "other",
    "should", "something", "their", "there", "these", "they", "this",
    "through", "under", "what", "when", "where", "which", "while", "with",
    "would", "your", "said", "says", "then", "that", "were", "been",
}

_OLD_CUES = (r"\blong ago\b", r"\bweeks? ago\b", r"\bmonths? ago\b",
             r"\bback then\b", r"\bearliest\b", r"\bfirst time\b",
             r"\boriginally\b")
_RECENT_CUES = (r"\brecently\b", r"\bjust now\b", r"\ba moment ago\b",
                r"\blast time\b", r"\bjust happened\b")


def _json_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _clamp(value, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo


def _blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()


def _vec(blob):
    return np.frombuffer(blob, dtype=np.float32) if blob else None


def _cos(a, b):
    # Plain dot product, NOT cosine-with-norms: both producers already
    # L2-normalise (embed_texts_meta and cheap_embed), so the two norm calls
    # would divide by 1.0 twice. The engine measured the shortcut at 4.4x with
    # scores agreeing to 8.7e-06; the precondition is every producer
    # normalising, which providers.py guarantees.
    if a is None or b is None or len(a) != len(b):
        return 0.0
    return float(np.dot(a, b))


def _gist(text, limit=240):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    out = ""
    for part in re.split(r"(?<=[.!?])\s+", text):
        candidate = (out + " " + part).strip()
        if len(candidate) > limit:
            break
        out = candidate
    return out or text[:limit].rsplit(" ", 1)[0]


def _extract_entities(text, limit=12):
    candidates = re.findall(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text or "")
    blocked = {"You", "The", "This", "That", "Then", "Your", "They",
               "Something", "What", "When", "Where"}
    out = []
    for c in candidates:
        c = c.strip()
        if c in blocked or c in out:
            continue
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _extract_key_phrases(text, entities=None, limit=12):
    text = str(text or "")
    phrases = []
    for quote in re.findall(r'["“](.{3,100}?)[”"]', text):
        quote = re.sub(r"\s+", " ", quote).strip()
        if quote and quote.lower() not in {p.lower() for p in phrases}:
            phrases.append(quote)
    words = re.findall(r"[A-Za-z0-9'-]{3,}", text.lower())
    counts = defaultdict(float)
    for i, w in enumerate(words):
        if w in _STOPWORDS:
            continue
        counts[w] += 1
        if i + 1 < len(words) and words[i + 1] not in _STOPWORDS:
            counts[f"{w} {words[i + 1]}"] += 1.5
    ranked = sorted(counts, key=lambda item: (-counts[item],
                                              -len(item.split()), item))
    for e in entities or []:
        if e.lower() not in {p.lower() for p in phrases}:
            phrases.append(e)
    for p in ranked:
        if p.lower() in {x.lower() for x in phrases}:
            continue
        phrases.append(p)
        if len(phrases) >= limit:
            break
    return phrases[:limit]


def _memory_document(data):
    """The text the full embedding is built from — a labelled block, so the
    vector carries structure (source, people, phrases) alongside the prose."""
    phrases = ", ".join(data.get("key_phrases") or [])
    entities = ", ".join(data.get("entities") or [])
    return "\n".join(p for p in (
        f"kind: {data.get('kind', 'episodic')}",
        f"turn: {data.get('turn_idx', '')}",
        f"people: {entities}",
        f"key phrases: {phrases}",
        f"gist: {data.get('gist', '')}",
        f"details: {data.get('content', '')}",
        f"source: {data.get('provenance', 'witnessed')}",
        f"url: {data.get('source_url', '')}",
    ) if not p.endswith(": "))


def _memory_cues(data):
    """Shorter, query-shaped text for the cue vector. It carries the highest
    weight of the four rankings precisely because it is built from the same
    short cue-like material a query is."""
    return "\n".join(p for p in (
        data.get("gist") or "",
        ", ".join(data.get("key_phrases") or []),
        ", ".join(data.get("entities") or []),
        data.get("kind") or "",
    ) if p)


def _replace_fts(memory_id, data):
    qi("DELETE FROM memory_retrieval_fts WHERE memory_id=?", (str(memory_id),))
    qi("INSERT INTO memory_retrieval_fts(memory_id,gist,content,key_phrases,"
       "entities) VALUES(?,?,?,?,?)",
       (str(memory_id), data.get("gist") or "", data.get("content") or "",
        ", ".join(data.get("key_phrases") or []),
        ", ".join(data.get("entities") or [])))


# ---- Importance and disputes ----
#
# How far one consequence moves a memory's importance, and the ceiling it
# climbs toward. Small and asymptotic: importance is evidence accumulating
# that a memory mattered, and one citation is evidence, not proof. Nothing
# here is driven by RETRIEVAL — a memory that gets recalled would then rank
# higher and get recalled more, a popularity loop wearing the word
# "importance". Only consequences move it (a belief built on it, a re-reading
# of it), which is also why access_count stays written and unread.
_IMPORTANCE_STEP = 0.12
_IMPORTANCE_CEILING = 0.97
# A memory the assistant has re-read moves further, because being wrong about
# something is a bigger fact about it than being cited once.
_IMPORTANCE_DISPUTE_STEP = 0.2
_MAX_DISPUTE_READING = 300


def _dispute_of(raw):
    """The stored re-reading, or None. Never raises on a malformed blob — a
    corrupt dispute must not make a memory unreadable."""
    if not raw:
        return None
    try:
        out = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, dict) and out.get("reading") else None


def effective_importance(mem):
    """How much this memory matters NOW: revised importance if it has one,
    else the salience it was minted with. The single place that fallback is
    decided, so no reader can rank on the raw column and see NULL for every
    row that was never revised."""
    value = mem["importance"] if not isinstance(mem, dict) \
        else mem.get("importance")
    if value is None:
        value = mem["salience"] if not isinstance(mem, dict) \
            else mem.get("salience")
    return _clamp(value)


def record_dispute(reading, turn_idx, *, memory_ref="", gist=""):
    """The assistant has re-read one of its own memories against new evidence.

    The row stays exactly as it was — "I read this" is still true; what is
    recorded beside it is that it is no longer read the way it first was.
    That is what a retracted source, a superseded benchmark, or a user
    correction actually do to a memory: they do not delete the experience of
    having read it, they change what it means. Collapsing the two would
    either erase the record or hide the correction.

    Stored as a column on the row, not an edge to the superseding memory —
    row ids do not survive delete-and-reinsert restore paths; the engine
    learned that the id-keyed version shreds on the first rollback.
    """
    needle = " ".join(str(gist or "").split()).casefold()
    reading = " ".join(str(reading or "").split())[:_MAX_DISPUTE_READING]
    memory_ref = str(memory_ref or "").strip()
    if not (needle or memory_ref) or not reading:
        return []
    rows = q("SELECT id, event_key, gist, content, disputed, salience, "
             "importance FROM memories")
    hits = ([r for r in rows if str(r["event_key"] or "") == memory_ref]
            if memory_ref else [])
    if not hits and needle:
        hits = [r for r in rows
                if " ".join((r["gist"] or "").split()).casefold() == needle]
    if not hits and needle:
        hits = [r for r in rows
                if needle in " ".join((r["gist"] or "").split()).casefold()
                or needle in " ".join((r["content"] or "").split()).casefold()]
    updated = []
    for row in hits:
        prior = _dispute_of(row["disputed"]) or {}
        blob = json.dumps({
            "turn_idx": turn_idx,
            "reading": reading,
            # Re-read twice means genuinely unstable, and that is worth
            # being able to see.
            "count": int(prior.get("count") or 0) + 1,
        }, ensure_ascii=False)
        # A dispute moves importance UP: a memory whose meaning changed is
        # more central to this mind, not less.
        base = effective_importance(row)
        raised = min(_IMPORTANCE_CEILING, base + _IMPORTANCE_DISPUTE_STEP)
        qi("UPDATE memories SET disputed=?, importance=? WHERE id=?",
           (blob, raised, row["id"]))
        updated.append(row["id"])
    return updated


def raise_importance(event_keys=(), *, only_unrevised=False,
                     step=_IMPORTANCE_STEP):
    """Nudge memories toward the ceiling because something happened that they
    turned out to matter for. Asymptotic — each consequence closes a fraction
    of the remaining distance — so repetition cannot run away. Never lowers,
    never touches salience.

    `only_unrevised` lifts a row exactly once, ever. The citation signal that
    feeds this is itself downstream of retrieval, so the popularity loop is
    closed structurally rather than hoped away."""
    keys = [str(k) for k in (event_keys or []) if str(k or "").strip()]
    if not keys:
        return 0
    clause = "event_key IN (%s)" % ",".join("?" for _ in keys)
    if only_unrevised:
        clause += " AND importance IS NULL"
    rows = q(f"SELECT id, salience, importance FROM memories WHERE {clause}",
             tuple(keys))
    changed = 0
    for row in rows:
        base = effective_importance(row)
        raised = min(_IMPORTANCE_CEILING, base + step * (1.0 - base))
        if raised - base > 1e-6:
            qi("UPDATE memories SET importance=? WHERE id=?",
               (raised, row["id"]))
            changed += 1
    return changed


# ---- Minting ----

def _row_memory(row):
    return {
        "id": row["id"], "session_id": row["session_id"],
        "turn_id": row["turn_id"], "turn_idx": row["turn_idx"],
        "kind": row["kind"], "provenance": row["provenance"],
        "salience": row["salience"], "content": row["content"],
        "gist": row["gist"] or _gist(row["content"]),
        "key_phrases": _json_list(row["key_phrases"]),
        "entities": _json_list(row["entities"]),
        "source_url": row["source_url"] or "",
        "confidence": row["confidence"] or 0.0,
        "importance": (row["salience"] if row["importance"] is None
                       else row["importance"]),
        "importance_revised": row["importance"] is not None,
        "disputed": _dispute_of(row["disputed"]),
        "archived": bool(row["archived"]),
        "event_key": row["event_key"] or "",
        "embedding_model": row["embedding_model"] or "",
        "embedding_dim": row["embedding_dim"],
    }


def prepare_memory(kind, provenance, salience, content, *, session_id=None,
                   turn_id=None, turn_idx=None, gist=None, key_phrases=None,
                   entities=None, source_url="", confidence=1.0,
                   event_key="", importance=None):
    content = re.sub(r"\s+", " ", str(content or "")).strip()
    entities = list(dict.fromkeys(
        entities if entities is not None else _extract_entities(content)))
    key_phrases = list(dict.fromkeys(
        key_phrases if key_phrases is not None
        else _extract_key_phrases(content, entities)))
    return {
        "session_id": session_id, "turn_id": turn_id, "turn_idx": turn_idx,
        "kind": kind if kind in MEMORY_KINDS else "episodic",
        "provenance": (provenance if provenance in MEMORY_PROVENANCE
                       else "witnessed"),
        "salience": _clamp(salience), "content": content,
        "gist": (gist or _gist(content)).strip(),
        "key_phrases": key_phrases[:16], "entities": entities[:16],
        "source_url": str(source_url or "").strip(),
        "confidence": _clamp(confidence),
        "event_key": str(event_key or "").strip(),
        # None, not 0.0: NULL is "never revised" and reads as the salience.
        # Defaulting to a number would freeze every new memory at mint value
        # and silently kill the fallback.
        "importance": None if importance is None else _clamp(importance),
    }


def prepare_memories_batch(memories):
    """Normalize and embed a batch WITHOUT touching the database. The commit
    path calls this before opening its write transaction, so a remote
    embedding round trip can never hold SQLite's writer."""
    prepared = [prepare_memory(**item) for item in memories]
    if not prepared:
        return {"prepared": [], "embedded": None}
    texts = []
    for data in prepared:
        texts.extend([_memory_document(data),
                      _memory_cues(data) or _memory_document(data)])
    return {"prepared": prepared, "embedded": embed_texts_meta(texts)}


def _upsert_memory(data, full_vec, cue_vec, embedded):
    existing = None
    if data["event_key"]:
        existing = q("SELECT id FROM memories WHERE event_key=?",
                     (data["event_key"],), one=True)
    values = (
        data["session_id"], data["turn_id"], data["turn_idx"], data["kind"],
        data["provenance"], data["salience"], data["content"], data["gist"],
        json.dumps(data["key_phrases"], ensure_ascii=False),
        json.dumps(data["entities"], ensure_ascii=False),
        data["source_url"], data["confidence"],
        _blob(full_vec), _blob(cue_vec),
        embedded.model_key, embedded.dimensions, data.get("importance"),
    )
    if existing:
        mid = existing["id"]
        qi("""UPDATE memories SET session_id=?,turn_id=?,turn_idx=?,kind=?,
            provenance=?,salience=?,content=?,gist=?,key_phrases=?,entities=?,
            source_url=?,confidence=?,embedding=?,cue_embedding=?,
            embedding_model=?,embedding_dim=?,importance=?,archived=0
            WHERE id=?""", values + (mid,))
    else:
        mid = qi("""INSERT INTO memories(session_id,turn_id,turn_idx,kind,
            provenance,salience,content,gist,key_phrases,entities,source_url,
            confidence,embedding,cue_embedding,embedding_model,embedding_dim,
            importance,event_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 values + (data["event_key"],))
    _replace_fts(mid, data)
    return mid


def add_memories_batch(memories=None, *, prepared_batch=None):
    if prepared_batch is None:
        prepared_batch = prepare_memories_batch(memories or [])
    prepared = prepared_batch.get("prepared") or []
    embedded = prepared_batch.get("embedded")
    if not prepared:
        return []
    if embedded is None or len(embedded.vectors) != len(prepared) * 2:
        raise ValueError("invalid prepared memory embedding batch")
    ids = []
    with transaction():
        for i, data in enumerate(prepared):
            ids.append(_upsert_memory(data, embedded.vectors[i * 2],
                                      embedded.vectors[i * 2 + 1], embedded))
    return ids


def add_memory(kind, provenance, salience, content, **kw):
    return add_memories_batch([dict(kind=kind, provenance=provenance,
                                    salience=salience, content=content,
                                    **kw)])[0]


# ---- Retirement: setting a memory aside without pretending it never was ----
#
# THE NEED IS REAL AND IT IS NOT ABOUT TRUTH. A long-memory assistant working
# on a project accumulates superseded context — the Postgres schema you spent
# three sessions on before moving to SQLite, the API shape from before the
# rewrite. None of it is FALSE. All of it is noise, and it competes for recall
# against the things that are current, which is a direct tax on every answer.
# `archived` does not help: an archived row has only left the rolling
# consolidation window and is still fully recallable, deliberately.
#
# SO WHY NOT DELETE. Because everything else in this system exists to stop a
# model rewriting the record by fiat. Contradiction becomes a dispute rather
# than an overwrite; a re-read memory keeps its content and carries the new
# reading beside it; reconciliation moves confidence on evidence and never on
# a verdict about truth. A DELETE reachable from model output would be the one
# path that erases rather than annotates — and it would be irreversible, which
# is the property that makes the mistake unrecoverable rather than merely
# wrong.
#
# And the judgement here is genuinely fallible in a specific way: "irrelevant
# to the current iteration" is a claim about SCOPE, and scope changes. The
# Postgres decision becomes relevant again the moment somebody asks why you
# moved. A relevance call that turns out wrong should cost a restore, not the
# information.
#
# So: retirement is immediate, total (retired rows leave recall entirely),
# reversible, batched, and it records WHY. Hard deletion exists — `purge` —
# and is reachable only from the host, because a human deciding to destroy
# their own records is a different act from a model deciding it.
#
# WHAT CANNOT BE RETIRED, and why each one:
#   commitment rows  — an open promise must nag, not fade. DESIGN.md argues
#                      this at length; a commitment the assistant can retire
#                      when it feels stale is not a commitment.
#   disputed rows    — the dispute IS the record that something was unstable.
#                      Retiring it hides the instability rather than the noise.
#   retirement notes — otherwise the assistant can forget that it forgot,
#                      which is strictly worse than forgetting.

_RETIRE_PROTECTED_KINDS = ("commitment",)
_RETIREMENT_NOTE_PREFIX = "retired:"


# Every embedded table spells "this vector cannot be compared with the current
# model" the same way, and the parentheses are part of the spelling: without
# them, ANDing another condition onto it silently widens the OR chain. Folded
# into one constant rather than retyped at each site — the identity lesson.
_STRANDED_SQL = ("(embedding_model<>? OR embedding_dim<>? "
                 "OR embedding IS NULL)")
# A summary window with no prose is never retrievable (search_memory_summaries
# skips it) and embedding an empty string is a provider error waiting to
# happen. Excluded from BOTH the work list and the remaining count, because a
# row counted as outstanding but never processed makes "run again to continue"
# an instruction that never terminates.
_SUMMARY_REBUILDABLE_SQL = _STRANDED_SQL + " AND TRIM(summary)<>''"


def _reembed_in_batches(rows, texts_of, write, batch_size):
    """Embed in batches, commit each batch, stop dead on a provider failure.

    Shared by both embedded tables so the mid-run guard is written once. A
    provider that fails part-way must not overwrite good vectors with
    hashing-trick ones — that turns a stalled migration into a corrupted bank —
    and a guard each caller has to remember to repeat is one that eventually
    gets forgotten.

    `texts_of(row)` returns that row's texts; `write(row, vectors, embedded)`
    stores them. Returns `(rebuilt, failed)`."""
    rebuilt = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        per_row = [texts_of(row) for row in chunk]
        texts = [text for group in per_row for text in group]
        embedded = embed_texts_meta(texts)
        if embedded.fallback or len(embedded.vectors) != len(texts):
            return rebuilt, len(chunk)
        with transaction():
            cursor = 0
            for row, group in zip(chunk, per_row):
                write(row, embedded.vectors[cursor:cursor + len(group)],
                      embedded)
                cursor += len(group)
                rebuilt += 1
    return rebuilt, 0


def _memory_rebuild_texts(row):
    """The document and cue text for a stored row.

    Every field `_memory_document` reads is passed, not just the prose ones.
    Reconstructing the dict from content/gist/phrases/entities alone let the
    document template fall back to its defaults — `kind: episodic`,
    `source: witnessed`, an empty turn and url — so a rebuilt vector encoded
    text the original never had, and a rebuild quietly rewrote the meaning of
    every non-episodic row it touched."""
    data = {
        "kind": row["kind"], "provenance": row["provenance"],
        "turn_idx": row["turn_idx"], "content": row["content"],
        "gist": row["gist"], "source_url": row["source_url"],
        "key_phrases": _json_list(row["key_phrases"]),
        "entities": _json_list(row["entities"]),
    }
    return [_memory_document(data), _memory_cues(data) or _memory_document(data)]


def rebuild_embeddings(batch_size=64, limit=None):
    """Re-embed rows whose vectors were written by a different model.

    DESIGN.md listed this as unbuilt while a provider change was hypothetical.
    It stops being hypothetical the moment somebody switches embedding
    providers, because a vector can only be compared with one from the same
    model: every existing row scores 0.0 against the new query embedding,
    forever, and retrieval keeps working on keyword match alone while looking
    perfectly healthy. Without a rebuild, the choice on the settings page is
    "keep your bank or upgrade your embeddings", which is not a choice anyone
    should be asked to make.

    Deliberately incremental and re-runnable. It embeds in batches, commits
    each batch, and selects by "stamp differs from current" — so an
    interrupted run leaves a partially-migrated bank that the next run simply
    continues, and a completed run is a no-op. Nothing is deleted: a row is
    updated in place, so a failure costs time and never data.

    Covers BOTH embedded tables. `memory_summaries` carries its own vector and
    `search_memory_summaries` skips a cross-model window outright rather than
    comparing it, so migrating only `memories` left every consolidated window
    unreachable — and reported the bank as fully comparable while it was. That
    is the same silent-success this function exists to prevent, one table over.

    Returns a progress dict; `rebuilt` and `remaining` span both tables so the
    "run again" prompt is never wrong about what is left."""
    probe = embed_texts_meta(["probe"])
    # `fallback` is TRUE only when a CONFIGURED provider failed; when none is
    # configured at all the batch comes back cheap with fallback False. Both
    # cases must refuse, and testing only the flag missed the likelier one —
    # a rebuild with nothing configured would have overwritten real vectors
    # with hashing-trick ones and called it a migration. Test the stamp, which
    # is the thing that actually decides comparability.
    if probe.fallback or str(probe.model_key).startswith("cheap:"):
        return {"ok": False, "rebuilt": 0, "remaining": None,
                "error": "no embeddings provider is reachable: "
                         + (probe.error
                            or "none is configured, and rebuilding onto the "
                               "local hashing fallback would destroy real "
                               "vectors rather than migrate them")}
    target_model, target_dim = probe.model_key, probe.dimensions
    stamp = (target_model, target_dim)
    rows = q("SELECT id, kind, provenance, turn_idx, content, gist, "
             "key_phrases, entities, source_url FROM memories WHERE "
             + _STRANDED_SQL, stamp)
    summaries = q("SELECT id, summary, key_phrases, unresolved_threads FROM "
                  "memory_summaries WHERE " + _SUMMARY_REBUILDABLE_SQL, stamp)
    total = len(rows) + len(summaries)
    if limit is not None:
        # One budget across both tables, memories first: they are the bank the
        # user is actually protecting, and a windowed summary is rebuilt from
        # rows that had better be comparable already.
        rows = rows[:max(0, limit)]
        summaries = summaries[:max(0, limit - len(rows))]

    def write_memory(row, vectors, embedded):
        qi("UPDATE memories SET embedding=?, cue_embedding=?, "
           "embedding_model=?, embedding_dim=? WHERE id=?",
           (_blob(vectors[0]), _blob(vectors[1]),
            embedded.model_key, embedded.dimensions, row["id"]))

    def write_summary(row, vectors, embedded):
        qi("UPDATE memory_summaries SET embedding=?, embedding_model=?, "
           "embedding_dim=? WHERE id=?",
           (_blob(vectors[0]), embedded.model_key, embedded.dimensions,
            row["id"]))

    rebuilt, failed = _reembed_in_batches(rows, _memory_rebuild_texts,
                                          write_memory, batch_size)
    summaries_rebuilt = 0
    # Only continue to summaries if memories came through clean: a failure
    # here means the provider is down, and the next call resumes from the
    # stamp anyway.
    if not failed:
        summaries_rebuilt, failed = _reembed_in_batches(
            summaries,
            lambda r: [_summary_retrieval_text(
                r["summary"], _json_list(r["key_phrases"]),
                _json_list(r["unresolved_threads"]))],
            write_summary, batch_size)
    remaining = (
        q("SELECT COUNT(*) AS c FROM memories WHERE " + _STRANDED_SQL,
          stamp, one=True)["c"]
        + q("SELECT COUNT(*) AS c FROM memory_summaries WHERE "
            + _SUMMARY_REBUILDABLE_SQL, stamp, one=True)["c"])
    return {"ok": failed == 0,
            "rebuilt": rebuilt + summaries_rebuilt, "remaining": remaining,
            "memories_rebuilt": rebuilt, "summaries_rebuilt": summaries_rebuilt,
            "total_needing_rebuild": total, "model": target_model,
            "dimensions": target_dim,
            **({"error": "the embeddings provider failed part-way; nothing "
                         "was overwritten with fallback vectors — re-run to "
                         "continue"} if failed else {})}


def retire_memories(event_keys, *, reason, turn_idx, batch=None):
    """Set rows aside as no longer relevant. Returns a result dict.

    Refuses without a reason: a row set aside for no stated cause cannot be
    reviewed later, and "why is this gone" is the question a restore has to
    answer."""
    reason = " ".join(str(reason or "").split())[:300]
    if not reason:
        return {"ok": False, "error": "retiring memories needs a reason",
                "retired": [], "refused": []}
    keys = [str(k).strip() for k in (event_keys or []) if str(k or "").strip()]
    if not keys:
        return {"ok": False, "error": "no memories named", "retired": [],
                "refused": []}
    batch = batch or f"turn:{turn_idx}"
    marks = ",".join("?" for _ in keys)
    rows = q(f"SELECT id, event_key, kind, disputed, retired, content "
             f"FROM memories WHERE event_key IN ({marks})", tuple(keys))
    found = {r["event_key"] for r in rows}
    retired, refused = [], []
    for key in keys:
        if key not in found:
            refused.append({"ref": key, "why": "no such memory"})
    with transaction():
        for row in rows:
            if row["retired"]:
                refused.append({"ref": row["event_key"],
                                "why": "already retired"})
                continue
            if row["kind"] in _RETIRE_PROTECTED_KINDS:
                refused.append({"ref": row["event_key"],
                                "why": "a commitment must nag, not fade"})
                continue
            if row["disputed"]:
                refused.append({
                    "ref": row["event_key"],
                    "why": "this row carries a dispute — the record that it "
                           "was unstable is the part worth keeping"})
                continue
            if str(row["content"] or "").startswith(_RETIREMENT_NOTE_PREFIX):
                refused.append({"ref": row["event_key"],
                                "why": "this is the note recording an earlier "
                                       "retirement; forgetting that you "
                                       "forgot is worse than forgetting"})
                continue
            qi("UPDATE memories SET retired=? WHERE id=?",
               (json.dumps({"turn_idx": turn_idx, "reason": reason,
                            "batch": batch}, ensure_ascii=False), row["id"]))
            retired.append(row["event_key"])
    return {"ok": bool(retired), "retired": retired, "refused": refused,
            "batch": batch, "reason": reason}


def restore_memories(event_keys=(), *, batch=""):
    """Bring rows back. By ref, or a whole batch at once — a retirement is
    usually one judgement about one topic, and undoing it one row at a time
    would be a worse experience than the mistake."""
    if batch:
        rows = q("SELECT id, event_key, retired FROM memories "
                 "WHERE retired<>''")
        ids = [r["id"] for r in rows
               if _retirement_of(r["retired"]).get("batch") == batch]
    else:
        keys = [str(k).strip() for k in event_keys or [] if str(k or "")]
        if not keys:
            return {"ok": False, "restored": 0}
        marks = ",".join("?" for _ in keys)
        ids = [r["id"] for r in
               q(f"SELECT id FROM memories WHERE event_key IN ({marks}) "
                 f"AND retired<>''", tuple(keys))]
    if not ids:
        return {"ok": False, "restored": 0}
    with transaction():
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            marks = ",".join("?" for _ in chunk)
            qi(f"UPDATE memories SET retired='' WHERE id IN ({marks})",
               tuple(chunk))
    return {"ok": True, "restored": len(ids)}


def _retirement_of(raw):
    try:
        out = json.loads(raw or "")
        return out if isinstance(out, dict) else {}
    except (TypeError, ValueError):
        return {}


def retired_rows(limit=200):
    """What has been set aside, newest first, grouped by the batch that did
    it. The host panel's view — a human should be able to see what the
    assistant decided to stop remembering."""
    rows = q("SELECT id, event_key, kind, provenance, gist, content, "
             "turn_idx, retired FROM memories WHERE retired<>'' "
             "ORDER BY id DESC LIMIT ?", (limit,))
    out = []
    for row in rows:
        mark = _retirement_of(row["retired"])
        out.append({"event_key": row["event_key"], "kind": row["kind"],
                    "provenance": row["provenance"],
                    "gist": row["gist"] or (row["content"] or "")[:200],
                    "turn_idx": row["turn_idx"],
                    "reason": mark.get("reason", ""),
                    "batch": mark.get("batch", ""),
                    "retired_at_turn": mark.get("turn_idx")})
    return out


def purge_retired(*, batch="", older_than_turns=None, current_turn_idx=None):
    """HARD DELETE of retired rows. Host-reachable only, deliberately.

    Nothing the model emits reaches this function. Retiring is a relevance
    judgement and is reversible; destroying the record is a different act,
    and it belongs to the person whose records they are.

    Rows cited by a summary's support set are skipped even here: those refs
    are the audit trail for a clause the assistant will keep asserting, and
    breaking it would leave a summary claiming support it can no longer
    show."""
    rows = q("SELECT id, event_key, retired FROM memories WHERE retired<>''")
    supported = set()
    for summary in q("SELECT support FROM memory_summaries"):
        for clause in _json_list(summary["support"]):
            if isinstance(clause, dict):
                supported.update(clause.get("support_refs") or [])
    doomed, kept = [], 0
    for row in rows:
        mark = _retirement_of(row["retired"])
        if batch and mark.get("batch") != batch:
            continue
        if older_than_turns is not None and current_turn_idx is not None:
            if (current_turn_idx - int(mark.get("turn_idx") or 0)
                    < older_than_turns):
                continue
        if row["event_key"] in supported:
            kept += 1
            continue
        doomed.append(row["id"])
    with transaction():
        for start in range(0, len(doomed), 400):
            chunk = doomed[start:start + 400]
            marks = ",".join("?" for _ in chunk)
            for mid in chunk:
                qi("DELETE FROM memory_retrieval_fts WHERE memory_id=?",
                   (str(mid),))
            qi(f"DELETE FROM memories WHERE id IN ({marks})", tuple(chunk))
    return {"ok": True, "purged": len(doomed),
            "kept_because_cited_by_a_summary": kept}


def delete_turn_memories(turn_id):
    """A re-run replaces rather than duplicates: the turn's rows go first,
    then the batch re-inserts (and event_key upserts catch anything shared)."""
    for r in q("SELECT id FROM memories WHERE turn_id=?", (turn_id,)):
        qi("DELETE FROM memory_retrieval_fts WHERE memory_id=?",
           (str(r["id"]),))
    qi("DELETE FROM memories WHERE turn_id=?", (turn_id,))


# ---- The one seam memory is read through --------------------------------
#
# One rule decides what the assistant may retrieve while deciding a turn, and
# it runs BEFORE any ranking: the turn cutoff. A mind deciding turn N must
# not read a memory of how turn N turned out. With no rerolls this looks
# hypothetical — it is not, because a regenerate/edit feature is one commit
# away, and the engine's audit found exactly this hole after assuming the
# same thing.
#
# The invariant-bearing arguments are REQUIRED and have no defaults, so a
# caller cannot omit one; it can only state it, including stating None (the
# host memory panel, where nobody is deciding a turn). Forgetting the rule is
# a TypeError instead of a leak. The engine's earlier design wrote the
# filters out at every call site and documented that repetition as the
# safety; that reasoning was backwards — repetition is exactly how a sixth
# call site forgets.

def visible_memory_rows(*, before_turn_idx, include_archived,
                        include_retired=False, since_turn_idx=None,
                        require_turn_idx=False):
    clauses, args = ["1=1"], []
    if not include_archived:
        clauses.append("archived=0")
    if not include_retired:
        # Retirement is the one filter here with a DEFAULT, and the default is
        # the safe direction. Everything else in this signature is defaultless
        # because forgetting it leaks; forgetting this one only means a
        # retired row stays hidden, which is what the assistant asked for.
        # The host panel states include_retired=True explicitly, because
        # inspecting what was set aside is exactly what a human needs.
        clauses.append("retired=''")
    if require_turn_idx:
        clauses.append("turn_idx IS NOT NULL")
    if since_turn_idx is not None:
        clauses.append("turn_idx>=?")
        args.append(since_turn_idx)
    if before_turn_idx is not None:
        # NULL turn_idx rows are kept explicitly: imported/authored rows
        # belong to no turn, so they cannot be this turn's leaked outcome —
        # and SQL's three-valued logic would silently drop them from a bare
        # `turn_idx < ?`.
        clauses.append("(turn_idx IS NULL OR turn_idx<?)")
        args.append(before_turn_idx)
    return q("SELECT * FROM memories WHERE " + " AND ".join(clauses),
             tuple(args))


# ---- Retrieval ----
#
# Four rankings fused with Reciprocal Rank Fusion, plus one ranking per
# "aspect" (a short facet of what the assistant brings to the turn — the open
# hypothesis, the unresolved threads). Aspects get their OWN rank lists
# because concatenating them onto a long query was measured doing nothing:
# a 1,015-char query with a 10–60-char fragment appended sits at cosine 0.994
# to the query alone. A short facet cannot compete for influence inside a
# long string; given its own list it does not have to.

# RRF output is arbitrary in magnitude (~0.02 at rank 1) and only its ORDER
# carries meaning; the bonuses below are hand-tuned on a 0..1 utility scale.
# Summed raw, the four rankings max out at 0.074 combined while one bonus
# alone reaches 0.12 — so an irrelevant but recent/salient memory outranked
# the best match on every relevance signal. 12 puts the rankings at ~0.9
# against a ~0.4 bonus band: relevance leads, bonuses break ties.
_RRF_SCALE = 12.0

# RECALL IS BOUNDED BY RELEVANCE, NOT BY AN ATTENTION BUDGET.
#
# The engine's number was 16, and its justification was a fact about MINDS:
# paraphrase recall went 7/16 → 11/16 → 13/16 across k=8/16/24 while mean
# relevance flattened, so 16 was where the curve stopped paying. But the reason
# it stopped there rather than at 40 was that a character has to be a person,
# and a person recalling forty things at once is not a person. "The attention
# budget is real" is true of a fictional mind and false of this.
#
# An assistant has the opposite obligation: if a memory is relevant, forgetting
# it is a failure, and a fixed k drops the seventeenth relevant memory for a
# reason that has nothing to do with the question asked. So the cut is made on
# RELEVANCE and not on count -- take memories while they are still earning
# their place, and stop when the fused score falls off a cliff rather than when
# a counter runs out.
#
# `RECALL_LIMIT` survives as a CEILING, not a target: a payload has to end
# somewhere, and an unbounded recall on a bank of a hundred thousand rows is a
# different failure. It is set where cost, not cognition, argues for it.
RECALL_LIMIT = 64

# Where the cliff is. A memory stays if it scores at least this fraction of the
# best match; below that it is being carried by tie-break bonuses (recency,
# salience) rather than by relevance to the question, and shipping it costs
# tokens and dilutes what is around it.
#
# 0.45 rather than something tighter because the fused score is a rank-fusion
# number, not a probability: the gap between rank 1 and rank 20 on a good query
# is genuinely narrow, and a tight floor would throw away most of a well-matched
# set. Tuned to keep an ordinary question in the 12-25 range and let a question
# that genuinely touches fifty memories return fifty.
_RELEVANCE_FLOOR_RATIO = 0.45

# Never fewer than this while candidates remain, whatever the ratio says. A
# query whose best match is weak produces a flat score curve where everything
# is "within 45% of the best" or nothing is; this stops a bad query from
# returning two rows and reading as an empty memory.
_RECALL_FLOOR = 6

# THE CEILING THAT EXISTS FOR COST WAS DENOMINATED IN THE WRONG UNIT.
# `RECALL_LIMIT` is a COUNT, and its own comment says it "is set where cost,
# not cognition, argues for it" — but cost is bytes. Memories in this bank run
# from a couple of hundred characters to 13,407, so 64 rows is anywhere from
# 8k to 500k characters and the ceiling bounds nothing that is actually paid
# for.
#
# Measured on turn 79, the turn this was found by: the memory block was
# 236,870 characters — 92% of the payload and 61% of every byte of memory
# content in the bank — inside a turn totalling ~282,000 tokens across four
# rounds. The model returned unparseable output, which is what a payload that
# size buys. Count was NOT the culprit: 56 rows (the median) at the 1,320-char
# mean is 74k, so relevance ranking is preferentially selecting the LARGEST
# rows, and a count cap cannot see that.
#
# Applied after ranking, never during it: which memories are relevant is a
# question for the ranker, and this only decides how many of its answers fit.
# Rows are kept in the order the ranker chose, so the cut always falls at the
# least relevant end.
RECALL_CHAR_BUDGET = 24000

_SUMMARY_RECALL_LIMIT = 2
_ASPECT_WEIGHT = 0.55


def _fit_recall_budget(rows, budget=RECALL_CHAR_BUDGET):
    """Keep recalled rows, in rank order, until the character budget is spent.

    Returns `(kept, spent, dropped)`. The first row is always kept whatever it
    costs: a single memory larger than the whole budget is a mint-side problem
    and returning nothing would hide it behind an empty recall.

    NEVER re-orders and never re-scores. The ranker decided what is relevant;
    this decides only how much of its answer fits in a payload that has to be
    read by a model. A cut here is reported, not silent — `recall.budget` in
    retrieval health carries what was dropped, because a recall that quietly
    shrank looks exactly like a bank that has nothing to say."""
    kept, spent = [], 0
    for row in rows:
        size = len(str(row.get("content") or ""))
        if kept and spent + size > budget:
            continue
        kept.append(row)
        spent += size
    return kept, spent, len(rows) - len(kept)


def _fts_query(text):
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9'-]{3,}", text or "")
              if t.lower() not in _STOPWORDS]
    tokens = list(dict.fromkeys(tokens))[:16]
    if not tokens:
        return None
    return " OR ".join(
        f'"{t.replace(chr(34), chr(34) + chr(34))}"' for t in tokens)


def _lexical_ranking(query_text, visible_ids=None, limit=60):
    """BM25 over the FTS index, FILTERED to rows the seam made visible.

    The FTS table has no idea what the turn cutoff is, and this lane used to
    take the global top-`limit` and hand it to RRF. Rows the seam had already
    excluded — future-turn rows, i.e. exactly the replayed-turn outcomes the
    cutoff exists to hide — occupied rank slots, diluting every visible row's
    contribution and, past `limit` matches, evicting visible rows from the
    lane outright. No outcome ever leaked (the fused sort only iterates
    visible rows), but "the cutoff runs BEFORE any ranking" was false here,
    and under cheap-embed — every fresh install — this is the primary honest
    lane. Over-fetch, then keep the first `limit` that survive the seam."""
    fq = _fts_query(query_text)
    if not fq:
        return []
    try:
        rows = q("""SELECT CAST(memory_id AS INTEGER) AS mid,
            bm25(memory_retrieval_fts) AS score FROM memory_retrieval_fts
            WHERE memory_retrieval_fts MATCH ? ORDER BY score LIMIT ?""",
                 (fq, limit if visible_ids is None else limit * 8))
    except Exception:
        return []
    ids = [r["mid"] for r in rows]
    if visible_ids is not None:
        ids = [mid for mid in ids if mid in visible_ids]
    return ids[:limit]


def _temporal_mode(query_text):
    text = (query_text or "").lower()
    if any(re.search(p, text) for p in _OLD_CUES):
        return "old"
    if any(re.search(p, text) for p in _RECENT_CUES):
        return "recent"
    return "neutral"


def _exact_cue_score(memory, query_text):
    ql = (query_text or "").lower()
    if not ql:
        return 0.0
    score = 0.0
    for phrase in memory.get("key_phrases") or []:
        pl = phrase.lower().strip()
        if pl and pl in ql:
            score = max(score, 1.0)
        elif pl and ql in pl and len(ql) >= 4:
            score = max(score, 0.8)
    for entity in memory.get("entities") or []:
        if entity.lower() in ql:
            score = max(score, 0.7)
    return score


def _jaccard_text(a, b):
    la = set(re.findall(r"[a-z0-9']{3,}", (a or "").lower()))
    lb = set(re.findall(r"[a-z0-9']{3,}", (b or "").lower()))
    if not la or not lb:
        return 0.0
    return len(la & lb) / len(la | lb)


def _memory_similarity(a, b):
    av, bv = a.get("_vector"), b.get("_vector")
    if av is not None and bv is not None and len(av) == len(bv):
        return max(0.0, _cos(av, bv))
    return _jaccard_text(f"{a.get('gist', '')} {a.get('content', '')}",
                         f"{b.get('gist', '')} {b.get('content', '')}")


def _rrf_add(scores, reasons, ranking, weight, reason):
    for rank, mid in enumerate(ranking, 1):
        scores[mid] += (weight * _RRF_SCALE) / (60.0 + rank)
        if rank <= 12 and reason not in reasons[mid]:
            reasons[mid].append(reason)


def _rank_normalized_importance(memories):
    """effective_importance, respaced across the rows THIS search can see —
    inside their own p10–p90. Ordering preserved exactly; only the gaps move.

    The engine replayed 270 real recalls to pick this shape, and both obvious
    alternatives lost: deleting the term moved 35.2% of top-16 membership (so
    it was never decoration), and normalising to [0,1] moved 59.6% — a 3.7x
    weight increase wearing the word "normalisation", because values living
    in a 0.27-wide band gain influence without reordering anything. Respacing
    inside the bank's own range moved 15.2%. The defect actually fixed: how
    much discrimination this term has no longer depends on how the minting
    process happened to spread its numbers — a bank minted at 0.70 ± 0.03 and
    one spanning 0.4–0.9 behave alike.

    Callers asking an ABSOLUTE question (archiving: "did this ever matter")
    keep reading effective_importance directly; that is why the respacing
    lives here and not there. Ties share a rank so a spreadless bank stays
    flat instead of being handed an ordering by row id."""
    values = [(effective_importance(mem), mid)
              for mid, mem in memories.items()]
    if len(values) < 2:
        return {mid: v for v, mid in values}
    values.sort()
    ordered = [v for v, _mid in values]
    lo = ordered[int(len(ordered) * 0.10)]
    hi = ordered[int(len(ordered) * 0.90)]
    if hi - lo <= 1e-9:
        return {mid: v for v, mid in values}
    out = {}
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1][0] == values[i][0]:
            j += 1
        pct = ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            out[values[k][1]] = lo + pct * (hi - lo)
        i = j + 1
    return out


def search_memories(query, k=RECALL_LIMIT, *, current_turn_idx,
                    include_archived=True, chronological=True, aspects=None,
                    embedded=None, count_access=True):
    """Hybrid retrieval. `current_turn_idx` is required — the turn cutoff is
    the seam's rule, and stating None (host browsing, no turn being decided)
    is a decision the caller must make visibly."""
    rows = visible_memory_rows(before_turn_idx=current_turn_idx,
                               include_archived=include_archived)
    if not rows:
        return []
    query_text = str(query or "").strip()
    _aspects = [(str(lbl), str(txt).strip()) for lbl, txt in (aspects or [])
                if str(txt or "").strip()]
    # One embedding call covers the query and every aspect. A caller that
    # already embedded this exact batch passes it in; the length check guards
    # against ranking on someone else's vectors.
    if (embedded is None
            or len(getattr(embedded, "vectors", ()) or ())
            != 1 + len(_aspects)):
        embedded = embed_texts_meta([query_text or "memory"]
                                    + [txt for _lbl, txt in _aspects])
    qv = embedded.vectors[0]
    aspect_vectors = list(zip((lbl for lbl, _t in _aspects),
                              embedded.vectors[1:]))
    memories, sem_scores, cue_scores, comparable = {}, [], [], {}
    for row in rows:
        mem = _row_memory(row)
        fv, cv = _vec(row["embedding"]), _vec(row["cue_embedding"])
        # A vector from another model is not comparable; it scores 0 on both
        # vector rankings and reaches recall through BM25/exact only.
        compatible = (row["embedding_model"] == embedded.model_key
                      and row["embedding_dim"] == embedded.dimensions)
        sem = _cos(qv, fv) if compatible and fv is not None else 0.0
        cue = _cos(qv, cv) if compatible and cv is not None else 0.0
        mem["_vector"] = fv if compatible else None
        memories[mem["id"]] = mem
        sem_scores.append((sem, mem["id"]))
        cue_scores.append((cue, mem["id"]))
        if compatible and aspect_vectors:
            comparable[mem["id"]] = (fv, cv)
    sem_rank = [m for s, m in sorted(sem_scores, reverse=True) if s > 0][:60]
    cue_rank = [m for s, m in sorted(cue_scores, reverse=True) if s > 0][:60]
    lex_rank = _lexical_ranking(query_text, visible_ids=memories.keys())
    # Scored once per row, not three times (twice in the sort key's
    # comparisons plus once in the filter) over the whole visible bank.
    exact_scores = {mid: _exact_cue_score(mem, query_text)
                    for mid, mem in memories.items()}
    exact_rank = sorted((mid for mid, s in exact_scores.items() if s > 0),
                        key=lambda mid: -exact_scores[mid])
    fused = defaultdict(float)
    reasons = defaultdict(list)
    # The cue vector outranks the full document because it is built from the
    # same short cue-shaped material a query is; exact match outranks both
    # because a literal phrase/entity hit is the strongest signal there is.
    _rrf_add(fused, reasons, sem_rank, 1.0, "semantic match")
    _rrf_add(fused, reasons, cue_rank, 1.15, "cue-vector match")
    _rrf_add(fused, reasons, lex_rank, 1.1, "keyword match")
    _rrf_add(fused, reasons, exact_rank, 1.25, "exact phrase or entity match")
    for label, av in aspect_vectors:
        scored = []
        for mid, (fv, cv) in comparable.items():
            best = max(_cos(av, fv) if fv is not None else 0.0,
                       _cos(av, cv) if cv is not None else 0.0)
            if best > 0:
                scored.append((best, mid))
        if scored:
            ranked = [mid for _s, mid in sorted(scored, reverse=True)][:60]
            _rrf_add(fused, reasons, ranked, _ASPECT_WEIGHT, label)
    tmode = _temporal_mode(query_text)
    known_turns = [m["turn_idx"] for m in memories.values()
                   if m["turn_idx"] is not None]
    max_turn = (current_turn_idx if current_turn_idx is not None
                else max(known_turns, default=0))
    ranked_importance = _rank_normalized_importance(memories)
    for mid, mem in memories.items():
        fused[mid] += 0.08 * ranked_importance[mid]
        fused[mid] += 0.04 * mem["confidence"]
        if mem["kind"] == "inference":
            # Belief-weighted recall, signed around 0.5: confidence on an
            # inference row is not a mint-time constant — reconciliation
            # tracks it to current credence — so a held belief is promoted
            # and an abandoned one demoted. Same band as the importance term:
            # it breaks ties between competing inferences, it does not
            # outrank an actual semantic match.
            fused[mid] += 0.10 * (mem["confidence"] - 0.5)
            if mem["confidence"] >= 0.6:
                reasons[mid].append("belief still held")
            elif mem["confidence"] <= 0.25:
                reasons[mid].append("belief since revised")
        fused[mid] += 0.08 * exact_scores[mid]
        ti = mem["turn_idx"]
        if ti is not None and max_turn and tmode != "neutral":
            # Temporal bonuses fire ONLY on explicit query language ("months
            # ago", "just now"). There is no unconditional recency term —
            # recency reaches the payload through recent_memory_buffer, which
            # is a separate field.
            age = _clamp((max_turn - ti) / max(max_turn, 1))
            if tmode == "old":
                fused[mid] += 0.12 * age
            else:
                fused[mid] += 0.12 * (1.0 - age)
            reasons[mid].append(f"{tmode}-memory cue")
    ranked = sorted(memories, key=lambda x: fused[x], reverse=True)
    # How many are actually worth carrying, before MMR picks WHICH. The count
    # is a consequence of the scores, not an input -- see _RELEVANCE_FLOOR_RATIO.
    if ranked:
        floor = fused[ranked[0]] * _RELEVANCE_FLOOR_RATIO
        worth_it = sum(1 for mid in ranked if fused[mid] >= floor)
        # The floor is a guard against a flat score curve, NOT a licence to
        # overrun the caller's stated budget. It used to be applied as
        # `max(min(_RECALL_FLOOR, len(ranked)), ...)`, which ignored `k`
        # entirely — so the ponder lane, which asks for 4, was handed 6 (then
        # 8 after neighbour padding). A caller that names a budget means it.
        k = max(min(_RECALL_FLOOR, len(ranked), k), min(k, worth_it))
    # MMR diversity over a pool wider than k, then chronological-neighbour
    # padding for the top episodes so a recalled moment arrives with its
    # immediate context.
    selected = []
    pool = ranked[:max(k * 8, 40)]
    # Redundancy against the selected set is a running MAX, so it only ever
    # increases: keeping it per-candidate turns the selection from
    # O(k^2 * |pool|) similarity computations into O(k * |pool|). At k=64
    # with a 512-row pool that is ~1M dot products against ~33k -- the same
    # choices, roughly thirty times less arithmetic, and it matters because
    # `k` is now bounded by relevance rather than by 16.
    redundancy = defaultdict(float)
    while pool and len(selected) < k:
        best_id, best = None, float("-inf")
        for mid in pool:
            mmr = 0.82 * fused[mid] - 0.18 * redundancy[mid]
            if mmr > best:
                best, best_id = mmr, mid
        selected.append(best_id)
        pool.remove(best_id)
        chosen = memories[best_id]
        for mid in pool:
            sim = _memory_similarity(memories[mid], chosen)
            if sim > redundancy[mid]:
                redundancy[mid] = sim
    expanded = list(selected)
    if len(expanded) < k + 2:
        by_turn = sorted((m for m in memories.values()
                          if m["turn_idx"] is not None),
                         key=lambda m: (m["turn_idx"], m["id"]))
        positions = {m["id"]: i for i, m in enumerate(by_turn)}
        for mid in selected[:3]:
            mem = memories[mid]
            if mem["kind"] != "episodic":
                continue
            pos = positions.get(mid)
            if pos is None:
                continue
            for np_ in (pos - 1, pos + 1):
                # Checked BEFORE appending. `break` only left the inner loop,
                # so each of the three padded episodes could add one more
                # neighbour after the ceiling was already reached: k+2 was
                # really k+4, and RECALL_LIMIT's "payload ceiling" could emit
                # 68. The existing test asserted <= 10 for k=8 and passed only
                # because its bank was dense enough that the neighbours were
                # already selected.
                if len(expanded) >= k + 2:
                    break
                if 0 <= np_ < len(by_turn):
                    nid = by_turn[np_]["id"]
                    if (nid not in expanded
                            and abs(by_turn[np_]["turn_idx"]
                                    - mem["turn_idx"]) <= 1):
                        expanded.append(nid)
                        reasons[nid].append(
                            "chronological neighbor of recalled episode")
            if len(expanded) >= k + 2:
                break
    result = []
    for mid in expanded:
        mem = dict(memories[mid])
        mem.pop("_vector", None)
        mem["score"] = round(fused[mid], 6)
        mem["retrieval_reasons"] = reasons[mid]
        result.append(mem)
    if chronological:
        # Chronological, not ranked: every consumer reads the result as a
        # narrative, and rank order presents a life out of sequence. Ranking
        # already did its work by choosing WHICH rows.
        result.sort(key=lambda m: (m["turn_idx"] is None,
                                   m["turn_idx"] if m["turn_idx"] is not None
                                   else 10 ** 12, m["id"]))
    if result and count_access:
        now = time.time()
        ids = [m["id"] for m in result]
        ph = ",".join("?" for _ in ids)
        qi(f"UPDATE memories SET access_count=access_count+1, last_accessed=?"
           f" WHERE id IN ({ph})", (now, *ids))
    return result


def recent_memory_buffer(current_turn_idx, turns=4, limit=12):
    rows = visible_memory_rows(
        before_turn_idx=current_turn_idx, include_archived=False,
        since_turn_idx=max(0, (current_turn_idx or 0) - turns),
        require_turn_idx=True)
    rows = sorted(rows, key=lambda r: (r["turn_idx"], r["id"]))
    return [_row_memory(r) for r in rows[-limit:]]


# ---- Projection: what a retrieved row looks like to the model ----
#
# An allow-list, not the row: numeric ids, access counters, archive state,
# embedding metadata and retrieval scores stay host-side. The durable
# event_key is the citable handle (`memory_ref`), because row ids do not
# survive restore paths and models cite what they are shown. temporal_status
# is stamped on every row rather than implied by the parent list, because a
# model flattens list-level meaning.

def _beats_ago(current_turn_idx, turn_idx):
    if current_turn_idx is None or turn_idx is None:
        return "at some earlier point"
    ago = max(0, int(current_turn_idx) - int(turn_idx))
    if ago <= 1:
        return "last turn"
    if ago <= 4:
        return "a few turns ago"
    if ago <= 12:
        return f"about {ago} turns ago"
    return f"about {ago} turns ago, some while back"


def project_memory(mem, current_turn_idx=None):
    out = {
        "memory_ref": mem.get("event_key") or "",
        "temporal_status": "remembered_past",
        "when": _beats_ago(current_turn_idx, mem.get("turn_idx")),
        "kind": mem.get("kind"),
        # Unknown provenance reads as testimony, never as experience — see
        # provenance_scope. The two defaults must agree, and they must both
        # claim the least.
        "epistemic_origin": _ORIGIN_LABELS.get(mem.get("provenance"),
                                               "what_i_was_told"),
        "gist": mem.get("gist") or "",
        "details": mem.get("content") or "",
        "confidence": round(float(mem.get("confidence") or 0.0), 3),
    }
    if mem.get("source_url"):
        out["source_url"] = mem["source_url"]
    disputed = mem.get("disputed")
    if isinstance(disputed, dict):
        # Both survive: the memory as formed AND the later re-reading. The
        # assistant still remembers reading what it read; it also remembers
        # having since decided it meant something else.
        out["i_now_read_this_differently"] = disputed.get("reading")
    return out


# ---- Consolidation ----
#
# Fires every N turns or M unarchived rows. Writes one summary row per
# epistemic scope per WINDOW — the (scope, end_turn_idx) key, because the
# engine's singleton design was overwriting every chapter but the last (53 of
# its 67 banks lost their opening turns before the key was completed). The
# first-hand row is written unconditionally even for a window with nothing
# first-hand, because its end_turn_idx IS the consolidation cursor: skip it
# and the same memories re-consolidate forever.

CONSOLIDATE_EVERY_TURNS = 10
CONSOLIDATE_ROW_PRESSURE = 40
_ARCHIVE_SALIENCE_FLOOR = 0.72
_ARCHIVE_KEEP_RECENT = 12
# Commitments never age out of the working set: a promise to the user is
# governed by being kept, not by consolidation.
_ARCHIVE_PROTECTED_KINDS = ("commitment",)

_SCOPE_FIELDS = (
    (SCOPE_FIRSTHAND, "summary"),
    (SCOPE_RECEIVED, "received_summary"),
    (SCOPE_SURMISE, "surmise_summary"),
)

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“‘])")
_SUPPORT_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her him his i "
    "in into is it its me my not of on or she that the their them then there "
    "they this to was were what when where which who will with you your"
    .split())
# Two shared content words is coincidence in prose this dense; three is a
# claim about the same thing. The engine calibrated this against its live
# corpus: at two, every clause matched every memory in its own window.
_SUPPORT_MIN_OVERLAP = 3
_SUPPORT_MAX_REFS = 3


def _content_words(text):
    return {w for w in re.findall(r"[a-z0-9']{3,}", str(text or "").casefold())
            if w not in _SUPPORT_STOPWORDS}


def derive_summary_support(summary, memories):
    """Which of this window's memories stand behind each clause of a summary.

    Derived HOST-SIDE by content-word overlap — deliberately not a model call
    and not embeddings: the question is "which stored rows does this sentence
    actually talk about", a lexical question with a checkable answer, and an
    audit trail produced by the same kind of process it audits is not one.

    An empty support_refs is a RESULT, not a failure: the clause generalises,
    compresses several rows, or was invented. Distinguishing those is a
    judgement; this is a measurement. Refs are event_keys, never row ids.

    Support is scoped by the CALLER to the summary's own epistemic class — a
    first-hand clause supported by something merely read would be an audit
    trail that launders sources into experience."""
    text = " ".join(str(summary or "").split())
    if not text:
        return []
    rows = []
    for mem in memories or []:
        if not isinstance(mem, dict):
            continue
        ref = str(mem.get("event_key") or "").strip()
        if not ref:
            continue
        words = _content_words(mem.get("gist")) \
            | _content_words(mem.get("content"))
        for phrase in (mem.get("key_phrases") or []):
            words |= _content_words(phrase)
        for entity in (mem.get("entities") or []):
            words |= _content_words(entity)
        rows.append((ref, words, mem.get("provenance")))
    out = []
    for clause in [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()]:
        cw = _content_words(clause)
        scored = sorted(
            ((len(cw & words), ref, prov) for ref, words, prov in rows
             if len(cw & words) >= _SUPPORT_MIN_OVERLAP),
            key=lambda item: (-item[0], item[1]))[:_SUPPORT_MAX_REFS]
        out.append({
            "claim": clause,
            "support_refs": [ref for _n, ref, _p in scored],
            # The class of the STRONGEST supporter; blank rather than
            # defaulted when there is none — the safest wrong answer is the
            # one that claims the least.
            "epistemic_origin": (_ORIGIN_LABELS.get(scored[0][2], "")
                                 if scored else ""),
        })
    return out


def thread_text(item):
    """A thread as text, whichever spelling it arrived in.

    Rows written before threads carried a stamp are bare strings and always
    will be; folding both spellings here is the alternative to every reader
    remembering which one it has (AGENTS.md: a guard that must be remembered
    will be forgotten)."""
    if isinstance(item, dict):
        return " ".join(str(item.get("thread") or "").split())
    return " ".join(str(item or "").split())


def fold_threads(threads, at_turn, previous=()):
    """Stamp each thread with the turn it was first opened.

    AN UNDATED THREAD IS INDISTINGUISHABLE FROM A CURRENT ONE, and that is
    not hypothetical: "the promised source-code upload has not arrived; the
    uploaded-files field is still empty" sat in a payload alongside 55
    uploaded files, and had done for turns. Threads are carried forward by a
    consolidator that is asked to let resolved detail go and has no way to
    check; nothing anywhere re-read current state. The prose said "nothing has
    landed yet" in the same breath.

    A stamp does not resolve the thread — nothing deterministic can, since
    "is this question still open" is a judgement. It makes the thread say how
    long it has been making its claim, which is what turns a confident stale
    sentence into a visibly suspicious one, for the assistant reading it and
    for the consolidator deciding whether to keep it.

    The stamp SURVIVES a merge: a thread carried forward keeps the turn it was
    opened on, or age would reset every ten turns and measure nothing."""
    prior = {}
    for item in previous or ():
        text = thread_text(item)
        if text:
            prior[text] = int(item.get("since_turn") or at_turn) \
                if isinstance(item, dict) else int(at_turn)
    out, seen = [], set()
    for item in threads or ():
        text = thread_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        own = item.get("since_turn") if isinstance(item, dict) else None
        out.append({"thread": text,
                    "since_turn": int(own or prior.get(text, at_turn) or 0)})
    return out


def close_threads(texts, turn_idx):
    """Retire threads the turn has just answered. Returns {closed, unknown}.

    NOTHING RE-READ CURRENT STATE. Threads were written once and only a
    consolidator — which runs every CONSOLIDATE_EVERY_TURNS turns — could ever
    drop one, so a question answered in the very payload that carried it went
    on being asked for up to ten more turns. Two threads were observed being
    answered by fields sitting beside them in the same payload.

    Matched on the thread's own text rather than an index, for the reason
    `chunks.expand` reports unknown ids: a positional handle silently closes
    the wrong thread when the list shifts, and the wrong closure is invisible
    while an unmatched string can be reported.

    The row's EMBEDDING is deliberately not recomputed. It is a retrieval aid
    over the window's prose, this runs inside the commit transaction, and an
    embedding call there would put a network round trip inside the one lock
    the whole design keeps short. A vector still carrying a closed thread
    makes the window slightly easier to surface, never harder — the safe
    direction for the error to point."""
    row = q("SELECT id, end_turn_idx, unresolved_threads FROM memory_summaries "
            "WHERE scope=? ORDER BY end_turn_idx DESC, id DESC",
            (SCOPE_FIRSTHAND,), one=True)
    wanted = {" ".join(str(t or "").split()) for t in (texts or [])}
    wanted.discard("")
    if row is None or not wanted:
        return {"closed": [], "unknown": sorted(wanted)}
    threads = fold_threads(_json_list(row["unresolved_threads"]),
                           row["end_turn_idx"])
    keep = [t for t in threads if t["thread"] not in wanted]
    closed = [t["thread"] for t in threads if t["thread"] in wanted]
    if closed:
        qi("UPDATE memory_summaries SET unresolved_threads=? WHERE id=?",
           (json.dumps(keep, ensure_ascii=False), row["id"]))
    return {"closed": closed, "unknown": sorted(wanted - set(closed))}


def _summary_retrieval_text(summary, key_phrases, unresolved_threads):
    return "\n".join(p for p in (summary or "",
                                 ", ".join(key_phrases or []),
                                 ", ".join(thread_text(t) for t
                                           in (unresolved_threads or []))) if p)


def save_memory_summary(summary, *, scope=SCOPE_FIRSTHAND, start_turn_idx=0,
                        end_turn_idx=0, key_phrases=None,
                        unresolved_threads=None, support=None):
    key_phrases = key_phrases or []
    # Folded HERE, where threads enter the store, rather than at each of the
    # four places that read them.
    unresolved_threads = fold_threads(
        unresolved_threads, int(end_turn_idx or 0),
        previous=get_memory_summary(scope).get("unresolved_threads"))
    embedded = embed_texts_meta(
        [_summary_retrieval_text(summary, key_phrases, unresolved_threads)])
    qi("""INSERT INTO memory_summaries(scope,start_turn_idx,end_turn_idx,
        summary,key_phrases,unresolved_threads,support,embedding,
        embedding_model,embedding_dim,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(scope,end_turn_idx) DO UPDATE SET
        start_turn_idx=excluded.start_turn_idx, summary=excluded.summary,
        key_phrases=excluded.key_phrases,
        unresolved_threads=excluded.unresolved_threads,
        support=excluded.support, embedding=excluded.embedding,
        embedding_model=excluded.embedding_model,
        embedding_dim=excluded.embedding_dim, updated=excluded.updated""",
       (scope, start_turn_idx, end_turn_idx, summary or "",
        json.dumps(key_phrases, ensure_ascii=False),
        json.dumps(unresolved_threads, ensure_ascii=False),
        json.dumps(support or [], ensure_ascii=False),
        _blob(embedded.vectors[0]), embedded.model_key, embedded.dimensions,
        time.time()))


def get_memory_summary(scope=SCOPE_FIRSTHAND, *, before_turn_idx=None):
    """The LATEST window for a scope. The cutoff applies here too: a window
    that closed at or after the deciding turn describes how this turn turned
    out and does not yet exist for the mind deciding it."""
    sql = "SELECT * FROM memory_summaries WHERE scope=?"
    params = [scope]
    if before_turn_idx is not None:
        sql += " AND end_turn_idx<?"
        params.append(int(before_turn_idx))
    row = q(sql + " ORDER BY end_turn_idx DESC, id DESC", tuple(params),
            one=True)
    if not row:
        return {"scope": scope, "start_turn_idx": 0, "end_turn_idx": 0,
                "summary": "", "key_phrases": [], "unresolved_threads": []}
    return {"scope": row["scope"], "start_turn_idx": row["start_turn_idx"],
            "end_turn_idx": row["end_turn_idx"], "summary": row["summary"],
            "key_phrases": _json_list(row["key_phrases"]),
            # Folded on read as well as on write: rows already in the table
            # are bare strings, and a reader that had to ask which it was
            # holding is exactly the guard that gets forgotten. An unstamped
            # thread takes its window's own end turn — it existed by then.
            "unresolved_threads": fold_threads(
                _json_list(row["unresolved_threads"]), row["end_turn_idx"])}


def search_memory_summaries(query, k=_SUMMARY_RECALL_LIMIT, *,
                            scope=SCOPE_FIRSTHAND, before_turn_idx=None,
                            exclude_latest=True, embedded=None):
    """Rank earlier windows against the turn's own query. No minimum score,
    deliberately: prose vectors score every window in a compressed band, so
    an absolute floor drops everything or nothing depending on the embedding
    model. Rank is trustworthy where magnitude is not; top-k is the honest
    way to use it."""
    sql = "SELECT * FROM memory_summaries WHERE scope=?"
    params = [scope]
    if before_turn_idx is not None:
        sql += " AND end_turn_idx<?"
        params.append(int(before_turn_idx))
    rows = q(sql + " ORDER BY end_turn_idx DESC", tuple(params))
    if exclude_latest and rows:
        rows = rows[1:]
    if not rows or k <= 0:
        return []
    if embedded is None or not getattr(embedded, "vectors", None):
        embedded = embed_texts_meta([str(query or "memory")])
    qv = embedded.vectors[0]
    scored = []
    for row in rows:
        if not str(row["summary"] or "").strip():
            continue
        if (row["embedding_model"] != embedded.model_key
                or row["embedding_dim"] != embedded.dimensions):
            # A cross-model vector is skipped rather than compared — compared,
            # its 0.0 silently ranks it last for a reason with nothing to do
            # with relevance.
            continue
        scored.append((_cos(qv, _vec(row["embedding"])), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"scope": r["scope"], "start_turn_idx": r["start_turn_idx"],
             "end_turn_idx": r["end_turn_idx"], "summary": r["summary"]}
            for _s, r in scored[:k]]


def consolidate_memory(current_turn_idx, consolidator):
    """One consolidation pass. `consolidator(payload) -> dict` is injected so
    this module stays free of prompt/provider knowledge and tests can run a
    deterministic one; the payload it receives is the window's memories in
    chronological order plus the previous first-hand summary.

    Returns the new latest first-hand summary, or None when nothing to do."""
    old = get_memory_summary(SCOPE_FIRSTHAND)
    rows = visible_memory_rows(
        before_turn_idx=int(current_turn_idx) + 1, include_archived=False,
        since_turn_idx=(old.get("end_turn_idx") or 0) + 1,
        require_turn_idx=True)
    rows = sorted(rows, key=lambda r: (r["turn_idx"], r["id"]))
    memories = [_row_memory(r) for r in rows]
    memories = [m for m in memories if (m.get("content") or "").strip()]
    if not memories:
        return None
    start_turn = min(m["turn_idx"] for m in memories)
    end_turn = max(m["turn_idx"] for m in memories)
    result = consolidator({
        # The threads arrive with their age attached, because the instruction
        # to let resolved detail go was being given to a reader who could not
        # tell resolved from merely old. A thread open for thirty turns while
        # the window below it discusses the thing it says has not happened is
        # the case this exists to make visible.
        "previous_summary": {
            **old,
            "unresolved_threads": [
                {"thread": t["thread"], "opened_at_turn": t["since_turn"],
                 "turns_open": max(0, end_turn - int(t["since_turn"] or 0))}
                for t in (old.get("unresolved_threads") or [])]},
        # All three scopes, not just first-hand. The prompt orders "merge the
        # previous summary's still-relevant content forward" — and the
        # consolidator was only ever shown the first-hand one, so for
        # received and surmise it was being told to carry forward something
        # it could not see. Each new window's received summary then shadowed
        # the last (they key on `end_turn_idx`, and only the latest is read),
        # so a fact the user told the assistant left the summary layer for
        # good the moment any later window contained a told/read row.
        "previous_received_summary": get_memory_summary(SCOPE_RECEIVED),
        "previous_surmise_summary": get_memory_summary(SCOPE_SURMISE),
        "memories_chronological": [
            {"turn_idx": m["turn_idx"], "kind": m["kind"],
             "provenance": m["provenance"], "salience": m["salience"],
             "confidence": m["confidence"], "gist": m["gist"],
             "details": m["content"], "key_phrases": m["key_phrases"],
             "source_url": m["source_url"]}
            for m in memories],
    }) or {}
    present = {provenance_scope(m.get("provenance")) for m in memories}
    # MODEL OUTPUT IS PROVISIONAL UNTIL DETERMINISTIC CODE VALIDATES IT — and
    # this was the one commit path that took the model's word for its own
    # output SHAPE. A consolidator returning valid JSON with a renamed or
    # dropped field (`{"recap": ...}`) sailed through: the first-hand summary
    # was written as "", its end_turn_idx — which IS the cursor — advanced,
    # and that chapter was gone from the summary layer forever. The
    # "first-hand is written even when empty" rule below is for a window with
    # nothing first-hand in it; it cannot tell that apart from a model that
    # lost the field, so the distinction has to be made here, where the rows
    # are still in hand. Raising leaves the cursor where it is and the
    # pipeline retries on the next trigger — a summary is a reconstructible
    # cache, so the safe failure is to not have one yet.
    if (SCOPE_FIRSTHAND in present
            and not str(result.get("summary") or "").strip()):
        raise RuntimeError(
            "consolidator returned no `summary` for a window containing "
            f"{sum(1 for m in memories if provenance_scope(m.get('provenance')) == SCOPE_FIRSTHAND)} "
            "first-hand rows; refusing to advance the cursor over them")
    # One transaction for every scope AND the archive sweep. The first-hand
    # row carries the cursor and was written FIRST, so a failure between
    # scope writes advanced the cursor past a window whose received/surmise
    # summaries were never written — the same lost-chapter shape, narrower
    # trigger. Consolidation runs post-commit, so nothing contends here.
    with transaction():
        for scope, field in _SCOPE_FIELDS:
            text = str(result.get(field) or "").strip()
            # First-hand is written even when empty — its end_turn_idx is the
            # cursor. The other scopes are skipped when the window has neither
            # text nor rows of that class.
            if scope != SCOPE_FIRSTHAND and not text and scope not in present:
                continue
            save_memory_summary(
                text, scope=scope, start_turn_idx=start_turn,
                end_turn_idx=end_turn,
                key_phrases=(result.get("key_phrases") or []
                             if scope == SCOPE_FIRSTHAND else []),
                unresolved_threads=(result.get("unresolved_threads") or []
                                    if scope == SCOPE_FIRSTHAND else []),
                support=derive_summary_support(
                    text, [m for m in memories
                           if provenance_scope(m.get("provenance")) == scope]))
        _archive_stale_rows(end_turn)
    return get_memory_summary(SCOPE_FIRSTHAND)


def _archive_stale_rows(end_turn):
    """Retire low-stakes rows that have fallen out of the rolling window —
    never from recall (search defaults include_archived=True), only from the
    recent buffer and future consolidation windows.

    Scoped to the whole bank, not to the window just summarised, and that is
    the fix rather than a widening. The sweep used to read
    `cutoff = max(start_turn, end_turn - _ARCHIVE_KEEP_RECENT)` over the
    window's own rows — but a window is `CONSOLIDATE_EVERY_TURNS` (10) turns
    wide and the keep-recent guard is 12, so `end_turn - 12` always landed
    BELOW `start_turn`, the clamp pinned the cutoff to `start_turn`, and
    nothing in the window was ever older than the window's own start.
    Measured: 0 rows archived across 40 turns and 4 consolidations. The
    mechanism could not fire by construction — two constants that were never
    compared to each other.

    Reads the HIGHER of salience and effective_importance: a memory that
    turned out to matter is not retired on the strength of how ordinary it
    looked at the time, which is the entire reason the two numbers are
    separate."""
    cutoff = int(end_turn) - _ARCHIVE_KEEP_RECENT
    if cutoff <= 0:
        return 0
    rows = q("SELECT id, salience, importance, kind FROM memories "
             "WHERE archived=0 AND turn_idx IS NOT NULL AND turn_idx<?",
             (cutoff,))
    archivable = [
        r["id"] for r in rows
        if r["kind"] not in _ARCHIVE_PROTECTED_KINDS
        and max(float(r["salience"] or 0.0),
                effective_importance({"salience": r["salience"],
                                      "importance": r["importance"]}))
        < _ARCHIVE_SALIENCE_FLOOR
    ]
    for start in range(0, len(archivable), 400):
        chunk = archivable[start:start + 400]
        marks = ",".join("?" for _ in chunk)
        qi(f"UPDATE memories SET archived=1 WHERE id IN ({marks})",
           tuple(chunk))
    return len(archivable)


def maybe_consolidate(current_turn_idx, consolidator):
    summary = get_memory_summary(SCOPE_FIRSTHAND)
    last_turn = summary.get("end_turn_idx") or 0
    count = q("SELECT COUNT(*) AS c FROM memories WHERE archived=0 AND "
              "turn_idx>?", (last_turn,), one=True)["c"]
    if (current_turn_idx - last_turn < CONSOLIDATE_EVERY_TURNS
            and count < CONSOLIDATE_ROW_PRESSURE):
        return None
    return consolidate_memory(current_turn_idx, consolidator)


# ---- Inference-confidence reconciliation ----
#
# An inference memory is minted with the confidence the assistant declared
# when it formed the belief; the belief store keeps moving (blend, explain
# away, decay, prune). Without reconciliation the assistant could hold one
# belief and preferentially RECALL the one it had already abandoned, because
# recall ranked on a number frozen at mint time.

# The mint formula, and its inverse. salience = 0.45 + 0.3 * confidence at
# mint lets the mint confidence be recovered later without a second column,
# BECAUSE salience is never revised.
def mint_salience(confidence):
    return round(0.45 + 0.3 * _clamp(confidence), 4)


def _mint_confidence_of(salience):
    try:
        return max(0.0, min(1.0, (float(salience) - 0.45) / 0.3))
    except (TypeError, ValueError):
        return 0.5


_ABANDONED_DECAY = 0.55
_ABANDONED_FLOOR = 0.08


def _abandoned_confidence(salience):
    """The resting confidence for an inference no live hypothesis carries.

    A FIXED FRACTION of the mint value, never a compounding per-turn decay —
    the engine measured the compounding rule crushing 76–80% of a long
    story's entire inference bank to the floor within 7–18 turns, removing
    inferences from recall wholesale. "No surviving hypothesis carries this"
    is usually expiry from a bounded working set, not revision: a belief that
    merely aged out was never concluded WRONG and must not rank as though it
    was. Pure function of untouched salience = idempotent, and a corpus
    previously crushed self-heals on its next pass."""
    mint = _mint_confidence_of(salience)
    return min(mint, max(_ABANDONED_FLOOR, mint * _ABANDONED_DECAY))


def reconcile_inference_confidence(state, turn_idx, credence_fn):
    """Re-weight inference memories to current belief. `credence_fn(state,
    subject, claim, turn_idx) -> float | None` is beliefs.belief_credence,
    injected to keep this module import-clean.

    The only inputs are the assistant's own rows and its own belief store —
    nothing consults whether a belief was TRUE. Revision comes from what was
    later learned, never from being graded against reality; for the engine
    that is the firewall, and here it is still the honest shape: the store
    records what the assistant concluded, and evidence (not fiat) moves it."""
    rows = q("SELECT id, entities, gist, salience, confidence FROM memories "
             "WHERE kind='inference'")
    updates = []
    for row in rows:
        subjects = _json_list(row["entities"])
        subject = str(subjects[0]).strip() if subjects else ""
        claim = str(row["gist"] or "").strip()
        if not subject or not claim:
            continue
        credence = credence_fn(state, subject, claim, turn_idx)
        abandoned = _abandoned_confidence(row["salience"])
        if credence is None:
            revised = abandoned
        else:
            # A claim STILL STORED must never rank below one that was pruned:
            # half-life decay on a surviving hypothesis measures staleness,
            # not disbelief. Held >= abandoned, always.
            revised = max(credence, abandoned)
        if abs(revised - float(row["confidence"] or 0.0)) > 1e-6:
            updates.append((round(revised, 4), row["id"]))
    for confidence, mid in updates:
        qi("UPDATE memories SET confidence=? WHERE id=?", (confidence, mid))
    return len(updates)


# ---- The per-turn memory payload ----

def build_memory_context(current_turn_idx, query_text, *, aspects=None,
                         ponder_query="", recall_limit=RECALL_LIMIT):
    """Everything the assistant remembers into one structured payload:
    recent buffer, ranked recall, the three scope summaries, up to two
    earlier first-hand chapters, and the deliberate-recall (ponder) lane.

    Returns (payload, internal) — internal holds the delivered refs the
    citation guard grounds against; it never reaches the model."""
    aspects = [(str(lbl), str(txt)) for lbl, txt in (aspects or [])
               if str(txt or "").strip()]
    recent = recent_memory_buffer(current_turn_idx)
    recent_ids = {m["id"] for m in recent}
    query_text = str(query_text or "").strip()
    if not query_text:
        query_text = " ".join(t for _l, t in aspects) or "memory"
    embedded = embed_texts_meta([query_text]
                                + [txt for _lbl, txt in aspects])
    recalled = search_memories(query_text, k=recall_limit,
                               current_turn_idx=current_turn_idx,
                               aspects=aspects, embedded=embedded)
    recalled = [m for m in recalled if m["id"] not in recent_ids]
    recalled, recall_spent, recall_dropped = _fit_recall_budget(recalled)
    # Deliberate recall: one query the assistant set for itself. Additive and
    # explicitly labelled; results already in normal recall are tagged, not
    # duplicated.
    ponder_query = " ".join(str(ponder_query or "").split())[:240]
    pondered = []
    if ponder_query:
        pondered = search_memories(ponder_query, k=4,
                                   current_turn_idx=current_turn_idx)
    normal_refs = {m["event_key"] for m in (*recent, *recalled)}
    summary = get_memory_summary(before_turn_idx=current_turn_idx)
    earlier = search_memory_summaries(
        query_text, before_turn_idx=current_turn_idx, embedded=embedded)
    payload = {
        "recent_exchanges": [project_memory(m, current_turn_idx)
                             for m in recent],
        "recalled_old_memories": [project_memory(m, current_turn_idx)
                                  for m in recalled],
        "what_happened_between_us": summary.get("summary") or "",
        # Each thread says how long it has been claiming to be unresolved. A
        # thread is written once and carried forward by a consolidator that
        # never re-reads the world, so an old one can assert something the
        # rest of this same payload contradicts; the age is what makes that
        # legible instead of authoritative.
        "unresolved_threads": [
            {"thread": t["thread"],
             "open_since": _beats_ago(current_turn_idx, t["since_turn"])}
            for t in (summary.get("unresolved_threads") or [])],
    }
    for scope in (SCOPE_RECEIVED, SCOPE_SURMISE):
        scoped = get_memory_summary(scope, before_turn_idx=current_turn_idx)
        text = str(scoped.get("summary") or "").strip()
        if text:
            # Absent, not empty, when there is nothing — an empty key still
            # spends attention.
            payload[scope_label(scope)] = text
    if earlier:
        # Chronological, oldest first: ranking chose WHICH chapters; it must
        # not choose the order a history is read in.
        payload["earlier_in_our_history"] = [
            {"what_i_lived_through_then": w["summary"],
             "when": _beats_ago(current_turn_idx, w["end_turn_idx"])}
            for w in sorted(earlier, key=lambda w: w["end_turn_idx"])]
    if ponder_query:
        additional = []
        tagged = 0
        for mem in pondered:
            if mem["event_key"] in normal_refs:
                tagged += 1
                continue
            item = project_memory(mem, current_turn_idx)
            item["retrieval_origin"] = "deliberate_ponder"
            additional.append(item)
        payload["deliberate_recall"] = {
            "query_i_chose": ponder_query,
            "temporal_status": "remembered_past",
            "already_in_normal_recall": tagged,
            "additional_memories": additional,
        }
    delivered = {str(m.get("event_key") or "")
                 for m in (*recent, *recalled, *pondered)
                 if str(m.get("event_key") or "")}
    health = retrieval_health(current_turn_idx, embedded)
    # HOW WIDE THE RELEVANCE GATE ACTUALLY OPENED, recorded every turn.
    #
    # `_RELEVANCE_FLOOR_RATIO` is documented as keeping an ordinary question
    # in the 12-25 range, and nobody had ever checked. There is a structural
    # reason to doubt it: RRF contributes 12/(60+rank) per lane, so rank 60
    # scores 50.8% of rank 1 WITHIN A LANE — and the floor is 0.45, below
    # that. No ratio under 0.5 can discriminate inside a lane at all; the cut
    # that does happen is really "how many of the four lanes did this row
    # appear in", a coarse four-valued signal rather than a cliff. Measured
    # over 120 synthetic turns against a 360-row bank, recall returned 55-65
    # rows per turn, saturating RECALL_LIMIT.
    #
    # This is NOT retuned here. The measurement above ran under cheap_embed,
    # which the README records at 0% recall on vocabulary-disjoint
    # paraphrases — a bank where everything correlates with everything is
    # exactly the corpus that would make a good gate look bad, and picking a
    # constant off it would be the "small-payload benchmark" scar again. So
    # the number is EXPORTED instead: every turn now records what the gate
    # let through against what it was offered, and the tuning decision gets
    # made on real traffic with real embeddings, the way the engine's numbers
    # were earned. See DESIGN.md § Theorycraft.
    health["recall"] = {
        "returned": len(recalled),
        "recent_buffer": len(recent),
        "pondered": len(pondered),
        "ceiling": recall_limit,
        "at_ceiling": len(recalled) >= recall_limit,
        # The cut that actually binds. `at_ceiling` answers a question about
        # rows; this answers the one about bytes, which is what a payload is
        # billed and read in. Reported so a recall that shrank for cost cannot
        # be mistaken for a bank with nothing to say.
        "chars": recall_spent,
        "char_budget": RECALL_CHAR_BUDGET,
        "dropped_for_size": recall_dropped,
    }
    return payload, {"delivered_refs": delivered,
                     "retrieval_health": health}


def retrieval_health(current_turn_idx, embedded):
    """How much of the ranking machinery could actually run this turn.

    MEASURE BEFORE YOU ENRICH, applied to the measurement itself. providers.py
    promises that stamping each vector with its model "is what lets retrieval
    count and announce the stranding instead of quietly splitting the bank" —
    and nothing anywhere counted or announced it. The mint path warns when
    embedding degrades; the READ path, where the damage actually lands, said
    nothing at all.

    Two ways three of the four lanes silently vanish:
      - one transient 5xx at recall time makes the query fall back to
        cheap_embed, so no stored vector is comparable and semantic, cue and
        every aspect lane score zero for that turn;
      - rows minted during an outage carry `cheap:...` forever (there is no
        re-embed path), so part of the bank is permanently unreachable by
        vector even after the provider comes back.

    Either way retrieval keeps working — on BM25 and exact match alone — and
    looks fine. That is the project's central scar exactly: a mechanism
    assumed live that is not running."""
    rows = q("SELECT embedding_model AS m, embedding_dim AS d, COUNT(*) AS c "
             "FROM memories WHERE embedding IS NOT NULL "
             "GROUP BY embedding_model, embedding_dim")
    total = sum(r["c"] for r in rows)
    stranded = sum(r["c"] for r in rows
                   if r["m"] != embedded.model_key
                   or r["d"] != embedded.dimensions)
    return {
        "query_embedding_fallback": bool(getattr(embedded, "fallback", False)),
        "query_embedding_error": str(getattr(embedded, "error", "") or ""),
        "embedded_rows": total,
        "vector_incomparable_rows": stranded,
        "vector_lanes_live": total > 0 and stranded < total,
    }
