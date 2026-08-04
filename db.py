# db.py — SQLite access and schema for Sonder Assistant.
#
# Inherited from Sonder Engine's db.py in shape: tiny q/qi/transaction helpers
# over one sqlite3 connection per thread, WAL mode, a meta table carrying a
# schema_version, and additive migrations. What was cut: frames, checkpoints,
# world state, steps/variants. This app has one mind and no reruns, so the
# persistence surface is a fraction of the engine's — but the memory tables
# keep the engine's columns almost exactly, because every column that looks
# decorative there turned out to be load-bearing (importance vs salience,
# disputed, event_key, embedding_model).

import json
import os
import sqlite3
import threading
import time

_DB_PATH = os.environ.get("ASSISTANT_DB", "assistant.db")
_local = threading.local()
_lock = threading.Lock()

# How long a writer waits for the lock before giving up. Two turns committing
# at once is ordinary (two browser tabs, a retry, an API client), and the
# right answer is to queue, not to fail the turn.
_BUSY_TIMEOUT = float(os.environ.get("ASSISTANT_DB_TIMEOUT", "10"))

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS settings(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL
);

-- One row per exchange. turn_idx is a GLOBAL ordinal across every session,
-- not per-session: the assistant's memory spans sessions (that is the point
-- of long-term memory), and the retrieval turn-cutoff needs one shared play
-- order to filter on. The engine's per-chat idx would give two sessions
-- overlapping ordinals and make "before this turn" unanswerable.
CREATE TABLE IF NOT EXISTS turns(
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_idx INTEGER NOT NULL,
    user_text TEXT NOT NULL DEFAULT '',
    reply_text TEXT NOT NULL DEFAULT '',
    trace TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_idx);

-- The memory bank. Columns mirror the engine's `memories` table minus the
-- fictional/embodied ones (frame_id, location, the four affect axes).
--
--   salience    how much it mattered WHEN FORMED. Set at mint, never revised.
--   importance  how central it BECAME. NULL means "never revised" and every
--               reader falls back to salience (effective_importance), so an
--               untouched bank behaves exactly as a bank with no such column.
--   confidence  how much the assistant credits it NOW. Revised every turn for
--               inference rows by reconcile_inference_confidence.
--   disputed    the assistant's own later re-reading of a row it still holds.
--               JSON {"turn_idx": n, "reading": "...", "count": k}; '' when
--               undisputed. A column, not an edge: refs by row id do not
--               survive delete-and-reinsert restore paths, so the engine
--               stores the re-reading on the row itself and so do we.
--   event_key   idempotency key. Re-minting the same key UPDATEs instead of
--               inserting, which is what makes a re-run replace rather than
--               duplicate.
--   retired     the assistant setting a row aside as no longer relevant to
--               the project it is working on. JSON {"turn_idx", "reason",
--               "batch"}; '' when live. NOT the same as `archived`: an
--               archived row has left the rolling consolidation window and
--               is still fully recallable, while a retired row is out of
--               recall entirely. Reversible by construction, because the
--               judgement being made here is about RELEVANCE and a relevance
--               judgement can be wrong in a way a deletion cannot be undone
--               from. A column rather than a deletion for the same reason
--               `disputed` is a column: the row is still the evidence.
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    turn_id INTEGER REFERENCES turns(id) ON DELETE SET NULL,
    turn_idx INTEGER,
    kind TEXT NOT NULL DEFAULT 'episodic',
    provenance TEXT NOT NULL DEFAULT 'witnessed',
    salience REAL NOT NULL DEFAULT 0.5,
    content TEXT NOT NULL,
    gist TEXT NOT NULL DEFAULT '',
    key_phrases TEXT NOT NULL DEFAULT '[]',
    entities TEXT NOT NULL DEFAULT '[]',
    source_url TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL,
    embedding BLOB,
    cue_embedding BLOB,
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dim INTEGER,
    archived INTEGER NOT NULL DEFAULT 0,
    event_key TEXT NOT NULL DEFAULT '',
    importance REAL,
    disputed TEXT NOT NULL DEFAULT '',
    retired TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memories_chronology ON memories(turn_idx, id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_event
    ON memories(event_key) WHERE event_key <> '';
-- reconcile_inference_confidence scans kind='inference' every single turn,
-- and maybe_consolidate counts unarchived rows past the cursor every turn.
-- Both were full table scans against a bank that only grows.
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_retired ON memories(retired);
CREATE INDEX IF NOT EXISTS idx_memories_window
    ON memories(archived, turn_idx);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_retrieval_fts USING fts5(
    memory_id UNINDEXED,
    gist,
    content,
    key_phrases,
    entities,
    tokenize='unicode61 remove_diacritics 2'
);

-- Summary windows, keyed (scope, end_turn_idx) — the engine learned the hard
-- way (schema v23) that a singleton per scope overwrites every chapter but
-- the last. `support` is the per-clause citation trail derived host-side at
-- consolidation; an empty support set on a clause is a finding (the clause
-- generalises or was invented), not an error.
CREATE TABLE IF NOT EXISTS memory_summaries(
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'autobiographical',
    start_turn_idx INTEGER NOT NULL DEFAULT 0,
    end_turn_idx INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    key_phrases TEXT NOT NULL DEFAULT '[]',
    unresolved_threads TEXT NOT NULL DEFAULT '[]',
    support TEXT NOT NULL DEFAULT '[]',
    embedding BLOB,
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dim INTEGER,
    updated REAL NOT NULL,
    UNIQUE(scope, end_turn_idx)
);

-- Large material broken into navigable pieces. `chunks.py` explains the
-- shape; what matters here is that a chunk's BODY lives in the database and
-- never in a turn payload unless it was explicitly expanded. The turn carries
-- gists and ids; the bodies are fetched by id, which is the whole point.
--
-- UNIQUE(session_id, chunk_key) because a re-chunk of an edited file must
-- update its rows rather than accumulate a second copy alongside them — an
-- expand that returned code no longer on disk would be worse than one that
-- returned nothing.
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    chunk_key TEXT NOT NULL,
    session_id INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'code',
    source_ref TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    gist TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    start_line INTEGER,
    end_line INTEGER,
    chars INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    UNIQUE(session_id, chunk_key)
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id, kind);

-- One row of persistent cognitive state: the belief store (mind models keyed
-- by subject — "the user" is simply the most important subject), the stable
-- hypothesis-sheet keys, and a pending ponder query. JSON blob because the
-- engine's chat_chars.state proved the shape churns faster than a schema.
CREATE TABLE IF NOT EXISTS state(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '{}'
);

-- Research: a question under active investigation. `statement` is the current
-- best answer, revised as evidence lands. `status` is open / answered /
-- disputed / abandoned. A dispute is two live contradictory readings held
-- side by side — never an average, for the same reason the engine's
-- record_dispute keeps the memory AND the re-reading.
CREATE TABLE IF NOT EXISTS hypotheses(
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    statement TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.3,
    status TEXT NOT NULL DEFAULT 'open',
    dispute TEXT NOT NULL DEFAULT '',
    created_turn INTEGER NOT NULL DEFAULT 0,
    updated_turn INTEGER NOT NULL DEFAULT 0
);

-- Evidence rows cite REAL urls. Each is also minted as a `read`-provenance
-- memory (event_key links the two), so evidence is retrievable later by the
-- same machinery as everything else the assistant knows — a source consulted
-- last month can surface again and dispute a newer claim.
CREATE TABLE IF NOT EXISTS evidence(
    id INTEGER PRIMARY KEY,
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    url TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    excerpt TEXT NOT NULL DEFAULT '',
    stance TEXT NOT NULL DEFAULT 'context',
    event_key TEXT NOT NULL DEFAULT '',
    fetched_turn INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_hypothesis ON evidence(hypothesis_id);

-- One code experiment: what was predicted, what was run, what happened.
-- `digest` is (hypothesis, source, command) so the SAME experiment repeated is
-- recognisable -- which is how non-determinism gets caught instead of silently
-- overwriting the previous answer. Rows are kept, never updated: the history of
-- what a thing did is the finding.
CREATE TABLE IF NOT EXISTS experiments(
    id INTEGER PRIMARY KEY,
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    digest TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL DEFAULT '[]',
    expect TEXT NOT NULL DEFAULT '{}',
    outcome TEXT NOT NULL DEFAULT 'inconclusive',
    observation TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    turn_idx INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis ON experiments(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_experiments_digest ON experiments(digest);
"""


def configure(path):
    """Point the module at a database file. Tests call this with a temp path
    BEFORE any query runs; closing the old handle here is what lets a test
    tear its file down on Windows and under pytest's tmp_path cleanup."""
    global _DB_PATH
    close()
    _DB_PATH = path


def close():
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def _restrict_permissions(path):
    """Owner-only on the database and its WAL sidecars.

    This file stopped being only a memory bank when `config.KEY_VALUE_FIELDS`
    let a provider credential be stored in the settings row. A default umask
    leaves it world-readable, which on a shared machine hands the key to every
    other account. Applied on open rather than at creation because the
    interesting case is an EXISTING database created before this mattered.

    Best-effort by design: a filesystem without POSIX modes (a mount, a
    container volume) must not stop the assistant from starting, and the
    permissions are a second line behind not sharing the file at all."""
    for suffix in ("", "-wal", "-shm"):
        try:
            target = path + suffix
            if os.path.exists(target) and (os.stat(target).st_mode & 0o077):
                os.chmod(target, 0o600)
        except OSError:
            pass


def _conn():
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != _DB_PATH:
        if conn is not None:
            conn.close()
        conn = sqlite3.connect(_DB_PATH, timeout=_BUSY_TIMEOUT)
        _restrict_permissions(_DB_PATH)
        # TRUE AUTOCOMMIT, and the reason it is not optional.
        #
        # sqlite3's legacy mode (isolation_level == '') opens an implicit
        # transaction before every DML statement. Under it, `qi`'s
        # `if not conn.in_transaction: commit()` guard could never fire after
        # a write, and `transaction.__enter__` saw in_transaction=True, judged
        # itself NESTED, and never issued a COMMIT either. Measured
        # consequence: the whole turn -- turns, memories, beliefs, state --
        # stayed in one never-ending write transaction. A second connection
        # (any other uvicorn threadpool thread) saw zero rows and got
        # "database is locked"; closing the handle rolled everything back.
        # Every test passed because a test is one thread on one connection,
        # which sees its own uncommitted writes.
        #
        # With isolation_level=None sqlite3 issues no BEGIN of its own: a
        # standalone statement autocommits, and an explicit BEGIN from
        # `transaction()` is the only thing that opens one. That is what makes
        # the commit discipline in this file real rather than described.
        conn.isolation_level = None
        conn.row_factory = sqlite3.Row
        # busy_timeout FIRST, because the next two statements can themselves
        # need a lock and there is nothing to wait on until it is set.
        conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT * 1000)}")
        # Changing journal_mode needs an exclusive lock, and it does NOT
        # reliably honour busy_timeout — eight threads opening connections
        # while one holds a write transaction produced "database is locked"
        # here, from a statement that had nothing to do. Once the file is
        # already WAL this is a no-op worth skipping entirely.
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass                     # another connection is mid-write; it set
                                     # the mode, or will
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _local.path = _DB_PATH
        _init(conn)
    return conn


def _init(conn):
    with _lock:
        # Only when the schema is actually absent. `executescript` issues an
        # implicit COMMIT and then runs a page of DDL; doing that on every
        # connection open meant every new thread took a write lock to
        # re-create tables that already existed, which is both wasted work
        # and a lock to contend over.
        have = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            "AND name='memories'").fetchone()[0]
        if not have:
            conn.executescript(SCHEMA)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        try:
            version = int(row["value"]) if row else 0
        except (TypeError, ValueError):
            version = 0
        # Migrations, keyed on the stored version so they run ONCE per upgrade
        # rather than on every connection open. Additive only, engine rule:
        # NULL importance reads as salience, empty dispute is undisputed — a
        # new column must default to "behaves as before", never to a value
        # that fabricates history.
        #
        # v2: uniqueness on the global turn ordinal — the constraint that
        # would have caught the duplicate-ordinal race structurally instead of
        # letting it silently overwrite a turn's memories. It lives here
        # rather than in SCHEMA because a database written before the fix may
        # already hold duplicates, and refusing to open is a worse outcome
        # than running without the index on a legacy file.
        if version < 2:
            try:
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_turns_idx "
                             "ON turns(turn_idx)")
            except sqlite3.IntegrityError:
                pass
        # v3: retirement. Additive and defaulting to "behaves as before" —
        # every existing row reads as live, which is what it was.
        if version < 3:
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN "
                             "retired TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass          # already present on a fresh schema
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_retired "
                         "ON memories(retired)")
        # v4: the chunks table. A NEW TABLE MUST BE A MIGRATION, not just a
        # line in SCHEMA — `executescript(SCHEMA)` above runs only when the
        # schema is absent, so a table added to SCHEMA alone appears on fresh
        # installs and never on an existing database. The symptom is
        # "no such table" on exactly the machines that have data worth
        # keeping, which is the worst possible distribution of a bug.
        if version < 4:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks(
                    id INTEGER PRIMARY KEY,
                    chunk_key TEXT NOT NULL,
                    session_id INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'code',
                    source_ref TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    gist TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL DEFAULT '',
                    start_line INTEGER,
                    end_line INTEGER,
                    chars INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL,
                    UNIQUE(session_id, chunk_key)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_session
                    ON chunks(session_id, kind);
            """)
        if version != SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),))


def q(sql, args=(), one=False):
    cur = _conn().execute(sql, args)
    rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


def qi(sql, args=()):
    conn = _conn()
    cur = conn.execute(sql, args)
    if not conn.in_transaction:
        conn.commit()
    return cur.lastrowid


class transaction:
    """`with transaction():` — one outer write transaction. Nested use is a
    no-op (SQLite has one writer), matching the engine's commit discipline:
    all primary turn mutations inside one transaction, slow provider work
    (embedding) done BEFORE entering it so a network round trip never holds
    the write lock."""

    def __enter__(self):
        conn = _conn()
        self._outer = not conn.in_transaction
        if self._outer:
            # IMMEDIATE, not the default DEFERRED. A deferred BEGIN takes a
            # READ snapshot on its first SELECT and only tries to upgrade to a
            # writer later -- and under WAL an upgrade whose snapshot has gone
            # stale returns SQLITE_BUSY *instantly*, a failure no busy_timeout
            # can ever wait out. The commit stage reads before it writes
            # (delete_turn_memories, the event_key lookups), so it is exactly
            # the shape that loses that race. IMMEDIATE takes the write lock
            # up front, which makes concurrent commits queue on busy_timeout
            # instead of aborting a turn.
            conn.execute("BEGIN IMMEDIATE")
        return conn

    def __exit__(self, exc_type, exc, tb):
        conn = _conn()
        if self._outer:
            if exc_type is None:
                conn.commit()
            else:
                conn.rollback()
        return False


def next_turn_idx():
    """RESERVE the global play-order ordinal for a new turn. Monotonic across
    every session — see the turns table comment.

    A durable counter bumped inside a write transaction, not `MAX(turn_idx)`.
    Two reasons, both measured:

    1. `SELECT MAX(...)` then INSERT is a read-modify-write with no lock
       between the halves. Two concurrent turns both read N and both claim
       N+1; nothing in the schema stopped them, and both then minted
       `event_key = "turn:N+1:episode"`, so the second turn's upsert silently
       OVERWROTE the first turn's episode. One exchange disappears from
       memory with no warning. Two browser tabs is enough to cause it.
    2. The ordinal must be reserved BEFORE the model call (stage 1 needs it
       for the retrieval cutoff) but the turn row is only written at commit.
       A counter can be reserved without a row; MAX() cannot.

    Gaps are fine and expected — a turn that fails after reserving leaves a
    hole. The contract is unique and increasing, never dense."""
    with transaction():
        row = q("SELECT value FROM meta WHERE key='turn_cursor'", one=True)
        if row is None:
            seed = q("SELECT MAX(turn_idx) AS m FROM turns", one=True)
            current = int((seed["m"] if seed else 0) or 0)
        else:
            current = int(row["value"] or 0)
        nxt = current + 1
        qi("INSERT INTO meta(key,value) VALUES('turn_cursor',?) "
           "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(nxt),))
    return nxt


def setting_get(key, default=None):
    row = q("SELECT value FROM settings WHERE key=?", (key,), one=True)
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return default


def setting_put(key, value):
    qi("INSERT INTO settings(key,value) VALUES(?,?) "
       "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
       (key, json.dumps(value, ensure_ascii=False)))


def state_get(key, default=None):
    row = q("SELECT value FROM state WHERE key=?", (key,), one=True)
    if row is None:
        return default if default is not None else {}
    try:
        out = json.loads(row["value"])
    except (TypeError, ValueError):
        return default if default is not None else {}
    return out


def state_put(key, value):
    qi("INSERT INTO state(key,value) VALUES(?,?) "
       "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
       (key, json.dumps(value, ensure_ascii=False)))


def ensure_session(session_id=None, title=""):
    if session_id is not None:
        row = q("SELECT id FROM sessions WHERE id=?", (session_id,), one=True)
        if row:
            return row["id"]
    return qi("INSERT INTO sessions(title,created) VALUES(?,?)",
              (title or "", time.time()))
