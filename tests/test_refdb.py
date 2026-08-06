"""The read-only lane into databases too large to copy.

Every other lane into data assumes the data can be MOVED — the sandbox
snapshot builds a payload, `read_file` returns a file, the chunk index stores
bodies. A 1.1 GB reference database is 2.1x the whole workspace ceiling and
550x the per-file snapshot limit, so it fits through none of them.
"""

from __future__ import annotations

import sqlite3

import pytest

import refdb


@pytest.fixture
def ref(tmp_path):
    """A small stand-in for a database that is too big to move."""
    path = tmp_path / "engine.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("CREATE TABLE blobs(id INTEGER PRIMARY KEY, payload BLOB)")
    conn.executemany("INSERT INTO turns(body) VALUES(?)",
                     [(f"turn {i}",) for i in range(500)])
    conn.execute("INSERT INTO blobs(payload) VALUES(?)", (b"\x00" * 4096,))
    conn.commit()
    conn.close()
    refdb.configure({"engine": str(path)})
    yield path
    refdb.configure({})


def test_a_database_too_large_to_copy_is_still_queryable(ref):
    """THE WHOLE POINT. `engine.db` is 1,118,785,536 bytes against a 512 MiB
    workspace ceiling and a 2 MiB per-file snapshot limit, so no lane that
    copies bytes can reach it. The size limits bind on what a SANDBOX sees,
    because a run gets its own copy; a fetch verb copies nothing."""
    out = refdb.query("engine", "SELECT COUNT(*) AS n FROM turns")
    assert out["ok"], out
    assert out["columns"] == ["n"]
    assert out["rows"] == [["500"]]


def test_a_write_is_refused_by_sqlite_and_not_by_a_string_check(ref):
    """A read-only lane defended only by parsing the statement is a lane whose
    safety depends on the parser being exhaustive. The connection is opened
    `mode=ro`, so the refusal comes from SQLite itself and holds for anything
    the statement gate has not thought of."""
    out = refdb.query("engine", "UPDATE turns SET body='x'")
    assert not out["ok"]
    # Refused before it ran, with a message naming the caller's mistake.
    assert "not a read" in out["error"], out["error"]

    # And the underlying connection would refuse it even so.
    conn = sqlite3.connect(f"file:{ref}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("UPDATE turns SET body='x'")
    conn.close()


def test_a_second_statement_cannot_ride_along(ref):
    """Only the last result of a batch would come back, so a query whose
    interesting half is the first statement returns someone else's answer with
    nothing saying so."""
    out = refdb.query("engine", "SELECT 1; SELECT 2")
    assert not out["ok"]
    assert "one statement" in out["error"]


def test_a_trailing_semicolon_is_not_a_second_statement(ref):
    """Rejecting ordinary SQL punctuation would make the gate a thing to be
    worked around rather than a thing that catches mistakes."""
    out = refdb.query("engine", "SELECT COUNT(*) FROM turns;")
    assert out["ok"], out


def test_the_row_cap_says_when_it_bit(ref):
    """A result set silently cut is a census that reads as complete. This
    repository has already paid for that once — a `skipped` list capped at 40
    beside a true `skipped_count` — and a truncated query result is the same
    defect with a table behind it."""
    out = refdb.query("engine", "SELECT id FROM turns", max_rows=10)
    assert out["ok"], out
    assert out["row_count"] == 10
    assert out["truncated"] is True
    assert "10-row cap" in out["why"]


def test_an_untruncated_result_does_not_claim_to_be_truncated(ref):
    """The flag has to mean something. If it were set whenever a cap existed,
    every answer would read as a sample and the real samples would not stand
    out."""
    out = refdb.query("engine", "SELECT id FROM turns LIMIT 3")
    assert out["ok"] and out["row_count"] == 3
    assert out["truncated"] is False and out["why"] == ""


def test_a_blob_is_named_rather_than_rendered(ref):
    """A 4 KB blob rendered as text is 4 KB of the character budget spent on
    bytes nobody can read, and a large one would eat the whole result."""
    out = refdb.query("engine", "SELECT payload FROM blobs")
    assert out["ok"], out
    assert out["rows"] == [["<4096 bytes blob>"]]


def test_a_query_that_runs_too_long_is_stopped_and_says_why(ref):
    """One careless scan of a 699 MiB table holds the turn open for minutes,
    and the assistant cannot know a table's size before it asks. `interrupted`
    on its own reads as a harness fault rather than as the query having been
    too expensive."""
    out = refdb.query(
        "engine",
        "WITH RECURSIVE spin(n) AS (SELECT 1 UNION ALL "
        "SELECT n+1 FROM spin WHERE n < 100000000) SELECT COUNT(*) FROM spin",
        time_limit=0.25)
    assert not out["ok"]
    assert "ran past" in out["error"], out["error"]


def test_an_unknown_database_names_the_ones_that_exist(ref):
    """"no such database" leaves the author guessing at their own typo when
    the registry is one line long and could simply be shown."""
    out = refdb.query("nope", "SELECT 1")
    assert not out["ok"]
    assert "engine" in out["error"]


def test_a_registered_database_whose_file_vanished_looks_absent(tmp_path):
    """A registry entry pointing at a moved file reads exactly like one whose
    file is there, and the failure surfaces later as an empty result that
    looks like a finding."""
    refdb.configure({"gone": str(tmp_path / "not_here.db")})
    try:
        listed = refdb.databases()
        assert listed[0]["ok"] is False and "unreadable" in listed[0]["error"]
        out = refdb.query("gone", "SELECT 1")
        assert not out["ok"] and "not there" in out["error"]
    finally:
        refdb.configure({})
