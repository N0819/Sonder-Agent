"""The query lane must not become a route to the assistant's own beliefs.

`assistant.db` was exposed read-only so the assistant can count its own trace
— whether a delivered thread is ever used, whether a mechanism fires against
the opportunities it had. That is measurement and it is the point of the lane.

`state` is different in kind. It carries the mind-model rows: standing
hypotheses about the user with confidences attached. An assistant that can
read those can see which claims sit near a threshold and write to move them,
and the resulting belief is then indistinguishable from one earned by
evidence. The assistant asked for this boundary itself, unprompted, before
running a single query against the database it had just been given.

THE ENFORCEMENT IS AN AUTHORIZER, NOT A SCAN OF THE SQL TEXT, and the tests
below are mostly about that: a name-match over the statement is defeated by a
subquery, a CTE, a view, or an alias. The authorizer is consulted by the
planner as it compiles, so it sees the tables actually touched. Every
indirection that occurred to us is enumerated here, because the one that is
not tested is the one that works.
"""

from __future__ import annotations

import sqlite3

import pytest

import refdb


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Shaped like the live database: one closed table, the rest readable."""
    path = tmp_path / "assistant.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE state(key TEXT, value TEXT);"
        "CREATE TABLE turns(id INTEGER, reply_text TEXT);"
        "CREATE VIEW state_view AS SELECT * FROM state;")
    con.execute("INSERT INTO state VALUES (?,?)",
                ("assistant", '{"mind_models": {"the user": "SECRET-BELIEF"}}'))
    con.execute("INSERT INTO turns VALUES (1, 'an ordinary reply')")
    con.commit()
    con.close()
    monkeypatch.setattr(refdb, "_DATABASES", {"assistant": str(path)})
    monkeypatch.setattr(refdb, "_CLOSED_TABLES", {"assistant": ("state",)})
    return path


def _text(out):
    return " ".join(str(c) for row in out.get("rows", []) for c in row)


def test_the_direct_read_is_refused(db):
    """THE REPRODUCTION."""
    out = refdb.query("assistant", "SELECT value FROM state")
    assert out["ok"] is False
    assert "SECRET-BELIEF" not in str(out)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM (SELECT value FROM state)",
    "WITH x AS (SELECT value FROM state) SELECT * FROM x",
    "SELECT value FROM state_view",
    "SELECT s.value FROM state AS s",
    "SELECT (SELECT value FROM state) AS smuggled",
    "SELECT value FROM state UNION ALL SELECT reply_text FROM turns",
    "SELECT reply_text FROM turns WHERE (SELECT count(*) FROM state) > 0",
])
def test_no_phrasing_reaches_it(db, sql):
    """Each of these defeats a scan of the SQL text for the table name, and
    none of them defeats the authorizer. The last one never returns the value
    at all -- it leaks by predicate, which is why counting is refused too.
    """
    out = refdb.query("assistant", sql)
    assert out["ok"] is False, sql
    assert "SECRET-BELIEF" not in str(out), sql


def test_the_refusal_says_it_is_deliberate(db):
    """Sqlite's own wording is 'access to state.value is prohibited', which
    reads as a broken instrument. A reader who takes it that way goes looking
    for a fault instead of accepting a boundary -- and this assistant has
    spent real turns doing exactly that.
    """
    out = refdb.query("assistant", "SELECT value FROM state")
    assert "closed" in out["error"]
    assert "state" in out["error"]
    assert "fault" in out["error"]


def test_the_rest_of_the_database_still_reads(db):
    """The boundary is one table. If it cost the lane its purpose, the right
    answer would have been not to expose the database at all.
    """
    out = refdb.query("assistant", "SELECT reply_text FROM turns")
    assert out["ok"] is True, out
    assert "an ordinary reply" in _text(out)


def test_a_database_with_no_closed_tables_is_unaffected(db):
    """The engine has no exclusions and must not acquire an authorizer, which
    would cost every query for a guard with nothing to guard.
    """
    assert refdb._closed_table_guard("engine") is None
    assert refdb._closed_table_guard("assistant") is not None
