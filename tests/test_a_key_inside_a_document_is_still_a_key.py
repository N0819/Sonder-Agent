"""A credential nested in a JSON document walked straight through redaction.

The query lane has three ways to spot a credential: the COLUMN is named like
one, the ROW KEY is named like one, or the VALUE announces itself. All three
are defeated by the same shape — a settings row whose key is an innocent word
and whose value is a JSON document with the key buried inside it.

`_SECRET_VALUE` is anchored with `^`, because it was written for a cell that
IS a credential. A cell that CONTAINS one starts with `{`, so the anchor never
reaches it. `4144d63` stopped this lane handing back the engine's credentials
and this is the same defect wearing a different shape.

Live, and this is why it was found: `assistant.db` holds exactly one settings
row, keyed `providers`, and its value is a JSON blob carrying live provider
keys. `providers` matches no secret-key pattern; the columns are `key` and
`value`; the cell starts with `{`. Verified against the real row before the
fix: all three layers returned False and a credential-shaped token was
present. Exposing that database read-only — which is the whole point of the
change this test was written for — would have handed the assistant working
credentials through `SELECT * FROM settings`.

The redaction must therefore be about what the cell CONTAINS, not only what it
starts with. `_render` truncates before redaction runs, so the scan has to
happen on the rendered text or a key past the truncation point is invisible —
that is asserted below rather than left to be rediscovered.
"""

from __future__ import annotations

import sqlite3

import pytest

import refdb


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A key/value settings table shaped like the live one."""
    path = tmp_path / "ref.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE settings(key TEXT, value TEXT);"
        "CREATE TABLE notes(id INTEGER, body TEXT);")
    con.execute(
        "INSERT INTO settings(key, value) VALUES (?,?)",
        ("providers", '{"openai": {"api_key": "sk-liveLIVEliveLIVElive0123",'
                      ' "base_url": "https://example.invalid"}}'))
    con.execute(
        "INSERT INTO settings(key, value) VALUES (?,?)",
        ("ui", '{"theme": "dark", "rows": 40}'))
    con.commit()
    con.close()
    # `configure` is the real registration path, so the lane under test is the
    # one that ships. Patching a lookup helper instead is how the first draft
    # of this file "passed" a leak test against a query that never ran.
    monkeypatch.setattr(refdb, "_DATABASES", {"ref": str(path)})
    return path


def _cells(out):
    return [c for row in out.get("rows", []) for c in row]


def test_a_key_buried_in_json_does_not_come_back(db):
    """THE REPRODUCTION. The whole cell goes, because a partial redaction of a
    document would have to parse it, and a parser that fails open leaks."""
    out = refdb.query("ref", "SELECT key, value FROM settings")
    assert out.get("ok") is True, out
    joined = " ".join(_cells(out))
    assert "sk-liveLIVEliveLIVElive0123" not in joined, joined[:400]


def test_it_says_it_redacted_rather_than_going_quiet(db):
    """A blanked cell that does not announce itself reads as a NULL in the
    source table, which is a finding and a wrong one."""
    out = refdb.query("ref", "SELECT key, value FROM settings")
    assert "value" in (out.get("redacted") or []), out.get("redacted")
    assert any("redacted" in c for c in _cells(out))


def test_the_innocent_row_of_the_same_table_survives(db):
    """The point of the lane is reading settings. Redacting the whole table
    because one row holds a key would make the fix worse than the defect."""
    out = refdb.query("ref", "SELECT value FROM settings WHERE key='ui'")
    assert "dark" in " ".join(_cells(out)), out


def test_ordinary_prose_is_not_eaten(db):
    """The scan is unanchored now, so a false positive costs real data. These
    are the near-misses: a word containing the letters, and a bare prefix with
    nothing after it."""
    con = sqlite3.connect(db)
    con.execute("INSERT INTO notes(id, body) VALUES (1, ?)",
                ("the task_key discussion, and sk- on its own, and a "
                 "sentence about AIza as a string",))
    con.commit()
    con.close()
    out = refdb.query("ref", "SELECT body FROM notes")
    assert "discussion" in " ".join(_cells(out)), out
    assert not (out.get("redacted") or []), out.get("redacted")


def test_a_key_past_the_truncation_point_is_still_caught(db):
    """`_render` truncates long cells BEFORE redaction runs. A key sitting
    past `max_cell` is not in the rendered text at all — so it cannot leak,
    and the assertion that matters is that the truncated head is still
    scanned rather than waved through as 'already cut'.
    """
    con = sqlite3.connect(db)
    con.execute("INSERT INTO notes(id, body) VALUES (2, ?)",
                ("x" * 50 + " ghp_ABCDEFGHIJKLMNOPQRST0123 " + "y" * 5000,))
    con.commit()
    con.close()
    out = refdb.query("ref", "SELECT body FROM notes WHERE id=2",
                      max_cell=200)
    assert "ghp_ABCDEFGHIJKLMNOPQRST0123" not in " ".join(_cells(out))
