"""A story reaches its lore two ways, and reading one answers wrong.

`chats.lorebook_id` is a single column and `chat_lorebooks` is a link table.
Both are live, and they routinely disagree. Live: chat 63's column points at
`The Doctor — Hinami — canon`, seven entries the engine minted during play,
while the authored `Tamamo's Shrine` book its entire story rests on — fifteen
entries describing a genkan, two staircases, a second-floor hallway, a
basement and a seal cavern — arrives only through the link table.

An investigator reading the column alone concludes the story has no layout
lore. That is false, and it looks like a finding rather than a missed join.

There was no verb for any of this. Stories, turns, memories and schema each
had one; lore had to be reached by hand-written SQL across a join nobody
remembers, which is why it kept not being reached at all.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import storydb


@pytest.fixture
def lore_db(tmp_path, monkeypatch):
    """The live shape in miniature: one book on each route, disagreeing."""
    path = tmp_path / "engine.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE chats(id INTEGER PRIMARY KEY, name TEXT,
                           lorebook_id INTEGER);
        CREATE TABLE lorebooks(id INTEGER PRIMARY KEY, name TEXT,
                              book_type TEXT);
        CREATE TABLE chat_lorebooks(chat_id INTEGER, lorebook_id INTEGER,
                                    enabled INTEGER);
        CREATE TABLE lore_entries(id INTEGER PRIMARY KEY, lorebook_id INTEGER,
            keys TEXT, content TEXT, category TEXT, canon_locked INTEGER,
            turn_added INTEGER, title TEXT, aliases TEXT, importance REAL,
            scope TEXT, relations TEXT, knowledge_tag TEXT,
            embedding_model TEXT);
        INSERT INTO chats VALUES (63, 'The Doctor', 184);
        INSERT INTO lorebooks VALUES (184, 'engine leavings', 'general');
        INSERT INTO lorebooks VALUES (187, 'Tamamo shrine', 'general');
        INSERT INTO chat_lorebooks VALUES (63, 187, 1);
        INSERT INTO lore_entries VALUES
            (2768, 184, 'spanner', 'a spanner', 'other', 1, 67,
             NULL, '[]', 0.5, NULL, NULL, NULL, NULL),
            (2728, 187, 'second floor,hallway', 'A narrow wooden staircase rises from the back of the first floor.', 'layout', 0, NULL,
             'Second Floor', '[]', 0.5, NULL, NULL, NULL, NULL),
            (2734, 187, 'upstairs', 'Upstairs Resting Area', 'layout', 0, 130,
             'Main Hall and Upstairs Resting Area', '[]', 0.5,
             NULL, NULL, NULL, NULL);
    """)
    con.commit()
    con.close()
    monkeypatch.setattr(storydb, "_rows", _reader(path))
    return path


def _reader(path):
    def _rows(sql, database, max_cell=None, **kw):
        con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = [{k: (str(r[k]) if r[k] is not None else None)
                     for k in r.keys()} for r in con.execute(sql)]
        finally:
            con.close()
        return rows, {}
    return _rows


def test_both_routes_are_resolved(lore_db):
    """THE REPRODUCTION. Reading `chats.lorebook_id` alone finds the engine's
    leavings and misses the entire authored world.
    """
    out = storydb.story_lorebooks(63)
    routes = {b["book_id"]: b["route"] for b in out["books"]}
    assert routes == {"184": "chats.lorebook_id", "187": "chat_lorebooks"}


def test_each_book_says_how_much_of_it_the_engine_wrote(lore_db):
    """`written_in_play` beside `entries` is the number that turns a book
    listing into a diagnosis.
    """
    books = {b["book_id"]: b for b in storydb.story_lorebooks(63)["books"]}
    assert books["184"]["written_in_play"] == "1"
    assert books["187"]["entries"] == "2"
    assert books["187"]["written_in_play"] == "1"


def test_entries_reach_across_both_routes_from_a_chat_id(lore_db):
    """The convenience the missing verb cost: one chat_id, every entry the
    story can retrieve, without anyone reconstructing the union.
    """
    ids = {e["id"] for e in storydb.lore_entries(chat_id=63)["entries"]}
    assert ids == {"2768", "2728", "2734"}


def test_turn_added_separates_authored_from_minted(lore_db):
    """The column the whole layout diagnosis turned on: entry 2734, written by
    the engine at turn 130, contradicting the authored geometry in the same
    book with neither locked.
    """
    entries = {e["id"]: e for e in storydb.lore_entries(book_id=187)["entries"]}
    assert entries["2728"]["turn_added"] is None      # authored
    assert entries["2734"]["turn_added"] == "130"     # minted in play


def test_listing_entries_withholds_content_and_the_entry_verb_returns_it(lore_db):
    """A book runs to thousands of characters an entry and the question is
    almost always WHICH entry. Content comes back one at a time, on request.
    """
    listed = storydb.lore_entries(book_id=187)["entries"][0]
    assert "content" not in listed and "content_chars" in listed
    whole = storydb.lore_entry(2728)["entry"]
    assert "narrow wooden staircase" in whole["content"]


def test_text_search_reaches_keys_title_and_content(lore_db):
    assert storydb.lore_entries(text="staircase")["returned"] == 1
    assert storydb.lore_entries(text="hallway")["returned"] == 1


def test_a_call_with_no_selector_refuses(lore_db):
    """Returning the whole corpus for an empty call is how a verb becomes
    unusable on a real database.
    """
    assert storydb.lore_entries()["ok"] is False
