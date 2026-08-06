"""Reading a story out of a live engine database by the name a person uses.

A bug report arrives as "the Blizzard story broke around turn 40" — a name and
a number counted off a screen. Everything needed was already reachable through
`query_db`, and the SQL was never the hard part: the chat is keyed by an
integer nobody knows, turns are numbered per chat, and what the agents actually
said is a join past that. Every investigation began by rediscovering the
schema, and the rediscovery cost more rounds than the defect.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import refdb
import storydb


@pytest.fixture
def engine(tmp_path):
    """A miniature engine database with the tables a story lives in."""
    path = tmp_path / "engine.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE chats(id INTEGER PRIMARY KEY, name TEXT, persona_id INT,
                           lorebook_id INT, scenario TEXT, created REAL,
                           branched_from TEXT);
        CREATE TABLE turns(id INTEGER PRIMARY KEY, chat_id INT, idx INT,
                           player_input TEXT, created REAL, frame_id INT);
        CREATE TABLE steps(id INTEGER PRIMARY KEY, turn_id INT, key TEXT,
                           label TEXT, ord INT, stale INT DEFAULT 0);
        CREATE TABLE variants(id INTEGER PRIMARY KEY, step_id INT,
                              content TEXT, created REAL, active INT,
                              reasoning TEXT);
        CREATE TABLE memories(id INTEGER PRIMARY KEY, chat_id INT, char_id INT,
                              turn_idx INT, kind TEXT, category TEXT,
                              provenance TEXT, salience REAL, content TEXT,
                              gist TEXT, importance REAL, confidence REAL,
                              disputed INT, archived INT);
        CREATE TABLE characters(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE chat_chars(chat_id INT, char_id INT, status TEXT);
    """)
    # One story and two branches of it, named the way this engine names
    # branches: by suffixing the parent.
    conn.executemany(
        "INSERT INTO chats(id,name,scenario,created,branched_from) "
        "VALUES(?,?,?,?,?)",
        [(46, "The Blizzard", "Snow, and a door that will not shut.", 1.0, "[]"),
         (56, "Run!", "An alley, and something metal behind you.", 2.0, "[]"),
         (57, "Run! ⎇10", "An alley.", 3.0, "[56]"),
         (58, "Run! ⎇10 ⎇20", "An alley.", 4.0, "[57, 56]")])
    conn.execute("INSERT INTO characters(id,name) VALUES(9,'Maren Holt')")
    conn.execute("INSERT INTO chat_chars(chat_id,char_id,status) "
                 "VALUES(46,9,'active')")
    for idx in range(5):
        tid = 700 + idx
        conn.execute("INSERT INTO turns(id,chat_id,idx,player_input,created) "
                     "VALUES(?,?,?,?,?)",
                     (tid, 46, idx, f"the player types {idx}", 10.0 + idx))
        for ord_, key in enumerate(("director", "perception", "narrator")):
            sid = tid * 10 + ord_
            # Turn 3's perception is marked stale — the defect this fixture
            # exists to make findable.
            stale = 1 if (idx == 3 and key == "perception") else 0
            conn.execute("INSERT INTO steps(id,turn_id,key,label,ord,stale) "
                         "VALUES(?,?,?,?,?,?)",
                         (sid, tid, key, key.title(), ord_, stale))
            if not (idx == 4 and key == "narrator"):
                conn.execute("INSERT INTO variants(step_id,content,active) "
                             "VALUES(?,?,1)",
                             (sid, json.dumps({"prose": f"{key} said {idx}"})))
    conn.executemany(
        "INSERT INTO memories(chat_id,char_id,turn_idx,kind,provenance,"
        "salience,gist,content) VALUES(?,?,?,?,?,?,?,?)",
        [(46, 9, 2, "episodic", "witnessed", 0.8, "the door blew open", "..."),
         (46, 9, 3, "inference", "inferred", 0.4, "she is cold", "...")])
    conn.commit()
    conn.close()
    refdb.configure({"engine": str(path)})
    yield path
    refdb.configure({})


def test_a_story_is_found_by_the_name_a_person_says(engine):
    """The report never carries the integer. If the name cannot be resolved
    here it gets resolved by guessing, and a guessed chat id is a whole
    investigation aimed at another transcript.
    """
    found = storydb.find_story("blizzard")
    assert found["ok"] and found["match_count"] == 1
    assert found["matches"][0]["chat_id"] == "46"
    assert found["ambiguous"] is False


def test_an_ambiguous_name_is_not_resolved_to_one_story(engine):
    """BRANCHES ARE NAMED BY SUFFIXING THE PARENT, so the name a person says
    is routinely a prefix of three real chats and the one they mean is usually
    not the one a LIKE ranks first. Picking silently produces a theory that is
    internally consistent, checkable against real turns, and about another
    story — and nothing downstream can tell.
    """
    found = storydb.find_story("Run!")
    assert found["match_count"] == 3
    assert found["ambiguous"] is True
    assert "settled" in found["note"]
    assert {m["chat_id"] for m in found["matches"]} == {"56", "57", "58"}


def test_a_name_that_matches_nothing_says_what_does_exist(engine):
    """"No story called that" and "I spelled it differently from the person
    who named it" look identical, and the second is far more common. An empty
    list ends the round; a list of what exists turns the next one into a
    choice.
    """
    found = storydb.find_story("the lighthouse")
    assert found["ok"] and found["matches"] == []
    assert found["nothing_matched"] is True
    assert any(s["name"] == "The Blizzard" for s in found["recent_stories"])


def test_a_turn_range_reports_which_agents_ran_and_which_went_stale(engine):
    """A step that never ran and a step whose output was superseded are
    different defects needing different fixes, and neither is visible in the
    prose — which is the only thing the person reporting the bug saw.
    """
    census = storydb.story_turns(46, 2, 4)
    assert census["returned"] == 3
    stale = [t for t in census["turns"] if t["stale_steps"]]
    assert [t["idx"] for t in stale] == ["3"]
    assert "perception" in census["turns"][0]["step_keys"]


def test_an_empty_range_says_what_the_story_actually_spans(engine):
    """An empty list here reads as "those turns went wrong in a way that left
    no record", which is a far more alarming finding than "off by one" — and
    it is the wrong one almost every time.
    """
    census = storydb.story_turns(46, 90, 99)
    assert census["turns"] == [] and census["nothing_in_range"] is True
    assert census["story_span"]["last_idx"] == "4"


def test_one_turn_gives_every_agent_output_in_order(engine):
    """THE PROSE IS THE LAST STAGE WHERE THE DEFECT BECAME VISIBLE. The engine
    runs a turn as a chain, each agent reading the one before, so a
    contradiction in the narration was usually decided several steps upstream.
    Reading only the narrator is reading the symptom.
    """
    detail = storydb.turn_detail(46, 2)
    assert [s["step"] for s in detail["steps"]] == ["director", "perception",
                                                    "narrator"]
    assert json.loads(detail["steps"][0]["content"])["prose"] == "director said 2"
    assert detail["player_input"] == "the player types 2"


def test_a_step_that_stored_nothing_is_not_a_step_that_never_ran(engine):
    """Both come back without content and they are different failures. One is
    an agent that produced nothing; the other is a chain that stopped early.
    """
    detail = storydb.turn_detail(46, 4)
    narrator = [s for s in detail["steps"] if s["step"] == "narrator"][0]
    assert narrator["no_active_variant"] is True
    assert len(detail["steps"]) == 3


def test_a_missing_turn_does_not_look_like_an_empty_one(engine):
    """An idx outside the story and a turn that died before its first agent
    wrote anything both return no steps. The note is the only thing that
    separates them.
    """
    detail = storydb.turn_detail(46, 77)
    assert detail["steps"] == [] and detail["nothing_found"] is True


def test_a_chat_id_that_is_not_an_integer_is_refused_with_the_cure(engine):
    """The lane is reached with a name in hand, so the wrong argument here is
    the name — and the fix is one verb away.
    """
    bad = storydb.story_turns("The Blizzard")
    assert bad["ok"] is False and "find_story" in bad["error"]


def test_memories_carry_the_provenance_that_makes_them_judgeable(engine):
    """A memory written with the wrong provenance reads as nothing on the turn
    it was written and as a continuity break two beats later. Without the
    provenance column the whole class is invisible.
    """
    mem = storydb.story_memories(46, 2, 3)
    assert mem["returned"] == 2
    assert {m["provenance"] for m in mem["memories"]} == {"witnessed",
                                                          "inferred"}
    assert mem["memories"][0]["character"] == "Maren Holt"


def test_the_overview_shows_every_descendant_of_a_branched_story(engine):
    """Repairing a story that has been branched, without knowing it was, means
    the person testing the fix is reading a different chat.

    `branched_from` holds the whole ANCESTRY as a JSON list — '[57, 56]' for a
    branch of a branch — so comparing it to an integer matched nothing and
    reported every story in the engine as unbranched.
    """
    over = storydb.story_overview(56)
    assert over["exists"] is True
    assert [b["chat_id"] for b in over["descendants"]] == ["57", "58"]


def test_an_unknown_chat_id_does_not_come_back_as_an_empty_story(engine):
    """A guessed integer that hits nothing must not read as a story with no
    turns — that is a finding, and it is false.
    """
    over = storydb.story_overview(999)
    assert over["exists"] is False and "find_story" in over["note"]


def test_a_quote_in_a_story_name_cannot_end_the_statement(engine):
    """The lane composes SQL because `refdb.query` takes a statement and
    nothing else. A name is user text and arrives with apostrophes in it.
    """
    found = storydb.find_story("O'Brien'; DROP TABLE chats--")
    assert found["ok"] is True and found["matches"] == []
    assert refdb.query("engine", "SELECT COUNT(*) FROM chats")["rows"][0][0] == "4"
