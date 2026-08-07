"""A turn in flight was counted as a turn played.

`_turns_in` answered `SELECT COUNT(*) FROM turns`, and the engine creates the
turn row at the START of a turn. So the lab listing counted the turn currently
executing. The caller is a status field, and the whole question a status field
answers is what has FINISHED.

Live: lab `stairs` reported `turns_played: 2` while turn 2 was mid-flight with
four steps written — director_interpret, mapping_stage, perception_act,
interaction_loop — and no narrator and no commit. An assistant polling that
number to decide whether to act would read a turn in progress as a turn done.

Same shape as the zombie-poll defect one lane over: a status field answering a
question adjacent to the one that was asked, and confidently.
"""

from __future__ import annotations

import sqlite3

import enginelab


def _lab(tmp_path, turns):
    """`turns` is a list of step-key lists, one per turn."""
    path = tmp_path / "engine.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE turns(id INTEGER PRIMARY KEY, idx INTEGER);
        CREATE TABLE steps(id INTEGER PRIMARY KEY, turn_id INTEGER, key TEXT);
    """)
    for i, keys in enumerate(turns, start=1):
        con.execute("INSERT INTO turns(id, idx) VALUES (?,?)", (i, i))
        for k in keys:
            con.execute("INSERT INTO steps(turn_id, key) VALUES (?,?)", (i, k))
    con.commit()
    con.close()
    return str(path)


def test_a_turn_mid_flight_is_not_played(tmp_path):
    """THE REPRODUCTION, with the live step list."""
    path = _lab(tmp_path, [
        ["director_interpret", "mapping_stage", "narrator", "commit"],
        ["director_interpret", "mapping_stage", "perception_act",
         "interaction_loop"],
    ])
    assert enginelab._turns_in(path) == 1


def test_finished_turns_are_counted(tmp_path):
    path = _lab(tmp_path, [["narrator", "commit"], ["narrator", "commit"]])
    assert enginelab._turns_in(path) == 2


def test_a_lab_that_has_played_nothing_is_zero(tmp_path):
    assert enginelab._turns_in(_lab(tmp_path, [])) == 0
    assert enginelab._turns_in(str(tmp_path / "absent.db")) == 0


def test_a_lab_with_no_steps_table_still_answers(tmp_path):
    """An older or partly-migrated lab has no `steps`. The row count is then
    the best available answer, and returning it beats reporting zero turns for
    a lab that has plainly played some.
    """
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE turns(id INTEGER PRIMARY KEY, idx INTEGER);"
                      "INSERT INTO turns(id, idx) VALUES (1,1),(2,2);")
    con.commit()
    con.close()
    assert enginelab._turns_in(str(path)) == 2


def test_a_lab_with_no_turns_table_is_zero_not_an_error(tmp_path):
    """Provisioned but never migrated. A listing must not fail whole because
    one lab is empty.
    """
    path = tmp_path / "bare.db"
    sqlite3.connect(path).close()
    assert enginelab._turns_in(str(path)) == 0
