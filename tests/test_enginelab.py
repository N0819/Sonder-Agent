"""A scratch engine of its own to break.

Reading a broken story out of the engine ends at a theory, and a theory about
interactive fiction is cheap: the defects live in what agents do to each
other's output across a turn, and the only instrument that settles those is
another turn run against the changed code. Until a lab existed, every proposed
repair to the engine could be argued and not observed — which is the failure
this repository names first.

These tests run the real child processes against a STAND-IN engine source, so
the whole mechanism — the generated script, the environment, the sentinel, the
copying of credentials — is exercised offline with no model and no network.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

import enginelab
import refdb

# A stand-in engine: the four db functions the provision and seed children
# call, and nothing else. Small enough to read, real enough that the child is
# not stubbed out — the point is to exercise the plumbing, not to simulate it.
FAKE_DB = '''
import os, sqlite3
_PATH = os.environ.get("ENGINE_DB", "")
def configure(path):
    global _PATH
    _PATH = path
def _conn():
    c = sqlite3.connect(_PATH)
    c.row_factory = sqlite3.Row
    return c
def init():
    c = _conn()
    c.executescript("""
      CREATE TABLE IF NOT EXISTS schema_meta(key TEXT PRIMARY KEY, value TEXT);
      CREATE TABLE IF NOT EXISTS providers(id INTEGER PRIMARY KEY, name TEXT,
        kind TEXT, base_url TEXT, api_key TEXT, enabled INTEGER);
      CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
      CREATE TABLE IF NOT EXISTS personas(id INTEGER PRIMARY KEY, name TEXT,
        sheet TEXT, source TEXT);
      CREATE TABLE IF NOT EXISTS characters(id INTEGER PRIMARY KEY, name TEXT,
        sheet TEXT, source TEXT, created REAL);
      CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY, name TEXT,
        persona_id INT, lorebook_id INT, scenario TEXT, created REAL);
      CREATE TABLE IF NOT EXISTS chat_chars(chat_id INT, char_id INT,
        status TEXT, state TEXT);
      CREATE TABLE IF NOT EXISTS chat_personas(chat_id INT, persona_id INT);
      CREATE TABLE IF NOT EXISTS world(chat_id INT, key TEXT, value TEXT);
      CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, chat_id INT,
        idx INT, player_input TEXT, created REAL, frame_id INT);
      CREATE TABLE IF NOT EXISTS steps(id INTEGER PRIMARY KEY, turn_id INT,
        key TEXT, label TEXT, ord INT, stale INT);
      CREATE TABLE IF NOT EXISTS variants(id INTEGER PRIMARY KEY, step_id INT,
        content TEXT, active INT);
      CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, chat_id INT,
        turn_idx INT);
    """)
    c.execute("INSERT OR REPLACE INTO schema_meta VALUES('version','25')")
    c.commit(); c.close()
def q(sql, args=(), one=False):
    c = _conn(); rows = c.execute(sql, args).fetchall(); c.close()
    return (rows[0] if rows else None) if one else rows
def qi(sql, args=()):
    c = _conn(); cur = c.execute(sql, args); c.commit()
    n = cur.lastrowid; c.close(); return n
def wset(chat_id, key, val):
    import json as _j
    qi("INSERT INTO world(chat_id,key,value) VALUES(?,?,?)",
       (chat_id, key, _j.dumps(val)))
'''

FAKE_CHARACTER_SCHEMA = '''
def normalize_character_data(sheet):
    out = dict(sheet or {})
    out.setdefault("identity", {}).setdefault("name", out.get("name", "?"))
    return out
def character_name(sheet):
    return (sheet.get("identity") or {}).get("name") or sheet.get("name") or "?"
'''


@pytest.fixture
def engine_src(tmp_path):
    """A stand-in engine source tree the children can actually bind."""
    src = tmp_path / "engine_src"
    src.mkdir()
    (src / "db.py").write_text(FAKE_DB)
    (src / "character_schema.py").write_text(FAKE_CHARACTER_SCHEMA)
    return str(src)


@pytest.fixture
def live_engine(tmp_path):
    """A stand-in for the live engine.db a lab copies its configuration from."""
    path = tmp_path / "engine.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE providers(id INTEGER PRIMARY KEY, name TEXT, "
                 "kind TEXT, base_url TEXT, api_key TEXT, enabled INTEGER)")
    conn.execute("INSERT INTO providers VALUES(1,'nanogpt','nanogpt',"
                 "'https://example.invalid/v1','sk-nano-SECRET-VALUE-XYZ',1)")
    conn.execute("CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany("INSERT INTO settings VALUES(?,?)", [
        ("agent_models", '{"director": "some/model"}'),
        ("max_output_tokens", "40000"),
        ("host_pw_hash", "THE-HOST-PASSWORD-HASH"),
        ("host_secret", "THE-HOST-SESSION-SECRET"),
        ("freesound_key", "THE-FREESOUND-KEY")])
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def lab(tmp_path, engine_src, live_engine):
    """A configured lab root, torn down with the tmp tree."""
    enginelab.configure(root=str(tmp_path / "labs"), source=engine_src,
                        engine_db=live_engine)
    yield
    enginelab.configure(root="", source="", engine_db="")
    refdb.configure({})


def test_a_lab_is_built_on_the_engines_own_schema(lab):
    """WRITTEN BY `db.init()`, not by a copy of the schema kept here. A second
    implementation of a schema drifts from the first, and the drift shows up as
    a lab result that cannot be reproduced in the engine it was supposed to be
    testing.
    """
    result = enginelab.provision("one")
    assert result["ok"], result.get("error")
    assert result["schema_version"] == "25"
    assert result["tables"] >= 12


def test_the_api_key_is_copied_into_the_lab_and_never_returned(lab):
    """A lab needs real credentials to reach a model, and everything a lane
    returns gets written down — an evidence excerpt, a turn trace,
    `assistant.db`, a public repository. So the child writes the key and
    returns a count, and the parent never holds the string at all.
    """
    result = enginelab.provision("one")
    assert "sk-nano-SECRET-VALUE-XYZ" not in json.dumps(result)
    provider = result["providers"][0]
    assert provider["key_present"] is True
    assert provider["key_chars"] == len("sk-nano-SECRET-VALUE-XYZ")
    # ...and it really is in the lab, or nothing would run.
    conn = sqlite3.connect(result["db"])
    assert conn.execute("SELECT api_key FROM providers").fetchone()[0] == \
        "sk-nano-SECRET-VALUE-XYZ"
    conn.close()


def test_only_the_settings_a_story_needs_are_copied(lab):
    """AN ALLOWLIST, because a denylist silently admits every setting added
    later and the one added later is the one nobody audits. The host password
    hash and the session secret have nothing to do with running a story.
    """
    result = enginelab.provision("one")
    assert "agent_models" in result["settings_copied"]
    assert "max_output_tokens" in result["settings_copied"]
    conn = sqlite3.connect(result["db"])
    copied = {k for (k,) in conn.execute("SELECT key FROM settings")}
    conn.close()
    assert "host_pw_hash" not in copied
    assert "host_secret" not in copied
    assert "freesound_key" not in copied


def test_reading_the_lab_back_redacts_the_key_it_was_given(lab):
    """The lab database has a real credential in it by design, and the read
    lane is the same one the engine is read through — so the redaction that
    protects the engine protects this too, without being remembered twice.
    """
    enginelab.provision("one")
    result = enginelab.lab_query("one", "SELECT name, api_key FROM providers")
    assert result["ok"]
    assert "sk-nano-SECRET-VALUE-XYZ" not in json.dumps(result)
    assert result["redacted"] == ["api_key"]


def test_provisioning_over_an_existing_lab_is_refused_not_merged(lab):
    """A half-reseeded database produces a result nobody can attribute: the
    story is partly the old one and the turns are partly the new one.
    """
    assert enginelab.provision("one")["ok"]
    again = enginelab.provision("one")
    assert again["ok"] is False and "reset=True" in again["error"]
    assert enginelab.provision("one", reset=True)["ok"]


def test_a_lab_name_cannot_walk_out_of_the_lab_root(lab):
    """The name becomes a directory. Anything that is not a plain name is a
    path escape, and it is refused before anything is created.
    """
    bad = enginelab.provision("../../etc")
    assert bad["ok"] is False and "path escape" in bad["error"]
    assert enginelab.destroy("..")["ok"] is False


def test_a_story_is_seeded_through_the_engines_own_normaliser(lab):
    """A sheet this accepts must be one the engine accepts. A second
    normaliser here would be a second thing to keep in step, and the failure
    would land inside an agent halfway through a turn.
    """
    enginelab.provision("one")
    result = enginelab.seed("one", {
        "name": "the lighthouse", "scenario": "The lamp room, before dawn.",
        "persona": {"name": "Ilse"},
        "characters": [{"name": "Maren Holt"}],
        "world": {"style_guide": {"tone": "plain"}}})
    assert result["ok"], result.get("error")
    assert result["cast"] == ["Maren Holt"]
    assert result["chat_id"] == 1
    assert enginelab.labs()[0]["story"] == "the lighthouse"


def test_a_story_with_no_scenario_is_refused_with_the_reason(lab):
    """The scenario is the opening situation every agent reads. A story
    without one runs, and produces a turn about nothing.
    """
    enginelab.provision("one")
    bad = enginelab.seed("one", {"name": "x", "characters": []})
    assert bad["ok"] is False and "scenario" in bad["error"]


def test_playing_into_a_lab_with_no_story_says_so(lab):
    """Otherwise the child starts, fails on a missing chat id somewhere inside
    the engine, and reports it as a pipeline defect.
    """
    enginelab.provision("one")
    bad = enginelab.play("one", "hello")
    assert bad["ok"] is False and "seed one first" in bad["error"]


def test_an_unprovisioned_lab_is_named_rather_than_created(lab):
    """A verb that silently provisions would make a typo into a second lab,
    and the second lab's fresh story would be attributed to the first lab's
    edit.
    """
    for call in (lambda: enginelab.seed("ghost", {"scenario": "x"}),
                 lambda: enginelab.play("ghost", "x"),
                 lambda: enginelab.lab_query("ghost", "SELECT 1")):
        result = call()
        assert result["ok"] is False
        assert "not provisioned" in result["error"]


def test_a_child_that_dies_before_writing_a_result_is_a_failure(lab, tmp_path):
    """A RUN WITH NO RESULT AND NO PROCESS IS A FAILURE, reported as one with
    the tail of its log. The alternative reads as pending forever, which is how
    a crashed child gets mistaken for a slow one.
    """
    broken = tmp_path / "broken_src"
    broken.mkdir()
    (broken / "db.py").write_text("raise RuntimeError('the engine is broken')")
    result = enginelab.provision("two", source=str(broken))
    assert result["ok"] is False
    assert "the engine is broken" in (result.get("traceback") or "") + \
        (result.get("log_tail") or "")


def test_a_source_tree_with_no_engine_in_it_is_named_before_launch(lab,
                                                                   tmp_path):
    """Launching anyway means a traceback about an import, several layers down
    from the thing that was actually wrong.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    result = enginelab.provision("three", source=str(empty))
    assert result["ok"] is False and "no db.py there" in result["error"]


def test_labs_survive_the_turn_that_made_them(lab):
    """A lab is on disk precisely so a run started in one turn can be read in
    the next. A listing that forgot them would have the assistant provision a
    second lab and attribute its fresh story to the first one's edit.
    """
    enginelab.provision("one")
    enginelab.seed("one", {"scenario": "The lamp room.",
                           "characters": [{"name": "Maren Holt"}]})
    enginelab.configure(root=enginelab._ROOT, source=enginelab._SOURCE,
                        engine_db=enginelab._ENGINE_DB)
    listed = enginelab.labs()
    assert [entry["name"] for entry in listed] == ["one"]
    assert listed[0]["provisioned"] is True
    assert listed[0]["chat_id"] == 1
    # COUNTED FROM THE DATABASE. The counter this used to report only advanced
    # on a blocking run, so a lab with two detached turns in it read as zero —
    # and "no turns yet" is what makes an assistant re-seed a story it has
    # already played.
    assert listed[0]["turns_played"] == 0


def test_a_finished_run_reports_done_and_carries_its_result(lab):
    """`runs` is the only way a detached turn is ever read. If it cannot tell
    finished from running, the whole asynchronous shape is unusable.
    """
    enginelab.provision("one")
    report = enginelab.runs("one")
    assert report["ok"] and report["runs"]
    first = report["runs"][0]
    assert first["state"] == "done"
    assert first["schema_version"] == "25"


def test_destroying_a_lab_that_is_not_there_says_so(lab):
    """Silently succeeding would let a typo read as a cleanup that happened."""
    assert enginelab.destroy("nope")["ok"] is False
    enginelab.provision("one")
    assert enginelab.destroy("one")["ok"] is True
    assert enginelab.labs() == []


def test_a_run_in_flight_is_visible_in_the_listing(lab, monkeypatch):
    """RUNS OUTLIVE THE TURN THAT STARTED THEM — that is the whole point of
    detaching. Without this the listing shows a lab that looks idle while a
    turn is halfway through writing it, and the obvious next move, starting
    another, is the one thing that makes the trace impossible to untangle.
    """
    enginelab.provision("one")
    assert "running" not in enginelab.labs()[0]
    runs_dir = os.path.join(enginelab._lab_dir("one"), "runs")
    with open(os.path.join(runs_dir, "9999.pid"), "w") as fh:
        fh.write(str(os.getpid()))  # a pid that is certainly alive
    listed = enginelab.labs()[0]
    assert listed["running"]["run"] == "9999"
    assert listed["running"]["pid"] == os.getpid()


def test_a_reroll_needs_one_kind_of_rerun_not_both(lab):
    """`from_key` recomputes the tail; `only_key` recomputes one step and
    leaves the tail stale. They end in different states, so a call naming both
    has not said which experiment it is running.
    """
    enginelab.provision("one")
    enginelab.seed("one", {"scenario": "The lamp room.",
                           "characters": [{"name": "Maren Holt"}]})
    bad = enginelab.reroll("one", 0, from_key="narrator", only_key="narrator")
    assert bad["ok"] is False and "Pick one" in bad["error"]


def test_a_reroll_of_a_turn_that_was_never_played_says_so(lab):
    """Otherwise the failure lands inside the engine as a null turn id, which
    reads as a pipeline defect rather than as a wrong turn number.
    """
    enginelab.provision("one")
    enginelab.seed("one", {"scenario": "The lamp room.",
                           "characters": [{"name": "Maren Holt"}]})
    result = enginelab.reroll("one", 7, only_key="narrator", wait=True,
                              timeout=60)
    assert result["ok"] is False
    assert "no turn with idx 7" in (result.get("error") or "") + \
        (result.get("traceback") or "")
