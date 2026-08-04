# Retiring memories: setting something aside without pretending it never was.
#
# The need is real and it is not about truth — a long-memory assistant working
# on a project accumulates superseded context that competes for recall against
# what is current. The risk is equally real: this is the ONE path reachable
# from model output that removes information rather than annotating it, and
# every other mechanism here exists to stop exactly that. These tests are the
# line between the two.

import json

import pytest

import db
import memory
import pipeline
import providers


@pytest.fixture
def stub_model():
    yield
    providers.set_chat_stub(None)


def _mint(key, content, **kw):
    return memory.add_memory(kw.pop("kind", "semantic"),
                             kw.pop("provenance", "told"), 0.7, content,
                             turn_idx=kw.pop("turn_idx", 1), event_key=key,
                             **kw)


def test_a_retired_memory_leaves_recall_entirely(temp_db):
    """This is the whole point. `archived` does not help — an archived row has
    only left the rolling consolidation window and is still fully recallable,
    deliberately. Superseded project context has to stop competing."""
    _mint("old:1", "the postgres schema uses a jsonb column for settings")
    _mint("new:1", "the sqlite schema uses a settings table")
    before = memory.search_memories("schema settings", current_turn_idx=10)
    assert {m["event_key"] for m in before} == {"old:1", "new:1"}

    memory.retire_memories(["old:1"], reason="we moved to SQLite",
                           turn_idx=5)
    after = memory.search_memories("schema settings", current_turn_idx=10)
    assert {m["event_key"] for m in after} == {"new:1"}


def test_a_retired_memory_still_exists_and_comes_back(temp_db):
    """Reversible by construction. "Irrelevant to the current iteration" is a
    claim about SCOPE, and scope changes — the Postgres decision becomes
    relevant again the moment somebody asks why you moved. A relevance call
    that turns out wrong should cost a restore, not the information."""
    _mint("old:1", "the postgres schema uses jsonb")
    memory.retire_memories(["old:1"], reason="moved to SQLite", turn_idx=5)
    assert db.q("SELECT COUNT(*) AS c FROM memories WHERE event_key='old:1'",
                one=True)["c"] == 1

    memory.restore_memories(["old:1"])
    found = memory.search_memories("postgres jsonb", current_turn_idx=10)
    assert "old:1" in {m["event_key"] for m in found}


def test_a_whole_batch_restores_at_once(temp_db):
    """A retirement is usually one judgement about one topic, and undoing it
    one row at a time would be a worse experience than the mistake."""
    for n in range(4):
        _mint(f"old:{n}", f"a postgres detail number {n}")
    out = memory.retire_memories([f"old:{n}" for n in range(4)],
                                 reason="moved to SQLite", turn_idx=5)
    assert len(out["retired"]) == 4
    restored = memory.restore_memories(batch=out["batch"])
    assert restored["restored"] == 4


def test_retiring_without_a_reason_is_refused(temp_db):
    """A row set aside for no stated cause cannot be reviewed later, and "why
    is this gone" is the question a restore has to answer."""
    _mint("old:1", "something")
    out = memory.retire_memories(["old:1"], reason="   ", turn_idx=5)
    assert out["ok"] is False and "reason" in out["error"]
    assert memory.search_memories("something", current_turn_idx=10)


def test_a_commitment_cannot_be_retired(temp_db):
    """An open promise must nag, not fade. A commitment the assistant can
    retire when it feels stale is not a commitment."""
    _mint("promise:1", "I promised to check the migration before Friday",
          kind="commitment")
    out = memory.retire_memories(["promise:1"], reason="old sprint",
                                 turn_idx=5)
    assert out["retired"] == []
    assert out["refused"][0]["why"].startswith("a commitment must nag")


def test_a_disputed_row_cannot_be_retired(temp_db):
    """The dispute IS the record that something was unstable. Retiring it
    hides the instability rather than the noise."""
    _mint("ev:1", "the docs say the flag defaults to true")
    memory.record_dispute("the docs were corrected; it defaults to false",
                          turn_idx=4, memory_ref="ev:1")
    out = memory.retire_memories(["ev:1"], reason="old docs", turn_idx=5)
    assert out["retired"] == []
    assert "dispute" in out["refused"][0]["why"]


def test_the_assistant_cannot_forget_that_it_forgot(temp_db, stub_model):
    """An assistant that can retire its own retirement notes will later tell
    you confidently that it never knew — a worse failure than the clutter the
    feature removes."""
    providers.set_chat_stub(lambda s, u, **kw: json.dumps({"reply": "ok"}))
    _mint("old:1", "a superseded decision about postgres")
    result = pipeline.run_turn("noop")
    memory.retire_memories(["old:1"], reason="moved to SQLite",
                           turn_idx=result["turn_idx"])
    note_key = f"turn:{result['turn_idx']}:retirement"
    memory.add_memory("episodic", "witnessed", 0.8,
                      "retired: I set aside 1 memory. Reason: moved to SQLite",
                      turn_idx=result["turn_idx"], event_key=note_key)
    out = memory.retire_memories([note_key], reason="tidying", turn_idx=9)
    assert out["retired"] == []
    assert "forgot" in out["refused"][0]["why"]


def test_retiring_is_grounded_against_delivered_refs(temp_db, stub_model):
    """The same rule that stops the model citing a memory it never saw stops
    it discarding one. A ref it was not shown is not a ref it may act on."""
    _mint("never:delivered", "a memory the model was never shown")

    providers.set_chat_stub(lambda s, u, **kw: json.dumps({
        "reply": "tidying up",
        "retire": {"memory_refs": ["never:delivered"],
                   "reason": "superseded"}}))
    result = pipeline.run_turn("clean up the old project notes")
    assert any("ungrounded" in w for w in result["warnings"])
    assert db.q("SELECT retired FROM memories "
                "WHERE event_key='never:delivered'", one=True)["retired"] == ""


def test_the_turn_records_what_was_retired_and_why(temp_db, stub_model):
    """Visibility is what makes the capability safe to grant: the user is
    told, in the same channel as every other warning, that the assistant
    stopped remembering something."""
    # Minted at turn 1 so the turn-cutoff lets turn 2 see it: a row belonging
    # to the turn being decided is correctly invisible to that turn.
    _mint("old:1", "the postgres schema uses jsonb", turn_idx=1)
    seen = {}
    # One warm-up turn: the row lives at turn 1, and a row belonging to the
    # turn being decided is correctly invisible to that turn. The retiring
    # turn has to be turn 2 for the seam to deliver it at all.
    providers.set_chat_stub(lambda s, u, **kw: json.dumps({"reply": "hi"}))
    session = pipeline.run_turn("first")["session_id"]

    def stub(system, user, **kw):
        payload = json.loads(user)
        refs = [m["memory_ref"] for m in
                (payload["memory"].get("recalled_old_memories") or [])
                + (payload["memory"].get("recent_exchanges") or [])
                if m.get("memory_ref") == "old:1"]
        seen["refs"] = refs
        return json.dumps({"reply": "tidied",
                           "retire": {"memory_refs": refs,
                                      "reason": "we moved to SQLite"}})

    providers.set_chat_stub(stub)
    result = pipeline.run_turn("we're on SQLite now, drop the postgres stuff",
                               session)
    assert seen.get("refs") == ["old:1"], "the fixture needs the row delivered"
    assert "retired" in result["trace"]
    assert result["trace"]["retired"]["reason"] == "we moved to SQLite"
    assert any("retired 1 memories" in w for w in result["warnings"])
    note = db.q("SELECT content FROM memories WHERE event_key=?",
                (f"turn:{result['turn_idx']}:retirement",), one=True)
    assert note and "we moved to SQLite" in note["content"]


def test_the_host_can_see_everything_set_aside(temp_db):
    _mint("old:1", "a superseded postgres detail")
    memory.retire_memories(["old:1"], reason="moved to SQLite", turn_idx=5)
    listed = memory.retired_rows()
    assert len(listed) == 1
    assert listed[0]["reason"] == "moved to SQLite"
    assert listed[0]["event_key"] == "old:1"


def test_purge_is_not_reachable_from_the_model(temp_db, stub_model):
    """Retiring is a reversible relevance judgement; destroying the record is
    a different act and belongs to the person whose records they are. There
    is no `purge` side channel, and this test is what keeps it that way."""
    import prompts
    assert "purge" not in prompts.RESPOND_SYSTEM.lower()
    assert "purge" not in open("pipeline.py").read()


def test_purge_destroys_only_retired_rows(temp_db):
    _mint("live:1", "something current")
    _mint("old:1", "something superseded")
    memory.retire_memories(["old:1"], reason="moved on", turn_idx=5)
    out = memory.purge_retired()
    assert out["purged"] == 1
    remaining = {r["event_key"] for r in db.q("SELECT event_key FROM memories")}
    assert remaining == {"live:1"}


def test_purge_spares_a_row_a_summary_still_cites(temp_db):
    """Those refs are the audit trail for a clause the assistant will keep
    asserting; breaking it leaves a summary claiming support it cannot
    show."""
    _mint("old:1", "the postgres schema uses jsonb for settings")
    memory.save_memory_summary(
        "We chose a jsonb settings column.", scope=memory.SCOPE_FIRSTHAND,
        start_turn_idx=1, end_turn_idx=2,
        support=[{"claim": "We chose a jsonb settings column.",
                  "support_refs": ["old:1"], "epistemic_origin": ""}])
    memory.retire_memories(["old:1"], reason="moved to SQLite", turn_idx=5)
    out = memory.purge_retired()
    assert out["purged"] == 0
    assert out["kept_because_cited_by_a_summary"] == 1


def test_a_retired_row_is_not_delivered_as_a_ref(temp_db, stub_model):
    """If a retired row could still be cited, retirement would be cosmetic."""
    _mint("old:1", "a superseded postgres detail about settings")
    memory.retire_memories(["old:1"], reason="moved to SQLite", turn_idx=2)
    _payload, internal = memory.build_memory_context(10, "settings schema")
    assert "old:1" not in internal["delivered_refs"]


def test_a_retired_row_does_not_reach_consolidation(temp_db):
    """Otherwise the summary layer would quietly reintroduce what was set
    aside."""
    for n in range(1, 12):
        _mint(f"m:{n}", f"a postgres detail number {n}",
              provenance="witnessed", turn_idx=n)
    memory.retire_memories([f"m:{n}" for n in range(1, 12)],
                           reason="moved to SQLite", turn_idx=12)
    seen = {}

    def consolidator(payload):
        seen["rows"] = payload["memories_chronological"]
        return {"summary": "nothing relevant happened"}

    memory.consolidate_memory(12, consolidator)
    assert all("postgres detail" not in row["details"]
               for row in seen.get("rows", []))
