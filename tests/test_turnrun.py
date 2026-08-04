# Turns as observable, haltable jobs.
#
# The rule under test throughout: a halt is a STAGE BOUNDARY, never a point
# inside the commit. Everything durable about a turn is written in one
# transaction, so an interruption inside it would leave the ordinal consumed
# with the turn half-written — which is the failure `run_turn` reserves the
# ordinal early to avoid in the first place.

import json

import pytest

import pipeline
import providers
import turnrun
from db import q


def _stub(reply="ok"):
    def fn(system, user):
        payload = json.loads(user)
        if "user_message" in payload:
            return json.dumps({"reply": reply})
        if "rounds_left" in payload:
            return json.dumps({"action": "conclude", "answer": "",
                               "citations": []})
        return json.dumps({"summary": "", "received_summary": "",
                           "surmise_summary": "", "key_phrases": [],
                           "unresolved_threads": []})
    providers.set_chat_stub(fn)


def test_an_unobserved_turn_is_unchanged(temp_db):
    """The null-object default. Every existing caller — tests, the blocking
    route, scripts — passes no run, and must not acquire a dependency on the
    streaming machinery to keep working."""
    _stub()
    try:
        out = pipeline.run_turn("hello")
    finally:
        providers.set_chat_stub(None)
    assert out["reply"] == "ok"
    assert out["respond_ok"] is True


def test_a_halt_before_the_commit_writes_nothing(temp_db):
    """The whole promise of the button. A halted turn must leave the database
    as it found it, or "stopped" means "stopped somewhere, good luck"."""
    _stub()
    run = turnrun.create("hello", None)
    before = q("SELECT COUNT(*) AS c FROM memories", one=True)["c"]
    turns_before = q("SELECT COUNT(*) AS c FROM turns", one=True)["c"]
    run.request_halt()          # halt before the worker starts a stage
    try:
        raised = False
        try:
            pipeline.run_turn("hello", None, run=run)
        except turnrun.TurnHalted:
            raised = True
    finally:
        providers.set_chat_stub(None)
    assert raised
    assert q("SELECT COUNT(*) AS c FROM memories", one=True)["c"] == before
    assert q("SELECT COUNT(*) AS c FROM turns", one=True)["c"] == turns_before


def test_a_halt_during_the_commit_is_refused_not_dropped(temp_db):
    """`too_late` is a real outcome and has to be reported as one. Answering a
    late halt with a bare acknowledgement would make the button lie about the
    single thing the user is watching, and answering it by halting would tear
    a transaction in half."""
    run = turnrun.create("hello", None)
    run.enter_commit()
    assert run.request_halt() == "too_late"
    # ...and the pipeline's checkpoint must agree: no exception once latched.
    assert run.halted() is False


def test_a_finished_run_cannot_be_halted(temp_db):
    """Clicking halt on a turn that already landed must say so rather than
    setting a flag nothing will ever read."""
    run = turnrun.create("hello", None)
    run.finish("done", result={"reply": "ok"})
    assert run.request_halt() == "not_running"


def test_the_stream_replays_from_the_beginning(temp_db):
    """Recall is usually over before the browser has opened the stream. A
    stream that only carried live events would drop the memory list — the part
    of the reasoning trail the user most wanted to see."""
    run = turnrun.create("hello", None)
    run.emit("recall", returned=3, pondered=1)
    run.emit("respond", state="answered")
    run.finish("done", result={"reply": "ok"})
    stages = [ev["stage"] for ev in run.follow(timeout=5.0)]
    assert stages == ["recall", "respond", "end"]


def test_a_worker_failure_ends_the_stream(temp_db):
    """A dead worker must not leave the page waiting forever on a stream that
    will never produce another event."""
    run = turnrun.create("hello", None)

    def explode(_r):
        raise RuntimeError("worker died")

    turnrun.start(run, explode)
    events = list(run.follow(timeout=5.0))
    assert events[-1]["stage"] == "end"
    assert events[-1]["status"] == "failed"
    assert "worker died" in events[-1]["error"]


def test_a_watched_turn_reports_the_stages_it_ran(temp_db):
    """The reasoning panel renders only what the server says happened, so the
    server has to actually say it. Pinned because an empty trail looks
    identical to a turn that did no thinking."""
    _stub()
    run = turnrun.create("hello", None)
    try:
        turnrun.start(run, lambda r: pipeline.run_turn("hello", None, run=r))
        events = list(run.follow(timeout=30.0))
    finally:
        providers.set_chat_stub(None)
    stages = [ev["stage"] for ev in events]
    assert "recall" in stages and "respond" in stages and "commit" in stages
    assert events[-1]["status"] == "done"
    assert events[-1]["result"]["reply"] == "ok"


# ---- A subagent must not be a silent gap ----

def test_a_scout_narrates_itself_to_the_watching_turn(temp_db, monkeypatch):
    """A subagent was a hole in the reasoning panel. The turn went quiet for
    up to DEEP_TIMEOUT — or four model calls for a scout — with no way to
    tell work from a hang, and the parent could see every step the whole
    time."""
    import subagents
    run = turnrun.create("investigate", None)
    turnrun.bind(run)
    calls = []

    def fake_chat(system, user):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({"action": "search", "query": "buildkite docs"})
        return json.dumps({"action": "report",
                           "report": {"summary": "found it", "claims": []}})

    monkeypatch.setattr(subagents, "chat_complete", fake_chat)
    monkeypatch.setattr(subagents.tools_web, "search", lambda q, **kw: [])
    try:
        subagents._run_scout("find the deploy tool", turn_idx=1)
    finally:
        turnrun.bind(None)
    stages = [e for e in run.events if e["stage"] == "subagent"]
    states = [e["state"] for e in stages]
    assert "started" in states and "search" in states and "reported" in states
    assert all(e["kind"] == "scout" for e in stages)
    # The task travels with every line, or a two-agent turn is unreadable.
    assert all(e["task"] for e in stages)


def test_halting_reaches_a_running_scout(temp_db, monkeypatch):
    """A halt that could only land between subagents would wait out the whole
    thing — from the button's side, a halt that did nothing."""
    import subagents
    run = turnrun.create("investigate", None)
    turnrun.bind(run)
    run.request_halt()

    def fake_chat(system, user):
        raise AssertionError("halted before the first model call")

    monkeypatch.setattr(subagents, "chat_complete", fake_chat)
    try:
        with pytest.raises(turnrun.TurnHalted):
            subagents._run_scout("find something", turn_idx=1)
    finally:
        turnrun.bind(None)


def test_an_unwatched_subagent_still_runs(temp_db, monkeypatch):
    """`turnrun.current()` is None for a blocking turn or a test, and the
    narration must be inert rather than a crash."""
    import subagents
    turnrun.bind(None)
    monkeypatch.setattr(subagents, "chat_complete", lambda s, u: json.dumps(
        {"action": "report", "report": {"summary": "fine"}}))
    out = subagents._run_scout("anything", turn_idx=1)
    assert out["summary"] == "fine"
