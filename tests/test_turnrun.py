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


# ---- A dropped connection is not a dropped turn ----

def test_a_reconnecting_client_resumes_instead_of_replaying(temp_db):
    """Losing the pipe used to lose the turn: the stream closed, a retry was
    offered, and taking it ran the whole thing a SECOND time while the first
    was still going. The run never stopped — only the connection did.

    Resume has to be exact in both directions. Replaying from zero redraws
    every step the page already has; starting from "now" loses whatever
    arrived during the gap. `since` is what makes it neither."""
    run = turnrun.create("hello", None)
    run.emit("recall", returned=3)
    run.emit("respond", state="calling the model")
    first = list(run.follow(timeout=1.0))
    assert [e["stage"] for e in first] == ["recall", "respond"]

    # ...the pipe drops here, and the turn carries on regardless.
    run.emit("commit", state="writing")
    run.finish("done", result={"reply": "ok"})

    last_id = first[-1]["i"]
    resumed = list(run.follow(timeout=1.0, since=last_id + 1))
    assert [e["stage"] for e in resumed] == ["commit", "end"]
    assert resumed[-1]["result"]["reply"] == "ok"


def test_a_client_that_missed_the_ending_is_still_told_how_it_went(temp_db):
    """The worst moment to drop is during the commit, because that is the one
    stage that cannot be halted and the one whose outcome matters most. A
    resume past the last event must still deliver `end` rather than waiting on
    a stream with nothing more to say."""
    run = turnrun.create("hello", None)
    run.emit("commit", state="writing")
    run.finish("done", result={"reply": "committed"})
    # `since` well past the end — the client saw everything and then dropped.
    events = list(run.follow(timeout=1.0, since=99))
    assert [e["stage"] for e in events] == ["end"]
    assert events[0]["result"]["reply"] == "committed"


def test_every_frame_carries_an_id_so_the_browser_can_resume(temp_db):
    """`Last-Event-ID` is the browser's own mechanism and it only works if the
    frames are labelled. Without `id:` EventSource reconnects with no cursor
    and the server has no choice but to replay everything."""
    run = turnrun.create("hello", None)
    run.emit("recall", returned=1)
    run.finish("done", result={"reply": "ok"})
    frames = "".join(turnrun.sse(run))
    assert "retry: " in frames, "the reconnect delay is left to the browser"
    ids = [ln for ln in frames.splitlines() if ln.startswith("id: ")]
    # One per event plus the terminal `end`.
    assert ids == ["id: 0", "id: 1"], frames


def test_a_resumed_stream_starts_where_it_was_told_to(temp_db):
    """The route turns `Last-Event-ID` into `since`, so the serialiser has to
    honour it — an off-by-one here shows up as one duplicated or one missing
    step, which is exactly the kind of thing nobody notices until they do."""
    run = turnrun.create("hello", None)
    run.emit("recall", returned=1)
    run.emit("respond", state="one")
    run.emit("commit", state="two")
    run.finish("done", result={"reply": "ok"})
    frames = "".join(turnrun.sse(run, since=2))
    assert "id: 0" not in frames and "id: 1" not in frames
    assert "id: 2" in frames and '"stage": "commit"' in frames


def test_no_cursor_and_a_cursor_of_zero_are_different(temp_db):
    """SHIPPED, AND SILENT. The route defaulted to the integer 0 and then
    tested `str(cursor).strip()` — falsey as a number, truthy as "0" — so a
    request with no `Last-Event-ID` took the resume branch and every fresh
    stream began at event 1. Nothing surfaced it: event 0 renders as nothing
    today, so the loss was invisible until a first event carried something.

    Verified against a live server before it was fixed: a stream that should
    have opened `id: 0` opened `id: 1`."""
    import app
    assert app.resume_cursor(None, None) == 0, "a fresh stream must start at 0"
    assert app.resume_cursor("", None) == 0
    assert app.resume_cursor("0", None) == 1, "id 0 was delivered; send 1 next"
    assert app.resume_cursor("7", None) == 8
    # The header wins over the query parameter, and only falls through when
    # the browser did not send one.
    assert app.resume_cursor("3", "99") == 4
    assert app.resume_cursor(None, "3") == 4
    # An unparseable cursor replays rather than skips: a duplicated trail is
    # cosmetic, a skipped one loses the answer.
    assert app.resume_cursor("nonsense", None) == 0
    assert app.resume_cursor("-5", None) == 0
