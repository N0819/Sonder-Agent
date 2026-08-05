"""The automation loop: many turns, one run, no turn limit.

What these defend is the boundary between "keeps going" and "runs away". The
user asked for no turn limit, so nothing here counts iterations — what is
bounded is REPETITION, and the difference matters because a fixed ceiling is
exactly what stopped the assistant mid-edit when deliberation capped at three.

The other half is that a run must remain talkable-to. A loop you can only
start and stop is a batch job with a cancel button; the correction that
arrives in the middle has to be READ in the middle, or the only way to
redirect is to halt and throw away everything the run established.
"""

from __future__ import annotations

import autoloop
import turnrun


def _turns(*scripted):
    """A fake `run_turn` that replays a list of results, one per call, and
    records the instruction each was given."""
    seen = []
    plan = list(scripted)

    def fake(text, session_id=None, run=None):
        seen.append(text)
        result = plan.pop(0) if plan else {}
        return dict({"reply": "", "warnings": [], "trace": {},
                     "respond_ok": True, "session_id": session_id}, **result)
    fake.seen = seen
    return fake


def _worked(step="", **trace):
    """An iteration that committed something and wants another."""
    return {"continue_work": step, "trace": dict({"edits": [{"path": "a"}]},
                                                 **trace)}


def test_a_run_continues_until_it_stops_asking(temp_db):
    """The whole feature: the assistant names its next step and the engine
    feeds it straight back, so landing a verified change — outline, edit, run,
    read the result — does not need the user to say "carry on" four times."""
    fake = _turns(_worked("outline coding.py"),
                  _worked("expand the judge chunk"),
                  {"continue_work": "", "reply": "done"})

    out = autoloop.run_session("fix the harness grading", run_turn=fake)

    assert fake.seen == ["fix the harness grading", "outline coding.py",
                         "expand the judge chunk"]
    assert out["iterations"] == 3
    assert out["stopped_because"] == "finished"
    assert [r["from_user"] for r in out["replies"]] == [True, False, False]


def test_an_absent_next_step_ends_the_run(temp_db):
    """Absence STOPS. Every way this mechanism can fail — a model that ignores
    the field, a malformed payload, a provider error — has to fail by
    finishing, because the alternative is an unattended loop billing all night
    against a field nobody set."""
    fake = _turns({"reply": "here is the answer"})

    out = autoloop.run_session("what is 2+2?", run_turn=fake)

    assert out["iterations"] == 1
    assert out["stopped_because"] == "finished"


def test_a_failed_respond_stage_cannot_drive_the_loop(temp_db):
    """A 429 is not an instruction. The pipeline already blanks
    `continue_work` when the respond stage fails; this pins the loop refusing
    it independently, because that is the file where an all-night bill for a
    provider outage would be written."""
    fake = _turns({"continue_work": "keep going", "respond_ok": False})

    out = autoloop.run_session("go", run_turn=fake)

    assert out["iterations"] == 1
    assert out["stopped_because"] == "the respond stage failed"


def test_a_run_that_stops_getting_anywhere_is_stopped_and_says_so(temp_db):
    """NOT a turn limit — the user asked for none, and a fixed cap is what
    stopped the assistant mid-edit at three deliberation rounds. What is
    bounded is repetition: iterations that commit nothing durable AND name
    the same next step. The stall is reported, because a run that stopped
    spinning reads exactly like a run that finished unless it says otherwise.
    """
    spin = {"continue_work": "read coding.py", "trace": {"minted": 1}}
    fake = _turns(spin, dict(spin), dict(spin), dict(spin))

    out = autoloop.run_session("look into it", run_turn=fake)

    assert out["iterations"] == 3          # first, then two that repeated it
    assert "same next step" in out["stopped_because"]


def test_a_different_next_step_is_not_a_stall(temp_db):
    """Reading is not spinning. An outline, an expand, a directory walk commit
    nothing durable and are often the right move — so a run that is thinking
    its way toward an edit must not be cut off for having nothing to show yet.
    """
    fake = _turns({"continue_work": "outline coding.py", "trace": {}},
                  {"continue_work": "expand chunk c1", "trace": {}},
                  {"continue_work": "expand chunk c2", "trace": {}},
                  {"continue_work": "", "reply": "now I can answer"})

    out = autoloop.run_session("look into it", run_turn=fake)

    assert out["iterations"] == 4
    assert out["stopped_because"] == "finished"


def test_progress_resets_the_stall_count(temp_db):
    """Repeating a step you just made progress on is a retry, not a loop —
    running the same suite after each of three edits names one step three
    times and is the correct behaviour."""
    fake = _turns({"continue_work": "run the suite", "trace": {"minted": 1}},
                  _worked("run the suite"),
                  {"continue_work": "run the suite", "trace": {"minted": 1}},
                  _worked("run the suite"),
                  {"continue_work": "", "reply": "green"})

    out = autoloop.run_session("verify it", run_turn=fake)

    assert out["iterations"] == 5
    assert out["stopped_because"] == "finished"


def test_the_user_can_redirect_a_running_loop(temp_db):
    """The reason this is not a batch job. A message arriving mid-run becomes
    the NEXT instruction outright rather than being appended to the plan it
    interrupts — a user who speaks up is redirecting, and letting the
    abandoned plan still set the agenda is how "stop doing that" turns into
    "do that, and also this"."""
    run = turnrun.create("start", None)
    fake = _turns(_worked("keep refactoring"), _worked("keep refactoring"),
                  {"continue_work": "", "reply": "stopped"})
    original = fake

    def with_interjection(text, session_id=None, run=None):
        result = original(text, session_id, run=run)
        if len(original.seen) == 1:
            run.say("actually, leave that file alone")
        return result

    out = autoloop.run_session("refactor everything", run=run,
                               run_turn=with_interjection)

    assert original.seen[1] == "actually, leave that file alone"
    assert "keep refactoring" not in original.seen[1]
    assert [r["from_user"] for r in out["replies"]][:2] == [True, True]


def test_a_message_to_a_finished_run_is_refused_rather_than_dropped(temp_db):
    """A message typed a moment too late has to become an ordinary new turn.
    Reporting it as delivered would lose it silently, which is the worst of
    the three available outcomes."""
    run = turnrun.create("x", None)
    assert run.say("hello") == "delivered"
    run.finish("done")

    assert run.say("too late") == "not_running"
    assert run.heard() == ["hello"]


def test_what_was_said_is_delivered_once(temp_db):
    """Drain-and-clear, not read-and-mark. Redelivering a correction reads to
    the model as the user repeating themselves — that is, as not having been
    listened to the first time."""
    run = turnrun.create("x", None)
    run.say("one")
    run.say("two")

    assert run.drain_inbox() == ["one", "two"]
    assert run.drain_inbox() == []
    assert run.heard() == ["one", "two"]      # the record survives the drain


def test_an_empty_instruction_does_nothing(temp_db):
    """The loop is driven by text; a blank one must not start a turn, or the
    first iteration of every mis-clicked run is a real model call."""
    fake = _turns({"reply": "should not happen"})

    out = autoloop.run_session("   ", run_turn=fake)

    assert out["iterations"] == 0
    assert fake.seen == []
