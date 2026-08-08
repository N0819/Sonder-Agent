"""Asking a run to stop was advisory, and it was overruled.

There were two ways to stop an automation run and neither is what "stop"
usually means. `request_halt` raises out of the pipeline: the work in flight
dies and everything learned but not yet written dies with it. `say` is
advisory ON PURPOSE — the loop folds a message in beside the plan rather than
over it, so "also check the tests" cannot throw away three iterations, and the
model decides whether it counts as a redirect.

That is right for steering and wrong for stopping, because it leaves a stop
request to the judgement of the thing being asked to stop. Live: two wind-down
messages delivered to run 4154ed79f4324773, both folded in as context, six
further research turns after the first. Neither instruction was refused and
neither was obeyed. The run had by then measured a reword rate and separated
59 artefacts from 103 gap slots, none of which was in any reply.

So the third form: one more ordinary turn whose job is to COMPILE, and then
the loop ends whatever the model asks for next. Compiling is the point — the
alternative to a report is not a shorter run, it is a run whose findings were
never written down.
"""

from __future__ import annotations

import autoloop
import turnrun


class _Run:
    """The loop's view of a run, with a wind-down that can be tripped."""

    def __init__(self, wind_after=None):
        self.wind_after = wind_after
        self.seen = []
        self.calls = 0
        self._wind = False

    def emit(self, *a, **k):
        pass

    def halted(self):
        pass

    def drain_inbox(self):
        return []

    def winding_down(self):
        return self._wind

    def turn(self, text, session_id=None, run=None, speaker="user",
             carried_plan=""):
        self.calls += 1
        self.seen.append(text)
        if self.wind_after is not None and self.calls >= self.wind_after:
            self._wind = True
        return {"reply": f"reply {self.calls}", "respond_ok": True,
                "continue_work": "keep going forever",
                "trace": {"minted": 5}, "session_id": session_id}


def test_a_run_that_never_stops_itself_still_stops():
    """THE REPRODUCTION. `continue_work` is always set and every iteration
    mints, so neither existing brake can fire — this is the live shape, where
    20 of 22 turns satisfied `made_progress` by minting alone."""
    r = _Run(wind_after=2)
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    assert out["stopped_because"] == "wound down: reported on request"
    assert r.calls == 3, r.seen          # two of work, one to report


def test_the_last_turn_is_told_to_compile_rather_than_research():
    """A run cut off mid-investigation loses what it found. The closing turn
    is an ordinary turn -- same stages, same commit -- so the findings get
    written down."""
    r = _Run(wind_after=1)
    autoloop.run_session("go", run=r, run_turn=r.turn)
    last = r.seen[-1]
    assert "STOP RESEARCHING AND REPORT" in last
    assert "no experiments" in last
    assert "unclosable" in last


def test_the_model_cannot_ask_for_another_round():
    """The whole point. `continue_work` is set on every iteration including
    the report, and it must not buy one."""
    r = _Run(wind_after=1)
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    assert r.calls == 2
    assert out["iterations"] == 2


def test_what_they_said_when_asking_travels_with_it():
    """The request usually carries a reason -- budget, a redirect, a deadline.
    Dropping it would make the report answer a question nobody asked."""
    r = _Run(wind_after=1)
    r.drain_inbox = lambda: ["we are out of budget; land what you have"]
    autoloop.run_session("go", run=r, run_turn=r.turn)
    assert "out of budget" in r.seen[-1]


def test_an_ordinary_message_is_still_advisory():
    """The steering behaviour must not become collateral damage: a message
    still goes BESIDE the plan and the run carries on.

    `stop_after` rather than an unbounded run, because the first draft of this
    test used a run that never stops -- which is the exact shape under test --
    and hung the suite instead of failing it.
    """
    class _Steerable(_Run):
        carried = []

        def turn(self, text, session_id=None, run=None, speaker="user",
                 carried_plan=""):
            out = super().turn(text, session_id, run, speaker, carried_plan)
            # PER CALL, not the last one. Recording a single attribute made
            # this assert the third iteration's empty plan and fail a working
            # mechanism.
            self.carried.append(carried_plan)
            if self.calls >= 3:
                out["continue_work"] = ""       # it finishes on its own terms
            return out

    r = _Steerable()
    r.carried = []
    said = ["also check the tests"]
    r.drain_inbox = lambda: [said.pop()] if said else []
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    assert out["stopped_because"] == "finished"
    assert r.seen[1] == "also check the tests"
    # Carried ALONGSIDE, not instead of: the plan survives the interjection,
    # which is the property that stops a steering user being punished for it.
    assert r.carried[1] == "keep going forever", r.carried


# --- the run object's own contract ---------------------------------------

def test_winding_down_a_finished_run_says_so():
    """A request typed at a run that has just ended must not report success:
    the caller has to know it became nothing, the way `say` already does."""
    run = turnrun.create("x", None)
    run.finish("done")
    assert run.request_wind_down() == "not_running"


def test_asking_twice_is_not_two_reports():
    """A double click, or a user and a script both asking."""
    run = turnrun.create("x", None)
    assert run.request_wind_down() == "winding-down"
    assert run.request_wind_down() == "already"
    assert run.winding_down() is True


def test_it_is_not_a_halt():
    """A halt kills the work in flight; this keeps it. If wind-down ever
    starts setting the halt flag, the findings die again.
    """
    run = turnrun.create("x", None)
    run.request_wind_down()
    assert run._halt is False
    assert run.status == "running"
