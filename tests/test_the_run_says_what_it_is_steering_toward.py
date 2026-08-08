"""A run could not say what it was working toward, so it got interrupted.

From outside, a run three iterations into a deliberate detour and a run going
in circles are indistinguishable. The cost of that ambiguity is paid by
interrupting both, which is why a user watching a productive run still has to
reach for the stop.

`continue_work` cannot supply the signal. The prompt asks for a DIFFERENT next
step every iteration -- prompts.py says so in as many words -- so it cannot
mean anything by changing. `next_action` is asked for the opposite: what the
step is FOR, held stable and narrowed as measurements come in. Repetition
there is meaningful exactly where repetition in the step is not.

NOTHING HERE GRADES IT, and that is deliberate. Five brakes were proposed for
this loop and each was a guess about a field nobody had ever written down; the
only one that would have caught the run that prompted the work fired because a
human ran it by hand. So the field is collected first and thresholded later,
against history it produces rather than against an argument.

The second field is `ended_itself`. Reading 18 past sessions showed that ZERO
ended because the work was finished -- 8 external tune-ups, 7 mid-task
redirects, 1 tool failure. Recovering that took a session of archaeology, and
it stayed invisible for a hundred turns because nothing recorded it. It also
bounds its own reading: a run stopped from outside cannot say whether its end
condition would have fired, so an interruption rate is not an efficacy
measurement, and with this recorded the two stop being conflatable.
"""

from __future__ import annotations

import autoloop


class _Run:
    def __init__(self, script, minted=5):
        self.script = list(script)
        self.events = []
        self.calls = 0
        # `minted > 1` is what `made_progress` reads, so a test that wants the
        # stall to fire has to stop minting. The first draft of the stall test
        # left this at 5, the counter reset every iteration, and the run ran
        # off the end of its own script instead of stalling.
        self.minted = minted

    def emit(self, stage, **detail):
        self.events.append((stage, detail))

    def halted(self):
        pass

    def winding_down(self):
        return False

    def drain_inbox(self):
        return []

    def turn(self, text, session_id=None, run=None, speaker="user",
             carried_plan=""):
        step, action = self.script[self.calls]
        self.calls += 1
        return {"reply": "r", "respond_ok": True, "continue_work": step,
                "next_action": action, "trace": {"minted": self.minted},
                "session_id": session_id}


def _actions(run):
    return [d for stage, d in run.events if stage == "action"]


def test_a_held_action_is_visible_while_the_step_keeps_changing():
    """THE POINT. Three iterations of genuinely different next steps, all
    serving one action -- which is what a converging investigation looks like
    and what no existing field could show."""
    r = _Run([("read the seam", "cap the list at 20"),
              ("read fold_threads", "cap the list at 20"),
              ("count the closures", "cap the list at 20"),
              ("", "cap the list at 20")])
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    # Four, not three: the closing iteration names an action too,
    # and it is emitted before the empty step ends the run.
    assert [a["held"] for a in _actions(r)] == [1, 2, 3, 4]
    assert out["actions"][:3] == ["cap the list at 20"] * 3


def test_returning_to_an_action_is_not_holding_to_one():
    """CONSECUTIVE, not a tally. An action returned to after a detour is a
    different thing from one never let go of, and a plain count of matches
    cannot tell them apart -- it would read a wandering run as a steady one.
    """
    r = _Run([("a", "cap it"), ("b", "park it"), ("c", "cap it"), ("", "")])
    autoloop.run_session("go", run=r, run_turn=r.turn)
    assert [a["held"] for a in _actions(r)] == [1, 1, 1]


def test_an_iteration_that_names_no_action_breaks_the_streak():
    """A turn that named nothing is not evidence of holding to anything, and
    counting it as continuity would manufacture the signal."""
    r = _Run([("a", "cap it"), ("b", ""), ("c", "cap it"), ("", "")])
    autoloop.run_session("go", run=r, run_turn=r.turn)
    assert [a["held"] for a in _actions(r)] == [1, 1]


def test_nothing_is_stopped_by_any_of_it():
    """Instrumentation, not a brake. If this ever ends a run, the field has
    silently become a threshold nobody calibrated."""
    r = _Run([("a", "cap it")] * 5 + [("", "cap it")])
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    assert out["iterations"] == 6
    assert out["stopped_because"] == "finished"


def test_a_run_that_finished_says_it_ended_itself():
    r = _Run([("a", "x"), ("", "x")])
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    assert out["ended_itself"] is True


def test_a_run_stopped_from_outside_says_so():
    """The distinction the 18-session archaeology could not make without
    reading every opening message by hand."""
    class _Wound(_Run):
        def winding_down(self):
            return self.calls >= 1

    r = _Wound([("a", "x"), ("b", "x"), ("c", "x")])
    out = autoloop.run_session("go", run=r, run_turn=r.turn)
    assert out["stopped_because"] == "wound down: reported on request"
    # Wound down ON REQUEST but it did produce its report, so this counts as
    # ending on its own terms -- the alternative would file every reported
    # run as an interruption and lose the distinction being drawn.
    assert out["ended_itself"] is True


def test_a_stalled_run_did_not_end_itself():
    """The stall is the engine giving up on it, not it finishing."""
    r = _Run([("same step", "x")] * 4, minted=1)
    out = autoloop.run_session("go", run=r, run_turn=r.turn,
                               stall_limit=1)
    r2 = [d for stage, d in r.events if stage == "loop"][0]
    assert out["ended_itself"] is False
    assert r2["ended_itself"] is False
