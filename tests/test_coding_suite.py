"""Coding as investigation: predict, run, judge, revise.

The claim this suite defends is that the assistant OBSERVES what code does
rather than predicting it, and that the observation is graded outside the thing
being graded. Every test below corresponds to a way that claim can quietly stop
being true.

Four rules, each inherited from a scar in Sonder Engine:

  1. A PREDICTION BEFORE A RUN, or it is not an experiment. Running code and
     then deciding what it proved is how a model talks itself into a fix. It is
     also the "small-payload benchmark" failure recorded in the engine's own
     bench results, where four separate rankings inverted because nobody said
     in advance what the measurement was supposed to show.
  2. REPRODUCE BEFORE YOU FIX. A fix for a defect never observed failing cannot
     afterwards be told apart from a fix for nothing.
  3. A FAILURE IS DATA, NEVER AN ERROR. The moment a harness raises on its own
     negative results, the loop starts avoiding them.
  4. CONTRADICTION IS A DISPUTE, NOT AN AVERAGE. Two runs that disagree mean
     the thing under test is not a function of its inputs, which is the most
     interesting fact available and the first one smoothing would destroy.
"""

from __future__ import annotations

import coding
import research
import sandbox
import ui_review


class TestTheSandboxIsHonestAboutFailure:

    def test_it_runs_something_and_reports_what_happened(self):
        result = sandbox.run_python("print(6 * 7)")

        assert result["ok"] and result["stdout"].strip() == "42"

    def test_a_loop_is_caught_and_named_as_a_timeout(self):
        """Distinguished from a non-zero exit on purpose: "it never finished"
        and "it finished and failed" are different findings, and a caller that
        conflates them will keep re-running the loop."""
        result = sandbox.run_python("while True:\n    pass", timeout=2)

        assert result["timed_out"] is True
        assert result["exit_code"] is None
        assert result["ok"] is False

    def test_a_crash_is_a_result_and_not_an_exception(self):
        result = sandbox.run_python("raise ValueError('boom')")

        assert result["ok"] is False
        assert "ValueError" in result["stderr"]

    def test_the_hosts_credentials_do_not_reach_the_sandbox(self, monkeypatch):
        """Code the assistant wrote should not be able to read the keys it is
        running under."""
        monkeypatch.setenv("ASSISTANT_CHAT_KEY", "sk-secret")
        result = sandbox.run_python(
            "import os; print(os.environ.get('ASSISTANT_CHAT_KEY', 'ABSENT'))")

        assert result["stdout"].strip() == "ABSENT"

    def test_it_refuses_to_write_outside_its_workspace(self):
        result = sandbox.run(
            {"../escape.py": "x = 1"}, ["python3", "-c", "pass"])

        assert result["ok"] is False
        assert "outside the workspace" in result["stderr"]

    def test_a_missing_interpreter_is_reported_not_raised(self):
        result = sandbox.run({}, ["definitely-not-a-real-binary"])

        assert result["ok"] is False
        assert "no such interpreter" in result["stderr"]


class TestJudgingIsMechanical:
    """A prediction a model grades itself against is not a prediction."""

    def test_every_stated_check_must_hold(self):
        result = {"ok": True, "exit_code": 0, "stdout": "hello world",
                  "stderr": "", "timed_out": False}

        assert coding.judge(result, {"exit_zero": True})[0] == "confirmed"
        assert coding.judge(result, {"stdout_has": "world"})[0] == "confirmed"
        assert coding.judge(result, {"stdout_has": "goodbye"})[0] == "refuted"
        assert coding.judge(result, {"exit_zero": False})[0] == "refuted"

    def test_no_prediction_is_not_a_passing_prediction(self):
        result = {"ok": True, "exit_code": 0, "stdout": "", "stderr": "",
                  "timed_out": False}

        outcome, why = coding.judge(result, {})
        assert outcome == "inconclusive"
        assert "no prediction" in why

    def test_a_broken_harness_is_inconclusive_not_a_refutation(self):
        """"The interpreter was missing" says nothing about the hypothesis, and
        folding it into `contradicts` would move confidence on the strength of
        a broken harness."""
        result = {"ok": False, "exit_code": None, "stdout": "",
                  "stderr": "no such interpreter", "timed_out": False}

        assert coding.judge(result, {"exit_zero": True})[0] == "inconclusive"


class TestTheLoopEndToEnd:

    def _hypothesis(self, temp_db):
        return research.open_hypothesis(
            "Does divide() handle a zero denominator?", turn_idx=1)["id"]

    BUGGY = "def divide(a, b):\n    return a / b\n\nprint(divide(6, 0))\n"

    def test_an_experiment_without_a_prediction_is_refused(self, temp_db):
        hid = self._hypothesis(temp_db)
        out = coding.run_experiment(hid, source=self.BUGGY, expect={},
                                    turn_idx=1)

        assert out["outcome"] == "inconclusive"
        assert "prediction" in out["why"]

    def test_a_confirmed_prediction_moves_confidence_and_leaves_evidence(
            self, temp_db):
        hid = self._hypothesis(temp_db)
        before = research.get_hypothesis(hid)["confidence"]
        out = coding.run_experiment(
            hid, source=self.BUGGY,
            expect={"exit_zero": False, "stderr_has": "ZeroDivisionError"},
            turn_idx=1, note="reproduce")

        assert out["outcome"] == "confirmed"
        assert research.get_hypothesis(hid)["confidence"] > before
        assert out["evidence"]["url"].startswith("experiment:")

    def test_a_fix_is_refused_until_something_has_been_seen_to_fail(
            self, temp_db):
        hid = self._hypothesis(temp_db)

        refused = coding.propose_fix(hid, description="guard it", turn_idx=1)
        assert refused["accepted"] is False

        coding.run_experiment(
            hid, source=self.BUGGY,
            expect={"exit_zero": False, "stderr_has": "ZeroDivisionError"},
            turn_idx=1)
        accepted = coding.propose_fix(hid, description="raise on b == 0",
                                      turn_idx=2)
        assert accepted["accepted"] is True

    def test_a_run_nobody_predicted_does_not_open_the_fix_gate(self, temp_db):
        hid = self._hypothesis(temp_db)
        coding.run_experiment(hid, source="print('hi')", expect={}, turn_idx=1)

        assert coding.propose_fix(hid, description="x", turn_idx=1)[
            "accepted"] is False

    def test_the_same_experiment_disagreeing_with_itself_is_a_dispute(
            self, temp_db):
        """Not a new answer. Two runs that disagree mean the behaviour is not
        a function of its inputs, and averaging would hide the only
        interesting fact available."""
        from db import qi

        hid = self._hypothesis(temp_db)
        source = "raise SystemExit(0)\n"
        coding.run_experiment(hid, source=source, expect={"exit_zero": True},
                              turn_idx=1, note="same")
        qi("UPDATE experiments SET outcome='refuted' WHERE note='same'")
        again = coding.run_experiment(hid, source=source,
                                      expect={"exit_zero": True},
                                      turn_idx=2, note="same")

        assert again["repeated"] is True
        notes = (research.get_hypothesis(hid)["dispute"] or {}).get("notes")
        assert notes and "not deterministic" in notes[0]


class TestReadingAnInterfaceWithoutEyes:
    """Most UI defects are not visible in a screenshot of the happy path. Each
    check below corresponds to one that bites and one that looking would
    miss."""

    def test_a_duplicate_id_is_an_error(self):
        found = ui_review.review_html(
            '<div id="panel"></div><span id="panel"></span>')

        assert any(f["kind"] == "duplicate-id" and f["severity"] == "error"
                   for f in found)

    def test_an_unclosed_element_is_found(self):
        found = ui_review.review_html("<div><p>hi</p>")

        assert any(f["kind"] == "unclosed" for f in found)

    def test_an_icon_button_with_no_accessible_name(self):
        found = ui_review.review_html("<button>🔔</button>")

        assert any(f["kind"] == "unnamed-control" for f in found)

    def test_a_named_icon_button_is_fine(self):
        found = ui_review.review_html('<button aria-label="Mute">🔔</button>')

        assert not any(f["kind"] == "unnamed-control" for f in found)

    def test_a_click_handler_on_a_div_locks_out_the_keyboard(self):
        found = ui_review.review_html('<div onclick="go()">Go</div>')

        assert any(f["kind"] == "keyboard-unreachable" for f in found)

    def test_a_later_rule_that_loses_the_cascade(self):
        """The defect that produces "I changed it and nothing happened"."""
        found = ui_review.review_css("#panel .row{color:red}\n.row{color:blue}")

        assert any(f["kind"] == "specificity-loss" for f in found)

    def test_focus_removed_with_nothing_in_its_place(self):
        found = ui_review.review_css(":focus{outline:none}")

        assert any(f["kind"] == "focus-removed" for f in found)

    def test_focus_replaced_is_not_flagged(self):
        found = ui_review.review_css(
            ":focus{outline:none}\n:focus-visible{box-shadow:0 0 0 2px blue}")

        assert not any(f["kind"] == "focus-removed" for f in found)

    def test_a_script_reaching_for_a_node_that_is_not_there(self):
        found = ui_review.review_js('document.getElementById("gone").click()',
                                    html='<div id="here"></div>')

        assert any(f["kind"] == "missing-node" for f in found)

    def test_findings_come_back_worst_first(self):
        found = ui_review.review({
            "a.html": '<div id="x"></div><span id="x"></span><button>+</button>',
        })

        assert found[0]["severity"] == "error"
