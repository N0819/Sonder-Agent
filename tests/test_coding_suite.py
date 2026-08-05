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


# ---- The prediction vocabulary is the ceiling on what can be investigated ----

def test_a_prediction_can_name_an_exact_exit_code(temp_db):
    """`exit_zero` could only say "worked" or "did not". A program whose
    failure MODE is the finding — exit 2 means a parse error, exit 1 means the
    check failed — could not have that predicted, so the sharpest available
    claim had to be blurred into "non-zero" before it could be stated."""
    result = sandbox.run_python("import sys; sys.exit(3)")
    assert coding.judge(result, {"exit_code": 3})[0] == coding.OUTCOME_CONFIRMED
    assert coding.judge(result, {"exit_code": 2})[0] == coding.OUTCOME_REFUTED
    # A non-integer prediction is a broken instrument, not a refutation.
    assert coding.judge(result, {"exit_code": "three"})[0] == \
        coding.OUTCOME_INCONCLUSIVE


def test_a_prediction_can_be_a_regex_and_a_broken_one_settles_nothing(temp_db):
    """A malformed pattern must not move confidence. Grading a typo against
    the hypothesis is the same error as grading a missing interpreter against
    it — rule 4, arriving through the prediction rather than the run."""
    result = sandbox.run_python("print('total: 42 items')")
    assert coding.judge(result, {"stdout_matches": r"total: \d+ items"})[0] \
        == coding.OUTCOME_CONFIRMED
    assert coding.judge(result, {"stdout_matches": r"total: \d+ rows"})[0] \
        == coding.OUTCOME_REFUTED
    outcome, why = coding.judge(result, {"stdout_matches": "unclosed ("})
    assert outcome == coding.OUTCOME_INCONCLUSIVE
    assert "regex" in why


def test_stderr_lacks_is_predictable(temp_db):
    """`stderr_has` existed and its negation did not, so "this runs clean" —
    the commonest thing anyone wants to assert about a fix — was unsayable."""
    result = sandbox.run_python("print('fine')")
    assert coding.judge(result, {"stderr_lacks": "Traceback"})[0] == \
        coding.OUTCOME_CONFIRMED


def test_a_prediction_can_be_about_a_file_the_run_wrote(temp_db):
    """THE ONE THAT UNLOCKS DURABLE WORK. "The patch applied and the file now
    reads X" could previously only be checked by having the program print the
    file back — which tests the print statement alongside the patch, and
    silently passes when the patch wrote nothing and the print was hardcoded.

    The sandbox destroys its workspace on the way out, so the files have to be
    read back before that happens or the predicate has nothing to look at."""
    source = ("open('out.txt', 'w').write('VERSION = 2\\n')\n"
              "print('done')\n")
    result = sandbox.run_python(source)
    assert result["files_after"] == {}, "nothing was requested, nothing read"

    import coding as c
    expect = {"file_contains": {"out.txt": "VERSION = 2"}}
    result = sandbox.run(
        {"main.py": source}, [sandbox.sys.executable, "-s", "main.py"],
        collect=c._expected_paths(expect))
    assert coding.judge(result, expect)[0] == coding.OUTCOME_CONFIRMED
    assert coding.judge(result, {"file_lacks": {"out.txt": "VERSION = 2"}})[0] \
        == coding.OUTCOME_REFUTED
    assert coding.judge(
        result, {"file_equals": {"out.txt": "VERSION = 2"}})[0] == \
        coding.OUTCOME_CONFIRMED


def test_a_prediction_about_a_file_that_was_never_written_is_inconclusive(
        temp_db):
    """"The file says something else" and "there is no file" are different
    findings and only the first is about the hypothesis. Folding the second
    into `contradicts` would move confidence because a run crashed early."""
    expect = {"file_contains": {"never.txt": "anything"}}
    result = sandbox.run(
        {"main.py": "print('did not write it')"},
        [sandbox.sys.executable, "-s", "main.py"],
        collect=coding._expected_paths(expect))
    outcome, why = coding.judge(result, expect)
    assert outcome == coding.OUTCOME_INCONCLUSIVE
    assert "never.txt" in why


def test_the_old_vocabulary_is_graded_identically(temp_db):
    """Every new key is read only when present. A prediction written before
    any of this must grade exactly as it did, or the archive of past
    experiments stops meaning what it says."""
    result = sandbox.run_python("print('hello')")
    for expect, want in (
            ({"exit_zero": True}, coding.OUTCOME_CONFIRMED),
            ({"stdout_has": "hello"}, coding.OUTCOME_CONFIRMED),
            ({"stdout_lacks": "goodbye"}, coding.OUTCOME_CONFIRMED),
            ({"output_equals": "hello"}, coding.OUTCOME_CONFIRMED),
            ({"stdout_has": "goodbye"}, coding.OUTCOME_REFUTED),
            ({}, coding.OUTCOME_INCONCLUSIVE)):
        assert coding.judge(result, expect)[0] == want, expect


# ---- The sharpest instrument was the one most often misgraded ----

def test_a_pytest_collection_error_is_a_harness_failure_not_a_refutation(
        temp_db):
    """REPRODUCED BEFORE IT WAS FIXED. A test module that cannot import comes
    back exit 2 with the ImportError on STDOUT and stderr completely EMPTY —
    so `_HARNESS_FAILURE_CUES`, which scans stderr and only when stdout is
    blank, could never have matched it by either half of its guard.

    Graded a REFUTATION, that moves confidence down and swings the
    reproduce-before-you-fix gate open on a tooling failure — the exact
    outcome rule 4 exists to prevent, reached through the module's own
    flagship instrument. A documented exit code beats a substring search over
    the wrong stream."""
    result = sandbox.run_pytest(
        {"test_thing.py": "import definitely_not_a_real_module\n\n"
                          "def test_x():\n    assert True\n"})
    assert result["exit_code"] == 2
    assert not (result["stderr"] or "").strip(), "the premise changed"
    outcome, why = coding.judge(result, {"exit_zero": True})
    assert outcome == coding.OUTCOME_INCONCLUSIVE, why
    assert "harness" in why


def test_a_real_test_failure_is_still_a_finding(temp_db):
    """The other side, and the one that matters more: exit 1 means the tests
    RAN and failed, which is exactly what an experiment is for. Grading that
    as a harness problem would make the instrument useless in the direction it
    is most often pointed."""
    result = sandbox.run_pytest(
        {"test_thing.py": "def test_x():\n    assert 1 == 2\n"})
    assert result["exit_code"] == 1
    assert coding.judge(result, {"exit_zero": True})[0] == \
        coding.OUTCOME_REFUTED
    assert coding.judge(result, {"exit_zero": False})[0] == \
        coding.OUTCOME_CONFIRMED


def test_a_run_that_collected_no_tests_settles_nothing(temp_db):
    """pytest exit 5. Zero tests passing is not the same as the suite passing,
    and `exit_zero: false` would otherwise read as a reproduced failure."""
    result = sandbox.run_pytest({"test_empty.py": "x = 1\n"})
    assert result["exit_code"] == 5
    assert coding.judge(result, {"exit_zero": True})[0] == \
        coding.OUTCOME_INCONCLUSIVE


# ---- The archive must admit when it is a partial copy ----

def test_the_experiment_record_says_when_its_source_is_truncated(temp_db):
    """`source[:8000]` stored a program that was not the one executed with
    nothing saying so, so a reviewer reading the record back sees a clipped
    script and no reason to doubt it. Same defect class as a list capped at 40
    beside a true count. The truncation stays; the record admits it."""
    from db import q
    hyp = research.open_hypothesis("does a long program run?", turn_idx=1)
    long_source = "# padding\n" * 1200 + "print('end')\n"
    assert len(long_source) > 8000
    coding.run_experiment(hyp["id"], source=long_source,
                          expect={"stdout_has": "end"}, turn_idx=1)
    row = q("SELECT source, source_chars FROM experiments "
            "WHERE hypothesis_id=?", (hyp["id"],), one=True)
    assert len(row["source"]) == 8000, "the truncation itself changed"
    assert row["source_chars"] == len(long_source)
    assert row["source_chars"] > len(row["source"]), "the record looks whole"


# ---- The missing verb: a change that outlives the turn ----

def test_an_edit_lands_on_disk_and_returns_a_reviewable_diff(temp_db, tmp_path):
    """THE CAPABILITY GAP. Everything else here could reproduce a defect,
    design a fix and prove it correct in the sandbox — and then had nowhere to
    put it, because `sandbox.run` writes into a directory deleted the moment
    the run ends. A coding suite whose only durable artefact is an opinion
    about code is a research suite."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "thing.py", b"def f():\n    return 1\n")

    done = coding.apply_edit("thing.py", "def f():\n    return 2\n",
                             turn_idx=1, why="the off-by-one")
    assert done["ok"], done
    assert workspace.read_file("thing.py")["text"] == "def f():\n    return 2\n"
    assert "-    return 1" in done["diff"]
    assert "+    return 2" in done["diff"]
    assert done["created"] is False


def test_an_edit_recuts_the_chunks_for_the_file_it_changed(temp_db, tmp_path):
    """A MAP THAT OUTLIVES THE CODE IT DESCRIBES IS WORSE THAN NO MAP. Once
    the assistant can edit, every chunk of an edited file is a claim about
    what is there, and `expand` returning the version from before the edit
    sends it confidently to code that no longer runs."""
    import chunks
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "thing.py",
                           b'def old_name():\n    """Before."""\n    return 1\n')
    chunks.ingest_workspace()
    before = {e["gist"] for e in chunks.digest(kind="code")["entries"]}
    assert any("old_name" in g for g in before)

    coding.apply_edit("thing.py",
                      'def new_name():\n    """After."""\n    return 2\n',
                      turn_idx=1)
    after = {e["gist"] for e in chunks.digest(kind="code")["entries"]}
    assert any("new_name" in g for g in after)
    assert not any("old_name" in g for g in after), "the map is stale"


def test_an_edit_that_claims_to_fix_something_needs_a_reproduction(
        temp_db, tmp_path):
    """RULE 2, EXTENDED TO THE THING THAT ACTUALLY CHANGES. `propose_fix` was
    gated and the edit itself was not, which would have made the gate
    ceremonial the moment an edit verb existed — the record would show a
    refused fix note beside a file that had been changed anyway."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "thing.py", b"x = 1\n")
    hyp = research.open_hypothesis("is x wrong?", turn_idx=1)

    refused = coding.apply_edit("thing.py", "x = 2\n", turn_idx=1,
                                hypothesis_id=hyp["id"])
    assert refused["ok"] is False
    assert "observed failing" in refused["why"]
    assert workspace.read_file("thing.py")["text"] == "x = 1\n", "it wrote anyway"

    # Reproduce it, and the same edit goes through.
    coding.run_experiment(hyp["id"], source="import sys; sys.exit(1)",
                          expect={"exit_zero": True}, turn_idx=1)
    allowed = coding.apply_edit("thing.py", "x = 2\n", turn_idx=1,
                                hypothesis_id=hyp["id"])
    assert allowed["ok"], allowed
    assert workspace.read_file("thing.py")["text"] == "x = 2\n"


def test_an_edit_cannot_escape_the_workspace(temp_db, tmp_path):
    """A path in a model's output is untrusted input exactly like a path in an
    archive member, and this one is a write."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    for path in ("../escaped.py", "/etc/passwd", "a/../../escaped.py"):
        done = coding.apply_edit(path, "pwned", turn_idx=1)
        assert done["ok"] is False, path
    assert not (tmp_path / "escaped.py").exists()
