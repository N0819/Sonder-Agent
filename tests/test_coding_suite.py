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

import sys

import coding
import memory
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


def test_pytest_works_through_the_path_experiments_actually_take(temp_db):
    """DEAD CODE ON THE LIVE PATH, and the unit tests could not see it.

    `run_pytest` made pytest importable and stamped `harness`, and
    `coding.run_experiment` calls `sandbox.run` directly — so an experiment
    that invoked pytest by naming it in `command`, which is the only way the
    assistant can, got neither. Measured before the fix: "No module named
    pytest", exit 1, every time; and `_PYTEST_HARNESS_EXITS` unreachable, its
    own tests passing because they set `harness` themselves.

    Asserted through `sandbox.run` with a model-written argv, never through
    `run_pytest`, because using the convenience function is precisely what
    hid this."""
    argv = [sandbox.sys.executable, "-s", "-m", "pytest", "-q", "--no-header"]
    passing = sandbox.run({"test_x.py": "def test_a():\n    assert True\n"},
                          argv)
    assert passing["harness"] == "pytest"
    assert passing["exit_code"] == 0, passing["stderr"][:200]

    # ...and with the harness reachable, the exit-code table finally applies
    # to a run the assistant could actually have written.
    broken = sandbox.run({"test_x.py": "import nope_not_real\n"}, argv)
    assert broken["exit_code"] == 2
    outcome, why = coding.judge(broken, {"exit_zero": True})
    assert outcome == coding.OUTCOME_INCONCLUSIVE, why


def test_an_ordinary_run_is_not_labelled_a_harness(temp_db):
    """The stamp is keyed on the command, so it has to stay silent for
    everything that is not a test run — a plain script mislabelled `pytest`
    would have its exit codes reinterpreted and a real failure graded away."""
    plain = sandbox.run({"main.py": "import sys; sys.exit(2)"},
                        [sandbox.sys.executable, "-s", "main.py"])
    assert plain["harness"] == ""
    assert coding.judge(plain, {"exit_zero": True})[0] == coding.OUTCOME_REFUTED


def test_an_experiment_may_run_the_projects_own_suite(temp_db, tmp_path):
    """The end the whole chain serves: the assistant verifying with the tests
    that are already in the repo instead of hand-rolling a harness that
    reimplements them."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    hyp = research.open_hypothesis("does the helper return 2?", turn_idx=1)
    out = coding.run_experiment(
        hyp["id"], source="",
        files={"lib.py": "def f():\n    return 2\n",
               "test_lib.py": "from lib import f\n\n"
                              "def test_f():\n    assert f() == 2\n"},
        command=[sandbox.sys.executable, "-s", "-m", "pytest", "-q",
                 "--no-header"],
        expect={"exit_zero": True, "stdout_has": "1 passed"}, turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_CONFIRMED, out["why"]


# ---- An anchored edit, for a file read in pieces ----

def test_an_anchored_edit_changes_only_what_it_names(temp_db, tmp_path):
    """Whole-file replacement asks the assistant to reproduce every line it is
    NOT changing, from memory. The failure is a confident-looking rewrite that
    silently drops one — invisible in a diff summary, invisible in the tests
    that do not cover it, and indistinguishable afterwards from an intended
    deletion. An anchor changes what it names and nothing else."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    original = ("import os\n\n\ndef alpha():\n    return 1\n\n\n"
                "def beta():\n    return 2\n")
    workspace.store_upload(1, "m.py", original.encode())

    done = coding.apply_edit(
        "m.py", turn_idx=1,
        replace=[{"old": "def beta():\n    return 2",
                  "new": "def beta():\n    return 22"}])
    assert done["ok"], done
    after = workspace.read_file("m.py")["text"]
    assert after == original.replace("return 2\n", "return 22\n")
    assert "def alpha():\n    return 1" in after, "an untouched line moved"


def test_an_anchor_that_is_not_unique_writes_nothing(temp_db, tmp_path):
    """Zero matches means the anchor is stale and the edit would land
    somewhere unintended; more than one means the assistant is describing a
    PATTERN while believing it is naming a PLACE. Both need the count, because
    "it did not apply" and "it applied three times" want opposite fixes."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    original = "x = 1\ny = 1\nz = 1\n"
    workspace.store_upload(1, "m.py", original.encode())

    missing = coding.apply_edit("m.py", turn_idx=1,
                                replace=[{"old": "q = 9", "new": "q = 8"}])
    assert missing["ok"] is False
    assert "0 times" in missing["why"]
    assert "read it again" in missing["why"]

    ambiguous = coding.apply_edit("m.py", turn_idx=1,
                                  replace=[{"old": "= 1", "new": "= 2"}])
    assert ambiguous["ok"] is False
    assert "3 times" in ambiguous["why"]
    assert "unique" in ambiguous["why"]
    # Neither attempt touched the file.
    assert workspace.read_file("m.py")["text"] == original


def test_a_failed_anchor_late_in_a_batch_writes_none_of_it(temp_db, tmp_path):
    """All or nothing. A batch that applied its first two edits and then
    refused the third would leave the file in a state nobody designed — and
    the assistant would be told the edit failed while half of it had
    happened."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    original = "a = 1\nb = 2\nc = 3\n"
    workspace.store_upload(1, "m.py", original.encode())
    done = coding.apply_edit("m.py", turn_idx=1, replace=[
        {"old": "a = 1", "new": "a = 9"},
        {"old": "b = 2", "new": "b = 9"},
        {"old": "nonexistent", "new": "boom"}])
    assert done["ok"] is False
    assert workspace.read_file("m.py")["text"] == original


def test_an_edit_with_neither_contents_nor_replace_is_refused(temp_db,
                                                              tmp_path):
    """`contents=None` used to be the only signal, and None is also what an
    absent key looks like — so a malformed edit would have written an empty
    file over a real one."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "m.py", b"keep me\n")
    done = coding.apply_edit("m.py", turn_idx=1)
    assert done["ok"] is False
    assert workspace.read_file("m.py")["text"] == "keep me\n"


def test_a_harness_that_died_after_the_program_spoke_settles_nothing(temp_db):
    """THE SAME BROKEN TOOL, GRADED TWO WAYS. The cue scan was confined to
    runs that produced no stdout, so a missing import was INCONCLUSIVE when
    the program was silent and a REFUTATION when it had printed one line
    first — confidence down and the fix gate open, on a tooling failure."""
    result = {"ok": False, "exit_code": 1, "stdout": "checking widget\n",
              "stderr": "Traceback (most recent call last):\n"
                        "    import widget\n"
                        "ModuleNotFoundError: No module named 'widget'\n",
              "timed_out": False}

    outcome, why = coding.judge(result, {"exit_zero": True,
                                         "stdout_has": "checking widget"})

    assert outcome == coding.OUTCOME_INCONCLUSIVE, why
    assert "harness" in why


def test_a_program_that_prints_a_cue_string_is_still_judged_on_its_merits(
        temp_db):
    """THE CASE THE OLD GUARD PROTECTED, and the reason the fix separates the
    streams instead of widening the scan: a test asserting on the text "no
    such file or directory" prints it to STDOUT, so it never reaches a
    stderr-scoped guard and needs no second special case."""
    passing = {"ok": True, "exit_code": 0,
               "stdout": "no such file or directory\n", "stderr": "",
               "timed_out": False}
    failing = dict(passing, ok=False, exit_code=1)

    assert (coding.judge(passing, {"exit_zero": True})[0]
            == coding.OUTCOME_CONFIRMED)
    assert (coding.judge(failing, {"exit_zero": True})[0]
            == coding.OUTCOME_REFUTED)


def test_an_experiment_can_ask_for_the_time_its_suite_needs(temp_db):
    """A CEILING THAT FORCES A WEAKER QUESTION. 20s is a sound default and was
    also the maximum, with no way for a spec to raise it — and this project's
    own suite takes ~34s, so "run your tests and show me they are green" was
    unreachable through the one verb that can run them.

    Measured on a live audit: two experiments timed out at 20.0s and the
    assistant fell back to `-x`, which stops at the first failure and so can
    report that something is broken but never that everything passes."""
    slow = "import time\ntime.sleep(2.5)\nprint('finished')\n"
    assert sandbox.run({"main.py": slow},
                       [sandbox.sys.executable, "-s", "main.py"],
                       timeout=1.0)["timed_out"] is True

    out = sandbox.run({"main.py": slow},
                      [sandbox.sys.executable, "-s", "main.py"], timeout=30.0)
    assert out["timed_out"] is False
    assert "finished" in out["stdout"]


def test_the_ceiling_is_applied_where_it_cannot_be_forgotten(temp_db):
    """Clamped inside `run`, not at each caller. A guard every caller must
    remember to apply is one the next caller will omit — and the omission
    would be a sandbox with no upper bound at all, which is the failure this
    limit exists to prevent."""
    out = sandbox.run({"main.py": "print('hi')\n"},
                      [sandbox.sys.executable, "-s", "main.py"],
                      timeout=10_000)
    assert out["ok"] is True
    assert sandbox.MAX_TIMEOUT < 10_000


def test_a_write_that_changed_nothing_is_not_reported_as_an_edit(temp_db,
                                                                 tmp_path):
    """THE MOST EXPENSIVE SILENT SUCCESS IN THE SUITE. `unchanged` was
    returned, carried into the trace and read by nothing, so a no-op write
    reported "edited m.py" with an empty diff and minted a `witnessed`
    memory saying so. The assistant then recalled making a change that was
    never on disk and defended it across two turns against the file itself,
    because its own memory was the evidence."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "m.py", b"a = 1\n")

    done = coding.apply_edit("m.py", "a = 1\n", turn_idx=1, why="no-op")
    assert done["ok"] is False
    assert "already reads" in done["why"]
    minted = memory.q("SELECT COUNT(*) AS n FROM memories "
                      "WHERE event_key='edit:1:m.py'", one=True)
    assert minted["n"] == 0


def test_a_replacement_that_produces_the_original_text_is_refused(temp_db,
                                                                  tmp_path):
    """The anchored path reaches the same no-op by a different road: every
    replacement matches exactly once, so nothing is refused upstream, and the
    text that comes out is the text that went in."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "m.py", b"a = 1\n")

    done = coding.apply_edit("m.py", turn_idx=1,
                             replace=[{"old": "a = 1", "new": "a = 1"}])
    assert done["ok"] is False
    assert workspace.read_file("m.py")["text"] == "a = 1\n"


def test_a_binary_in_the_workspace_does_not_break_the_experiment(temp_db):
    """The workspace could not carry a SQLite database, so "run this project's
    suite" failed at the first query on any project that keeps state on disk.
    Once it could, `_digest` — the identity function, several frames below the
    caller — raised `Object of type bytes is not JSON serializable`, and the
    experiment never ran at all. The turn reported `experiment harness failed`
    and named neither the file nor the stage."""
    hyp = research.open_hypothesis("does the binary reach the sandbox?", 1)
    out = coding.run_experiment(
        hyp["id"],
        source="import os\nprint('SIZE', os.path.getsize('state.db'))",
        files={"state.db": b"SQLite format 3\x00\x01\x02"},
        expect={"output_equals": "SIZE 18"}, turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_CONFIRMED, out["why"]


def test_two_different_databases_are_two_different_experiments(temp_db):
    """A binary belongs in the experiment's identity for the same reason
    `{"lib.py": "V=1"}` and `{"lib.py": "V=2"}` do: collapsing them makes one
    run's outcome look like the other's non-determinism, which manufactures a
    dispute out of nothing."""
    first = coding._digest(1, "x", ["python3"], {}, {"a.db": b"\x00one"})
    second = coding._digest(1, "x", ["python3"], {}, {"a.db": b"\x00two"})
    assert first != second


def test_pytest_is_reachable_from_a_program_that_shells_out_to_it(temp_db):
    """The way you run a suite and capture its output is to write a program
    that invokes pytest itself — so the argv is `python3 -s main.py` and the
    fold keyed on the outer command never fires. Measured in production: the
    nested interpreter inherited a PYTHONPATH holding only the workspace and
    returned `No module named pytest`, the same sentence the fold was written
    to end. The outer process exited 0, because the program caught the failure
    and printed it as data, so both harness guards in `judge` were blind to it
    and a broken tool was graded a refutation."""
    source = (
        "import subprocess, sys\n"
        "out = subprocess.run([sys.executable, '-s', '-m', 'pytest',"
        " '--version'], capture_output=True, text=True)\n"
        "print('RC', out.returncode)\n"
        "print('ERR', out.stderr.strip()[:80])\n")
    result = sandbox.run({"main.py": source}, [sys.executable, "-s", "main.py"])
    assert "No module named pytest" not in result["stdout"], result["stdout"]
    assert "RC 0" in result["stdout"], result["stdout"]


def test_a_nested_pytest_does_not_restamp_the_outer_harness(temp_db):
    """`harness` decides how the OUTER run is graded — `_PYTEST_HARNESS_EXITS`
    reads the outer exit code — and a program that happens to invoke pytest
    inside itself is not a pytest run. Only the path is unconditional."""
    result = sandbox.run({"main.py": "print('hi')"},
                         [sys.executable, "-s", "main.py"])
    assert result["harness"] == ""


def test_a_long_observation_keeps_the_end_a_test_runner_puts_the_answer_in(temp_db):
    """`observation[:4000]` kept the head, which is the wrong end for every
    test runner there is. `sandbox._tail` truncates each stream from the FRONT
    so the summary survives, and this cut the summary straight back off.
    Observed on a real collect run: 9.9 seconds, the full list of collected
    tests stored, and the one line saying how many were collected and how many
    errored was the line that did not make it into the row."""
    body = "\n".join(f"tests/test_thing.py::test_case_{n}" for n in range(600))
    text = f"exit 1 in 9.9s — stdout: {body}\n3 errors, 4571 tests collected"
    assert len(text) > coding.OBSERVATION_CHARS, "fixture must exceed the cap"
    fitted = coding._fit_observation(text)
    assert len(fitted) <= coding.OBSERVATION_CHARS
    assert fitted.endswith("3 errors, 4571 tests collected")
    # The head is what tells a crash from a clean run with noisy output.
    assert fitted.startswith("exit 1 in 9.9s")
    # Elision declared, not silent — a reader must not read the join as real.
    assert "elided from the middle" in fitted


def test_a_short_observation_is_stored_exactly(temp_db):
    """The common case must not grow a marker it did not earn."""
    assert coding._fit_observation("exit 0 in 0.1s — stdout: ok") == \
        "exit 0 in 0.1s — stdout: ok"


def test_the_excerpt_the_model_reads_keeps_the_counts(temp_db):
    """The 4,000-char experiments row is for forensics; the EVIDENCE excerpt
    is what the model reads back, and `record_evidence` cuts head-first —
    right for a web page, exactly wrong for a test runner, which writes
    `N passed, M failed` last. Measured on a real suite run: 19,776 characters
    of stdout, of which the excerpt kept the first 600 — a mid-list slice of
    node ids, no counts, no error roster. The assistant reported the count as
    unreachable and rebuilt its instrument around the gap."""
    hyp = research.open_hypothesis("does the suite pass?", 1)
    noise = "\n".join(f"tests/test_m.py::case_{n}" for n in range(900))
    out = coding.run_experiment(
        hyp["id"],
        source=f"print({noise!r})\nprint('4571 passed, 3 failed')",
        expect={"stdout_contains_nothing_useful": 1,
                "exit_zero": True}, turn_idx=1)
    from db import q
    row = q("SELECT excerpt, excerpt_chars FROM evidence WHERE id=?",
            (out["evidence"]["id"],), one=True)
    assert "4571 passed, 3 failed" in row["excerpt"]
    assert len(row["excerpt"]) <= research.EXCERPT_CHARS
    assert "elided from the middle" in row["excerpt"]


def test_a_budget_smaller_than_the_head_still_leaves_a_tail(temp_db):
    """A fixed head wider than the budget yields a negative tail — a slice
    from the front wearing the marker of a slice from both ends."""
    text = "HEAD" + ("x" * 5000) + "TAIL"
    fitted = coding._fit_observation(text, 200)
    assert len(fitted) <= 200
    assert fitted.endswith("TAIL")


def test_an_experiment_can_run_inside_the_project_it_is_testing(temp_db):
    """A project is not a loose pile of files. An unpacked repository sits at
    `<name>/<name>/`, and its own code resolves paths against the process
    directory — the Engine mounts `StaticFiles(directory="static")` at import
    time — so a suite invoked from the workspace root failed on every module
    with `Directory 'static' does not exist` no matter how completely the
    files were delivered. Measured: 24 modules, 226 tests. The only way to say
    "run in that directory" was `sh -c "cd X && ..."`, which makes the command
    `sh` and so loses the pytest provisioning keyed on it — two workarounds
    cancelling out, and four turns spent between them."""
    files = {"proj/static/index.html": "<!doctype html>",
             "proj/app.py": "import os\nprint('STATIC', os.path.isdir('static'))"}
    outside = sandbox.run(files, [sys.executable, "-s", "proj/app.py"])
    assert "STATIC False" in outside["stdout"]
    inside = sandbox.run(files, [sys.executable, "-s", "app.py"], cwd="proj")
    assert "STATIC True" in inside["stdout"]


def test_a_run_directory_outside_the_workspace_is_refused(temp_db):
    """A generated path is untrusted input here as everywhere else — and a cwd
    that quietly fell back to the root would reproduce the exact failure the
    argument exists to end, with the argument set."""
    out = sandbox.run({"main.py": "print('hi')"},
                      [sys.executable, "-s", "main.py"], cwd="../../etc")
    assert out["ok"] is False and "refused to run outside" in out["stderr"]
    missing = sandbox.run({"main.py": "print('hi')"},
                          [sys.executable, "-s", "main.py"], cwd="nope")
    assert missing["ok"] is False and "no such directory" in missing["stderr"]


def test_a_file_can_be_read_back_without_predicting_its_contents(temp_db):
    """Deriving the collected list from the prediction is what stops a file
    predicate being judged against a file nobody read back — but it also made
    "give me that file" inexpressible without inventing a prediction about a
    number not yet known. A measurement run wrote its counts to a ladder file
    and had no way to ask for it: four turns of writing predicates about files
    that were never returned, each recorded as "the run left no X to check"."""
    hyp = research.open_hypothesis("what does the run count?", 1)
    out = coding.run_experiment(
        hyp["id"],
        source="open('counts.txt','w').write('FAILED=60 ERRORS=9')\nprint('ok')",
        expect={"exit_zero": True}, collect=["counts.txt"], turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_CONFIRMED
    assert out["result"]["files_after"]["counts.txt"] == "FAILED=60 ERRORS=9"


def test_source_overwriting_your_own_file_is_reported(temp_db):
    """`source` always lands in main.py and silently overwrote whatever the
    caller put there. A caller who named its program `probe.py` in `files` and
    passed the real body as `source` got both — a stub at probe.py and the
    body at main.py — and the command it wrote ran the stub. Observed: a probe
    whose entire output was `NameError: PLACEHOLDER_REPLACED_BY_SOURCE`."""
    hyp = research.open_hypothesis("does the collision surface?", 1)
    out = coding.run_experiment(
        hyp["id"], source="print('the real body')",
        files={"main.py": "PLACEHOLDER"},
        expect={"output_equals": "the real body"}, turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_CONFIRMED
    assert out["shadowed"] == "main.py"
    # Not resolved, reported: guessing which one was meant is how the wrong
    # file wins quietly.
    clean = coding.run_experiment(
        hyp["id"], source="print('x')", expect={"exit_zero": True}, turn_idx=1)
    assert clean["shadowed"] == ""
