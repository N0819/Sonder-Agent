# coding.py — coding as the scientific method, using the machinery already here.
#
# THE FOUNDING CLAIM. Writing code is not composition, it is investigation:
# you hold a belief about what a system does, you design something that would
# come out differently depending on whether the belief is true, you run it, and
# you revise. That is the loop research.py already implements for the web —
# hypothesis, evidence, confidence, dispute — and a code run is a strictly
# better source than a web page, because it cannot be out of date, cannot be
# opinion, and answers the exact question you asked.
#
# So this module adds no epistemics. It supplies the missing SOURCE TYPE: an
# experiment whose URL is `experiment:<id>` and whose excerpt is what actually
# happened. Everything downstream — confidence movement, disputes, grounded
# citation, the read-memory that lets an old result resurface — is inherited
# unchanged.
#
# FOUR RULES, each of them a scar from Sonder.
#
# 1. A PREDICTION BEFORE A RUN, or it is not an experiment. `expect` is
#    required and is compared mechanically to the observation. Running code and
#    then deciding what it proved is how a model talks itself into a fix; it is
#    also, exactly, the "small-payload benchmark" failure recorded in
#    docs/bench-2026-08-03/RESULTS.md, where four rankings inverted because
#    nobody had said in advance what the measurement was supposed to show.
#
# 2. REPRODUCE BEFORE YOU FIX. `propose_fix` refuses a fix for a defect that
#    has never been observed failing. Sonder's CLAUDE.md puts it first among
#    working rules, and the reason is not discipline for its own sake: a fix
#    for an unreproduced defect cannot be distinguished from a fix for nothing,
#    and the suite has no way to tell you later which it was.
#
# 3. A FAILURE IS DATA, NEVER AN ERROR. A crash, a timeout, a missing
#    interpreter all come back as observations. Sonder's rule that "a warning
#    means the system WORKED" is the same instinct: the moment a harness treats
#    its own negative results as exceptions, the loop starts avoiding them.
#
# 4. CONTRADICTION IS A DISPUTE, NOT AN AVERAGE. Two runs that disagree mean
#    something is non-deterministic, and that is a finding worth surfacing
#    rather than smoothing. `run_experiment` detects it and says so.

import hashlib
import json
import re
import sys
import time

import memory
import research
import sandbox
from db import q, qi

# A prediction is met, contradicted, or the run never got far enough to say.
# The third is not a soft version of the second: "the interpreter was missing"
# tells you nothing about the hypothesis, and folding it into `contradicts`
# would move confidence on the strength of a broken harness.
OUTCOME_CONFIRMED = "confirmed"
OUTCOME_REFUTED = "refuted"
OUTCOME_INCONCLUSIVE = "inconclusive"


def _experiment_ref(digest):
    return f"experiment:{digest}"


def _digest(hypothesis_id, source, command, expect=None, files=None):
    """What makes two runs THE SAME EXPERIMENT.

    `expect` and `files` are part of the identity and were missing from it,
    so two runs of one deterministic program under DIFFERENT predictions
    collided — and the outcome differed for that reason alone, which the
    non-determinism check then reported as "the behaviour under test is not
    deterministic" and pinned the confidence to the dispute band. It
    manufactured a dispute out of nothing, inverting the invariant it was
    built to serve. `files` likewise: `{"lib.py": "V=1"}` and
    `{"lib.py": "V=2"}` were "the same experiment".

    Non-determinism now means what it says: identical inputs, identical
    prediction, different outcome."""
    payload = "|".join((
        str(hypothesis_id), source or "", json.dumps(command),
        json.dumps(expect or {}, sort_keys=True),
        json.dumps(sorted((files or {}).items()))))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# stderr shapes that mean THE HARNESS broke, not the hypothesis. Rule 4 says
# inconclusive must never fold into `contradicts`, and the arithmetic honours
# that — but `judge` only ever returned inconclusive for a missing
# interpreter, so every other harness breakage exited non-zero and was graded
# a REFUTATION: confidence moved down, and the reproduce-before-you-fix gate
# swung open, on the strength of a broken tool. Rule 4 is only as good as
# this classifier.
_HARNESS_FAILURE_CUES = (
    "no module named", "can't open file", "cannot open file",
    "permission denied", "command not found", "no such file or directory",
    "is not recognized as an internal", "could not start the command",
    "could not write ", "no such interpreter",
)


def _observation_text(result):
    """What happened, in the form the evidence row keeps.

    stderr first when the run failed, because the traceback is the finding and
    burying it under the program's own chatter is how a useful failure gets
    read as a boring one.
    """
    if result.get("timed_out"):
        head = f"TIMED OUT after {result['seconds']}s"
    elif result.get("exit_code") is None:
        head = "DID NOT RUN"
    else:
        head = f"exit {result['exit_code']} in {result['seconds']}s"
    streams = []
    if not result.get("ok") and result.get("stderr"):
        streams.append("stderr: " + result["stderr"])
    if result.get("stdout"):
        streams.append("stdout: " + result["stdout"])
    if result.get("ok") and result.get("stderr"):
        streams.append("stderr: " + result["stderr"])
    return (head + " — " + " | ".join(streams)) if streams else head


def judge(result, expect):
    """Did the observation match the prediction?

    `expect` is a dict, and every key present must hold:

        exit_zero      True/False   the command succeeded / failed
        stdout_has     substring    appears in stdout
        stdout_lacks   substring    does NOT appear in stdout
        stderr_has     substring
        output_equals  exact stripped stdout

    Deliberately mechanical. A prediction a model grades itself against is not
    a prediction, and the whole value of this module is that the grading
    happens outside the thing being graded.
    """
    if result.get("exit_code") is None and not result.get("timed_out"):
        return OUTCOME_INCONCLUSIVE, "the command never ran"
    # A broken harness says nothing about the hypothesis. The program under
    # test produced no output of its own and the interpreter complained about
    # its own ability to run: that is the tool failing, and folding it into
    # `contradicts` would move confidence on the strength of a missing
    # module. Confined to runs that produced NO stdout, so a program that
    # legitimately prints one of these strings is still judged normally.
    stderr_low = (result.get("stderr") or "").lower()
    if (not result.get("ok") and not (result.get("stdout") or "").strip()
            and any(cue in stderr_low for cue in _HARNESS_FAILURE_CUES)):
        return (OUTCOME_INCONCLUSIVE,
                "the harness failed before the hypothesis was tested: "
                + (result.get("stderr") or "").strip().splitlines()[-1][:160])
    checks = []
    if "exit_zero" in expect:
        want = bool(expect["exit_zero"])
        checks.append((bool(result.get("ok")) == want,
                       f"expected exit {'0' if want else 'non-zero'}, "
                       f"got {result.get('exit_code')}"))
    if "stdout_has" in expect:
        needle = str(expect["stdout_has"])
        checks.append((needle in (result.get("stdout") or ""),
                       f"expected {needle!r} in stdout"))
    if "stdout_lacks" in expect:
        needle = str(expect["stdout_lacks"])
        checks.append((needle not in (result.get("stdout") or ""),
                       f"expected {needle!r} absent from stdout"))
    if "stderr_has" in expect:
        needle = str(expect["stderr_has"])
        checks.append((needle in (result.get("stderr") or ""),
                       f"expected {needle!r} in stderr"))
    if "output_equals" in expect:
        want = str(expect["output_equals"]).strip()
        checks.append(((result.get("stdout") or "").strip() == want,
                       f"expected stdout to equal {want!r}"))
    if not checks:
        # No prediction is not a passing prediction.
        return OUTCOME_INCONCLUSIVE, "no prediction was stated"
    failed = [why for held, why in checks if not held]
    if failed:
        return OUTCOME_REFUTED, "; ".join(failed)
    return OUTCOME_CONFIRMED, "prediction held"


def run_experiment(hypothesis_id, *, source, expect, turn_idx,
                   files=None, command=None, timeout=None, note=""):
    """One experiment: predict, run, judge, record as evidence.

    Returns {outcome, why, result, evidence, repeated}. The evidence row is an
    ordinary `research` evidence row, so the hypothesis's confidence moves
    through the same bounded arithmetic a web source moves it through, and the
    observation becomes a `read` memory that can dispute a later claim.
    """
    if not isinstance(expect, dict) or not expect:
        return {"outcome": OUTCOME_INCONCLUSIVE,
                "why": "an experiment needs a prediction stated first",
                "result": None, "evidence": None, "repeated": False}

    # ONE code path writes the workspace, and the command RECORDED is the
    # command RUN. `payload["main.py"] = source` used to live inside the
    # `command is None` branch — where `payload` was then unused, because
    # that branch called run_python instead. So with an explicit command the
    # source under test never reached the disk: `python3 main.py` reported
    # "can't open file", judge called it a refutation, confidence went down,
    # the fix gate opened, and the experiments row recorded a `source` that
    # had never executed. The recorded command was also a literal "python3"
    # while `sys.executable` was what actually ran.
    payload = dict(files or {})
    payload["main.py"] = source
    if command is None:
        command = [sys.executable, "-s", "main.py"]
    result = sandbox.run(payload, command,
                         timeout=timeout or sandbox.DEFAULT_TIMEOUT)

    outcome, why = judge(result, expect)
    digest = _digest(hypothesis_id, source, command, expect, payload)
    observation = _observation_text(result)

    # NON-DETERMINISM IS A FINDING. The same experiment reaching a different
    # outcome than last time does not mean the newer answer is right; it means
    # the thing under test is not a function of its inputs, and averaging the
    # two would hide the only interesting fact available.
    prior = q("SELECT * FROM experiments WHERE digest=?", (digest,), one=True)
    repeated = False
    if prior and prior["outcome"] != outcome:
        repeated = True
        research.record_dispute_note(
            hypothesis_id,
            f"the same experiment produced {prior['outcome']!r} before and "
            f"{outcome!r} now — the behaviour under test is not deterministic")

    qi("INSERT INTO experiments(hypothesis_id,digest,source,command,expect,"
       "outcome,observation,note,turn_idx,created) "
       "VALUES(?,?,?,?,?,?,?,?,?,?)",
       (hypothesis_id, digest, source[:8000], json.dumps(command),
        json.dumps(expect), outcome, observation[:4000], str(note)[:400],
        turn_idx, time.time()))

    stance = {OUTCOME_CONFIRMED: "supports",
              OUTCOME_REFUTED: "contradicts"}.get(outcome, "context")
    evidence = research.record_evidence(
        hypothesis_id,
        url=_experiment_ref(digest),
        title=f"experiment ({outcome}): {note or 'code run'}",
        excerpt=f"{why}. {observation}",
        stance=stance, turn_idx=turn_idx)
    return {"outcome": outcome, "why": why, "result": result,
            "evidence": evidence, "repeated": repeated}


def observed_failing(hypothesis_id):
    """Has anything actually been seen to fail for this hypothesis?

    The gate `propose_fix` uses. An experiment counts only if it was DESIGNED
    to fail and did -- either it predicted a failure and got one, or it
    predicted success and was refuted. A run nobody made a prediction about
    proves nothing and does not open the gate.
    """
    for row in q("SELECT expect, outcome, observation, turn_idx FROM "
                 "experiments WHERE hypothesis_id=? ORDER BY id",
                 (hypothesis_id,)):
        try:
            expect = json.loads(row["expect"] or "{}")
        except (TypeError, ValueError):
            continue
        # "Refuted" means THE PREDICTION WAS WRONG, not that anything failed.
        # Accepting any refutation let a typo'd prediction open the gate:
        # `source="print('hi')", expect={"stdout_has": "goodbye"}` exits 0,
        # ok=True, outcome refuted — and propose_fix returned
        # "a failing observation exists" for a run in which nothing failed.
        # A reproduction has to carry an actual failure signal.
        if row["outcome"] == OUTCOME_REFUTED and _observation_failed(row):
            return True
        if (row["outcome"] == OUTCOME_CONFIRMED
                and expect.get("exit_zero") is False):
            return True
    return False


def _observation_failed(row):
    """Did the recorded run actually fail, as opposed to merely surprising
    whoever predicted it? Read from the stored observation, whose first token
    is written by `_observation_text`."""
    head = str(row["observation"] or "")
    if head.startswith("TIMED OUT") or head.startswith("DID NOT RUN"):
        return True
    match = re.match(r"exit (-?\d+)", head)
    return bool(match) and match.group(1) != "0"


def propose_fix(hypothesis_id, *, description, turn_idx):
    """Record an intended fix, or refuse because nothing has been reproduced.

    REPRODUCE BEFORE YOU FIX. A fix for a defect never observed failing cannot
    be distinguished afterwards from a fix for nothing, and the record cannot
    tell you which it was. Refusing here is cheaper than a codebase full of
    changes whose necessity nobody can reconstruct.
    """
    if not observed_failing(hypothesis_id):
        return {"accepted": False,
                "why": "nothing has been observed failing yet — write an "
                       "experiment that reproduces the defect first"}
    # EACH FIX CONSUMES ITS OWN REPRODUCTION. The gate was per-hypothesis and
    # never spent: after one genuine repro and one accepted fix, every later
    # `propose_fix` on that hypothesis was accepted forever with no new
    # experiment — "reproduce before you fix" holding for the first fix and
    # for no fix after it.
    # Keyed on the experiments row id, not on turn_idx: a reproduction and
    # the fix it justifies routinely land in the SAME turn, so a turn-based
    # cursor let that one reproduction go on justifying every later fix.
    last_fix = q("SELECT MAX(id) AS i FROM experiments "
                 "WHERE hypothesis_id=? AND outcome='fix-marker'",
                 (hypothesis_id,), one=True)
    since = int((last_fix["i"] if last_fix else 0) or 0)
    if not _failing_since(hypothesis_id, since):
        return {"accepted": False,
                "why": "the failure this would fix was already answered by a "
                       "previous fix — reproduce it again before changing "
                       "anything else"}
    # The fix is recorded BESIDE the hypothesis, not on top of it. Writing it
    # into `statement` destroyed the claim every existing evidence row had
    # been recorded against.
    qi("INSERT INTO experiments(hypothesis_id,digest,source,command,expect,"
       "outcome,observation,note,turn_idx,created) "
       "VALUES(?,?,?,?,?,?,?,?,?,?)",
       (hypothesis_id, f"fix:{hypothesis_id}:{turn_idx}", "", "[]", "{}",
        "fix-marker", "proposed fix", str(description)[:400], turn_idx,
        time.time()))
    memory.add_memory(
        "semantic", "inferred", 0.7,
        f"Proposed fix after reproducing the defect: {description}",
        turn_idx=turn_idx, confidence=0.7,
        event_key=f"fix:{hypothesis_id}:{turn_idx}")
    return {"accepted": True, "why": "a failing observation exists"}


def _failing_since(hypothesis_id, row_id):
    for row in q("SELECT expect, outcome, observation FROM "
                 "experiments WHERE hypothesis_id=? AND id>?",
                 (hypothesis_id, row_id)):
        try:
            expect = json.loads(row["expect"] or "{}")
        except (TypeError, ValueError):
            continue
        if row["outcome"] == OUTCOME_REFUTED and _observation_failed(row):
            return True
        if (row["outcome"] == OUTCOME_CONFIRMED
                and expect.get("exit_zero") is False):
            return True
    return False


def experiments_for(hypothesis_id, limit=20):
    return [dict(r) for r in q(
        "SELECT id,digest,outcome,observation,note,turn_idx,created "
        "FROM experiments WHERE hypothesis_id=? ORDER BY id DESC LIMIT ?",
        (hypothesis_id, limit))]
