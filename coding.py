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
import workspace
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
        json.dumps(sorted((path, _identity_of(body))
                          for path, body in (files or {}).items()))))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _identity_of(body):
    """A file's contribution to the digest.

    A binary is hashed rather than carried: it belongs in the identity — two
    runs against different databases are different experiments — but a 2 MB
    database inlined into the string that is then hashed is 2 MB of work per
    digest, and `sorted()` would be comparing bytes against str the moment a
    workspace holds both. Once the workspace could carry a SQLite file at all,
    `json.dumps` on the raw dict raised `Object of type bytes is not JSON
    serializable` — out of the identity function, so the experiment never ran
    and the failure named the harness rather than the file that caused it."""
    if isinstance(body, (bytes, bytearray)):
        return "sha1:" + hashlib.sha1(bytes(body)).hexdigest()
    return body


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


OBSERVATION_CHARS = 4000
# Enough for the verdict line and the start of whatever spoke first.
_OBSERVATION_HEAD = 1200


def _fit_observation(text, limit=OBSERVATION_CHARS):
    """Keep both ends of a long observation, and say what fell out.

    `observation[:4000]` kept the HEAD, which is the wrong end for every test
    runner there is: `sandbox._tail` deliberately truncates each stream from
    the front so the SUMMARY survives, and then this cut the summary back off.
    Observed on a real collect run — 9.9 seconds, a full list of collected
    tests, and the one line saying how many were collected and how many errored
    was the line that did not make it into the row.

    Head as well as tail because the head carries `exit N in Xs` and the first
    thing either stream said, and a reader who loses that cannot tell a crash
    from a clean run with noisy output."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    marker = f"\n… [{len(text) - limit:,} chars elided from the middle] …\n"
    # A THIRD OF WHATEVER BUDGET IT IS GIVEN. The head is a fixed 1,200 for
    # the 4,000-char row, but the evidence excerpt is 600 — and a fixed head
    # wider than the budget yields a negative tail, which is a slice from the
    # front wearing the marker of a slice from both ends. The tail must always
    # be the larger share: it is the half that carries the answer.
    head = min(_OBSERVATION_HEAD, max(0, (limit - len(marker)) // 3))
    tail = limit - head - len(marker)
    if tail <= 0:                       # a budget too small to say anything
        return text[:limit]
    return text[:head] + marker + text[-tail:]


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


# pytest's documented exit codes. 1 is a FINDING — tests ran and failed, which
# is exactly what an experiment is for. Everything above it means the run never
# reached the hypothesis.
_PYTEST_HARNESS_EXITS = {
    2: "the test run was interrupted before it finished collecting",
    3: "an internal pytest error",
    4: "pytest was invoked wrongly",
    5: "no tests were collected, so nothing was tested",
}


def _expected_paths(expect):
    """Files a prediction talks about, so the runner knows to read them back.

    Derived from the prediction rather than named separately by the caller.
    A second field the model must remember to fill is a field it will
    eventually forget, and the failure would be a file predicate silently
    judged against an absent file."""
    paths = []
    for key in ("file_contains", "file_equals", "file_lacks"):
        spec = expect.get(key)
        if isinstance(spec, dict):
            paths.extend(str(p) for p in spec)
    return sorted(set(paths))


def judge(result, expect):
    """Did the observation match the prediction?

    `expect` is a dict, and every key present must hold:

        exit_zero      True/False   the command succeeded / failed
        exit_code      int          the exact status, when zero/non-zero is
                                    too coarse to state what you predicted
        stdout_has     substring    appears in stdout
        stdout_lacks   substring    does NOT appear in stdout
        stdout_matches regex        matches stdout (re.search)
        stderr_has     substring
        stderr_lacks   substring
        output_equals  exact stripped stdout
        file_contains  {path: substring}   in a file the run left behind
        file_lacks     {path: substring}
        file_equals    {path: exact stripped contents}

    THE VOCABULARY IS THE CEILING ON WHAT CAN BE INVESTIGATED. Every claim an
    experiment can ever make has to be expressible here, and with five keys
    the honest ones were unreachable: "the patch applied and the file now
    reads X" could only be approximated by having the program print the file
    back, which tests the print statement alongside the patch. The file
    predicates are the ones that make a durable edit checkable.

    Additive by construction — each key is read only when present, so a
    prediction written against the old vocabulary is graded identically.

    Deliberately mechanical. A prediction a model grades itself against is not
    a prediction, and the whole value of this module is that the grading
    happens outside the thing being graded.
    """
    if result.get("exit_code") is None and not result.get("timed_out"):
        return OUTCOME_INCONCLUSIVE, "the command never ran"
    # THE SHARPEST INSTRUMENT WAS THE ONE MOST OFTEN MISGRADED, and the cue
    # scan below could never have caught it: pytest writes collection errors
    # to STDOUT and leaves stderr completely empty, so a test module that
    # cannot import came back exit 2, stderr "", and was graded a REFUTATION
    # of whatever hypothesis it was testing — confidence down and the
    # reproduce-before-you-fix gate open, on a tooling failure. Reproduced
    # before it was fixed: `import definitely_not_a_real_module` in a test
    # file, judged `refuted` against `{"exit_zero": True}`.
    #
    # A documented exit code beats a substring search over the wrong stream.
    # Where the engine can decide, it decides.
    if result.get("harness") == "pytest":
        why = _PYTEST_HARNESS_EXITS.get(result.get("exit_code"))
        if why:
            return (OUTCOME_INCONCLUSIVE,
                    f"the test harness did not run the hypothesis: {why}")
    # A broken harness says nothing about the hypothesis. The program under
    # test produced no output of its own and the interpreter complained about
    # its own ability to run: that is the tool failing, and folding it into
    # `contradicts` would move confidence on the strength of a missing
    # module. Confined to runs that produced NO stdout, so a program that
    # legitimately prints one of these strings is still judged normally.
    # A HARNESS THAT DIED AFTER THE PROGRAM SPOKE IS STILL A BROKEN HARNESS.
    # The guard below is confined to runs that produced NO stdout, so the same
    # missing import was INCONCLUSIVE when the program was silent and a
    # REFUTATION when it had printed one line first: same broken tool,
    # opposite verdict, decided by whether the program happened to speak.
    # Confidence moved down and the reproduce-before-you-fix gate swung open,
    # on a tooling failure.
    #
    # Separated by STREAM rather than by emptiness. The tooling complains on
    # stderr and complains LAST, so only the final non-empty stderr line is
    # scanned, and only when the run actually exited non-zero. The case the
    # guard below protects survives by construction: a test asserting on the
    # text "no such file or directory" prints it to STDOUT and never reaches
    # here, and a program that merely mentions a cue mid-stderr is judged on
    # its merits. Residual, stated rather than hidden: a failing program whose
    # own last stderr line ends with a cue phrase is still misgraded, and
    # closing that needs the sandbox to say which side wrote the failure.
    stderr_lines = [ln for ln in (result.get("stderr") or "").splitlines()
                    if ln.strip()]
    if stderr_lines and result.get("exit_code") not in (0, None):
        if any(cue in stderr_lines[-1].lower()
               for cue in _HARNESS_FAILURE_CUES):
            return (OUTCOME_INCONCLUSIVE,
                    "the harness failed before the hypothesis was tested: "
                    + stderr_lines[-1].strip()[:160])
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
    if "exit_code" in expect:
        try:
            want_code = int(expect["exit_code"])
        except (TypeError, ValueError):
            return (OUTCOME_INCONCLUSIVE,
                    f"exit_code must be an integer, got "
                    f"{expect['exit_code']!r}")
        checks.append((result.get("exit_code") == want_code,
                       f"expected exit {want_code}, "
                       f"got {result.get('exit_code')}"))
    if "stderr_lacks" in expect:
        needle = str(expect["stderr_lacks"])
        checks.append((needle not in (result.get("stderr") or ""),
                       f"expected {needle!r} absent from stderr"))
    if "stdout_matches" in expect:
        pattern = str(expect["stdout_matches"])
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            # A malformed prediction is a broken instrument, not a refutation
            # — the same rule that keeps a missing interpreter out of
            # `contradicts`. Grading it against the hypothesis would move
            # confidence on the strength of a typo.
            return (OUTCOME_INCONCLUSIVE,
                    f"the prediction's regex does not compile: {exc}")
        checks.append((bool(rx.search(result.get("stdout") or "")),
                       f"expected stdout to match {pattern!r}"))
    if "output_equals" in expect:
        want = str(expect["output_equals"]).strip()
        checks.append(((result.get("stdout") or "").strip() == want,
                       f"expected stdout to equal {want!r}"))
    # A file predicate against a run that never wrote the file is INCONCLUSIVE,
    # not refuted. "The file says something else" and "there is no file" are
    # different findings, and only the first is about the hypothesis.
    after = result.get("files_after")
    if after is None:
        after = {}
    for key in ("file_contains", "file_lacks", "file_equals"):
        spec = expect.get(key)
        if not isinstance(spec, dict):
            continue
        for path, wanted in spec.items():
            path = str(path)
            if path not in after:
                return (OUTCOME_INCONCLUSIVE,
                        f"the run left no {path!r} to check, so the "
                        f"prediction about it was never tested")
            body = after[path]
            if key == "file_contains":
                checks.append((str(wanted) in body,
                               f"expected {str(wanted)!r} in {path!r}"))
            elif key == "file_lacks":
                checks.append((str(wanted) not in body,
                               f"expected {str(wanted)!r} absent from "
                               f"{path!r}"))
            else:
                checks.append((body.strip() == str(wanted).strip(),
                               f"expected {path!r} to equal {str(wanted)!r}"))
    if not checks:
        # No prediction is not a passing prediction.
        return OUTCOME_INCONCLUSIVE, "no prediction was stated"
    failed = [why for held, why in checks if not held]
    if failed:
        return OUTCOME_REFUTED, "; ".join(failed)
    return OUTCOME_CONFIRMED, "prediction held"


def run_experiment(hypothesis_id, *, source="", expect, turn_idx,
                   files=None, command=None, timeout=None, note="",
                   cwd="", collect=(), session_id=None):
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
    source = str(source or "")

    # ONE code path writes the workspace, and the command RECORDED is the
    # command RUN. `payload["main.py"] = source` used to live inside the
    # `command is None` branch — where `payload` was then unused, because
    # that branch called run_python instead. So with an explicit command the
    # source under test never reached the disk: `python3 main.py` reported
    # "can't open file", judge called it a refutation, confidence went down,
    # the fix gate opened, and the experiments row recorded a `source` that
    # had never executed. The recorded command was also a literal "python3"
    # while `sys.executable` was what actually ran.
    # THE MOST OBVIOUS EXPERIMENT IN THE REPOSITORY WAS THE ONE IT COULD NOT
    # RUN. `source` was required and always written to main.py, so "run the
    # suite that is already here" had to invent a program in order to ask a
    # question about programs that exist. The assistant tried it twice in one
    # turn — a baseline `pytest tests/` and its two new tests by name — and
    # both were dropped before they executed, which is why it could report a
    # fix landing and no run confirming it. An experiment over the workspace
    # as it stands is the normal case for a repository, not a special one.
    #
    # A named command needs no source. Only the bare `python3 main.py` default
    # does, and that is the one path that now insists on it.
    payload = dict(files or {})
    shadowed = ""
    if str(source).strip():
        # `source` ALWAYS lands in main.py, and silently overwrote whatever
        # the caller had put there. Worse in the other direction: a caller who
        # named its program `probe.py` in `files` and passed the real body as
        # `source` got both — a stub at probe.py and the body at main.py — and
        # the command it wrote ran the stub. Observed: a probe whose entire
        # output was `NameError: PLACEHOLDER_REPLACED_BY_SOURCE`. Silent
        # either way, so it is reported rather than resolved: guessing which
        # one was meant is how the wrong file wins quietly.
        if payload.get("main.py") not in (None, source):
            shadowed = "main.py"
        payload["main.py"] = source
    elif command is None:
        return {"outcome": OUTCOME_INCONCLUSIVE,
                "why": "an experiment needs either source to run or a command "
                       "naming what to run",
                "result": None, "evidence": None, "repeated": False}
    if command is None:
        command = [sys.executable, "-s", "main.py"]
    # `collect` UNION the predicted paths, never instead of them. Deriving
    # the list from the prediction is what stops a file predicate being judged
    # against a file nobody read back — but it also made "give me that file"
    # inexpressible without inventing a prediction about contents not yet
    # known. A measurement run wrote its counts to a ladder file and had no way
    # to ask for it: four turns of writing predicates about files that were
    # never returned, recorded as "the run left no X to check".
    want = sorted(set(_expected_paths(expect))
                  | {str(p) for p in (collect or ())})
    result = sandbox.run(payload, command,
                         timeout=timeout or sandbox.DEFAULT_TIMEOUT,
                         collect=want, cwd=cwd)

    outcome, why = judge(result, expect)
    digest = _digest(hypothesis_id, source, command, expect, payload)
    observation = _observation_text(result)

    # AND NOW SAY WHERE THEY WENT. Gathering the files and telling nobody is
    # what made `collect` look like a broken transport for three turns: the
    # grader scored predicates against contents the caller was never shown,
    # so "the run left no counts.txt to check" and "the file is here and you
    # cannot see it" were the same observation. The location goes on the END
    # of the text because the excerpt that gets read back keeps ~375
    # characters of tail against ~187 of head — the last line survives, the
    # middle does not.
    stored = None
    if session_id is not None and (result or {}).get("files_after"):
        try:
            stored = workspace.store_run_output(session_id, digest,
                                                result["files_after"])
        except OSError as exc:
            observation += f"\n[collected files could not be stored: {exc}]"
    if stored:
        names = ", ".join(sorted(stored["files"]))
        observation += f"\ncollected → {stored['dir']}/ ({names})"

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

    # THE ARCHIVE SAYS WHEN IT IS A PARTIAL COPY. `source[:8000]` stored a
    # program that was not the one executed, with nothing anywhere saying so —
    # for anything file-sized, a reviewer reading the record back sees a
    # truncated script and no reason to doubt it. Same defect class as a
    # `skipped` list capped at 40 beside a true `skipped_count`. The
    # truncation stays (the row is a record, not a filesystem); what changes
    # is that the record admits it.
    qi("INSERT INTO experiments(hypothesis_id,digest,source,source_chars,"
       "command,expect,outcome,observation,note,turn_idx,created) "
       "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
       (hypothesis_id, digest, source[:8000], len(source), json.dumps(command),
        json.dumps(expect), outcome, _fit_observation(observation),
        str(note)[:400],
        turn_idx, time.time()))

    stance = {OUTCOME_CONFIRMED: "supports",
              OUTCOME_REFUTED: "contradicts"}.get(outcome, "context")
    evidence = research.record_evidence(
        hypothesis_id,
        url=_experiment_ref(digest),
        title=f"experiment ({outcome}): {note or 'code run'}",
        # FIT IT HERE, WHERE THE SHAPE OF THE TEXT IS KNOWN. The evidence
        # excerpt is what the model reads back — the 4,000-char row above is
        # for forensics — and `record_evidence` cuts head-first, which is
        # right for a web page and exactly wrong for a test runner: pytest
        # writes `N passed, M failed` as its LAST line. Measured on a real
        # suite run: 19,776 characters of stdout, of which the excerpt kept
        # the first 600 — a mid-list slice of node ids, no counts, no error
        # roster. The assistant reported the count as unreachable and rebuilt
        # its instrument around the gap.
        excerpt=_fit_observation(f"{why}. {observation}",
                                 research.EXCERPT_CHARS),
        stance=stance, turn_idx=turn_idx)
    return {"outcome": outcome, "why": why, "result": result,
            "evidence": evidence, "repeated": repeated,
            "shadowed": shadowed}


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


def apply_replacements(before, replacements):
    """Apply anchored old→new substitutions. Returns (text, error).

    THE SAFE WAY TO CHANGE A FILE YOU HAVE ONLY READ IN PIECES. Whole-file
    replacement asks the assistant to reproduce every line it is NOT changing,
    from memory, and the failure mode is a confident-looking rewrite that
    silently drops one — invisible in the diff summary, invisible in the
    tests that do not cover that line, and indistinguishable from an intended
    deletion afterwards.

    EXACTLY ONE MATCH, or nothing is written. Zero means the anchor is stale
    or misremembered and the edit would land somewhere unintended; more than
    one means the assistant is describing a pattern while believing it is
    naming a place. Both are refused with the count, because "it did not
    apply" and "it applied three times" need different corrections.

    Applied in sequence against the accumulating text, so a later anchor may
    legitimately match something an earlier replacement produced."""
    text = before
    for n, item in enumerate(replacements or [], 1):
        if not isinstance(item, dict):
            return before, f"replacement {n} is not an object"
        old = str(item.get("old") or "")
        if not old:
            return before, f"replacement {n} has no `old` text to anchor on"
        found = text.count(old)
        if found != 1:
            excerpt = " ".join(old.split())[:80]
            return before, (
                f"replacement {n} matched {found} times, not once "
                f"({excerpt!r}) — "
                + ("widen it until it is unique"
                   if found > 1 else
                   "the file does not contain that text; read it again"))
        text = text.replace(old, str(item.get("new") or ""), 1)
    return text, ""


def apply_edit(path, contents=None, *, turn_idx, replace=None,
               hypothesis_id=None, why="", session_id=None):
    """Write a change back to the workspace and return the diff for review.

    THE MISSING VERB. Everything else in this module could reproduce a defect,
    design a fix and prove it correct in the sandbox — and then had nowhere to
    put it, because `sandbox.run` writes into a directory that is deleted the
    moment the run ends. A coding suite whose only durable artefact is an
    opinion about code is a research suite.

    REPRODUCE BEFORE YOU FIX, EXTENDED TO THE THING THAT ACTUALLY CHANGES.
    When a `hypothesis_id` is given, the same gate `propose_fix` applies holds
    here: nothing observed failing, no edit. It is optional because not every
    edit is a fix — writing a new test, or a file the user asked for outright,
    is not repairing a defect and inventing a defect to justify it would make
    the gate a ritual. What the gate cannot be is silently skipped, so the
    return value always says which of the two happened.

    Returns {ok, path, diff, created, rechunked, gated_on}. Never raises for
    an ordinary refusal: a rejected edit is a finding the assistant has to
    read and act on, and rule 3 applies to its own tools too."""
    import chunks
    import workspace

    if hypothesis_id is not None and not observed_failing(hypothesis_id):
        return {"ok": False, "path": str(path),
                "why": "nothing has been observed failing for that "
                       "hypothesis — reproduce the defect before editing the "
                       "code that supposedly causes it",
                "gated_on": hypothesis_id}
    if replace:
        # Anchored mode reads the file itself, so the assistant never has to
        # reproduce the parts it is not changing.
        current = workspace.read_file(path, session_id)
        if not current.get("ok"):
            return {"ok": False, "path": str(path),
                    "why": current.get("error")
                           or "cannot anchor an edit in a file I cannot read"}
        contents, problem = apply_replacements(current["text"], replace)
        if problem:
            return {"ok": False, "path": str(path), "why": problem}
    elif contents is None:
        return {"ok": False, "path": str(path),
                "why": "an edit needs either `contents` (the whole new file) "
                       "or `replace` (anchored old/new pairs)"}
    result = workspace.write_file(path, contents, session_id)
    if not result.get("ok"):
        return {"ok": False, "path": str(path),
                "why": result.get("error") or "the write was refused"}
    # A WRITE THAT CHANGED NOTHING IS NOT AN EDIT, AND CALLING IT ONE COSTS
    # FAR MORE THAN THE WASTED CALL. `unchanged` was returned here, carried
    # through the pipeline into the trace, and never read by anything — so a
    # no-op write reported "edited {path}" with an empty diff AND minted the
    # `witnessed` memory below saying so. The assistant then RECALLED making
    # a change that was never on disk and defended it across two turns
    # against the file itself, because its own memory was the evidence.
    # This is the last stage where a no-op and a real edit can still be told
    # apart; after it they are the same row.
    if result.get("unchanged"):
        return {"ok": False, "path": result["path"],
                "why": "the file already reads exactly like that — nothing "
                       "changed, so there is no edit to report",
                "gated_on": hypothesis_id}
    # The map is corrected in the same call that invalidated it.
    rechunked = chunks.reingest_path(result["path"], session_id or 0)
    # An edit is something the assistant DID, witnessed, not concluded. The
    # diff head goes in the memory so that "what did I change and why" is
    # answerable from recall alone, without reading the file back.
    head = "\n".join(result["diff"].splitlines()[:24])
    memory.add_memory(
        "episodic", "witnessed", 0.75,
        f"I edited {result['path']} in the workspace"
        + (f" because {why}" if why else "")
        + (" (new file)" if result["created"] else "")
        + f". Diff:\n{head}",
        gist=f"edited {result['path']}" + (f": {why}" if why else ""),
        turn_idx=turn_idx, session_id=session_id,
        event_key=f"edit:{turn_idx}:{result['path']}")
    return {"ok": True, "path": result["path"], "diff": result["diff"],
            "created": result["created"], "unchanged": result["unchanged"],
            "rechunked": rechunked,
            "gated_on": hypothesis_id,
            "why": "no reproduction required for this edit"
                   if hypothesis_id is None else
                   "a failing observation exists for the hypothesis"}


def experiments_for(hypothesis_id, limit=20):
    return [dict(r) for r in q(
        "SELECT id,digest,outcome,observation,note,turn_idx,created "
        "FROM experiments WHERE hypothesis_id=? ORDER BY id DESC LIMIT ?",
        (hypothesis_id, limit))]
