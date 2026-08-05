# subagents.py — delegating work to a second mind, and the permission that
# has to exist before it can happen.
#
# TWO TYPES, AND THE DIFFERENCE IS PRIVILEGE, NOT SIZE.
#
#   deep   A smaller version of this assistant with the full cognitive suite:
#          its own memory bank, its own beliefs, its own research loop, the
#          coding suite and the sandbox. It runs in a SUBPROCESS against its
#          OWN temporary database, emits one structured report, and is then
#          torn down — its working directory ARCHIVED for human forensics
#          (see the archiving block) and removed from the live tree. Nothing
#          of it remains reachable by the parent except the report. The
#          subprocess is not
#          ceremony: `db` resolves its path from a module global, so a child
#          sharing this process would have to repoint the parent's own
#          connection mid-turn. Isolation by process makes "erases itself" a
#          fact about the filesystem and makes the parent's bank structurally
#          unreachable — the child cannot corrupt what it was never handed.
#
#   scout  A single model call on a prompt it is given, with READ-ONLY
#          privileges: web search and fetch, the parent's memory as text, the
#          session workspace as text. It has no database handle, no sandbox,
#          and no write path of any kind — enforced by never passing it one,
#          not by instructing it to behave. A prohibition in a prompt is a
#          suggestion; an absent capability is a guarantee.
#
# THE PERMISSION MODEL IS HOST-HELD, AND THAT IS THE WHOLE POINT.
#
# The assistant may READ its allowance (it appears in every turn payload, so
# "how many of which type am I allowed" is answerable without asking) and may
# REQUEST more (a typed side channel the user approves). It cannot grant
# itself anything: `spawn` refuses when the grant is absent, the decrement
# happens in this file, and no prompt text is load-bearing anywhere in that
# sequence. This is the research loop's `rounds_left` discipline applied to a
# far more expensive resource — autonomy is safe to grant exactly when it
# provably halts, and a budget the model cannot edit is what makes it
# provable.
#
# WHAT A REPORT IS, EPISTEMICALLY.
#
# Testimony. A subagent is a second mind, and its report is something the
# parent was TOLD — not something it experienced. Absorbing it as first-hand
# would be precisely the information-layer collapse this project exists to
# prevent, happening at the one seam where a second mind actually exists.
# README records that the engine's information firewall was cut because "an
# assistant has no second mind to be kept out of"; subagents make that
# sentence false, and this module is where the firewall's internal form —
# provenance labelling — earns its keep instead.

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import codemap
import config
import memory
import prompts
import chunks
import research
import tools_web
import turnrun
import workspace
from db import state_get, state_put, transaction
from providers import chat_complete, chat_configured, parse_model_json

DEEP = "deep"
SCOUT = "scout"
KINDS = (DEEP, SCOUT)

_STATE_KEY = "assistant"
_GRANTS = "subagent_grants"
_REQUESTS = "subagent_requests"

# A deep child is bounded in its own right: it is a full assistant, and an
# unbounded one would be a way to spend the parent's budget without the
# parent's knowledge. Same reasoning as RESEARCH_MAX_ROUNDS.
DEEP_MAX_TURNS = 6
# TWO BOUNDS, NOT ONE — the split `providers.py` already made for the CLI and
# nobody carried up to here. Its comment is exactly this defect one layer
# down: "180 seconds killed real answers mid-sentence and reported it as a
# provider fault."
#
# A single wall clock cannot tell a wedged child from a productive one, so it
# killed both at the same moment. Observed: fourteen minutes of work, seven
# hypotheses, four completed experiments, destroyed for being slow rather than
# for being stuck. A deep read of a large codebase is legitimately long.
#
# `DEEP_IDLE_TIMEOUT` is the real guard: the child now reports every completed
# turn, so silence is measurable and silence is the only thing that means
# stuck. `DEEP_TIMEOUT` survives as a livelock ceiling — a child that emits
# forever without concluding — and is set where cost, not patience, argues
# for it.
DEEP_TIMEOUT = 3600.0
DEEP_IDLE_TIMEOUT = 900.0
# Held back from the working budget so the report can be written. The report
# is a separate model call and the ONLY artefact that outlives the child, so
# spending the last second of the budget on one more experiment is spending
# the whole run: everything learned goes down with the process.
REPORT_RESERVE = 240.0
SCOUT_MAX_ROUNDS = 4

# A ceiling on what the USER can grant in one go, so an accidental extra zero
# in the settings box is not a runaway.
MAX_GRANT = {DEEP: 8, SCOUT: 40}


# ---- The grant ledger ----

def _blank():
    return {kind: {"granted": 0, "used": 0} for kind in KINDS}


def _read(state=None):
    state = state if state is not None else state_get(_STATE_KEY)
    grants = state.get(_GRANTS)
    if not isinstance(grants, dict):
        return _blank()
    out = _blank()
    for kind in KINDS:
        entry = grants.get(kind)
        if isinstance(entry, dict):
            try:
                out[kind]["granted"] = max(0, int(entry.get("granted") or 0))
                out[kind]["used"] = max(0, int(entry.get("used") or 0))
            except (TypeError, ValueError):
                pass
    return out


def allowance(state=None):
    """What the assistant is allowed to spawn RIGHT NOW, per type.

    Goes into every turn payload. The model is required to consult this
    rather than guess, which is only reasonable if consulting it is free —
    so it is delivered, not fetched."""
    grants = _read(state)
    return {
        kind: {
            "remaining": max(0, grants[kind]["granted"] - grants[kind]["used"]),
            "granted": grants[kind]["granted"],
            "used": grants[kind]["used"],
        } for kind in KINDS
    }


def grant(kind, count):
    """The user's approval. The ONLY function that increases an allowance,
    and nothing the model emits reaches it."""
    if kind not in KINDS:
        return {"ok": False, "error": f"unknown subagent type {kind!r}"}
    try:
        count = int(count)
    except (TypeError, ValueError):
        return {"ok": False, "error": "count must be a whole number"}
    if count < 0 or count > MAX_GRANT[kind]:
        return {"ok": False,
                "error": f"grant between 0 and {MAX_GRANT[kind]} {kind} "
                         "subagents"}
    with transaction():
        state = state_get(_STATE_KEY)
        grants = _read(state)
        grants[kind]["granted"] = grants[kind]["used"] + count
        state[_GRANTS] = grants
        # An approval answers whatever was pending for that type.
        state[_REQUESTS] = [r for r in (state.get(_REQUESTS) or [])
                            if isinstance(r, dict) and r.get("kind") != kind]
        state_put(_STATE_KEY, state)
    return {"ok": True, "allowance": allowance()}


def revoke_all():
    with transaction():
        state = state_get(_STATE_KEY)
        state[_GRANTS] = _blank()
        state[_REQUESTS] = []
        state_put(_STATE_KEY, state)
    return {"ok": True, "allowance": allowance()}


def record_request(kind, count, why, turn_idx):
    """The assistant asking. Recorded for the user to answer; grants nothing.

    Kept deliberately separate from `grant` so that the code path the model
    can reach and the code path that increases a budget have no overlap at
    all — not a shared function with a flag, two functions."""
    if kind not in KINDS or not str(why or "").strip():
        return None
    try:
        count = max(1, min(int(count or 1), MAX_GRANT[kind]))
    except (TypeError, ValueError):
        count = 1
    state = state_get(_STATE_KEY)
    requests = [r for r in (state.get(_REQUESTS) or []) if isinstance(r, dict)]
    entry = {"kind": kind, "count": count, "why": str(why)[:400],
             "turn_idx": turn_idx, "asked": time.time()}
    requests = [r for r in requests if r.get("kind") != kind] + [entry]
    state[_REQUESTS] = requests[-6:]
    state_put(_STATE_KEY, state)
    return entry


def pending_requests():
    return [r for r in (state_get(_STATE_KEY).get(_REQUESTS) or [])
            if isinstance(r, dict)]


def _consume(kind):
    """Spend one. Returns False when there is nothing to spend — the refusal
    the whole permission model rests on."""
    with transaction():
        state = state_get(_STATE_KEY)
        grants = _read(state)
        if grants[kind]["granted"] - grants[kind]["used"] <= 0:
            return False
        grants[kind]["used"] += 1
        state[_GRANTS] = grants
        state_put(_STATE_KEY, state)
    return True


# ---- The structured report, and the rule that it is provisional ----

_REPORT_FIELDS = ("summary", "claims", "evidence", "open_questions",
                  "could_not_establish")


def validate_report(raw, kind, assignment=None):
    """Turn whatever the child returned into a report this parent will accept.

    Same discipline as every other model boundary here: claims are kept only
    when their support names evidence the child actually filed, and an
    ungrounded claim loses its citation, not the benefit of the doubt. A
    child is a model like any other — being a whole assistant does not buy it
    trust, and the seam that would have to be relaxed to let it through is
    the one that keeps the parent honest."""
    raw = raw if isinstance(raw, dict) else {}
    warnings = []
    evidence = []
    for item in raw.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        url = research.canonical_url(item.get("url"))
        excerpt = " ".join(str(item.get("excerpt") or "").split())[:600]
        if not url or not excerpt:
            warnings.append("dropped a subagent evidence row with no url or "
                            "no excerpt")
            continue
        stance = item.get("stance")
        evidence.append({
            "url": url, "title": str(item.get("title") or "")[:200],
            "excerpt": excerpt,
            "stance": stance if stance in ("supports", "contradicts",
                                           "context") else "context",
        })
    known = {row["url"] for row in evidence}
    claims = []
    for item in raw.get("claims") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim") or "").strip()
        if not text:
            continue
        support = [research.canonical_url(s)
                   for s in (item.get("support") or [])]
        support = [s for s in support if s in known]
        if not support:
            warnings.append(
                f"subagent claim kept as UNSUPPORTED: {text[:80]!r}")
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        claims.append({"claim": text[:400], "support": support,
                       "confidence": confidence,
                       "grounded": bool(support)})
    notes = []
    for item in raw.get("coding_notes") or []:
        if not isinstance(item, dict):
            continue
        what = str(item.get("what_changed") or "").strip()
        why = str(item.get("why") or "").strip()
        if not what:
            continue
        if not why:
            warnings.append(
                f"a coding note for {item.get('path')!r} says what changed "
                "and not why — an unexplained change cannot be reviewed")
        notes.append({"path": str(item.get("path") or "")[:200],
                      "what_changed": what[:600], "why": why[:600],
                      "evidence": str(item.get("evidence") or "")[:200],
                      "risk": str(item.get("risk") or "")[:300]})
    owned = set((assignment or {}).get("owns") or [])
    changes, maps = [], []
    for field, sink in (("file_changes", changes), ("map_updates", maps)):
        for item in raw.get(field) or []:
            if not isinstance(item, dict):
                continue
            path = normalize_path(item.get("path"))
            content = item.get("content")
            if not path or not isinstance(content, str):
                warnings.append(
                    f"refused a {field} entry: "
                    + (f"{item.get('path')!r} is not a path inside the "
                       "workspace" if not path else
                       "no complete file content was sent"))
                continue
            # COORDINATION IS ENFORCED HERE, not in the prompt. A child that
            # was assigned tokenizer.py and returns a change to app.py is
            # refused — which is what makes "their jobs do not overlap" a
            # property of the system rather than a hope about the model.
            if owned and path not in owned and field == "file_changes":
                warnings.append(
                    f"refused a change to {path!r}: this subagent was "
                    f"assigned {', '.join(sorted(owned))} and another agent "
                    "may own that file")
                continue
            sink.append({"path": path, "content": content[:400_000],
                         "why": str(item.get("why") or "")[:300]})
    described = {n["path"] for n in notes if n["path"]}
    for change in changes:
        if change["path"] not in described:
            warnings.append(
                f"{change['path']} was changed with no coding note saying "
                "why — the change is kept and the reason is not recoverable")
    return {
        "kind": kind,
        "summary": str(raw.get("summary") or "").strip()[:4000],
        "claims": claims,
        "evidence": evidence,
        "coding_notes": notes,
        "file_changes": changes,
        "map_updates": maps,
        "open_questions": [str(q)[:300]
                           for q in (raw.get("open_questions") or [])][:10],
        "could_not_establish": [
            str(q)[:300]
            for q in (raw.get("could_not_establish") or [])][:10],
        "warnings": warnings,
    }


def absorb(report, *, task, turn_idx, session_id=None, hypothesis_id=None):
    """Write a validated report into the parent's memory AS TESTIMONY.

    provenance `told`, attributed to the child by name, one row for the
    report and one per grounded claim. The child's evidence is replayed into
    the parent's evidence table when the work was attached to a hypothesis,
    so it moves confidence through the ordinary bounded arithmetic rather
    than through a second, privileged path — a subagent that fetched three
    supporting pages should move a hypothesis exactly as far as the parent
    fetching those three pages would have."""
    label = f"subagent {report['kind']}"
    minted = []
    summary = report["summary"] or "(the subagent returned no summary)"
    minted.append(dict(
        kind="dialogue", provenance="told",
        salience=0.8,
        content=f"I asked a {label} to: {task}. It reported: {summary}",
        entities=[label], turn_idx=turn_idx, session_id=session_id,
        confidence=0.7,
        event_key=f"turn:{turn_idx}:subagent:{report['kind']}:report"))
    for n, claim in enumerate(report["claims"]):
        if not claim["grounded"]:
            # An unsupported claim is recorded as the conjecture it is, not
            # as a finding. It still reaches memory, because "the subagent
            # thought this and could not show why" is worth remembering.
            continue
        minted.append(dict(
            kind="semantic", provenance="told",
            salience=0.75,
            content=f"A {label} reported, citing {', '.join(claim['support'][:3])}: "
                    f"{claim['claim']}",
            entities=[label], turn_idx=turn_idx, session_id=session_id,
            source_url=claim["support"][0], confidence=claim["confidence"],
            event_key=f"turn:{turn_idx}:subagent:{report['kind']}:claim:{n}"))
    for n, note in enumerate(report.get("coding_notes") or []):
        # WHY a file changed is the part that decays first and matters
        # longest. Six weeks on, the diff is still in the tree and the
        # reasoning is gone unless something kept it — so it is minted as an
        # ordinary memory row, retrievable by the same machinery as anything
        # else, rather than living only in a report the parent read once.
        minted.append(dict(
            kind="semantic", provenance="told", salience=0.85,
            content=(f"A {label} changed {note['path']}: "
                     f"{note['what_changed']}. Why: {note['why'] or 'not stated'}."
                     + (f" Justified by: {note['evidence']}."
                        if note["evidence"] else "")
                     + (f" Risk: {note['risk']}." if note["risk"] else "")),
            entities=[label, note["path"]], turn_idx=turn_idx,
            session_id=session_id, confidence=0.75,
            event_key=f"turn:{turn_idx}:subagent:{report['kind']}:note:{n}"))
    memory.add_memories_batch(minted)
    if hypothesis_id:
        for row in report["evidence"]:
            research.record_evidence(
                hypothesis_id, url=row["url"], title=row["title"],
                excerpt=row["excerpt"], stance=row["stance"],
                turn_idx=turn_idx)
    return len(minted)


# ---- Spawning ----

def spawn_cohort(specs, *, turn_idx, session_id=None, context=""):
    """Run a batch of subagents TOGETHER, brokering between them.

    Serial execution was the earlier shape and it makes sibling collaboration
    meaningless: by the time B starts, A has already reported and been
    erased, so anything A learned can only reach B through the parent's
    memory of a finished job. Running them concurrently is what makes "they
    can interact if they wind up in the same space" a live property rather
    than a description of the past.

    Threads, not processes, on this side: each child is ALREADY its own
    process, so the parent's threads spend their lives blocked on a pipe and
    the GIL is irrelevant. Returns (reports, warnings)."""
    import threading
    specs = [s for s in specs or [] if isinstance(s, dict)]
    if not specs:
        return [], []
    assignments, warnings = plan_assignments(specs)
    cohort = Cohort(assignments, session_id=session_id, turn_idx=turn_idx)
    results = [None] * len(specs)

    def work(n):
        results[n] = spawn(specs[n].get("kind"), specs[n].get("task"),
                           turn_idx=turn_idx, session_id=session_id,
                           context=context, assignment=assignments[n],
                           cohort=cohort)

    # `turnrun.inherit` HERE, on the parent thread, is what makes the cohort
    # visible at all: `current()` is thread-local, so without it every child
    # thread runs unobserved and each subagent's own emits — the ones written
    # to close the gap in the reasoning panel — go nowhere.
    threads = [threading.Thread(target=turnrun.inherit(work), args=(n,),
                                daemon=True)
               for n in range(len(specs))]
    for thread in threads:
        thread.start()
    # A generous ceiling: each child already enforces its own deadline, and
    # this only stops a wedged thread from holding the turn open forever.
    for thread in threads:
        thread.join(timeout=DEEP_TIMEOUT + 120)
    reports = []
    for n, outcome in enumerate(results):
        if outcome is None:
            warnings.append(f"subagent {n + 1} did not finish in time")
            continue
        if not outcome.get("ok"):
            warnings.append(outcome.get("error") or "a subagent failed")
            continue
        reports.append(outcome["report"])
    if cohort.log:
        warnings.extend(
            f"subagent cross-talk: {entry['from']!r} -> "
            f"{', '.join(entry['delivered_to'])}"
            for entry in cohort.log if entry["delivered_to"])
    return reports, warnings


# ---- Archiving a finished subagent ----
#
# ARCHIVE THE EVIDENCE, NOT THE MIND. A finished subagent leaves two very
# different things behind, and conflating them is the whole risk:
#
#   its REPORT      — testimony, absorbed into the parent's memory, subject to
#                     the same citation discipline as any other model output.
#   its WORKING     — the scratch database, the transcript, the experiments it
#                     ran and the pages it read. Forensics.
#
# The second is archived and the second is NEVER retrievable by the parent.
# That distinction is not fussiness: `search_memories` reads one bank, and a
# subagent's bank contains its own episodic rows, its own beliefs, and its own
# half-formed inferences. Making those reachable would mean the parent
# recalling a dead agent's private working-out as though it were its own
# experience — the layer collapse this project exists to prevent, wearing a
# helpful-sounding feature.
#
# So an archive is a FILE ON DISK for a human to open when a report looks
# wrong, and nothing in the running system reads it. What the assistant keeps
# of a subagent is exactly what it was told.
#
# The child is still told its working does not survive, and that stays honest:
# nothing outside its report reaches its parent. It is told for a reason —
# "everything not in this report is lost" is a real forcing function on report
# quality, and the report is what the whole design rests on.

ARCHIVE_ROOT = os.environ.get(
    "ASSISTANT_SUBAGENT_ARCHIVE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "subagent-archive"))
# Bounded, because an unbounded audit trail is a disk-full outage wearing the
# word "thorough".
ARCHIVE_KEEP_RUNS = 40
ARCHIVE_MAX_BYTES = 512 << 20


def archive_run(home, *, kind, task, report, turn_idx, seconds):
    """Preserve a finished subagent's working directory for later inspection.

    Returns the archive id, or "" when archiving is off or fails — a failed
    archive must never fail the turn, for the same reason a failed
    consolidation does not: it is a reconstructible-ish convenience, and the
    report already landed."""
    import tarfile
    if not ARCHIVE_ROOT:
        return ""
    try:
        os.makedirs(ARCHIVE_ROOT, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        run_id = f"{stamp}-{kind}-turn{turn_idx}-{os.getpid()}"
        manifest = {
            "id": run_id, "kind": kind, "task": task, "turn_idx": turn_idx,
            "seconds": seconds, "archived": time.time(),
            "report": report,
            "note": "Forensics only. Nothing here is readable by the "
                    "assistant; what it kept of this subagent is its report, "
                    "absorbed as testimony.",
        }
        with open(os.path.join(home, "manifest.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=1)
        target = os.path.join(ARCHIVE_ROOT, run_id + ".tar.gz")
        with tarfile.open(target, "w:gz") as tar:
            tar.add(home, arcname=run_id)
        _prune_archive()
        return run_id
    except Exception:
        return ""


def _prune_archive():
    """Oldest-first, by count and by total size."""
    try:
        entries = []
        for name in os.listdir(ARCHIVE_ROOT):
            if not name.endswith(".tar.gz"):
                continue
            full = os.path.join(ARCHIVE_ROOT, name)
            entries.append((os.path.getmtime(full), os.path.getsize(full),
                            full))
        entries.sort()
        total = sum(size for _m, size, _f in entries)
        while entries and (len(entries) > ARCHIVE_KEEP_RUNS
                           or total > ARCHIVE_MAX_BYTES):
            _mtime, size, full = entries.pop(0)
            os.remove(full)
            total -= size
    except OSError:
        pass


def list_archives(limit=50):
    """What is on disk, newest first. Metadata only — opening one is a
    deliberate act, not something a listing does for you."""
    import tarfile
    out = []
    try:
        names = sorted(os.listdir(ARCHIVE_ROOT), reverse=True)
    except OSError:
        return out
    for name in names[:limit]:
        if not name.endswith(".tar.gz"):
            continue
        full = os.path.join(ARCHIVE_ROOT, name)
        entry = {"id": name[:-7], "bytes": 0, "task": "", "kind": "",
                 "archived": 0}
        try:
            entry["bytes"] = os.path.getsize(full)
            entry["archived"] = os.path.getmtime(full)
            with tarfile.open(full) as tar:
                member = next(
                    (m for m in tar.getmembers()
                     if m.name.endswith("manifest.json")), None)
                if member:
                    manifest = json.loads(
                        tar.extractfile(member).read().decode())
                    entry.update({k: manifest.get(k, entry.get(k))
                                  for k in ("task", "kind", "turn_idx",
                                            "seconds")})
                    entry["summary"] = (
                        (manifest.get("report") or {}).get("summary") or ""
                    )[:400]
        except Exception:
            pass
        out.append(entry)
    return out


def normalize_path(raw):
    """A workspace-relative path, or None if it is not one.

    `lstrip("./")` was the first version and it is the classic form of this
    bug: it strips a CHARACTER SET, not a prefix, so "../../ESCAPED.txt"
    became "ESCAPED.txt" — a traversal silently rewritten into a different,
    valid filename. That is worse than refusing, because the write then
    succeeds against a file nobody named and the report says it changed
    something else. Refuse instead: mangling a path is a decision the caller
    did not ask for."""
    text = str(raw or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or ":" in text.split("/")[0]:
        return None
    parts = []
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return None
        parts.append(part)
    return "/".join(parts) or None


def plan_assignments(specs):
    """Divide work between subagents so two of them do not do the same job.

    THE PARENT COORDINATES, AND THE ENGINE CHECKS THE COORDINATION IS REAL.
    A parent that spawns three agents at one codebase without saying who owns
    what gets three agents reading the same entry point, and — worse — three
    sets of edits to the same file, of which the last writer wins silently.
    The overlap is not a model-behaviour problem to be prompted away; it is a
    property of the assignment, and a property can be checked.

    So each spec may declare a `scope`: the paths it owns. This function
    refuses the overlap deterministically (first claimant keeps a path,
    later ones lose it), and hands every child a ROSTER of what its siblings
    are doing. The roster matters as much as the boundary: an agent that
    knows a sibling is already reproducing the bug does not reproduce it
    again, and an agent that knows nobody is checking the callers will say
    so in `could_not_establish` instead of assuming someone else did.

    Returns (assignments, warnings). An assignment with no scope owns
    nothing exclusively — legitimate for a research question, and reported
    rather than inferred, because "everyone owns everything" is exactly the
    unplanned case."""
    assignments, claimed, warnings = [], {}, []
    for n, spec in enumerate(specs or []):
        if not isinstance(spec, dict):
            continue
        scope = [p for p in (normalize_path(s) for s in spec.get("scope") or [])
                 if p]
        mine, lost = [], []
        for path in scope:
            owner = claimed.get(path)
            if owner is None:
                claimed[path] = n
                mine.append(path)
            elif owner != n:
                lost.append(path)
        if lost:
            warnings.append(
                f"subagent {n + 1} asked for {', '.join(lost[:4])}, already "
                f"assigned to subagent {claimed[lost[0]] + 1} — the later "
                "claim was dropped so two agents cannot edit one file")
        assignments.append({"index": n, "kind": spec.get("kind"),
                            "task": str(spec.get("task") or ""),
                            "owns": mine})
    for assignment in assignments:
        assignment["siblings"] = [
            {"task": other["task"][:200], "owns": other["owns"]}
            for other in assignments if other["index"] != assignment["index"]]
    return assignments, warnings


def coordination_brief(assignment):
    """What one child is told about the division of labour.

    Stated as ownership rather than prohibition. "You own tokenizer.py" is an
    occasion; "do not touch other files" is a bare prohibition, and this
    project has measured what those do — a clause that only forbids reads as
    a suggestion of the thing it forbids."""
    if not assignment:
        return ""
    lines = []
    if assignment.get("owns"):
        lines.append(
            "YOU OWN these paths for this run: "
            + ", ".join(assignment["owns"])
            + ". Changes you send for anything else are refused by the "
            "engine, so if the fix genuinely belongs elsewhere, say so in "
            "`could_not_establish` and name the file — that is a finding, "
            "not a failure.")
    for sibling in assignment.get("siblings") or []:
        owns = ", ".join(sibling["owns"]) or "no files exclusively"
        lines.append(f"A SIBLING AGENT is working on: {sibling['task']} "
                     f"(owns {owns}). Do not duplicate that work; if you need "
                     "its result, say so rather than redoing it.")
    return "\n".join(lines)


# ---- Sibling collaboration, brokered by the parent ----
#
# WHY EVERYTHING GOES THROUGH THE PARENT. A child is a subprocess holding
# exactly one pipe, to the process that started it. It has no way to reach a
# sibling and is not given one — so "route cross-agent information through the
# main agent" is not a policy that could be violated, it is the only topology
# that exists. The parent is the broker because the parent is the only thing
# both children can see.
#
# WHY DELIVERY IS A MAILBOX AND NOT A CALL. Blocking child-to-child RPC
# deadlocks the first time A waits on B while B waits on A, and two agents
# working the same area is precisely when that happens. Messages are queued
# and handed over at the start of the recipient's next turn instead: nobody
# ever blocks on a sibling, and the worst case is a message that arrives one
# turn later than it could have.
#
# WHY RELEVANCE IS CHECKED IN CODE. "Only talk to a sibling if it is job
# relevant" is exactly the kind of instruction a model follows loosely and
# expensively, and the check is cheap and objective: do their assigned paths
# touch, or does the message name something in the other's brief? An
# irrelevant message is dropped with a reason rather than delivered, because
# the cost of chatter here is a subagent turn.

_STOP = frozenset(
    "a an and are as at be by for from has have how in into is it its of on or "
    "that the this to was what when where which who will with you your do does "
    "did can could should would i we my our not no if then than there".split())


def _content_words(text):
    import re
    return {w for w in re.findall(r"[a-z0-9_./-]{3,}", str(text or "").lower())
            if w not in _STOP}


def job_relevant(sender, recipient, text):
    """Would this message plausibly help the recipient do ITS job?

    Three ways to qualify, cheapest first: their assigned paths intersect;
    the paths sit in the same directory (agents editing siblings in one
    package are working the same space whether or not they were told so); or
    the message's content words overlap the recipient's brief. Returns
    (relevant, why) so a refusal can say what it was."""
    mine = {p for p in (sender or {}).get("owns") or []}
    theirs = {p for p in (recipient or {}).get("owns") or []}
    shared = mine & theirs
    if shared:
        return True, f"you both hold {', '.join(sorted(shared))}"
    import os as _os
    my_dirs = {_os.path.dirname(p) for p in mine}
    their_dirs = {_os.path.dirname(p) for p in theirs}
    if my_dirs & their_dirs:
        return True, ("you are working in the same area: "
                      + ", ".join(sorted(d or "." for d in my_dirs
                                         & their_dirs)))
    words = _content_words(text)
    brief = _content_words(recipient.get("task", "")) | {
        w for p in theirs for w in _content_words(p)}
    overlap = words & brief
    if len(overlap) >= 2:
        return True, ("it touches their brief: "
                      + ", ".join(sorted(overlap)[:4]))
    return False, ("their task does not touch this — the message was dropped "
                   "rather than costing them a turn")


class Cohort:
    """One batch of subagents running together, with the parent between them.

    Holds each child's mailbox and does the routing. Deliberately small: the
    interesting decisions (who may talk to whom, what a report is worth) live
    in `job_relevant` and `validate_report`, and this is only the plumbing
    that lets them happen while the children are still alive."""

    def __init__(self, assignments, *, session_id, turn_idx):
        import threading
        self.assignments = {a["index"]: a for a in assignments}
        self.session_id = session_id
        self.turn_idx = turn_idx
        self.mail = {a["index"]: [] for a in assignments}
        self.lock = threading.Lock()
        self.log = []

    def take_mail(self, index):
        with self.lock:
            pending, self.mail[index] = self.mail[index], []
        return pending

    def route(self, sender_index, message):
        """Deliver a child's message to whichever siblings it actually
        concerns. Returns what the sender is told back."""
        sender = self.assignments.get(sender_index) or {}
        text = str(message.get("text") or message.get("question") or "")
        delivered, skipped = [], []
        with self.lock:
            for index, recipient in self.assignments.items():
                if index == sender_index:
                    continue
                relevant, why = job_relevant(sender, recipient, text)
                if not relevant:
                    skipped.append({"task": recipient["task"][:80],
                                    "why": why})
                    continue
                self.mail[index].append({
                    "from": sender.get("task", "")[:160],
                    "from_owns": sender.get("owns") or [],
                    "text": text[:1200],
                    "because": why,
                })
                delivered.append(recipient["task"][:80])
        self.log.append({"from": sender.get("task", "")[:80],
                         "delivered_to": delivered, "text": text[:160]})
        if delivered:
            return {"delivered_to": delivered,
                    "note": "Your siblings will see this at the start of "
                            "their next turn. They are not blocked waiting "
                            "for you and you are not waiting for them — keep "
                            "working."}
        return {"delivered_to": [],
                "note": "No sibling's job touches this, so it was not "
                        "delivered. If you need the information yourself, "
                        "ask your parent instead.",
                "considered": skipped}


def apply_changes(report, session_id):
    """Write a child's file changes and map updates into the parent's
    workspace.

    WITHOUT THIS THE CODING WORK IS LOST. A deep subagent edits files inside a
    temporary directory that is deleted the moment it reports, so a report
    saying "I fixed the tokenizer" describes a fix that no longer exists
    anywhere. The parent receives the sentence and not the change.

    Every path goes through the same resolution an upload does, so a child
    cannot write outside the session workspace whatever it returns — the
    child is a model, and a model's output is provisional here as everywhere
    else. Map updates are applied AFTER code so that a map describing the new
    state is not briefly true of the old one.

    Returns (written, refused)."""
    if session_id is None:
        return [], [c["path"] for c in report.get("file_changes") or []]
    root = workspace.session_root(session_id)
    written, refused = [], []
    for change in ((report.get("file_changes") or [])
                   + (report.get("map_updates") or [])):
        target = workspace._resolved_under(root, change["path"])
        if target is None:
            refused.append(change["path"])
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(change["content"])
        except OSError:
            refused.append(change["path"])
            continue
        written.append(change["path"])
    return written, refused


def spawn(kind, task, *, turn_idx, session_id=None, context="",
          assignment=None, cohort=None):
    """Run one subagent, if and only if the user has allowed one.

    Returns {"ok": bool, "report"|"error": ...}. The permission check and the
    decrement both happen here, in deterministic code, before any model is
    reached."""
    if kind not in KINDS:
        return {"ok": False, "error": f"unknown subagent type {kind!r}"}
    task = str(task or "").strip()
    if not task:
        return {"ok": False, "error": "a subagent needs a task"}
    if not chat_configured():
        return {"ok": False, "error": "no chat model is configured"}
    if not _consume(kind):
        return {"ok": False, "denied": True,
                "error": f"no {kind} subagents are allowed right now — ask "
                         "the user for permission first"}
    started = time.perf_counter()
    try:
        brief = coordination_brief(assignment)
        full_context = "\n\n".join(p for p in (context, brief) if p)
        raw = (_run_deep(task, session_id=session_id, context=full_context,
                         turn_idx=turn_idx, assignment=assignment,
                         cohort=cohort)
               if kind == DEEP
               else _run_scout(task, session_id=session_id,
                               context=full_context, turn_idx=turn_idx))
    except Exception as exc:
        return {"ok": False, "error": f"the {kind} subagent failed: "
                                      f"{str(exc)[:300]}"}
    report = validate_report(raw, kind, assignment=assignment)
    report["seconds"] = round(time.perf_counter() - started, 2)
    report["task"] = task
    return {"ok": True, "report": report}


# ---- scout: one call, read-only, no state ----

def _run_scout(task, *, session_id=None, context="", turn_idx=0):
    """A single investigator on a given prompt.

    Read-only is structural: this function has no database handle, no
    sandbox, and no mint path. The only capabilities it passes on are web
    search, web fetch, and text it was already given. Nothing it returns is
    written anywhere by this function — the caller decides, after
    validation."""
    gathered, seen, search_note = [], [], ""
    # A scout sent at a codebase gets the map AND a way to open it.
    #
    # It used to get `codemap_for` and nothing else, which is a list of names
    # it could never read: its three actions were search, fetch-a-URL and
    # report, and a local file is not a URL. Measured on three real scouts
    # sent to investigate this project: 7 claims between them, `evidence: 0`
    # and `grounded_claims: 0` on every one. They were not lazy — the support
    # a claim about local code needs was structurally unavailable, so every
    # claim they made was a guess about a filename.
    #
    # Read-only does not mean blind: the constraint is that a scout cannot
    # CHANGE anything, and reading is not changing.
    given_map = (chunks.digest(kind="code", query=task)
                 if session_id is not None
                 and workspace.has_files(session_id) else None)
    read_chunks = []
    # A scout runs IN PROCESS, so it has no child to register and no stdout
    # protocol to narrate it. Without these it was the quieter of the two
    # gaps: the turn simply paused for four model calls.
    run = turnrun.current()
    label = str(task or "")[:80]
    if run is not None:
        run.emit("subagent", state="started", kind="scout", task=label)
    for _round in range(SCOUT_MAX_ROUNDS):
        if run is not None:
            # A scout is several model calls long, so a halt that could only
            # land between subagents would wait out the whole thing.
            run.halted()
        payload = {
            "task": task,
            "context_from_the_parent": context[:4000],
            **({"code": given_map} if given_map else {}),
            "code_you_have_read": read_chunks,
            "pages_read_so_far": gathered,
            "search_results": seen,
            # WHY it is empty, in the payload the scout actually reads. A
            # scout told only "no results" rephrases and searches again —
            # the one move that cannot help a blocked backend — and then
            # reports to its parent that the web has nothing on the subject.
            **({"search_unavailable": search_note} if search_note else {}),
            "rounds_left": SCOUT_MAX_ROUNDS - _round,
        }
        action = parse_model_json(chat_complete(
            prompts.render(prompts.SCOUT_SYSTEM),
            json.dumps(payload, ensure_ascii=False))) or {}
        name = str(action.get("action") or "").strip()
        if run is not None:
            run.emit("subagent", state=name or "(no action)", kind="scout",
                     task=label, rounds_left=SCOUT_MAX_ROUNDS - _round,
                     detail=str(action.get("query") or action.get("url")
                                or "")[:120])
        if name == "read":
            ids = action.get("chunk_ids")
            got = chunks.expand(ids=ids if isinstance(ids, list) else [])
            for piece in got:
                if piece.get("unknown") or not piece.get("text"):
                    continue
                read_chunks.append({"id": piece["id"],
                                    "source": piece.get("source", ""),
                                    "title": piece.get("title", ""),
                                    "text": piece["text"][:4000]})
            gathered.extend({"url": p["id"], "title": p.get("source", ""),
                             "excerpt": p["text"][:600], "stance": "context"}
                            for p in read_chunks[-len(got or []):]
                            if p.get("text"))
        elif name == "search":
            # A scout that cannot tell a dead lane from a quiet one reports
            # "nothing found" about the web, and its parent banks that as
            # testimony. Same seam as the research loop and the turn payload.
            found = tools_web.search_detail(
                str(action.get("query") or task), max_results=5)
            seen = found["results"]
            search_note = (found["detail"][:200]
                           if not seen
                           and found["status"] in ("blocked", "error") else "")
        elif name == "fetch":
            page = tools_web.fetch(str(action.get("url") or ""))
            excerpt = (str(action.get("excerpt") or "").strip()
                       or (page.get("text") or "")[:400])
            if page.get("error") or not excerpt:
                continue
            gathered.append({"url": page["url"], "title": page.get("title",
                                                                   ""),
                             "excerpt": excerpt,
                             "stance": action.get("stance") or "context"})
        elif name == "report":
            out = dict(action.get("report") or {})
            out.setdefault("evidence", gathered)
            if run is not None:
                run.emit("subagent", state="reported", kind="scout",
                         task=label, evidence=len(out.get("evidence") or []),
                         summary=str(out.get("summary") or ""),
                         found=[{"claim": str(c.get("claim") or ""),
                                 "confidence": c.get("confidence")}
                                for c in (out.get("claims") or [])[:8]
                                if isinstance(c, dict)],
                         sources=[str(e.get("url") or e.get("title") or "")
                                  for e in (out.get("evidence") or [])[:8]
                                  if isinstance(e, dict)],
                         could_not_establish=[
                             str(x) for x in
                             (out.get("could_not_establish") or [])[:6]])
            return out
    # Budget spent without a report: return what was actually gathered rather
    # than nothing, and say plainly that it is incomplete.
    if run is not None:
        run.emit("subagent", state="ran out of rounds", kind="scout",
                 task=label, evidence=len(gathered))
    return {"summary": "The scout ran out of rounds before reporting.",
            "evidence": gathered,
            "could_not_establish": [task]}


# ---- deep: a whole assistant, in its own process, then erased ----

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "subagent_runner.py")


def answer_query(question, *, session_id, turn_idx, allow_scout=True):
    """The parent answering a child's question mid-investigation.

    INFORMATION ROUTING, and why it is worth the protocol. A deep subagent
    working on one file of a project routinely needs something only the
    parent knows — what the user actually asked for, what was decided three
    sessions ago, how this module is used elsewhere. Without a way to ask, it
    has exactly two options: assume, or report an answer hedged into
    uselessness. Both are worse than a round trip.

    The parent answers in the order that spends least: its own memory first
    (free, and the thing it is actually for), then — only if recall came back
    thin and the user has allowed one — a scout, whose findings are returned
    as the testimony they are. A scout dispatched here spends the SAME grant
    ledger as one the assistant spawns directly, because a subagent
    commissioning a subagent must not be a way around the budget.

    What comes back is labelled by source. The child is told whether it is
    holding recalled memory or fresh investigation, because a child that
    cannot tell them apart will cite them alike."""
    question = str(question or "").strip()
    if not question:
        return {"answer": "", "source": "none",
                "note": "the query was empty"}
    recalled = memory.search_memories(
        question, k=6, current_turn_idx=turn_idx, count_access=False)
    projected = [memory.project_memory(row, turn_idx) for row in recalled]
    # "Thin" is deliberately a low bar. The scout costs a grant, and the
    # parent's own memory is the cheaper answer whenever it is an answer at
    # all; escalating on anything less than genuine emptiness would turn
    # every query into a spawn.
    thin = len(projected) < 2
    out = {
        "question": question,
        "source": "the parent's memory",
        "recalled": projected,
        "note": ("This is what your parent already knew — remembered, not "
                 "freshly checked. Its epistemic_origin fields say how each "
                 "row was known."),
    }
    if thin and allow_scout:
        scouted = spawn(SCOUT, question, turn_idx=turn_idx,
                        session_id=session_id,
                        context="A deep subagent asked this mid-task.")
        if scouted.get("ok"):
            out["source"] = "the parent's memory, plus a scout it dispatched"
            out["scout_report"] = scouted["report"]
            out["note"] = ("Your parent had little in memory, so it sent a "
                           "read-only scout. The scout's claims are "
                           "TESTIMONY — cite its urls, not the scout.")
        else:
            out["scout_unavailable"] = scouted.get("error")
            out["note"] += (" Your parent had little on this and could not "
                            "dispatch a scout: "
                            + str(scouted.get("error") or ""))
    return out


def _run_deep(task, *, session_id=None, context="", turn_idx=0,
              assignment=None, cohort=None):
    """A full cognitive suite on its own database, erased afterwards.

    The temporary directory is the child's entire world: its memory bank, its
    beliefs, its hypotheses and its consolidation windows all live in one
    SQLite file inside it, and `finally: rmtree` is what "spits out a report
    and erases itself" actually means. Nothing survives except the report the
    parent validates."""
    home = tempfile.mkdtemp(prefix="assistant-subagent-")
    seeded = os.path.join(home, "given")
    os.makedirs(seeded, exist_ok=True)
    snapshot = (workspace.snapshot_for_sandbox(session_id)
                if session_id is not None else {})
    if snapshot:
        _seed(seeded, snapshot)
        # SEEDED ON DISK, NOT THROUGH THE PAYLOAD.
        #
        # `files` used to carry every body in this dict into `task_payload`,
        # which is the child's model context. Measured on a 119-file
        # workspace: 78 files, 859,445 characters, ~215k tokens — the entire
        # source tree, in a single prompt, before the child had done
        # anything. That is the crash.
        #
        # The dict was load-bearing, which is why deleting it is not the fix:
        # the runner used it to populate the child's own workspace, since the
        # child runs with its own ASSISTANT_WORKSPACE in this temp directory.
        # But the parent knows that path — it sets it below — so it can write
        # the files there directly. The child ends up with exactly the same
        # workspace and none of it in its context.
        #
        # What the child gets in the payload instead is the MAP: the same
        # gist-and-id contract the parent works under. A subagent that reads
        # its whole corpus to answer one question is not cheaper than the
        # parent doing it.
        # `workspace.root_under` rather than a second spelling of the join.
        # This line and the ASSISTANT_WORKSPACE set below were computing the
        # same directory two different ways and disagreeing by one level, so
        # every deep subagent started with an empty workspace and correctly
        # reported that it could not open the code it was sent at.
        _seed(workspace.root_under(_child_workspace(home)), snapshot)
    task_payload = {
        "task": task,
        "context": context[:8000],
        "max_turns": DEEP_MAX_TURNS,
        # A DEADLINE IT CANNOT SEE IS ONE IT CANNOT PLAN AGAINST, and this one
        # was invisible: the child worked until the parent killed it, and the
        # report — a SEPARATE final call, the only thing that survives it —
        # never ran. A run that had done its work and formed its conclusions
        # returned nothing at all. Observed: fourteen minutes, seven
        # hypotheses, four completed experiments, discarded whole.
        #
        # The same rule the deliberation loop already follows for rounds, in
        # seconds: say how much is left so the worker can stop in time to
        # deliver. Sent as a duration rather than a wall-clock instant because
        # the child is a separate process and clocks are the parent's to keep.
        "budget_seconds": DEEP_TIMEOUT,
        "report_reserve_seconds": REPORT_RESERVE,
        # The map, and the project's own instructions if it carries any.
        # Handed over at the start so the child never has to spend a turn
        # discovering the shape of what it was given.
        "codemap": codemap.for_prompt(seeded),
        # HOW to reach a model, but nothing about what the parent knows. The
        # child gets a fresh database by design, and provider settings live
        # in that database — so without this the child fell back to the
        # environment defaults and reported "no language model configured"
        # while cheerfully consuming its grant.
        #
        # THE KEY VALUES ARE STRIPPED AND PASSED THROUGH THE ENVIRONMENT.
        # This used to be `config.get_config()` with a comment promising that
        # secrets were never serialised here — true when a settings row could
        # only hold the NAME of an env var, and false the moment
        # `config.KEY_VALUE_FIELDS` let a credential be stored outright. The
        # child's runner writes whatever it is handed straight into its own
        # database (`config.save_config`), and `archive_run` tars the child's
        # entire home directory afterwards — so a stored key would have come
        # to rest in a .tar.gz on disk, in a file whose whole purpose is to
        # be kept and read later. A comment asserting an invariant is not the
        # same as enforcing it; this enforces it.
        "provider_config": _provider_config_without_secrets(),
    }
    env = dict(os.environ)
    # The credential travels here instead: process environment, inherited by
    # the child, never written to its database and never archived.
    for field, variable in _SUBAGENT_KEY_VARS.items():
        secret = config.secret_for(field)
        if secret:
            env[variable] = secret
    env["ASSISTANT_DB"] = os.path.join(home, "subagent.db")
    env["ASSISTANT_WORKSPACE"] = _child_workspace(home)
    proc = subprocess.Popen(
        [sys.executable, _RUNNER], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    started = time.perf_counter()
    report = None
    try:
        report = _converse(proc, task_payload, session_id=session_id,
                           turn_idx=turn_idx, assignment=assignment,
                           cohort=cohort)
        return report
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        # ARCHIVED BEFORE ERASED, and only the DEEP type — see the archiving
        # block above. A scout has no working directory to keep: no database,
        # no sandbox, no memory bank, one model call whose entire product is
        # the report, whose evidence is absorbed. There is nothing of a scout
        # that outlives the answer it gave, which is why only this path
        # archives. Archiving runs in `finally` so a child that crashed —
        # the case you most want to inspect — is kept too.
        archived = archive_run(home, kind=DEEP, task=task, report=report,
                               turn_idx=turn_idx,
                               seconds=round(time.perf_counter() - started, 2))
        if archived and isinstance(report, dict):
            report["archive_id"] = archived
        shutil.rmtree(home, ignore_errors=True)


# Where a stripped key is handed to a child instead. Names chosen to be the
# ones `config._DEFAULTS` already falls back to, so a child that reads them is
# reading its ordinary configuration rather than a special case.
_SUBAGENT_KEY_VARS = {"chat_key_env": "ASSISTANT_CHAT_KEY",
                      "embed_key_env": "ASSISTANT_EMBED_KEY"}


def _provider_config_without_secrets():
    """The parent's provider settings with every stored credential removed
    and the key-name fields pointed at `_SUBAGENT_KEY_VARS`.

    Stripped by iterating `config.KEY_VALUE_FIELDS` rather than by naming
    fields here: a credential field added later is removed by default instead
    of leaking because this function was not updated."""
    cfg = dict(config.get_config())
    for field in config.KEY_VALUE_FIELDS:
        cfg[field] = ""
    for field, variable in _SUBAGENT_KEY_VARS.items():
        cfg[field] = variable
    return cfg


def _child_workspace(home):
    """The value of the child's ASSISTANT_WORKSPACE. One caller sets the
    variable, another seeds files under it, and they must agree — so neither
    spells the path itself."""
    return os.path.join(home, "workspace")


def _seed(root, files):
    for relative, text in (files or {}).items():
        target = os.path.normpath(os.path.join(root, relative))
        if not target.startswith(os.path.realpath(root) + os.sep) \
                and target != os.path.realpath(root):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            # A child gets the same workspace an experiment gets, binaries
            # included — otherwise the two disagree about what exists.
            if isinstance(text, (bytes, bytearray)):
                with open(target, "wb") as handle:
                    handle.write(text)
                continue
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            continue


# The child speaks one JSON object per line on stdout and reads one per line
# on stdin. Line-delimited rather than a single blob because the conversation
# is now bidirectional: the child asks, the parent answers, and only the last
# message is the report.
MAX_CHILD_QUERIES = 6


def _converse(proc, task_payload, *, session_id, turn_idx, assignment=None,
              cohort=None):
    """Run the child to completion, answering its questions as they arrive.

    A hard cap on queries, for the same reason every other loop here is
    bounded: a child that can ask forever is a child that never reports, and
    the parent is holding a user's turn open while it happens."""
    # THE PARENT ALREADY SEES EVERYTHING THE CHILD DOES — every question,
    # every sibling message, the report — and until now it told nobody. A
    # subagent was a gap in the reasoning panel: the turn went quiet for up to
    # DEEP_TIMEOUT with no way to tell work from a hang.
    #
    # `turnrun.current()` is thread-local and a subagent is spawned on the
    # turn's own worker thread, so the live run is reachable here without
    # threading a parameter through `spawn`, `_run_deep` and `_converse`.
    # None for a blocking turn or a test, and then this is inert.
    run = turnrun.current()
    label = str(task_payload.get("task") or "")[:80]
    if run is not None:
        run.emit("subagent", state="started", kind="deep", task=label)
        # Registering the child means HALT REACHES IT. Without this a halt
        # during a subagent set the flag and then waited out the child's
        # remaining timeout, which from the button's side is a halt that did
        # nothing for two minutes.
        run.register_process(proc)
    # The parent's own deadline sits PAST the child's, by the reserve it told
    # the child to hold back. Two bounds, not one: the child stops working in
    # time to report, and the parent waits long enough to receive it. Without
    # the second half the first buys nothing — the report would be composed
    # and then killed on the way out.
    deadline = time.monotonic() + DEEP_TIMEOUT + REPORT_RESERVE
    last_word = time.monotonic()
    proc.stdin.write(json.dumps(task_payload) + "\n")
    proc.stdin.flush()
    answered = 0
    while True:
        now = time.monotonic()
        if now - last_word > DEEP_IDLE_TIMEOUT:
            raise RuntimeError(
                f"the deep subagent said nothing for {DEEP_IDLE_TIMEOUT:.0f}s "
                "— it reports every completed turn, so this is a stuck child "
                "rather than a slow one")
        if now > deadline:
            raise RuntimeError(
                "the deep subagent was still going after "
                f"{DEEP_TIMEOUT + REPORT_RESERVE:.0f}s without concluding")
        line = proc.stdout.readline()
        if not line:
            stderr = (proc.stderr.read() or "").strip()[-400:]
            raise RuntimeError(stderr or "the deep subagent exited without "
                                         "reporting")
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (TypeError, ValueError):
            continue          # the child printed something that was not
            # protocol; ignore rather than die, since a stray print is not a
            # failure of the investigation
        # ANY word from the child is proof it is alive; the idle clock is
        # about silence, not about which message broke it.
        last_word = time.monotonic()
        kind = str(message.get("type") or "")
        if kind == "progress":
            # Straight to the panel. A subagent was a hole in the trail —
            # the turn went quiet for the whole of its run with no way to tell
            # work from a hang — and this is the signal that closes it.
            if run is not None:
                run.emit("subagent", state="working", kind="deep", task=label,
                         turn=message.get("turn"),
                         seconds=message.get("seconds"),
                         experiments=message.get("experiments"),
                         edits=message.get("edits"),
                         minted=message.get("minted"),
                         said=message.get("said"))
            continue
        if kind == "report":
            report = message.get("report") or {}
            if run is not None:
                run.unregister_process(proc)
                # THE COUNTS WERE NOT THE REPORT. This emitted "3 claims, 4
                # evidence" and 160 characters of summary, so the one thing a
                # reader wanted — what the subagent actually concluded, and
                # what it could not settle — reached the browser as arithmetic.
                # The report is already in hand here; sending its text costs
                # nothing and is the only place the UI can get it, because the
                # subagent's scratch database is torn down after this.
                run.emit("subagent", state="reported", kind="deep",
                         task=label,
                         claims=len(report.get("claims") or []),
                         evidence=len(report.get("evidence") or []),
                         summary=str(report.get("summary") or ""),
                         found=[{"claim": str(c.get("claim") or ""),
                                 "confidence": c.get("confidence")}
                                for c in (report.get("claims") or [])[:8]
                                if isinstance(c, dict)],
                         sources=[str(e.get("url") or e.get("title") or "")
                                  for e in (report.get("evidence") or [])[:8]
                                  if isinstance(e, dict)],
                         could_not_establish=[
                             str(x) for x in
                             (report.get("could_not_establish") or [])[:6]],
                         open_questions=[
                             str(x) for x in
                             (report.get("open_questions") or [])[:6]])
            return report
        if kind == "message" and cohort is not None:
            if run is not None:
                run.emit("subagent", state="messaged a sibling", kind="deep",
                         task=label,
                         detail=str(message.get("text") or "")[:120])
            # A child talking to its siblings. Routed, relevance-checked, and
            # non-blocking in both directions.
            reply = {"type": "routed",
                     **cohort.route((assignment or {}).get("index", 0),
                                    message)}
            proc.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            continue
        if kind == "mailcheck":
            pending = (cohort.take_mail((assignment or {}).get("index", 0))
                       if cohort is not None else [])
            proc.stdin.write(json.dumps({"type": "mail", "messages": pending},
                                        ensure_ascii=False) + "\n")
            proc.stdin.flush()
            continue
        if kind == "query":
            answered += 1
            if run is not None:
                run.emit("subagent", state="asked a question", kind="deep",
                         task=label,
                         detail=str(message.get("question") or "")[:140],
                         questions_left=max(0, MAX_CHILD_QUERIES - answered))
            if answered > MAX_CHILD_QUERIES:
                reply = {"type": "answer", "answer": "",
                         "note": "You have used all your questions. Finish "
                                 "with what you have and report."}
            else:
                reply = {"type": "answer",
                         "remaining_questions": MAX_CHILD_QUERIES - answered,
                         **answer_query(message.get("question"),
                                        session_id=session_id,
                                        turn_idx=turn_idx)}
            proc.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            proc.stdin.flush()
