# pipeline.py — one chat turn, stage by stage. The engine's discipline in
# miniature: model output is provisional until deterministic code validates
# it; slow provider work (embedding) happens before the write transaction;
# all primary turn mutations commit atomically; reconstructible caches
# (consolidation) run after, outside the transaction, so their failure can
# never roll back a valid turn.
#
# Stages (see DESIGN.md for the full write-up):
#   1 recall   — deterministic. Build the memory payload, belief payload,
#                hypothesis sheet; consume any pending ponder query.
#   2 respond  — one model call. Reply + typed side-channels (user-model
#                updates, remember marks, dispute, ponder, research request).
#   3 ground   — deterministic. Citations against delivered refs; ungrounded
#                ones dropped with a warning; belief updates without grounded
#                evidence dropped.
#   4 research — only when requested: the automated loop in research.py
#                (search/fetch/ponder rounds until grounded conclusion or
#                budget), producing the reply instead of stage 2's.
#   5 commit   — deterministic, one transaction. Mint memories (episode,
#                told-rows, inference-rows with the reconstructible salience
#                formula), apply belief updates, reconcile inference
#                confidence, lift importance for load-bearing citations
#                (once-ever), record disputes, reselect the hypothesis
#                sheet, store the pending ponder.
#   6 settle   — post-commit: consolidation when due.

import json

import beliefs
import coding
import memory
import persona
import prompts
import chunks
import research
import subagents
import tools_web
import turnrun
import workspace
from db import (ensure_session, next_turn_idx, qi, state_get, state_put,
                transaction)
from providers import chat_complete, chat_configured, parse_model_json

USER_SUBJECT = "the user"

# Assistant-tuned salience markers. Same deterministic floor as the engine's
# 15-word list (0.45 + len/1600 + 0.08 per hit, cap 0.95) with the vocabulary
# moved from knives-and-betrayal to the things that make an assistant
# exchange worth ranking later: decisions, deadlines, corrections, identity.
_SALIENCE_MARKERS = (
    "deadline", "decided", "decision", "prefer", "always", "never",
    "important", "remember", "budget", "name is", "birthday", "project",
    "launch", "wrong", "correct", "instead", "moved", "cancel", "promise",
)


def _salience_of(text):
    s = 0.45 + min(len(text or ""), 400) / 1600.0
    low = (text or "").lower()
    for w in _SALIENCE_MARKERS:
        if w in low:
            s += 0.08
    return round(min(s, 0.95), 3)


def _ground_evidence_list(refs, delivered, warnings, path):
    """Engine citation discipline: keep refs that were delivered (or the
    literal "current", which names the user's message this turn); drop the
    rest with a warning. Never invent one the model omitted."""
    grounded = []
    for ref in refs or []:
        name = str(ref or "").strip()
        if name == "current" or name in delivered:
            if name not in grounded:
                grounded.append(name)
        else:
            warnings.append(f"dropped ungrounded {path} citation {name!r}")
    return grounded


def _aspects(sheet_entries, threads, persona_sheet):
    """What the assistant BRINGS to the turn, as separate retrieval facets —
    each gets its own RRF ranking (see memory.search_memories for why
    concatenation was measured useless)."""
    return [
        ("what i am wondering about",
         " ".join(e.get("i_suspect", "") for e in sheet_entries[:2])),
        ("what is still unsettled", " ".join(threads or [])),
        ("my standing commitments",
         " ".join(persona_sheet.get("standing_commitments") or [])),
    ]


# How many times the respond stage may go back for more before it must answer.
#
# Bounded, and the bound is small. Each round is a full model call, so this is
# the difference between a turn that thinks and a turn that costs four times
# as much to say the same thing. The model is TOLD how many rounds remain, the
# way the research loop tells it: a deadline you cannot see is one you cannot
# plan against, and an agent that discovers it is out of rounds has already
# wasted the last one.
DELIBERATION_MAX_ROUNDS = 3
# What one ponder is allowed to pull back into the turn.
PONDER_RECALL_LIMIT = 8
PONDER_SEARCH_RESULTS = 5


def _deliberate(payload, persona_sheet, turn_idx, session_id, run, warnings):
    """The respond stage, with the option to go back for more before it
    answers.

    `ponder` used to be DEFERRED: the model named something it wanted from its
    own memory and the answer arrived on the following turn, by which point
    the question that prompted it had already been answered without it. For a
    hard question that is the wrong shape — the point of asking your own
    memory is to ask it BEFORE you commit to an answer.

    So a round may come back with `need_more`, and the engine gathers exactly
    three things, deterministically, and asks again:

      ponder         → its own memory
      expand_chunks  → the bodies behind gists it has only seen summarised
      search         → the web

    The ordering rule lives in the prompt, not here: ask memory before the
    web. What lives HERE is the honest reporting of an empty result — a ponder
    that returned nothing is stated as nothing, so the next round can search
    instead of quietly assuming memory had the answer. That is the difference
    between a loop that converges and one that repeats itself.

    Returns `(out, deliberation)`."""
    system = prompts.render(prompts.RESPOND_SYSTEM,
                            persona=persona.persona_prompt(persona_sheet))
    deliberation = []
    out = None
    for round_no in range(1, DELIBERATION_MAX_ROUNDS + 1):
        run.halted()
        run.emit("respond", state=("calling the model" if round_no == 1
                                   else f"deliberating (round {round_no})"),
                 round=round_no)
        body = dict(payload)
        if deliberation:
            body["what_i_went_and_got"] = deliberation
        body["deliberation_rounds_left"] = DELIBERATION_MAX_ROUNDS - round_no
        out = parse_model_json(chat_complete(system,
                                             json.dumps(body,
                                                        ensure_ascii=False)))
        if out is None:
            return None, deliberation
        more = out.get("need_more")
        if not isinstance(more, dict) or round_no == DELIBERATION_MAX_ROUNDS:
            break
        step = _gather(more, turn_idx, session_id, run, warnings)
        if not step:
            break                 # nothing to fetch: the answer it has stands
        deliberation.append(step)
    return out, deliberation


def _gather(more, turn_idx, session_id, run, warnings):
    """Fetch what a deliberation round asked for. Deterministic; no judgement
    about whether the request was wise — that is the model's to make and the
    engine's to bound."""
    step = {}
    query = str(more.get("ponder") or "").strip()
    if query:
        run.emit("ponder", query=query)
        rows = memory.search_memories(query, k=PONDER_RECALL_LIMIT,
                                      current_turn_idx=turn_idx)
        step["ponder"] = {
            "query": query,
            "returned": len(rows),
            "memories": [{"ref": r.get("ref"),
                          "gist": str(r.get("gist") or r.get("content")
                                      or "")[:200]} for r in rows],
            # Said explicitly, because "no memories" and "I forgot to look"
            # are indistinguishable from an empty list — and the whole point
            # of the fallback is that the model KNOWS memory came up dry.
            **({"nothing_found": True,
                "note": "your memory has nothing on this. If the question "
                        "needs external or current information, search the "
                        "web next round instead of guessing."}
               if not rows else {}),
        }
        run.emit("ponder", query=query, returned=len(rows))
    # Navigation, one directory at a time. `list_dir` is the step; the payload
    # deliberately carries only the top level, so going deeper is an act
    # rather than something that already happened to the context.
    where = more.get("list_dir")
    if isinstance(where, str) or isinstance(where, list):
        paths = [where] if isinstance(where, str) else [str(p) for p in where]
        listings = [workspace.list_dir(p) for p in paths[:6]]
        step["listings"] = listings
        run.emit("navigate", paths=paths[:6],
                 entries=sum(l.get("count", 0) for l in listings))
    ids = more.get("expand_chunks")
    if isinstance(ids, list) and ids:
        expanded = chunks.expand(session_id, ids)
        step["expanded_chunks"] = expanded
        run.emit("expand", ids=[str(i)[:24] for i in ids[:8]],
                 got=len([e for e in expanded if not e.get("unknown")]))
    search_query = str(more.get("search") or "").strip()
    if search_query:
        run.emit("search", query=search_query)
        try:
            results = tools_web.search(search_query,
                                       max_results=PONDER_SEARCH_RESULTS)
        except Exception as exc:
            warnings.append(f"web search failed: {str(exc)[:120]}")
            results = []
        step["search"] = {"query": search_query, "results": results,
                          **({"nothing_found": True} if not results else {})}
        run.emit("search", query=search_query, results=len(results))
    return step


class _Unobserved:
    """The default `run` — a turn nobody is watching and nobody can halt.

    A null object rather than `if run is not None` at every checkpoint: the
    checks are the point, and a guard that each new stage must remember to
    wrap is a guard the next stage will omit."""

    def emit(self, stage, **detail):
        pass

    def halted(self):
        return False

    def enter_commit(self):
        pass


_UNOBSERVED = _Unobserved()


def run_turn(user_text, session_id=None, run=None):
    """One full exchange. Returns {reply, warnings, trace, session_id,
    turn_idx}.

    `run` is an optional `turnrun.TurnRun`: it receives a step event per stage
    and is asked, between stages, whether the user has halted. Halting raises
    `turnrun.TurnHalted` out of here, so nothing after the halt point runs and
    — because every checkpoint is before stage 5 — nothing is committed."""
    run = run or _UNOBSERVED
    warnings = []
    trace = {}
    persona_sheet = persona.get_persona()
    warnings.extend(persona.persona_warnings(persona_sheet))
    session_id = ensure_session(session_id)
    # The ordinal is RESERVED here (atomically, durably) because stage 1's
    # retrieval cutoff needs it — but the turn ROW is not written until the
    # commit transaction. Writing it here made the turn's own record the one
    # durable mutation outside the "all turn mutations in one transaction"
    # invariant: an exception mid-commit rolled the memories back and left a
    # turns row with a user message, no reply, and a consumed ordinal.
    turn_idx = next_turn_idx()

    # -- Stage 1: recall (deterministic) --
    state = state_get("assistant")
    sheet_entries, _prev_keys = beliefs.select_active_hypotheses(
        state.get("mind_models"), state.get("active_hypothesis_keys"),
        turn_idx)
    threads = memory.get_memory_summary(
        before_turn_idx=turn_idx).get("unresolved_threads") or []
    pending_ponder = str(state.get("pending_ponder") or "")
    memory_payload, internal = memory.build_memory_context(
        turn_idx, user_text,
        aspects=_aspects(sheet_entries, threads, persona_sheet),
        ponder_query=pending_ponder)
    delivered = set(internal["delivered_refs"])
    # What surfaced, not just how much. The count alone cannot answer "why did
    # it say that" — the refs are what tie a claim back to a row.
    run.emit("recall",
             returned=len(delivered),
             pondered=(internal.get("retrieval_health") or {})
             .get("recall", {}).get("pondered", 0),
             ponder_query=pending_ponder,
             refs=sorted(delivered)[:24],
             gists=[str(m.get("gist") or m.get("content") or "")[:120]
                    for m in (memory_payload.get("recalled_old_memories")
                              or [])][:12])
    run.halted()
    # Degraded retrieval is announced, not inferred later from bad answers.
    # Three of the four ranking lanes are vector lanes; when they cannot run,
    # recall still returns rows and still looks healthy.
    health = internal.get("retrieval_health") or {}
    trace["retrieval_health"] = health
    if health.get("query_embedding_fallback"):
        warnings.append(
            "retrieval degraded: the query embedding fell back to the local "
            "hashing trick, so semantic, cue and aspect ranking scored zero "
            "this turn"
            + (f" ({health['query_embedding_error'][:120]})"
               if health.get("query_embedding_error") else ""))
    elif health.get("embedded_rows") and not health.get("vector_lanes_live"):
        warnings.append(
            f"retrieval degraded: all {health['embedded_rows']} embedded "
            "memories were written by a different embedding model and cannot "
            "be compared; recall is running on keyword match alone")
    elif health.get("vector_incomparable_rows"):
        warnings.append(
            f"{health['vector_incomparable_rows']} of "
            f"{health['embedded_rows']} memories are stranded on an older "
            "embedding model and reachable only by keyword")
    user_model = beliefs.beliefs_for_payload(
        state.get("mind_models"), turn_idx)

    # -- Stage 2: respond (the model call) --
    payload = {
        "user_message": {"text": str(user_text or ""),
                         "temporal_status": "present",
                         "ref": "current"},
        "memory": memory_payload,
        "what_i_believe_about_people_and_topics": user_model,
        "active_hypotheses": sheet_entries,
        "open_research": [
            {"question": h["question"], "status": h["status"],
             "confidence": h["confidence"]}
            for h in research.list_hypotheses(status="open", limit=5)],
        # Delivered, not fetched: "how many of which type am I allowed" has
        # to be answerable without spending anything to ask, or the model
        # will guess instead — and a guess about its own permissions is the
        # one guess this design cannot tolerate.
        "subagent_allowance": subagents.allowance(state),
        # Not just a file list: the STRUCTURE of what was uploaded, plus any
        # AGENTS.md / CLAUDE.md the project carries. An assistant that can
        # see where things are does not have to spend turns discovering it,
        # and one that can see the project's own house rules does not have to
        # guess at them.
        "files_the_user_gave_me": workspace.describe(session_id),
        # A DIGEST, not the whole map. `codemap_for` bounded by file count and
        # produced 97 KB for a 115-file upload — enough to break the turn
        # outright, and long before that, enough to spend the context on
        # describing the code instead of thinking about it. The digest is a
        # ranked list of gists with ids; bodies arrive only via
        # `need_more.expand_chunks`. Measured on the same upload: 13 KB.
        **({"code": chunks.digest(session_id, kind="code",
                                  query=str(user_text or ""))}
           if workspace.has_files(session_id) else {}),
    }
    out = None
    run.halted()
    if chat_configured():
        try:
            out, deliberation = _deliberate(payload, persona_sheet, turn_idx,
                                            session_id, run, warnings)
            if deliberation:
                trace["deliberation"] = deliberation
            if out is None:
                warnings.append("respond stage returned unparseable output")
        except turnrun.TurnHalted:
            raise
        except Exception as exc:
            warnings.append(f"respond stage failed: {str(exc)[:200]}")
    else:
        warnings.append("no chat model configured; memory still records "
                        "this exchange")
    # Whether a MODEL composed this reply, stated as a fact rather than left
    # for the client to infer from warning text. The retry control needs to
    # know, and a UI that decides by string-matching "failed" in a warning
    # would silently stop offering retry the day a message is reworded.
    respond_ok = out is not None
    run.emit("respond", state="answered" if respond_ok else "no usable answer",
             ok=respond_ok,
             ponder=((out or {}).get("ponder") or {}).get("query") or "",
             asks_research=bool(((out or {}).get("research") or {})
                                .get("question")))
    if out is None:
        # No fabricated reply: memory still records the exchange, and the
        # reply says plainly what happened. The deterministic floor does not
        # impersonate the model — and it does not misdiagnose itself either.
        # One message served both "no provider is configured" and "the
        # configured provider just failed", so a truncated response or a 429
        # told the operator to go configure a model they had already
        # configured.
        out = {"reply": (
            "(I have no language model configured, so I can't compose a real "
            "reply — but I've recorded what you said and will remember it.)"
            if not chat_configured() else
            "(My language model didn't come back with a usable answer this "
            "turn — see the warnings for why. I've recorded what you said "
            "and will remember it.)")}

    # -- Stage 3: ground (deterministic) --
    out["memory_evidence_used"] = _ground_evidence_list(
        out.get("memory_evidence_used"), delivered, warnings,
        "memory_evidence")
    model_updates = []
    for update in out.get("user_model_updates") or []:
        if not isinstance(update, dict) or not str(
                update.get("claim") or "").strip():
            continue
        evidence = _ground_evidence_list(update.get("evidence"), delivered,
                                         warnings, "user_model evidence")
        if not evidence:
            warnings.append(
                "dropped user_model update with no grounded evidence: "
                f"{str(update.get('claim'))[:80]!r}")
            continue
        update = dict(update)
        update["evidence"] = evidence
        model_updates.append(update)

    # -- Stage 4: research (only when requested) --
    research_result = None
    req = out.get("research")
    if isinstance(req, dict) and str(req.get("question") or "").strip():
        hyp = research.open_hypothesis(req["question"], turn_idx, session_id)

        run.emit("research", state="opened", question=req["question"])

        def ask_model(round_payload):
            # The halt checkpoint that matters most: research is the long
            # stage, so a turn the user wants to stop is almost always
            # stopped here. Between rounds, never mid-round — a half-recorded
            # round would leave evidence attached to a hypothesis nobody
            # concluded.
            run.halted()
            raw = chat_complete(
                prompts.render(
                    prompts.RESEARCH_SYSTEM,
                    persona=persona.persona_prompt(persona_sheet)),
                json.dumps(round_payload, ensure_ascii=False))
            act = parse_model_json(raw)
            # Emitted from here rather than from research.py: the loop already
            # builds its own trace, and this is the seam that knows a human is
            # watching. research.py stays free of the idea.
            run.emit("research",
                     state="round",
                     rounds_left=round_payload.get("rounds_left"),
                     action=str((act or {}).get("action") or "?"),
                     detail={k: str(v)[:200]
                             for k, v in (act or {}).items()
                             if k != "action"})
            return act

        try:
            research_result = research.research_loop(hyp["id"], ask_model,
                                                     turn_idx)
        except turnrun.TurnHalted:
            raise           # a halt is not a research failure; let it unwind
        except Exception as exc:
            warnings.append(f"research loop failed: {str(exc)[:200]}")
        if research_result:
            warnings.extend(research_result.get("warnings") or [])
            answer = str(research_result.get("answer") or "").strip()
            if answer:
                cites = research_result.get("citations") or []
                out["reply"] = answer + (
                    "\n\nSources: " + ", ".join(cites) if cites else "")
            trace["research"] = {
                "hypothesis_id": hyp["id"],
                "rounds": research_result.get("rounds"),
                "status": research_result.get("status"),
                "hedged": research_result.get("hedged"),
                "confidence": research_result.get("confidence"),
            }

    # -- Stage 4a: experiments (the coding suite, finally wired in) --
    #
    # `coding.py` and `sandbox.py` had no caller anywhere outside tests: a
    # complete implementation of coding-as-the-scientific-method that no turn
    # could ever reach. Built, tested, and unreachable is indistinguishable
    # from absent at runtime, which is the same scar as a mechanism that
    # never fires — one step earlier.
    #
    # The four rules are enforced by coding.py, not here: `expect` is
    # required and graded mechanically, a failure comes back as an
    # observation rather than an exception, an inconclusive run moves
    # nothing, and `propose_fix` refuses a fix for a defect never observed
    # failing. This stage only routes.
    experiments = []
    for spec in (out.get("experiment") or [])[:3]:
        if not isinstance(spec, dict):
            continue
        question = str(spec.get("hypothesis") or "").strip()
        source = str(spec.get("source") or "")
        if not question or not source.strip():
            warnings.append("dropped an experiment with no hypothesis or no "
                            "source")
            continue
        hyp = research.open_hypothesis(question, turn_idx, session_id)
        # The user's uploaded files are the working set: "run the tests in
        # this zip" is the same workspace as "here is a zip".
        files = workspace.snapshot_for_sandbox(session_id)
        files.update({str(k): str(v)
                      for k, v in (spec.get("files") or {}).items()
                      if isinstance(spec.get("files"), dict)})
        try:
            outcome = coding.run_experiment(
                hyp["id"], source=source, expect=spec.get("expect"),
                turn_idx=turn_idx, files=files,
                command=spec.get("command") or None,
                note=str(spec.get("note") or "")[:200])
        except Exception as exc:
            warnings.append(f"experiment harness failed: {str(exc)[:200]}")
            continue
        experiments.append({"hypothesis_id": hyp["id"],
                            "question": question[:120],
                            "outcome": outcome["outcome"],
                            "why": outcome["why"][:200],
                            "repeated": outcome["repeated"]})
        if outcome["repeated"]:
            warnings.append(
                "the same experiment disagreed with its earlier run — that is "
                "non-determinism, recorded as a dispute rather than averaged")
    fix = out.get("propose_fix")
    if isinstance(fix, dict) and str(fix.get("description") or "").strip():
        target = fix.get("hypothesis_id")
        if target is None and experiments:
            target = experiments[-1]["hypothesis_id"]
        if target is None:
            warnings.append("dropped a proposed fix naming no hypothesis")
        else:
            verdict = coding.propose_fix(
                int(target), description=str(fix["description"]),
                turn_idx=turn_idx)
            trace["proposed_fix"] = verdict
            if not verdict["accepted"]:
                warnings.append(f"fix refused: {verdict['why']}")
    if experiments:
        trace["experiments"] = experiments

    # -- Stage 4b: subagents (only when the user has allowed one) --
    #
    # Deliberately BEFORE the commit and outside it: a subagent is a network
    # round trip measured in minutes, and the embed-before-the-write-lock rule
    # applies with far more force to something that slow. The grant ledger is
    # its own small transaction inside `spawn`, so a spawn that crashes has
    # still spent its permission — which is the conservative direction. A
    # crashed child that cost nothing would be a retry loop.
    reports = []
    request = out.get("request_subagents")
    if isinstance(request, dict) and str(request.get("why") or "").strip():
        entry = subagents.record_request(
            request.get("kind"), request.get("count"), request.get("why"),
            turn_idx)
        if entry:
            trace["subagent_request"] = entry
            warnings.append(
                f"the assistant asked for {entry['count']} {entry['kind']} "
                f"subagent(s): {entry['why'][:120]}")
    specs = [s for s in (out.get("spawn") or [])[:4] if isinstance(s, dict)]
    # The parent divides the work; the engine checks the division is real.
    # Two agents assigned one file is not a prompting problem, it is a
    # property of the assignment — and a property can be checked before
    # anything is spent.
    batch, batch_warnings = subagents.spawn_cohort(
        specs, turn_idx=turn_idx, session_id=session_id,
        context=f"The user asked: {user_text}")
    warnings.extend(batch_warnings)
    for report in batch:
        warnings.extend(report.get("warnings") or [])
        # The child's edits, written back before it is forgotten.
        written, refused = subagents.apply_changes(report, session_id)
        if written:
            report["applied"] = written
            warnings.append(
                f"a {report['kind']} subagent changed {len(written)} file(s) "
                f"in your workspace: {', '.join(written[:6])}")
        for path in refused:
            warnings.append(f"refused a subagent change to {path!r}")
        reports.append(report)
    if reports:
        trace["subagents"] = [
            {"kind": r["kind"], "task": r["task"][:120],
             "claims": len(r["claims"]),
             "grounded_claims": sum(1 for c in r["claims"] if c["grounded"]),
             "evidence": len(r["evidence"]), "seconds": r["seconds"]}
            for r in reports]
        # The reply is REPLACED by what came back, the way research does it,
        # rather than the model's pre-spawn guess being kept beside a report
        # that may contradict it.
        digest = "\n\n".join(
            f"[{r['kind']} subagent] {r['summary']}" for r in reports
            if r["summary"])
        if digest:
            out["reply"] = ((str(out.get("reply") or "").strip() + "\n\n"
                             + digest).strip())

    reply = str(out.get("reply") or "").strip()

    # THE LAST POINT A HALT IS HONOURED. Past this line every durable mutation
    # of the turn happens in one transaction, and an interruption inside it
    # would leave the ordinal consumed with the turn half-written — precisely
    # what reserving the ordinal early was meant to prevent. `enter_commit`
    # latches the run so a halt arriving from here on is answered "too late"
    # rather than silently dropped, which is what lets the button tell the
    # truth about what it did.
    run.halted()
    run.emit("commit", state="committing")
    run.enter_commit()

    # -- Stage 5: commit (one transaction) --
    # Build + embed the memory batch BEFORE the transaction: embedding is a
    # network round trip and must never hold SQLite's writer.
    to_mint = []
    exchange = f"User said: {user_text}\nI replied: {reply}"
    to_mint.append(dict(
        kind="episodic", provenance="witnessed",
        salience=_salience_of(exchange), content=exchange,
        session_id=session_id, turn_idx=turn_idx,
        event_key=f"turn:{turn_idx}:episode"))
    for n, mark in enumerate(out.get("remember") or []):
        if not isinstance(mark, dict):
            continue
        content = str(mark.get("content") or "").strip()
        if not content:
            continue
        prov = mark.get("provenance")
        prov = prov if prov in ("told", "witnessed", "read") else "told"
        # The character decides what was worth hearing: the engine measured
        # its fixed phrase-list gate catching 0 of 125 model marks while
        # marked lines were retrieved at 3x the base rate. The mark is the
        # gate; salience gets a floor above the episode's so kept facts
        # outrank the exchange that carried them.
        to_mint.append(dict(
            kind="dialogue" if prov == "told" else "semantic",
            provenance=prov, salience=max(0.82, _salience_of(content)),
            content=content, session_id=session_id,
            turn_idx=turn_idx, event_key=f"turn:{turn_idx}:kept:{n}"))
    for n, update in enumerate(model_updates):
        claim = str(update.get("claim") or "").strip()
        confidence = float(update.get("confidence") or 0.5)
        about = str(update.get("about") or USER_SUBJECT).strip() \
            or USER_SUBJECT
        # salience = 0.45 + 0.3*confidence — the reconstructible mint
        # formula. Reconciliation later recovers the mint confidence by
        # inverting it, which only works because salience is never revised.
        to_mint.append(dict(
            kind="inference", provenance="inferred",
            salience=memory.mint_salience(confidence),
            content=f"I concluded about {about}: {claim}",
            gist=claim, entities=[about], confidence=confidence,
            session_id=session_id, turn_idx=turn_idx,
            event_key=f"turn:{turn_idx}:inference:{n}"))
    prepared = memory.prepare_memories_batch(to_mint)
    if prepared.get("embedded") is not None \
            and getattr(prepared["embedded"], "fallback", False):
        err = getattr(prepared["embedded"], "error", "")
        if err:
            warnings.append(f"embeddings degraded to lexical fallback: {err}")

    with transaction():
        turn_id = qi("INSERT INTO turns(session_id,turn_idx,user_text,created) "
                     "VALUES(?,?,?,strftime('%s','now'))",
                     (session_id, turn_idx, str(user_text or "")))
        # The prepared batch was embedded before the lock and so could not
        # carry a row id that did not exist yet. Stamping it here keeps the
        # network round trip outside the transaction without giving up the
        # turn_id link that delete_turn_memories reruns on.
        for data in prepared.get("prepared") or []:
            data["turn_id"] = turn_id
        # Re-read state INSIDE the write lock. The copy stage 1 took is
        # seconds old by now — a model call old — and stage 5 writes the whole
        # blob back. Two turns overlapping meant the later writer silently
        # erased the earlier one's belief updates, hypothesis keys and pending
        # ponder. Read-modify-write is only atomic if the read is in here too.
        state = state_get("assistant")
        # Rerun-safety even though there is no rerun UI yet: a replayed turn
        # replaces its memories rather than duplicating them. delete-then-
        # insert plus event_key upserts, same as the engine.
        memory.delete_turn_memories(turn_id)
        memory.add_memories_batch(prepared_batch=prepared)
        beliefs.apply_belief_updates(state, model_updates, turn_idx)
        # Deliberately AFTER the belief write, so this turn's own fresh
        # inference rows are re-weighted by the same reconciled store
        # everything else now reads.
        memory.reconcile_inference_confidence(
            state, turn_idx, beliefs.belief_credence)
        # Importance: consequence, not popularity. Only refs the model built
        # something on (belief evidence) or leaned its reply on
        # (memory_evidence_used with a formed update) get lifted, once ever.
        # Both halves of what this comment has always claimed. `evidence` on a
        # formed belief update was the only half implemented; refs the model
        # leaned its REPLY on were named here and never read, so on a corpus
        # where the model grounds in "current" (the ordinary case) importance
        # was lifted zero times — measured 0 revisions across 40 turns. A
        # memory that visibly carried an answer is consequence, which is the
        # standard this gate applies; it is still once-ever, so the popularity
        # loop stays structurally closed.
        load_bearing = {r for u in model_updates
                        for r in u.get("evidence", []) if r != "current"}
        load_bearing |= {r for r in out.get("memory_evidence_used") or []
                         if r != "current"}
        if load_bearing:
            memory.raise_importance(sorted(load_bearing),
                                    only_unrevised=True)
        # Subagent reports enter as TESTIMONY, inside the same transaction
        # as everything else this turn learned. provenance `told`, attributed
        # to the child: a subagent is a second mind, and absorbing its report
        # as first-hand would be the information-layer collapse this whole
        # system exists to prevent, at the one seam where a second mind
        # actually exists.
        for report in reports:
            subagents.absorb(report, task=report["task"], turn_idx=turn_idx,
                             session_id=session_id)
        # RETIREMENT: relevance, not truth, and grounded like every other
        # citation. A ref the model was not shown cannot be retired — the
        # same rule that stops it citing a memory it never saw stops it
        # discarding one.
        retire = out.get("retire")
        if isinstance(retire, dict) and (retire.get("memory_refs") or None):
            refs = _ground_evidence_list(retire.get("memory_refs"), delivered,
                                         warnings, "retire")
            refs = [r for r in refs if r != "current"]
            if refs:
                outcome = memory.retire_memories(
                    refs, reason=str(retire.get("reason") or ""),
                    turn_idx=turn_idx)
                if not outcome["ok"] and outcome.get("error"):
                    warnings.append(f"retirement refused: {outcome['error']}")
                for refusal in outcome.get("refused") or []:
                    warnings.append(
                        f"kept {refusal['ref']}: {refusal['why']}")
                if outcome["retired"]:
                    trace["retired"] = outcome
                    warnings.append(
                        f"retired {len(outcome['retired'])} memories as no "
                        f"longer relevant: {outcome['reason']}")
                    # The act of forgetting is itself remembered, and the note
                    # is protected from retirement. An assistant that can
                    # forget that it forgot will confidently tell you it never
                    # knew — which is a worse failure than the clutter this
                    # feature removes.
                    memory.add_memory(
                        "episodic", "witnessed", 0.8,
                        f"retired: I set aside {len(outcome['retired'])} "
                        f"memories as no longer relevant to what we are "
                        f"working on. Reason: {outcome['reason']}. "
                        f"Refs: {', '.join(outcome['retired'][:12])}",
                        turn_idx=turn_idx, session_id=session_id,
                        event_key=f"turn:{turn_idx}:retirement")
        dispute = out.get("dispute")
        if isinstance(dispute, dict) and str(
                dispute.get("reading") or "").strip():
            ref = str(dispute.get("memory_ref") or "").strip()
            if ref in delivered:
                updated = memory.record_dispute(
                    dispute["reading"], turn_idx, memory_ref=ref)
                if updated:
                    trace["disputed"] = ref
            else:
                warnings.append(
                    f"dropped dispute citing undelivered ref {ref!r}")
        # The stable hypothesis sheet, selected at commit where the
        # reconciled beliefs exist; hysteresis lives in beliefs.py.
        entries, keys = beliefs.select_active_hypotheses(
            state.get("mind_models"), state.get("active_hypothesis_keys"),
            turn_idx)
        state["active_hypothesis_keys"] = keys
        # Ponder: one pending query, consumed next turn by stage 1.
        ponder = out.get("ponder")
        if (isinstance(ponder, dict)
                and str(ponder.get("query") or "").strip()
                and str(ponder.get("why") or "").strip()):
            state["pending_ponder"] = str(ponder["query"])[:240]
        else:
            state["pending_ponder"] = ""
        state_put("assistant", state)
        trace["warnings"] = warnings
        trace["minted"] = len(to_mint)
        qi("UPDATE turns SET reply_text=?, trace=? WHERE id=?",
           (reply, json.dumps(trace, ensure_ascii=False), turn_id))

    # -- Stage 6: settle (post-commit; reconstructible cache) --
    if chat_configured():
        try:
            memory.maybe_consolidate(turn_idx, _llm_consolidator)
        except Exception as exc:
            # A consolidation failure must never look like a turn failure:
            # summaries are reconstructible caches over rows that survived.
            warnings.append(f"consolidation failed: {str(exc)[:200]}")

    return {"reply": reply, "warnings": warnings, "trace": trace,
            "respond_ok": respond_ok,
            "session_id": session_id, "turn_idx": turn_idx,
            "turn_id": turn_id}


def _llm_consolidator(payload):
    raw = chat_complete(prompts.render(prompts.CONSOLIDATE_SYSTEM),
                        json.dumps(payload, ensure_ascii=False),
                        temperature=0.1, max_tokens=3000)
    out = parse_model_json(raw)
    if out is None:
        raise RuntimeError("consolidator returned invalid JSON")
    return out
