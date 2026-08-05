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
import sandbox
import subagents
import tools_web
import turnrun
import workspace
from db import (ensure_session, next_turn_idx, q, qi, state_get, state_put,
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


# How much of one side of an exchange an episode keeps. Generous enough that
# an ordinary turn is stored whole — the median memory in this bank is well
# under it — and small enough that a pasted brief cannot become the largest
# row in the corpus and then outrank everything in recall forever.
EPISODE_HALF_CHARS = 1200


def _episode_half(text):
    """One side of an exchange, cut with the cut declared."""
    text = str(text or "")
    if len(text) <= EPISODE_HALF_CHARS:
        return text
    marker = f" …[cut: {len(text)} chars total]"
    return text[:EPISODE_HALF_CHARS - len(marker)].rstrip() + marker


def _ref_in_bank(name):
    """The STORED spelling of a ref, or "" when no such row exists anywhere.

    TWO SPELLINGS OF ONE THING, folded where the data enters rather than at
    each caller who must remember. Memory rows are keyed bare
    (`turn:61:episode`) while evidence rows are keyed WITH their prefix
    (`evidence:1335…`), so a model that writes `event:` in front of a memory
    ref names a real row under a name nothing stores — and six citations to
    rows that plainly existed were destroyed as inventions for exactly that."""
    name = str(name or "").strip()
    if not name:
        return ""
    candidates = [name]
    if name.startswith("event:"):
        candidates.append(name[len("event:"):])
    for cand in candidates:
        for sql in ("SELECT 1 FROM memories WHERE event_key=? LIMIT 1",
                    "SELECT 1 FROM evidence WHERE event_key=? LIMIT 1",
                    "SELECT 1 FROM chunks WHERE chunk_key=? LIMIT 1"):
            if q(sql, (cand,), one=True):
                return cand
    return ""


def _ground_evidence_list(refs, delivered, warnings, path, *, bank_ok=False):
    """Engine citation discipline: keep refs that were delivered (or the
    literal "current", which names the user's message this turn); drop the
    rest with a warning. Never invent one the model omitted.

    `bank_ok` SEPARATES CITING FROM ACTING, and they are not the same risk.
    Citing a real row the assistant produced earlier is honest bookkeeping.
    DISCARDING a row it was never shown is a destructive act on something it
    cannot have read — so retirement stays strictly delivered-only and passes
    this flag false. A ref it was not shown is not a ref it may act on.

    "I INVENTED THIS" AND "RECALL DID NOT SURFACE IT" NEED OPPOSITE
    CORRECTIONS, and this told them apart by comparing against the delivered
    set alone — so a row still sitting in the bank read exactly like a
    fabrication. Measured over 71 turns: of 23 distinct dropped citations, 15
    named rows that existed. The assistant was citing experiment results it
    had produced itself 8 to 34 turns earlier, being told they were
    ungrounded, and re-running the experiment to get them back — nine
    hypotheses on one question across turns 38 to 70.

    So a ref the bank can resolve is KEPT, and the fact that recall missed it
    is reported rather than the citation destroyed. What the gate still
    refuses is the thing it was actually built to refuse: a name that
    corresponds to nothing."""
    grounded = []
    for ref in refs or []:
        name = str(ref or "").strip()
        if name == "current" or name in delivered:
            if name not in grounded:
                grounded.append(name)
            continue
        stored = _ref_in_bank(name) if bank_ok else ""
        if stored:
            if stored not in grounded:
                grounded.append(stored)
            # Not a failure of the citation; a failure of retrieval. Named as
            # such so the recall miss is countable instead of arriving
            # disguised as the model making things up.
            warnings.append(
                f"kept {path} citation {name!r}: it is in the bank, but "
                "recall did not surface it this turn")
            continue
        warnings.append(f"dropped ungrounded {path} citation {name!r} — no "
                        "such row exists")
    return grounded


def _aspects(sheet_entries, threads, persona_sheet):
    """What the assistant BRINGS to the turn, as separate retrieval facets —
    each gets its own RRF ranking (see memory.search_memories for why
    concatenation was measured useless)."""
    return [
        ("what i am wondering about",
         " ".join(e.get("i_suspect", "") for e in sheet_entries[:2])),
        ("what is still unsettled",
         " ".join(memory.thread_text(t) for t in (threads or []))),
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
# RAISED FROM 3, AND THE MEASUREMENT IS WHY. Landing one anchored edit costs
# orient → outline → expand → answer: four rounds, and three was the ceiling.
# A real turn ran out with the assistant reporting "surfacing ids and then
# expanding them is two rounds; I have one" — so the budget, not its
# judgement, is what stopped the work. The model is told how many remain and
# stops as soon as it is satisfied, so this raises the ceiling on hard turns
# without spending anything extra on easy ones.
DELIBERATION_MAX_ROUNDS = 5
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

    Returns `(out, deliberation, delivered, cost)` — `delivered` being the
    memory refs this stage handed over MID-TURN. They have to reach the
    citation gate or the assistant is given material it is then forbidden to
    cite, which is indistinguishable from not having looked.

    `cost` is the measurement, not an estimate: the system prompt is re-sent
    on EVERY round and the payload grows as `what_i_went_and_got` accumulates,
    so the price of a turn is a sum over rounds and nothing recorded it. Every
    proposal to make this cheaper — trim recall, route sections, cache the
    prefix — was an argument about numbers no one had. Recorded against every
    turn, so the denominator is turns that HAD the opportunity to deliberate
    rather than turns that did."""
    system = prompts.render(prompts.RESPOND_SYSTEM,
                            persona=persona.persona_prompt(persona_sheet))
    deliberation = []
    delivered = set()
    cost = {"system_chars": len(system), "rounds": []}
    out = None
    for round_no in range(1, DELIBERATION_MAX_ROUNDS + 1):
        run.halted()
        run.emit("respond", state=("calling the model" if round_no == 1
                                   else f"deliberating (round {round_no})"),
                 round=round_no)
        # THE USER MAY SPEAK WHILE IT IS THINKING, and be heard THIS turn.
        # A correction that arrives mid-turn and is not read until the turn
        # ends is a correction applied to work already finished — so the only
        # way to redirect was to halt, which throws away everything the turn
        # had established to say one sentence. Read at a round boundary, which
        # is where the loop already re-decides what to do next.
        #
        # Carried in `what_i_went_and_got` rather than by rewriting the
        # original message: the model must be able to tell "you asked me this"
        # from "you have since said this", and it cannot if they are merged.
        #
        # ADDITIONAL, NOT OVERRIDING. The engine takes no view on whether an
        # interjection cancels the work in flight — "also check the tests" and
        # "stop, wrong file" arrive through the same channel and only the
        # model can tell them apart. A rule here would have to guess, and the
        # cost of guessing wrong is a user who learns not to speak up.
        for said in run.drain_inbox():
            deliberation.append({"got": "the user, mid-turn", "said": said,
                                 "note": "this arrived AFTER the message you "
                                         "are answering. Fold it into what "
                                         "you are already doing; abandon that "
                                         "only if they have actually "
                                         "redirected you."})
        body = dict(payload)
        if deliberation:
            body["what_i_went_and_got"] = deliberation
        body["deliberation_rounds_left"] = DELIBERATION_MAX_ROUNDS - round_no
        # Serialised once and measured, rather than measured by serialising a
        # second time: the instrument must not become a cost of its own.
        sent = json.dumps(body, ensure_ascii=False)
        cost["rounds"].append(len(sent))
        # THE RAW OUTPUT DIES INSIDE THIS EXPRESSION, and it is the only
        # evidence of why a turn failed. `parse_model_json(chat_complete(...))`
        # threw the model's actual words away at the moment they became
        # interesting: every "respond stage returned unparseable output" since
        # turn 79 has been unfalsifiable, because the thing that would say
        # whether it was truncation, a fence, a refusal or a provider error was
        # already gone. Observed again at turn 117 — a 496,743-character
        # payload over four rounds, and nothing anywhere recording what came
        # back.
        raw = chat_complete(system, sent)
        out = parse_model_json(raw)
        if out is None:
            # Both ends, because the two diagnoses live at opposite ones: a
            # refusal or a prose preamble shows at the head, and truncation at
            # max_tokens shows as a sentence that simply stops at the tail.
            text = " ".join(str(raw or "").split())
            cost["unparseable"] = {
                "chars": len(str(raw or "")),
                "head": text[:400],
                "tail": text[-400:] if len(text) > 800 else "",
                "round": round_no,
                "sent_chars": len(sent),
            }
            return None, deliberation, delivered, cost
        more = out.get("need_more")
        if not isinstance(more, dict) or round_no == DELIBERATION_MAX_ROUNDS:
            break
        step, refs = _gather(more, turn_idx, session_id, run, warnings)
        delivered |= refs
        if not step:
            break                 # nothing to fetch: the answer it has stands
        deliberation.append(step)
    return out, deliberation, delivered, cost


def _gather(more, turn_idx, session_id, run, warnings):
    """Fetch what a deliberation round asked for. Deterministic; no judgement
    about whether the request was wise — that is the model's to make and the
    engine's to bound.

    Returns `(step, delivered_refs)`. The refs matter as much as the material:
    a memory handed over mid-turn that the citation gate has never heard of is
    a memory the assistant cannot use, and the gate is right to strip it."""
    step = {}
    delivered = set()
    query = str(more.get("ponder") or "").strip()
    if query:
        run.emit("ponder", query=query)
        rows = memory.search_memories(query, k=PONDER_RECALL_LIMIT,
                                      current_turn_idx=turn_idx)
        # `event_key` IS the ref, everywhere. This read `r.get("ref")`, which
        # is not a key any memory row has ever carried, so every pondered
        # memory arrived as `ref: null` — ten of them in one measured turn.
        # Under the citation rule a memory that cannot be named cannot be
        # used, so the whole mid-turn ponder lane returned material the
        # assistant then had to answer as though it had never seen.
        #
        # It failed silently in both directions: `.get` on a missing key is
        # None rather than an error, and a null ref reads as "this memory
        # happens not to have one" rather than "the lane is broken".
        delivered |= {str(r.get("event_key") or "") for r in rows
                      if r.get("event_key")}
        step["ponder"] = {
            "query": query,
            "returned": len(rows),
            "memories": [{"ref": r.get("event_key") or "",
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
    # Filename → ids. The step between knowing what to edit and being able to
    # read it, which did not exist: `digest` ranks the workspace against the
    # turn's message, so "Go for it" surfaced no code at all and the file the
    # assistant had been asked to change was unreachable.
    wants = more.get("outline")
    if isinstance(wants, str) or isinstance(wants, list):
        paths = [wants] if isinstance(wants, str) else [str(p) for p in wants]
        step["outlines"] = [chunks.outline(p, session_id) for p in paths[:4]]
        run.emit("outline", paths=paths[:4],
                 chunks=sum(o.get("chunks", 0) for o in step["outlines"]))
    ids = more.get("expand_chunks")
    if isinstance(ids, list) and ids:
        expanded = chunks.expand(session_id, ids)
        step["expanded_chunks"] = expanded
        run.emit("expand", ids=[str(i)[:24] for i in ids[:8]],
                 got=len([e for e in expanded if not e.get("unknown")]))
    search_query = str(more.get("search") or "").strip()
    if search_query:
        run.emit("search", query=search_query)
        # `search_detail`, NOT `search`. The collapsing version reports a
        # blocked backend as `nothing_found`, which is the exact defect that
        # function was written to end — fixed in the research loop and left
        # standing here, the lane the assistant reaches for most. Measured
        # against the live backend: `BBC News` returns 5 results while
        # `AI therapy chatbot randomized` is served an anti-bot challenge,
        # deterministically, four attempts apart. So a control query PROVES
        # THE LANE WORKS while the real question is silently refused, and the
        # assistant spent six searches across two agents concluding the web
        # had nothing to say.
        detail = {"results": [], "status": "error", "detail": ""}
        try:
            detail = tools_web.search_detail(
                search_query, max_results=PONDER_SEARCH_RESULTS)
        except Exception as exc:
            detail["detail"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        results = detail["results"]
        step["search"] = {"query": search_query, "results": results,
                          "status": detail["status"],
                          **({"unavailable": detail["detail"]}
                             if detail["status"] in ("blocked", "error")
                             else {}),
                          **({"nothing_found": True}
                             if not results and detail["status"] == "empty"
                             else {})}
        if detail["status"] in ("blocked", "error"):
            warnings.append(f"search unavailable: {detail['detail'][:160]}")
        run.emit("search", query=search_query, results=len(results),
                 status=detail["status"], detail=detail["detail"][:200])
    return step, delivered


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

    def drain_inbox(self):
        return ()


_UNOBSERVED = _Unobserved()


def run_turn(user_text, session_id=None, run=None, speaker="user",
             carried_plan=""):
    """One full exchange. Returns {reply, warnings, trace, session_id,
    turn_idx}.

    `speaker` SAYS WHO IS TALKING, and it is not cosmetic. An automation
    iteration is driven by the assistant's own `continue_work`, and with one
    speaker the episode row for it read "User said: <the assistant's own
    plan>" — a witnessed memory of words the user never uttered, minted into
    the bank every iteration and recalled later as fact about them. A memory
    system whose provenance can be wrong about WHO SPOKE is worse than one
    with less memory.

    `carried_plan` is what the assistant was about to do when the user
    interrupted. Delivered as context beside their message, never merged into
    it: steering must not silently discard the work in flight, and the model
    is the only thing that can tell "stop doing that" from "also, note this".

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
                         "ref": "current",
                         "spoken_by": ("the user" if speaker == "user" else
                                       "you — this is your own next step from "
                                       "last iteration, not a new message")},
        # WHAT YOU WERE ABOUT TO DO, when the user spoke over it.
        #
        # Steering must not be an interrupt. Their message used to REPLACE the
        # next step outright, so "also, check the tests" threw away a plan
        # three iterations deep — the user paying attention was penalised for
        # it, which is the opposite of what a steerable run should feel like.
        #
        # Carried as context beside their words rather than merged into them,
        # and the engine takes no view on which wins. Only the model can tell
        # "stop doing that" from "also, note this", and a rule here would have
        # to guess at exactly the moment guessing is most expensive.
        **({"work_in_progress": {
            "you_were_about_to": carried_plan,
            "note": "the user spoke while you were working, and their message "
                    "is `user_message` above. It does not automatically "
                    "cancel this. Fold it in and carry on where you can; drop "
                    "the plan only if they have actually redirected you, and "
                    "say which you did."}} if carried_plan else {}),
        "memory": memory_payload,
        "what_i_believe_about_people_and_topics": user_model,
        "active_hypotheses": sheet_entries,
        # The tally travels with the confidence, always. A bare number cannot
        # say whether it moved, and "open at 0.545" reads as an untouched
        # default until you can see the one supporting row that put it there.
        # `status` is worth the same care: an experiment whose prediction held
        # is evidence and moves confidence — it does not close a question, so
        # "open" after a confirmed run is correct rather than stuck.
        "open_research": [
            # THE ID, BECAUSE TWO GATES ASK FOR IT BY NUMBER. `propose_fix`
            # and an anchored `edit_files` both take a `hypothesis_id`, and
            # this payload described the open questions in prose without ever
            # naming one — so the only way to satisfy those fields was to
            # guess an integer, which is precisely the ritual the
            # reproduce-before-you-fix gate exists to prevent. The assistant
            # hit it and declined to guess, correctly, and reported the gap
            # rather than routing around it. A field the engine requires and
            # the payload withholds is the engine's defect.
            {"id": h["id"],
             "question": h["question"], "status": h["status"],
             "confidence": h["confidence"],
             # The prior travels too. The tally alone was not enough: three
             # hypotheses at 0.545 with `supports: 1` apiece still read as
             # "unmoved by evidence that supports them", because nothing in
             # the payload said what they had moved FROM.
             "opened_at": research.PRIOR_CONFIDENCE,
             "evidence": research.evidence_tally(h["id"]),
             "last_moved_turn": h["updated_turn"]}
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
            out, deliberation, mid_turn_refs, cost = _deliberate(
                payload, persona_sheet, turn_idx, session_id, run, warnings)
            # WHAT THE TURN COST, BROKEN DOWN WHERE A DECISION COULD ACT ON
            # IT. A total says the turn was expensive; a section says which
            # proposal would have helped. Recorded every turn, including the
            # single-round ones — the interesting question is what share of
            # turns pay the multiplier, and that needs the cheap turns in the
            # denominator too.
            cost["sections"] = {k: len(json.dumps(v, ensure_ascii=False,
                                                  default=str))
                                for k, v in payload.items()}
            cost["total_chars"] = (cost["system_chars"] * len(cost["rounds"])
                                   + sum(cost["rounds"]))
            trace["payload_cost"] = cost
            # The gate's definition of "delivered" has to include what the
            # deliberation loop went and fetched. It was fixed at stage 1, so
            # a memory the assistant asked for and received DURING the turn
            # was stripped from its own citations as if invented.
            delivered |= mid_turn_refs
            if deliberation:
                trace["deliberation"] = deliberation
            if out is None:
                # SAY WHICH FAILURE IT WAS. "Unparseable output" names the
                # symptom and nothing else, and the operator's next move
                # differs completely between a payload that hit max_tokens and
                # a provider that returned an error page.
                bad = (cost or {}).get("unparseable") or {}
                if bad:
                    warnings.append(
                        "respond stage returned unparseable output: "
                        f"{bad['chars']:,} chars back on a "
                        f"{bad['sent_chars']:,}-char payload — begins "
                        f"{bad['head'][:120]!r}"
                        + (f", ends {bad['tail'][-120:]!r}"
                           if bad.get("tail") else ""))
                else:
                    warnings.append(
                        "respond stage returned unparseable output")
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
    # WHAT IT INTENDS TO DO NEXT, if anything — read here, acted on by
    # `autoloop`. A turn that fails to compose a reply asks for nothing: a
    # provider error must not be able to drive an automation loop, and an
    # absent field stops rather than continues, so the way this mechanism
    # fails is by halting.
    continue_work = ""
    if respond_ok:
        nxt = out.get("continue_work")
        if isinstance(nxt, dict):
            nxt = nxt.get("next")
        continue_work = str(nxt or "").strip()[:2000]
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
        "memory_evidence", bank_ok=True)
    model_updates = []
    for update in out.get("user_model_updates") or []:
        if not isinstance(update, dict) or not str(
                update.get("claim") or "").strip():
            continue
        evidence = _ground_evidence_list(update.get("evidence"), delivered,
                                         warnings, "user_model evidence",
                                         bank_ok=True)
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
        command = spec.get("command") or None
        # A COMMAND IS A WAY OF RUNNING SOMETHING TOO. Requiring `source`
        # meant an experiment over the files already in the workspace — "run
        # the suite in this repository" — had to invent a program first, so
        # the assistant's own verification runs were dropped here before they
        # executed and it could only report a fix landing with nothing that
        # ran it. `coding.run_experiment` refuses the genuinely empty case.
        #
        # The warning also says WHICH half is missing. "no hypothesis or no
        # source" left the author guessing at their own mistake.
        if not question or not (source.strip() or command):
            missing = "hypothesis" if not question else "source or command"
            warnings.append(f"dropped an experiment with no {missing}")
            continue
        hyp = research.open_hypothesis(question, turn_idx, session_id)
        # The user's uploaded files are the working set: "run the tests in
        # this zip" is the same workspace as "here is a zip".
        files = workspace.snapshot_for_sandbox(session_id)
        files.update({str(k): str(v)
                      for k, v in (spec.get("files") or {}).items()
                      if isinstance(spec.get("files"), dict)})
        # THE MOST INTERESTING THING A TURN DOES, AND IT RAN SILENTLY. The
        # stage emitted nothing at all: a suite run holds the turn for up to
        # three minutes with no step in the panel, which is the same "is it
        # working or is it hung" ambiguity the streaming work removed from the
        # respond stage — reintroduced at the one stage where the answer takes
        # longest.
        #
        # Announced BEFORE the run, carrying the prediction. That ordering is
        # the discipline made visible: `expect` was written before anything
        # executed, and a panel that only ever showed the verdict could not
        # tell that apart from a result narrated afterwards.
        run.emit("experiment", state="running", hypothesis=question[:200],
                 predicted=json.dumps(spec.get("expect") or {},
                                      ensure_ascii=False)[:300],
                 command=" ".join(str(c) for c in (command or []))[:200],
                 timeout=spec.get("timeout") or sandbox.DEFAULT_TIMEOUT)
        try:
            outcome = coding.run_experiment(
                hyp["id"], source=source, expect=spec.get("expect"),
                turn_idx=turn_idx, files=files,
                command=command, timeout=spec.get("timeout"),
                note=str(spec.get("note") or "")[:200],
                cwd=str(spec.get("cwd") or ""),
                collect=[str(p) for p in (spec.get("collect") or [])
                         if isinstance(spec.get("collect"), list)])
        except Exception as exc:
            run.emit("experiment", state="the harness itself failed",
                     hypothesis=question[:200], detail=str(exc)[:200])
            warnings.append(f"experiment harness failed: {str(exc)[:200]}")
            continue
        # The verdict, with what was actually observed beside it. `why` is the
        # grader's sentence — which predicate failed, or which harness cue
        # fired — and without it "refuted" is a label the panel cannot justify.
        result = outcome.get("result") or {}
        run.emit("experiment", state=outcome["outcome"],
                 hypothesis=question[:200], why=outcome["why"][:300],
                 exit_code=result.get("exit_code"),
                 seconds=result.get("seconds"),
                 timed_out=bool(result.get("timed_out")),
                 stdout=(result.get("stdout") or "")[-600:],
                 stderr=(result.get("stderr") or "")[-600:],
                 repeated=outcome["repeated"])
        experiments.append({"hypothesis_id": hyp["id"],
                            "question": question[:120],
                            "outcome": outcome["outcome"],
                            "why": outcome["why"][:200],
                            "repeated": outcome["repeated"]})
        if outcome.get("shadowed"):
            # A warning is the system WORKING: the run happened, and this says
            # the program that ran was not the one the caller thought it wrote.
            warnings.append(
                f"`source` overwrote your own {outcome['shadowed']} — source "
                "always lands in main.py, so name the program you want run "
                "either in `source` or in `files`, never both")
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
            # The gate is the point of the module, and it was invisible. A
            # refusal is not an error — it is the mechanism working — so it
            # belongs in the trail beside the run that failed to justify it.
            run.emit("fix", state=("accepted" if verdict["accepted"]
                                   else "refused"),
                     why=str(verdict.get("why") or "")[:300],
                     description=str(fix["description"])[:200])
            if not verdict["accepted"]:
                warnings.append(f"fix refused: {verdict['why']}")
    if experiments:
        trace["experiments"] = experiments

    # -- Stage 4c: edits (the durable half of the coding suite) --
    #
    # Before this the assistant could reproduce a defect, design a fix and
    # prove it correct in the sandbox — and then had nowhere to put it, because
    # `sandbox.run` writes into a directory deleted the moment the run ends.
    # The deliverable of a coding turn is a changed file and a diff somebody
    # can review; neither existed, so the loop terminated in an opinion.
    #
    # AFTER the experiments deliberately. An edit naming a hypothesis is gated
    # on that hypothesis having been observed failing, and running the
    # reproduction in the same turn as the fix is the ordinary case — so the
    # gate has to read a table the experiments above have already written.
    edits = []
    for spec in (out.get("edit_files") or [])[:8]:
        if not isinstance(spec, dict):
            continue
        path = str(spec.get("path") or "").strip()
        replace = spec.get("replace")
        replace = replace if isinstance(replace, list) and replace else None
        if not path or (spec.get("contents") is None and not replace):
            warnings.append("dropped an edit with no path, no contents and "
                            "no replacements")
            continue
        target = spec.get("hypothesis_id")
        if target is None and str(spec.get("fixes") or "").strip() and experiments:
            target = experiments[-1]["hypothesis_id"]
        try:
            done = coding.apply_edit(
                path, spec.get("contents"), turn_idx=turn_idx,
                replace=replace,
                hypothesis_id=int(target) if target is not None else None,
                why=str(spec.get("why") or "")[:200], session_id=session_id)
        except Exception as exc:
            warnings.append(f"edit harness failed: {str(exc)[:200]}")
            run.emit("edit", state="failed", path=path, why=str(exc)[:200])
            continue
        if not done["ok"]:
            # AS LOUD AS A SUCCESS, AND FOR THE SAME REASON THE SUCCESS IS.
            # This path appended a warning while the one below emitted, so
            # the live trace showed every edit that landed and none that was
            # turned away. A refusal that is quieter than an acceptance is
            # the exact shape of failure that gets read as success.
            warnings.append(f"edit refused for {path}: {done['why']}")
            run.emit("edit", state="refused",
                     path=done.get("path") or path, why=done["why"])
            continue
        run.emit("edit", state="applied", path=done["path"],
                 created=done["created"], rechunked=done["rechunked"])
        edits.append({k: done[k] for k in
                      ("path", "diff", "created", "unchanged", "rechunked")})
    if edits:
        # The DIFF goes in the trace, not a line count. An edit reported as
        # "wrote 812 lines" is unreviewable — the reader has to hold both
        # versions to find the change, which is the work the diff exists to do.
        trace["edits"] = edits
        warnings.append(
            "edited " + ", ".join(e["path"] for e in edits)
            + " — the diffs are in this turn's reasoning trail")

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
        context=(f"The user asked: {user_text}" if speaker == "user" else
                 f"Continuing its own work, the assistant set out to: "
                 f"{user_text}"))
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
    # WHO SPOKE IS PART OF THE MEMORY. An automation iteration is driven by
    # the assistant's own `continue_work`, and this line minted it as "User
    # said: <the assistant's own plan>" — a witnessed episode of words the
    # user never uttered, once per iteration, recalled later as fact about
    # them. Retrieval cannot repair a row whose provenance is wrong, and
    # nothing downstream could have caught it: the row is well-formed, richly
    # salient, and false.
    # AN EPISODE IS A RECORD OF AN EXCHANGE, NOT A TRANSCRIPT OF IT. Both
    # halves were stored whole, so a long prompt became a permanent memory as
    # long as itself — the largest in this bank is 13,407 characters, and it is
    # a pasted audit brief. Those rows then dominate every later recall,
    # because a long prompt is also a rich match for questions about its own
    # topic: turn 79's memory block reached 236,870 characters, 92% of the
    # payload, and the model returned unparseable output.
    #
    # Truncated with the true length recorded, exactly as `research.py` does
    # for evidence excerpts. The archive says when it is a partial copy —
    # silently storing half an exchange as though it were the whole one is the
    # failure that pattern exists to prevent.
    said, told = _episode_half(user_text), _episode_half(reply)
    exchange = (f"User said: {said}\nI replied: {told}"
                if speaker == "user" else
                f"Continuing my own work, I set out to: {said}\n"
                f"I then reported: {told}")
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
        # A thread the turn answered is closed by the turn that answered it.
        # Waiting for the consolidator meant a question could go on being
        # asked for ten more turns after the payload beside it had settled
        # it — the same append-only staleness as the summary prose, one level
        # down. Quoted text, not an index, and what does not match is said.
        closing = out.get("resolved_threads")
        if isinstance(closing, list) and closing:
            outcome = memory.close_threads([str(t) for t in closing][:12],
                                           turn_idx)
            if outcome["closed"]:
                trace["closed_threads"] = outcome["closed"]
            for miss in outcome["unknown"]:
                warnings.append(
                    f"no open thread matched {miss[:60]!r}; nothing closed")
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
            "respond_ok": respond_ok, "continue_work": continue_work,
            "session_id": session_id, "turn_idx": turn_idx,
            "turn_id": turn_id}


def _llm_consolidator(payload):
    raw = chat_complete(prompts.render(prompts.CONSOLIDATE_SYSTEM),
                        json.dumps(payload, ensure_ascii=False),
                        temperature=0.1)
    out = parse_model_json(raw)
    if out is None:
        raise RuntimeError("consolidator returned invalid JSON")
    return out
