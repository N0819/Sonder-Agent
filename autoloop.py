# autoloop.py — many turns, one run, no turn limit.
#
# A turn ends when the assistant has something to say. That is the right unit
# for a conversation and the wrong one for a task: landing a verified change
# costs orient, outline, read, edit, run, read the result — and the honest
# ones are separated by a COMMIT, because what one iteration establishes has
# to be durable before the next builds on it. Raising the deliberation ceiling
# does not buy that. Everything an unbounded single turn learns is held in
# memory until stage 5, so a failure at round forty loses forty rounds; and
# the mid-turn lane cannot run an experiment, apply an edit, or mint the
# evidence row that makes the run citable next time.
#
# So the loop is OUTSIDE the turn. Each iteration is an ordinary
# `pipeline.run_turn` — same stages, same grounding gate, same one-transaction
# commit — and the loop only decides whether there is another one. Nothing
# here is a second pipeline, and nothing here writes to the database.
#
# WHAT ENDS A RUN, in the order it is checked:
#
#   1. The user halts. `run.halted()` already raises out of the pipeline;
#      this loop just does not start another iteration.
#   2. The assistant omits `continue_work`. Absence stops, so every way this
#      mechanism can fail — a provider error, a malformed payload, a model
#      that ignores the field — fails by finishing.
#   3. It stops making progress. NOT a turn limit: the user asked for none and
#      a fixed cap is exactly the thing that stopped the assistant mid-edit
#      when the ceiling was three rounds. What is bounded is repetition —
#      consecutive iterations that commit nothing durable and name the same
#      next step. A loop with no stall check is a loop that bills for spinning,
#      and the engine's own scar here is a mechanism that fired zero times in
#      181 eligible beats without anybody noticing.
#
# The stall is REPORTED, never silent. A run that stopped because it stopped
# getting anywhere reads identically to a run that finished, unless it says so.

import turnrun

# Consecutive no-progress iterations tolerated before the run is stopped.
#
# Two, not one: an iteration that only reads — an outline, an expand, a
# directory walk — commits nothing durable and is often the correct move
# straight after a surprise. One such iteration is thinking; two in a row
# naming the same next step is a loop.
STALL_LIMIT = 2

# Trace keys that mean the iteration left something behind. Read from the
# trace rather than counted here, because the pipeline already knows what it
# wrote and a second tally kept in this file would drift from it.
_PROGRESS_KEYS = ("experiments", "edits", "research", "proposed_fix",
                  "subagents", "retired", "disputed", "closed_threads")


# WHAT "STOP" MEANS. Not "drop it" — compile it.
#
# The alternative to a report is not a shorter run; it is a run whose findings
# were never written down. This run had measured a reword rate, separated 59
# artefacts from 103 gap slots and corrected its own denominator twice, none of
# which was in a reply anyone would read. Halting loses that. Asking politely
# gets folded in beside the plan and overruled.
#
# So the closing iteration is an ordinary turn with an unusual instruction: no
# new evidence, compile what exists. The prohibitions name the specific verbs
# rather than saying "no research", because a bare prohibition reads as a
# suggestion of the thing it forbids — and every one of these was observed
# being reached for after a stop was requested.
WIND_DOWN_INSTRUCTION = (
    "STOP RESEARCHING AND REPORT. This is the last iteration of this run: "
    "whatever you ask for next, there will not be another one, so nothing you "
    "leave for later survives.\n\n"
    "Run no experiments, no labs, no subagents, no database queries and no "
    "file edits this turn. Everything you have already established is enough "
    "to write from, and anything that is not established is a known unknown "
    "rather than one more query.\n\n"
    "Compile, in this order:\n"
    "1. What you now know that you did not know when the run started, with "
    "the measurement behind each claim rather than the claim alone.\n"
    "2. What you got WRONG during the run and how you found out — a run that "
    "reports only its surviving conclusions is unreadable as evidence.\n"
    "3. What is unfinished, and for each one the cheapest thing that would "
    "close it, or 'unclosable' with what evidence was destroyed.\n"
    "4. Anything committed to a branch: the sha, and what a reviewer should "
    "look at hardest.\n\n"
    "Quote your numbers with their denominators. A figure whose opportunity "
    "count is missing is the failure this project has the most scars from.")


def made_progress(result):
    """Did this iteration leave anything durable behind?

    Deliberately NOT "did the reply change". A model can produce a different
    paragraph forever; the question is whether the turn moved the record —
    evidence, an edit, a hypothesis, a retirement. `minted` is compared
    against 1 because every turn mints its own episode, so the episode alone
    is the floor rather than a sign of life.
    """
    trace = (result or {}).get("trace") or {}
    if any(trace.get(key) for key in _PROGRESS_KEYS):
        return True
    return int(trace.get("minted") or 0) > 1


def _normalise(text):
    return " ".join(str(text or "").lower().split())


def run_session(text, session_id=None, run=None, run_turn=None,
                stall_limit=STALL_LIMIT):
    """Iterate turns until the assistant stops asking for another.

    Returns the LAST iteration's result, with the loop's own account added:
    `iterations`, `stopped_because`, and `replies` — every reply in order, so
    a client that joined late can still read what happened rather than only
    how it ended.

    `run_turn` is injectable so this can be tested without a pipeline; it
    defaults to the real one. Imported lazily for the same reason `pipeline`
    is not imported at module scope: `pipeline` imports a dozen modules and
    this one is reached from the HTTP layer.
    """
    if run_turn is None:
        import pipeline
        run_turn = pipeline.run_turn
    run = run or _UNOBSERVED

    nxt = str(text or "").strip()
    from_user = True
    carried = ""
    result = None
    replies = []
    iteration = 0
    stalls = 0
    previous_step = ""
    stopped = "finished"
    reported = False

    while nxt:
        iteration += 1
        run.halted()
        run.emit("iteration", n=iteration,
                 instruction=nxt[:400], source="user" if from_user else "self")

        result = run_turn(nxt, session_id, run=run,
                          speaker="user" if from_user else "self",
                          carried_plan=carried)
        session_id = result.get("session_id", session_id)
        replies.append({"n": iteration, "reply": result.get("reply") or "",
                        "from_user": from_user})

        step = str(result.get("continue_work") or "").strip()

        # SPEAKING UP MUST NOT COST THE WORK IN FLIGHT. `_deliberate` drains
        # the same inbox mid-turn, so what reaches here arrived after the last
        # round — during the commit, or while a subagent held the turn.
        #
        # Their words become the next instruction and the plan is CARRIED
        # ALONGSIDE, not discarded. Replacing it meant "also, check the tests"
        # threw away a plan three iterations deep: the user who was paying
        # attention was punished for it, and the way that failure shows up is
        # that they stop steering. Whether the interjection actually cancels
        # the plan is the model's call, made with both in front of it.
        # WINDING DOWN ENDS THE RUN, WHOEVER DISAGREES. Checked before the
        # inbox, because it is not a message and must not be foldable into the
        # plan the way a message deliberately is.
        #
        # `reported` is set on the iteration that has just produced the report,
        # so the report itself gets a whole ordinary turn — same stages, same
        # commit — and the loop ends after it rather than before it.
        if reported:
            stopped = "wound down: reported on request"
            break
        if run.winding_down():
            said = [s for s in run.drain_inbox() if s.strip()]
            nxt = WIND_DOWN_INSTRUCTION + (
                "\n\nWHAT THEY SAID WHEN THEY ASKED:\n" + "\n\n".join(said)
                if said else "")
            from_user, carried, reported = True, step, True
            stalls, previous_step = 0, ""
            continue

        said = [s for s in run.drain_inbox() if s.strip()]
        if said:
            nxt, from_user, carried = "\n\n".join(said), True, step
            # A message is new information by definition, so nothing about the
            # iteration before it can be a repetition.
            stalls, previous_step = 0, ""
            continue

        if not result.get("respond_ok"):
            # Belt and braces: `run_turn` already blanks `continue_work` when
            # the respond stage failed. Restated here because a loop that can
            # be driven by a provider error is a loop that bills all night for
            # a 429, and this is the file where that would happen.
            stopped = "the respond stage failed"
            break
        if not step:
            stopped = "finished"
            break

        if made_progress(result) or _normalise(step) != _normalise(previous_step):
            stalls = 0
        else:
            stalls += 1
            run.emit("stall", n=iteration, count=stalls,
                     step=step[:200], limit=stall_limit)
            if stalls >= stall_limit:
                stopped = (f"stopped after {stalls} iterations that committed "
                           f"nothing and named the same next step")
                break

        previous_step, nxt, from_user, carried = step, step, False, ""

    run.emit("loop", state="stopped", iterations=iteration, why=stopped)
    if result is None:
        return {"reply": "", "warnings": ["nothing to do"], "trace": {},
                "iterations": 0, "stopped_because": "no instruction",
                "replies": [], "session_id": session_id}
    return dict(result, iterations=iteration, stopped_because=stopped,
                replies=replies)


class _Unobserved:
    """A loop nobody is watching. Same null-object reasoning as the
    pipeline's: the checkpoints are the point, and a guard each new one must
    remember to wrap is a guard the next one will omit."""

    def emit(self, stage, **detail):
        pass

    def winding_down(self):
        # A run nobody is watching is a run nobody has asked to wind down.
        return False

    def halted(self):
        return False

    def drain_inbox(self):
        return ()


_UNOBSERVED = _Unobserved()


def start(text, session_id=None):
    """Create a run and drive an automation loop on its worker thread."""
    run = turnrun.create(text, session_id)
    run.auto = True
    turnrun.start(run, lambda r: run_session(text, session_id, run=r))
    return run
