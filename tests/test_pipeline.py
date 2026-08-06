# The chat turn end to end, with a scripted model. What these prove: the
# deterministic floor — minting, grounding, belief application, the ponder
# hand-off, the research splice — works with the model replaced by a stub,
# which is the engine's standing claim that no correctness property may
# depend on a model cooperating.

import json
import os

import beliefs
import memory
import pipeline
import providers
from db import q, state_get


def _stub(respond=None, research_actions=None, consolidate=None):
    """Route by payload shape: respond payloads carry user_message, research
    rounds carry rounds_left, the consolidator carries
    memories_chronological."""
    actions = list(research_actions or [])

    def fn(system, user):
        payload = json.loads(user)
        if "user_message" in payload:
            return json.dumps(respond or {"reply": "ok"})
        if "rounds_left" in payload:
            act = actions.pop(0) if len(actions) > 1 else (
                actions[0] if actions else {"action": "conclude",
                                            "answer": "", "citations": []})
            return json.dumps(act)
        if "memories_chronological" in payload:
            return json.dumps(consolidate or {
                "summary": "", "received_summary": "",
                "surmise_summary": "", "key_phrases": [],
                "unresolved_threads": []})
        raise AssertionError("unrecognised payload shape")
    providers.set_chat_stub(fn)


def test_turn_mints_episode_kept_facts_and_inferences(temp_db):
    """One turn produces: the witnessed episode, a told-row for a fact the
    model marked worth keeping, and an inference row whose salience encodes
    its mint confidence (the reconstructible formula). Every row carries a
    stable event_key so a replayed turn replaces rather than duplicates."""
    _stub(respond={
        "reply": "Noted — congratulations on the new job.",
        "remember": [{"content": "The user starts a new job at Acme in "
                                 "September", "provenance": "told"}],
        "user_model_updates": [
            {"about": "the user", "kind": "stated_fact",
             "claim": "starts a new job at Acme in September",
             "confidence": 0.9, "evidence": ["current"]}],
    })
    out = pipeline.run_turn("I got the job at Acme! I start in September.")
    assert "congratulations" in out["reply"].lower()
    rows = {r["event_key"]: dict(r) for r in q("SELECT * FROM memories")}
    assert "turn:1:episode" in rows
    assert rows["turn:1:episode"]["provenance"] == "witnessed"
    assert "turn:1:kept:0" in rows
    assert rows["turn:1:kept:0"]["provenance"] == "told"
    inf = rows["turn:1:inference:0"]
    assert inf["provenance"] == "inferred"
    assert abs(inf["salience"] - memory.mint_salience(0.9)) < 1e-6
    # The belief store moved too, and the two agree.
    state = state_get("assistant")
    credence = beliefs.belief_credence(
        state, "the user", "starts a new job at Acme in September", 1)
    assert credence is not None and credence > 0.5


def test_ungrounded_user_model_update_is_dropped(temp_db):
    """A claim about the user citing a ref the model was never shown is
    dropped with a warning — the mechanism that keeps the user model earned
    rather than invented. This is the same guard, same failure mode, as the
    engine's belief-update grounding."""
    _stub(respond={
        "reply": "hm",
        "user_model_updates": [
            {"about": "the user", "kind": "trait",
             "claim": "is chronically late", "confidence": 0.9,
             "evidence": ["event:never-delivered"]}],
    })
    out = pipeline.run_turn("hello")
    assert any("no grounded evidence" in w or "ungrounded" in w
               for w in out["warnings"])
    state = state_get("assistant")
    assert beliefs.belief_credence(state, "the user",
                                   "is chronically late", 1) is None
    assert not q("SELECT * FROM memories WHERE kind='inference'")


def test_ponder_hands_off_to_next_turn(temp_db):
    """A ponder set this turn is consumed by the NEXT turn's recall stage as
    a deliberate_recall lane — the engine's one-pending-query contract. The
    query requires a concrete why; results are labelled with their origin so
    remembering-on-purpose is distinguishable from remembering-by-cue."""
    seen_payloads = []

    def fn(system, user):
        payload = json.loads(user)
        if "user_message" in payload:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return json.dumps({
                    "reply": "Let me think about that.",
                    "ponder": {"query": "greenhouse tomato varieties",
                               "why": "the user asked before"}})
            return json.dumps({"reply": "Recalled."})
        return json.dumps({"summary": "", "received_summary": "",
                           "surmise_summary": "", "key_phrases": [],
                           "unresolved_threads": []})
    providers.set_chat_stub(fn)
    memory.add_memory("dialogue", "told", 0.82,
                      "The user grows cherry tomato varieties in the "
                      "greenhouse", turn_idx=0, event_key="seed:tomato")
    pipeline.run_turn("what was I growing again?")
    assert state_get("assistant").get("pending_ponder") == \
        "greenhouse tomato varieties"
    pipeline.run_turn("thanks")
    deliberate = seen_payloads[1]["memory"].get("deliberate_recall")
    assert deliberate is not None
    assert deliberate["query_i_chose"] == "greenhouse tomato varieties"
    got = (deliberate["additional_memories"]
           or deliberate["already_in_normal_recall"])
    assert got, "the pondered memory must reach the payload somewhere"
    # Consumed: a ponder is one query, not a standing tax.
    assert state_get("assistant").get("pending_ponder") == ""


def test_research_request_splices_grounded_answer(temp_db):
    """A research request from the respond stage runs the automated loop and
    the final reply is the loop's grounded answer with its citations — not
    the respond stage's guess. One user turn, several internal rounds."""
    import tools_web
    tools_web.set_search_stub(lambda q_, n: [
        {"title": "Release notes", "url": "https://ex.example/notes",
         "snippet": "version 5 shipped in June"},
        {"title": "Blog", "url": "https://blog.example/v5",
         "snippet": "v5 arrived in June"}])
    tools_web.set_fetch_stub(lambda url: {
        "url": url, "title": "Release notes",
        "text": "version 5 shipped in June"})
    _stub(
        respond={"reply": "I'll look that up.",
                 "research": {"question": "when did version 5 ship"}},
        research_actions=[
            {"action": "search", "query": "version 5 release date"},
            {"action": "fetch", "url": "https://ex.example/notes",
             "stance": "supports", "excerpt": "version 5 shipped in June",
             "statement": "version 5 shipped in June"},
            # A SECOND distinct source: the idempotency rule deliberately
            # refuses to count the same page twice (see test_research), so
            # corroboration requires another URL.
            {"action": "fetch", "url": "https://blog.example/v5",
             "stance": "supports", "excerpt": "v5 arrived in June"},
            {"action": "conclude",
             "answer": "Version 5 shipped in June.",
             "citations": ["ev:1", "ev:2"]},
        ])
    out = pipeline.run_turn("when did version 5 ship?")
    assert "shipped in June" in out["reply"]
    assert "ev:1" in out["reply"]  # citations travel with the answer
    assert out["trace"]["research"]["status"] in ("answered", "disputed")
    # The evidence survived as memory for future turns.
    assert q("SELECT * FROM memories WHERE provenance='read'")


def test_no_model_configured_still_remembers(temp_db):
    """With no chat model at all, the turn does not fabricate a reply — and
    it still records the exchange. Memory is the deterministic floor; the
    model is the optional ceiling."""
    out = pipeline.run_turn("remember that my sister is called Maud")
    assert any("no chat model configured" in w for w in out["warnings"])
    rows = q("SELECT * FROM memories")
    assert len(rows) == 1
    assert "Maud" in rows[0]["content"]


def test_replayed_turn_replaces_not_duplicates(temp_db):
    """Event keys + delete-turn-memories: re-running a turn's commit path
    yields the same rows, not doubled ones. There is no rerun UI yet, which
    is exactly why the property is pinned now — the engine's audit found
    this hole AFTER assuming reruns were hypothetical."""
    _stub(respond={"reply": "ok"})
    out = pipeline.run_turn("hello there")
    n_before = q("SELECT COUNT(*) AS c FROM memories", one=True)["c"]
    # Simulate the commit half re-running for the same turn.
    memory.delete_turn_memories(out["turn_id"])
    exchange = "User said: hello there\nI replied: ok"
    memory.add_memory("episodic", "witnessed", 0.5, exchange,
                      turn_id=out["turn_id"], turn_idx=out["turn_idx"],
                      event_key=f"turn:{out['turn_idx']}:episode")
    n_after = q("SELECT COUNT(*) AS c FROM memories", one=True)["c"]
    assert n_after == n_before


def test_a_failed_respond_stage_says_so_as_a_fact(temp_db):
    """The retry control has to know whether a MODEL composed the reply. The
    client had nothing to go on but warning text, and deciding by searching it
    for "failed" would stop offering retry the day a message was reworded —
    and would stop silently."""
    import providers

    def boom(system, user):
        raise RuntimeError("respond exploded")

    providers.set_chat_stub(boom)
    try:
        out = pipeline.run_turn("hello")
    finally:
        providers.set_chat_stub(None)
    assert out["respond_ok"] is False
    assert any("respond stage failed" in w for w in out["warnings"])


def test_a_successful_turn_does_not_offer_retry(temp_db):
    """The other half: a flag that is always False is not a signal. Pinned
    because the retry bar appearing under every good reply would train the
    user to ignore it, which is how the warnings line stops working."""
    _stub(respond={"reply": "a real reply"})
    assert pipeline.run_turn("hello")["respond_ok"] is True


def test_an_experiment_may_name_a_command_instead_of_writing_a_program(
        temp_db, tmp_path):
    """THE MOST OBVIOUS EXPERIMENT IN THE REPOSITORY WAS THE ONE IT COULD NOT
    RUN. This stage required `source`, so "run the suite that is already
    here" had to invent a program in order to ask a question about programs
    that exist, and the spec was dropped before it executed. Measured on a
    live turn: the assistant landed a fix, queued a baseline `pytest tests/`
    and its two new tests by name, and both were dropped here — so it could
    report an edit and nothing that ran it, and said so.

    `coding.run_experiment` already accepted a blank source; only this gate
    refused, which is why no unit test saw it."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    _stub(respond={"reply": "checking", "experiment": [{
        "hypothesis": "does the shipped helper still return 2?",
        "command": ["python3", "-c", "import sys; print(2); sys.exit(0)"],
        "expect": {"exit_zero": True, "stdout_has": "2"}}]})

    out = pipeline.run_turn("verify it")

    ran = out["trace"].get("experiments") or []
    assert [e["outcome"] for e in ran] == ["confirmed"], (ran, out["warnings"])
    assert not any("dropped an experiment" in w for w in out["warnings"])


def test_a_dropped_experiment_says_which_half_is_missing(temp_db, tmp_path):
    """"no hypothesis or no source" left the author guessing at their own
    mistake, and the guess it invited was the wrong one — the assistant read
    it as a hypothesis problem and kept resupplying the field that was
    already there."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    _stub(respond={"reply": "x", "experiment": [
        {"hypothesis": "", "source": "print(1)", "expect": {"exit_zero": True}},
        {"hypothesis": "a real question", "expect": {"exit_zero": True}}]})

    warns = pipeline.run_turn("go")["warnings"]

    assert "dropped an experiment with no hypothesis" in warns
    assert "dropped an experiment with no source or command" in warns


def test_a_file_the_index_does_not_cover_can_still_be_read(temp_db, tmp_path):
    """Every file verb went through the code index, which skips anything with
    no recognised language — so a `.txt` was listable by `list_dir`, sized,
    named, and unopenable. Five files a run collected out of a sandbox landed
    in exactly that hole: delivered, visible in the listing, and `outline`
    answering "no indexed file matches" for every one. A gap in the map became
    a file that could not be read, and the gap is invisible from the reading
    side."""
    import chunks
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_run_output(0, "ab12cd34", {"counts.txt": "FNF=0\n"})
    chunks.ingest_workspace(0)
    # The premise: the index genuinely does not cover it.
    assert chunks.outline("_runs/ab12cd34/counts.txt", 0).get("chunks", 0) == 0

    _stub(respond={"reply": "ok"}, )
    step, _refs = pipeline._gather(
        {"read_file": "_runs/ab12cd34/counts.txt"}, 1, 0,
        pipeline.turnrun.current() or _NullRun(), [])

    assert step["files_read"][0]["text"] == "FNF=0\n"


class _NullRun:
    def emit(self, *args, **kwargs):
        pass


def test_a_refused_edit_corrects_the_reply_that_claimed_it_landed(temp_db,
                                                                  tmp_path):
    """The reply is composed BEFORE the edit stage runs, so it is written in
    the belief every edit will land. When the reproduce-before-you-fix gate
    turned one away, the refusal reached the trace and the live panel and the
    reply went on saying the change was made — observed on a live turn, a
    reply reporting a section "now marked WITHDRAWN in place, with the run id
    on the row" against a file whose line still read exactly as before.
    Nobody reads a trace to check a sentence."""
    import research
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "notes.md", b"a line that will not change\n")
    hyp = research.open_hypothesis("does anything fail?", turn_idx=1)

    _stub(respond={
        "reply": "The correction is now marked WITHDRAWN in place.",
        "edit_files": [{"path": "notes.md", "contents": "rewritten\n",
                        "hypothesis_id": hyp["id"], "why": "withdraw it"}]})
    out = pipeline.run_turn("withdraw that claim")

    # The gate held: nothing was observed failing, so nothing was written.
    assert workspace.read_file("notes.md", 1)["text"] == (
        "a line that will not change\n")
    assert any("edit refused" in w for w in out["trace"]["warnings"])
    # And the reply itself says so, not only the trace.
    assert "Not applied." in out["reply"], out["reply"]
    assert "notes.md" in out["reply"]
    # The model's own account survives beside it — the disagreement is the
    # finding, so substituting the reply would hide which one was written.
    assert "WITHDRAWN in place" in out["reply"]


def test_a_turn_that_edits_and_runs_says_the_run_came_first(temp_db, tmp_path):
    """Stage 4a runs before 4c, because the reproduce-before-you-fix gate
    reads the experiments table and the reproduction has to precede the repair
    it justifies. So a run written to VERIFY an edit measures the tree as it
    stood before it. A fixture repair that took a suite from 31 failed to 0
    was graded refuted twice in the same turn on exactly that — and a
    refutation moves confidence DOWN through the same arithmetic a real one
    does, so the artefact argues against a correct repair in the record."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "notes.md", b"before\n")

    _stub(respond={
        "reply": "Edited it and verified.",
        "experiment": [{"hypothesis": "does the edit show up?",
                        "source": "print('checked')\n",
                        "expect": {"stdout_has": "checked"}}],
        "edit_files": [{"path": "notes.md", "contents": "after\n",
                        "why": "the user asked"}]})
    out = pipeline.run_turn("change it and check")

    assert workspace.read_file("notes.md", 1)["text"] == "after\n"
    assert "Ordering caveat" in out["reply"], out["reply"]
    assert "BEFORE it" in out["reply"]


def test_a_hypothesis_carries_what_its_evidence_said_not_only_how_much(
        temp_db, tmp_path):
    """A run's output was unreachable to the thing that wrote the run. The
    reply is composed BEFORE the experiment executes, and the next turn was
    handed a counter — `contradicts: 1` — and nothing else. A repair that took
    a suite from 31 failures to 0 was graded refuted on a stage-ordering
    artefact, and its author could not read which of four predictions had
    failed, or by how much, from anything it was given. It declined to guess a
    failure count out of a confidence number and re-ran a 47-second suite to
    recover a measurement it had already made."""
    import research
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    _stub(respond={"reply": "checking",
                   "experiment": [{"hypothesis": "does the marker print?",
                                   "source": "print('MARKER-9713')\n",
                                   "expect": {"stdout_has": "MARKER-9713"}}]})
    pipeline.run_turn("run it")

    hyp = research.list_hypotheses(status="open", limit=5)[0]
    latest = research.latest_evidence(hyp["id"])
    assert latest, "the hypothesis carries no evidence at all"
    assert "MARKER-9713" in latest[0]["excerpt"], latest[0]
    assert latest[0]["stance"] == "supports"
    assert latest[0]["ref"].startswith("ev:")


def test_a_turn_that_only_edits_gets_no_ordering_caveat(temp_db, tmp_path):
    """A caveat appended to every turn is a caveat nobody reads. It has to
    mean that a run and an edit actually shared a turn."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "notes.md", b"before\n")
    _stub(respond={"reply": "Edited it.",
                   "edit_files": [{"path": "notes.md", "contents": "after\n",
                                   "why": "asked"}]})
    out = pipeline.run_turn("change it")
    assert "Ordering caveat" not in out["reply"]


def test_a_reply_whose_edits_all_landed_is_left_alone(temp_db, tmp_path):
    """A correction appended to every turn is a correction nobody reads. It
    has to mean that something was actually turned away."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "notes.md", b"before\n")

    _stub(respond={"reply": "Edited it.",
                   "edit_files": [{"path": "notes.md", "contents": "after\n",
                                   "why": "the user asked"}]})
    out = pipeline.run_turn("change it")

    assert workspace.read_file("notes.md", 1)["text"] == "after\n"
    assert "Not applied." not in out["reply"]


def test_a_database_too_large_for_any_copy_lane_is_still_reachable(temp_db,
                                                                   tmp_path):
    """Every lane into data copies bytes — the sandbox snapshot builds a
    payload, `read_file` returns a file, the index stores bodies. A reference
    database of 1,118,785,536 bytes is 2.1x the workspace ceiling and 550x the
    per-file snapshot limit, so it reaches none of them, and raising the
    limits would write a gigabyte into a fresh temp directory per run. The
    ceilings bind on what a sandbox sees; a fetch verb copies nothing."""
    import sqlite3

    import refdb
    path = tmp_path / "big.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE steps(key TEXT)")
    conn.executemany("INSERT INTO steps VALUES(?)", [("dispute",)] * 3)
    conn.commit()
    conn.close()
    refdb.configure({"engine": str(path)})
    try:
        _stub(respond={"reply": "ok"})
        step, _refs = pipeline._gather(
            {"query_db": {"database": "engine",
                          "sql": "SELECT COUNT(*) AS n FROM steps "
                                 "WHERE key='dispute'"}},
            1, 0, pipeline.turnrun.current() or _NullRun(), [])
        assert step["db_query"]["ok"], step["db_query"]
        assert step["db_query"]["rows"] == [["3"]]
    finally:
        refdb.configure({})


def test_open_research_carries_the_id_the_gates_ask_for(temp_db):
    """`propose_fix` and an anchored `edit_files` both take a numeric
    `hypothesis_id`, and this payload described the open questions in prose
    without ever naming one — so the only way to fill those fields was to
    guess an integer, which is exactly the ritual the reproduce-before-you-fix
    gate exists to prevent. The assistant hit it, declined to guess, and
    reported the gap. A field the engine requires and the payload withholds is
    the engine's defect."""
    import research
    hyp = research.open_hypothesis("does the harness grade a break?", 1)
    _stub(respond={"reply": "noted"})
    captured = {}

    def watch(system, user):
        captured.update(json.loads(user))
        return json.dumps({"reply": "noted"})
    providers.set_chat_stub(watch)

    pipeline.run_turn("what is open?")

    ids = [h["id"] for h in captured["open_research"]]
    assert hyp["id"] in ids


def test_a_message_sent_mid_turn_is_read_by_the_same_turn(temp_db):
    """THE DIFFERENCE BETWEEN A BATCH JOB AND SOMETHING STEERABLE. A
    correction read only after the turn ends is a correction applied to work
    already finished, so the only way to redirect was to halt — throwing away
    everything the turn had established in order to say one sentence.

    Delivered as a separate item rather than merged into the original message:
    the model has to be able to tell "you asked me this" from "you have since
    said this", because the second usually overrides the first."""
    import turnrun
    run = turnrun.create("x", None)
    rounds = []

    def fn(system, user):
        payload = json.loads(user)
        if "user_message" not in payload:
            return json.dumps({"summary": "", "received_summary": "",
                               "surmise_summary": "", "key_phrases": [],
                               "unresolved_threads": []})
        rounds.append(payload.get("what_i_went_and_got") or [])
        if len(rounds) == 1:
            run.say("actually, leave that file alone")
            return json.dumps({"reply": "",
                               "need_more": {"ponder": "what do I know?"}})
        return json.dumps({"reply": "understood"})
    providers.set_chat_stub(fn)

    pipeline.run_turn("refactor everything", run=run)

    assert len(rounds) == 2, "the turn never went back for a second round"
    said = [item for item in rounds[1] if item.get("got") == "the user, mid-turn"]
    assert [s["said"] for s in said] == ["actually, leave that file alone"]
    assert run.drain_inbox() == []          # delivered once, not re-delivered


def test_a_self_driven_iteration_is_not_minted_as_something_the_user_said(
        temp_db):
    """PROVENANCE CANNOT BE REPAIRED DOWNSTREAM. An automation iteration is
    driven by the assistant's own `continue_work`, and the episode row for it
    read "User said: <the assistant's own plan>" — a witnessed memory of words
    the user never uttered, minted once per iteration and recalled later as
    fact about them.

    Nothing else could have caught this: the row is well-formed, richly
    salient, and false. A memory system that can be wrong about WHO SPOKE is
    worse than one that remembers less."""
    _stub(respond={"reply": "ran the suite"})

    pipeline.run_turn("run the suite next", speaker="self")

    rows = q("SELECT content FROM memories WHERE kind='episodic'")
    assert rows, "no episode was minted at all"
    content = rows[-1]["content"]
    assert "User said" not in content, content
    assert "Continuing my own work" in content, content


def test_the_user_still_speaks_as_the_user(temp_db):
    """The other half. A fix that made every episode say "continuing my own
    work" would be the same defect pointed the other way, and the default
    path is the one no test would have covered."""
    _stub(respond={"reply": "hello back"})

    pipeline.run_turn("hello")

    content = q("SELECT content FROM memories WHERE kind='episodic'")[-1]
    assert content["content"].startswith("User said: hello")


def test_an_interjection_arrives_beside_the_plan_not_instead_of_it(temp_db):
    """STEERING MUST NOT COST THE WORK IN FLIGHT. A mid-run message used to
    replace the next step outright, so "also, check the tests" threw away a
    plan three iterations deep — the user who was paying attention was
    punished for it, and the way that failure shows is that they stop
    speaking up.

    The engine takes NO view on which wins: "also check X" and "stop, wrong
    file" arrive through the same channel and only the model can tell them
    apart."""
    captured = {}

    def watch(system, user):
        captured.update(json.loads(user))
        return json.dumps({"reply": "ok"})
    providers.set_chat_stub(watch)

    pipeline.run_turn("also, check the tests", speaker="user",
                      carried_plan="finish rewriting judge()")

    assert captured["user_message"]["text"] == "also, check the tests"
    assert captured["user_message"]["spoken_by"] == "the user"
    assert (captured["work_in_progress"]["you_were_about_to"]
            == "finish rewriting judge()")


def test_no_plan_means_no_work_in_progress_key(temp_db):
    """An empty field fails silently: `you_were_about_to: ""` reads as "you
    were about to do nothing", which is a claim, and a false one on every
    ordinary turn."""
    captured = {}

    def watch(system, user):
        captured.update(json.loads(user))
        return json.dumps({"reply": "ok"})
    providers.set_chat_stub(watch)

    pipeline.run_turn("hello")

    assert "work_in_progress" not in captured


def test_a_turn_records_what_it_cost_and_where(temp_db):
    """NOTHING RECORDED THE PRICE OF A TURN, so every proposal to lower it —
    trim recall, route sections by question type, cache the prefix — was an
    argument about numbers no one had. The system prompt is re-sent on every
    deliberation round and the payload grows as the loop gathers, so the cost
    is a sum over rounds that cannot be reconstructed from a total. Broken
    down by section because a total says the turn was expensive and a section
    says which proposal would have helped."""
    _stub(respond={"reply": "ok"})
    result = pipeline.run_turn("what did I say about the deploy?")

    cost = result["trace"]["payload_cost"]
    assert cost["system_chars"] > 0
    assert len(cost["rounds"]) >= 1
    # The system prompt is paid once PER ROUND, not once per turn — the whole
    # point of the number is that it multiplies.
    assert cost["total_chars"] == (cost["system_chars"] * len(cost["rounds"])
                                   + sum(cost["rounds"]))
    # Recall is the section every proposal is about, so it has to be nameable
    # rather than folded into a total.
    assert "memory" in cost["sections"]
    assert cost["sections"]["user_message"] > 0


def _mint_ref(key, content):
    return memory.add_memory("semantic", "told", 0.7, content,
                             turn_idx=1, event_key=key)


def test_a_citation_recall_missed_is_kept_not_called_invented(temp_db):
    """"I INVENTED THIS" AND "RECALL DID NOT SURFACE IT" NEEDED OPPOSITE
    CORRECTIONS AND GOT THE SAME ONE. The gate compared refs against the
    delivered set alone, so a row still sitting in the bank read exactly like
    a fabrication. Measured over 71 live turns: of 23 distinct dropped
    citations, 15 named rows that existed — experiment results the assistant
    had produced itself 8 to 34 turns earlier. Told its own findings were
    ungrounded, it re-ran the experiments to get them back, minting nine
    hypotheses on one question between turns 38 and 70."""
    _mint_ref("real:but:not:recalled", "a finding from many turns ago")
    warnings = []
    kept = pipeline._ground_evidence_list(
        ["real:but:not:recalled"], set(), warnings, "memory_evidence",
        bank_ok=True)

    assert kept == ["real:but:not:recalled"]
    assert any("recall did not surface it" in w for w in warnings)
    assert not any("dropped" in w for w in warnings)


def test_a_citation_to_nothing_is_still_refused(temp_db):
    """The gate keeps the job it was actually built for. Resolving real rows
    must not become "accept any string the model wrote"."""
    warnings = []
    kept = pipeline._ground_evidence_list(
        ["no:such:row"], set(), warnings, "memory_evidence", bank_ok=True)

    assert kept == []
    assert any("ungrounded" in w and "no such row" in w for w in warnings)


def test_a_memory_ref_written_with_the_event_prefix_still_resolves(temp_db):
    """TWO SPELLINGS OF ONE THING. Memory rows are keyed bare
    (`turn:61:episode`) and evidence rows keyed WITH their prefix, so a model
    writing `event:` in front of a memory ref named a real row under a name
    nothing stores. Six live citations died that way."""
    _mint_ref("turn:61:episode", "what happened on turn 61")
    warnings = []
    kept = pipeline._ground_evidence_list(
        ["event:turn:61:episode"], set(), warnings, "memory_evidence",
        bank_ok=True)

    assert kept == ["turn:61:episode"], "the stored spelling is what is kept"


def test_discarding_a_memory_still_requires_having_been_shown_it(temp_db):
    """CITING IS NOT DISCARDING. Letting a citation resolve from the bank must
    not also let the model retire a row it was never shown — that is a
    destructive act on something it cannot have read."""
    _mint_ref("real:but:not:recalled", "a memory the model was never shown")
    warnings = []
    kept = pipeline._ground_evidence_list(
        ["real:but:not:recalled"], set(), warnings, "retire")

    assert kept == []
    assert any("ungrounded" in w for w in warnings)


def test_a_parse_failure_keeps_the_words_that_explain_it(temp_db, monkeypatch):
    """`parse_model_json(chat_complete(...))` threw the model's actual output
    away at the moment it became interesting, so every "respond stage returned
    unparseable output" since turn 79 has been unfalsifiable — the thing that
    would say whether it was truncation, a fence, a refusal or a provider
    error was already gone. Observed again at turn 117: a 496,743-character
    payload over four rounds, and nothing anywhere recording what came back.

    Both ends, because the two diagnoses live at opposite ones: a refusal or a
    prose preamble shows at the head, truncation at max_tokens shows as a
    sentence that simply stops at the tail."""
    import providers

    truncated = 'Here is the answer: {"reply": "' + "x" * 900
    providers.set_chat_stub(lambda system, user, **kw: truncated)
    try:
        out = pipeline.run_turn("what happened?")
    finally:
        providers.set_chat_stub(None)

    bad = out["trace"]["payload_cost"]["unparseable"]
    assert bad["chars"] == len(truncated)
    assert bad["head"].startswith("Here is the answer:")
    assert bad["tail"].endswith("x")
    assert bad["sent_chars"] > 0
    # And the operator is told without opening the trace, because the next
    # move differs completely between max_tokens and a provider error page.
    assert any("unparseable output:" in w and "chars back on a" in w
               for w in out["warnings"]), out["warnings"]


def test_a_turn_that_parses_records_no_failure_sample(temp_db):
    """The diagnosis must not become a field that is always there — a key
    present on every turn is one nobody reads on the turn that matters."""
    import providers

    providers.set_chat_stub(lambda system, user, **kw: '{"reply": "fine"}')
    try:
        out = pipeline.run_turn("hello")
    finally:
        providers.set_chat_stub(None)
    assert "unparseable" not in out["trace"]["payload_cost"]


def test_a_defect_reproduced_and_fixed_in_one_turn_can_name_its_own_run(
        temp_db, tmp_path):
    """The id of THIS turn's reproduction does not exist when the edit citing
    it is written — `open_hypothesis` mints it at stage 4a, after the model has
    composed its response. The prompt promised otherwise ("run the experiment
    that reproduces the defect in the same turn and it will be there"), and
    four consecutive live turns reproduced a defect, cited the newest id they
    could actually see — the previous turn's — and were refused. `fixes` is how
    that promise is kept; it existed in the pipeline and was documented
    nowhere, so nothing could reach it."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "db.py", b"def conn():\n    return None\n")

    _stub(respond={
        "reply": "Reproduced it, then guarded it.",
        "experiment": [{"hypothesis": "does conn() skip the schema check?",
                        "source": "print('GUARD_PRESENT=False')\n",
                        "expect": {"exit_zero": True,
                                   "stdout_has": "GUARD_PRESENT=False",
                                   "reproduces": True}}],
        "edit_files": [{"path": "db.py",
                        "fixes": "does conn() skip the schema check?",
                        "contents": "def conn():\n    check()\n",
                        "why": "name the cause"}]})
    out = pipeline.run_turn("guard it")

    assert not [w for w in out["trace"]["warnings"] if "edit refused" in w], (
        out["trace"]["warnings"])
    assert workspace.read_file("db.py", 1)["text"] == "def conn():\n    check()\n"
    assert "Not applied." not in out["reply"]


def test_fixes_aims_at_the_run_that_failed_not_the_last_one(temp_db, tmp_path):
    """A turn that reproduces a defect and then runs a baseline would aim the
    gate at the run where nothing failed, and be refused for the reproduction
    it did make. `fixes` names a defect, and the only runs that can answer for
    one are the runs that observed it."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "db.py", b"old\n")

    _stub(respond={
        "reply": "Reproduced, then took a baseline.",
        "experiment": [
            {"hypothesis": "does the defect reproduce?",
             "source": "print('BROKEN=True')\n",
             "expect": {"exit_zero": True, "stdout_has": "BROKEN=True",
                        "reproduces": True}},
            {"hypothesis": "is the harness itself sound?",
             "source": "print('ok')\n",
             "expect": {"exit_zero": True, "stdout_has": "ok"}}],
        "edit_files": [{"path": "db.py", "fixes": "does the defect reproduce?",
                        "contents": "new\n", "why": "repair it"}]})
    out = pipeline.run_turn("fix it")

    assert not [w for w in out["trace"]["warnings"] if "edit refused" in w], (
        out["trace"]["warnings"])
    assert workspace.read_file("db.py", 1)["text"] == "new\n"


def test_the_trace_says_whether_an_edit_was_gated_at_all(temp_db, tmp_path):
    """An edit that names no hypothesis takes the one path the reproduce-
    before-you-fix gate does not check, and in the trace it was indistinguish-
    able from one that satisfied it. `apply_edit` reports which happened and
    its docstring promises the gate "cannot be silently skipped"; the pipeline
    dropped the field that said so. Reviewing a live turn whose edit landed,
    the record could not answer whether the gate had run."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "notes.md", b"before\n")
    workspace.store_upload(1, "db.py", b"old\n")

    _stub(respond={
        "reply": "Both done.",
        "experiment": [{"hypothesis": "does it reproduce?",
                        "source": "print('BROKEN=True')\n",
                        "expect": {"exit_zero": True, "stdout_has": "BROKEN=True",
                                   "reproduces": True}}],
        "edit_files": [
            {"path": "db.py", "fixes": "does it reproduce?",
             "contents": "new\n", "why": "repair"},
            {"path": "notes.md", "contents": "after\n", "why": "asked for it"}]})
    out = pipeline.run_turn("fix it and write the note")

    by_path = {e["path"]: e for e in out["trace"]["edits"]}
    assert by_path["db.py"]["gated_on"] is not None
    assert "failing observation" in by_path["db.py"]["gate"]
    # The unchecked path is named as such rather than looking like the other.
    assert by_path["notes.md"]["gated_on"] is None
    assert "no reproduction required" in by_path["notes.md"]["gate"]


def test_an_experiment_can_attach_to_a_hypothesis_already_open(temp_db,
                                                               tmp_path):
    """Without this, `open_hypothesis` mints a fresh id from each run's own
    text and no two runs ever share one — which is why three mechanisms sat at
    zero fires against zero opportunities. The id is the only lever that says
    two differently-worded probes measure one claim."""
    import research
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    hyp = research.open_hypothesis("does f() return 1?", turn_idx=1)

    _stub(respond={"reply": "second method",
                   "experiment": [{"hypothesis": "measured by exit code",
                                   "hypothesis_id": hyp["id"],
                                   "source": "raise SystemExit(3)\n",
                                   "expect": {"exit_zero": True}}]})
    out = pipeline.run_turn("check it another way")

    assert out["trace"]["experiments"][0]["hypothesis_id"] == hyp["id"]
    assert research.list_hypotheses(status="open", limit=50)
    got = [h["id"] for h in research.list_hypotheses(limit=50)]
    assert got.count(hyp["id"]) == 1 and len(got) == 1, (
        "attaching minted a second hypothesis anyway: %r" % got)


def test_an_experiment_naming_an_unknown_hypothesis_says_so(temp_db, tmp_path):
    """A guessed integer is the ritual the gate discipline exists to prevent,
    and an attached run is where a wrong id is invisible afterwards — a
    misfiled row reads exactly like a correct one. Falling back silently would
    make the guess look honoured."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    _stub(respond={"reply": "x",
                   "experiment": [{"hypothesis": "a fresh question",
                                   "hypothesis_id": 9999,
                                   "source": "print('ok')\n",
                                   "expect": {"stdout_has": "ok"}}]})
    out = pipeline.run_turn("run it")
    assert any("9999" in w and "not among the ids offered" in w
               for w in out["trace"]["warnings"]), out["trace"]["warnings"]
    assert out["trace"]["experiments"][0]["hypothesis_id"] != 9999


def _story_engine(tmp_path):
    """A miniature engine database, registered the way the real one is."""
    import sqlite3
    import refdb
    path = tmp_path / "engine.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE chats(id INTEGER PRIMARY KEY, name TEXT, scenario TEXT,
                           created REAL, branched_from TEXT);
        CREATE TABLE turns(id INTEGER PRIMARY KEY, chat_id INT, idx INT,
                           player_input TEXT, created REAL);
        CREATE TABLE steps(id INTEGER PRIMARY KEY, turn_id INT, key TEXT,
                           label TEXT, ord INT, stale INT DEFAULT 0);
        CREATE TABLE variants(id INTEGER PRIMARY KEY, step_id INT,
                              content TEXT, active INT, reasoning TEXT);
        INSERT INTO chats VALUES(46,'The Blizzard','Snow.',1.0,'[]');
        INSERT INTO turns VALUES(700,46,0,'the player types',1.0);
        INSERT INTO steps VALUES(7000,700,'narrator','Narrator',0,0);
        INSERT INTO variants VALUES(1,7000,'{"prose":"the door blew open"}',1,'');
    """)
    conn.commit()
    conn.close()
    refdb.configure({"engine": str(path)})
    return path


def test_a_story_can_be_reached_by_the_name_a_bug_report_uses(tmp_path):
    """The verb exists because the SQL was never the hard part: a story is
    named by a person and keyed by an integer nobody knows, and every
    investigation used to begin by rediscovering that.
    """
    import refdb
    _story_engine(tmp_path)
    try:
        step, _refs = pipeline._gather(
            {"story": {"verb": "find_story", "name": "Blizzard"}}, 1, 0,
            _NullRun(), [])
        assert step["story"]["matches"][0]["chat_id"] == "46"
        detail, _refs = pipeline._gather(
            {"story": {"verb": "turn_detail", "chat_id": 46, "turn": 0}}, 1, 0,
            _NullRun(), [])
        assert "the door blew open" in detail["story"]["steps"][0]["content"]
    finally:
        refdb.configure({})


def test_an_unknown_verb_names_the_verbs_that_exist(tmp_path):
    """A round spent on a typo costs the same as a round spent on the defect,
    and the round after it goes to a second guess at the spelling.
    """
    import refdb
    _story_engine(tmp_path)
    try:
        step, _refs = pipeline._gather(
            {"story": {"verb": "find_chat", "name": "Blizzard"}}, 1, 0,
            _NullRun(), [])
        assert step["story"]["ok"] is False
        assert "find_story" in step["story"]["error"]
        assert "turn_detail" in step["story"]["error"]
    finally:
        refdb.configure({})


def test_a_verb_that_raises_does_not_take_the_round_with_it(tmp_path):
    """A deliberation round asks for several things at once. A traceback in
    one lane would lose the answers to all the others, and the failure IS the
    answer for the lane that failed.
    """
    step, _refs = pipeline._gather(
        {"engine_lab": {"verb": "seed", "lab": "nope", "story": None}}, 1, 0,
        _NullRun(), [])
    assert step["engine_lab"]["ok"] is False
    assert step["engine_lab"]["error"]


def test_the_labs_on_disk_are_shown_in_the_payload(tmp_path):
    """A LAB SURVIVES THE TURN THAT MADE IT — that is the whole reason it is on
    disk. An assistant that cannot see the one it provisioned last turn will
    provision another, and then attribute the second one's fresh story to the
    first one's edit.
    """
    import enginelab
    enginelab.configure(root=str(tmp_path / "labs"), source="", engine_db="")
    try:
        os.makedirs(str(tmp_path / "labs" / "lamp"), exist_ok=True)
        step, _refs = pipeline._gather(
            {"engine_lab": {"verb": "list"}}, 1, 0, _NullRun(), [])
        assert [entry["name"] for entry in step["engine_lab"]["labs"]] == ["lamp"]
        assert step["engine_lab"]["labs"][0]["provisioned"] is False
    finally:
        enginelab.configure(root="", source="", engine_db="")
