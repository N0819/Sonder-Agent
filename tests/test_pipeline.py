# The chat turn end to end, with a scripted model. What these prove: the
# deterministic floor — minting, grounding, belief application, the ponder
# hand-off, the research splice — works with the model replaced by a stub,
# which is the engine's standing claim that no correctness property may
# depend on a model cooperating.

import json

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
