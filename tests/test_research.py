# Research: hypotheses, evidence, disputes, grounding, and the automated
# loop. This is the engine's citation/dispute discipline promoted to the
# centre of the product, so these tests are the backbone tier: every rule
# here is one that, silently broken, produces an assistant that sounds
# right instead of being right.

import memory
import research
import tools_web


def _stub_search(results):
    tools_web.set_search_stub(lambda q, n: results)


def _stub_fetch(pages):
    tools_web.set_fetch_stub(
        lambda url: pages.get(url, {"url": url, "title": "", "text": "",
                                    "error": "404"}))


def _scripted(actions):
    """ask_model replacement: plays actions in order, repeats the last."""
    it = list(actions)

    def ask(payload):
        return it.pop(0) if len(it) > 1 else it[0]
    return ask


def test_supporting_evidence_moves_confidence_boundedly(temp_db):
    """One source closes a bounded fraction of the distance to certainty —
    never a jump to 1.0 on a single page. The bounded-update machinery is
    affect.py's in spirit: clamped step, convex blend, repetition required."""
    hyp = research.open_hypothesis("does the framework support streaming",
                                   turn_idx=1)
    assert hyp["confidence"] == 0.3
    research.record_evidence(hyp["id"], url="https://a.example/docs",
                             title="Docs", excerpt="Streaming is supported "
                             "since version four", stance="supports",
                             turn_idx=1)
    conf = research.get_hypothesis(hyp["id"])["confidence"]
    assert 0.3 < conf < 0.7


def test_same_url_cannot_pump_confidence_by_repetition(temp_db):
    """Re-recording the same page for the same hypothesis is one evidence
    row, not corroboration — without the idempotency key, a loop that
    re-fetches its best source manufactures certainty out of repetition
    (the importance popularity-loop, wearing 'corroboration')."""
    hyp = research.open_hypothesis("is the library maintained", 1)
    for _ in range(4):
        research.record_evidence(hyp["id"], url="https://a.example/repo",
                                 title="Repo", excerpt="last commit "
                                 "yesterday", stance="supports", turn_idx=1)
    assert len(research.evidence_for(hyp["id"])) == 1
    conf = research.get_hypothesis(hyp["id"])["confidence"]
    # Exactly one bounded step from 0.3, not four.
    assert conf < 0.56


def test_contradiction_becomes_dispute_not_average(temp_db):
    """Source A says X, source B says not-X: the honest state is 'sources
    disagree', with both sides kept and cited. A confidence drifting to the
    middle reads exactly like 'moderately likely' — a number that silently
    claims a state of knowledge nobody has."""
    hyp = research.open_hypothesis("is the API rate limited", 1)
    research.record_evidence(hyp["id"], url="https://a.example/one",
                             title="", excerpt="rate limits apply",
                             stance="supports", turn_idx=1)
    research.record_evidence(hyp["id"], url="https://b.example/two",
                             title="", excerpt="no rate limits",
                             stance="contradicts", turn_idx=1)
    got = research.get_hypothesis(hyp["id"])
    assert got["status"] == "disputed"
    assert got["dispute"]["supporting"] and got["dispute"]["contradicting"]
    lo, hi = research._DISPUTE_BAND
    assert lo <= got["confidence"] <= hi


def test_evidence_is_also_memory_with_read_provenance(temp_db):
    """An evidence row is minted as a `read`-provenance memory carrying its
    URL, so a source consulted long ago resurfaces through ordinary
    retrieval and can dispute a newer claim. Without this, research findings
    evaporate at the end of the loop that made them."""
    hyp = research.open_hypothesis("what year was sqlite released", 1)
    research.record_evidence(hyp["id"], url="https://sqlite.example/history",
                             title="History", excerpt="SQLite was released "
                             "in the year two thousand", stance="supports",
                             turn_idx=1)
    got = memory.search_memories("when was sqlite released", k=4,
                                 current_turn_idx=10)
    hit = [m for m in got if m["provenance"] == "read"]
    assert hit and hit[0]["source_url"] == "https://sqlite.example/history"


def test_ungrounded_citations_dropped_with_warning():
    """A citation naming a ref the model was never shown is dropped, never
    repaired into something plausible — audit metadata describes the model's
    reasoning, it does not fix it after the fact."""
    grounded, warnings = research.ground_citations(
        ["ev:1", "ev:99", "event:abc"], {"ev:1", "event:abc"})
    assert grounded == ["ev:1", "event:abc"]
    assert any("ev:99" in w for w in warnings)


def test_loop_researches_then_concludes_grounded(temp_db):
    """The automated loop: search → fetch → fetch → conclude, ending early
    once the conclusion grounds and confidence clears the bar. The rounds
    are internal — one user turn, several engine turns."""
    _stub_search([{"title": "Doc A", "url": "https://a.example/1",
                   "snippet": "supports streaming"},
                  {"title": "Doc B", "url": "https://b.example/2",
                   "snippet": "streaming guide"}])
    _stub_fetch({"https://a.example/1": {"url": "https://a.example/1",
                                         "title": "Doc A",
                                         "text": "streaming supported"},
                 "https://b.example/2": {"url": "https://b.example/2",
                                         "title": "Doc B",
                                         "text": "how to stream"}})
    hyp = research.open_hypothesis("does it support streaming", 1)
    ask = _scripted([
        {"action": "search", "query": "framework streaming support"},
        {"action": "fetch", "url": "https://a.example/1",
         "stance": "supports", "excerpt": "streaming supported",
         "statement": "yes, streaming is supported"},
        {"action": "fetch", "url": "https://b.example/2",
         "stance": "supports", "excerpt": "how to stream"},
        {"action": "conclude", "answer": "Yes — streaming is supported.",
         "citations": ["ev:1", "ev:2"]},
    ])
    out = research.research_loop(hyp["id"], ask, turn_idx=1)
    assert out["hedged"] is False
    assert out["citations"] == ["ev:1", "ev:2"]
    assert out["rounds"] == 4
    assert research.get_hypothesis(hyp["id"])["status"] == "answered"


def test_loop_rejects_ungrounded_conclusion_then_hedges_at_budget(temp_db):
    """A conclusion with no surviving citations is prose, not an answer: the
    loop rejects it and keeps working, and at budget exhaustion the ENGINE
    writes the hedge deterministically from the evidence table — the model
    is never asked to sound more finished than the evidence is."""
    _stub_search([])
    hyp = research.open_hypothesis("an unanswerable question", 1)
    ask = _scripted([
        {"action": "conclude", "answer": "Trust me, it is fine.",
         "citations": ["ev:404"]},
    ])
    out = research.research_loop(hyp["id"], ask, turn_idx=1, max_rounds=3)
    assert out["hedged"] is True
    assert "couldn't establish" in out["answer"] \
        or "Sources disagree" in out["answer"]
    assert any("no grounded citations" in w for w in out["warnings"])


def test_loop_ponder_lane_answers_from_own_memory(temp_db):
    """Strategic ponder: before burning web rounds, the loop can ask the
    assistant's own bank — and a conclusion grounded entirely in delivered
    memory refs is accepted even though no web evidence moved the
    hypothesis. Without this exit, a question the assistant already knew
    the answer to burns the whole budget and comes back hedged."""
    memory.add_memory("semantic", "read", 0.7,
                      "Regarding release dates: the tool shipped in March "
                      "twenty twenty-two", turn_idx=1,
                      source_url="https://old.example/notes",
                      event_key="evidence:old1")
    hyp = research.open_hypothesis("when did the tool ship", 5)
    ask_calls = []

    def ask(payload):
        ask_calls.append(payload)
        if len(ask_calls) == 1:
            return {"action": "ponder", "query": "tool shipped release date",
                    "why": "I may have read this already"}
        remembered = payload.get("remembered") or []
        assert remembered, "pondered memories must be delivered next round"
        return {"action": "conclude",
                "answer": "It shipped in March 2022.",
                "citations": [remembered[0]["memory_ref"]]}

    out = research.research_loop(hyp["id"], ask, turn_idx=5)
    assert out["hedged"] is False
    assert out["from_memory"] is True
    assert out["citations"] == ["evidence:old1"]
    assert out["rounds"] == 2


def test_loop_terminates_on_budget(temp_db):
    """'Ponder until satisfied' must terminate: a model that searches
    forever hits max_rounds and gets the deterministic hedge. The budget is
    what makes autonomy safe to grant."""
    _stub_search([{"title": "T", "url": "https://t.example", "snippet": ""}])
    hyp = research.open_hypothesis("a rabbit hole", 1)
    ask = _scripted([{"action": "search", "query": "deeper"}])
    out = research.research_loop(hyp["id"], ask, turn_idx=1, max_rounds=4)
    assert out["hedged"] is True
    assert out["rounds"] == 4


def test_dispute_against_old_evidence_memory(temp_db):
    """The re-read loop closed end to end: new evidence changes what an old
    evidence MEMORY means, and the dispute is recorded beside it while the
    original text survives. This is the engine's rarely-fired machinery in
    its natural habitat — superseded docs are common where staged deceptions
    are rare."""
    memory.add_memory("semantic", "read", 0.7,
                      "The docs said the default port is eight thousand",
                      turn_idx=1, source_url="https://docs.example/old",
                      event_key="evidence:port")
    updated = research.dispute_memory_against_evidence(
        "evidence:port", "That was v1; v2 changed the default port", 9)
    assert updated
    row = memory.q("SELECT * FROM memories WHERE event_key='evidence:port'",
                   one=True)
    assert "eight thousand" in row["content"]
    assert memory._row_memory(row)["disputed"]["reading"].startswith(
        "That was v1")


def test_a_confidence_travels_with_the_evidence_that_moved_it(temp_db):
    """A NUMBER WITHOUT ITS DENOMINATOR IS UNREADABLE. Four open hypotheses
    were shown at 0.545 apiece and read, reasonably, as four questions sharing
    an untouched default. They were not: 0.3 + 0.35 * (1 - 0.3) is 0.545, so
    each had received exactly one supporting row and the identical values were
    the arithmetic working. The one fact that would have settled it — the
    count — was not in the payload, and a mechanism that fired looks exactly
    like one that never ran."""
    hid = research.open_hypothesis("does the cap already bite?", turn_idx=1)["id"]
    assert research.evidence_tally(hid) == {}, "an untouched hypothesis"

    research.record_evidence(hid, url="https://example.com/a", title="a",
                             excerpt="it does", stance="supports", turn_idx=1)
    assert research.evidence_tally(hid) == {"supports": 1}
    moved = research.get_hypothesis(hid)["confidence"]
    assert moved == 0.545, moved       # pins the arithmetic the payload explains

    research.record_evidence(hid, url="https://example.com/b", title="b",
                             excerpt="it does not", stance="contradicts",
                             turn_idx=2)
    assert research.evidence_tally(hid) == {"supports": 1, "contradicts": 1}


def test_the_turn_payload_carries_the_tally(temp_db):
    """The tally is only worth having where the confusion happened, which was
    in the turn payload and not in a helper nobody calls."""
    import json

    import pipeline
    import providers

    hid = research.open_hypothesis("an open question", turn_idx=1)["id"]
    research.record_evidence(hid, url="https://example.com/a", title="a",
                             excerpt="supporting", stance="supports",
                             turn_idx=1)
    seen = {}

    def capture(system, user):
        payload = json.loads(user)
        if "user_message" in payload:
            seen.update(payload)
            return json.dumps({"reply": "ok"})
        return json.dumps({"summary": "", "received_summary": "",
                           "surmise_summary": "", "key_phrases": [],
                           "unresolved_threads": []})

    providers.set_chat_stub(capture)
    try:
        pipeline.run_turn("anything")
    finally:
        providers.set_chat_stub(None)
    entry = seen["open_research"][0]
    assert entry["evidence"] == {"supports": 1}
    assert entry["last_moved_turn"] == 1
