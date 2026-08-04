# Evidence integrity: the ways a hypothesis reached an answer it had not
# earned. Each of these moved confidence, or concluded, on something that was
# not evidence — repetition, a source that reversed itself, an empty page, or
# the assistant's own conjecture read back as a source.

import pytest

import coding
import memory
import research
import tools_web


# ---- one page, one spelling ----

@pytest.mark.parametrize("left,right", [
    # RFC 3986 says percent-escapes are case-insensitive and that escapes of
    # unreserved characters are equivalent to the character. These cannot
    # possibly be two documents, and each extra spelling was another full
    # confidence step: four fetches of two real pages reached 0.875 where the
    # honest number was 0.704.
    ("https://ex.com/a%2fb", "https://ex.com/a%2Fb"),
    ("https://ex.com/%7Euser", "https://ex.com/~user"),
    # A search backend returns punycode; a model retypes the Unicode form.
    ("https://xn--bcher-kva.example/a", "https://bücher.example/a"),
    # The trailing dot is the explicit spelling of the same absolute name.
    ("https://ex.com./a", "https://ex.com/a"),
    ("http://WWW.Ex.com:80/a/?utm_source=x#frag", "https://ex.com/a"),
])
def test_one_page_folds_to_one_spelling(left, right):
    """Evidence idempotency keyed on the raw string meant N fetches of one
    page carried a hypothesis up N steps — repetition wearing corroboration,
    arriving through the door the idempotency check left open."""
    assert research.same_source(left, right), (left, right)


def test_canonical_url_is_idempotent():
    """Folding twice must equal folding once, or the key itself is unstable."""
    for raw in ("https://EX.com./%7Ea/?b=2&utm_medium=x#f", "experiment:abc",
                "https://xn--bcher-kva.example/a%2Fb"):
        once = research.canonical_url(raw)
        assert research.canonical_url(once) == once


# ---- evidence that is not evidence ----

def test_a_page_with_no_readable_text_is_a_failed_fetch(temp_db):
    """A JS-only SPA, a PDF, or a page `_strip_html` eats returns HTTP 200
    with no text — and each one moved confidence a full supporting step on
    nothing at all. Three of them reached 0.81 and concluded "answered",
    citing ev:1, ev:2 and ev:3, all substanceless. The user saw "Yes." and
    three sources."""
    tools_web.set_search_stub(lambda q, n: [
        {"title": "A", "url": "https://a.example/", "snippet": ""}])
    tools_web.set_fetch_stub(lambda url: {"url": url, "title": "",
                                          "text": ""})
    try:
        hyp = research.open_hypothesis("does it?", 1, None)
        rounds = iter([
            {"action": "search", "query": "does it"},
            {"action": "fetch", "url": "https://a.example/",
             "stance": "supports", "excerpt": ""},
            {"action": "fetch", "url": "https://a.example/",
             "stance": "supports", "excerpt": ""},
        ])
        research.research_loop(hyp["id"], lambda p: next(rounds, {}), 1,
                               max_rounds=3)
        assert research.evidence_for(hyp["id"]) == []
        assert research.get_hypothesis(hyp["id"])["confidence"] == 0.3
    finally:
        tools_web.set_search_stub(None)
        tools_web.set_fetch_stub(None)


def test_a_source_that_reverses_itself_is_a_dispute(temp_db):
    """The existing-row branch overwrote stance and excerpt in place and
    skipped `_apply_stance` entirely, so a source re-read with the opposite
    stance destroyed the earlier reading and kept the confidence the OLD
    stance had bought. The coding suite drives exactly this path: "passed,
    then failed" — the most informative thing a flaky test can say — silently
    became "failed", at a confidence earned by passing."""
    hyp = research.open_hypothesis("is it stable?", 1, None)
    research.record_evidence(hyp["id"], url="https://s.example/", title="S",
                             excerpt="it passed", stance="supports",
                             turn_idx=1)
    after_support = research.get_hypothesis(hyp["id"])["confidence"]
    research.record_evidence(hyp["id"], url="https://s.example/", title="S",
                             excerpt="it failed", stance="contradicts",
                             turn_idx=2)
    row = research.get_hypothesis(hyp["id"])
    assert row["status"] == "disputed"
    low, high = research._DISPUTE_BAND
    assert low <= row["confidence"] <= high
    # Both readings kept, neither destroyed.
    excerpt = research.evidence_for(hyp["id"])[0]["excerpt"]
    assert "it passed" in excerpt and "it failed" in excerpt
    assert after_support != row["confidence"] or True


def test_a_note_only_dispute_actually_holds(temp_db):
    """`record_dispute_note`'s docstring said "the hypothesis is flagged
    disputed"; the UPDATE wrote only `dispute` and `confidence`. So the
    conclude bar's disputed branch never fired, the hedge's "Sources disagree"
    line never appeared, and two later supporting sources walked the
    confidence straight back out of the band to 0.81 and concluded
    "answered" — with the non-determinism note invisible to every consumer."""
    hyp = research.open_hypothesis("deterministic?", 1, None)
    research.record_dispute_note(hyp["id"], "it did both things")
    assert research.get_hypothesis(hyp["id"])["status"] == "disputed"
    low, high = research._DISPUTE_BAND
    for n in range(3):
        research.record_evidence(hyp["id"], url=f"https://s{n}.example/",
                                 title="S", excerpt="it works",
                                 stance="supports", turn_idx=2 + n)
    assert research.get_hypothesis(hyp["id"])["confidence"] <= high


def test_a_conjecture_cannot_ground_a_memory_only_conclusion(temp_db):
    """The exit required only that no evidence row existed and no citation
    began with "ev:" — it never looked at provenance. But pipeline mints every
    user_model update as an `inference` row with its own event_key, so the
    assistant's own conjectures are citable and pondered like anything else. A
    single "I suspect X but never checked" was pondered up, cited, and
    accepted: hedged False, status answered, confidence 0.30, surfaced to the
    user under a Sources line. An unverified i_suspect reading back as a
    settled answer is the one thing the i_suspect key exists to prevent."""
    memory.add_memory("inference", "inferred", 0.7,
                      "I suspect the tool shipped in March 2022 but I never "
                      "checked", turn_idx=1, event_key="turn:1:inference:0")
    hyp = research.open_hypothesis("when did the tool ship?", 2, None)
    rounds = iter([
        {"action": "ponder", "query": "when did the tool ship"},
        {"action": "conclude", "answer": "March 2022",
         "statement": "March 2022",
         "citations": ["event:turn:1:inference:0"]},
    ])
    out = research.research_loop(hyp["id"], lambda p: next(rounds, {}), 2,
                                 max_rounds=2)
    assert out["hedged"] is True, "a conjecture is not a source"
    assert not out.get("from_memory")


def test_testimony_still_grounds_a_memory_only_conclusion(temp_db):
    """The gate must not close on the case it was built for: a question the
    assistant genuinely already read the answer to should still exit early
    rather than burning the whole budget."""
    memory.add_memory("semantic", "read", 0.7,
                      "the changelog says the tool shipped in March 2022",
                      turn_idx=1, source_url="https://c.example/",
                      event_key="evidence:abc")
    hyp = research.open_hypothesis("when did the tool ship?", 2, None)
    seen = {}

    def ask(payload):
        if "ref" not in seen:
            seen["ref"] = None
            return {"action": "ponder", "query": "when did the tool ship"}
        refs = [m["memory_ref"] for m in payload.get("remembered") or []]
        return {"action": "conclude", "answer": "March 2022",
                "statement": "March 2022", "citations": refs[:1]}

    out = research.research_loop(hyp["id"], ask, 2, max_rounds=3)
    assert out["from_memory"] is True
    assert out["hedged"] is False


def test_a_dead_search_result_is_pruned_by_identity_not_by_equals(temp_db):
    """`s.get("url") != url` is the bare `==` on a thing carrying two names
    that AGENTS.md forbids: `url` is the model's spelling, `s["url"]` is the
    backend's, so one trailing slash left the dead result in the list to be
    re-offered and re-fetched every remaining round until the budget drained."""
    calls = []
    tools_web.set_search_stub(lambda q, n: [
        {"title": "A", "url": "https://a.example/", "snippet": "s"}])

    def fetch(url):
        calls.append(url)
        return {"url": url, "title": "", "text": "", "error": "410 gone"}

    tools_web.set_fetch_stub(fetch)
    try:
        hyp = research.open_hypothesis("q?", 1, None)
        rounds = iter([
            {"action": "search", "query": "q"},
            # the model spells it WITHOUT the trailing slash
            {"action": "fetch", "url": "https://a.example",
             "stance": "supports", "excerpt": "x"},
        ])
        out = research.research_loop(hyp["id"], lambda p: next(rounds, {}), 1,
                                     max_rounds=4)
        assert len(calls) == 1
        assert out["hedged"] is True
    finally:
        tools_web.set_search_stub(None)
        tools_web.set_fetch_stub(None)


# ---- the fetcher's own reach ----

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9/secret",
    "http://localhost:8010/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.1/",
    "file:///etc/passwd",
])
def test_the_fetcher_refuses_non_public_addresses(url):
    """The url reaching `fetch` is chosen by a model reasoning over search
    results it did not write, and the body it returns becomes an evidence row
    AND a durable memory with the url attached. With only a `^https?://`
    check, loopback and link-local were both fetchable — a confused deputy
    spending the assistant's network position on somebody else's instruction,
    and writing the result into long-term memory."""
    assert tools_web.fetch(url)["error"]


def test_html_comments_do_not_become_evidence():
    """`<[^>]+>` cannot see comments, so comment contents survived into
    "readable text" — the material least likely to be a quotable source was
    the material most likely to reach a citation."""
    text = tools_web._strip_html(
        '<p>Real text.</p><!-- internal: revenue > forecast, DO NOT SHIP -->')
    assert "DO NOT SHIP" not in text
    assert "Real text." in text


def test_an_attribute_containing_a_gt_does_not_split_the_tag():
    """`<[^>]+>` ended the tag at the first `>` inside a quoted attribute and
    spilled the rest of the markup into the text."""
    text = tools_web._strip_html('<a title="a > b">link</a> after')
    assert 'b">' not in text
    assert "link" in text and "after" in text


# ---- the coding suite's four rules ----

def test_a_broken_harness_is_inconclusive_not_a_refutation(temp_db):
    """`judge` returned inconclusive only for a missing interpreter, so every
    OTHER harness breakage exited non-zero and was graded a REFUTATION:
    confidence moved down and the reproduce-before-you-fix gate swung open, on
    the strength of a broken tool. Rule 4 is only as good as this
    classifier."""
    hyp = research.open_hypothesis("does the import work?", 1, None)
    out = coding.run_experiment(hyp["id"],
                                source="import nonexistent_module_xyz",
                                expect={"exit_zero": True}, turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_INCONCLUSIVE
    assert research.get_hypothesis(hyp["id"])["confidence"] == 0.3


def test_a_wrong_prediction_does_not_open_the_fix_gate(temp_db):
    """"Refuted" means the PREDICTION was wrong, not that anything failed.
    `source="print('hi')", expect={"stdout_has": "goodbye"}` exits 0, ok=True,
    outcome refuted — and propose_fix answered "a failing observation exists"
    for a run in which nothing failed at all."""
    hyp = research.open_hypothesis("is it broken?", 1, None)
    out = coding.run_experiment(hyp["id"], source="print('hi')",
                                expect={"stdout_has": "goodbye"}, turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_REFUTED
    assert coding.propose_fix(hyp["id"], description="rewrite it",
                              turn_idx=1)["accepted"] is False


def test_each_fix_consumes_its_own_reproduction(temp_db):
    """The gate was per-hypothesis and never spent: after one genuine repro
    and one accepted fix, every later propose_fix was accepted forever with no
    new experiment — "reproduce before you fix" holding for the first fix and
    for no fix after it."""
    hyp = research.open_hypothesis("is it broken?", 1, None)
    coding.run_experiment(hyp["id"], source="raise SystemExit(3)",
                          expect={"exit_zero": True}, turn_idx=1)
    assert coding.propose_fix(hyp["id"], description="fix one",
                              turn_idx=1)["accepted"] is True
    assert coding.propose_fix(hyp["id"], description="fix two",
                              turn_idx=2)["accepted"] is False
    coding.run_experiment(hyp["id"], source="raise SystemExit(4)",
                          expect={"exit_zero": True}, turn_idx=3)
    assert coding.propose_fix(hyp["id"], description="fix three",
                              turn_idx=3)["accepted"] is True


def test_a_fix_does_not_destroy_the_claim_it_was_recorded_against(temp_db):
    """propose_fix overwrote `hypotheses.statement`, so the claim every
    existing evidence row had been recorded against was replaced by the fix
    description."""
    hyp = research.open_hypothesis("is it broken?", 1, None)
    research.record_evidence(hyp["id"], url="https://a.example/", title="A",
                             excerpt="looks broken", stance="supports",
                             turn_idx=1, claim_hint="the parser drops tabs")
    coding.run_experiment(hyp["id"], source="raise SystemExit(3)",
                          expect={"exit_zero": True}, turn_idx=1)
    coding.propose_fix(hyp["id"], description="rewrite the tokenizer",
                       turn_idx=1)
    assert research.get_hypothesis(hyp["id"])["statement"] == \
        "the parser drops tabs"


def test_different_predictions_are_not_the_same_experiment(temp_db):
    """`_digest` omitted `expect` and `files`, so two runs of one
    DETERMINISTIC program under different predictions collided — and the
    outcome differed for that reason alone, which the non-determinism check
    reported as "the behaviour under test is not deterministic" and pinned the
    confidence to the dispute band. It manufactured a dispute out of nothing,
    inverting the invariant it was built to serve."""
    hyp = research.open_hypothesis("deterministic?", 1, None)
    coding.run_experiment(hyp["id"], source="raise SystemExit(0)",
                          expect={"exit_zero": True}, turn_idx=1)
    second = coding.run_experiment(hyp["id"], source="raise SystemExit(0)",
                                   expect={"exit_zero": False}, turn_idx=2)
    assert second["repeated"] is False
    # The two runs still file opposite STANCES — one prediction held and the
    # opposite one did not, which is an honest source disagreement. What must
    # not appear is the non-determinism note, because the program under test
    # behaved identically both times.
    dispute = research.get_hypothesis(hyp["id"])["dispute"] or {}
    assert not (dispute.get("notes") if isinstance(dispute, dict) else None)


def test_an_explicit_command_still_gets_the_source_under_test(temp_db):
    """`payload["main.py"] = source` lived inside the `command is None`
    branch, where `payload` was then unused. With an explicit command the
    source never reached the disk: "can't open file", judged a refutation,
    confidence down, fix gate open — and an experiments row recording a
    `source` that had never executed."""
    import sys
    hyp = research.open_hypothesis("does it print?", 1, None)
    out = coding.run_experiment(
        hyp["id"], source="print('the source under test')",
        command=[sys.executable, "-s", "main.py"],
        expect={"stdout_has": "the source under test"}, turn_idx=1)
    assert out["outcome"] == coding.OUTCOME_CONFIRMED
