# Belief revision: blend, explain away, decay, prune, and the calibration
# guards. The machinery is the engine's theory_of_mind.py; here it IS the
# user model, so a defect stops being a fiction bug and becomes the
# assistant confidently misremembering its user.

import beliefs


def test_reinforcement_blends_rather_than_jumps():
    """A restated belief moves toward the new evidence by the kind's
    plasticity — never a max() jump. The old exact-text max()-only
    accumulation meant one enthusiastic restatement set the ceiling
    forever."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "prefers short answers", "confidence": 0.5}], 1)
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "prefers short answers", "confidence": 0.9}], 2)
    hyp = state["mind_models"]["the user"]["hypotheses"][0]
    assert 0.5 < hyp["confidence"] < 0.9  # moved toward, not to, the evidence


def test_competing_claim_suppresses_but_never_erases():
    """Explaining away is partial: real revision needs repeated
    disconfirmation, not one data point. The displaced belief survives with
    reduced confidence — erasure would make one noisy observation delete a
    standing model of the user."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "goal",
         "claim": "wants to migrate the service to Rust",
         "confidence": 0.6}], 1)
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "goal",
         "claim": "wants to abandon fixing performance and rewrite the "
                  "frontend entirely", "confidence": 0.7}], 2)
    hyps = state["mind_models"]["the user"]["hypotheses"]
    assert len(hyps) == 2
    rust = next(h for h in hyps if "Rust" in h["claim"])
    assert 0 < rust["confidence"] < 0.6


def test_effective_kind_takes_the_stricter_ceiling():
    """A model cannot buy confidence by mislabelling: 'the user always cuts
    corners' declared as observation (cap 1.0) is capped as trait (0.45),
    because the claim's own language votes and the stricter reading wins. A
    misfire can only make the assistant LESS sure — under-confidence
    degrades, over-confidence launders a guess into a fact."""
    assert beliefs.effective_kind("observation",
                                  "the user always cuts corners") == "trait"
    # And a wrong inference never RAISES the ceiling.
    assert beliefs.effective_kind("identity",
                                  "wants to ship on Friday") == "identity"


def test_unreinforced_beliefs_decay_and_prune():
    """Ebbinghaus decay per kind, and subjects whose every hypothesis fell
    through the floor leave storage — this is what keeps years of history
    from accumulating unbounded tracked subjects."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "some passing topic", "kind": "observation",
         "claim": "was mentioned once in passing", "confidence": 0.3}], 1)
    # observation half-life is 5 turns; 60 turns later it is dust.
    beliefs.apply_belief_updates(state, [], 61)
    assert "some passing topic" not in state.get("mind_models", {})


def test_preferences_outlive_observations():
    """The assistant-specific kind: a preference unmentioned for fifty turns
    is usually still held (half-life 300), while an observation from fifty
    turns ago is stale (half-life 5). Forgetting a preference means
    re-asking a question the user already answered — the exact failure a
    long-memory assistant exists to avoid."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "prefers metric units", "confidence": 0.8},
        {"about": "the user", "kind": "observation",
         "claim": "was debugging a css layout", "confidence": 0.8}], 1)
    pref = beliefs.belief_credence(state, "the user",
                                   "prefers metric units", 51)
    obs = beliefs.belief_credence(state, "the user",
                                  "was debugging a css layout", 51)
    assert pref is not None and pref > 0.6
    assert obs is None or obs < 0.1


def test_credence_uses_the_same_matcher_as_the_merge():
    """A terse restatement ('prefers short answers') must resolve to the
    stored longer claim — one similarity rule for both merge and lookup, or
    a memory and a hypothesis drift into being judged by two rules."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "prefers short answers over long thorough ones",
         "confidence": 0.7}], 1)
    got = beliefs.belief_credence(state, "the user",
                                  "prefers short answers", 2)
    assert got is not None and got > 0.5


def test_sheet_hysteresis_keeps_incumbents():
    """Without the incumbent margin the sheet churns every turn as
    confidences wobble and stops being 'what I am actively wondering
    about' — which is the whole point of it being stable."""
    state = {}
    updates = [{"about": f"topic {i}", "kind": "goal",
                "claim": f"question number {i} still open",
                "confidence": 0.5} for i in range(8)]
    beliefs.apply_belief_updates(state, updates, 1)
    entries, keys = beliefs.select_active_hypotheses(
        state["mind_models"], [], 1, capacity=3)
    # A challenger inside the margin must not displace an incumbent.
    challenger = [{"about": "topic 7", "kind": "goal",
                   "claim": "question number 7 still open",
                   "confidence": 0.54}]
    beliefs.apply_belief_updates(state, challenger, 2)
    entries2, _ = beliefs.select_active_hypotheses(
        state["mind_models"], keys, 2, capacity=3)
    assert {e["about"] for e in entries} == {e["about"] for e in entries2}


def test_sheet_entries_carry_epistemic_status():
    """The payload key is `i_suspect`: the field itself says this is a
    conjecture, so the assistant cannot read its own guess back as settled
    fact — the information-layer collapse the engine polices between minds,
    prevented inside one."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "goal",
         "claim": "is preparing a conference talk", "confidence": 0.5}], 1)
    entries, _ = beliefs.select_active_hypotheses(state["mind_models"], [], 1)
    assert entries and "i_suspect" in entries[0]
    assert "claim" not in entries[0]
