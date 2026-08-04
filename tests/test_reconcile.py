# Inference-confidence reconciliation: recall follows belief.
#
# The engine measured its first (compounding) decay rule crushing 76–80% of
# a long story's inference bank to the floor within 7–18 turns — beliefs
# that merely aged out of a bounded working set ranked as though they had
# been concluded WRONG. The fixed-fraction resting value and the
# held>=abandoned floor are the fix, and both are idempotence-critical.

import beliefs
import memory


def _mint_inference(claim, confidence, key, turn_idx=1):
    return memory.add_memory(
        "inference", "inferred", memory.mint_salience(confidence),
        f"I concluded: {claim}", gist=claim, entities=["the user"],
        confidence=confidence, turn_idx=turn_idx, event_key=key)


def test_mint_confidence_reconstructible_from_salience():
    """salience = 0.45 + 0.3*confidence is load-bearing, not a style choice:
    it is how reconciliation recovers the mint confidence without a second
    column — which only works because salience is never revised."""
    for conf in (0.0, 0.3, 0.7, 1.0):
        sal = memory.mint_salience(conf)
        assert abs(memory._mint_confidence_of(sal) - conf) < 1e-6


def test_abandoned_belief_rests_at_fixed_fraction_idempotently(temp_db):
    """No live hypothesis carries the claim → the row rests at a fixed
    fraction of mint confidence. Reconciling again lands on the SAME number:
    a compounding rule here is the measured 76–80% bank-crush."""
    _mint_inference("the user works at a bakery", 0.8, "inf:bakery")
    state = {}  # no mind models at all — everything is 'abandoned'
    memory.reconcile_inference_confidence(state, 10,
                                          beliefs.belief_credence)
    first = memory.q("SELECT confidence FROM memories WHERE event_key="
                     "'inf:bakery'", one=True)["confidence"]
    for turn in (11, 12, 20, 50):
        memory.reconcile_inference_confidence(state, turn,
                                              beliefs.belief_credence)
    final = memory.q("SELECT confidence FROM memories WHERE event_key="
                     "'inf:bakery'", one=True)["confidence"]
    assert final == first, "reconciliation must be idempotent"
    assert final == round(memory._abandoned_confidence(
        memory.mint_salience(0.8)), 4)


def test_held_belief_never_ranks_below_abandoned(temp_db):
    """Half-life decay on a SURVIVING hypothesis measures staleness, not
    disbelief — so a still-stored belief's credence is floored at the
    abandoned resting value. Held >= abandoned, always; violating this made
    the engine rank a belief a character still held below one they had
    dropped."""
    _mint_inference("the user prefers early meetings", 0.8, "inf:held")
    _mint_inference("the user owns a motorcycle", 0.8, "inf:gone")
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "prefers early meetings", "confidence": 0.8}], 1)
    # Far enough for heavy decay on the surviving hypothesis.
    memory.reconcile_inference_confidence(state, 200,
                                          beliefs.belief_credence)
    held = memory.q("SELECT confidence FROM memories WHERE event_key="
                    "'inf:held'", one=True)["confidence"]
    gone = memory.q("SELECT confidence FROM memories WHERE event_key="
                    "'inf:gone'", one=True)["confidence"]
    assert held >= gone


def test_salience_untouched_by_reconciliation(temp_db):
    """salience records how much the inference mattered when formed (and
    drives consolidation/archiving); confidence records how much it is
    credited now. Reconciliation must move only the second — touching the
    first also destroys the mint-confidence reconstruction."""
    _mint_inference("the user dislikes surprises", 0.6, "inf:sal")
    before = memory.q("SELECT salience FROM memories WHERE event_key="
                      "'inf:sal'", one=True)["salience"]
    memory.reconcile_inference_confidence({}, 30, beliefs.belief_credence)
    after = memory.q("SELECT salience FROM memories WHERE event_key="
                     "'inf:sal'", one=True)["salience"]
    assert before == after
