# A closed thread must record WHERE it sat in the delivered list.
#
# WHY THIS FILE EXISTS. `trace.closed_threads` is the only persisted record of
# a thread being answered — 105 instances across 397 turns — and it stored the
# thread's TEXT and nothing else. The question it is wanted for is whether the
# threads that actually get closed concentrate by POSITION in the delivered
# list (which argues for a cap), by position after re-ranking (which argues for
# a re-rank), or by AGE since opening (which argues for expiry). None of the
# three is answerable from a string.
#
# It cannot be recovered afterwards either: `close_threads` UPDATEs the same
# row the payload was assembled from, so the instrument destroys its own
# denominator at the moment it records the numerator. Measured on the live
# bank: `pos` and `open_since` NULL for all 105 historical instances, and 79 of
# the 105 matching no stored thread at any prefix length down to 15 characters.
#
# `of` travels with `idx` because a raw index is not comparable across time.
# The delivered list has averaged 42.5 slots historically and holds 87 now, so
# "index 20" is the bottom quarter in one window and the top quarter in
# another; without the length, any threshold computed on raw idx is computed
# against a moving denominator.
#
# WITHOUT THESE TESTS the return quietly reverts to bare strings and every
# future closure becomes unjoinable again — and the record still looks
# complete, because what it cannot answer is invisible from the record.

import memory


def test_a_closed_thread_records_its_position_age_and_list_length(temp_db):
    """Without this, `closed` is a bare string: the position and the age are
    gone the instant the UPDATE below removes the thread from the row the
    payload was built from, and no later join can recover them."""
    memory.save_memory_summary(
        "early", start_turn_idx=1, end_turn_idx=4,
        unresolved_threads=["the oldest question"])
    memory.save_memory_summary(
        "later", start_turn_idx=5, end_turn_idx=9,
        unresolved_threads=["the oldest question",
                            "a middle question",
                            "the newest question"])
    outcome = memory.close_threads(["a middle question"], 10)
    assert outcome["closed"] == [{"thread": "a middle question", "idx": 1,
                                  "open_since": 9, "of": 3}]


def test_positions_are_read_from_the_delivered_list_not_the_closed_set(temp_db):
    """Without this, `idx` could be enumerated over the threads being closed
    rather than over the list they were delivered in — which reads as a
    position, is comparable across turns, and means nothing at all."""
    memory.save_memory_summary(
        "window", start_turn_idx=1, end_turn_idx=6,
        unresolved_threads=["zero", "one", "two", "three"])
    outcome = memory.close_threads(["three", "one"], 7)
    assert [(c["thread"], c["idx"]) for c in outcome["closed"]] == [
        ("one", 1), ("three", 3)]
    assert {c["of"] for c in outcome["closed"]} == {4}


def test_an_unmatched_thread_is_still_reported_as_a_string(temp_db):
    """`unknown` is the caller's own strings and must stay strings. This also
    guards the crash: the set difference that computes it is `wanted -
    set(closed)`, which raises TypeError: unhashable type: 'dict' the moment
    `closed` stops being strings — inside the commit transaction, on a turn
    that answered a question, which is the worst place to learn it."""
    memory.save_memory_summary("early", start_turn_idx=1, end_turn_idx=4,
                               unresolved_threads=["a live question"])
    outcome = memory.close_threads(["a question nobody asked"], 5)
    assert outcome["closed"] == []
    assert outcome["unknown"] == ["a question nobody asked"]
