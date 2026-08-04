# Retrieval: RRF fusion, rank-normalised importance, MMR, the recall limit.
#
# The constants under test were measured in the engine, not guessed; what
# these tests protect is not the numbers but the STRUCTURE the numbers hang
# from — the failure each one encodes actually happened there.

import memory


def _mint(turn_idx, content, salience=0.6, **kw):
    return memory.add_memory(
        "episodic", "witnessed", salience, content, turn_idx=turn_idx,
        event_key=kw.pop("event_key", f"t{turn_idx}:{abs(hash(content))}"),
        **kw)


def test_lexical_and_exact_rankings_reach_recall(temp_db):
    """With cheap_embed (the offline fallback) the vector rankings are
    lexical noise; BM25 and exact-match are what actually retrieve. A shared
    vocabulary query must find its memory through those channels — this is
    the degraded mode every fresh install runs in, so it gets its own test
    rather than being assumed from the semantic path."""
    _mint(1, "the user's dog is called Biscuit and hates thunderstorms")
    for i in range(2, 8):
        _mint(i, f"an unrelated exchange about spreadsheets number {i}")
    got = memory.search_memories("what is the dog called", k=4,
                                 current_turn_idx=20)
    assert any("Biscuit" in m["content"] for m in got)


def test_rank_normalized_importance_preserves_order_and_respaces():
    """The engine replayed 270 recalls: normalising salience to [0,1] moved
    MORE retrievals than deleting the term (59.6% vs 35.2%), because values
    in a 0.27-wide band gain 3.7x influence while reordering nothing.
    Respacing inside the bank's own p10–p90 keeps ordering exactly and keeps
    the influence budget flat — this asserts both properties, which is what
    stops a future 'fix the compression' from reintroducing the bug."""
    mems = {i: {"importance": None, "salience": s}
            for i, s in enumerate([0.68, 0.70, 0.72, 0.69, 0.71, 0.70])}
    out = memory._rank_normalized_importance(mems)
    # Order preserved exactly.
    original = sorted(mems, key=lambda i: mems[i]["salience"])
    respaced = sorted(mems, key=lambda i: out[i])
    assert [mems[i]["salience"] for i in original] == \
           [mems[i]["salience"] for i in respaced]
    # Influence budget unchanged: values stay inside the bank's own range.
    lo = min(m["salience"] for m in mems.values())
    hi = max(m["salience"] for m in mems.values())
    assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in out.values())
    # Ties share a rank: identical inputs stay identical, no ordering is
    # invented from dict insertion order.
    assert out[1] == out[5]


def test_rank_normalized_importance_flat_bank_stays_flat():
    """A bank with no spread must NOT be handed an ordering by row id — the
    degenerate range collapses to the constant it already was."""
    mems = {i: {"importance": None, "salience": 0.7} for i in range(5)}
    out = memory._rank_normalized_importance(mems)
    assert set(out.values()) == {0.7}


def test_results_return_chronological_not_ranked(temp_db):
    """Every consumer reads results as a narrative; rank order presents a
    history out of sequence. Ranking chooses WHICH rows, never the order
    they are read in."""
    _mint(5, "later conversation about the garden fence")
    _mint(1, "first conversation about the garden gate")
    _mint(3, "middle conversation about the garden path")
    got = memory.search_memories("garden", k=8, current_turn_idx=10)
    idxs = [m["turn_idx"] for m in got if m["turn_idx"] is not None]
    assert idxs == sorted(idxs)


def test_recall_limit_and_neighbor_padding(temp_db):
    """The k+2 ceiling: chronological neighbours of recalled episodes ride
    along (a moment arrives with its context) but never blow the budget —
    the attention budget is real and the payload must stay bounded."""
    for i in range(1, 30):
        _mint(i, f"exchange about topic-{i} with some shared words garden")
    got = memory.search_memories("garden topic", k=8, current_turn_idx=40)
    assert len(got) <= 10  # k + 2


def test_belief_weighting_is_signed_around_half(temp_db):
    """An inference the assistant still holds ranks above the same-relevance
    inference it has since revised. Frozen mint confidence caused the engine
    to preferentially recall ABANDONED beliefs; the signed term is the fix,
    and it reads the confidence column reconciliation maintains."""
    a = memory.add_memory("inference", "inferred", 0.7,
                          "I concluded the user prefers tabs for the editor",
                          gist="prefers tabs", entities=["the user"],
                          turn_idx=1, confidence=0.9, event_key="inf:a")
    b = memory.add_memory("inference", "inferred", 0.7,
                          "I concluded the user prefers spaces for the editor",
                          gist="prefers spaces", entities=["the user"],
                          turn_idx=2, confidence=0.1, event_key="inf:b")
    got = memory.search_memories("does the user prefer tabs or spaces",
                                 k=4, current_turn_idx=10,
                                 chronological=False)
    scores = {m["id"]: m["score"] for m in got}
    assert scores.get(a, 0) > scores.get(b, 0)


def test_access_count_is_written_and_never_ranked(temp_db):
    """access_count is instrumentation. Ranking on it would be a popularity
    loop — recalled memories ranking higher and being recalled more. The
    engine keeps it written-and-unread deliberately; this pins that."""
    mid = _mint(1, "a memory about the annual budget meeting")
    for _ in range(5):
        memory.search_memories("budget meeting", k=4, current_turn_idx=10)
    row = memory.q("SELECT access_count FROM memories WHERE id=?", (mid,),
                   one=True)
    assert row["access_count"] >= 5
    # A pure read must exist for mid-pipeline callers: count_access=False.
    before = row["access_count"]
    memory.search_memories("budget meeting", k=4, current_turn_idx=10,
                           count_access=False)
    after = memory.q("SELECT access_count FROM memories WHERE id=?",
                     (mid,), one=True)["access_count"]
    assert after == before
