# Consolidation: windows, scope separation, support sets, archiving.
#
# The scope separation exists because the engine watched a single melted
# summary hand a character its own inference back as something witnessed a
# few turns later — belief laundering into knowledge inside one mind. The
# window key exists because the singleton design silently overwrote every
# chapter but the last (53 of 67 banks lost their opening turns).

import memory


def _consolidator(payload):
    """Deterministic consolidator: proves the machinery without a model.
    Joins gists per scope, which also makes support derivation exact."""
    rows = payload["memories_chronological"]
    def _join(scopes):
        return " ".join(m["gist"] for m in rows
                        if memory.provenance_scope(m["provenance"]) in scopes)
    return {
        "summary": _join({memory.SCOPE_FIRSTHAND}),
        "received_summary": _join({memory.SCOPE_RECEIVED}),
        "surmise_summary": _join({memory.SCOPE_SURMISE}),
        "key_phrases": [],
        "unresolved_threads": ["the unresolved rollout question"],
    }


def _seed(turn, provenance, content, kind="episodic"):
    memory.add_memory(kind, provenance, 0.6, content, turn_idx=turn,
                      event_key=f"{provenance}:{turn}")


def test_scopes_stay_separate(temp_db):
    """What happened, what was told/read, and what was concluded are three
    ROWS, not three paragraphs of one blob — a separate row is the one form
    a model cannot collapse by dropping a convention."""
    _seed(1, "witnessed", "We debugged the deployment script together today")
    _seed(2, "read", "The changelog says version nine drops python support",
          kind="semantic")
    _seed(3, "inferred", "The user is probably preparing a major upgrade",
          kind="inference")
    memory.consolidate_memory(4, _consolidator)
    first = memory.get_memory_summary(memory.SCOPE_FIRSTHAND)
    received = memory.get_memory_summary(memory.SCOPE_RECEIVED)
    surmise = memory.get_memory_summary(memory.SCOPE_SURMISE)
    assert "debugged" in first["summary"]
    assert "changelog" in received["summary"]
    assert "changelog" not in first["summary"]
    assert "preparing a major upgrade" in surmise["summary"]


def test_windows_accumulate_instead_of_overwriting(temp_db):
    """Two consolidation passes produce two windows. Under the engine's
    pre-v23 singleton the second UPDATE destroyed the first chapter — the
    raw rows survived but the summary layer lost the era, unrecoverably
    until a backfill tool was written. The (scope, end_turn_idx) key is the
    entire fix."""
    _seed(1, "witnessed", "The first chapter about the greenhouse build")
    memory.consolidate_memory(2, _consolidator)
    _seed(12, "witnessed", "The second chapter about the irrigation system")
    memory.consolidate_memory(13, _consolidator)
    rows = memory.q("SELECT * FROM memory_summaries WHERE scope=?",
                    (memory.SCOPE_FIRSTHAND,))
    assert len(rows) == 2
    texts = " | ".join(r["summary"] for r in rows)
    assert "greenhouse" in texts and "irrigation" in texts


def test_cursor_advances_so_rows_consolidate_once(temp_db):
    """The first-hand row's end_turn_idx is the cursor: the next pass sends
    only memories after it. Stalling the cursor re-consolidates the same
    rows forever (and re-bills a model call for them every ten turns)."""
    _seed(1, "witnessed", "An early exchange about the harvest festival")
    memory.consolidate_memory(2, _consolidator)
    seen = []

    def _spy(payload):
        seen.extend(m["gist"] for m in payload["memories_chronological"])
        return _consolidator(payload)

    _seed(12, "witnessed", "A later exchange about winter storage")
    memory.consolidate_memory(13, _spy)
    assert any("winter storage" in g for g in seen)
    assert not any("harvest festival" in g for g in seen)


def test_support_sets_derived_and_empty_support_is_a_finding(temp_db):
    """Per-clause support by content-word overlap, host-side. A clause no
    memory supports gets an EMPTY support_refs — recorded, not erased,
    because 'this sentence generalises or was invented' is the finding the
    audit trail exists to make countable."""
    _seed(1, "witnessed",
          "We spent the afternoon repairing the observatory telescope mount")

    def _con(payload):
        return {"summary": "We repaired the observatory telescope mount. "
                           "Everything in the garden was serene.",
                "received_summary": "", "surmise_summary": "",
                "key_phrases": [], "unresolved_threads": []}

    memory.consolidate_memory(2, _con)
    import json
    row = memory.q("SELECT support FROM memory_summaries WHERE scope=?",
                   (memory.SCOPE_FIRSTHAND,), one=True)
    support = json.loads(row["support"])
    assert len(support) == 2
    grounded = [c for c in support if c["support_refs"]]
    ungrounded = [c for c in support if not c["support_refs"]]
    assert len(grounded) == 1 and "telescope" in grounded[0]["claim"]
    assert len(ungrounded) == 1 and "serene" in ungrounded[0]["claim"]
    assert ungrounded[0]["epistemic_origin"] == ""  # claims the least


def test_archiving_reads_the_higher_of_salience_and_importance(temp_db):
    """A memory that turned out to matter is not retired on the strength of
    how ordinary it looked at the time — which is the entire reason the two
    numbers are separate columns."""
    memory.add_memory("episodic", "witnessed", 0.5,
                      "an ordinary-looking exchange that became load-bearing",
                      turn_idx=1, event_key="lb")
    memory.add_memory("episodic", "witnessed", 0.5,
                      "an ordinary exchange that stayed ordinary",
                      turn_idx=2, event_key="ord")
    memory.raise_importance(["lb"], step=0.9)  # became important
    for t in range(3, 20):
        _seed(t, "witnessed", f"filler exchange number {t}")
    memory.consolidate_memory(20, _consolidator)
    rows = {r["event_key"]: r["archived"]
            for r in memory.q("SELECT event_key, archived FROM memories")}
    assert rows["ord"] == 1
    assert rows["lb"] == 0


def test_commitments_never_archive(temp_db):
    """A promise to the user is governed by being kept, not by
    consolidation — the protected-kinds set is what stops the rolling
    window from quietly retiring it."""
    memory.add_memory("commitment", "remembered", 0.5,
                      "I promised to re-check the visa requirements monthly",
                      turn_idx=1, event_key="promise")
    for t in range(2, 20):
        _seed(t, "witnessed", f"filler exchange number {t}")
    memory.consolidate_memory(20, _consolidator)
    row = memory.q("SELECT archived FROM memories WHERE event_key='promise'",
                   one=True)
    assert row["archived"] == 0


def test_summary_cutoff_respected_on_read(temp_db):
    """A window that closed at or after the deciding turn describes how this
    stretch turned out; get_memory_summary(before_turn_idx=N) must not
    return it. Matters on any replayed turn."""
    _seed(1, "witnessed", "chapter one about the pottery class")
    memory.consolidate_memory(5, _consolidator)
    got = memory.get_memory_summary(before_turn_idx=1)
    assert got["summary"] == ""
    got = memory.get_memory_summary(before_turn_idx=10)
    assert "pottery" in got["summary"]
