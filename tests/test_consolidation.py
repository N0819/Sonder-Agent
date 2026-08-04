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


# ---- A thread that is carried forward is making a fresh claim ----

def test_a_carried_thread_keeps_the_turn_it_was_opened_on(temp_db):
    """THE STALE-NARRATIVE DEFECT. `unresolved_threads` held "the promised
    source-code upload has not arrived; the uploaded-files field is still
    empty" in the same payload as 55 uploaded files, and had for turns. The
    consolidator is told to let resolved detail go and has no way to check;
    nothing re-reads current state.

    Nothing deterministic can resolve a thread — "is this still open" is a
    judgement. What code can do is refuse to let the thread hide its age. The
    stamp has to SURVIVE the merge, or age resets every window and measures
    nothing."""
    memory.save_memory_summary("early", start_turn_idx=1, end_turn_idx=4,
                               unresolved_threads=["the upload has not landed"])
    first = memory.get_memory_summary()
    assert first["unresolved_threads"][0]["since_turn"] == 4

    # The same thread, word for word, carried into a much later window.
    memory.save_memory_summary("later", start_turn_idx=5, end_turn_idx=40,
                               unresolved_threads=["the upload has not landed",
                                                   "a genuinely new question"])
    threads = {t["thread"]: t["since_turn"]
               for t in memory.get_memory_summary()["unresolved_threads"]}
    assert threads["the upload has not landed"] == 4, "the stamp reset"
    assert threads["a genuinely new question"] == 40


def test_the_turn_payload_says_how_long_a_thread_has_been_open(temp_db):
    """A stale thread reads as authoritative right up until you can see its
    age beside it. The assistant is the reader that has to weigh it against
    the rest of the payload, so the age has to reach the payload."""
    _seed(1, "witnessed", "we talked about the upload")
    memory.save_memory_summary("early", start_turn_idx=1, end_turn_idx=3,
                               unresolved_threads=["the upload has not landed"])
    payload, _internal = memory.build_memory_context(40, "what is left open?")
    thread = payload["unresolved_threads"][0]
    assert thread["thread"] == "the upload has not landed"
    assert "37 turns ago" in thread["open_since"]


def test_the_consolidator_is_told_each_thread_s_age(temp_db):
    """It was being asked to drop what was resolved while shown nothing that
    would distinguish resolved from merely old."""
    memory.save_memory_summary("early", start_turn_idx=1, end_turn_idx=2,
                               unresolved_threads=["the old rollout question"])
    for turn in range(3, 9):
        _seed(turn, "witnessed", f"turn {turn} happened")
    seen = {}

    def spy(payload):
        seen.update(payload)
        return _consolidator(payload)

    memory.consolidate_memory(8, spy)
    threads = seen["previous_summary"]["unresolved_threads"]
    assert threads[0]["opened_at_turn"] == 2
    assert threads[0]["turns_open"] == 6


def test_a_legacy_thread_written_as_a_bare_string_still_reads(temp_db):
    """Every thread already in the table is a bare string and always will be.
    A reader that had to ask which spelling it was holding is the guard that
    gets forgotten, so both fold at the door."""
    from db import qi
    qi("INSERT INTO memory_summaries(scope,start_turn_idx,end_turn_idx,summary,"
       "key_phrases,unresolved_threads,support,embedding,embedding_model,"
       "embedding_dim,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
       (memory.SCOPE_FIRSTHAND, 1, 7, "old row", "[]",
        '["a thread from before the stamp"]', "[]", b"", "", 0, 0.0))
    threads = memory.get_memory_summary()["unresolved_threads"]
    assert threads == [{"thread": "a thread from before the stamp",
                        "since_turn": 7}]
