# The one seam memory is read through, and the rules it must never lose.
#
# The engine's audit F1: a mind deciding turn N retrieved memories of how
# turn N turned out, because the cutoff fed only the recency SCORING — which
# ranked those rows highly instead of dropping them. The fix was a required,
# defaultless argument on one seam, so that forgetting the rule is a
# TypeError instead of a leak. These tests assert the seam's behaviour and
# that the invariant arguments stay defaultless.

import inspect

import pytest

import memory


def _mint(turn_idx, content, **kw):
    return memory.add_memory("episodic", "witnessed", 0.6, content,
                             turn_idx=turn_idx,
                             event_key=f"t{turn_idx}:{content[:12]}", **kw)


def test_invariant_arguments_have_no_defaults():
    """Deleting the requiredness of before_turn_idx/include_archived would
    let a new call site forget the rules silently. The engine proved by
    mutation testing that a duplicated filter is a guard nothing can observe
    failing; requiredness is the observable form."""
    sig = inspect.signature(memory.visible_memory_rows)
    for name in ("before_turn_idx", "include_archived"):
        param = sig.parameters[name]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        memory.visible_memory_rows()  # noqa — the TypeError is the guard


def test_turn_cutoff_is_strict_and_null_rows_survive(temp_db):
    """Turn N itself and everything later are excluded; rows with NULL
    turn_idx (imported, no place in play order) are kept — SQL three-valued
    logic would silently drop them from a bare `turn_idx < ?`, which is a
    real bug the engine wrote a comment about so nobody reintroduces it."""
    _mint(1, "the early exchange about apples")
    _mint(5, "the current turn outcome about pears")
    memory.add_memory("semantic", "told", 0.6, "an imported fact about plums",
                      event_key="imported:1")  # turn_idx None
    rows = memory.visible_memory_rows(before_turn_idx=5,
                                      include_archived=True)
    contents = {r["content"] for r in rows}
    assert "the early exchange about apples" in contents
    assert "an imported fact about plums" in contents
    assert "the current turn outcome about pears" not in contents


def test_search_applies_cutoff_before_ranking(temp_db):
    """The outcome of the turn being decided must not be retrievable no
    matter how well it matches the query — the failure mode is a replayed
    turn seeing its own committed outcome."""
    _mint(3, "we discussed the meteor shower over the lake")
    _mint(9, "the meteor shower discussion concluded it was a satellite")
    got = memory.search_memories("meteor shower", k=8, current_turn_idx=9)
    assert got, "the earlier memory should be retrievable"
    assert all(m["turn_idx"] is None or m["turn_idx"] < 9 for m in got)


def test_archived_rows_stay_recallable_but_leave_the_buffer(temp_db):
    """Archiving removes a row from the rolling window, never from recall:
    search defaults include_archived=True, the recent buffer excludes them.
    Collapsing those two policies into one flag-read is how a 'cleanup'
    quietly deletes a life."""
    mid = _mint(1, "an archived memory about the lighthouse keeper")
    memory.qi("UPDATE memories SET archived=1 WHERE id=?", (mid,))
    got = memory.search_memories("lighthouse keeper", k=8,
                                 current_turn_idx=10)
    assert any(m["id"] == mid for m in got)
    buffer = memory.recent_memory_buffer(current_turn_idx=3)
    assert not any(m["id"] == mid for m in buffer)
