# salience vs importance, and re-reading a memory without rewriting it.
#
# Two different questions: how much did this matter when it happened (set at
# mint, never revised) and how central did it become (revised by
# consequences). The engine's first two versions of the citation reader
# matched nothing at all — importance revised on 9 of 6,460 rows — so the
# discipline these tests pin is cheap to break silently.

import memory


def _mint(content, salience=0.5, key=None):
    return memory.add_memory("episodic", "witnessed", salience, content,
                             turn_idx=1,
                             event_key=key or f"k:{abs(hash(content))}")


def test_effective_importance_null_reads_as_salience(temp_db):
    """NULL means 'never revised' and must read as the salience — defaulting
    the column to a number would freeze every memory at mint value and
    silently kill the fallback (the engine left a comment at the exact line
    where that mistake would be made)."""
    _mint("an ordinary chat about weather", salience=0.61, key="w1")
    row = memory.q("SELECT * FROM memories WHERE event_key='w1'", one=True)
    assert row["importance"] is None
    assert abs(memory.effective_importance(row) - 0.61) < 1e-9


def test_raise_importance_is_asymptotic_and_never_lowers(temp_db):
    """Each consequence closes a fraction of the remaining distance to the
    ceiling, so repetition cannot run away; and the function never lowers,
    because 'this stopped mattering' is not a signal it receives."""
    _mint("the user's server rack overheated", salience=0.5, key="rack")
    memory.raise_importance(["rack"])
    first = memory.q("SELECT importance FROM memories WHERE event_key='rack'",
                     one=True)["importance"]
    assert first > 0.5
    for _ in range(50):
        memory.raise_importance(["rack"])
    final = memory.q("SELECT importance FROM memories WHERE event_key='rack'",
                     one=True)["importance"]
    assert final <= memory._IMPORTANCE_CEILING + 1e-9
    sal = memory.q("SELECT salience FROM memories WHERE event_key='rack'",
                   one=True)["salience"]
    assert sal == 0.5, "salience is never revised"


def test_only_unrevised_lifts_exactly_once_ever(temp_db):
    """The citation signal is downstream of retrieval, so the popularity
    loop is closed structurally: only_unrevised=True lifts a given row once
    for its whole life, no matter how often it is cited afterwards."""
    _mint("the migration plan the team agreed on", key="plan")
    assert memory.raise_importance(["plan"], only_unrevised=True) == 1
    lifted = memory.q("SELECT importance FROM memories WHERE event_key="
                      "'plan'", one=True)["importance"]
    assert memory.raise_importance(["plan"], only_unrevised=True) == 0
    again = memory.q("SELECT importance FROM memories WHERE event_key="
                     "'plan'", one=True)["importance"]
    assert again == lifted


def test_dispute_records_beside_never_over(temp_db):
    """A re-read memory keeps its content, provenance and salience exactly —
    'I read this' stays true; what changed is what it MEANS. Collapsing the
    two either erases the record or hides the correction, and both halves
    must be visible to the mind afterwards."""
    _mint("The docs said the API defaults to UTC timestamps",
          salience=0.7, key="api-docs")
    updated = memory.record_dispute(
        "That page described v1; v2 defaults to local time", turn_idx=9,
        memory_ref="api-docs")
    assert updated
    row = memory.q("SELECT * FROM memories WHERE event_key='api-docs'",
                   one=True)
    assert "defaults to UTC" in row["content"]  # untouched
    assert row["salience"] == 0.7               # untouched
    mem = memory._row_memory(row)
    assert mem["disputed"]["reading"].startswith("That page described v1")
    projected = memory.project_memory(mem, 10)
    assert "i_now_read_this_differently" in projected
    assert projected["details"]  # the original still travels


def test_dispute_raises_importance_further_than_citation(temp_db):
    """Being wrong about something is a bigger fact about it than using it
    once — a dispute moves importance UP by a larger step, so archiving can
    never retire a memory whose meaning turned out to be contested."""
    _mint("a claim later contested", salience=0.5, key="c1")
    _mint("a claim merely cited", salience=0.5, key="c2")
    memory.record_dispute("it meant something else", 5, memory_ref="c1")
    memory.raise_importance(["c2"])
    imp = {r["event_key"]: r["importance"]
           for r in memory.q("SELECT event_key, importance FROM memories")}
    assert imp["c1"] > imp["c2"]


def test_dispute_count_accumulates(temp_db):
    """A memory re-read twice has been genuinely unstable, and that is worth
    being able to see — the count must survive successive re-readings."""
    _mint("the disputed benchmark number", key="bench")
    memory.record_dispute("first re-reading", 3, memory_ref="bench")
    memory.record_dispute("second re-reading", 7, memory_ref="bench")
    mem = memory._row_memory(
        memory.q("SELECT * FROM memories WHERE event_key='bench'", one=True))
    assert mem["disputed"]["count"] == 2
    assert mem["disputed"]["reading"] == "second re-reading"
