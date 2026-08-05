# The scar this file exists for: "mechanisms assumed live were not running at
# all." Each test below pins a mechanism that was measured firing ZERO times
# against the opportunities it had — not one of them raised, warned, or failed
# a test. Reading the code was not enough to find any of them; counting was.

import json

import pytest

import beliefs
import db
import memory
import persona
import pipeline
import prompts
import providers


@pytest.fixture
def stub_model():
    yield
    providers.set_chat_stub(None)


# ---- prompts ----

def test_every_prompt_is_rendered_through_one_seam(tmp_path):
    """CONSOLIDATE_SYSTEM was the one template with no variable, so it was
    passed to the model raw while its siblings went through `.format`. Its
    whole JSON schema block therefore reached the consolidator as literal
    `{{ "summary": ... }}`, a model shown doubled braces mirrors them,
    parse fails, and the entire summary layer dies behind a warning."""
    for template in (prompts.CONSOLIDATE_SYSTEM,):
        assert "{{" not in prompts.render(template)
    rendered = prompts.render(prompts.RESPOND_SYSTEM, persona="X")
    assert "{{" not in rendered and '"reply"' in rendered


# Measured, not chosen: RESPOND_SYSTEM shares 0.9997 of its prefix across two
# personas and RESEARCH_SYSTEM 0.9977, so 0.9 clears both with room while
# still failing anything that moves the variable meaningfully up the prompt.
# The old floor was 0.5, which a mutant with the persona hoisted to the 55%
# mark passed — the guard could not fail for the reason it exists.
_PREFIX_FLOOR = 0.9


def test_the_contract_comes_before_the_variable(tmp_path):
    """Cacheability is a prefix property: a provider caches up to the first
    byte that differs, so a variable near the top costs the whole prompt."""
    for template in (prompts.RESPOND_SYSTEM, prompts.RESEARCH_SYSTEM):
        a = prompts.render(template, persona="alpha")
        b = prompts.render(template, persona="beta-and-longer")
        shared = len(os_common_prefix(a, b))
        assert shared > len(a) * _PREFIX_FLOOR, template[:40]


def test_a_variable_moved_to_the_middle_fails_the_guard():
    """The floor above was 0.5 for as long as the guard existed, so a persona
    hoisted into the middle of the contract scored 0.5486 and PASSED. The
    guard could not fail for the reason it was written, which makes it a
    comment. This is that mutant, held against the floor."""
    # Placed at ~55% deliberately: the mutant has to land BETWEEN the old
    # floor and the new one, or it proves nothing about the loosening. A
    # persona at the exact midpoint scores 0.4994 and the old guard catches
    # it — which is why the hole went unnoticed. This is the one that got
    # through: 0.5493, comfortably over 0.5, nowhere near 0.9.
    template = ("HEAD " * 440) + "{persona}" + (" TAIL" * 360)
    a = prompts.render(template, persona="alpha")
    b = prompts.render(template, persona="beta-and-longer")
    shared = len(os_common_prefix(a, b))
    assert shared < len(a) * _PREFIX_FLOOR


def os_common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return a[:i]
    return a[:min(len(a), len(b))]


# ---- persona ----

def test_the_drive_reaches_the_model(temp_db):
    """persona.py argues at length that the drive is the field whose absence
    costs most and shows up latest — and `persona_prompt` never rendered it.
    Worse than an empty field failing silently: a FULL field that nothing
    reads, so even persona_warnings could not see the problem."""
    sheet = persona.get_persona()
    rendered = persona.persona_prompt(sheet)
    assert sheet["drive"] in rendered
    assert "Assist the user" in prompts.render(prompts.RESPOND_SYSTEM,
                                               persona=rendered)


def test_a_partial_save_cannot_blank_a_field(temp_db):
    """`save_persona` writes EVERY field, coercing a missing one to "". The
    default-merge then tested `if field in stored`, and the field WAS in
    stored — as an empty string. So a partial PUT /api/persona permanently
    blanked the rest of the sheet through the merge meant to protect it."""
    persona.save_persona({"identity": "just this one field"})
    sheet = persona.get_persona()
    assert sheet["identity"] == "just this one field"
    assert sheet["drive"] == persona.DEFAULT_PERSONA["drive"]
    assert sheet["standing_commitments"]


# ---- beliefs ----

def test_a_correction_competes_instead_of_reinforcing(temp_db):
    """"not" is a stopword and the matcher counts overlap, so "prefers dark
    mode" and "does not prefer dark mode" scored 0.667 — over the 0.4
    threshold. A user CORRECTION therefore reinforced the belief it
    contradicted: the blend pulled confidence toward it, the stored claim was
    overwritten with its own negation, and explaining-away never fired once.
    That is contradiction being averaged, which AGENTS.md forbids outright."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "prefers dark mode", "confidence": 0.8}], 1)
    beliefs.apply_belief_updates(state, [
        {"about": "the user", "kind": "preference",
         "claim": "does not prefer dark mode", "confidence": 0.9}], 2)
    claims = [h["claim"]
              for h in state["mind_models"]["the user"]["hypotheses"]]
    assert "prefers dark mode" in claims
    assert "does not prefer dark mode" in claims
    original = next(h for h in state["mind_models"]["the user"]["hypotheses"]
                    if h["claim"] == "prefers dark mode")
    assert original["confidence"] < 0.8, "the competitor must suppress it"


def test_credence_uses_the_same_matcher_as_the_merge(temp_db):
    """The docstring promised one rule; there were two. The merge strips the
    subject's own tokens (`ignore=about`) before comparing, credence did not,
    so naming the subject inside a claim inflated every same-subject pair
    toward a match — and reconcile_inference_confidence ranked an inference
    row on a DIFFERENT belief's credence, which is the exact case
    reconciliation exists to fix."""
    state = {}
    beliefs.apply_belief_updates(state, [
        {"about": "the user Alice", "kind": "preference",
         "claim": "the user Alice prefers rust for tooling",
         "confidence": 0.75}], 1)
    # A claim the MERGE would file as a separate competitor must not be
    # matched by credence either.
    merge_side = beliefs.claim_similarity(
        "the user Alice mentioned rust once",
        "the user Alice prefers rust for tooling", ignore="the user Alice")
    credence = beliefs.belief_credence(
        state, "the user Alice", "the user Alice mentioned rust once", 1)
    if merge_side < beliefs._SIMILARITY_THRESHOLD:
        assert credence is None


# ---- memory ----

def test_archiving_can_actually_fire(temp_db):
    """`cutoff = max(start_turn, end_turn - _ARCHIVE_KEEP_RECENT)` over the
    window's own rows, with a window CONSOLIDATE_EVERY_TURNS (10) wide and a
    keep-recent guard of 12: `end_turn - 12` always landed below `start_turn`,
    the clamp pinned the cutoff to `start_turn`, and nothing in a window was
    ever older than that window's start. Measured: 0 rows archived across 40
    turns and 4 consolidations. Two constants nobody compared."""
    assert memory._ARCHIVE_KEEP_RECENT >= memory.CONSOLIDATE_EVERY_TURNS, (
        "if this stops being true the old clamp would have worked and this "
        "test no longer pins anything")
    for turn in range(1, 40):
        memory.add_memory("episodic", "witnessed", 0.5,
                          f"an ordinary exchange on turn {turn}",
                          turn_idx=turn, event_key=f"t:{turn}")
    assert memory._archive_stale_rows(end_turn=39) > 0
    archived = db.q("SELECT COUNT(*) AS c FROM memories WHERE archived=1",
                    one=True)["c"]
    assert archived > 0


def test_a_high_stakes_row_is_not_archived(temp_db):
    """Archiving reads the HIGHER of salience and effective_importance: a
    memory that turned out to matter is not retired on the strength of how
    ordinary it looked at the time."""
    memory.add_memory("episodic", "witnessed", 0.95, "the deadline moved",
                      turn_idx=1, event_key="keep:1")
    memory.add_memory("commitment", "witnessed", 0.1, "I promised to check",
                      turn_idx=1, event_key="promise:1")
    memory.add_memory("episodic", "witnessed", 0.2, "small talk",
                      turn_idx=1, event_key="drop:1")
    memory._archive_stale_rows(end_turn=40)
    kept = {r["event_key"] for r in
            db.q("SELECT event_key FROM memories WHERE archived=0")}
    assert "keep:1" in kept and "promise:1" in kept
    assert "drop:1" not in kept


def test_consolidation_refuses_to_advance_the_cursor_on_a_bad_shape(temp_db):
    """A consolidator returning VALID JSON with a renamed field sailed
    through: the first-hand summary was written as "", its end_turn_idx — the
    cursor — advanced, and that chapter was gone from the summary layer
    forever. Model output is provisional until deterministic code validates
    it, and this was the one commit path taking the model's word for its own
    output shape."""
    for turn in range(1, 12):
        memory.add_memory("episodic", "witnessed", 0.6,
                          f"we discussed the launch on turn {turn}",
                          turn_idx=turn, event_key=f"c:{turn}")
    with pytest.raises(RuntimeError):
        memory.consolidate_memory(11, lambda payload: {"recap": "wrong key"})
    assert memory.get_memory_summary().get("end_turn_idx", 0) == 0


def test_the_lexical_lane_respects_the_turn_cutoff(temp_db):
    """The FTS table knows nothing about the seam, so this lane took the
    global BM25 top-60. Rows the cutoff had excluded — the replayed-turn
    outcomes the seam exists to hide — occupied rank slots, diluting every
    visible row and, past 60 matches, evicting them from the lane entirely.
    No outcome leaked, but "the cutoff runs BEFORE any ranking" was false
    here, and under cheap-embed this is the primary honest lane."""
    memory.add_memory("episodic", "witnessed", 0.6,
                      "the migration script uses postgres",
                      turn_idx=1, event_key="past:1")
    for n in range(70):
        memory.add_memory("episodic", "witnessed", 0.6,
                          f"future note {n} about the migration script "
                          "and postgres", turn_idx=50 + n,
                          event_key=f"future:{n}")
    visible = {r["id"] for r in memory.visible_memory_rows(
        before_turn_idx=2, include_archived=True)}
    ranked = memory._lexical_ranking("migration script postgres",
                                     visible_ids=visible)
    assert ranked, "the visible row must survive the lane"
    assert set(ranked) <= visible


def test_neighbour_padding_respects_its_own_ceiling(temp_db):
    """`break` only left the INNER loop, so each of the three padded episodes
    could add one more neighbour after the ceiling was reached: k+2 was really
    k+4, and RECALL_LIMIT's "payload ceiling" could emit 68."""
    for turn in range(1, 40):
        memory.add_memory("episodic", "witnessed", 0.6,
                          f"an exchange about topic {turn % 7} on turn {turn}",
                          turn_idx=turn, event_key=f"n:{turn}")
    got = memory.search_memories("topic", k=6, current_turn_idx=100)
    assert len(got) <= 6 + 2


def test_the_recall_floor_never_exceeds_the_callers_budget(temp_db):
    """_RECALL_FLOOR was applied as `max(min(FLOOR, len(ranked)), ...)`,
    ignoring `k` entirely — so the ponder lane, which asks for 4, was handed 6
    and then 8 after padding. A caller that names a budget means it."""
    for turn in range(1, 30):
        memory.add_memory("episodic", "witnessed", 0.6,
                          f"a note about scheduling on turn {turn}",
                          turn_idx=turn, event_key=f"p:{turn}")
    assert len(memory.search_memories("scheduling", k=4,
                                      current_turn_idx=100)) <= 4 + 2


def test_unknown_provenance_is_testimony_not_experience(temp_db):
    """Both maps defaulted to the most-trusted class, so a corrupt string or
    a provenance added to one map and forgotten in the other was classified
    as something the assistant personally experienced — sources laundering
    into experience, the direction the three scopes exist to prevent."""
    assert memory.provenance_scope("something-nobody-defined") == \
        memory.SCOPE_RECEIVED
    projected = memory.project_memory({"provenance": "unheard-of", "id": 1,
                                       "gist": "g", "content": "c",
                                       "turn_idx": 1, "event_key": "x",
                                       "kind": "episodic", "salience": 0.5,
                                       "confidence": 1.0, "importance": None,
                                       "disputed": "", "source_url": "",
                                       "key_phrases": [], "entities": []}, 2)
    assert projected["epistemic_origin"] == "what_i_was_told"


def test_degraded_retrieval_is_announced(temp_db, stub_model):
    """providers.py promises the model stamp "is what lets retrieval count and
    announce the stranding instead of quietly splitting the bank" — and
    nothing anywhere counted or announced it. The mint path warned; the READ
    path, where the damage lands, said nothing, so three of the four ranking
    lanes could score zero for a turn and recall still looked healthy."""
    providers.set_chat_stub(lambda system, user, **kw: '{"reply": "ok"}')
    pipeline.run_turn("first, so there is something in the bank")
    db.qi("UPDATE memories SET embedding_model='some-other-model'")
    result = pipeline.run_turn("second")
    health = result["trace"]["retrieval_health"]
    assert health["vector_incomparable_rows"] > 0
    assert not health["vector_lanes_live"]
    assert any("keyword" in w for w in result["warnings"]), result["warnings"]


def test_importance_lifts_for_a_ref_the_reply_leaned_on(temp_db, stub_model):
    """The comment claimed both belief evidence AND memory_evidence_used were
    load-bearing; only the first was implemented. On a corpus where the model
    grounds in "current" — the ordinary case — importance was lifted zero
    times across 40 turns."""
    state = {"ref": None}

    def stub(system, user, **kw):
        payload = json.loads(user)
        if "memories_chronological" in payload:
            return json.dumps({"summary": "x"})
        refs = [m["memory_ref"]
                for m in payload["memory"].get("recent_exchanges") or []
                if m.get("memory_ref")]
        state["ref"] = refs[0] if refs else None
        return json.dumps({"reply": "noted",
                           "memory_evidence_used": refs[:1]})

    providers.set_chat_stub(stub)
    pipeline.run_turn("the deadline moved to April")
    pipeline.run_turn("what did I say about the deadline?")
    assert state["ref"], "the fixture needs a delivered ref to be meaningful"
    revised = db.q("SELECT COUNT(*) AS c FROM memories "
                   "WHERE importance IS NOT NULL", one=True)["c"]
    assert revised > 0
