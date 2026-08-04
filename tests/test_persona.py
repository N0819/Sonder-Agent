# Persona: the sheet that replaced the character card, and the one authoring
# lesson that came with it — an empty field fails silently. The engine's
# measured case: a sheet with rich traits and an empty drive read as
# complete, and the character stopped wanting things fifty beats later with
# the cause looking like a model problem. The warning is the only defence,
# so it must fire everywhere a sheet is written, not just on one path.

import persona


def test_default_persona_has_no_empty_fields(temp_db):
    """The shipped default must never demonstrate the failure the warnings
    exist to catch."""
    assert persona.persona_warnings(persona.get_persona()) == []


def test_empty_field_warns_on_save(temp_db):
    warnings = persona.save_persona({
        "identity": "a helpful assistant",
        "expertise": "",  # the silent failure
        "working_style": "evidence first",
        "standing_commitments": ["cite sources"],
        "preferences": ["primary sources"],
    })
    assert any("expertise" in w for w in warnings)


def test_saved_persona_round_trips_and_merges_defaults(temp_db):
    """A partial sheet keeps defaults for fields it does not name — an
    author editing one field must not silently blank four others."""
    persona.save_persona({**persona.DEFAULT_PERSONA,
                          "identity": "a terse research gremlin"})
    sheet = persona.get_persona()
    assert sheet["identity"] == "a terse research gremlin"
    assert sheet["expertise"] == persona.DEFAULT_PERSONA["expertise"]


def test_persona_prompt_contains_every_field(temp_db):
    """The prompt is the only place the sheet reaches the model; a field the
    prompt omits is a field that silently does nothing, which is the same
    failure as an empty field with extra steps.

    This test used to check a HAND-WRITTEN list of five headings — and the
    list omitted the drive, which was the one field `persona_prompt` did not
    render. A test that names the failure and then enumerates the cases by
    hand will forget the same case the code forgot. So it reads `_FIELDS`,
    which is the same list the warning path reads: adding a field to the
    sheet and not to the prompt now fails here."""
    text = persona.persona_prompt()
    for field in persona._FIELDS:
        value = persona.DEFAULT_PERSONA[field]
        needle = value[0] if isinstance(value, list) else value
        assert needle in text, f"persona.{field} never reaches the model"
