# persona.py — the assistant's sheet: who it is, what it knows how to do, how
# it works, and what it has standingly committed to. Replaces the engine's
# character sheet; deliberately has NO emotional fields — no drive, no affect,
# no affect, no somatic anything — but ONE drive, for the reason Sonder's own
# CLAUDE.md gives at length: a character with rich traits and an empty drive
# reads as complete and is not, because every motivation then lives in goals,
# and goals are built to be completable and abandonable. A courier walked
# sixteen optimal rooms to his destination and turned away, because nothing
# underneath the spent goals wanted it. A drive survives goal decay, and the
# rule for authoring one is that it must be UNSATISFIABLE — otherwise it is a
# goal wearing the word.
#
# "Assist the user" is exactly that shape. It cannot be completed, it cannot
# decay, and it is what should still be true after every task in the queue is
# done. Everything else here is subordinate to it, which is why it sits at the
# top of the sheet rather than among the preferences.
#
# It also keeps the engine's hardest-won authoring lesson:
# an empty field fails silently. A sheet with rich prose in three fields and
# an empty fourth reads as complete and is not; the failure shows up as wrong
# behaviour much later, looking like a model problem. So persona_warnings
# flags every empty field, and the default sheet ships with every field
# filled.
#
# `preferences` are STABLE dispositions of the assistant itself (how it likes
# to work), not emotions. What it learns about the USER lives in the belief
# store (beliefs.py) — the mind-model machinery is the user model, and it is
# earned through conversation, never authored here.

from db import setting_get, setting_put

DEFAULT_PERSONA = {
    # The one thing that is never finished. Read first in every prompt.
    "drive": (
        "Assist the user. Not 'answer the question asked' — understand what "
        "they are actually trying to do and make that go better, including "
        "when it means saying the question is the wrong one, doing the "
        "unglamorous checking, or volunteering the thing they did not think "
        "to ask for. This is never complete and never traded away."),
    "identity": (
        "A research assistant with real long-term memory. Direct, curious, "
        "unhurried; comfortable saying 'I don't know yet' and then finding "
        "out."),
    "expertise": (
        "Finding, reading and weighing sources; keeping track of what was "
        "decided and why; noticing when new information contradicts "
        "something already believed."),
    "working_style": (
        "Evidence first: claims carry citations to sources actually read, "
        "and a claim that cannot be grounded is dropped or clearly hedged. "
        "Disagreement between sources is reported as disagreement, never "
        "averaged into false confidence."),
    "standing_commitments": [
        "Never present a guess as a settled fact — hypotheses are labelled.",
        "Cite the evidence actually used, or say there is none.",
        "When the user corrects a stored fact, record the correction beside "
        "the original rather than pretending it was never held.",
    ],
    "preferences": [
        "Prefers primary sources over aggregators.",
        "Prefers a short honest answer over a long confident-sounding one.",
    ],
}

# `drive` first: an empty one is the failure mode that costs most and shows
# up latest, so it is checked like every other field and is never optional.
_FIELDS = ("drive", "identity", "expertise", "working_style",
           "standing_commitments", "preferences")


def get_persona():
    stored = setting_get("persona")
    if not isinstance(stored, dict):
        return dict(DEFAULT_PERSONA)
    out = dict(DEFAULT_PERSONA)
    for field in _FIELDS:
        # `field in stored` is not the question — `save_persona` writes EVERY
        # field, coercing a missing one to ""/[]. So a partial PUT /api/persona
        # blanked fields straight through this merge: the key was present, the
        # default was overridden by emptiness, and the sheet the docstring
        # promises ("ships with every field filled") was silently gone. The
        # warning fired, but a warning is the detector, not the defence.
        value = stored.get(field)
        filled = (any(str(v).strip() for v in value)
                  if isinstance(value, list) else str(value or "").strip())
        if field in stored and filled:
            out[field] = value
    return out


def save_persona(sheet):
    clean = {}
    for field in _FIELDS:
        if field in ("standing_commitments", "preferences"):
            value = [str(v).strip() for v in (sheet.get(field) or [])
                     if str(v or "").strip()]
        else:
            value = str(sheet.get(field) or "").strip()
        clean[field] = value
    setting_put("persona", clean)
    return persona_warnings(clean)


def persona_warnings(sheet):
    """Empty fields, named. The engine's import-path warning generalised:
    these parameters do not error at runtime and do not show up in tests —
    they show up as an assistant that behaves wrongly fifty turns later, by
    which time the cause looks like a model problem."""
    warnings = []
    for field in _FIELDS:
        value = sheet.get(field)
        if not value or (isinstance(value, list)
                         and not any(str(v).strip() for v in value)):
            warnings.append(
                f"persona.{field} is empty — an empty field fails silently; "
                "fill it or restore the default")
    return warnings


def persona_prompt(sheet=None):
    sheet = sheet or get_persona()
    commitments = "\n".join(f"- {c}"
                            for c in sheet.get("standing_commitments") or [])
    prefs = "\n".join(f"- {p}" for p in sheet.get("preferences") or [])
    return (
        # The drive goes FIRST, and it goes in at all. This module's own
        # header says the drive is "read first in every prompt" and argues at
        # length that it is the field whose absence costs most — and it was
        # the one field this function never rendered. The sheet was authored,
        # the warning stayed silent because the field was full, and the
        # motivation underneath every goal simply never reached the model.
        # An empty field fails silently; a field nothing reads fails harder,
        # because even the emptiness check cannot see it.
        f"YOUR DRIVE\n{sheet.get('drive', '')}\n\n"
        f"WHO YOU ARE\n{sheet.get('identity', '')}\n\n"
        f"WHAT YOU ARE GOOD AT\n{sheet.get('expertise', '')}\n\n"
        f"HOW YOU WORK\n{sheet.get('working_style', '')}\n\n"
        f"STANDING COMMITMENTS\n{commitments}\n\n"
        f"STABLE PREFERENCES\n{prefs}")
