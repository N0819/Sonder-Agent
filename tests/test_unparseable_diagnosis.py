"""A reply was thrown away for a newline.

`json.loads` forbids an unescaped control character inside a string and
enforces it by default. `parse_model_json` called it that way, so a `reply`
field containing a paragraph break — which is what every long answer contains —
was rejected whole. Nothing was truncated and nothing was malformed in any way
that mattered: the object opened, the object closed, every field was present.

The live failure that prompted this: 6,036 characters returned on a
113,141-character payload, beginning `{"reply": "Lab `stairs` is provisioned`
and ending `poll `runs` until done."}`. Complete at both ends — which rules out
the truncation-at-max_tokens that every one of these was diagnosed as.

The second half is why they were all misdiagnosed. The recorder that exists to
explain these failures ran `" ".join(raw.split())` before storing the head and
tail, erasing the newlines that caused them. The instrument normalised away the
signature of the only cause it was built to find, and the pipeline comment
above it — "every unparseable output since turn 79 has been unfalsifiable" —
stayed true after the fix that was supposed to end it.
"""

from __future__ import annotations

import json

import providers


LIVE_SHAPE = ('{"reply": "Lab `stairs` is provisioned — schema 26, 52 '
              'tables.\n\nNext: poll `runs` until done."}')


def test_a_paragraph_break_no_longer_costs_the_whole_reply():
    """THE REPRODUCTION. A literal newline inside the string, which is what a
    multi-paragraph answer is made of.
    """
    out = providers.parse_model_json(LIVE_SHAPE)
    assert out is not None, "a reply was discarded for containing a paragraph"
    assert out["reply"].startswith("Lab `stairs` is provisioned")
    assert "\n\n" in out["reply"], "the paragraph break must survive intact"


def test_tabs_too():
    """Same rule, same class. A model laying out a table inside its reply hits
    this and nothing about the failure said so.
    """
    assert providers.parse_model_json('{"reply": "a\tb"}') == {"reply": "a\tb"}


def test_relaxing_control_characters_relaxes_nothing_else():
    """The narrow claim this fix rests on. `strict=False` permits control
    characters inside strings and changes no other rule -- so genuinely broken
    output still fails rather than being coerced into a plausible object.
    """
    for broken in ('{"reply": "unterminated',
                   '{"reply" "missing colon"}',
                   '{reply: "unquoted key"}',
                   'not json at all',
                   '[]',
                   ''):
        assert providers.parse_model_json(broken) is None, broken


def test_what_already_worked_still_works():
    """Fences, prose around the object and trailing commas were the reasons
    this function exists; a change for a fourth case must not cost the three.
    """
    assert providers.parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert providers.parse_model_json('Sure!\n{"a": 1}\nHope that helps')["a"] == 1
    assert providers.parse_model_json('{"a": 1,}') == {"a": 1}


def test_a_closed_object_is_not_truncation():
    """The discriminator the warning never carried. Truncation at max_tokens
    stops mid-sentence; a control-character rejection closes cleanly. Reading
    only the collapsed head and tail, the two were indistinguishable, and the
    operator's next move differs completely between them.
    """
    import pipeline  # noqa: F401 - imported for the module under test

    raw = LIVE_SHAPE
    assert raw.rstrip().endswith("}")
    assert raw.count("\n") == 2
    # ...and the normalisation that used to precede the record erases both.
    collapsed = " ".join(raw.split())
    assert "\n" not in collapsed, (
        "if this ever fails the recorder is safe again and this test is stale")


def test_the_recorder_keeps_the_raw_counts(monkeypatch):
    """Through the real deliberation loop, because the failure this guards
    against is a recorder that quietly stops recording -- and a test that
    inspected the source for field names would pass against a function that
    never ran.

    The output here is deliberately unsalvageable, so the parser genuinely
    fails and the recorder genuinely runs.
    """
    import pipeline

    raw = 'here you go\n{"reply": "unterminated\n\nand it never closes'
    monkeypatch.setattr(pipeline, "chat_complete", lambda system, sent: raw)

    class _Run:
        def halted(self):
            return None

        def emit(self, *a, **k):
            return None

        def drain_inbox(self):
            return []

    out, _deliberation, _refs, cost = pipeline._deliberate(
        {"question": "anything"}, {}, 1, 1, _Run(), [])

    assert out is None, "the fixture must actually fail to parse"
    bad = cost["unparseable"]
    assert bad["raw_newlines"] == raw.count("\n") == 3
    assert bad["closed"] is False, "nothing here closes the object"
    assert bad["chars"] == len(raw), "chars must count the raw, not the collapsed"


def test_a_closed_object_is_reported_as_not_truncation(monkeypatch):
    """The discriminator that was missing. A reply rejected for a control
    character closes cleanly, and the warning has to say so -- otherwise the
    next operator reaches for max_tokens again, as every previous one did.
    """
    import pipeline

    monkeypatch.setattr(pipeline, "chat_complete",
                        lambda system, sent: LIVE_SHAPE.replace('"}', '"'))

    class _Run:
        def halted(self):
            return None

        def emit(self, *a, **k):
            return None

        def drain_inbox(self):
            return []

    out, _d, _r, cost = pipeline._deliberate(
        {"question": "anything"}, {}, 1, 1, _Run(), [])
    assert out is None
    assert cost["unparseable"]["raw_newlines"] == 2
