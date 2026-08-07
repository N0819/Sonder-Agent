"""The one editing mode built for large files refused to touch them.

An anchored edit sends old/new pairs instead of the whole file, so the
assistant never has to reproduce the parts it is not changing. That is the
mode for a file too big to hand back — and it read the file through
`MAX_READ_BYTES`, the cap that exists to stop a file being put in front of the
MODEL. Any file over 200,000 characters was refused with "work on it in
chunks", which is incoherent advice here: anchors ARE the way to edit without
reading the file, and there is no chunked form of an anchored edit to fall
back to.

Live: `Sonder_Engine_working/prompts.py`, 286,803 bytes, refused.

The text an anchored edit reads is consumed by `apply_replacements` and thrown
away; it never reaches a prompt. So the context cap was protecting nothing on
this path while blocking the only path it mattered on. `MAX_EDIT_BYTES` is the
separate, much larger bound, and it is about memory rather than context.
"""

from __future__ import annotations

import pytest

import coding
import workspace


@pytest.fixture
def big_file(tmp_path, monkeypatch):
    """Bigger than the read cap, the way the live one was."""
    root = tmp_path / "ws"
    root.mkdir()
    body = ("# line\n" * 40_000) + "TARGET_CLAUSE\n" + ("# tail\n" * 5_000)
    (root / "prompts.py").write_text(body)
    monkeypatch.setattr(workspace, "session_root", lambda sid=None: str(root))
    assert len(body) > workspace.MAX_READ_BYTES
    return root


def test_the_two_limits_are_not_the_same_number(big_file):
    """If these are ever merged again the defect returns whole."""
    assert workspace.MAX_EDIT_BYTES > workspace.MAX_READ_BYTES


def test_reading_for_the_model_still_refuses(big_file):
    """The context cap is not weakened. A plain read of a 280 KB file is still
    turned away, because that one WOULD go in front of the model.
    """
    out = workspace.read_file("prompts.py")
    assert out["ok"] is False
    assert "read limit" in out["error"]


def test_an_anchored_edit_reaches_past_it(big_file):
    """THE REPRODUCTION."""
    done = coding.apply_edit(
        "prompts.py", None, turn_idx=1,
        replace=[{"old": "TARGET_CLAUSE", "new": "REPLACED_CLAUSE"}],
        hypothesis_id=None, why="", session_id=None)
    assert done["ok"] is True, done.get("why")
    assert "REPLACED_CLAUSE" in (big_file / "prompts.py").read_text()


def test_a_missing_anchor_still_fails_on_its_own_terms(big_file):
    """Widening the read must not turn a bad anchor into a silent success:
    the refusal has to stay about the anchor, not become about size.
    """
    done = coding.apply_edit(
        "prompts.py", None, turn_idx=1,
        replace=[{"old": "NOT_IN_THE_FILE", "new": "x"}],
        hypothesis_id=None, why="", session_id=None)
    assert done["ok"] is False
    assert "read limit" not in str(done.get("why", "")), done
