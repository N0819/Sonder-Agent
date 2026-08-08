"""The assistant's own source is editable; the checks that govern it are not.

Its source is checked out live, on its own branch, so its edits reach real
code with real tests instead of an inert copy — the copy was the actual
defect, because an edit that cannot be run cannot be verified, and every
question about its own behaviour then cost one of the user's turns.

What must not move with it is the small set of files that ARE the checks: the
reproduce-before-you-fix gate, the workspace confinement, the credential
redaction, the closed-table authorizer, and git's own state. An edit there is
the gated thing editing its own gate.

THIS IS NOT A SECURITY BOUNDARY and the comment in `workspace.py` says so at
length. The tree is Turing-complete; a new module imported from an unlisted
file does whatever it likes. What the list buys is that the gates cannot come
off BY ACCIDENT, and that taking them off deliberately requires an edit that
is conspicuous in a diff a human is already reading. The merge and the restart
are the real check. These tests are about the accident.

The engine clone in the same workspace is deliberately unconstrained: it is
not the code running this process, and a rule that could not tell the two
apart would forbid the work the lane exists for.
"""

from __future__ import annotations

import pytest

import workspace


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "session_root", lambda sid=None: str(tmp_path))
    (tmp_path / workspace.SELF_TREE).mkdir()
    (tmp_path / workspace.SELF_TREE / "coding.py").write_text("gate = True\n")
    (tmp_path / workspace.SELF_TREE / "pipeline.py").write_text("x = 1\n")
    (tmp_path / "Sonder_Engine_working").mkdir()
    (tmp_path / "Sonder_Engine_working" / "coding.py").write_text("y = 2\n")
    return tmp_path


@pytest.mark.parametrize("name", ["workspace.py", "coding.py", "refdb.py",
                                  "config.py", "providers.py"])
def test_the_gate_files_refuse_the_write(ws, name):
    """THE REPRODUCTION, one per governing file."""
    out = workspace.write_file(f"{workspace.SELF_TREE}/{name}", "gate = False")
    assert out["ok"] is False
    assert "refused" in out["error"]


def test_the_refusal_says_what_to_do_instead(ws):
    """A refusal that only forbids gets read as an obstacle to route around.
    This one has to name the route that works.
    """
    out = workspace.write_file(f"{workspace.SELF_TREE}/coding.py", "x")
    assert "diff" in out["error"], out["error"]
    assert "human" in out["error"], out["error"]


def test_deleting_it_is_refused_too(ws):
    """Guarding the write alone leaves the gate removable by deletion — the
    same outcome by a quieter route.
    """
    out = workspace.delete_file(None, f"{workspace.SELF_TREE}/coding.py")
    assert out["ok"] is False
    assert "refused" in out["error"]
    assert (ws / workspace.SELF_TREE / "coding.py").exists()


def test_the_rest_of_its_own_source_is_writable(ws):
    """The capability, not merely the constraint. If this fails the design is
    a refusal wearing a feature's name.
    """
    out = workspace.write_file(f"{workspace.SELF_TREE}/pipeline.py", "x = 2\n")
    assert out["ok"] is True, out
    assert (ws / workspace.SELF_TREE / "pipeline.py").read_text() == "x = 2\n"


def test_the_engine_clone_is_untouched_by_the_rule(ws):
    """Same filename, different tree. The engine is not the code running this
    process and constraining it would forbid the work the lane exists for.
    """
    out = workspace.write_file("Sonder_Engine_working/coding.py", "y = 3\n")
    assert out["ok"] is True, out


def test_git_state_is_closed(ws):
    """The merge is the review step, so what a merge would bring must not be
    editable from inside. A worktree's `.git` is a FILE, not a directory,
    which is why the check is on the name rather than on the type.
    """
    for path in (f"{workspace.SELF_TREE}/.git",
                 f"{workspace.SELF_TREE}/.gitignore",
                 f"{workspace.SELF_TREE}/.git/config"):
        out = workspace.write_file(path, "[core]\n")
        assert out["ok"] is False, path


def test_a_nested_file_of_the_same_name_is_allowed(ws):
    """`tests/test_coding.py` is not the gate, and a rule that matched on
    basename anywhere in the tree would block the tests for the very files it
    protects — which is how a guard starts getting switched off.
    """
    (ws / workspace.SELF_TREE / "tests").mkdir()
    out = workspace.write_file(f"{workspace.SELF_TREE}/tests/coding.py",
                               "def test_x(): pass\n")
    assert out["ok"] is True, out
