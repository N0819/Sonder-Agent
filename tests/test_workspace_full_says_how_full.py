"""'This workspace is full' named no quantity and suggested nothing.

`edit_files` refuses a write once the session workspace exceeds
MAX_WORKSPACE_BYTES, and the refusal is honest — it returns ok=False. What it
did not do was say by how much, or what was taking the room. A reader could
not tell a document 200 bytes too large from a workspace 300 MB over, and had
nothing to delete.

Live, and it cost two turns and three attempted writes. A sync had copied
685 MB of assets — a 220 MB virtualenv, 228 MB of ambience audio, 182 MB of
generated backdrops — into a 512 MiB workspace. Every write was refused with
this string. The assistant read it, could not reconcile "full" with a
workspace it believed held a handful of documents, and ran a file-by-file hunt
to prove its own document was absent. It was absent, and for exactly the
reason this message had already been told and did not pass on.

The assistant's diagnosis was right and arrived the long way round: it found
MAX_WORKSPACE_BYTES by grepping a *copy* of the source and compared it against
a byte count in its own payload. That is three rounds of work to recover a
number the refusal was holding.
"""

from __future__ import annotations

import os

import pytest

import workspace


@pytest.fixture
def full_session(tmp_path, monkeypatch):
    """A workspace over cap, made the way the live one got there: one big
    directory copied in whole.
    """
    root = tmp_path / "ws"
    (root / "Sonder_Engine_working" / "ambience").mkdir(parents=True)
    (root / "Sonder_Engine_working" / "ambience" / "bed.wav").write_bytes(
        b"\0" * 40_000)
    (root / "notes.md").write_text("small")
    monkeypatch.setattr(workspace, "session_root", lambda sid: str(root))
    monkeypatch.setattr(workspace, "MAX_WORKSPACE_BYTES", 50_000)
    return root


def test_the_refusal_says_how_full_and_by_how_much(full_session, monkeypatch):
    """THE REPRODUCTION. Every number in the message is one the caller needed
    and had to go and derive.

    The real numbers, not the fixture's toy ones: the live workspace stood at
    808 MiB against a 512 MiB cap. Sized here so the MiB figures are the ones
    an operator would actually read.
    """
    monkeypatch.setattr(workspace, "MAX_WORKSPACE_BYTES", 512 * 1024 * 1024)
    monkeypatch.setattr(workspace, "workspace_bytes",
                        lambda sid: 808 * 1024 * 1024)

    out = workspace.write_file("PROPOSAL_MAPS.md", "x" * 30_000, session_id="s")
    error = str(out.get("error") or "")
    assert "full" in error
    assert "808.0 MiB used of 512 MiB" in error, error
    assert "29.3 KiB more" in error, error


def test_the_refusal_names_what_is_taking_the_room(full_session):
    """Naming the largest entry is the difference between a dead end and an
    action. Top level only: what fills a workspace is a directory copied in
    whole, and pointing inside it helps nobody.
    """
    out = workspace.write_file("PROPOSAL_MAPS.md", "x" * 30_000, session_id="s")
    error = str(out.get("error") or "")
    assert "Sonder_Engine_working" in error, error


def test_largest_entries_measures_a_directory_whole(full_session):
    largest = workspace._largest_entries("s")
    assert largest.startswith("Sonder_Engine_working"), largest
    assert "notes.md" in largest or "MiB" in largest


def test_a_write_that_fits_is_untouched(full_session):
    """The refusal must stay a refusal about SIZE. A workspace with room takes
    the write and says nothing about capacity.
    """
    monkey = workspace.MAX_WORKSPACE_BYTES
    assert monkey == 50_000
    out = workspace.write_file("tiny.md", "ok", session_id="s")
    assert out.get("ok") is True, out
    assert os.path.isfile(full_session / "tiny.md")
