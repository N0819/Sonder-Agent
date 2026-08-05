# The files the user hands over, and the archives among them.
#
# The threat model is not the user, it is the ARCHIVE: a zip is untrusted
# structure even when the person who uploaded it is entirely trustworthy,
# because they downloaded it from somewhere, or it is the thing they want
# examined precisely because they do not know what is in it. Every test here
# is a way "just unzip it" becomes an arbitrary file write.

import os
import tarfile
import zipfile

import pytest

import workspace


@pytest.fixture
def ws(tmp_path):
    workspace.configure(str(tmp_path / "workspaces"))
    yield tmp_path


def _zip(path, build):
    with zipfile.ZipFile(path, "w") as zf:
        build(zf)
    return path


def test_zip_slip_writes_nothing_outside_the_workspace(ws):
    """A member named `../../ESCAPED.txt` writes outside the destination. The
    check resolves the final real path rather than looking for ".." in the
    name, because string inspection is defeated by encoding and by a
    component that is itself a symlink."""
    root = workspace.session_root(1)
    archive = _zip(os.path.join(root, "slip.zip"),
                   lambda z: z.writestr("../../ESCAPED.txt", "pwned"))
    out = workspace.extract(1, os.path.basename(archive))
    assert out["ok"] and out["written"] == 0
    assert out["refused"] == ["../../ESCAPED.txt"]
    assert not os.path.exists(str(ws / "ESCAPED.txt"))


def test_an_absolute_member_name_is_refused(ws):
    """`/etc/cron.d/x` ignores the destination entirely."""
    root = workspace.session_root(1)
    archive = _zip(os.path.join(root, "abs.zip"),
                   lambda z: z.writestr("/tmp/ABSOLUTE_MEMBER.txt", "pwned"))
    out = workspace.extract(1, os.path.basename(archive))
    assert out["refused"]
    assert not os.path.exists("/tmp/ABSOLUTE_MEMBER.txt")


def test_a_symlink_member_stops_the_whole_extraction(ws):
    """A zip can carry a symlink `link -> /etc` and a LATER member
    `link/passwd` then writes THROUGH it — every individual path check passes
    and the write still lands outside. Extracting members in order makes that
    a live sequence, not a curiosity, so the archive is refused whole."""
    root = workspace.session_root(1)

    def build(z):
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        z.writestr(info, "/etc")
        z.writestr("link/passwd", "pwned")

    archive = _zip(os.path.join(root, "sym.zip"), build)
    out = workspace.extract(1, os.path.basename(archive))
    assert out["ok"] is False and "symlink" in out["error"]


def test_a_decompression_bomb_is_refused_before_anything_is_written(ws):
    """42.zip is 42 KB and 4.5 PB expanded. The ratio is read from the
    central directory BEFORE a byte is written, because checking afterwards
    means the disk is already full."""
    root = workspace.session_root(1)

    def build(z):
        z.writestr("bomb.bin", b"\0" * (40 << 20))

    archive = os.path.join(root, "bomb.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        build(zf)
    out = workspace.extract(1, "bomb.zip")
    assert out["ok"] is False and "bomb" in out["error"]
    assert not os.path.exists(os.path.join(root, "bomb-extracted"))


def test_a_tar_link_member_is_refused(ws):
    """tarfile's extraction filters landed in 3.12; relying on the version
    would be a guard that must be remembered."""
    root = workspace.session_root(1)
    archive = os.path.join(root, "l.tar")
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc"
        tf.addfile(info)
    out = workspace.extract(1, "l.tar")
    assert out["ok"] is False and "link" in out["error"]


def test_an_ordinary_archive_still_extracts(ws):
    """The guards have to leave the honest case working, or they will be
    turned off."""
    root = workspace.session_root(1)
    archive = _zip(os.path.join(root, "good.zip"),
                   lambda z: (z.writestr("proj/main.py", "print('hi')"),
                              z.writestr("proj/README.md", "# hi")))
    out = workspace.extract(1, os.path.basename(archive))
    assert out["ok"] and out["written"] == 2
    assert not out["refused"]
    assert os.path.exists(os.path.join(root, out["into"], "proj", "main.py"))


def test_an_upload_never_silently_replaces_another(ws):
    """Two files called report.pdf are two files. Clobbering the first is a
    surprise the user cannot undo."""
    workspace.store_upload(1, "report.pdf", b"first")
    second = workspace.store_upload(1, "report.pdf", b"second")
    assert second["ok"] and second["name"] != "report.pdf"
    assert len(workspace.list_files(1)) == 2


def test_a_filename_cannot_be_a_path(ws):
    """An uploaded name arrives from a browser and is untrusted like any
    other input."""
    out = workspace.store_upload(1, "../../etc/passwd", b"x")
    assert out["ok"] and "/" not in out["name"] and ".." not in out["name"]


def test_an_oversized_upload_is_refused_with_a_reason(ws):
    """A refusal is a finding the user should read, never a 500."""
    out = workspace.store_upload(1, "big.bin",
                                 b"x" * (workspace.MAX_UPLOAD_BYTES + 1))
    assert out["ok"] is False and "limit" in out["error"]


def test_a_binary_file_travels_whole_rather_than_being_dropped(ws):
    """A silently corrupted binary is worse than an absent one — which is why
    these were skipped — but an absent one broke the case that matters:
    a SQLite database could never reach a sandbox, so "run this project's
    suite" failed at the first query with `no such table` on any project that
    keeps state on disk, and the experiment recorded the fault against the
    project under test."""
    workspace.store_upload(1, "a.py", b"print('hi')")
    workspace.store_upload(1, "state.db", b"SQLite format 3\x00\x01\x02")
    snapshot = workspace.snapshot_for_sandbox(1)
    assert snapshot["a.py"] == "print('hi')"
    # Bytes, byte-for-byte. `str(raw)` would deliver "b'SQLite format 3...'".
    assert snapshot["state.db"] == b"SQLite format 3\x00\x01\x02"


def test_a_binary_too_large_to_carry_is_named_rather_than_dropped(ws):
    """Silence is the defect, not the limit."""
    workspace.store_upload(1, "huge.db", b"\x00" * 4096)
    snapshot = workspace.snapshot_for_sandbox(1, binary_max=16)
    assert "huge.db" not in snapshot
    assert "huge.db" in snapshot[workspace.WITHHELD_MANIFEST]


def test_the_workspace_says_what_it_could_not_bring(ws):
    """A deep subagent was handed a 591-file tree, received the 200 newest,
    walked what it had, and reported that the project HAS NO TEST SUITE —
    with a recursive walk, a root listing and a pytest run as evidence, all
    honest, all about a workspace three-quarters absent. It then reasoned
    about why the suite had been "stripped". The tree has 300 test files.
    The cap was never the defect; a cap nothing downstream could see was."""
    for n in range(12):
        workspace.store_upload(1, f"f{n}.py", b"x = 1")
    snapshot = workspace.snapshot_for_sandbox(1, max_files=5)
    assert len(snapshot) == 6, "five files and the manifest"
    note = snapshot[workspace.WITHHELD_MANIFEST]
    # The COUNT is the part a truncating loop cannot report: `break` left the
    # workspace unable to say how far past the ceiling the tree went.
    assert "5 of 12" in note
    assert "may still exist in the real tree" in note


# ---- One persistent workspace, navigated a directory at a time ----

def test_the_workspace_survives_a_new_session(ws):
    """It was `session-{id}`, so a new conversation started empty and you
    re-uploaded a codebase to ask a second question about it the next day.
    Worse, it split the work silently: Monday's memories were still there on
    Tuesday, describing files that were not."""
    workspace.store_upload(1, "keep.py", b"print('hi')")
    assert [f["path"] for f in workspace.list_files(2)] == ["keep.py"]
    assert workspace.session_root(1) == workspace.session_root(99)


def test_legacy_session_directories_are_folded_in_once(tmp_path):
    """A migration that loses the material is, from where the user sits,
    indistinguishable from the feature not working."""
    import os
    root = tmp_path / "ws"
    (root / "session-2").mkdir(parents=True)
    (root / "session-2" / "old.py").write_text("print('old')")
    (root / "session-5").mkdir()
    (root / "session-5" / "old.py").write_text("print('other')")
    workspace.configure(str(root))
    names = sorted(f["path"] for f in workspace.list_files(1))
    # Both survive: a name collision is suffixed, never resolved by deleting.
    assert "old.py" in names and any(n.startswith("old--") for n in names)
    assert not os.path.exists(root / "session-2")


def test_bookkeeping_never_appears_as_a_users_file(ws):
    """The migration marker was written INTO the workspace and showed up in
    `list_files`, in the byte total, and would have been chunked and offered
    as something to read."""
    workspace.store_upload(1, "real.py", b"x = 1")
    assert [f["path"] for f in workspace.list_files(1)] == ["real.py"]


def test_listing_is_one_level_not_the_whole_tree(ws):
    """The point of navigation. A flat dump of every path is the context-bloat
    version: the assistant never goes anywhere, it is handed everything."""
    import os
    root = workspace.workspace_root()
    os.makedirs(os.path.join(root, "src", "deep"), exist_ok=True)
    open(os.path.join(root, "top.py"), "w").write("x = 1")
    open(os.path.join(root, "src", "mid.py"), "w").write("y = 2")
    open(os.path.join(root, "src", "deep", "low.py"), "w").write("z = 3")

    top = workspace.list_dir("")
    paths = [e["path"] for e in top["entries"]]
    assert "top.py" in paths and "src" in paths
    assert "src/mid.py" not in paths           # one level, not recursive
    assert any(e.get("dir") and e["entries"] == 2 for e in top["entries"])

    inner = workspace.list_dir("src")
    assert sorted(e["path"] for e in inner["entries"]) == ["src/deep",
                                                           "src/mid.py"]


def test_listing_outside_the_workspace_is_refused(ws):
    """The same boundary as writes, enforced by the same resolver — symlinks
    and `..` alike."""
    out = workspace.list_dir("../../etc")
    assert out["ok"] is False and "outside the workspace" in out["error"]


def test_listing_a_file_says_to_read_it_instead(ws):
    """"not a directory" alone leaves the assistant to guess the next move,
    and the guess costs a round."""
    workspace.store_upload(1, "a.py", b"x = 1")
    out = workspace.list_dir("a.py")
    assert out["ok"] is False and "expand its chunks" in out["error"]
