# workspace.py — the files the user hands the assistant to look at.
#
# One directory per session, holding whatever was uploaded plus whatever was
# extracted from it. `sandbox.run` can be pointed at it, so "have a look at
# this zip" and "run the tests in it" are the same workspace rather than two
# disconnected ideas.
#
# THE THREAT MODEL IS NOT THE USER. It is the archive. A zip is an untrusted
# structure even when the person who uploaded it is entirely trustworthy —
# they downloaded it from somewhere, or built it with a tool, or it is the
# thing they want examined precisely because they do not know what is in it.
# Everything below exists because an extractor that believes its input is the
# classic way a "just unzip it" feature becomes an arbitrary file write.
#
#   * zip slip — a member named `../../.ssh/authorized_keys` writes outside
#     the workspace. Checked by resolving the final path and requiring it to
#     stay under the root, not by looking for ".." in the name (which
#     `a/..%2f..` and friends defeat).
#   * absolute member names — `/etc/cron.d/x` ignores the destination
#     entirely.
#   * symlink members — a zip can contain a symlink `link -> /etc`, and a
#     LATER member `link/passwd` then writes through it. Extracting members in
#     order makes this a live sequence, not a curiosity, so symlinks are
#     refused outright.
#   * zip bombs — 42.zip is 42 KB and 4.5 PB expanded. Bounded three ways:
#     total uncompressed bytes, member count, and per-member compression
#     ratio, all checked from the CENTRAL DIRECTORY BEFORE a single byte is
#     written, because checking afterwards means the disk is already full.
#
# A refusal here returns a reason, never an exception: this is user-facing
# input handling, and "that archive declares 4.5 PB" is a finding the user
# should read, not a 500.

import difflib
import os
import re
import shutil
import tarfile
import time
import zipfile

import codemap

# Per session. Generous enough for a source tree or a document set, small
# enough that a runaway upload cannot fill a disk.
MAX_UPLOAD_BYTES = 64 << 20            # 64 MiB per file
MAX_WORKSPACE_BYTES = 512 << 20        # 512 MiB per session, everything in
MAX_EXTRACT_BYTES = 256 << 20          # 256 MiB out of one archive
MAX_EXTRACT_MEMBERS = 5000
# What one read may pull into a turn. Far below the upload limit on purpose:
# this bound is about a model's context, not a disk, and a file above it is
# what `chunks` exists for.
MAX_READ_BYTES = 200_000
# A diff long enough to review is short enough to read. Past this the change
# is not a patch, it is a rewrite, and the honest report is that it was
# truncated rather than a wall the reviewer skims.
MAX_DIFF_LINES = 400
# 42.zip's outer layer is ~1000x. Legitimate text compresses ~10x, and a big
# uniform file (a DB dump, a log) can reach 200x honestly, so the line is
# drawn where the ratio stops being explicable by content.
MAX_COMPRESSION_RATIO = 400

_ROOT = os.environ.get("ASSISTANT_WORKSPACE",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "workspaces"))

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


# THE WORKSPACE IS PERSISTENT, NOT PER SESSION.
#
# It used to be `session-{id}`, and a new session started with an empty one.
# That is the wrong lifetime for what this directory actually holds. A session
# is a conversation — it ends when you close the tab — while the code you are
# working on is the thing you come back to tomorrow. Tying them together meant
# re-uploading a codebase to ask a second question about it the next day, and
# it silently split the work: the memories from Monday's session were still
# there on Tuesday, describing files that were not.
#
# It also made the chunk map's lifetime wrong in the same way, and it is why
# `subagent_runner` had `store_upload(1, ...)` hardcoded — with a per-session
# root there was no correct id to pass from a background worker, so it guessed.
#
# One directory, one map, one project. Multiple named workspaces would be the
# next step if this ever needs to hold two unrelated projects at once; that is
# a real want but not this one, and the seam below is where it would go.
_WORKSPACE_DIR = "workspace"
_LEGACY_PREFIX = "session-"
_migrated = False


def configure(root):
    """Point the module at a workspace root. Tests call this with tmp_path."""
    global _ROOT, _migrated
    _ROOT = root
    _migrated = False


def root_under(base):
    """Where a workspace lives given an ASSISTANT_WORKSPACE of `base`.

    EXISTS BECAUSE TWO PLACES COMPUTED IT AND DISAGREED. `subagents` seeded a
    child's files into `<home>/workspace` and set its ASSISTANT_WORKSPACE to
    the same path — but this module joins `_WORKSPACE_DIR` onto that, so the
    child looked in `<home>/workspace/workspace` and found nothing. Every deep
    subagent ran against an empty workspace: no files, no chunks, no ids to
    read by, and its honest report was that it had no way to open the code it
    was sent at. The tool it said it lacked was there; the corpus was not.

    Folded to one function rather than fixed at the one caller who was wrong,
    per AGENTS.md — a guard that must be remembered will be forgotten, and
    this one already was."""
    return os.path.join(base, _WORKSPACE_DIR)


def workspace_root():
    """The one persistent workspace directory, created on demand."""
    path = root_under(_ROOT)
    os.makedirs(path, exist_ok=True)
    _migrate_legacy_sessions(path)
    return path


def session_root(session_id=None):
    """Deprecated alias. The workspace no longer depends on the session.

    Kept because every caller passes an id and silently ignoring it at each
    call site would be worse than ignoring it in one place, where the reason
    is written down. The argument is accepted and unused."""
    return workspace_root()


def _migrate_legacy_sessions(target):
    """Fold any `session-N` directories into the persistent workspace, once.

    A user who already had files must not have to re-upload them to keep
    them — a migration that loses the material is indistinguishable, from
    where they sit, from the feature not working. Collisions are suffixed
    rather than overwritten: two sessions holding different `main.py` files is
    the expected case, and picking a winner silently would destroy one."""
    global _migrated
    if _migrated:
        return
    _migrated = True
    # The marker lives BESIDE the workspace, never inside it. Bookkeeping put
    # in the workspace is indistinguishable from the user's own material: it
    # appeared in `list_files`, in `list_dir`, in the byte total, and it would
    # have been chunked and offered as something to read.
    marker = os.path.join(_ROOT, ".workspace-migrated")
    if os.path.exists(marker):
        return
    try:
        legacy = sorted(d for d in os.listdir(_ROOT)
                        if d.startswith(_LEGACY_PREFIX)
                        and os.path.isdir(os.path.join(_ROOT, d)))
    except OSError:
        legacy = []
    for name in legacy:
        source = os.path.join(_ROOT, name)
        for entry in sorted(os.listdir(source)):
            src = os.path.join(source, entry)
            dst = os.path.join(target, entry)
            if os.path.exists(dst):
                stem, ext = os.path.splitext(entry)
                dst = os.path.join(target, f"{stem}--{name}{ext}")
            try:
                shutil.move(src, dst)
            except OSError:
                pass
        try:
            os.rmdir(source)          # only when it emptied cleanly
        except OSError:
            pass
    try:
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("workspace is persistent; see workspace.py\n")
    except OSError:
        pass


def safe_name(name):
    """A stored filename that cannot be a path. Basename first (so `a/b.txt`
    becomes `b.txt` rather than a directory traversal), then anything outside
    a conservative alphabet folded to `_`."""
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    base = _SAFE_NAME.sub("_", base).lstrip(".") or "upload"
    return base[:120]


def _resolved_under(root, *parts):
    """The absolute path of `parts` under `root`, or None if it escapes.

    `os.path.realpath` resolves symlinks as well as `..`, so a component that
    is itself a link out of the tree is caught here too — the check the
    string-matching version of this misses."""
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root, *parts))
    if target == root_real or target.startswith(root_real + os.sep):
        return target
    return None


def workspace_bytes(session_id):
    total = 0
    for dirpath, _dirs, files in os.walk(session_root(session_id)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def store_upload(session_id, filename, data):
    """Write one uploaded file into the session workspace.

    Returns {"ok": bool, "name"|"error": ...}. Never raises for ordinary
    refusals — the caller is an HTTP route and the user needs the reason."""
    if not data:
        return {"ok": False, "error": "the file was empty"}
    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False,
                "error": f"{len(data) / 1e6:.1f} MB exceeds the "
                         f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB per-file limit"}
    root = session_root(session_id)
    if workspace_bytes(session_id) + len(data) > MAX_WORKSPACE_BYTES:
        return {"ok": False,
                "error": "this session's workspace is full; delete something "
                         "first"}
    name = safe_name(filename)
    target = _resolved_under(root, name)
    if target is None:
        return {"ok": False, "error": f"refused the filename {filename!r}"}
    # Never silently replace: two uploads called report.pdf are two files, and
    # clobbering the first is a surprise the user cannot undo.
    stem, ext = os.path.splitext(name)
    n = 1
    while os.path.exists(target):
        target = os.path.join(root, f"{stem}-{n}{ext}")
        n += 1
    with open(target, "wb") as handle:
        handle.write(data)
    return {"ok": True, "name": os.path.basename(target),
            "bytes": len(data)}


def read_file(relative, session_id=None, limit=MAX_READ_BYTES):
    """The current text of one workspace file, or an explicit absence.

    Returns {"ok", "text"|"error", "bytes"}. An unreadable file is a RESULT,
    never an exception — the callers are an edit path and an HTTP route, and
    both need the reason rather than a traceback."""
    root = session_root(session_id)
    target = _resolved_under(root, str(relative or ""))
    if target is None:
        return {"ok": False, "error": f"path escapes the workspace: "
                                      f"{relative!r}"}
    if not os.path.isfile(target):
        return {"ok": False, "error": f"no such file: {relative!r}"}
    try:
        with open(target, "r", encoding="utf-8") as handle:
            text = handle.read(limit + 1)
    except (OSError, UnicodeDecodeError) as exc:
        return {"ok": False, "error": f"could not read {relative!r}: {exc}"}
    if len(text) > limit:
        size = os.path.getsize(target)
        # "WORK ON IT IN CHUNKS" IS THE WRONG ADVICE FOR RAW DATA, and it is
        # the expensive one. A recorded story is tens of megabytes of JSON: read
        # sequentially it costs a fortune in context and returns almost nothing
        # per page, because the answer is one record somewhere in the middle
        # rather than something a reader accumulates. Chunking through it is
        # not a smaller version of reading it, it is the same cost paid slowly.
        #
        # So the refusal names the size and the shape and points at the only
        # approach that is cheap: search for the record, then read around it.
        data_like = str(relative).lower().rsplit(".", 1)[-1] in (
            "json", "jsonl", "ndjson", "csv", "log")
        if data_like and size > 200_000:
            return {"ok": False,
                    "error": f"{relative!r} is {size:,} bytes of raw data, not "
                             "source. Reading it in pages is expensive and "
                             "usually answers nothing — one record in the "
                             "middle is the thing you want. Search it for the "
                             "key or id you are after and read around the hit "
                             "instead of walking it."}
        return {"ok": False,
                "error": f"{relative!r} is {size:,} bytes, larger than the "
                         f"{limit:,} character read limit; work on it in "
                         "chunks"}
    return {"ok": True, "text": text, "bytes": len(text.encode())}


def write_file(relative, contents, session_id=None):
    """Replace one workspace file's contents, and say exactly what changed.

    THE ASSISTANT COULD NOT CHANGE ANYTHING IT COULD SEE. `sandbox.run` writes
    into a throwaway directory that is deleted the moment the run ends, and
    nothing anywhere wrote back to the workspace — so an assistant asked to
    edit its own source could reproduce a defect, design a fix, prove the fix
    correct in the sandbox, and then had no way to put it anywhere. The
    deliverable of a coding turn is a changed file and a reviewable diff, and
    neither existed.

    Returns {"ok", "path", "created", "diff", "before_bytes", "after_bytes"}.

    UNLIKE `store_upload` THIS DELIBERATELY REPLACES. An upload arriving twice
    is two files and clobbering the first is a surprise; an edit that wrote
    `module-1.py` beside `module.py` would leave the tree holding two versions
    and the map indexing both, which is the failure where an expand returns
    code that is no longer what runs. The undo is the diff, which is returned
    rather than merely logged.

    Directories are created as needed and the path is resolved with symlinks
    followed, so a component that is itself a link out of the tree is refused
    here rather than discovered afterwards."""
    text = "" if contents is None else str(contents)
    if len(text.encode()) > MAX_UPLOAD_BYTES:
        return {"ok": False, "error": "the new contents exceed the per-file "
                                      "limit"}
    root = session_root(session_id)
    relative = str(relative or "").strip().replace("\\", "/")
    if not relative:
        return {"ok": False, "error": "no path given"}
    # REFUSED, NOT REINTERPRETED. Stripping the leading slash contained the
    # write — `/etc/passwd` landed at `<workspace>/etc/passwd` — but silently
    # turned a path the caller meant absolutely into a different path it never
    # named. A write that goes somewhere other than where it was addressed is
    # a surprise whichever side of the boundary it lands on, and the caller
    # cannot see it happened.
    if relative.startswith("/") or (len(relative) > 1 and relative[1] == ":"):
        return {"ok": False,
                "error": f"paths are relative to the workspace; {relative!r} "
                         "is absolute"}
    # The parent must resolve inside the workspace even when the file does not
    # exist yet, so the check is on the directory and then on the whole path.
    parent = _resolved_under(root, os.path.dirname(relative) or ".")
    if parent is None:
        return {"ok": False,
                "error": f"path escapes the workspace: {relative!r}"}
    target = os.path.join(parent, os.path.basename(relative))
    if _resolved_under(root, relative) is None and os.path.exists(target):
        return {"ok": False,
                "error": f"path escapes the workspace: {relative!r}"}
    existed = os.path.isfile(target)
    before = ""
    if existed:
        try:
            with open(target, "r", encoding="utf-8") as handle:
                before = handle.read()
        except (OSError, UnicodeDecodeError):
            # Present but unreadable as text: the diff would be a lie, so it
            # says so rather than showing the whole file as an addition.
            before = ""
    if (workspace_bytes(session_id) - len(before.encode())
            + len(text.encode()) > MAX_WORKSPACE_BYTES):
        return {"ok": False, "error": "this workspace is full"}
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"could not write {relative!r}: {exc}"}
    return {"ok": True, "path": relative, "created": not existed,
            "diff": diff_text(before, text, relative),
            "before_bytes": len(before.encode()),
            "after_bytes": len(text.encode()),
            "unchanged": before == text}


def diff_text(before, after, path="", context=3):
    """A unified diff, or a sentence saying there is nothing to show.

    A REVIEWER READS THE DIFF, NOT THE FILE. An edit reported as "wrote 812
    lines to memory.py" is unreviewable — the reader has to hold both versions
    to find the change, which is precisely the work the diff exists to do.
    Truncated at MAX_DIFF_LINES, and it says when it truncated: a diff that
    silently stops is worse than none, because it reads as the whole change."""
    lines = list(difflib.unified_diff(
        (before or "").splitlines(), (after or "").splitlines(),
        fromfile=f"a/{path}" if path else "before",
        tofile=f"b/{path}" if path else "after",
        lineterm="", n=context))
    if not lines:
        return "(no change)"
    if len(lines) > MAX_DIFF_LINES:
        kept = lines[:MAX_DIFF_LINES]
        kept.append(f"... diff truncated: {len(lines)} lines total, "
                    f"{MAX_DIFF_LINES} shown")
        lines = kept
    return "\n".join(lines)


# Directories that are machinery, not material.
#
# `.git` alone contributed 60-odd entries to a 119-"file" workspace — object
# files and refs, counted in the totals, walked by the chunk mapper, listed
# back to the assistant as things it might want to read. None of it is the
# code the user handed over, all of it competes with the code for attention,
# and git objects are compressed binary that chunk into noise.
#
# `codemap` has had its own `_SKIP_DIRS` since before this module existed;
# the mistake was that `list_files` never consulted one. Same set, one
# meaning — the module that already knew is the one to ask.
# Machinery, and then FIXTURES. The two are skipped for the same reason from
# the index's point of view: neither is the code the assistant is being asked
# about. A project's `demo/` and `demos/` hold recorded stories and captured
# runs — 18.2 MB and 11.7 MB of JSON in the one that prompted this — and
# chunking them buys thousands of entries nobody can navigate by gist while
# spending the budget that real source needed. Twenty-seven modules went
# unindexed behind two fixtures, and `outline` answered "no indexed file
# matches" for a file that was plainly on disk.
#
# An archive is not a live codebase. Pruned at the WALK so nothing downstream
# has to remember, which is the same reason the machinery names are here.
_SKIP_WALK_DIRS = frozenset(codemap._SKIP_DIRS) | {
    ".git", ".hg", ".svn", ".pytest_cache", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".tox", "node_modules", ".venv", "venv",
}

# Recorded runs and captured stories. NOT pruned from the walk — the assistant
# must still be able to list them and read one on purpose — but the indexer
# takes only their prose. See `chunks.ingest_workspace`.
ARCHIVE_DIRS = frozenset({"demo", "demos", "fixtures", "snapshots", "backups"})


def in_archive_dir(relative):
    """True when a workspace-relative path sits under a recorded-run folder."""
    parts = str(relative or "").replace("\\", "/").split("/")
    return any(p in ARCHIVE_DIRS for p in parts[:-1])


def list_files(session_id=None):
    """Everything in the workspace, newest first, with paths relative to the
    workspace root so nothing absolute reaches the browser.

    Machinery directories are skipped — see `_SKIP_WALK_DIRS`."""
    root = session_root(session_id)
    out = []
    for dirpath, dirs, files in os.walk(root):
        # Pruned IN PLACE, which is what stops os.walk descending into them
        # at all rather than filtering their contents afterwards.
        dirs[:] = [d for d in dirs if d not in _SKIP_WALK_DIRS]
        for name in files:
            full = os.path.join(dirpath, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            out.append({
                "path": os.path.relpath(full, root),
                "bytes": stat.st_size,
                "modified": stat.st_mtime,
                "archive": is_archive(name),
            })
    out.sort(key=lambda f: f["modified"], reverse=True)
    return out


def delete_file(session_id, relative):
    root = session_root(session_id)
    target = _resolved_under(root, relative)
    if target is None or not os.path.exists(target):
        return {"ok": False, "error": "no such file in this workspace"}
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    else:
        os.remove(target)
    return {"ok": True}


def clear(session_id):
    shutil.rmtree(session_root(session_id), ignore_errors=True)
    return session_root(session_id)


def is_archive(name):
    low = str(name or "").lower()
    return (low.endswith(".zip")
            or low.endswith(".tar") or low.endswith(".tar.gz")
            or low.endswith(".tgz") or low.endswith(".tar.bz2")
            or low.endswith(".tar.xz"))


# ---- Extraction ----

def _plan_zip(archive):
    """Inspect the central directory and decide before writing anything."""
    members, total = [], 0
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_EXTRACT_MEMBERS:
            return None, (f"the archive declares {len(infos)} members, over "
                          f"the {MAX_EXTRACT_MEMBERS} limit")
        for info in infos:
            if info.is_dir():
                continue
            # The high 16 bits of external_attr are the unix mode; 0xA000 is
            # S_IFLNK. A symlink member is refused rather than followed: a
            # later member writing THROUGH it is how extraction escapes a
            # directory that every path check said was safe.
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                return None, (f"the archive contains a symlink "
                              f"({info.filename!r}); refusing to extract it")
            total += info.file_size
            if total > MAX_EXTRACT_BYTES:
                return None, (f"the archive expands to over "
                              f"{MAX_EXTRACT_BYTES / 1e6:.0f} MB")
            if (info.compress_size > 0
                    and info.file_size / info.compress_size
                    > MAX_COMPRESSION_RATIO):
                return None, (
                    f"{info.filename!r} expands {info.file_size / info.compress_size:.0f}x, "
                    f"over the {MAX_COMPRESSION_RATIO}x limit — that is the "
                    "shape of a decompression bomb, not of a document")
            members.append(info.filename)
    return members, ""


def _plan_tar(archive):
    members, total = [], 0
    with tarfile.open(archive) as tf:
        for info in tf.getmembers():
            if info.isdir():
                continue
            if info.issym() or info.islnk():
                return None, (f"the archive contains a link "
                              f"({info.name!r}); refusing to extract it")
            if not info.isfile():
                return None, (f"{info.name!r} is not a regular file "
                              "(device, fifo); refusing to extract it")
            total += info.size
            if len(members) > MAX_EXTRACT_MEMBERS:
                return None, f"over {MAX_EXTRACT_MEMBERS} members"
            if total > MAX_EXTRACT_BYTES:
                return None, (f"the archive expands to over "
                              f"{MAX_EXTRACT_BYTES / 1e6:.0f} MB")
            members.append(info.name)
    return members, ""


def extract(session_id, relative):
    """Unpack one archive into a sibling directory inside the workspace.

    Every member is planned first, then written one at a time through the
    same path resolution `store_upload` uses. Extraction is NOT delegated to
    `ZipFile.extractall` / `TarFile.extractall`, because the safety of those
    depends on the Python version (tar filters landed in 3.12) and on nobody
    passing a path that resolves elsewhere — a guard that must be remembered
    will be forgotten, so the resolution happens per member, here, always."""
    root = session_root(session_id)
    archive = _resolved_under(root, relative)
    if archive is None or not os.path.isfile(archive):
        return {"ok": False, "error": "no such file in this workspace"}
    if not is_archive(archive):
        return {"ok": False, "error": "not an archive this can unpack "
                                      "(.zip, .tar, .tar.gz, .tgz, .tar.bz2, "
                                      ".tar.xz)"}
    is_zip = zipfile.is_zipfile(archive)
    try:
        planned, why = (_plan_zip(archive) if is_zip else _plan_tar(archive))
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
        return {"ok": False, "error": f"could not read the archive: {exc}"}
    if planned is None:
        return {"ok": False, "error": why}
    if workspace_bytes(session_id) + sum_sizes(archive, is_zip) \
            > MAX_WORKSPACE_BYTES:
        return {"ok": False, "error": "extracting this would fill the "
                                      "session workspace"}

    stem = os.path.splitext(os.path.basename(archive))[0]
    stem = stem[:-4] if stem.endswith(".tar") else stem
    dest = os.path.join(root, safe_name(stem) + "-extracted")
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(root, f"{safe_name(stem)}-extracted-{n}")
        n += 1
    os.makedirs(dest, exist_ok=True)

    written, refused = [], []
    opener = zipfile.ZipFile if is_zip else tarfile.open
    with opener(archive) as handle:
        for name in planned:
            # Resolve against the DESTINATION, then re-check containment.
            # This is the zip-slip check, and it is done on the resolved real
            # path rather than by looking for ".." in the member name —
            # string inspection is defeated by encoding tricks and by a
            # component that is itself a symlink.
            target = _resolved_under(dest, name)
            if target is None:
                refused.append(name)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = (handle.open(name) if is_zip
                      else handle.extractfile(name))
            if source is None:
                refused.append(name)
                continue
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            written.append(os.path.relpath(target, root))
    return {"ok": True, "into": os.path.relpath(dest, root),
            "written": len(written), "files": written[:200],
            "refused": refused}


def sum_sizes(archive, is_zip):
    try:
        if is_zip:
            with zipfile.ZipFile(archive) as zf:
                return sum(i.file_size for i in zf.infolist())
        with tarfile.open(archive) as tf:
            return sum(m.size for m in tf.getmembers())
    except Exception:
        return 0


def list_dir(relative="", limit=200):
    """ONE level of the workspace tree. The unit of navigation.

    `describe` used to put a flat list of every file into every turn payload,
    which is the context-bloat version of navigation: the assistant never
    "goes" anywhere, it is simply handed everything and left to find the
    relevant part by reading. A 115-file upload spent the payload on paths
    before a single question was asked.

    This is the opposite shape — you are somewhere, you see what is here, and
    you step into what looks right. Directories are marked and carry their
    child counts, so the next step is an informed one rather than a guess.

    Escapes are refused by `_resolved_under`, which resolves symlinks as well
    as `..`; the workspace boundary is enforced here exactly as it is for
    writes."""
    root = workspace_root()
    target = _resolved_under(root, str(relative or "").strip("/") or ".")
    if target is None:
        return {"ok": False, "path": relative,
                "error": "that path is outside the workspace"}
    if not os.path.isdir(target):
        return {"ok": False, "path": relative,
                "error": "not a directory"
                         + (" — it is a file; expand its chunks to read it"
                            if os.path.exists(target) else "; it does not "
                            "exist. List its parent to see what does.")}
    entries, truncated = [], False
    try:
        names = sorted(os.listdir(target))
    except OSError as exc:
        return {"ok": False, "path": relative, "error": str(exc)[:120]}
    for name in names:
        if len(entries) >= limit:
            truncated = True
            break
        full = os.path.join(target, name)
        # The same skip set the walker uses. Navigation that offers `.git` as
        # somewhere to go is navigation into machinery, and the assistant has
        # no way to know from the name that it is a dead end.
        if os.path.isdir(full) and name in _SKIP_WALK_DIRS:
            continue
        rel = os.path.relpath(full, root)
        if os.path.isdir(full):
            try:
                count = len(os.listdir(full))
            except OSError:
                count = 0
            entries.append({"path": rel, "dir": True, "entries": count})
        else:
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            entries.append({"path": rel, "dir": False, "bytes": size,
                            "language": codemap.language_of(name) or ""})
    # `relpath` returns "." for the root itself, and the fix for that stripped
    # EVERY dot from the path — so `Sonder_Engine-alpha-7.2/` was echoed back as
    # `Sonder_Engine-alpha-72/` while every entry inside it kept its real name,
    # and any path with a version or an extension in it came back subtly wrong.
    # A path that is not the path asked for is worse than an error, because it
    # reads as an answer.
    return {"ok": True,
            "path": "" if os.path.relpath(target, root) == "." else
                    os.path.relpath(target, root),
            "entries": entries, "count": len(entries), "truncated": truncated,
            "how_to_use_this": (
                "One level only. Step into a directory by listing its path; "
                "read a FILE by expanding its chunk ids, never by listing it.")}


def describe(session_id=None, limit=40):
    """The workspace as a STARTING POINT, not an inventory.

    Deliberately shallow: the top level plus totals. Everything below is
    reached with `list_dir`, which is what makes navigation real rather than
    decorative — an assistant handed the whole tree has no reason to navigate
    and no way to tell which part of it matters."""
    files = list_files(session_id)
    top = list_dir("")
    return {
        "count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "top_level": top.get("entries", [])[:limit],
        "persistent": True,
        "root_hint": ("one persistent workspace, kept across sessions. "
                      "This is only the TOP level — use `need_more.list_dir` "
                      "to go deeper, and expand chunk ids to read files."),
    }


def has_files(session_id):
    return bool(list_files(session_id))


def codemap_for(session_id, max_files=80):
    """The structure of what the user uploaded, for navigation.

    Built from the workspace directory itself rather than from a snapshot, so
    extracted archives are mapped in place with their real paths — the paths
    the assistant will name when it asks the sandbox to run something."""
    import codemap
    return codemap.for_prompt(session_root(session_id), max_files=max_files)


# A binary file travels as bytes, per-file bounded. The sandbox contract was
# text-only, which meant a SQLite database could never reach an experiment —
# so "run this project's suite" against any project that keeps state on disk
# failed at the first query with `no such table`, and the failure looked like
# a defect in the project under test rather than a file that never arrived.
SNAPSHOT_BINARY_MAX = 2 << 20
# Where the snapshot declares itself. A leading underscore and a full sentence
# for a name, because it shares a namespace with the user's own files.
WITHHELD_MANIFEST = "_withheld_from_this_workspace.md"
# How much of the manifest goes to naming individual files. The
# per-directory roll-up above it is never bounded — it is the part that
# answers "is this subtree here?", and a bound on THAT is what let a whole
# `tools/` directory vanish from a list of what had vanished.
WITHHELD_DETAIL_CHARS = 40_000


# THE BYTE CEILING IS THE HONEST CONSTRAINT; the file count was costing
# coverage it was not buying anything for. Measured on a 593-file workspace
# holding two repositories: at 200 files the snapshot delivered 6.2 MB and 27
# of the engine's test files; raising the count alone delivered 404 files,
# 230 test files, and 8.4 MB — the same ceiling, reached properly. Past 600
# the number changes nothing at all, because bytes bind first. So the count
# stays only as a backstop against a workspace of a million tiny files.
SNAPSHOT_MAX_FILES = 2000


# WHAT THE TREE ACTUALLY WEIGHS, once archives are excluded — not a round
# number. At 8 MB the Engine's suite could not even be COLLECTED: pytest died
# on `RuntimeError: Directory 'static' does not exist` for every module,
# because the 21 files under `static/` sit at rank 537 of 593 in a
# newest-first listing and fell past the ceiling. Delivered tests that cannot
# import are not a delivered suite.
#
# Measured on the real workspace, two repositories in it: 8 MB gives 403 files
# and 0 of 21 static; 12 MB gives all 559 and all 21, weighing 10.5 MB; 16 MB
# gives exactly the same, because the workspace saturates. So this is not a
# budget with headroom bolted on, it is the size of the thing being copied,
# and the archive rule above is what keeps raw story data out of it.
SNAPSHOT_MAX_BYTES = 12 << 20


def snapshot_for_sandbox(session_id, max_bytes=SNAPSHOT_MAX_BYTES,
                         max_files=SNAPSHOT_MAX_FILES,
                         binary_max=SNAPSHOT_BINARY_MAX):
    """The workspace as a {relative_path: text-or-bytes} dict `sandbox.run`
    accepts, TOGETHER WITH what it could not bring.

    ABSENCE FROM THE SANDBOX IS NOT ABSENCE FROM THE REPOSITORY, and until
    this said so nothing downstream could tell the two apart. The caps are
    real and have to be: `list_files` is newest-first, so what a child
    receives is a 200-file window ordered by modification time — for an
    unpacked archive that ordering is arbitrary. A deep subagent was given a
    591-file tree, received 200, walked what it had, found no `tests/`
    directory, and reported that the project HAS NO TEST SUITE — with a
    recursive walk, a root listing and a pytest run as evidence, all of them
    honest, all of them about a workspace that was three-quarters absent. It
    went on to reason about why the suite had been "stripped". The tree has
    300 test files.

    A cap is not the defect. A cap nobody downstream can see is: every
    observation drawn inside the sandbox silently changes meaning, and no
    amount of care by the reader recovers it. So the omission travels WITH
    the files, as one more file, rather than as a return value each caller
    would have to remember to look at (AGENTS.md: a guard that must be
    remembered will be forgotten). Both callers write this dict to disk, so
    the child reads the manifest exactly where it would look for the code."""
    root = session_root(session_id)
    out, total, withheld = {}, 0, []
    for entry in list_files(session_id):
        path = entry["path"]
        # `continue`, NOT `break`. Stopping at the cap was what made the
        # count unknowable: the loop that hit the ceiling was also the only
        # thing that could have said how far past it the workspace went.
        if len(out) >= max_files:
            withheld.append((path, f"past the {max_files}-file ceiling"))
            continue
        if total >= max_bytes:
            withheld.append((path, "past the total size ceiling"))
            continue
        # AN ARCHIVE IS NOT A LIVE CODEBASE, AND THE SAME RULE THE INDEX
        # APPLIES HAS TO APPLY HERE. `chunks` has skipped non-prose inside a
        # recorded-run folder since the day it was told to; this walk never
        # learned it, and one 2.4 MB story JSON took 29% of the whole byte
        # budget while the engine's 300 test files got NONE of it. Two
        # spellings of one rule, and the one that governs what a subagent can
        # actually see was the one missing it.
        if in_archive_dir(path) and not path.lower().endswith(
                (".md", ".markdown", ".rst", ".txt")):
            withheld.append((path, "recorded run — only prose is taken from "
                                   "an archive folder"))
            continue
        full = os.path.join(root, path)
        try:
            raw = open(full, "rb").read(max_bytes - total + 1)
        except OSError as exc:
            withheld.append((path, f"could not be read: {exc.strerror}"))
            continue
        binary = b"\x00" in raw[:8192]
        if not binary:
            try:
                out[path] = raw.decode("utf-8")
            except UnicodeDecodeError:
                binary = True
        if binary:
            if len(raw) > binary_max:
                withheld.append((path, "binary, over the per-file size limit"))
                continue
            out[path] = raw
        total += len(raw)
    if withheld:
        out[WITHHELD_MANIFEST] = _withheld_note(len(out), withheld)
    return out


def _withheld_note(delivered, withheld):
    """The manifest. Written to be read by something that is about to
    conclude a file does not exist.

    THE DIRECTORY ROLL-UP IS COMPLETE AND COMES FIRST; the per-file list is
    the bounded part. It was the other way round, capped at 60 entries of 192,
    and the cap fell exactly where it did the most harm: not one of the 33
    files under `tools/` was named. A subagent classified nine failing test
    modules, found five of them raising `ModuleNotFoundError` on `tools`, and
    recorded that against the project — the manifest built to catch precisely
    that could not answer, because the whole directory had fallen off the end
    of a display limit.

    "Named, not counted" was the rule this file was written to serve, and a
    list truncated at 60 is counted again for everything past 60. Every
    withheld DIRECTORY is now always named with its file count, so "is this
    subtree here?" — the question that keeps going wrong — is always
    answerable, and only the file-by-file detail is subject to a budget."""
    by_dir = {}
    for path, _why in withheld:
        parent = os.path.dirname(str(path).replace("\\", "/")) or "."
        by_dir[parent] = by_dir.get(parent, 0) + 1
    total = delivered + len(withheld)
    lines = [
        "# Files NOT in this workspace",
        "",
        f"This workspace holds {delivered} of {total} files from the source "
        "tree. The rest were withheld by size and count limits.",
        "",
        "**A path missing from here may still exist in the real tree.** Do "
        "not conclude that a file, a directory or a whole test suite is "
        "absent from the project because a listing or a recursive walk inside "
        "this workspace did not find it. Say what you searched and say that "
        "the workspace is partial.",
        "",
        f"## Every directory with something withheld ({len(by_dir)}, complete)",
        "",
    ]
    lines += [f"- `{d}/` — {n} file(s) withheld"
              for d, n in sorted(by_dir.items())]
    lines += ["", f"## The files themselves ({len(withheld)})", ""]
    spent, shown = 0, 0
    for path, why in withheld:
        entry = f"- `{path}` — {why}"
        if spent + len(entry) > WITHHELD_DETAIL_CHARS:
            break
        lines.append(entry)
        spent += len(entry)
        shown += 1
    if shown < len(withheld):
        lines.append(f"- …and {len(withheld) - shown} more — the directory "
                     "roll-up above is complete, so check it before "
                     "concluding anything is absent")
    return "\n".join(lines) + "\n"
