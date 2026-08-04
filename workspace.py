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


def workspace_root():
    """The one persistent workspace directory, created on demand."""
    path = os.path.join(_ROOT, _WORKSPACE_DIR)
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


def list_files(session_id):
    """Everything in the workspace, newest first, with paths relative to the
    session root so nothing absolute reaches the browser."""
    root = session_root(session_id)
    out = []
    for dirpath, _dirs, files in os.walk(root):
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
    return {"ok": True, "path": os.path.relpath(target, root).replace(".", ""),
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


def snapshot_for_sandbox(session_id, max_bytes=8 << 20, max_files=200):
    """The workspace as a {relative_path: text} dict `sandbox.run` accepts.

    Binary and oversized files are skipped rather than mangled: the sandbox
    contract is text files, and a silently corrupted binary is worse than an
    absent one."""
    root = session_root(session_id)
    out, total = {}, 0
    for entry in list_files(session_id):
        if len(out) >= max_files or total >= max_bytes:
            break
        full = os.path.join(root, entry["path"])
        try:
            with open(full, "rb") as handle:
                raw = handle.read(max_bytes - total + 1)
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out[entry["path"]] = text
        total += len(raw)
    return out
