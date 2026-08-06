# sandbox.py — a bounded place to run something and find out.
#
# The assistant's coding capability rests on one claim: that it can OBSERVE
# what code does rather than predict it. Everything in coding.py depends on
# that observation being real, cheap, and safe to repeat, because the whole
# method is repetition — run it, read the result, revise, run it again.
#
# So this is deliberately small and deliberately hostile:
#
#   * a fresh temporary workspace per experiment, thrown away after;
#   * a wall-clock timeout AND a CPU ceiling, because the commonest failure of
#     generated code is not a crash, it is a loop;
#   * output capped AT THE PIPE, because the second commonest is a loop that
#     prints — and a cap applied after the fact is not a cap;
#   * the process tree killed as a group, because the third is a child that
#     outlives its parent.
#
# THE HONEST LIMIT, stated precisely rather than gestured at. This is a guard
# rail, not a jail, and the earlier wording implied protections that
# measurement showed were absent -- "writing files somewhere surprising" and
# "reading the host's environment" were both listed as stopped, and neither
# was.
#
# WHAT IS BOUNDED: wall clock (timeout), CPU seconds (RLIMIT_CPU), address
# space (RLIMIT_AS, 1 GiB), written-file size (RLIMIT_FSIZE), open descriptors
# (RLIMIT_NOFILE), output actually READ into the parent (bounded at the pipe,
# so runaway printing can no longer exhaust the host's memory), and the
# process TREE, which is killed as a group on every exit path.
#
# WHAT IS NOT: the filesystem is not confined -- code can read and write
# outside the workspace, including this repository. The network is not
# blocked; the proxy variables are a speed bump for libraries that honour
# them and nothing else. Process count is not capped (see _limits for the
# measurement behind that). The environment is now an ALLOWLIST, so secrets
# no longer leak by naming convention, but the sandbox runs as the host user
# with the host user's privileges.
#
# Real confinement is one line away and deliberately not taken here:
# `bwrap --unshare-all --bind <workspace> <workspace>` if bubblewrap is
# available. Until it is, the list above is the whole truth.

import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:
    import resource
except ImportError:                                   # non-POSIX
    resource = None

# Long enough for a real test run, short enough that a loop is caught while
# the user is still watching.
DEFAULT_TIMEOUT = 20.0
# The most an experiment may ask for. 20s is the right DEFAULT — a runaway
# loop should die quickly — but it was also the ceiling, and the ceiling was
# measured wrong: this project's own suite takes ~34s, so "run your tests and
# show me they pass" was unreachable through the one verb that can run them.
# The assistant hit it twice in one audit and worked around it with `-x`,
# which stops at the first failure and therefore cannot answer "is the suite
# green". A limit that forces a weaker question is a limit set too low.
MAX_TIMEOUT = 180.0
# Per stream. A truncated tail is more useful than a truncated head: the
# traceback is at the end.
MAX_OUTPUT = 20_000

# Host environment variables that must not reach the sandbox. The assistant's
# own API keys are the ones that matter -- code it wrote should not be able to
# read the credentials it is running under.
_SCRUBBED_PREFIXES = ("ASSISTANT_", "OPENAI_", "ANTHROPIC_", "AWS_", "GH_",
                      "GITHUB_", "SSH_", "GOOGLE_")


# Resource ceilings. None of these existed: measured from inside the sandbox,
# RLIMIT_AS, RLIMIT_CPU and RLIMIT_FSIZE were all unlimited and NPROC was
# 53,093. "A guard rail, not a jail" is an honest thing to say about
# filesystem and network confinement; it was not an honest description of
# bounds that simply were not set. These are the cheap half of the jail and
# they cost one preexec_fn.
MAX_MEMORY_BYTES = 1 << 30          # 1 GiB
MAX_FILE_BYTES = 64 << 20           # 64 MiB
MAX_OPEN_FILES = 256

# Environment allowlist, replacing a prefix DENYLIST. `_SCRUBBED_PREFIXES`
# caught the names it knew and passed 76 other variables straight through --
# a `MY_SERVICE_TOKEN` was printed back out of the sandbox verbatim. A
# denylist of secrets has to enumerate every naming convention anyone will
# ever use; an allowlist has to enumerate what a Python subprocess needs.
_ENV_ALLOWED = ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")


def _clean_env(workspace, extra_paths=()):
    env = {k: v for k, v in os.environ.items()
           if k in _ENV_ALLOWED and not k.startswith(_SCRUBBED_PREFIXES)}
    env["PATH"] = "/usr/bin:/bin"
    env["HOME"] = workspace
    env["TMPDIR"] = workspace
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # `-I` implied `-P` (no script directory on sys.path) AND `-s` (no user
    # site-packages) AND `-E` (PYTHONPATH ignored). Between them the
    # documented "one script plus whatever it imports" could not import
    # anything -- `import lib` with lib.py right beside it raised
    # ModuleNotFoundError, which `judge` then graded as a REFUTATION of the
    # hypothesis under test. The env is allowlisted above, so `-E` was
    # buying nothing that the allowlist does not already buy; the commands
    # now use `-s` alone and this path is what they see.
    env["PYTHONPATH"] = os.pathsep.join(
        [workspace] + [p for p in (extra_paths or []) if p])
    # Not a network sandbox -- a speed bump that makes an accidental fetch
    # fail fast and loudly instead of hanging until the timeout.
    env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:1"
    env["NO_PROXY"] = ""
    return env


def _tail(text, limit=MAX_OUTPUT):
    if isinstance(text, bytes):
        # CPython builds TimeoutExpired.stdout/.stderr with b"".join() even
        # under text=True, so the timeout path handed bytes to a caller
        # expecting str. `judge`'s `"hello" in result["stdout"]` raised
        # TypeError, and so did `_observation_text` -- rule 3 ("a failure is
        # data, never an error") broken by the exact path that exists to
        # serve it. The existing test only timed out a SILENT loop, where
        # b"" or "" collapses to "" and hides it.
        text = text.decode("utf-8", "replace")
    text = text or ""
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


def _limits(timeout=DEFAULT_TIMEOUT):
    """Applied in the child between fork and exec."""
    if resource is None:
        return None
    cpu = max(1, int(math.ceil(timeout)) + 1)

    # NO RLIMIT_NPROC, DELIBERATELY, AND THIS IS THE MEASUREMENT THAT DECIDED
    # IT. The limit counts every TASK (threads included) belonging to the real
    # UID across the whole machine, not the descendants of this run. A flat
    # ceiling of 64 made the sandbox's first fork die with BlockingIOError, so
    # any experiment that shelled out was graded a REFUTATION of the
    # hypothesis under test — a harness failure wearing the clothes of a
    # finding, which is the worst outcome this module has. Deriving the
    # ceiling from the host's current usage does not rescue it either: /proc
    # enumerates thread-group leaders while the kernel counts tasks, so the
    # derived number is wrong in the unsafe direction and races anything else
    # the user starts. A per-UID knob cannot bound a per-run sandbox without
    # a UID of its own.
    #
    # What does bound a fork bomb here: the wall-clock timeout and the
    # process-GROUP kill below, which reap the whole tree rather than the
    # direct child. That is a time bound, not a count bound. Saying so is the
    # point — a guard whose limits are undocumented gets trusted past them.
    def apply():
        os.setsid()          # its own process group, so the whole tree dies
        for what, limit in (
                (resource.RLIMIT_AS, MAX_MEMORY_BYTES),
                # A CPU ceiling as well as a wall clock: a tight loop that
                # ignores SIGTERM still gets SIGXCPU.
                (resource.RLIMIT_CPU, cpu),
                (resource.RLIMIT_FSIZE, MAX_FILE_BYTES),
                (resource.RLIMIT_NOFILE, MAX_OPEN_FILES)):
            try:
                resource.setrlimit(what, (limit, limit))
            except (ValueError, OSError):
                pass
    return apply


def _drain(stream, sink, cap):
    """Read one pipe, keeping only the last `cap` characters.

    `capture_output=True` buffered the WHOLE stream in the parent before
    MAX_OUTPUT ever got a chance to truncate it: a script doing
    `while True: sys.stdout.write('x'*4096)` took the parent from 12 MB RSS
    to 3,181 MB and returned 11.1s after a 5s deadline. The cap was being
    applied to a string that had already been materialised in full — which
    means model-written code could OOM-kill the process hosting the
    assistant. Bound the read, not the result."""
    try:
        # read1, not read: `read(n)` on a buffered stream blocks until it has
        # n bytes OR end-of-file, so a short line sat unread whenever anything
        # still held the pipe open. read1 returns whatever has arrived.
        while True:
            chunk = stream.read1(8192)
            if not chunk:
                break
            sink.append(chunk)
            # Keep a little more than the cap so the tail is exact.
            total = sum(len(c) for c in sink)
            while total > cap * 2 and len(sink) > 1:
                total -= len(sink.pop(0))
    except (ValueError, OSError):
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_tree(proc):
    """SIGKILL the whole process group, not just the direct child.

    Without it, a script that spawns a background sleeper and spins left the
    GRANDCHILD alive after the timeout — burning CPU and writing into a
    workspace `shutil.rmtree` had already deleted — and, because that orphan
    held the stdout pipe open, a script that printed "ALL TESTS PASSED" and
    exited 0 was reported as `timed_out` and graded `refuted`. Any experiment
    leaving a server or worker running was silently marked a failure."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


MAX_COLLECT_BYTES = 200_000


def run(files, command, *, timeout=DEFAULT_TIMEOUT, stdin="",
        import_paths=(), collect=(), cwd="", cwd_importable=False):
    """Write `files` into a throwaway workspace, run `command`, report.

    `files` is {relative_path: contents}. `command` is an argv list. Returns a
    dict that is the OBSERVATION an experiment is judged against:

        ok          the command exited 0 and did not time out
        exit_code   None when it timed out
        stdout      tail-truncated
        stderr      tail-truncated
        truncated   whether either stream lost its head
        seconds     wall clock
        timed_out   distinguished from a non-zero exit, because "it never
                    finished" and "it finished and failed" are different
                    findings and a caller that conflates them will keep
                    re-running the loop
        files_after {path: contents} for each path named in `collect`

    `cwd` IS A SUBDIRECTORY TO RUN IN, and it exists because a project is
    not a loose pile of files. An unpacked repository sits at
    `<name>/<name>/`, and its own code resolves paths against the process
    cwd — the Engine's `app.py` mounts `StaticFiles(directory="static")` at
    import time — so a suite invoked from the workspace root fails on every
    module with `Directory 'static' does not exist` no matter how completely
    the files were delivered. Measured: 24 modules, 226 tests. The only way
    to say "run in that directory" was `sh -c "cd X && ..."`, which changes
    the argv to `sh` and so loses the pytest provisioning keyed on it — two
    workarounds cancelling out, and the assistant spent four turns there.
    Refused if it escapes the workspace, like every other path here.

    `collect` IS THE WHOLE REASON A RUN CAN BE ASKED ABOUT ITS EFFECTS. The
    workspace is destroyed in the `finally` below, so before this existed the
    only observable a program had was what it printed — "the patch applied and
    the file now reads X" could only be checked by having the program print
    the file back, which tests the print statement as much as the patch. A
    path that the run did not produce comes back absent from `files_after`
    rather than empty: absent and empty are different findings, and a
    predicate over the wrong one is a prediction about nothing.

    Never raises for ordinary failure. A crash, a timeout and a missing
    interpreter are all RESULTS -- an experiment that blows up is data, and a
    harness that raises on it makes the assistant afraid of its own tools.
    """
    # PYTEST SUPPORT WAS ATTACHED TO A FUNCTION NOBODY ON THE LIVE PATH
    # CALLED. `run_pytest` added the interpreter's real site-packages and
    # stamped the harness — and `coding.run_experiment` calls `run` directly,
    # so an experiment that invoked pytest by naming it in `command` got
    # neither. Measured: "No module named pytest", exit 1, every time; and
    # `_PYTEST_HARNESS_EXITS` unreachable, its unit tests passing only because
    # they construct the key themselves.
    #
    # Folded HERE, where the command is known, rather than left to each
    # caller to remember (AGENTS.md). `run_pytest` now only builds an argv;
    # everything that makes pytest work happens once, below.
    # Clamped HERE rather than at each caller, so the ceiling cannot be
    # bypassed by a caller who forgets it and cannot be forgotten by one who
    # should apply it (AGENTS.md: a guard that must be remembered will be
    # forgotten). A request above the ceiling is silently lowered rather than
    # refused — the run still happens, and the timeout it gets is reported in
    # `seconds` either way.
    timeout = max(1.0, min(float(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    harness = _harness_of(command)
    # WHERE PYTEST LIVES MUST NOT DEPEND ON HOW THE COMMAND SPELLS IT.
    #
    # This used to be inside `if harness == "pytest"`, which is the same guard
    # one layer too shallow: it reads the OUTER argv, and the way you run a
    # suite and capture its output is to write a program that shells out to
    # pytest itself. That program's argv is `python3 -s main.py`, so the fold
    # did not fire, the nested interpreter inherited a PYTHONPATH holding only
    # the workspace, and the run came back `No module named pytest` — the same
    # sentence, measured the same way, that the fold was written to end.
    #
    # Worse than a repeat: the outer process exited 0, because the program
    # caught the failure and printed it as data. So neither harness guard in
    # `coding.judge` could see it — they read the outer exit code and the
    # outer stderr — and a broken tool was graded a REFUTATION of the
    # hypothesis. Rule 4, defeated by nesting rather than by argument.
    #
    # `harness` stays keyed on the command, because that decides how the
    # OUTER run is GRADED and a nested pytest is not the outer harness. Only
    # the path is unconditional.
    import_paths = list(import_paths or []) + _pytest_path()
    workspace = tempfile.mkdtemp(prefix="assistant-exp-")
    try:
        for relative, contents in (files or {}).items():
            # Never outside the workspace: a generated path is untrusted input
            # like any other.
            target = os.path.normpath(os.path.join(workspace, relative))
            if not target.startswith(workspace + os.sep):
                return {"ok": False, "exit_code": None, "stdout": "",
                        "stderr": f"refused to write outside the workspace: "
                                  f"{relative!r}",
                        "truncated": False, "seconds": 0.0,
                        "timed_out": False}
            # A generated `files` dict is untrusted input all the way down,
            # not just in its paths. {"a": "x", "a/b": "y"} raised
            # FileExistsError and {"a.txt": None} raised TypeError, both
            # straight out of `run` — plausible model payloads escaping as
            # exceptions instead of coming back as observations, which is
            # rule 3 again.
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                # BYTES ARRIVE AS BYTES. A text-only writer turned a SQLite
                # database into the literal characters `b'SQLite format 3...'`,
                # so a suite that opens one failed at its first query and the
                # experiment recorded a fault in the project under test.
                if isinstance(contents, (bytes, bytearray)):
                    with open(target, "wb") as handle:
                        handle.write(contents)
                    continue
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write("" if contents is None else str(contents))
            except (OSError, TypeError, ValueError) as exc:
                return {"ok": False, "exit_code": None, "stdout": "",
                        "stderr": f"could not write {relative!r}: {exc}",
                        "truncated": False, "seconds": 0.0,
                        "timed_out": False, "harness": harness}
        # Resolved AFTER the files are written, so naming a directory the run
        # itself creates works, and refused rather than silently ignored: a
        # cwd that quietly falls back to the root reproduces the exact failure
        # this argument exists to end, with the argument set.
        run_dir = os.path.normpath(os.path.join(workspace, str(cwd or "")))
        if not (run_dir == workspace or run_dir.startswith(workspace + os.sep)):
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"refused to run outside the workspace: "
                              f"{cwd!r}",
                    "truncated": False, "seconds": 0.0, "timed_out": False,
                    "harness": harness}
        if not os.path.isdir(run_dir):
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"no such directory in the workspace: {cwd!r}",
                    "truncated": False, "seconds": 0.0, "timed_out": False,
                    "harness": harness}
        # ONLY WHEN WE MOVED THE PROGRAM. A run whose program we synthesised
        # at the payload root and then pointed at from a `cwd` below cannot
        # import that directory, because Python puts the SCRIPT's directory on
        # sys.path and never the working one — so `source` plus `cwd` came
        # back `ModuleNotFoundError: No module named 'db'` from a run whose
        # whole purpose was to read `db.py`.
        #
        # BUT A CALLER'S OWN COMMAND MUST SEE THE REAL ENVIRONMENT. Adding
        # `cwd` unconditionally made the sandbox unlike the machine it is
        # standing in for, and a real project noticed within the hour:
        #   ROOT = Path(__file__).resolve().parents[1]
        #   if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
        # is a guard that inserts the project root at the FRONT when it is
        # absent. Put it on PYTHONPATH and the guard sees it present, skips
        # the insert, and leaves the script's own directory first — where the
        # import finds the script instead of the module it names. One
        # previously-passing test went red, in a suite that had just been
        # repaired to zero, on a change that was supposed to be invisible.
        # A developer standing in that directory does not have it on
        # PYTHONPATH, so neither should a run that is imitating them.
        paths = list(import_paths or [])
        if cwd_importable and run_dir != workspace:
            paths.insert(0, run_dir)
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                command, cwd=run_dir,
                env=_clean_env(workspace, paths),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=_limits(timeout))
        except FileNotFoundError as missing:
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"no such interpreter: {missing}",
                    "truncated": False, "seconds": 0.0, "timed_out": False,
                    "harness": harness}
        except OSError as exc:
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"could not start the command: {exc}",
                    "truncated": False, "seconds": 0.0, "timed_out": False,
                    "harness": harness}
        out_chunks, err_chunks = [], []
        readers = [
            threading.Thread(target=_drain,
                             args=(proc.stdout, out_chunks, MAX_OUTPUT),
                             daemon=True),
            threading.Thread(target=_drain,
                             args=(proc.stderr, err_chunks, MAX_OUTPUT),
                             daemon=True)]
        for reader in readers:
            reader.start()
        try:
            proc.stdin.write((stdin or "").encode())
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        elapsed = time.perf_counter() - started
        # Kill the group on EVERY path, not only on timeout. A child that
        # exits cleanly can still leave a background worker behind, and that
        # orphan (a) survives the rmtree of the workspace it is writing into
        # and (b) holds the stdout pipe open, which used to hang the readers
        # and lose the output of a run that had actually succeeded.
        _kill_tree(proc)
        # A short join, then move on regardless: nothing the sandbox spawns
        # gets to hold the ASSISTANT here.
        for reader in readers:
            reader.join(timeout=2)
        out, cut_out = _tail(b"".join(out_chunks))
        err, cut_err = _tail(b"".join(err_chunks))
        # Read back BEFORE the rmtree below, and after the tree is dead, so a
        # surviving writer cannot change what is reported.
        # Relative to where the program RAN, because that is where a relative
        # write lands. Identical to the old behaviour when `cwd` is unset.
        after = _collect(run_dir, collect, workspace)
        if timed_out:
            return {"ok": False, "exit_code": None, "stdout": out,
                    "stderr": (err
                               + f"\n[timed out after {timeout}s]").strip(),
                    "truncated": True, "seconds": round(elapsed, 3),
                    "timed_out": True, "files_after": after,
                    "harness": harness}
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
                "stdout": out, "stderr": err,
                "truncated": cut_out or cut_err,
                "seconds": round(elapsed, 3), "timed_out": False,
                "files_after": after, "harness": harness}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _collect(run_dir, paths, workspace=None):
    """Read named files out of the workspace before it is destroyed.

    Every failure to read is silent ABSENCE rather than an exception or an
    empty string, for rule 3's reason: a run that crashed before writing its
    output file is data, and a harness that raises on it teaches the loop to
    avoid the interesting cases. The path guard is the same one the write side
    uses — a path in a prediction is untrusted input exactly like a path in
    `files`."""
    out = {}
    bound = workspace if workspace is not None else run_dir
    for relative in (paths or ()):
        target = os.path.normpath(os.path.join(run_dir, str(relative)))
        if not target.startswith(bound + os.sep):
            continue
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                out[str(relative)] = fh.read(MAX_COLLECT_BYTES)
        except (OSError, ValueError):
            continue
    return out


def run_python(source, *, extra_files=None, timeout=DEFAULT_TIMEOUT):
    """The common case: one script, run once."""
    files = dict(extra_files or {})
    files["main.py"] = source
    # `-E -s` rather than `-I`: same isolation from the host's environment
    # and user site-packages, WITHOUT `-P`, which stripped the script's own
    # directory from sys.path and made `extra_files` unimportable.
    return run(files, [sys.executable, "-s", "main.py"], timeout=timeout)


def run_pytest(files, *, timeout=DEFAULT_TIMEOUT):
    """A test file plus whatever it imports.

    Tests are the sharpest instrument this module offers, because a test
    states a PREDICTION before it is run -- which is the difference between an
    experiment and a demonstration.

    Nothing but the argv happens here now. Making pytest importable and
    stamping the harness used to live in this function, which meant they
    applied only to callers who knew to use it -- and the live experiment path
    did not. Both moved into `run`, keyed on the command itself.
    """
    return run(files, [sys.executable, "-s", "-m", "pytest", "-q",
                       "--no-header", "-p", "no:cacheprovider"],
               timeout=timeout)


def _harness_of(command):
    """Which known test harness this argv invokes, if any.

    `-m pytest` and a bare `pytest` are the two spellings that reach here. A
    harness recognised by NAME rather than declared by the caller is what lets
    a command the model wrote get the same treatment as one this module built
    -- and the model writes most of them."""
    argv = [str(a) for a in (command or [])]
    for i, arg in enumerate(argv):
        if arg == "-m" and i + 1 < len(argv) and argv[i + 1] == "pytest":
            return "pytest"
        if os.path.basename(arg) == "pytest" and i == 0:
            return "pytest"
    return ""


def _pytest_path():
    """Where pytest lives on THIS host, or nothing. Absent pytest stays an
    inconclusive harness failure, never a refutation."""
    try:
        import pytest
        return [os.path.dirname(os.path.dirname(pytest.__file__))]
    except Exception:
        return []
