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


def run(files, command, *, timeout=DEFAULT_TIMEOUT, stdin="",
        import_paths=()):
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

    Never raises for ordinary failure. A crash, a timeout and a missing
    interpreter are all RESULTS -- an experiment that blows up is data, and a
    harness that raises on it makes the assistant afraid of its own tools.
    """
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
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write("" if contents is None else str(contents))
            except (OSError, TypeError, ValueError) as exc:
                return {"ok": False, "exit_code": None, "stdout": "",
                        "stderr": f"could not write {relative!r}: {exc}",
                        "truncated": False, "seconds": 0.0,
                        "timed_out": False}
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                command, cwd=workspace,
                env=_clean_env(workspace, import_paths),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=_limits(timeout))
        except FileNotFoundError as missing:
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"no such interpreter: {missing}",
                    "truncated": False, "seconds": 0.0, "timed_out": False}
        except OSError as exc:
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"could not start the command: {exc}",
                    "truncated": False, "seconds": 0.0, "timed_out": False}
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
        if timed_out:
            return {"ok": False, "exit_code": None, "stdout": out,
                    "stderr": (err
                               + f"\n[timed out after {timeout}s]").strip(),
                    "truncated": True, "seconds": round(elapsed, 3),
                    "timed_out": True}
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
                "stdout": out, "stderr": err,
                "truncated": cut_out or cut_err,
                "seconds": round(elapsed, 3), "timed_out": False}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


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
    """
    # `-I` implied `-s`, which hides user site-packages -- where pytest
    # actually lives on an ordinary install. `run_pytest` could not succeed
    # AT ALL, and "No module named pytest" exits non-zero, so `judge` graded
    # the sharpest instrument this module offers as a REFUTATION of whatever
    # hypothesis it was testing. Keeping `-s` (the sandbox must not inherit
    # the host's user site wholesale) while naming pytest's own directory
    # explicitly is what makes it importable without reopening that door --
    # and it has to be named explicitly because HOME points at the
    # workspace, so the interpreter cannot find the real one by itself.
    return run(files, [sys.executable, "-s", "-m", "pytest", "-q",
                       "--no-header", "-p", "no:cacheprovider"],
               timeout=timeout, import_paths=_pytest_path())


def _pytest_path():
    """Where pytest lives on THIS host, or nothing. Absent pytest stays an
    inconclusive harness failure, never a refutation."""
    try:
        import pytest
        return [os.path.dirname(os.path.dirname(pytest.__file__))]
    except Exception:
        return []
