"""A zombie is not a slow run.

`enginelab.runs` decides whether a lab turn is still going by asking whether
its pid is alive, and the obvious way to ask — `os.kill(pid, 0)` — answers YES
for a zombie. A child that has exited and has not been reaped still owns its
pid, and nothing here ever reaps: the turn is launched with Popen, the poll
reads a pid FILE, and the Popen object is long gone by the time anyone asks.

So a crashed run reads `running` forever, which is the precise failure the
`runs()` docstring promises not to have.

Live: lab `stairs`, run 0003. The child exited after 1.3 seconds of CPU and the
poll reported `state: "running", log_bytes: 228` on four consecutive polls
across two turns. The assistant doing the polling wrote that it could not tell
a long model call from a wedged run "from the poll output alone" — a correct
reading of a lying instrument.
"""

from __future__ import annotations

import os
import time

import enginelab


def _make_zombie():
    """A real one. Simulating this with a mock would test the mock -- the whole
    defect is that a genuine zombie answers signal 0 without error.
    """
    pid = os.fork()
    if pid == 0:            # child
        os._exit(0)
    for _ in range(200):    # let it die, do NOT reap it
        try:
            with open("/proc/%d/stat" % pid) as fh:
                if fh.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return pid
        except OSError:
            break
        time.sleep(0.01)
    return pid


def test_signal_zero_really_does_lie_about_a_zombie():
    """The premise, pinned. If this ever fails the platform changed and the
    guard below is solving a problem that no longer exists.
    """
    pid = _make_zombie()
    try:
        os.kill(pid, 0)     # no exception: the old check called this "alive"
    except OSError:
        os.waitpid(pid, os.WNOHANG)
        raise AssertionError("signal 0 now raises for a zombie; test is stale")
    os.waitpid(pid, os.WNOHANG)


def test_a_zombie_is_not_alive():
    """THE REPRODUCTION."""
    pid = _make_zombie()
    assert enginelab._pid_alive(pid) is False


def test_the_zombie_is_reaped_on_the_way_past():
    """A dead run should stop lingering in the process table too, not only in
    the report -- otherwise every crashed lab turn leaks an entry for the life
    of the server.
    """
    pid = _make_zombie()
    enginelab._pid_alive(pid)
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return              # already reaped, which is the point
    raise AssertionError("the zombie was still there to reap")


def test_a_live_process_is_still_alive():
    """The half that must not break: this process is running, and calling it
    dead would report every healthy lab turn as a failure.
    """
    assert enginelab._pid_alive(os.getpid()) is True


def test_a_pid_that_does_not_exist_is_not_alive():
    for pid in (999_999, 2_147_483_646):
        if not os.path.exists("/proc/%d" % pid):
            assert enginelab._pid_alive(pid) is False
            return


def test_an_unreadable_stat_does_not_kill_a_live_run():
    """Fail-open on the unknown. Calling a run dead because /proc could not be
    read would turn a permissions quirk into a lost turn, and the cost is
    asymmetric: a run wrongly called dead is abandoned, a run wrongly called
    alive is polled again.
    """
    import builtins

    real_open = builtins.open

    def refuse(path, *a, **k):
        if str(path).startswith("/proc/"):
            raise OSError("nope")
        return real_open(path, *a, **k)

    builtins.open = refuse
    try:
        assert enginelab._pid_alive(os.getpid()) is True
    finally:
        builtins.open = real_open
