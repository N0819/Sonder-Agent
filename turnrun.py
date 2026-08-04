# turnrun.py — a turn as an observable, haltable job.
#
# `pipeline.run_turn` is one long synchronous call, and for most of this
# project's life that was right: a turn is atomic, and the HTTP layer had
# nothing useful to say while it ran. Two requirements broke that. Watching the
# assistant think — which memories surfaced, what it pondered, what it searched
# — needs the stages to be visible AS they happen rather than summarised after.
# And a halt button needs something to halt.
#
# This module holds the registry and nothing else. The pipeline stays free of
# HTTP, the routes stay free of threading, and the rule about where a turn may
# be interrupted lives in one place:
#
# HALTING IS A STAGE BOUNDARY, NEVER A POINT INSIDE THE COMMIT.
#
# Every durable mutation of a turn happens in one transaction (`pipeline`
# stage 5, `db.transaction`). A halt that landed inside it would either roll
# back a partially-written turn or, worse, leave the ordinal consumed with
# memories half-committed — the exact failure `run_turn` reserves the ordinal
# early to avoid. So a halt is CHECKED between stages and REFUSED once the
# commit begins. The user-visible consequence is honest and easy to state: a
# halted turn wrote nothing, and a turn that reached the commit finishes.

import json
import threading
import time
import uuid

# Finished runs stay readable so a client that reconnects can still collect the
# result and the reasoning trail. Bounded because this is a registry, not a
# second copy of the turns table — the durable record is in the database.
_MAX_RETAINED = 32

_runs = {}
_order = []
_registry_lock = threading.Lock()


class TurnHalted(Exception):
    """Raised inside the pipeline when the user halted at a stage boundary.

    An exception rather than a return value because it has to unwind out of
    the research loop, which is several frames down and knows nothing about
    turns being cancellable."""


class TurnRun:
    """One in-flight or finished turn."""

    def __init__(self, run_id, text, session_id):
        self.id = run_id
        self.text = text
        self.session_id = session_id
        self.events = []
        self.status = "running"      # running | done | halted | failed
        self.result = None
        self.error = ""
        self.started = time.time()
        # `_committing` is the latch that makes the rule above enforceable:
        # once the pipeline says it has entered the commit, a halt request is
        # answered "too late" instead of being honoured.
        self._committing = False
        self._halt = False
        # Child processes the turn is currently blocked on. A halt that only
        # sets a flag is checked at the NEXT stage boundary, and the stage
        # boundaries are on either side of a model call — measured at 47s for
        # the Claude Code CLI. An interrupt that takes 47 seconds is not an
        # interrupt, so halting also kills what the turn is waiting on, which
        # makes the flag observable almost immediately.
        self._procs = set()
        self._cond = threading.Condition()

    # -- producer side, called from the worker thread --

    def emit(self, stage, **detail):
        with self._cond:
            self.events.append({"i": len(self.events), "stage": stage,
                                "t": round(time.time() - self.started, 3),
                                **detail})
            self._cond.notify_all()

    def enter_commit(self):
        with self._cond:
            self._committing = True

    def halted(self):
        """The pipeline's checkpoint. Raises so the caller cannot forget to
        act on a False-y return — a halt that must be remembered to be honoured
        is a halt that will eventually be ignored."""
        with self._cond:
            if self._halt and not self._committing:
                raise TurnHalted()
        return False

    def register_process(self, proc):
        """Track a child the turn is blocked on, so a halt can end the wait.

        A halt that arrived while the process was being spawned would
        otherwise be missed entirely — the flag was set before the process
        existed, so nothing killed it — so a run already halted kills on
        registration."""
        with self._cond:
            if self._halt and not self._committing:
                self._kill(proc)
                return
            self._procs.add(proc)

    def unregister_process(self, proc):
        with self._cond:
            self._procs.discard(proc)

    @staticmethod
    def _kill(proc):
        try:
            proc.terminate()
        except Exception:
            pass

    def finish(self, status, result=None, error=""):
        with self._cond:
            self.status = status
            self.result = result
            self.error = error
            self._cond.notify_all()

    # -- consumer side, called from request threads --

    def request_halt(self):
        """Returns what actually happened, rather than a bare acknowledgement:
        'halting' or 'too_late'. A button that always says "stopped" while the
        turn keeps running is worse than no button."""
        with self._cond:
            if self.status != "running":
                return "not_running"
            if self._committing:
                return "too_late"
            self._halt = True
            for proc in list(self._procs):
                self._kill(proc)
            self._procs.clear()
            self._cond.notify_all()
            return "halting"

    def follow(self, timeout=600.0):
        """Yield events from the beginning, then live ones until the run ends.

        Replays from index 0 so a client that connects a moment after POST
        misses nothing — the recall stage in particular is usually over before
        the browser has opened the stream."""
        sent = 0
        deadline = time.time() + timeout
        while True:
            with self._cond:
                while sent >= len(self.events) and self.status == "running":
                    if not self._cond.wait(timeout=1.0):
                        if time.time() > deadline:
                            return
                batch = self.events[sent:]
                sent += len(batch)
                finished = self.status != "running"
            for event in batch:
                yield event
            if finished and sent >= len(self.events):
                with self._cond:
                    yield {"i": sent, "stage": "end", "status": self.status,
                           "error": self.error,
                           "result": self.result}
                return


# The run that owns THIS thread.
#
# Thread-local rather than a parameter threaded through `chat_complete`,
# because the provider seam is reached from the respond stage, the research
# loop, the consolidator and the subagents, and adding a `run=` argument to
# every one of them would put the same forgettable plumbing in five places to
# serve one feature. A turn already owns its worker thread — `db` relies on
# the same fact for its per-thread connection — so the thread IS the turn.
_current = threading.local()


def bind(run):
    _current.run = run


def current():
    """The run on this thread, or None. Providers use it to register the
    child they are about to block on."""
    return getattr(_current, "run", None)


def create(text, session_id):
    run = TurnRun(uuid.uuid4().hex[:16], text, session_id)
    with _registry_lock:
        _runs[run.id] = run
        _order.append(run.id)
        while len(_order) > _MAX_RETAINED:
            _runs.pop(_order.pop(0), None)
    return run


def get(run_id):
    with _registry_lock:
        return _runs.get(run_id)


def start(run, target):
    """Run `target(run)` on a worker thread.

    `db` keeps one sqlite connection per thread, so a turn on a worker thread
    gets its own connection and the commit discipline is unchanged."""
    def body():
        bind(run)
        try:
            run.emit("start", text=run.text)
            result = target(run)
            run.emit("done")
            run.finish("done", result=result)
        except TurnHalted:
            run.emit("halted")
            run.finish("halted")
        except Exception as exc:      # a dead worker must not hang the stream
            run.emit("failed", error=str(exc)[:300])
            run.finish("failed", error=str(exc)[:300])

    thread = threading.Thread(target=body, name=f"turn-{run.id}", daemon=True)
    thread.start()
    return run


def sse(run):
    """Server-sent events for one run."""
    for event in run.follow():
        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
