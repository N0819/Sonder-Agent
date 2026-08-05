# providers.py — LLM chat and embeddings, with deterministic degradation.
#
# A cut-down port of the engine's providers.py. Two roles: `chat` and
# `embeddings`, each an OpenAI-compatible endpoint configured by environment
# (or left unconfigured). Two properties carried over on purpose:
#
# 1. **Embedding failure degrades to cheap_embed, loudly.** cheap_embed is a
#    signed hashing trick over character 3/4-grams — a fuzzy LEXICAL signature,
#    not a semantic vector. The engine measured it at 0% recall on
#    vocabulary-disjoint paraphrases (median rank 228/441, i.e. random), so the
#    fallback keeps retrieval alive on shared vocabulary and nothing else.
#    Every EmbeddingBatch carries model_key + dimensions because a vector can
#    only be compared with one from the same model: a mismatched row scores
#    0.0 forever, silently, and the model stamp is what lets retrieval count
#    and announce the stranding instead of quietly splitting the bank.
#
# 2. **Tests never touch the network.** set_chat_stub installs a deterministic
#    responder; embeddings fall back to cheap_embed when no provider is
#    configured, which is also what makes the whole test tier runnable
#    offline. This mirrors the engine's rule that the deterministic floor
#    must not depend on a model cooperating — or being reachable.

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import http.client
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field

import turnrun

import numpy as np

import config

REQUEST_TIMEOUT = float(os.environ.get("ASSISTANT_HTTP_TIMEOUT", "60"))

# GENERATION IS NOT A LOOKUP, AND ONE TIMEOUT CANNOT SERVE BOTH. Embeddings
# return in well under a second; a respond round sends a 17k-character system
# prompt plus an accumulating payload and may generate thousands of tokens
# before the first byte comes back. They shared the 60s ceiling, which was
# never exercised while the Claude Code CLI was the provider — the CLI is a
# subprocess and never touches urllib — so the first HTTP provider configured
# lost turns to a limit chosen for the other role.
#
# A timeout that fires on a healthy-but-slow provider is indistinguishable
# from a broken one from the operator's side, and the recovery is opposite:
# wait, versus go and fix something.
CHAT_TIMEOUT = float(os.environ.get("ASSISTANT_CHAT_TIMEOUT", "300"))


@dataclass
class EmbeddingBatch:
    vectors: list = field(default_factory=list)
    model_key: str = "cheap:crc32:256"
    dimensions: int = 256
    fallback: bool = False
    error: str = ""


def cheap_embed(text, dim=256):
    v = np.zeros(dim, dtype=np.float32)
    t = " " + (text or "").lower() + " "
    for n in (3, 4):
        for i in range(max(len(t) - n, 0)):
            h = zlib.crc32(t[i:i + n].encode("utf-8", "ignore"))
            v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def _embed_config():
    """Where embeddings come from — read from SETTINGS, not from the
    environment behind the settings' back.

    This function used to read `os.environ` directly while the Settings tab
    happily displayed and saved `embed_base` / `embed_model` fields that
    nothing consulted. Configuring embeddings in the UI did nothing at all,
    silently, and retrieval stayed on the lexical fallback while the page
    said otherwise. That is the persona-drive scar exactly — a field nothing
    reads defeats even the check built to catch an empty one — reintroduced
    in the same commit that fixed the original.

    The embeddings provider is deliberately INDEPENDENT of the chat provider:
    the Claude Code CLI has no embeddings endpoint at all, so an assistant
    talking through it still needs somewhere else to vectorise. They are two
    roles, not one setting."""
    cfg = config.get_config()
    base = str(cfg.get("embed_base") or "").rstrip("/")
    model = str(cfg.get("embed_model") or "")
    key = config.secret_for("embed_key_env") or os.environ.get(
        "ASSISTANT_API_KEY", "")
    return (base, model, key) if base and model else None


def embed_texts_meta(texts) -> EmbeddingBatch:
    texts = [str(t or "") for t in texts]
    if not texts:
        return EmbeddingBatch(vectors=[])
    cfg = _embed_config()
    if cfg:
        base, model, key = cfg
        try:
            req = urllib.request.Request(
                base + "/embeddings",
                data=json.dumps({"model": model, "input": texts}).encode(),
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {key}"} if key else {})})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                data = json.loads(r.read().decode())
            items = sorted(data.get("data") or [],
                           key=lambda item: item.get("index", 0))
            if len(items) != len(texts):
                raise RuntimeError("embedding provider returned wrong count")
            vectors, dim = [], None
            for item in items:
                vec = np.asarray(item["embedding"], dtype=np.float32)
                dim = dim or len(vec)
                nrm = np.linalg.norm(vec)
                # L2-normalise HERE so ranking can use a plain dot product —
                # the engine measured 4.4x from dropping the per-comparison
                # norms, and it only works if every producer normalises.
                vectors.append(vec / nrm if nrm > 0 else vec)
            return EmbeddingBatch(vectors=vectors,
                                  model_key=f"api:{model}",
                                  dimensions=dim or 0)
        except Exception as exc:  # degrade, and say why — verbatim, because
            # "no embeddings provider" is the wrong sentence when one IS
            # configured and is simply not an embeddings model.
            detail = str(exc)[:300]
            # A bare "HTTP Error 401: Unauthorized" names neither the field
            # nor the fix, and the two causes need opposite actions: no key at
            # all is a settings problem, a rejected key is a credential
            # problem. Distinguishing them here is the difference between a
            # message that ends the investigation and one that starts it.
            if "401" in detail or "403" in detail:
                detail += (
                    " — the embeddings key was rejected. It is currently "
                    f"coming from {config.secret_source('embed_key_env')
                                   or 'nowhere'}"
                    + ("; paste a key into the embeddings key field in "
                       "Settings" if not key else
                       "; check that key is correct and not revoked"))
            return EmbeddingBatch(
                vectors=[cheap_embed(t) for t in texts],
                fallback=True, error=detail[:400])
    return EmbeddingBatch(vectors=[cheap_embed(t) for t in texts])


# ---- Chat ----

_chat_stub = None


def set_chat_stub(fn):
    """Install a deterministic responder for tests: fn(system, user) -> str.
    Pass None to remove. The pipeline is written so that everything OUTSIDE
    the model call is deterministic; the stub is how tests prove that claim
    rather than assuming it."""
    global _chat_stub
    _chat_stub = fn


def chat_configured():
    if _chat_stub is not None:
        return True
    cfg = config.get_config()
    if cfg["chat_provider"] == config.PROVIDER_CLAUDE_CODE:
        return bool(shutil.which(cfg["claude_binary"] or "claude"))
    return bool(cfg["chat_base"] and cfg["chat_model"])


# GENEROUS BY DEFAULT, BECAUSE THE CEILING IS FREE UNTIL IT IS HIT AND THE
# WHOLE TURN WHEN IT IS. This was 2000, and the Claude Code CLI path ignores
# it entirely — so the default was never exercised for as long as the CLI was
# the provider, and the first HTTP provider to be configured lost a turn to a
# number nobody had chosen for it. A ceiling that must be overridden at each
# call site to be safe is a guard that must be remembered; the respond stage,
# which emits the largest output of any stage, was the one call site that
# never overrode it. Output tokens are billed as generated, not as budgeted,
# so a high ceiling costs nothing until it is needed.
DEFAULT_MAX_TOKENS = 8000


def chat_complete(system, user, *, temperature=0.4,
                  max_tokens=DEFAULT_MAX_TOKENS):
    """One chat completion. Raises RuntimeError with a legible reason when no
    provider is configured — the caller decides how to surface that; nothing
    here fabricates a reply."""
    if _chat_stub is not None:
        return _chat_stub(system, user)
    cfg = config.get_config()
    if cfg["chat_provider"] == config.PROVIDER_CLAUDE_CODE:
        return _claude_code_complete(cfg, system, user)
    base = str(cfg["chat_base"] or "").rstrip("/")
    model = cfg["chat_model"]
    key = config.secret_for("chat_key_env") or os.environ.get(
        "ASSISTANT_API_KEY", "")
    if not base or not model:
        raise RuntimeError(
            "no chat model configured: set a base URL and model in Settings, "
            "or switch the provider to the Claude Code CLI")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    data = _post_with_retry(req)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("chat provider returned no choices: %s"
                           % str(data)[:200])
    # TRUNCATION IS NOT A PARSE FAILURE, AND SAYING SO SENDS THE OPERATOR TO
    # THE WRONG SUBSYSTEM. `finish_reason` was never read: a reply that hit
    # max_tokens came back as invalid JSON, the pipeline dropped the WHOLE
    # turn's side channels (reply, remember marks, belief updates, research
    # request) behind "respond stage returned unparseable output", and the
    # user was shown "I have no language model configured" — which is false.
    # Fix the earliest stage where the data first becomes wrong: name the
    # real cause here, where the cause is still visible.
    if str(choices[0].get("finish_reason") or "") == "length":
        raise RuntimeError(
            f"model output was truncated at max_tokens={max_tokens}; the "
            "reply and every side channel in it were lost. Raise max_tokens "
            "or shorten the payload.")
    return str((choices[0].get("message") or {}).get("content") or "")


# One blip must not cost a whole turn. Bounded, so a provider that is down
# stays down quickly rather than holding the request open for minutes.
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRY_ATTEMPTS = 3


def _post_with_retry(req, attempts=_RETRY_ATTEMPTS):
    """POST with bounded exponential backoff on transient failures.

    A single 429 or 502 used to raise straight out of `urlopen`, degrading
    the entire turn, and the provider's own error body — the part that says
    WHICH limit was hit — was never read, so the operator got only
    "HTTP Error 429: Too Many Requests"."""
    delay = 1.0
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT) as r:
                return json.loads(r.read().decode())
        # A READ TIMEOUT IS NOT A `URLError` AND SO ESCAPED THIS LOOP ENTIRELY.
        # `urlopen` raises the bare socket `TimeoutError` when the socket is
        # open and the body never finishes, so the one failure the backoff
        # exists for was the one it never caught — and what reached the turn
        # was "The read operation timed out", which names no provider, no
        # stage and no number the operator could change.
        except TimeoutError as exc:
            last = RuntimeError(
                f"chat provider did not respond within {CHAT_TIMEOUT:.0f}s. "
                "The request was sent and the connection stayed open, so this "
                "is a slow or overloaded provider rather than an unreachable "
                "one — raise ASSISTANT_CHAT_TIMEOUT, or use a faster model.")
            if attempt == attempts - 1:
                raise last from exc
            time.sleep(delay)
            delay *= 2
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            last = RuntimeError(f"chat provider HTTP {exc.code}: {body}")
            if exc.code not in _RETRY_STATUSES or attempt == attempts - 1:
                raise last from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers \
                else None
            try:
                wait = float(retry_after) if retry_after else delay
            except (TypeError, ValueError):
                wait = delay
            time.sleep(min(wait, 30.0))
            delay *= 2
        # THE FAMILY, NOT THE MEMBER THAT BIT LAST. This clause caught only
        # `URLError`, and the transport kept producing failures that are not
        # one: a read timeout is a bare `TimeoutError`, and a connection
        # dropped mid-response is `http.client.RemoteDisconnected`, because
        # urllib wraps what `request()` raises and not what `getresponse()`
        # does. Each escaped the backoff, unretried, and reached the turn as a
        # raw library string naming no provider and no stage.
        #
        # Enumerating them one at a time is a guard that must be remembered,
        # and it was forgotten twice. Everything the transport can raise is
        # `OSError` or `http.client.HTTPException`; the two cases that need
        # their own handling — a real HTTP status, and a timeout — are caught
        # above, so what reaches here is transient by construction.
        except (OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None) or exc
            last = RuntimeError(
                f"chat provider connection failed: {reason}. The connection "
                "closed before a reply arrived — usually an unreachable host, "
                "a rejected request (wrong model name or a payload the "
                "provider will not accept), or a proxy limit.")
            if attempt == attempts - 1:
                raise last from exc
            time.sleep(delay)
            delay *= 2
    raise last or RuntimeError("chat provider failed")


# ---- The Claude Code CLI as a chat provider ----
#
# Same contract as the HTTP path: (system, user) -> text. Everything around it
# — memory, grounding, the commit discipline — is unchanged, because the
# provider seam is the only thing that knows how a reply got composed.
#
# Three things are deliberate here:
#
# `cwd` is an EMPTY throwaway directory. Without `--bare` the CLI discovers
# CLAUDE.md by walking up from its working directory, so running it inside
# this repository would silently prepend this project's engineering
# instructions to the assistant's own system prompt — the assistant would
# start answering its user as though it were maintaining Sonder. An empty cwd
# has nothing to discover.
#
# `--bare` is NOT used, even though it would skip that discovery directly,
# because it also forces auth to ANTHROPIC_API_KEY only and never reads the
# OAuth login. Measured: `--bare` on a logged-in machine returns
# "Not logged in · Please run /login". Neutralising the cwd achieves the same
# isolation without breaking the common case.
#
# `--tools ""` because this role is COMPOSITION, not agency. The assistant's
# tools are its own — memory, research, the sandbox, the coding suite — and
# they are governed by the epistemics in this repository. A provider that
# quietly ran a second, ungoverned tool loop would put actions outside every
# guard here.
#
# It must be `--tools ""` and NOT `--allowed-tools ""`, which was the first
# attempt and is a filter rather than a switch: the model still had tools, so
# a payload mentioning a file made it reach for Read, that consumed a turn,
# and the run died at --max-turns with `stop_reason: "tool_use"`,
# `is_error: true` and an EMPTY result string. Measured, not guessed — the
# deep subagent failed all three of its turns this way while reporting the
# failure honestly enough that the cause was still invisible. `--tools ""`
# removes the tools rather than hiding them, so there is nothing to reach for.
#
# THE PAYLOAD GOES ON STDIN, NOT IN ARGV. Passing it as a trailing argument
# worked until a session had real material in it and then failed as
# `OSError: [Errno 7] Argument list too long` — which surfaced to the user as
# "respond stage failed" and named nothing anybody could act on. Linux caps a
# SINGLE argv entry at MAX_ARG_STRLEN (128 KiB), independent of the 2 MiB
# total, and the turn payload crosses it as soon as a codemap is in it:
# measured at 102 KB of codemap alone for a 115-file upload, before memory,
# beliefs or hypotheses were added.
#
# `claude -p` reads the prompt from stdin when no prompt argument is given, so
# the fix is a channel change rather than a size limit — verified against the
# installed CLI. The system prompt has no such channel in this version (there
# is no --system-prompt-file), so it stays in argv and is checked below
# instead: a limit that is going to be hit deserves its own sentence.
class _Completed:
    """What `subprocess.run` would have returned. Kept so the parsing below
    is unchanged by the switch to Popen."""

    def __init__(self, stdout, stderr, returncode):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


MAX_ARGV_ENTRY_BYTES = 131072       # Linux MAX_ARG_STRLEN, 32 pages
_SYSTEM_PROMPT_BUDGET = MAX_ARGV_ENTRY_BYTES - 8192   # headroom for the rest

# THE OLD TIMEOUT MEASURED THE WRONG THING. A single wall clock cannot tell a
# model that is thinking hard from one that has hung: at 180 seconds a
# deliberation round over a large payload was killed mid-answer and surfaced
# as "the respond stage failed", which reads as a fault and is not one.
#
# Two bounds instead, because there are two failures:
#   IDLE      the CLI has gone SILENT. Nothing is arriving, so nothing is
#             happening. This is the one that catches a hang, and it can be
#             short precisely because a working run resets it constantly.
#   CEILING   a total, for the livelocked case that keeps emitting forever.
#             Generous, because its only job is to be finite.
#
# Streaming is what makes the distinction available at all — with a single
# blocking read there is no signal between "started" and "finished".
_DEFAULT_IDLE_TIMEOUT = 120.0
_STREAM_POLL = 0.5


def _stream_text(event, seen):
    """The text carried by one stream-json event, and what kind it is.

    TWO SHAPES ARRIVE AND ONLY ONE MAY BE COUNTED. With
    `--include-partial-messages` the CLI emits token-level
    `content_block_delta`s AND, at the end, a complete `assistant` message
    holding the same text — so consuming both would show every answer twice.
    Deltas win when any have been seen; the `assistant` branch is the fallback
    for a CLI build that does not emit them.

    `thinking_delta` is kept separate rather than merged into the answer. It
    is the model's reasoning, it is not what it decided to say, and running
    the two together in one pane is how a draft gets read as a conclusion."""
    kind = event.get("type")
    if kind == "stream_event":
        inner = event.get("event") or {}
        if inner.get("type") != "content_block_delta":
            return "", ""
        delta = inner.get("delta") or {}
        if delta.get("type") == "text_delta":
            seen["saw_delta"] = True
            return delta.get("text") or "", "text"
        if delta.get("type") == "thinking_delta":
            seen["saw_delta"] = True
            return delta.get("thinking") or "", "thinking"
        return "", ""
    if kind == "assistant" and not seen["saw_delta"]:
        return "".join(
            part.get("text") or ""
            for part in (event.get("message") or {}).get("content") or []
            if isinstance(part, dict) and part.get("type") == "text"), "text"
    return "", ""


def _narrate_stream(line, run, seen):
    """Turn one stream-json event into visible progress, if it carries text.

    Best-effort by construction: an unparseable or unfamiliar event is skipped
    rather than raised on. The stream is a progress channel, and a provider
    that dies because it could not classify a notification would be trading
    the answer for the commentary."""
    if run is None:
        return
    try:
        event = json.loads(line)
    except (TypeError, ValueError):
        return
    text, kind = _stream_text(event, seen)
    if not text:
        return
    buffer = seen[kind]
    buffer["text"] += text
    now = time.monotonic()
    # Throttled, and the throttle is why this is affordable: the run keeps
    # every event it is given, so an unthrottled delta channel would trade a
    # timeout for an unbounded list.
    pending = len(buffer["text"]) - buffer["sent"]
    if now - seen["last_emit"] < 0.4 and pending < 400:
        return
    seen["last_emit"] = now
    if seen["events"] >= _MAX_STREAM_EVENTS:
        # Past the cap the deltas stop and a heartbeat takes over: the user
        # needs to know it is alive, not to read every token twice.
        run.emit("stream", kind=kind, chars=len(buffer["text"]),
                 truncated=True)
        buffer["sent"] = len(buffer["text"])
        return
    run.emit("stream", kind=kind, delta=buffer["text"][buffer["sent"]:],
             chars=len(buffer["text"]))
    buffer["sent"] = len(buffer["text"])
    seen["events"] += 1


_MAX_STREAM_EVENTS = 400


def _stream_cli(proc, user, run, idle_timeout, ceiling):
    """Feed the prompt in and read events out, concurrently.

    THE WRITE HAS TO BE CONCURRENT WITH THE READ. `communicate` handled that
    invisibly; doing it by hand does not get to skip it. A turn payload is
    routinely over 64 KiB and a pipe buffer is 64 KiB, so writing the whole
    prompt before reading anything deadlocks the moment the child fills the
    other direction — the parent blocked on write, the child blocked on
    write, neither draining the other.

    stderr gets its own drain for the same reason: a chatty child that fills
    the stderr pipe stops, and it stops without having produced the answer.

    Returns a `_Completed`. Raises with the bound that was hit, named, because
    "it timed out" sends someone to raise the wrong number."""
    lines, errs = [], []
    last_event = [time.monotonic()]
    seen = {"text": {"text": "", "sent": 0},
            "thinking": {"text": "", "sent": 0},
            "saw_delta": False, "last_emit": 0.0, "events": 0}

    def feed():
        try:
            proc.stdin.write(user)
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def read_events():
        try:
            for line in proc.stdout:
                last_event[0] = time.monotonic()
                lines.append(line)
                _narrate_stream(line, run, seen)
        except (ValueError, OSError):
            pass

    def read_errors():
        try:
            for line in proc.stderr:
                errs.append(line)
        except (ValueError, OSError):
            pass

    threads = [threading.Thread(target=fn, daemon=True)
               for fn in (feed, read_events, read_errors)]
    for thread in threads:
        thread.start()
    reader = threads[1]
    started = time.monotonic()
    while reader.is_alive():
        reader.join(timeout=_STREAM_POLL)
        now = time.monotonic()
        if now - last_event[0] > idle_timeout:
            proc.kill()
            raise RuntimeError(
                f"the Claude Code CLI went silent for {idle_timeout:.0f}s "
                f"(it had been running {now - started:.0f}s). Raise "
                "`claude_idle_timeout` in Settings if it is genuinely this "
                "slow to start.")
        if now - started > ceiling:
            proc.kill()
            raise RuntimeError(
                f"the Claude Code CLI passed the {ceiling:.0f}s ceiling while "
                "still producing output. Raise `claude_timeout` in Settings.")
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    for thread in threads:
        thread.join(timeout=2)
    # THE THROTTLE HOLDS THE TAIL. Whatever arrived since the last emit is
    # still in the buffer when the stream ends, so without this the live view
    # is permanently missing the end of every answer — and the shorter the
    # final burst, the more of it is lost. Caught by comparing the streamed
    # text against the returned reply, which is the only assertion that would
    # have noticed.
    _flush_stream(run, seen)
    return _Completed("".join(lines), "".join(errs), proc.returncode)


def _flush_stream(run, seen):
    if run is None:
        return
    for kind in ("thinking", "text"):
        buffer = seen[kind]
        if len(buffer["text"]) > buffer["sent"]:
            run.emit("stream", kind=kind,
                     delta=buffer["text"][buffer["sent"]:],
                     chars=len(buffer["text"]))
            buffer["sent"] = len(buffer["text"])


def _result_envelope(stdout):
    """The final `result` event out of a stream-json transcript.

    Read from the END backwards: a transcript holds many objects and only the
    last one is the answer. Scanning forward would find an `assistant` event
    and mistake a partial message for the result."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return None


def _claude_code_complete(cfg, system, user):
    binary = str(cfg.get("claude_binary") or "claude").strip() or "claude"
    resolved = shutil.which(binary)
    if not resolved:
        raise RuntimeError(
            f"the Claude Code CLI ({binary!r}) is not on PATH; install it or "
            "point Settings at the right binary")
    # `stream-json` rather than `json`: the events are the liveness signal the
    # idle timeout needs, and the assistant's text arrives as it is written
    # instead of all at once at the end. `--verbose` is required by the CLI
    # for this format and is not optional.
    argv = [resolved, "-p", "--output-format", "stream-json", "--verbose",
            # Token-level deltas. Without this the CLI emits one complete
            # message at the END of generation, which means the idle bound
            # sees total silence for the whole of a long answer and kills the
            # very run it exists to protect.
            "--include-partial-messages",
            # 2, not 1: headroom so a single stray step cannot end the run
            # outright. With --tools "" there is nothing to step toward, but
            # a ceiling of exactly one leaves no margin for the CLI's own
            # internal turns.
            "--max-turns", "2", "--tools", "",
            "--system-prompt", system]
    model = str(cfg.get("claude_model") or "").strip()
    if model:
        argv += ["--model", model]
    try:
        ceiling = float(cfg.get("claude_timeout") or 900.0)
    except (TypeError, ValueError):
        ceiling = 900.0
    try:
        idle_timeout = float(cfg.get("claude_idle_timeout")
                             or _DEFAULT_IDLE_TIMEOUT)
    except (TypeError, ValueError):
        idle_timeout = _DEFAULT_IDLE_TIMEOUT
    # The one argv entry that can still grow. Checked before spawning, because
    # the kernel's own answer for this is E2BIG on the whole exec, which names
    # neither which argument was too long nor what to do about it.
    system_bytes = len(str(system).encode("utf-8"))
    if system_bytes > _SYSTEM_PROMPT_BUDGET:
        raise RuntimeError(
            f"the system prompt is {system_bytes:,} bytes, over the "
            f"{_SYSTEM_PROMPT_BUDGET:,}-byte limit the CLI can take as a "
            "command-line argument. The persona sheet or the standing "
            "contract has grown past what this provider can carry; shorten "
            "it, or switch to the OpenAI-compatible HTTP provider, which has "
            "no argv limit.")
    scratch = tempfile.mkdtemp(prefix="assistant-claude-")
    # Popen rather than subprocess.run so the child can be REGISTERED and
    # therefore killed. A halt is checked at stage boundaries, and the
    # boundaries sit on either side of this call — measured at 47s for a real
    # turn — so without this the button's latency is one whole model call.
    # `turnrun.current()` is None for a blocking turn or a test, and then this
    # behaves exactly as `subprocess.run` did.
    run = turnrun.current()
    proc = None
    try:
        try:
            proc = subprocess.Popen(argv, cwd=scratch, text=True,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except OSError as exc:
            raise RuntimeError(f"could not run the Claude Code CLI: {exc}")
        if run is not None:
            run.register_process(proc)
        done = _stream_cli(proc, user, run, idle_timeout, ceiling)
    finally:
        if run is not None and proc is not None:
            run.unregister_process(proc)
        shutil.rmtree(scratch, ignore_errors=True)
    # A killed child exits without writing JSON. Ask the run whether that was
    # us: a halt must surface as a halt, not as "the CLI returned nothing",
    # which would send the user looking for a provider fault that never was.
    if run is not None:
        run.halted()
    if not (done.stdout or "").strip():
        raise RuntimeError(
            "the Claude Code CLI returned nothing: "
            + ((done.stderr or "").strip()[:300] or f"exit {done.returncode}"))
    envelope = _result_envelope(done.stdout)
    if envelope is None:
        raise RuntimeError(
            "the Claude Code CLI produced no result event: "
            + ((done.stderr or "").strip()[:200]
               or done.stdout.strip()[-300:] or f"exit {done.returncode}"))
    # `is_error` is the CLI's own verdict and it is NOT the exit code: a "Not
    # logged in" answer comes back as subtype "success" with is_error true and
    # the reason in `result`. Surfacing that text is the difference between
    # "run /login" and a silent dead assistant.
    if envelope.get("is_error"):
        raise RuntimeError("the Claude Code CLI reported an error: "
                           + str(envelope.get("result") or "")[:300])
    return str(envelope.get("result") or "")


def parse_model_json(raw):
    """Tolerant JSON extraction from model output — fenced blocks, prose
    around the object, trailing commas. Engine lineage: model output is
    provisional until deterministic code validates it, and a parse failure is
    a None the caller must handle, never an exception mid-pipeline."""
    import re
    text = str(raw or "").strip()
    if not text:
        return None
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text,
                                        re.S)):
        candidate = candidate.strip()
        match = re.search(r"\{.*\}", candidate, re.S)
        if not match:
            continue
        blob = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            out = json.loads(blob)
        except (TypeError, ValueError):
            continue
        if isinstance(out, dict):
            return out
    return None
