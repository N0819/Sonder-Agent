# Provider configuration, and the two ways it silently did nothing.
#
# Both defects here are the same shape as the persona-drive scar: a field that
# is authored, saved, displayed — and read by nothing. An empty field fails
# silently; a field nothing reads defeats even the check built to catch an
# empty one.

import http.client
import json
import os

import pytest

import config
import memory
import providers


def test_the_embeddings_settings_are_actually_read(temp_db, monkeypatch):
    """`_embed_config` read os.environ directly while the Settings tab
    displayed and saved `embed_base`/`embed_model` that nothing consulted.
    Configuring embeddings in the UI did nothing at all, and retrieval stayed
    on the lexical fallback while the page said otherwise."""
    monkeypatch.delenv("ASSISTANT_EMBED_BASE", raising=False)
    monkeypatch.delenv("ASSISTANT_EMBED_MODEL", raising=False)
    assert providers._embed_config() is None

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    config.save_config({"embed_base": "https://openrouter.ai/api/v1",
                        "embed_model": "perplexity/pplx-embed-v1-4b",
                        "embed_key_env": "OPENROUTER_API_KEY"})
    got = providers._embed_config()
    assert got is not None
    base, model, key = got
    assert base == "https://openrouter.ai/api/v1"
    assert model == "perplexity/pplx-embed-v1-4b"
    assert key == "sk-test"


def test_embeddings_are_configured_separately_from_chat(temp_db):
    """Not a convenience — the Claude Code CLI has no embeddings endpoint, so
    an assistant composing replies through it still needs somewhere to
    vectorise or three of the four ranking lanes go dark."""
    config.save_config({"chat_provider": config.PROVIDER_CLAUDE_CODE,
                        "embed_base": "https://openrouter.ai/api/v1",
                        "embed_model": "perplexity/pplx-embed-v1-4b"})
    cfg = config.get_config()
    assert cfg["chat_provider"] == config.PROVIDER_CLAUDE_CODE
    assert providers._embed_config()[1] == "perplexity/pplx-embed-v1-4b"


def test_a_preset_sets_a_key_name_and_never_a_key(temp_db):
    """The whole storage design in one assertion: a settings row names the
    variable a secret lives in and never carries the secret."""
    cfg, _warnings = config.apply_preset("embed", "openrouter-pplx-4b")
    assert cfg["embed_base"] == "https://openrouter.ai/api/v1"
    assert cfg["embed_model"] == "perplexity/pplx-embed-v1-4b"
    assert cfg["embed_key_env"] == "OPENROUTER_API_KEY"
    # nothing anywhere in the stored row looks like a credential
    import db
    stored = db.setting_get("providers")
    assert not any("sk-" in str(v) for v in stored.values())


def test_the_status_view_never_carries_a_secret(temp_db, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret-value")
    config.apply_preset("embed", "openrouter-pplx-4b")
    status = config.redacted_status()
    assert "sk-super-secret-value" not in str(status)
    assert status["secrets"]["embed_key_env"]["present"] is True
    assert status["secrets"]["embed_key_env"]["env"] == "OPENROUTER_API_KEY"


def test_changing_the_embedding_model_warns_that_the_bank_is_stranded(
        temp_db, monkeypatch):
    """A vector can only be compared with one from the same model, so every
    existing row scores 0.0 against a query embedded by a different one.
    Retrieval keeps working, on keyword match alone, and looks fine — so the
    warning has to arrive at the moment of the DECISION, not later from bad
    answers."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    memory.add_memory("semantic", "told", 0.7, "something worth keeping",
                      turn_idx=1, event_key="m:1")
    assert not [w for w in config.config_warnings() if "stranded" in w]

    config.apply_preset("embed", "openrouter-pplx-4b")
    warnings = config.config_warnings()
    assert any("cannot be compared" in w for w in warnings), warnings
    assert any("perplexity/pplx-embed-v1-4b" in w for w in warnings)


def test_the_identity_stamp_matches_what_will_be_written(temp_db):
    """The warning above compares against this, so it has to be the same
    string `embed_texts_meta` stamps on a new vector."""
    assert config.embedding_identity() == "cheap:crc32:256"
    config.apply_preset("embed", "openrouter-pplx-4b")
    assert config.embedding_identity() == "api:perplexity/pplx-embed-v1-4b"
    batch = providers.embed_texts_meta(["x"])
    if not batch.fallback:                    # only when a key is present
        assert batch.model_key == config.embedding_identity()


# ---- The CLI provider's transport ----

def _fake_cli(seen, out_lines, err_lines=(), returncode=0, stall=0.0):
    """A stand-in for the streamed CLI process.

    Models the pipes rather than `communicate`, because that is what the
    provider now drives: a writer thread feeding stdin while readers drain
    stdout and stderr. A fake that accepts the whole prompt in one call cannot
    exercise the deadlock this shape exists to avoid."""
    import io
    import time as _time

    class FakeStdin:
        def __init__(self):
            self.parts = []

        def write(self, data):
            self.parts.append(data)
            seen["stdin"] = "".join(self.parts)

        def flush(self):
            pass

        def close(self):
            pass

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = io.StringIO("".join(l + "\n" for l in out_lines))
            self.stderr = io.StringIO("".join(l + "\n" for l in err_lines))
            self.returncode = returncode
            self.killed = False
            if stall:
                _time.sleep(0)

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.killed = True

    return FakeProc()

def test_a_large_payload_goes_on_stdin_not_argv(temp_db, monkeypatch):
    """`[Errno 7] Argument list too long`. The payload was passed as a
    trailing argv entry, and Linux caps a SINGLE entry at MAX_ARG_STRLEN
    (128 KiB) independently of the 2 MiB total — so the respond stage died as
    soon as a session had real material in it. Measured at the time: 102 KB of
    codemap alone for a 115-file upload, before memory or beliefs were added.
    The user saw "respond stage failed", which named nothing actionable."""
    seen = {}
    proc = _fake_cli(seen, ['{"type": "result", "is_error": false, '
                            '"result": "ok"}'])
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")

    def fake_popen(argv, **kw):
        seen["argv"] = argv
        return proc

    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    payload = "x" * 200_000
    out = providers._claude_code_complete(
        {"claude_binary": "claude", "claude_timeout": 30.0}, "sys", payload)
    assert out == "ok"
    assert seen["stdin"] == payload
    assert not any(len(a) > providers.MAX_ARGV_ENTRY_BYTES
                   for a in seen["argv"])


def test_an_oversized_system_prompt_is_named_not_left_to_execve(temp_db,
                                                                monkeypatch):
    """The system prompt has no stdin channel (this CLI has no
    --system-prompt-file), so it is the one argv entry that can still grow.
    The kernel's answer for that is E2BIG on the whole exec, which says
    neither which argument was too long nor what to do — the failure that
    started this. Checked before spawning, where the cause is still known."""
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")
    with pytest.raises(RuntimeError) as caught:
        providers._claude_code_complete(
            {"claude_binary": "claude", "claude_timeout": 30.0},
            "s" * 200_000, "user")
    assert "system prompt" in str(caught.value)
    assert "HTTP provider" in str(caught.value)


# ---- The two ways a correct provider still fails to connect ----

def test_a_base_url_that_already_names_its_route_is_folded(temp_db):
    """Copying the URL from a provider's docs gives you the full endpoint, and
    the provider layer appends the route itself — so
    `https://openrouter.ai/api/v1/embeddings` became `…/embeddings/embeddings`
    and 404'd. The UI then reported "HTTP Error 404", naming neither the cause
    nor the field. Folded on the way in, per the canonical_url rule."""
    config.save_config({"embed_base": "https://openrouter.ai/api/v1/embeddings",
                        "chat_base": "https://api.openai.com/v1/chat/completions"})
    got = config.get_config()
    assert got["embed_base"] == "https://openrouter.ai/api/v1"
    assert got["chat_base"] == "https://api.openai.com/v1"
    # Idempotent: an already-correct base survives a re-save untouched.
    config.save_config({"embed_base": got["embed_base"]})
    assert config.get_config()["embed_base"] == "https://openrouter.ai/api/v1"


def test_a_stored_key_is_what_actually_gets_sent(temp_db, monkeypatch):
    """The point of storing a key at all: type it, save, and the next request
    is authenticated — no export, no restart. The restart is what made the
    environment-only path fail invisibly, because a running process cannot see
    an export made after it launched."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config.save_config({"embed_base": "https://openrouter.ai/api/v1",
                        "embed_model": "perplexity/pplx-embed-v1-4b",
                        "embed_key_env": "OPENROUTER_API_KEY",
                        "embed_key": "sk-or-v1-abc123def"})
    assert config.secret_for("embed_key_env") == "sk-or-v1-abc123def"
    assert config.secret_source("embed_key_env") == "stored"
    assert providers._embed_config()[2] == "sk-or-v1-abc123def"
    # And the 401 warning must stop firing, or it teaches users to ignore it.
    assert not any("401" in w for w in config.config_warnings())


def test_a_stored_key_beats_a_stale_export(temp_db, monkeypatch):
    """Precedence is not arbitrary. The other order lets an export made months
    ago silently shadow the key just typed into Settings, and the symptom is a
    401 against configuration that looks correct — the exact failure the field
    was added to end."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-stale-from-the-shell")
    config.save_config({"embed_key_env": "OPENROUTER_API_KEY"})
    assert config.secret_for("embed_key_env") == "sk-stale-from-the-shell"
    config.save_config({"embed_key": "sk-typed-just-now"})
    assert config.secret_for("embed_key_env") == "sk-typed-just-now"


def test_a_stored_key_never_reaches_the_browser(temp_db):
    """`get_config` now returns credentials, so `redacted_status` is the only
    thing between them and the page. It strips by KEY_VALUE_FIELDS rather than
    listing what to send, so a credential field added later is redacted by
    default instead of shipped by omission."""
    config.save_config({"embed_key": "sk-or-v1-secret", "chat_key": "sk-chat"})
    status = config.redacted_status()
    assert "sk-or-v1-secret" not in json.dumps(status)
    assert "sk-chat" not in json.dumps(status)
    assert status["config"]["embed_key"] == ""
    # Presence still has to be reportable, or the page cannot tell "no key"
    # from "a key you are not allowed to see".
    assert status["secrets"]["embed_key_env"]["present"] is True
    assert status["secrets"]["embed_key_env"]["source"] == "stored"


def test_a_key_pasted_into_the_name_field_is_routed_not_refused(temp_db):
    """Refusing was right only while there was nowhere to put it. A key typed
    into the variable-name box is the obvious user action, so it is folded
    into the right field on the way in — the `_fold_base_url` rule — rather
    than rejected with an explanation of a distinction the user did not want
    to learn."""
    config.save_config({"embed_key_env": "OPENROUTER_API_KEY"})
    _cfg, warnings = config.save_config({"embed_key_env": "sk-or-v1-pasted"})
    assert config.secret_for("embed_key_env") == "sk-or-v1-pasted"
    # The name field keeps its name — the paste must not destroy it.
    assert config.get_config()["embed_key_env"] == "OPENROUTER_API_KEY"
    assert any("looks like a key" in w for w in warnings)


def test_a_blank_key_field_keeps_the_stored_key(temp_db):
    """The page cannot re-display a stored key, so the input it draws is
    always blank — and a blank submit that cleared the key would delete the
    credential every time any unrelated setting was saved. Clearing is
    explicit; blank means unchanged."""
    config.save_config({"embed_key": "sk-or-v1-keepme"})
    config.save_config({"embed_key": "", "embed_model": "some/other-model"})
    assert config.secret_for("embed_key_env") == "sk-or-v1-keepme"
    config.save_config({"embed_key": config.CLEAR_SECRET})
    assert config.secret_for("embed_key_env") == ""
    from db import setting_get
    assert "keepme" not in json.dumps(setting_get("providers") or {})


def test_configuring_one_provider_does_not_wipe_the_other(temp_db):
    """`save_config` wrote its cleaned fields as the WHOLE settings row, so a
    partial save reset every field it did not mention — and `apply_preset` is
    exactly that: choosing an embeddings preset sends three embed_* fields and
    wiped chat_base, chat_model and the claude_* block back to environment
    defaults. Configuring one provider destroyed the other while the page
    reported success."""
    config.save_config({"chat_base": "https://api.example.com/v1",
                        "chat_model": "my-chat-model"})
    config.apply_preset("embed", "openrouter-pplx-4b")
    got = config.get_config()
    assert got["chat_model"] == "my-chat-model"
    assert got["chat_base"] == "https://api.example.com/v1"
    assert got["embed_model"] == "perplexity/pplx-embed-v1-4b"


def test_a_save_that_changed_your_input_says_so_on_the_page(temp_db):
    """`settings_put` threw `save_config`'s warnings away and returned a fresh
    status, so a save that quietly rerouted a field redrew looking untouched —
    which is what "I pressed save and nothing happened" actually was."""
    from fastapi.testclient import TestClient
    import app as app_module
    client = TestClient(app_module.app)
    out = client.put("/api/settings",
                     json={"settings": {"chat_key_env": "sk-live-nope"}}).json()
    assert any("looks like a key" in w for w in out["warnings"])
    # ...and the response that carries the warning still carries no secret.
    assert "sk-live-nope" not in json.dumps(out["config"])


# ---- Rebuilding, so a provider change is not a one-way door ----

def test_a_rebuild_without_a_provider_refuses_rather_than_destroying(temp_db):
    """The failure mode to avoid is overwriting good vectors with
    hashing-trick ones, which turns a stalled migration into a corrupted
    bank."""
    memory.add_memory("semantic", "told", 0.7, "a memory", turn_idx=1,
                      event_key="m:1")
    before = memory.visible_memory_rows(before_turn_idx=None,
                                        include_archived=True)[0]["embedding"]
    out = memory.rebuild_embeddings()
    assert out["ok"] is False and "no embeddings provider" in out["error"]
    after = memory.visible_memory_rows(before_turn_idx=None,
                                       include_archived=True)[0]["embedding"]
    assert before == after


def test_a_rebuild_re_embeds_stranded_rows(temp_db, monkeypatch):
    """DESIGN.md listed this unbuilt while a provider change was
    hypothetical. It stops being hypothetical the moment somebody switches
    providers: without it the settings page offers a choice between your
    existing bank and better retrieval."""
    import numpy as np
    memory.add_memory("semantic", "told", 0.7, "a memory to migrate",
                      turn_idx=1, event_key="m:1")
    import db
    db.qi("UPDATE memories SET embedding_model='api:old-model'")
    assert config.config_warnings()

    # A deterministic stand-in for a real provider: distinct model stamp,
    # different dimensionality, no network.
    def fake(texts):
        return providers.EmbeddingBatch(
            vectors=[np.ones(8, dtype=np.float32) / np.sqrt(8)
                     for _ in texts],
            model_key="api:perplexity/pplx-embed-v1-4b", dimensions=8)

    monkeypatch.setattr(memory, "embed_texts_meta", fake)
    out = memory.rebuild_embeddings()
    assert out["ok"] and out["rebuilt"] == 1 and out["remaining"] == 0
    row = db.q("SELECT embedding_model, embedding_dim FROM memories",
               one=True)
    assert row["embedding_model"] == "api:perplexity/pplx-embed-v1-4b"
    assert row["embedding_dim"] == 8


def test_a_completed_rebuild_is_a_no_op(temp_db, monkeypatch):
    """Re-runnable by construction: it selects by "stamp differs from
    current", so an interrupted run is continued by the next one and a
    finished run does nothing."""
    import numpy as np
    memory.add_memory("semantic", "told", 0.7, "a memory", turn_idx=1,
                      event_key="m:1")

    def fake(texts):
        return providers.EmbeddingBatch(
            vectors=[np.ones(8, dtype=np.float32) / np.sqrt(8)
                     for _ in texts],
            model_key="api:e", dimensions=8)

    monkeypatch.setattr(memory, "embed_texts_meta", fake)
    assert memory.rebuild_embeddings()["rebuilt"] == 1
    assert memory.rebuild_embeddings()["rebuilt"] == 0


def test_a_rebuild_migrates_summary_windows_too(temp_db, monkeypatch):
    """The rebuild covered `memories` and not `memory_summaries`, which also
    stores a vector. A cross-model window is SKIPPED by
    search_memory_summaries rather than scored, so every consolidated window
    left retrieval permanently on a provider switch — while the rebuild
    reported the bank fully comparable, which is the exact silent success this
    whole mechanism exists to prevent."""
    import numpy as np
    import db

    def fake(texts):
        return providers.EmbeddingBatch(
            vectors=[np.ones(8, dtype=np.float32) / np.sqrt(8)
                     for _ in texts],
            model_key="api:new", dimensions=8)

    memory.save_memory_summary("the deploy pipeline moved to buildkite",
                               end_turn_idx=4, key_phrases=["buildkite"])
    db.qi("UPDATE memory_summaries SET embedding_model='api:old'")
    assert any("summary windows" in w for w in config.config_warnings())

    monkeypatch.setattr(memory, "embed_texts_meta", fake)
    assert memory.search_memory_summaries("buildkite", exclude_latest=False,
                                          embedded=fake(["q"])) == []
    out = memory.rebuild_embeddings()
    assert out["ok"] and out["summaries_rebuilt"] == 1 and out["remaining"] == 0
    assert memory.search_memory_summaries("buildkite", exclude_latest=False,
                                          embedded=fake(["q"]))


def test_a_blank_summary_window_never_becomes_perpetual_work(temp_db,
                                                             monkeypatch):
    """A window with no prose is unretrievable by construction and cannot be
    embedded from an empty string. Counted as outstanding but never processed,
    it would make "run again to continue" an instruction that never
    terminates."""
    import numpy as np
    import db
    memory.save_memory_summary("", end_turn_idx=2)
    db.qi("UPDATE memory_summaries SET embedding_model='api:old'")

    monkeypatch.setattr(memory, "embed_texts_meta", lambda texts:
                        providers.EmbeddingBatch(
                            vectors=[np.ones(8, dtype=np.float32) / np.sqrt(8)
                                     for _ in texts],
                            model_key="api:new", dimensions=8))
    out = memory.rebuild_embeddings()
    assert out["ok"] and out["remaining"] == 0 and out["summaries_rebuilt"] == 0


def test_a_rebuilt_document_is_the_document_that_was_stored(temp_db,
                                                            monkeypatch):
    """The rebuild rebuilt each document from content/gist/phrases/entities
    only, so `_memory_document` fell back to its defaults — `kind: episodic`,
    `source: witnessed`, an empty turn and url. A rebuilt vector therefore
    encoded text the row never had, and the SELECT was already fetching the
    fields it then discarded."""
    import numpy as np
    import db
    memory.add_memory("semantic", "told", 0.7, "the invoice was paid",
                      turn_idx=9, source_url="https://example.com/a",
                      event_key="m:1")
    db.qi("UPDATE memories SET embedding_model='api:old'")
    seen = []

    def fake(texts):
        seen.extend(texts)
        return providers.EmbeddingBatch(
            vectors=[np.ones(8, dtype=np.float32) / np.sqrt(8)
                     for _ in texts],
            model_key="api:new", dimensions=8)

    monkeypatch.setattr(memory, "embed_texts_meta", fake)
    memory.rebuild_embeddings()
    document = next(t for t in seen if t.startswith("kind:"))
    assert "kind: semantic" in document and "source: told" in document
    assert "turn: 9" in document
    assert "url: https://example.com/a" in document


def test_rebuilt_rows_are_retrievable_again(temp_db, monkeypatch):
    """The point of the exercise: a stranded row is reachable by keyword only,
    and after a rebuild it is back in the vector lanes."""
    import numpy as np
    memory.add_memory("semantic", "told", 0.7,
                      "the deployment pipeline uses buildkite", turn_idx=1,
                      event_key="m:1")
    import db
    db.qi("UPDATE memories SET embedding_model='api:stranded'")
    _payload, internal = memory.build_memory_context(5, "deployment pipeline")
    assert internal["retrieval_health"]["vector_incomparable_rows"] == 1

    def fake(texts):
        return providers.EmbeddingBatch(
            vectors=[np.ones(256, dtype=np.float32) / 16.0 for _ in texts],
            model_key="api:e", dimensions=256)

    monkeypatch.setattr(memory, "embed_texts_meta", fake)
    memory.rebuild_embeddings()
    _payload, internal = memory.build_memory_context(5, "deployment pipeline")
    assert internal["retrieval_health"]["vector_incomparable_rows"] == 0


# ---- Two bounds, because there are two failures ----

def test_a_slow_but_talking_cli_is_not_killed(temp_db, monkeypatch):
    """THE TIMEOUT MEASURED THE WRONG THING. One wall clock cannot tell a
    model that is thinking hard from one that has hung, so a real answer past
    180 seconds was killed mid-sentence and surfaced as "respond stage
    failed" — which reads as a fault and is not one.

    This run takes longer than its own idle bound in total, and never goes
    silent for that long. It must survive."""
    import io
    seen = {}

    class Trickle(io.StringIO):
        """Events arriving slowly, one per read, as a real stream does."""

        def __iter__(self):
            import time as _t
            for line in ('{"type": "assistant", "message": {"content": '
                         '[{"type": "text", "text": "thinking..."}]}}',
                         '{"type": "assistant", "message": {"content": '
                         '[{"type": "text", "text": " still going"}]}}',
                         '{"type": "result", "is_error": false, '
                         '"result": "the answer"}'):
                _t.sleep(0.15)
                yield line + "\n"

    proc = _fake_cli(seen, [])
    proc.stdout = Trickle()
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(providers.subprocess, "Popen",
                        lambda argv, **kw: proc)
    out = providers._claude_code_complete(
        # An idle bound SHORTER than the total run: the point of the split.
        {"claude_binary": "claude", "claude_idle_timeout": 0.3,
         "claude_timeout": 30.0}, "sys", "user")
    assert out == "the answer"
    assert proc.killed is False, "a talking process was killed"


def test_a_silent_cli_is_killed_and_the_bound_is_named(temp_db, monkeypatch):
    """The other half. An error that says only "it timed out" sends whoever
    reads it to raise whichever number they happen to find."""
    import io
    seen = {}

    class Silence(io.StringIO):
        def __iter__(self):
            import time as _t
            _t.sleep(5)
            return iter(())

    proc = _fake_cli(seen, [])
    proc.stdout = Silence()
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(providers.subprocess, "Popen",
                        lambda argv, **kw: proc)
    with pytest.raises(RuntimeError) as caught:
        providers._claude_code_complete(
            {"claude_binary": "claude", "claude_idle_timeout": 0.4,
             "claude_timeout": 30.0}, "sys", "user")
    assert "went silent" in str(caught.value)
    assert "claude_idle_timeout" in str(caught.value), "the fix is unnamed"
    assert proc.killed is True


def test_the_result_is_read_from_the_end_of_the_transcript(temp_db):
    """A stream-json transcript holds many objects and only the last is the
    answer. Scanning forward finds an `assistant` event and mistakes a partial
    message for the result — which would silently return the first sentence of
    a long reply as though it were the whole thing."""
    transcript = "\n".join([
        '{"type": "system", "subtype": "init"}',
        '{"type": "assistant", "message": {"content": '
        '[{"type": "text", "text": "partial"}]}}',
        'not json at all',
        '{"type": "result", "is_error": false, "result": "the whole answer"}',
    ])
    envelope = providers._result_envelope(transcript)
    assert envelope["result"] == "the whole answer"
    assert providers._result_envelope("") is None
    assert providers._result_envelope("garbage") is None


def test_the_streamed_text_reaches_the_watching_turn(temp_db, monkeypatch):
    """"See it as it streams" is only true if the tokens reach the run. The
    deltas are throttled, so the assertion is on the reassembled text rather
    than on a count of events."""
    import io

    import turnrun
    seen = {}
    run = turnrun.create("hello", None)
    turnrun.bind(run)

    class Trickle(io.StringIO):
        def __iter__(self):
            import time as _t
            for word in ("Once ", "upon ", "a ", "time"):
                _t.sleep(0.45)      # past the throttle, so each one emits
                yield ('{"type": "assistant", "message": {"content": '
                       '[{"type": "text", "text": "%s"}]}}\n' % word)
            yield ('{"type": "result", "is_error": false, '
                   '"result": "Once upon a time"}\n')

    proc = _fake_cli(seen, [])
    proc.stdout = Trickle()
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(providers.subprocess, "Popen",
                        lambda argv, **kw: proc)
    try:
        providers._claude_code_complete(
            {"claude_binary": "claude", "claude_idle_timeout": 5.0}, "s", "u")
    finally:
        turnrun.bind(None)
    streamed = "".join(e.get("delta", "") for e in run.events
                       if e["stage"] == "stream")
    # EXACT, not "starts with". The emit is throttled, so whatever arrived
    # since the last emit is still buffered when the stream ends — the live
    # view was permanently missing the tail of every answer, and the shorter
    # the final burst the more of it was lost. Only a full-equality assertion
    # notices; `in` or a length check passes throughout.
    assert streamed == "Once upon a time", streamed


def test_token_deltas_and_the_final_message_are_not_both_counted(temp_db,
                                                                 monkeypatch):
    """With --include-partial-messages the CLI emits token-level deltas AND a
    complete `assistant` message at the end holding the same text. Consuming
    both shows every answer twice; consuming neither shows nothing. The
    fallback exists for a build that emits no deltas, so it has to be
    conditional rather than removed."""
    import turnrun
    seen = {}
    run = turnrun.create("hello", None)
    turnrun.bind(run)
    lines = [
        '{"type": "stream_event", "event": {"type": "content_block_delta",'
        ' "delta": {"type": "thinking_delta", "thinking": "hmm"}}}',
        '{"type": "stream_event", "event": {"type": "content_block_delta",'
        ' "delta": {"type": "text_delta", "text": "Hello"}}}',
        '{"type": "stream_event", "event": {"type": "content_block_delta",'
        ' "delta": {"type": "text_delta", "text": " there"}}}',
        '{"type": "assistant", "message": {"content": '
        '[{"type": "text", "text": "Hello there"}]}}',
        '{"type": "result", "is_error": false, "result": "Hello there"}',
    ]
    proc = _fake_cli(seen, lines)
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(providers.subprocess, "Popen",
                        lambda argv, **kw: proc)
    try:
        out = providers._claude_code_complete(
            {"claude_binary": "claude"}, "s", "u")
    finally:
        turnrun.bind(None)
    assert out == "Hello there"
    text = "".join(e.get("delta", "") for e in run.events
                   if e["stage"] == "stream" and e.get("kind") == "text")
    assert text == "Hello there", "the final message was counted twice"
    # Reasoning is kept apart from what it decided to say: a draft shown in
    # the answer pane reads as a conclusion.
    thinking = "".join(e.get("delta", "") for e in run.events
                       if e["stage"] == "stream" and e.get("kind") == "thinking")
    assert thinking == "hmm"


def test_the_partial_message_flag_is_actually_requested(temp_db, monkeypatch):
    """Without it the CLI emits ONE complete message at the end of generation,
    so the idle bound sees total silence for the whole of a long answer and
    kills the very run it exists to protect. Measured against the installed
    CLI: 11 events with the flag, 1 without."""
    seen = {}
    proc = _fake_cli(seen, ['{"type": "result", "is_error": false, '
                            '"result": "ok"}'])
    monkeypatch.setattr(providers.shutil, "which", lambda b: "/usr/bin/claude")

    def fake_popen(argv, **kw):
        seen["argv"] = argv
        return proc

    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    providers._claude_code_complete({"claude_binary": "claude"}, "s", "u")
    assert "--include-partial-messages" in seen["argv"]
    assert "stream-json" in seen["argv"]


def test_no_stage_gets_a_smaller_output_ceiling_than_the_default():
    """A CEILING NOBODY CHOSE COST A WHOLE TURN. `chat_complete` defaulted to
    max_tokens=2000, and the Claude Code CLI path ignores the argument
    entirely — so for as long as the CLI was the provider the default was
    never exercised, and the first HTTP provider configured lost a turn to it.
    The respond stage, which emits the largest output of any stage (the reply
    plus every side channel), was the one call site that never overrode it,
    while consolidation and subagent reports both did.

    Output tokens are billed as generated rather than as budgeted, so a stage
    pinning a LOWER ceiling than the default is buying nothing and risking the
    same silent loss. Structural rather than behavioural on purpose: the
    failure is a call site that forgot, and only reading the call sites
    catches that."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    pinned = []
    for path in sorted(root.glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for found in re.finditer(r"max_tokens\s*=\s*(\d+)", line):
                pinned.append((path.name, n, int(found.group(1))))
    too_small = [p for p in pinned if p[2] < providers.DEFAULT_MAX_TOKENS]
    assert not too_small, f"ceiling below the default: {too_small}"
    assert providers.DEFAULT_MAX_TOKENS >= 8000


def test_a_slow_provider_is_retried_and_named_rather_than_leaking_a_socket(
        monkeypatch):
    """THE ONE FAILURE THE BACKOFF EXISTS FOR WAS THE ONE IT NEVER CAUGHT. A
    read timeout raises the bare socket `TimeoutError`, which is not a
    `URLError`, so it escaped `_post_with_retry` entirely — unretried, and
    surfaced to the turn as "The read operation timed out". That names no
    provider, no stage, and no number the operator could change, and it is
    indistinguishable from an unreachable host when the recovery is the
    opposite one: wait, versus go and fix something."""
    tries = []

    def slow(req, timeout=None):
        tries.append(timeout)
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(providers.urllib.request, "urlopen", slow)
    with pytest.raises(RuntimeError) as caught:
        providers._post_with_retry(object(), attempts=2)

    assert len(tries) == 2, "a timeout must be retried like any other blip"
    assert tries[0] == providers.CHAT_TIMEOUT
    message = str(caught.value)
    assert "did not respond within" in message
    assert "ASSISTANT_CHAT_TIMEOUT" in message


def test_generation_does_not_inherit_the_lookup_timeout():
    """Embeddings return in well under a second; a respond round sends a 17k
    system prompt and may generate thousands of tokens before the first byte.
    They shared one 60s ceiling, which the Claude Code CLI never exercised
    because it is a subprocess and never touches urllib."""
    assert providers.CHAT_TIMEOUT > providers.REQUEST_TIMEOUT
    assert providers.CHAT_TIMEOUT >= 300


@pytest.mark.parametrize("failure", [
    TimeoutError("The read operation timed out"),
    http.client.RemoteDisconnected(
        "Remote end closed connection without response"),
    ConnectionResetError("Connection reset by peer"),
    providers.urllib.error.URLError("Name or service not known"),
    http.client.BadStatusLine("''"),
])
def test_every_transport_failure_is_retried_and_named(failure, monkeypatch):
    """CAUGHT THE MEMBER, NOT THE FAMILY, AND WAS FORGOTTEN TWICE. The clause
    handled `URLError` only, but urllib wraps what `request()` raises and not
    what `getresponse()` does — so a read timeout arrived as a bare
    `TimeoutError` and a dropped connection as `http.client.RemoteDisconnected`.
    Both escaped the bounded backoff unretried and surfaced as raw library
    strings naming no provider and no stage. Enumerating the classes as each
    one bites is the guard that must be remembered; this asserts the family."""
    tries = []

    def fail(req, timeout=None):
        tries.append(timeout)
        raise failure

    monkeypatch.setattr(providers.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError) as caught:
        providers._post_with_retry(object(), attempts=2)

    assert len(tries) == 2, f"{type(failure).__name__} was not retried"
    assert "chat provider" in str(caught.value), \
        f"{type(failure).__name__} leaked a raw library string"


def test_the_search_settings_are_actually_read(temp_db, monkeypatch):
    """THE EMBEDDINGS SCAR, IN THE SAME APP TWICE. `_embed_config` exists
    because the Settings tab displayed and saved embeddings fields that
    nothing consulted — configuring them did nothing at all, silently, while
    the page said otherwise. A search key the settings page accepts and no
    code reads would be that defect reintroduced, and this time in the lane
    that had just been found dead."""
    import tools_web
    tools_web.set_search_stub(None)
    tools_web.set_search_backend(None)

    config.save_config({"search_provider": "brave", "search_key": "sk-test"})
    seen = {}

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["token"] = req.get_header("X-subscription-token")
        raise TimeoutError("stop here; the wiring is what is under test")

    monkeypatch.setattr(tools_web._opener, "open", fake)
    got = tools_web.search_detail("BBC News")

    assert "api.search.brave.com" in seen.get("url", ""), \
        "the configured provider was never reached"
    assert seen.get("token") == "sk-test", "the stored key was not sent"
    assert got["status"] == "error"


def test_a_keyless_backend_choice_is_honoured(temp_db, monkeypatch):
    """DuckDuckGo is kept selectable rather than deleted: the challenge that
    killed it was observed from ONE network, and an anti-bot block is a
    property of the requesting address as much as of the endpoint."""
    import tools_web
    tools_web.set_search_stub(None)
    tools_web.set_search_backend(None)
    config.save_config({"search_provider": "ddg", "search_key": ""})

    seen = {}

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        raise TimeoutError("stop here")

    monkeypatch.setattr(tools_web._opener, "open", fake)
    tools_web.search_detail("BBC News")
    assert "duckduckgo.com" in seen.get("url", "")


def test_a_blocked_backend_names_itself_rather_than_the_query(temp_db):
    """"The lane is down" and "this question has no answer" need opposite
    responses and were one outcome. The loop kept rephrasing, which is the one
    move that cannot help."""
    import tools_web
    challenge = ("<html><title>Captcha</title>Please complete the following "
                 "challenge to confirm this search was made by a human.</html>")
    assert tools_web._parse_ddg(challenge, 5) == []
    assert any(c in challenge.lower() for c in tools_web._BLOCK_CUES)


def test_a_search_key_nobody_reads_is_reported(temp_db):
    """`_configured_backend` matches "brave" and nothing else, so any other
    provider means a stored search key is never used — and the page shows a
    filled key field either way, because `redacted_status` cannot re-display
    it. Measured live: a working 31-character Brave key stored, provider
    reading "openai-compatible", every search for six turns falling through to
    the keyless scrapers — which refuse real research queries while answering
    short controls, so a control query proved the lane worked. Nothing
    anywhere connected those three facts."""
    config.save_config({"search_provider": "mojeek", "search_key": "abc123"})
    warnings = config.config_warnings()
    assert any("does not read it" in w for w in warnings), warnings
    # ...and the honest configuration is quiet.
    config.save_config({"search_provider": "brave"})
    assert not [w for w in config.config_warnings() if "search" in w]


def test_a_provider_that_is_not_a_search_provider_is_refused(temp_db):
    """`search_provider` was found holding "openai-compatible" — a CHAT
    provider value, written by an earlier build that rendered both selects
    from the chat list. The only copy of the valid set lived in controls.js
    and nothing on this side ever checked."""
    config.save_config({"search_provider": "brave"})
    _cfg, warnings = config.save_config(
        {"search_provider": "openai-compatible"})
    assert any("is not a search provider" in w for w in warnings), warnings
    # REFUSED, not folded: the previous choice stands rather than being
    # silently rewritten to a default the user never picked.
    assert config.get_config()["search_provider"] == "brave"


def test_brave_without_a_key_says_so(temp_db):
    """The other half of the same silence."""
    config.save_config({"search_provider": "brave",
                        "search_key": config.CLEAR_SECRET})
    assert any("no key is set" in w for w in config.config_warnings())


def test_the_settings_page_cannot_silently_rewrite_a_value_it_cannot_show():
    """A <select> silently discards a value that is not among its options, and
    Save then writes back what it is DISPLAYING. `search_provider` held
    "openai-compatible", so the page showed "Mojeek (default)", and a user who
    pasted a Brave key and pressed Save had their provider rewritten to the
    option the box happened to be resting on. Their key stored correctly and
    was never read. It then happened a SECOND time, to a provider that had
    been set correctly from outside the page — which is how it was found.

    A form that cannot display its own data must not be allowed to overwrite
    it by omission."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    js = open(os.path.join(root, "static/js/controls.js")).read()
    body = js.split("input.value = out.config[key];", 1)[1][:1200]
    assert "SELECT" in body, "the mismatch has to be detected on selects"
    assert "not a valid choice" in body, "and shown, not silently corrected"


def test_the_cli_saying_it_stopped_early_is_not_a_parse_failure(monkeypatch):
    """`chat_complete` returns the CLI path before it ever reaches the
    `finish_reason == "length"` branch, so the truncation check existed on the
    HTTP provider only. On the live provider a run that stopped early came
    back as ordinary text, failed to parse, and was reported as "respond stage
    returned unparseable output" — the exact wrong subsystem, named in that
    branch's own comment, reachable by the path it does not cover."""
    import providers

    envelope = {"type": "result", "subtype": "error_max_turns",
                "is_error": False, "result": "{\"reply\": \"half an ans"}
    monkeypatch.setattr(providers, "_result_envelope", lambda out: envelope)

    class Done:
        stdout, stderr, returncode = "", "", 0

    with pytest.raises(RuntimeError) as caught:
        providers._finish_claude_result(Done())
    assert "stopped early (error_max_turns)" in str(caught.value)
    # The length of what DID come back is the operator's next clue.
    assert "22 characters" in str(caught.value)


def test_a_successful_cli_result_is_returned_untouched(monkeypatch):
    """The guard must leave the honest case alone."""
    import providers

    envelope = {"type": "result", "subtype": "success", "is_error": False,
                "result": '{"reply": "fine"}'}
    monkeypatch.setattr(providers, "_result_envelope", lambda out: envelope)

    class Done:
        stdout, stderr, returncode = "", "", 0

    assert providers._finish_claude_result(Done()) == '{"reply": "fine"}'
