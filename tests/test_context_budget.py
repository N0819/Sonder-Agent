# Nothing may pack a corpus into a model payload.
#
# THE SCAR. `_run_deep` built its task payload with
# `"files": workspace.snapshot_for_sandbox(session_id)` — every body, up to
# 8 MiB and 400 files, in the child's model context before it had done
# anything. Measured on a 119-file workspace: 78 files, 859,445 characters,
# roughly 215k tokens. The child was handed the entire source tree to answer
# one question about one function.
#
# It survived into a running system because NO TEST BOUNDED AN ASSEMBLED
# PAYLOAD. Every module was individually correct: `snapshot_for_sandbox` is
# right to return whole files (the sandbox needs real files on disk),
# `chunks.digest` is right to be small, and nothing asserted the relationship
# between them. The defect lived in the seam, which is where the tests were
# not.
#
# So these tests assert the PROPERTY, not the fix: no assembled payload
# carries file bodies, whatever the mechanism. A future change that
# reintroduces the leak by another route fails here.

import json

import pytest

import chunks
import pipeline
import providers
import subagents
import workspace


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace root of its own, so a test never walks the real one.

    ARCHIVE_ROOT is redirected for the same reason, and it is not
    hypothetical: `_run_deep` archives in a `finally`, so exercising it here
    wrote seven real tarballs into the user's live `subagent-archive/` before
    this line existed. A test that leaves artefacts in the directory a human
    audits is a test that corrupts the evidence it exists to protect."""
    workspace.configure(str(tmp_path / "workspaces"))
    monkeypatch.setattr(subagents, "ARCHIVE_ROOT",
                        str(tmp_path / "subagent-archive"))
    yield tmp_path


# Generous, and still two orders of magnitude below what was measured. The
# point is not to tune a number — it is that SOME bound exists and a corpus
# cannot slip under it.
MAX_PAYLOAD_CHARS = 120_000
_NEEDLE = "MARKER_THAT_MUST_NOT_REACH_A_MODEL"


def _fill_workspace(count=40):
    """A workspace big enough that packing it would be obvious."""
    for i in range(count):
        workspace.store_upload(
            1, f"mod{i}.py",
            (f'"""Module {i}."""\n\n\ndef f{i}():\n'
             f'    # {_NEEDLE}\n'
             f'    return {i}\n' + ("# padding line\n" * 400)).encode())
    chunks.ingest_workspace()


def test_a_deep_subagent_is_handed_a_map_not_a_corpus(ws, temp_db,
                                                      monkeypatch):
    """The crash itself, caught at the seam it actually happened in.

    Asserting that a digest contains no bodies would pass whether or not the
    bug existed — the digest was never the leak. What has to be inspected is
    the bytes `_run_deep` writes to the child's stdin, so the real payload is
    captured here and the spawn is faked."""
    _fill_workspace()
    written = []

    class FakeStdin:
        def write(self, data):
            written.append(data)

        def flush(self):
            pass

    class FakeProc:
        stdin = FakeStdin()
        stdout = None
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subagents.subprocess, "Popen",
                        lambda *a, **kw: FakeProc())
    # `_converse` reads a reply the fake will never produce; the payload is
    # already on the wire by then, which is all this test is about.
    monkeypatch.setattr(subagents, "_converse",
                        lambda proc, payload, **kw: (
                            proc.stdin.write(json.dumps(payload)), None)[1])
    subagents._run_deep("look at mod3", session_id=1, turn_idx=1)

    assert written, "no payload was sent"
    blob = "".join(written)
    snapshot = workspace.snapshot_for_sandbox(None)
    assert any(_NEEDLE in text for text in snapshot.values()), "fixture is wrong"
    assert _NEEDLE not in blob, "a file body reached the child's context"
    assert len(blob) < MAX_PAYLOAD_CHARS, f"payload was {len(blob):,} chars"


def test_the_turn_payload_never_carries_file_bodies(ws, temp_db):
    """The parent's own payload, assembled for real. `code` is a digest of
    gists and ids; the bodies are fetched by id or not at all."""
    _fill_workspace()
    seen = {}

    def capture(system, user):
        seen["user"] = user
        return json.dumps({"reply": "ok"})

    providers.set_chat_stub(capture)
    try:
        pipeline.run_turn("tell me about mod3")
    finally:
        providers.set_chat_stub(None)
    body = seen["user"]
    assert _NEEDLE not in body, "a file body reached the model payload"
    assert len(body) < MAX_PAYLOAD_CHARS, f"payload was {len(body):,} chars"


def test_the_map_resolves_whatever_scope_a_caller_passes(ws, temp_db):
    """The other half of the failure. Chunks were written at the workspace
    scope and read at a live session id, so `digest` reported "0 chunks across
    0 sources" while the rows sat in the table — the intended read channel was
    empty and nothing said so."""
    _fill_workspace()
    totals = {sid: chunks.digest(sid, kind="code")["total_chunks"]
              for sid in (0, 2, 9, None)}
    assert len(set(totals.values())) == 1, totals
    assert all(t > 0 for t in totals.values()), totals


def test_machinery_directories_are_not_material(ws):
    """`.git` alone contributed 60-odd entries to a 119-"file" workspace:
    counted in the totals, walked by the mapper, offered to the assistant as
    things it might read."""
    import os
    root = workspace.workspace_root()
    for junk in (".git", ".pytest_cache", "__pycache__"):
        os.makedirs(os.path.join(root, junk), exist_ok=True)
        with open(os.path.join(root, junk, "obj"), "w") as handle:
            handle.write("x" * 5000)
    workspace.store_upload(1, "real.py", b"x = 1")

    paths = [f["path"] for f in workspace.list_files()]
    assert paths == ["real.py"], paths
    listed = [e["path"] for e in workspace.list_dir("")["entries"]]
    assert listed == ["real.py"], listed


def test_a_subagent_payload_carries_no_credential(ws, temp_db):
    """`provider_config` was `config.get_config()` under a comment promising
    secrets were never serialised — true when a settings row could only hold
    a variable NAME, false once a key could be stored outright. The child
    writes what it is handed into its own database, and `archive_run` tars
    that database, so a stored key came to rest in a .tar.gz whose entire
    purpose is to be kept and read later."""
    import config
    config.save_config({"chat_key": "sk-live-parent-secret",
                        "embed_key": "sk-live-embed-secret"})
    payload = subagents._provider_config_without_secrets()
    blob = json.dumps(payload)
    assert "sk-live-parent-secret" not in blob
    assert "sk-live-embed-secret" not in blob
    # ...and the child is still told where to find one.
    assert payload["chat_key_env"] == "ASSISTANT_CHAT_KEY"
    assert payload["embed_key_env"] == "ASSISTANT_EMBED_KEY"


def test_a_scout_can_read_the_code_it_is_sent_at(ws, temp_db, monkeypatch):
    """Three real scouts investigating this project returned 7 claims with
    `evidence: 0` and `grounded_claims: 0` between them. Not laziness: a
    scout's actions were search, fetch-a-URL and report, and a local file is
    not a URL — so it was handed a map of names it could never open, and
    every claim it could make about code was a guess about a filename."""
    _fill_workspace(count=3)
    seen_payloads = []

    def fake_chat(system, user):
        payload = json.loads(user)
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            ids = [e["id"] for e in payload["code"]["entries"][:2]]
            return json.dumps({"action": "read", "chunk_ids": ids,
                               "why": "look at the bodies"})
        return json.dumps({"action": "report", "report": {
            "summary": "read it", "claims": [], "open_questions": []}})

    monkeypatch.setattr(subagents, "chat_complete", fake_chat)
    out = subagents._run_scout("what does f1 return", session_id=1, turn_idx=1)

    # The map it was given is a digest, not the corpus.
    assert "code" in seen_payloads[0]
    assert _NEEDLE not in json.dumps(seen_payloads[0])
    # ...and after one read, it actually holds source it can quote.
    read = seen_payloads[1]["code_you_have_read"]
    assert read and any(_NEEDLE in r["text"] for r in read)
    # Which means the claim it makes is supportable: the read is evidence.
    assert out["evidence"]
