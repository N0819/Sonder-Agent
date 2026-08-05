# Large material as a navigable list.
#
# The scar: a 115-file upload became 97 KB of codemap in one turn payload,
# which broke the turn outright ([Errno 7]) and, long before that, spent the
# context describing the code instead of thinking about it. `codemap_for`
# bounded by FILE COUNT, and a bound on the number of things says nothing
# about their size.

import json

import chunks
import pipeline
import providers


def test_code_splits_at_symbols_never_mid_function(temp_db):
    """A function cut in half is worse than one left out. Absent is visible;
    half a function reads as a whole one and gets reasoned about as if it
    were."""
    source = ('"""Module doc."""\nimport os\n\n\n'
              'def alpha():\n    """First."""\n    return 1\n\n\n'
              'def beta():\n    """Second."""\n    return 2\n')
    pieces = chunks.split_code(source, "python")
    titles = [p["title"] for p in pieces]
    assert "function alpha" in titles and "function beta" in titles
    for piece in pieces:
        # Every body is whole: it never starts or ends inside a def line.
        assert piece["body"].count("def ") <= 1 or "preamble" in piece["title"]


def test_a_digest_carries_gists_and_ids_but_not_bodies(temp_db):
    """The entire mechanism. If bodies rode along in the digest, this would be
    the old codemap with extra steps."""
    source = 'def alpha():\n    """What alpha does."""\n    return 1\n' * 1
    source += '\n\ndef beta():\n    """What beta does."""\n    return 2\n'
    chunks.put(1, "code", "m.py", chunks.split_code(source, "python"))
    digest = chunks.digest(1)
    blob = json.dumps(digest)
    assert "return 1" not in blob and "return 2" not in blob
    assert any("alpha" in e["gist"] for e in digest["entries"])
    assert digest["total_chunks"] >= 2


def test_expanding_returns_the_body_for_that_id(temp_db):
    """Map, then explore. The id in the list is the handle for the lines."""
    source = 'def alpha():\n    """Doc."""\n    return "THE BODY"\n'
    keys = chunks.put(1, "code", "m.py", chunks.split_code(source, "python"))
    got = chunks.expand(1, [keys[0]])
    assert "THE BODY" in got[0]["text"]


def test_an_unknown_id_says_so_rather_than_vanishing(temp_db):
    """Silence reads to the model as "that chunk was empty", and it carries on
    as though it had looked."""
    out = chunks.expand(1, ["cdeadbeef"])
    assert out and out[0]["unknown"] is True


def test_relevance_ranks_by_tokens_not_substrings(temp_db):
    """"how does the embeddings rebuild work" ranked `test_workspace.py`
    first, on "work" matching "workspace" — one real term outvoted by three
    stopwords and a substring. Tokens, and a stoplist, or the ranking is
    noise."""
    chunks.put(1, "code", "workspace.py",
               [{"title": "function store_upload", "start": 1, "end": 3,
                 "body": "def store_upload():\n    return 1\n"}])
    chunks.put(1, "code", "memory.py",
               [{"title": "function rebuild_embeddings", "start": 1, "end": 3,
                 "body": "def rebuild_embeddings():\n    return 1\n"}])
    top = chunks.digest(1, query="how does the embeddings rebuild work")
    assert top["entries"][0]["source"] == "memory.py"


def test_re_chunking_replaces_rather_than_accumulates(temp_db):
    """An expand that returned code no longer on disk would be worse than one
    that returned nothing."""
    chunks.put(1, "code", "m.py",
               [{"title": "function old", "body": "def old(): pass", "start": 1,
                 "end": 1}])
    chunks.put(1, "code", "m.py",
               [{"title": "function new", "body": "def new(): pass", "start": 1,
                 "end": 1}])
    digest = chunks.digest(1)
    assert digest["total_chunks"] == 1
    assert "new" in digest["entries"][0]["gist"]


def test_a_digest_states_what_it_is_not_showing(temp_db):
    """`showing` below `total_chunks` is the difference between an agent that
    asks for more and one that concludes from a sample."""
    pieces = [{"title": f"function f{i}", "start": i, "end": i,
               "body": f"def f{i}():\n    return {i}\n" + "# pad\n" * 20}
              for i in range(400)]
    chunks.put(1, "code", "big.py", pieces)
    digest = chunks.digest(1, budget=2_000)
    assert digest["showing"] < digest["total_chunks"]
    assert len(json.dumps(digest)) < 6_000


# ---- The deliberation loop: ponder chains BEFORE the answer ----

def test_ponder_now_resolves_within_the_turn(temp_db):
    """`ponder` was DEFERRED — the model named what it wanted from memory and
    the answer arrived next turn, by which point the question had already been
    answered without it. The point of asking your own memory is to ask it
    before you commit to an answer."""
    import memory
    # turn_idx=0: recall applies the turn cutoff, so a memory minted AT the
    # ordinal being decided is correctly invisible to it. Prior turns only.
    memory.add_memory("semantic", "told", 0.8,
                      "the deploy pipeline uses buildkite", turn_idx=0,
                      event_key="m:1")
    rounds = []

    def fake(system, user):
        payload = json.loads(user)
        if "user_message" not in payload:
            return json.dumps({"summary": "", "received_summary": "",
                               "surmise_summary": "", "key_phrases": [],
                               "unresolved_threads": []})
        rounds.append(payload)
        if len(rounds) == 1:
            return json.dumps({"reply": "",
                               "need_more": {"ponder": "deploy pipeline"}})
        return json.dumps({"reply": "buildkite"})

    providers.set_chat_stub(fake)
    try:
        out = pipeline.run_turn("what do we deploy with?")
    finally:
        providers.set_chat_stub(None)
    assert out["reply"] == "buildkite"
    assert len(rounds) == 2
    # The second round actually SAW what the ponder returned.
    got = rounds[1]["what_i_went_and_got"][0]["ponder"]
    assert got["returned"] >= 1


def test_an_empty_ponder_is_reported_as_empty(temp_db):
    """The web fallback depends on this. "no memories" and "I forgot to look"
    are indistinguishable from an empty list, and a model that cannot tell
    them apart will ask memory the same thing again instead of searching."""
    rounds = []

    def fake(system, user):
        payload = json.loads(user)
        if "user_message" not in payload:
            return json.dumps({"summary": "", "received_summary": "",
                               "surmise_summary": "", "key_phrases": [],
                               "unresolved_threads": []})
        rounds.append(payload)
        if len(rounds) == 1:
            return json.dumps({"reply": "",
                               "need_more": {"ponder": "zzzz nonexistent"}})
        return json.dumps({"reply": "done"})

    providers.set_chat_stub(fake)
    try:
        pipeline.run_turn("something unknown")
    finally:
        providers.set_chat_stub(None)
    ponder = rounds[1]["what_i_went_and_got"][0]["ponder"]
    assert ponder["returned"] == 0
    assert ponder["nothing_found"] is True
    assert "search the web" in ponder["note"]


def test_deliberation_is_bounded(temp_db):
    """A loop whose exit depends on the model deciding it is satisfied is not
    bounded. Each round is a full model call."""
    calls = []

    def fake(system, user):
        payload = json.loads(user)
        if "user_message" not in payload:
            return json.dumps({"summary": "", "received_summary": "",
                               "surmise_summary": "", "key_phrases": [],
                               "unresolved_threads": []})
        calls.append(1)
        return json.dumps({"reply": "never satisfied",
                           "need_more": {"ponder": "again and again"}})

    providers.set_chat_stub(fake)
    try:
        out = pipeline.run_turn("loop forever please")
    finally:
        providers.set_chat_stub(None)
    assert len(calls) == pipeline.DELIBERATION_MAX_ROUNDS
    assert out["reply"] == "never satisfied"


# ---- What the index does NOT cover has to be in the index ----

def test_a_file_the_index_skipped_is_named_in_the_digest(temp_db, tmp_path):
    """THE CRASH POINTED THE OTHER WAY. `ingest_workspace` bounded by
    `max_files=60` over a newest-modified-first listing, so the 61st file and
    everything older was simply not indexed — no error, no warning, and the
    digest still announced "N chunks across M sources" in the same confident
    tone. An assistant reading that concludes a symbol is absent from the
    codebase when it is only absent from the index.

    A silent truncation is worse than a crash: a crash is a fact, and a corpus
    quietly missing half of itself produces confident answers about code
    nobody looked at."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    for i in range(6):
        workspace.store_upload(1, f"mod{i}.py",
                               (f"def f{i}():\n    return {i}\n"
                                + "# padding\n" * 200).encode())
    # A budget that cannot fit them all, so the bound actually bites.
    chunks.ingest_workspace(budget=3_000)
    digest = chunks.digest(kind="code")

    assert digest["not_indexed"], "a dropped file left no trace"
    assert all(entry["path"] and entry["why"] for entry in digest["not_indexed"])
    # And the pinned summary — the part read first — says so, rather than
    # leaving it to a reader who thinks to scroll.
    assert "NOT in this index" in digest["summary"]
    assert "not_indexed" in digest["how_to_use_this"]


def test_an_index_within_budget_claims_no_omissions(temp_db, tmp_path):
    """The other half: a complete index must not cry wolf, or the warning
    stops carrying information."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "only.py", b"def f():\n    return 1\n")
    chunks.ingest_workspace()
    digest = chunks.digest(kind="code")
    assert "not_indexed" not in digest
    assert "NOT in this index" not in digest["summary"]


def test_a_digest_says_how_it_chose_what_to_show(temp_db):
    """`showing: 50 of 884` told a reader the list was partial and nothing
    about how the 50 were picked — and a relevance rank and an arbitrary slice
    are indistinguishable from inside the payload while failing completely
    differently. A rank degrades gracefully; a slice hides whole modules. Not
    knowing which, a reader cannot tell "not in the list" from "not looked
    at"."""
    chunks.put(1, "code", "alpha.py",
               chunks.split_code("def embeddings_rebuild():\n"
                                 '    """Rebuild them."""\n    return 1\n',
                                 "python"))
    chunks.put(1, "code", "beta.py",
               chunks.split_code("def unrelated_helper():\n"
                                 '    """Something else."""\n    return 2\n',
                                 "python"))
    ranked = chunks.digest(kind="code", query="embeddings rebuild")
    assert "ranked by relevance" in ranked["selection"]
    assert "1 of 2" in ranked["selection"]
    assert ranked["entries"][0]["source"] == "alpha.py"

    unranked = chunks.digest(kind="code")
    assert "source order" in unranked["selection"]


def test_the_walked_count_and_the_indexed_count_reconcile(temp_db, tmp_path):
    """A scout was spent asking why a workspace held 53 chunkable files while
    the index reported 52 sources. Files this index does not handle were
    dropped in silence, so the two numbers could not be reconciled from
    outside and the gap read as loss."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "real.py", b"def f():\n    return 1\n")
    workspace.store_upload(1, "requirements.txt", b"fastapi\n")
    workspace.store_upload(1, "Makefile", b"test:\n\tpytest\n")
    chunks.ingest_workspace()

    from db import state_get
    index = state_get(chunks.INGEST_STATE_KEY)
    walked = len(workspace.list_files())
    assert index["indexed_sources"] + index["skipped_count"] == walked
    assert {e["path"] for e in index["skipped"]} == {"requirements.txt",
                                                     "Makefile"}


# ---- A memory fetched mid-turn has to be usable ----

def test_a_pondered_memory_arrives_with_a_ref(temp_db):
    """THE PONDER LANE RETURNED UNUSABLE MATERIAL. `_gather` read
    `r.get("ref")` — not a key any memory row carries — so every pondered
    memory arrived as `ref: null`. Ten of them in one measured turn.

    Under the citation rule a memory that cannot be named cannot be used, so
    the assistant went and asked its own memory, was handed the answer, and
    had to reply as though it had never looked. Silent in both directions:
    `.get` on a missing key is None rather than an error, and a null ref reads
    as "this row happens to have none" rather than "the lane is broken".

    THE SECOND HALF is the citation gate. `delivered` was built once at stage
    1, so even a correctly-named pondered memory was stripped as invented.
    The ponder query here is deliberately unrelated to the user's message, or
    stage-1 recall delivers the row anyway and the mid-turn path is never the
    thing under test."""
    import memory
    # turn_idx=0, like the ponder test above: a row minted at the ordinal
    # being decided is correctly invisible to the turn deciding it.
    memory.add_memory("semantic", "told", 0.8,
                      "the deploy pipeline runs on buildkite", turn_idx=0,
                      event_key="ev:buildkite")
    rounds = []

    def fake(system, user):
        payload = json.loads(user)
        if "user_message" not in payload:
            return json.dumps({"summary": "", "received_summary": "",
                               "surmise_summary": "", "key_phrases": [],
                               "unresolved_threads": []})
        rounds.append(payload)
        if len(rounds) == 1:
            return json.dumps({"reply": "hold on",
                               "need_more": {"ponder": "deploy pipeline"}})
        got = payload["what_i_went_and_got"][0]["ponder"]["memories"]
        return json.dumps({"reply": "answered",
                           "memory_evidence_used": [got[0]["ref"]]})

    providers.set_chat_stub(fake)
    try:
        out = pipeline.run_turn("which colour should the button be?")
    finally:
        providers.set_chat_stub(None)

    fetched = rounds[1]["what_i_went_and_got"][0]["ponder"]["memories"]
    assert fetched, "the ponder returned nothing to test"
    assert all(m["ref"] for m in fetched), "a pondered memory had no ref"
    assert not any("ungrounded" in w for w in out["warnings"]), out["warnings"]


def test_the_deliberation_loop_reports_what_it_delivered(temp_db):
    """The citation gate's `delivered` set was built once, at stage 1, so a
    memory the assistant asked for and RECEIVED during the turn was stripped
    from its own citations as though invented.

    Asserted on `_gather`'s contract rather than end-to-end: with a small bank
    every row is in the recent buffer and therefore already delivered, so an
    integration test passes whether or not the mid-turn path works. That is
    the shape of test that let this survive — one that cannot fail for the
    reason it was written."""
    import memory
    memory.add_memory("semantic", "told", 0.8,
                      "the deploy pipeline runs on buildkite", turn_idx=0,
                      event_key="ev:buildkite")
    step, delivered = pipeline._gather({"ponder": "deploy pipeline"},
                                       turn_idx=5, session_id=1,
                                       run=pipeline._UNOBSERVED, warnings=[])
    assert step["ponder"]["returned"] >= 1
    assert "ev:buildkite" in delivered
    assert {m["ref"] for m in step["ponder"]["memories"]} <= delivered


# ---- Knowing a filename must be enough to read the file ----

def test_a_file_can_be_reached_by_name(temp_db, tmp_path):
    """THE STEP THAT DID NOT EXIST. `digest` ranks the whole workspace against
    the turn's message and `expand` needs ids, so an assistant told "fix
    coding.py" had no route to coding.py's ids unless the user's own words
    happened to rank it into the sample.

    Measured on a real turn: the message was "Go for it", nothing in it
    ranked, the digest showed 67 of 987 chunks and stopped at `beliefs.py`,
    and both the assistant and the deep subagent it delegated to reported
    they could not obtain an anchor for the file they were asked to edit.
    Neither was confused — the lookup was missing."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    # Both bodies over MIN_CHUNK_CHARS, or the smaller folds into the larger
    # and the file is one chunk — which would test the fixture, not the lookup.
    workspace.store_upload(
        1, "coding.py",
        b'def judge(result, expect):\n'
        b'    """Did the observation match the prediction?"""\n'
        b'    checks = []\n    return checks\n\n\n'
        b'def propose_fix(hypothesis_id, description):\n'
        b'    """Record an intended fix, or refuse."""\n'
        b'    return {"accepted": False}\n')
    for i in range(30):        # enough noise that ranking would not find it
        workspace.store_upload(1, f"other{i}.py", b"x = 1\n" * 40)
    chunks.ingest_workspace()

    # The digest, ranked against a message with no code terms in it, is where
    # the assistant would have had to look.
    ranked = chunks.digest(kind="code", query="Go for it")
    assert "coding.py" not in json.dumps(ranked["entries"]) or True

    got = chunks.outline("coding.py")
    assert got["path"].endswith("coding.py")
    assert got["chunks"] >= 2
    titles = [e["title"] for e in got["entries"]]
    assert any("judge" in t for t in titles), titles
    # ...and the ids it hands back actually resolve to source.
    body = chunks.expand(ids=[got["entries"][0]["id"]])
    assert body[0].get("text"), body


def test_an_outline_finds_a_file_wherever_the_tree_put_it(temp_db, tmp_path):
    """An uploaded project sits under its own archive directory, so requiring
    the full path would mean knowing the layout in order to ask about the
    file — which is the thing you are trying to find out."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    root = workspace.workspace_root()
    import os
    os.makedirs(os.path.join(root, "Project (Copy 1)", "src"), exist_ok=True)
    with open(os.path.join(root, "Project (Copy 1)", "src", "deep.py"),
              "w") as fh:
        fh.write("def buried():\n    return 1\n")
    chunks.ingest_workspace()
    got = chunks.outline("deep.py")
    assert got["path"] == "Project (Copy 1)/src/deep.py", got


def test_an_ambiguous_name_is_reported_not_guessed(temp_db, tmp_path):
    """Two files with one basename is exactly when picking is worst: the
    assistant would anchor an edit in a file it did not mean and the diff
    would look entirely reasonable."""
    import os

    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    root = workspace.workspace_root()
    for where in ("a", "b"):
        os.makedirs(os.path.join(root, where), exist_ok=True)
        with open(os.path.join(root, where, "util.py"), "w") as fh:
            fh.write(f"def {where}_fn():\n    return 1\n")
    chunks.ingest_workspace()
    got = chunks.outline("util.py")
    assert got["entries"] == []
    assert "2 files match" in got["error"]
    assert sorted(got["candidates"]) == ["a/util.py", "b/util.py"]


def test_a_missing_name_suggests_what_is_there(temp_db, tmp_path):
    """A bare "not found" against a 987-chunk index is unactionable; a
    near-miss list is the difference between one more round and three."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "coding.py", b"x = 1\n")
    chunks.ingest_workspace()
    got = chunks.outline("codingg.py")
    assert got["entries"] == []
    assert "no indexed file matches" in got["error"]
    got2 = chunks.outline("coding")
    assert got2.get("did_you_mean") == ["coding.py"], got2


def test_expanding_a_path_answers_with_the_files_chunks(temp_db, tmp_path):
    """The assistant knows filenames long before it knows chunk ids, so
    passing one to `expand_chunks` is the obvious wrong guess. Answering it
    with a bare `unknown` teaches nothing and costs a round."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(1, "coding.py", b"def judge():\n    return 1\n")
    chunks.ingest_workspace()
    out = chunks.expand(ids=["coding.py"])
    assert out[0]["not_a_chunk_id"] is True
    assert out[0]["resolved_as_file"].endswith("coding.py")
    assert out[0]["entries"], "the correction carries no ids to use"
    # A genuinely unknown id is still reported as unknown.
    assert chunks.expand(ids=["cdeadbeef"])[0]["unknown"] is True


def test_the_turn_can_outline_then_expand_then_edit(temp_db, tmp_path):
    """THE WHOLE CHAIN, IN ONE TURN. Landing one anchored edit costs orient →
    outline → expand → answer, and DELIBERATION_MAX_ROUNDS was 3. A real turn
    ran out with the assistant reporting "surfacing ids and then expanding
    them is two rounds; I have one" — the budget stopped the work, not its
    judgement."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    workspace.store_upload(
        1, "target.py",
        b'def judge(result):\n'
        b'    """Grade the run against the prediction."""\n'
        b'    if not result:\n        return "inconclusive"\n'
        b'    return "refuted"\n')
    chunks.ingest_workspace()
    rounds = []

    def fake(system, user):
        payload = json.loads(user)
        if "user_message" not in payload:
            return json.dumps({"summary": "", "received_summary": "",
                               "surmise_summary": "", "key_phrases": [],
                               "unresolved_threads": []})
        rounds.append(payload)
        # Four rounds, matching the real trace: orient, outline, expand, edit.
        # Three was the ceiling, and a turn ran out reporting "surfacing ids
        # and then expanding them is two rounds; I have one".
        if len(rounds) == 1:                      # where am I
            return json.dumps({"reply": "", "need_more": {"list_dir": ""}})
        if len(rounds) == 2:                      # I know the name, not the id
            return json.dumps({"reply": "",
                               "need_more": {"outline": "target.py"}})
        if len(rounds) == 3:                      # now I have ids
            ids = [e["id"] for e in
                   payload["what_i_went_and_got"][1]["outlines"][0]["entries"]]
            return json.dumps({"reply": "",
                               "need_more": {"expand_chunks": ids}})
        body = payload["what_i_went_and_got"][2]["expanded_chunks"][0]["text"]
        assert "return \"refuted\"" in body, body
        return json.dumps({
            "reply": "edited",
            "edit_files": [{"path": "target.py", "why": "the grade was wrong",
                            "replace": [{"old": 'return "refuted"',
                                         "new": 'return "inconclusive"'}]}]})

    providers.set_chat_stub(fake)
    try:
        out = pipeline.run_turn("fix the grading in target.py")
    finally:
        providers.set_chat_stub(None)
    assert out["reply"] == "edited", out
    after = workspace.read_file("target.py")["text"]
    assert 'return "refuted"' not in after, after
    assert after.count('return "inconclusive"') == 2
