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
