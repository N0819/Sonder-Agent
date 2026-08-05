# Subagents, and the permission the assistant cannot give itself.
#
# The whole design rests on one property: every path that INCREASES an
# allowance is reachable only from the user, and every path the model can
# reach either spends one or asks for one. These tests are that property,
# stated as code — because a permission model whose enforcement lives in a
# prompt is a suggestion, and the difference is invisible until it matters.

import json
import os

import pytest

import codemap
import providers
import subagents
import workspace


@pytest.fixture
def stub_model():
    yield
    providers.set_chat_stub(None)


# ---- The grant ledger ----

def test_spawning_is_refused_before_anything_is_granted(temp_db, stub_model):
    """Explicit permission means explicit: the default is zero, and a spawn
    at zero never reaches a model at all."""
    providers.set_chat_stub(lambda s, u, **kw: '{"summary": "should not run"}')
    for kind in subagents.KINDS:
        out = subagents.spawn(kind, "do something", turn_idx=1)
        assert out["ok"] is False and out["denied"] is True


def test_the_allowance_is_readable_without_spending_anything(temp_db):
    """The assistant is required to consult its allowance rather than guess,
    which is only reasonable if consulting it is free. It is delivered in
    every turn payload for exactly that reason."""
    before = subagents.allowance()
    assert before["deep"]["remaining"] == 0
    subagents.grant("deep", 2)
    assert subagents.allowance()["deep"]["remaining"] == 2
    # reading it again changed nothing
    assert subagents.allowance()["deep"]["remaining"] == 2


def test_a_grant_is_spent_one_at_a_time_and_then_refuses(temp_db,
                                                         stub_model):
    """The decrement happens in deterministic code before any model is
    reached, so a model that ignores its allowance entirely still cannot
    exceed it."""
    providers.set_chat_stub(
        lambda s, u, **kw: json.dumps({"action": "report",
                                       "report": {"summary": "done"}}))
    subagents.grant("scout", 2)
    assert subagents.spawn("scout", "look something up", turn_idx=1)["ok"]
    assert subagents.spawn("scout", "look again", turn_idx=1)["ok"]
    third = subagents.spawn("scout", "and again", turn_idx=1)
    assert third["ok"] is False and third["denied"] is True


def test_a_request_grants_nothing(temp_db, stub_model):
    """The path the model can reach and the path that increases a budget
    have no overlap at all — two functions, not one with a flag."""
    subagents.record_request("deep", 3, "I need to refactor the parser", 1)
    assert subagents.allowance()["deep"]["remaining"] == 0
    pending = subagents.pending_requests()
    assert pending and pending[0]["kind"] == "deep"


def test_approving_a_request_clears_it(temp_db):
    subagents.record_request("scout", 2, "three lookups", 1)
    subagents.grant("scout", 2)
    assert subagents.pending_requests() == []
    assert subagents.allowance()["scout"]["remaining"] == 2


def test_a_grant_cannot_exceed_the_ceiling(temp_db):
    """A stray extra zero in the box is not a runaway."""
    out = subagents.grant("deep", subagents.MAX_GRANT["deep"] + 1)
    assert out["ok"] is False


def test_revoking_stops_everything_immediately(temp_db, stub_model):
    providers.set_chat_stub(
        lambda s, u, **kw: json.dumps({"action": "report",
                                       "report": {"summary": "done"}}))
    subagents.grant("scout", 5)
    subagents.revoke_all()
    assert subagents.spawn("scout", "x", turn_idx=1)["denied"] is True


def test_a_failed_spawn_still_spends_its_permission(temp_db, stub_model):
    """The conservative direction, deliberately. A crashed child that cost
    nothing would be a retry loop against the user's budget."""
    def explode(system, user, **kw):
        raise RuntimeError("provider died")

    providers.set_chat_stub(explode)
    subagents.grant("scout", 1)
    out = subagents.spawn("scout", "x", turn_idx=1)
    assert out["ok"] is False
    assert subagents.allowance()["scout"]["remaining"] == 0


# ---- Reports are provisional, like every other model output ----

def test_a_claim_citing_nothing_the_child_filed_loses_its_citation(temp_db):
    """Being a whole assistant does not buy a child trust. The seam that
    would have to be relaxed to let its citations through unchecked is the
    one that keeps the parent honest."""
    report = subagents.validate_report({
        "summary": "found it",
        "evidence": [{"url": "https://real.example/a", "excerpt": "the text",
                      "stance": "supports"}],
        "claims": [
            {"claim": "grounded one", "support": ["https://real.example/a"]},
            {"claim": "invented one", "support": ["https://never.fetched/"]},
        ],
    }, subagents.DEEP)
    grounded = {c["claim"]: c for c in report["claims"]}
    assert grounded["grounded one"]["grounded"] is True
    assert grounded["invented one"]["grounded"] is False
    assert grounded["invented one"]["support"] == []
    assert any("UNSUPPORTED" in w for w in report["warnings"])


def test_evidence_urls_are_folded_to_one_spelling(temp_db):
    """The same canonicalisation the research loop uses, for the same reason:
    a claim citing `HTTP://Ex.com/a/` must match evidence filed as
    `https://ex.com/a`, or every claim would read as ungrounded."""
    report = subagents.validate_report({
        "evidence": [{"url": "HTTP://WWW.Ex.com/a/?utm_source=x",
                      "excerpt": "text"}],
        "claims": [{"claim": "c", "support": ["https://ex.com/a"]}],
    }, subagents.SCOUT)
    assert report["claims"][0]["grounded"] is True


def test_a_report_enters_memory_as_testimony_not_experience(temp_db):
    """A subagent is a second mind. Absorbing its report as first-hand would
    be the information-layer collapse this whole project exists to prevent,
    at the one seam where a second mind actually exists — and README's reason
    for cutting the engine's information firewall ("no second mind to be kept
    out of") stops being true the moment this feature ships."""
    import db
    report = subagents.validate_report({
        "summary": "the parser drops tabs",
        "evidence": [{"url": "https://a.example/", "excerpt": "line 12"}],
        "claims": [{"claim": "tabs are dropped at line 12",
                    "support": ["https://a.example/"]}],
    }, subagents.DEEP)
    subagents.absorb(report, task="check the parser", turn_idx=1)
    rows = db.q("SELECT provenance, kind, content FROM memories")
    assert rows, "the report must reach memory"
    assert {r["provenance"] for r in rows} == {"told"}
    assert all("subagent" in r["content"] for r in rows)


def test_an_unsupported_claim_does_not_become_a_remembered_finding(temp_db):
    """It is recorded in the report and NOT minted as a claim row: "the
    subagent thought this and could not show why" must not read back later
    as something the assistant knows."""
    import db
    report = subagents.validate_report({
        "summary": "s",
        "claims": [{"claim": "a guess with no source", "support": []}],
    }, subagents.SCOUT)
    subagents.absorb(report, task="t", turn_idx=1)
    contents = [r["content"] for r in db.q("SELECT content FROM memories")]
    assert not any("a guess with no source" in c for c in contents)


# ---- Information routing ----

def test_the_parent_answers_from_memory_before_spending_a_scout(temp_db,
                                                                stub_model):
    """Memory is the cheaper answer whenever it is an answer at all.
    Escalating on anything less than genuine emptiness would turn every query
    into a spawn."""
    import memory
    for n in range(4):
        memory.add_memory("semantic", "told", 0.7,
                          f"the deploy pipeline uses buildkite, note {n}",
                          turn_idx=1, event_key=f"m:{n}")
    subagents.grant("scout", 1)
    out = subagents.answer_query("what does the deploy pipeline use",
                                 session_id=None, turn_idx=5)
    assert out["recalled"]
    assert "scout_report" not in out
    # nothing was spent
    assert subagents.allowance()["scout"]["remaining"] == 1


def test_a_scout_dispatched_for_a_child_spends_the_same_budget(temp_db,
                                                               stub_model):
    """A subagent commissioning a subagent must not be a way around the
    ledger."""
    providers.set_chat_stub(
        lambda s, u, **kw: json.dumps({"action": "report",
                                       "report": {"summary": "looked it up"}}))
    subagents.grant("scout", 1)
    out = subagents.answer_query("something nobody has ever mentioned",
                                 session_id=None, turn_idx=5)
    assert "scout_report" in out
    assert subagents.allowance()["scout"]["remaining"] == 0


def test_a_query_with_no_scout_available_says_so_rather_than_inventing(
        temp_db, stub_model):
    out = subagents.answer_query("an unknown thing", session_id=None,
                                 turn_idx=5)
    assert out.get("scout_unavailable")


# ---- Navigation ----

def test_the_codemap_indexes_symbols_and_finds_house_rules(tmp_path):
    """An agent handed a 400-file upload has two bad options: read everything
    or guess. The map is an INDEX so it can do neither — it says where things
    are and never what they mean."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "tokenizer.py").write_text(
        "MAX = 3\n\n\nclass Lexer:\n    def scan(self):\n        pass\n\n\n"
        "def tokenize(text):\n    return text.split()\n")
    (tmp_path / "AGENTS.md").write_text("# rules\n- never change signatures\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("function x(){}")

    out = codemap.build(str(tmp_path))
    paths = {f["path"] for f in out["files"]}
    assert "node_modules/junk.js" not in paths, "vendored code is not the code"
    entry = next(f for f in out["files"] if f["path"].endswith("tokenizer.py"))
    names = {s["name"] for s in entry["symbols"]}
    assert {"Lexer", "Lexer.scan", "tokenize", "MAX"} <= names
    assert [i["path"] for i in out["project_instructions"]] == ["AGENTS.md"]


def test_regex_extracted_symbols_are_labelled_as_such(tmp_path):
    """A symbol list that is 80% right is useful to NAVIGATE by and dangerous
    to reason from, so the agent is told which it is holding."""
    (tmp_path / "a.js").write_text("export function go(){}\nclass K{}\n")
    entry = next(f for f in codemap.build(str(tmp_path))["files"]
                 if f["path"] == "a.js")
    assert entry["symbols"]
    assert "signpost" in entry.get("caveat", "")


def test_a_python_file_that_does_not_parse_does_not_kill_the_map(tmp_path):
    (tmp_path / "broken.py").write_text("def (((:\n")
    (tmp_path / "fine.py").write_text("def ok():\n    pass\n")
    out = codemap.build(str(tmp_path))
    assert len(out["files"]) == 2
    fine = next(f for f in out["files"] if f["path"] == "fine.py")
    assert fine["symbols"]


# ---- Coordination: the parent divides the work, the engine checks it ----

def test_two_agents_cannot_be_assigned_the_same_file(temp_db):
    """Three agents at one codebase with no division of labour read the same
    entry point and, worse, produce three sets of edits to one file of which
    the last writer wins silently. Overlap is a property of the ASSIGNMENT,
    so it is checked rather than prompted against."""
    assignments, warnings = subagents.plan_assignments([
        {"kind": "deep", "task": "fix the tokenizer",
         "scope": ["tokenizer.py", "shared.py"]},
        {"kind": "deep", "task": "fix the parser",
         "scope": ["parser.py", "shared.py"]},
    ])
    assert assignments[0]["owns"] == ["tokenizer.py", "shared.py"]
    assert assignments[1]["owns"] == ["parser.py"]
    assert any("shared.py" in w for w in warnings)


def test_every_agent_is_told_what_its_siblings_are_doing(temp_db):
    """The roster matters as much as the boundary: an agent that knows a
    sibling is already reproducing the bug does not reproduce it again."""
    assignments, _ = subagents.plan_assignments([
        {"kind": "deep", "task": "fix the tokenizer", "scope": ["tok.py"]},
        {"kind": "deep", "task": "update the docs", "scope": ["README.md"]},
    ])
    brief = subagents.coordination_brief(assignments[0])
    assert "YOU OWN" in brief and "tok.py" in brief
    assert "update the docs" in brief and "README.md" in brief


def test_a_change_outside_an_agents_assignment_is_refused(temp_db):
    """This is what makes coordination structural. A child assigned
    tokenizer.py that returns a change to app.py is refused at validation —
    the boundary is enforced when writing, not merely described in a
    prompt."""
    assignments, _ = subagents.plan_assignments([
        {"kind": "deep", "task": "fix the tokenizer", "scope": ["tok.py"]}])
    report = subagents.validate_report({
        "summary": "done",
        "coding_notes": [{"path": "tok.py", "what_changed": "split on \\s+",
                          "why": "tabs were dropped"}],
        "file_changes": [
            {"path": "tok.py", "content": "ok"},
            {"path": "app.py", "content": "should be refused"},
        ],
    }, subagents.DEEP, assignment=assignments[0])
    assert [c["path"] for c in report["file_changes"]] == ["tok.py"]
    assert any("app.py" in w and "refused" in w for w in report["warnings"])


# ---- Collaboration: siblings talk, brokered, only when it is job relevant ----

def test_a_message_reaches_a_sibling_holding_the_same_file(temp_db):
    assignments, _ = subagents.plan_assignments([
        {"kind": "deep", "task": "fix the tokenizer",
         "scope": ["pkg/tok.py"]},
        {"kind": "deep", "task": "fix the lexer", "scope": ["pkg/lex.py"]},
    ])
    cohort = subagents.Cohort(assignments, session_id=None, turn_idx=1)
    out = cohort.route(0, {"text": "the shared helper strips tabs"})
    # same directory counts as the same space
    assert out["delivered_to"] == ["fix the lexer"]
    assert cohort.take_mail(1)[0]["text"].startswith("the shared helper")
    assert cohort.take_mail(1) == [], "mail is handed over once"


def test_an_irrelevant_message_is_dropped_with_a_reason(temp_db):
    """An irrelevant message costs the recipient a turn, so it is not
    delivered — and the sender is told why rather than assuming it landed."""
    assignments, _ = subagents.plan_assignments([
        {"kind": "deep", "task": "fix the tokenizer in src",
         "scope": ["src/tok.py"]},
        {"kind": "deep", "task": "rewrite the billing invoice template",
         "scope": ["billing/invoice.html"]},
    ])
    cohort = subagents.Cohort(assignments, session_id=None, turn_idx=1)
    out = cohort.route(0, {"text": "tabs are dropped by the tokenizer"})
    assert out["delivered_to"] == []
    assert out["considered"] and out["considered"][0]["why"]
    assert cohort.take_mail(1) == []


def test_relevance_can_be_earned_by_the_message_itself(temp_db):
    """Agents in different directories still share a job when the message
    names what the other is working on."""
    assignments, _ = subagents.plan_assignments([
        {"kind": "deep", "task": "fix the tokenizer", "scope": ["a/tok.py"]},
        {"kind": "deep", "task": "update the tokenizer documentation",
         "scope": ["docs/tokenizer.md"]},
    ])
    cohort = subagents.Cohort(assignments, session_id=None, turn_idx=1)
    out = cohort.route(0, {"text": "tokenizer documentation is now wrong: it "
                                   "splits on whitespace not spaces"})
    assert out["delivered_to"]


# ---- Coding notes and the work surviving the worker ----

def test_a_changed_file_is_written_back_into_the_workspace(temp_db, tmp_path):
    """WITHOUT THIS THE CODING WORK IS LOST. The child edits inside a
    directory deleted the moment it reports, so "I fixed the tokenizer" would
    describe a fix that exists nowhere — the parent receives the sentence and
    not the change."""
    workspace.configure(str(tmp_path / "ws"))
    workspace.store_upload(1, "tok.py", b"old contents")
    report = subagents.validate_report({
        "summary": "fixed",
        "coding_notes": [{"path": "tok.py", "what_changed": "split on \\s+",
                          "why": "the reproduction dropped tabs"}],
        "file_changes": [{"path": "tok.py", "content": "new contents"}],
    }, subagents.DEEP)
    written, refused = subagents.apply_changes(report, 1)
    assert written == ["tok.py"] and not refused
    root = workspace.session_root(1)
    with open(os.path.join(root, "tok.py")) as handle:
        assert handle.read() == "new contents"


def test_a_change_cannot_escape_the_workspace(temp_db, tmp_path):
    """A child is a model, and its output is provisional here as everywhere
    else."""
    workspace.configure(str(tmp_path / "ws"))
    report = subagents.validate_report({
        "file_changes": [{"path": "../../ESCAPED.txt", "content": "pwned"},
                         {"path": "/etc/passwd", "content": "pwned"}],
    }, subagents.DEEP)
    # Refused at VALIDATION, and refused rather than rewritten: `lstrip("./")`
    # turned "../../ESCAPED.txt" into "ESCAPED.txt", so the write succeeded
    # against a file nobody named while the report claimed something else.
    assert report["file_changes"] == []
    assert len([w for w in report["warnings"] if "refused" in w]) == 2
    written, refused = subagents.apply_changes(report, 1)
    assert written == []
    assert not os.path.exists(str(tmp_path / "ESCAPED.txt"))


def test_why_a_file_changed_becomes_a_memory(temp_db):
    """The diff stays in the tree; the reasoning is gone in six weeks unless
    something keeps it. So it is minted as an ordinary memory row, retrievable
    by the same machinery as anything else."""
    import db
    report = subagents.validate_report({
        "summary": "s",
        "coding_notes": [{"path": "tok.py", "what_changed": "split on \\s+",
                          "why": "the reproduction showed tabs were dropped",
                          "evidence": "experiment:abc123"}],
    }, subagents.DEEP)
    subagents.absorb(report, task="fix tokenizer", turn_idx=1)
    rows = db.q("SELECT provenance, content FROM memories")
    note = [r for r in rows if "tok.py" in r["content"]]
    assert note and note[0]["provenance"] == "told"
    assert "experiment:abc123" in note[0]["content"]
    assert "tabs were dropped" in note[0]["content"]


def test_a_change_with_no_note_is_flagged(temp_db):
    """An unexplained change cannot be reviewed, and the reason is not
    recoverable later."""
    report = subagents.validate_report({
        "file_changes": [{"path": "tok.py", "content": "x"}],
    }, subagents.DEEP)
    assert any("no coding note" in w for w in report["warnings"])


def test_a_stale_map_is_updated_alongside_the_code(temp_db, tmp_path):
    """A map that has silently drifted from the code is worse than no map,
    because the next agent will trust it."""
    workspace.configure(str(tmp_path / "ws"))
    workspace.store_upload(1, "AGENTS.md", b"# old\n")
    report = subagents.validate_report({
        "coding_notes": [{"path": "tok.py", "what_changed": "x", "why": "y"}],
        "file_changes": [{"path": "tok.py", "content": "new"}],
        "map_updates": [{"path": "AGENTS.md", "content": "# new\n",
                         "why": "module list no longer matched"}],
    }, subagents.DEEP)
    written, _ = subagents.apply_changes(report, 1)
    assert set(written) == {"tok.py", "AGENTS.md"}


# ---- Archiving: forensics, deliberately unreachable by the assistant ----

def test_a_deep_run_is_archived_and_a_scout_has_nothing_to_archive(
        temp_db, tmp_path, monkeypatch):
    """Archive the evidence, not the mind. A finished deep subagent leaves a
    scratch database, a transcript and its experiments — worth keeping when a
    report later looks wrong. A scout leaves nothing: no database, no sandbox,
    no bank, one call whose entire product is the report and whose evidence is
    already absorbed."""
    monkeypatch.setattr(subagents, "ARCHIVE_ROOT", str(tmp_path / "arch"))
    home = tmp_path / "run"
    home.mkdir()
    (home / "subagent.db").write_bytes(b"scratch")
    run_id = subagents.archive_run(str(home), kind=subagents.DEEP,
                                   task="fix the tokenizer",
                                   report={"summary": "did it"}, turn_idx=3,
                                   seconds=12.0)
    assert run_id
    listed = subagents.list_archives()
    assert listed and listed[0]["task"] == "fix the tokenizer"
    assert listed[0]["summary"] == "did it"
    # A scout never creates a working directory in the first place.
    import inspect
    assert "archive_run" not in inspect.getsource(subagents._run_scout)


def test_an_archive_is_not_reachable_by_recall(temp_db, tmp_path,
                                               monkeypatch):
    """The distinction the whole feature rests on: a subagent's bank holds its
    own episodes, beliefs and half-formed inferences. Making those retrievable
    would mean the parent recalling a dead agent's private working-out as
    though it were its own experience."""
    import memory
    monkeypatch.setattr(subagents, "ARCHIVE_ROOT", str(tmp_path / "arch"))
    home = tmp_path / "run"
    home.mkdir()
    (home / "subagent.db").write_bytes(b"scratch")
    subagents.archive_run(str(home), kind=subagents.DEEP,
                          task="a distinctive archived phrase",
                          report={"summary": "s"}, turn_idx=1, seconds=1.0)
    found = memory.search_memories("a distinctive archived phrase",
                                   current_turn_idx=None)
    assert found == []


def test_the_archive_is_bounded(temp_db, tmp_path, monkeypatch):
    """An unbounded audit trail is a disk-full outage wearing the word
    'thorough'."""
    monkeypatch.setattr(subagents, "ARCHIVE_ROOT", str(tmp_path / "arch"))
    monkeypatch.setattr(subagents, "ARCHIVE_KEEP_RUNS", 3)
    for n in range(6):
        home = tmp_path / f"run{n}"
        home.mkdir()
        (home / "f.txt").write_text(f"run {n}")
        subagents.archive_run(str(home), kind=subagents.DEEP, task=f"t{n}",
                              report={}, turn_idx=n, seconds=1.0)
    assert len(subagents.list_archives()) <= 3


def test_a_deep_child_can_find_the_files_it_was_seeded(temp_db, tmp_path,
                                                       monkeypatch):
    """EVERY DEEP SUBAGENT RAN AGAINST AN EMPTY WORKSPACE. The parent seeded
    `<home>/workspace` and set the child's ASSISTANT_WORKSPACE to the same
    path — but `workspace_root()` joins `_WORKSPACE_DIR` onto that, so the
    child looked one level down and found nothing.

    The symptom was not a missing file. It was a child reporting, honestly and
    at length, that it had no tool capable of opening local code: with no
    files there were no chunks, with no chunks there were no ids, and the one
    read channel it had resolved to nothing. Three subagent runs were spent
    concluding the tool was absent when the corpus was.

    Asserted against what `_run_deep` ACTUALLY does — the directory it seeds
    versus the directory the child's own ASSISTANT_WORKSPACE resolves to.
    A test that computes both sides itself agrees with itself and would have
    passed throughout."""
    import workspace
    workspace.configure(str(tmp_path / "workspaces"))
    monkeypatch.setattr(subagents, "ARCHIVE_ROOT",
                        str(tmp_path / "subagent-archive"))
    workspace.store_upload(1, "coding.py", b"def judge():\n    return 1\n")
    seen = {}

    real_seed = subagents._seed

    def spy_seed(root, files):
        seen["seeded"] = root
        return real_seed(root, files)

    class FakeProc:
        stdin, stdout, returncode = None, None, 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def spy_popen(*args, **kwargs):
        seen["env"] = kwargs.get("env") or {}
        return FakeProc()

    monkeypatch.setattr(subagents, "_seed", spy_seed)
    monkeypatch.setattr(subagents.subprocess, "Popen", spy_popen)
    monkeypatch.setattr(subagents, "_converse", lambda *a, **kw: None)
    subagents._run_deep("look at coding.py", session_id=1, turn_idx=1)

    assert seen.get("seeded"), "nothing was seeded"
    child_root = workspace.root_under(seen["env"]["ASSISTANT_WORKSPACE"])
    assert seen["seeded"] == child_root, (
        f"seeded {seen['seeded']} but the child reads {child_root}")
