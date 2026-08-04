# Durability and concurrency. Every test here pins a defect that the other 80
# tests could not see, because a test is one thread holding one connection and
# a connection can always read its own uncommitted writes. That blind spot is
# the point: the suite was green while nothing the app wrote ever reached the
# disk.

import sqlite3
import threading

import pytest

import db
import memory
import pipeline
import providers


@pytest.fixture
def stub_model():
    providers.set_chat_stub(lambda system, user, **kw: '{"reply": "ok"}')
    yield
    providers.set_chat_stub(None)


def test_a_committed_turn_is_visible_to_another_connection(tmp_path,
                                                           stub_model):
    """Nothing ever committed. sqlite3's legacy isolation_level opened an
    implicit transaction before every write, so `qi`'s `if not in_transaction`
    guard could never fire and `transaction()` mistook itself for nested and
    skipped its COMMIT too. A whole session of chat lived in one open write
    transaction: a second thread saw zero rows and got "database is locked",
    and closing the handle rolled all of it back."""
    path = str(tmp_path / "durable.db")
    db.configure(path)
    pipeline.run_turn("remember the deadline is April 3")

    other = sqlite3.connect(path, timeout=2.0)
    try:
        assert other.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1
        assert other.execute("SELECT COUNT(*) FROM memories").fetchone()[0] > 0
        # And the writer is not still holding the lock.
        other.execute("INSERT INTO sessions(title,created) VALUES('x',1)")
        other.commit()
    finally:
        other.close()
        db.close()


def test_rows_survive_closing_the_connection(tmp_path, stub_model):
    """`db.close()` rolled back everything, because the turn's transaction was
    still open. A restart of uvicorn lost the entire memory bank."""
    path = str(tmp_path / "survive.db")
    db.configure(path)
    pipeline.run_turn("my name is Nathan")
    db.close()

    after = sqlite3.connect(path)
    try:
        assert after.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1
    finally:
        after.close()


def test_concurrent_turns_get_unique_ordinals(tmp_path, stub_model):
    """`SELECT MAX(turn_idx)` then INSERT is a read-modify-write with no lock
    between the halves. Two turns both read N, both claimed N+1, and both
    minted event_key "turn:N+1:episode" — so the second turn's upsert silently
    OVERWROTE the first turn's episode and one exchange vanished from memory
    with no warning. Two browser tabs was enough."""
    db.configure(str(tmp_path / "race.db"))
    ordinals, errors = [], []

    def turn(n):
        try:
            ordinals.append(pipeline.run_turn(f"message {n}")["turn_idx"])
        except Exception as exc:                      # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=turn, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(set(ordinals)) == len(ordinals) == 8
    assert db.q("SELECT COUNT(*) AS c FROM turns", one=True)["c"] == 8
    # Every episode kept its own row rather than overwriting a namesake.
    episodes = db.q("SELECT COUNT(*) AS c FROM memories "
                    "WHERE event_key LIKE 'turn:%:episode'", one=True)["c"]
    assert episodes == 8
    db.close()


def test_a_failed_commit_leaves_no_partial_turn(tmp_path, stub_model,
                                                monkeypatch):
    """The turns row was INSERTed and autocommitted before the model call, so
    it sat outside the "all durable turn mutations in one transaction"
    invariant. An exception mid-commit rolled the memories back and left a
    turn with a user message, no reply, and a consumed ordinal."""
    db.configure(str(tmp_path / "partial.db"))

    def boom(*args, **kwargs):
        raise RuntimeError("commit stage exploded")

    monkeypatch.setattr(memory, "reconcile_inference_confidence", boom)
    with pytest.raises(RuntimeError):
        pipeline.run_turn("this turn should leave nothing behind")

    assert db.q("SELECT COUNT(*) AS c FROM turns", one=True)["c"] == 0
    assert db.q("SELECT COUNT(*) AS c FROM memories", one=True)["c"] == 0
    db.close()


def test_state_is_re_read_inside_the_write_lock(tmp_path, stub_model):
    """Stage 1 read the state blob, the model call took seconds, and stage 5
    wrote the whole blob back. Two overlapping turns meant the later writer
    silently erased the earlier one's belief updates, hypothesis keys and
    pending ponder. Read-modify-write is only atomic if the read is inside."""
    db.configure(str(tmp_path / "state.db"))
    pipeline.run_turn("first")
    # A write that lands between another turn's stage 1 and stage 5 must
    # survive that turn's commit.
    state = db.state_get("assistant")
    state["written_by_someone_else"] = True
    db.state_put("assistant", state)
    pipeline.run_turn("second")
    assert db.state_get("assistant").get("written_by_someone_else") is True
    db.close()
