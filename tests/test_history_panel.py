# Prior sessions, and being able to get back to one.
#
# `sessionId` in the page began null and was only ever set from the reply of a
# turn that page had run, so closing the tab ended the conversation as far as
# the browser was concerned: every reload founded a new session and left the
# previous one unreachable. Twenty-nine of them accumulated that way. The
# turns were on disk the whole time and the routes to read them had existed
# since they were written — nothing in the interface ever called either one.

import os
import re

from fastapi.testclient import TestClient

import db


def _client():
    import app as app_module
    return TestClient(app_module.app)


def test_a_session_is_listed_by_what_was_said_in_it(temp_db):
    """`title` is empty on every session this project has ever recorded —
    nothing writes it — so a list keyed on it is twenty-nine rows reading
    "(untitled)". That is `persona_warnings` again: an empty field reads as
    present, and the list looks like a rendering bug rather than like missing
    data."""
    with db.transaction():
        db.q("INSERT INTO sessions (id, title, created) VALUES (1, '', 100.0)")
        db.q("INSERT INTO turns (session_id, turn_idx, user_text, reply_text,"
             " trace, created) VALUES (1, 1, ?, 'sure', '{}', 101.0)",
             ("what did the subagent conclude about the engine?",))
        db.q("INSERT INTO turns (session_id, turn_idx, user_text, reply_text,"
             " trace, created) VALUES (1, 2, 'and after that?', 'ok',"
             " '{}', 102.0)")
    rows = _client().get("/api/sessions").json()
    assert len(rows) == 1 and rows[0]["turns"] == 2
    # The FIRST thing said, not the last: it is what a person remembers a
    # conversation by, and the last line of a long session is usually a
    # follow-up that means nothing on its own.
    assert rows[0]["opened_with"].startswith("what did the subagent")


def test_a_session_that_committed_nothing_is_still_listed(temp_db):
    """A reload that started a thread and abandoned it is exactly the event
    this panel exists to make visible. Hiding the empty ones would hide the
    defect."""
    with db.transaction():
        db.q("INSERT INTO sessions (id, title, created) VALUES (7, '', 100.0)")
    rows = _client().get("/api/sessions").json()
    assert [r["id"] for r in rows] == [7]
    assert rows[0]["turns"] == 0 and not rows[0]["opened_with"]


def test_the_trace_comes_back_parsed_and_whole(temp_db):
    """The durable artefact of this project is not the reply, it is the
    trace — which memories surfaced, what the payload cost, what the
    subagents reported. Delivered as a JSON string it would render as a wall
    of braces, which is the same as not delivering it."""
    with db.transaction():
        db.q("INSERT INTO sessions (id, title, created) VALUES (1, '', 100.0)")
        db.q("INSERT INTO turns (session_id, turn_idx, user_text, reply_text,"
             " trace, created) VALUES (1, 1, 'hi', 'hello',"
             " '{\"payload_cost\": {\"total\": 8}}', 101.0)")
    turns = _client().get("/api/sessions/1/turns").json()
    assert turns[0]["trace"]["payload_cost"]["total"] == 8


def test_the_page_can_pick_a_session_back_up(temp_db):
    """A read-only history is a log file with a stylesheet. Recall, beliefs
    and the episode chain all key off the session id, so continuing a thread
    under a new one is a different conversation as far as the assistant is
    concerned — adopting the id is what makes the panel worth having."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat = open(os.path.join(root, "static/js/chat.js")).read()
    history = open(os.path.join(root, "static/js/history.js")).read()
    index = open(os.path.join(root, "static/index.html")).read()
    assert "function adoptSession(" in chat
    assert "sessionId = id" in chat
    assert "adoptSession(sid, turns)" in history
    # Browser globals, no modules: script ORDER in index.html is the whole
    # dependency graph, and history.js calls `readable` out of chat.js.
    order = re.findall(r'/static/js/(\w+)\.js', index)
    assert order.index("chat") < order.index("history")
    assert order.index("history") < order.index("app")


def test_swapping_the_session_mid_turn_is_refused(temp_db):
    """A turn commits against the id it started with. Changing it underneath
    would file the exchange in a thread it never happened in — the same class
    of provenance defect as `speaker` on an automation iteration."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat = open(os.path.join(root, "static/js/chat.js")).read()
    body = chat.split("function adoptSession(")[1].split("\n}")[0]
    assert "if (sending) return" in body
