# app.py — FastAPI assembly. Thin by design: every route body is a call into
# a module that owns the behaviour, so the app layer cannot quietly grow a
# second copy of a rule (the engine's lesson about repeated filters applies
# to route handlers too).

import json

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import autoloop
import beliefs
import chunks
import config
import memory
import persona
import pipeline
import providers
import research
import subagents
import turnrun
import workspace
from db import q, state_get

app = FastAPI(title="Sonder Assistant")


# A chat message becomes an episode row, an FTS document, and an embedding
# payload sent over the network before the write lock — in full. Uvicorn caps
# nothing, so a multi-megabyte body went all the way through. Single-user, so
# this is a stray-script guard rather than a hostile-input one, but the cost
# of the guard is one line.
MAX_MESSAGE_CHARS = 32_000
MAX_PANEL_ROWS = 60


class ChatIn(BaseModel):
    text: str = Field(max_length=MAX_MESSAGE_CHARS)
    session_id: int | None = None


@app.post("/api/chat")
def chat(body: ChatIn):
    """The blocking turn. Kept because it is the honest shape for a script or
    a test: one call in, one finished turn out. The UI uses the streaming pair
    below; both run the same `pipeline.run_turn`, so there is one turn
    implementation and two ways to watch it, not two pipelines."""
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty message"}, status_code=400)
    return pipeline.run_turn(text, body.session_id)


@app.post("/api/chat/start")
def chat_start(body: ChatIn):
    """Begin a turn on a worker thread and return its id immediately, so the
    caller can watch it and stop it."""
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty message"}, status_code=400)
    run = turnrun.create(text, body.session_id)
    turnrun.start(run, lambda r: pipeline.run_turn(text, body.session_id,
                                                  run=r))
    return {"turn_run_id": run.id}


@app.post("/api/chat/auto")
def chat_auto(body: ChatIn):
    """Begin an automation run: turns chain until the assistant stops asking
    for another, the user halts, or it stops making progress.

    A separate route rather than a flag on `/api/chat/start`, because the two
    differ in the only way a caller cares about — when they end — and a client
    that got the mode wrong would either stop a long task after one turn or
    leave an unattended loop running against a provider bill."""
    text = (body.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty message"}, status_code=400)
    run = autoloop.start(text, body.session_id)
    return {"turn_run_id": run.id, "auto": True}


@app.post("/api/chat/{run_id}/say")
def chat_say(run_id: str, body: ChatIn):
    """Speak to a run that is already working.

    The whole point of the automation loop: a correction arriving mid-run is
    read at the next round boundary rather than queued behind work it would
    have changed. Reports whether it was actually delivered — a message typed
    into a run that has just finished has to become an ordinary new turn, and
    a client told "delivered" would have dropped it."""
    run = turnrun.get(run_id)
    if run is None:
        return JSONResponse({"error": "unknown turn"}, status_code=404)
    outcome = run.say(body.text)
    if outcome == "empty":
        return JSONResponse({"error": "empty message"}, status_code=400)
    return {"outcome": outcome, "status": run.status}


def resume_cursor(header, param):
    """Which event index a reconnecting client should be sent from.

    NO CURSOR AND A CURSOR OF ZERO ARE DIFFERENT THINGS, and conflating them
    ate the first event of every fresh stream. The default was the integer 0,
    which is falsey as a number and truthy as `"0"` — so a request with no
    `Last-Event-ID` at all took the resume branch and started at 1. It was
    invisible only because event 0 happens to render as nothing today; the day
    a first event carries something, it would go missing with no error.

    A function rather than an expression inside the route because that is what
    made it untestable, and untestable is how it shipped. An unparseable
    cursor replays from the start: a duplicated trail is a cosmetic fault and
    a skipped one is a lost answer."""
    cursor = header if header is not None else param
    if cursor is None or not str(cursor).strip():
        return 0
    try:
        # The id names the LAST event delivered, so resume from the next one.
        return max(0, int(str(cursor).strip()) + 1)
    except (TypeError, ValueError):
        return 0


@app.get("/api/chat/{run_id}/events")
def chat_events(run_id: str, request: Request):
    """Server-sent events for one turn: a step per stage, then `end`.

    RESUMABLE. `Last-Event-ID` is the browser's own reconnect mechanism —
    EventSource remembers the last id it received and sends it back
    automatically — so a dropped connection picks up after the step it already
    had rather than replaying the trail or losing it. The run never stopped;
    only the pipe did.

    `X-Accel-Buffering: no` because a buffering proxy in front of this would
    deliver the whole stream at the end, which is indistinguishable from the
    blocking endpoint and would make the feature look broken rather than
    absent."""
    run = turnrun.get(run_id)
    if run is None:
        # 404 is load-bearing: EventSource does NOT reconnect after a non-2xx,
        # so a run the registry has dropped ends the client's retry loop
        # instead of leaving it reconnecting forever against nothing.
        return JSONResponse({"error": "unknown turn"}, status_code=404)
    since = resume_cursor(request.headers.get("last-event-id"),
                          request.query_params.get("since"))
    return StreamingResponse(
        turnrun.sse(run, since=since), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.get("/api/chat/{run_id}")
def chat_status(run_id: str):
    """Where a turn got to. What a page reloaded mid-turn asks first.

    A stream can be re-opened, but only if the client still knows there is
    something to re-open — so the run id outlives the page and this is how it
    is redeemed."""
    run = turnrun.get(run_id)
    if run is None:
        return JSONResponse({"error": "unknown turn"}, status_code=404)
    return {"id": run.id, "status": run.status, "text": run.text,
            "events": len(run.events), "result": run.result,
            "error": run.error}


@app.delete("/api/chat/{run_id}")
def chat_halt(run_id: str):
    """Halt at the next stage boundary.

    Reports what actually happened — `halting`, `too_late`, `not_running` —
    rather than a bare acknowledgement. Once the commit transaction has begun
    the turn must finish, and a button that claimed otherwise would be lying
    about the one thing the user is watching."""
    run = turnrun.get(run_id)
    if run is None:
        return JSONResponse({"error": "unknown turn"}, status_code=404)
    return {"outcome": run.request_halt(), "status": run.status}


@app.get("/api/sessions")
def sessions():
    rows = q("SELECT s.id, s.title, s.created, COUNT(t.id) AS turns "
             "FROM sessions s LEFT JOIN turns t ON t.session_id=s.id "
             "GROUP BY s.id ORDER BY s.id DESC")
    return [dict(r) for r in rows]


@app.get("/api/sessions/{sid}/turns")
def session_turns(sid: int):
    rows = q("SELECT id, turn_idx, user_text, reply_text, trace, created "
             "FROM turns WHERE session_id=? ORDER BY turn_idx", (sid,))
    out = []
    for r in rows:
        row = dict(r)
        try:
            row["trace"] = json.loads(row["trace"] or "{}")
        except (TypeError, ValueError):
            row["trace"] = {}
        out.append(row)
    return out


@app.get("/api/memories")
def memories(query: str = "", limit: int = 40):
    """The host memory panel. before_turn_idx=None is deliberate and
    documented at the seam: nobody is deciding a turn here, so there is no
    future to withhold. count_access=False — browsing is not recall."""
    # Clamped at BOTH ends. `rows[-0:]` is `rows[0:]`, so limit=0 returned the
    # entire table, and a negative limit slipped past the cap that was only
    # applied on the search branch.
    limit = max(1, min(int(limit), MAX_PANEL_ROWS))
    if query.strip():
        rows = memory.search_memories(query, k=limit,
                                      current_turn_idx=None,
                                      chronological=True, count_access=False)
    else:
        rows = [memory._row_memory(r) for r in memory.visible_memory_rows(
            before_turn_idx=None, include_archived=True,
            include_retired=True)]
        rows.sort(key=lambda m: (m["turn_idx"] is None,
                                 m["turn_idx"] or 0, m["id"]))
        rows = rows[-limit:]
    for m in rows:
        m.pop("_vector", None)
    return rows


@app.get("/api/memories/retired")
def memories_retired():
    """What the assistant has set aside, and why. A human should be able to
    see what it decided to stop remembering — that visibility is what makes
    the capability safe to grant."""
    return {"retired": memory.retired_rows()}


class RestoreIn(BaseModel):
    event_keys: list[str] = []
    batch: str = ""


@app.post("/api/memories/restore")
def memories_restore(body: RestoreIn):
    return memory.restore_memories(body.event_keys, batch=body.batch)


class PurgeIn(BaseModel):
    batch: str = ""
    confirm: bool = False


@app.post("/api/memories/purge")
def memories_purge(body: PurgeIn):
    """HARD DELETE, host-only. Nothing the model emits reaches this route:
    retiring is a reversible relevance judgement, destroying the record is a
    different act and it belongs to the person whose records they are."""
    if not body.confirm:
        return {"ok": False, "error": "purging is irreversible; confirm it"}
    return memory.purge_retired(batch=body.batch)


@app.get("/api/hypotheses")
def hypotheses():
    return research.list_hypotheses()


@app.get("/api/hypotheses/{hid}/evidence")
def hypothesis_evidence(hid: int):
    return research.evidence_for(hid)


@app.get("/api/beliefs")
def belief_view():
    state = state_get("assistant")
    row = q("SELECT MAX(turn_idx) AS m FROM turns", one=True)
    turn_idx = (row["m"] or 0) if row else 0
    return {
        "user_model": beliefs.beliefs_for_payload(
            state.get("mind_models"), turn_idx),
        "active_hypotheses": beliefs.select_active_hypotheses(
            state.get("mind_models"),
            state.get("active_hypothesis_keys"), turn_idx)[0],
        "pending_ponder": state.get("pending_ponder") or "",
    }


@app.get("/api/persona")
def persona_get():
    sheet = persona.get_persona()
    return {"persona": sheet, "warnings": persona.persona_warnings(sheet)}


class PersonaIn(BaseModel):
    persona: dict


@app.put("/api/persona")
def persona_put(body: PersonaIn):
    warnings = persona.save_persona(body.persona or {})
    return {"ok": True, "warnings": warnings}


# ---- Settings: providers, and the keys they are NOT allowed to hold ----

@app.get("/api/settings")
def settings_get():
    """Configuration plus whether each named env var currently resolves.
    Never the value — see config.py for why that is not a convenience being
    withheld but the design."""
    return config.redacted_status()


class SettingsIn(BaseModel):
    settings: dict


@app.put("/api/settings")
def settings_put(body: SettingsIn):
    """`save_config` can REFUSE a field — a pasted credential in a key field
    is dropped rather than stored. Its warnings were discarded here, so the
    one save that most needed to say something said nothing and the page
    redrew looking saved."""
    _config, warnings = config.save_config(body.settings or {})
    return {**config.redacted_status(), "warnings": warnings}


class PresetIn(BaseModel):
    role: str
    preset: str


@app.post("/api/settings/preset")
def settings_preset(body: PresetIn):
    """Fill the fields for a known endpoint. Sets a base, a model and the NAME
    of a key variable — never a key."""
    config.apply_preset(body.role, body.preset)
    return config.redacted_status()


@app.post("/api/settings/rebuild-embeddings")
def settings_rebuild_embeddings(limit: int | None = None):
    """Re-embed rows stranded on an older embedding model.

    Incremental and re-runnable: call it again to continue. Without this,
    changing embedding provider means choosing between your existing memory
    bank and better retrieval."""
    return memory.rebuild_embeddings(limit=limit)


@app.post("/api/settings/test")
def settings_test():
    """Actually call the configured provider. A settings page that only
    validates the SHAPE of a configuration tells you nothing about the thing
    that usually breaks — the key, the binary, the login."""
    if not providers.chat_configured():
        return {"ok": False, "error": "no chat provider is configured"}
    try:
        raw = providers.chat_complete(
            'Reply with exactly this JSON and nothing else: {"reply":"ok"}',
            "Answer with the JSON object.")
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:400]}
    parsed = providers.parse_model_json(raw)
    return {"ok": True, "provider": config.get_config()["chat_provider"],
            "reply": (parsed or {}).get("reply") or str(raw)[:200]}


# ---- Files: what the user hands the assistant to look at ----

@app.get("/api/files")
def files_list(session_id: int):
    return {"files": workspace.list_files(session_id),
            "limits": {"per_file_bytes": workspace.MAX_UPLOAD_BYTES,
                       "workspace_bytes": workspace.MAX_WORKSPACE_BYTES},
            "used_bytes": workspace.workspace_bytes(session_id)}


@app.post("/api/files")
async def files_upload(session_id: int = Form(...),
                       files: list[UploadFile] = File(...)):
    """Drag-and-drop lands here. Each file is size-checked and name-folded
    before it touches the disk; a refusal is a reason, never a 500."""
    results = []
    for upload in files:
        data = await upload.read(workspace.MAX_UPLOAD_BYTES + 1)
        results.append({"filename": upload.filename,
                        **workspace.store_upload(session_id, upload.filename,
                                                 data)})
    # Map on the way in. A workspace whose chunks are built lazily is a
    # workspace whose first question after an upload sees an empty map and
    # concludes there is no code — the same silent-absence failure as a
    # mechanism that never fires.
    mapped = chunks.ingest_workspace(session_id)
    return {"results": results, "chunks": mapped,
            "files": workspace.list_files(session_id)}


class PathIn(BaseModel):
    session_id: int
    path: str


@app.post("/api/files/extract")
def files_extract(body: PathIn):
    """Unpack an archive inside the workspace. Every guard is in
    workspace.extract; this route only carries the answer."""
    out = workspace.extract(body.session_id, body.path)
    # An archive is the case that matters most: it is how a whole codebase
    # arrives, and it is the upload that most needs mapping rather than
    # ingesting.
    return {**out, "chunks": chunks.ingest_workspace(body.session_id),
            "files": workspace.list_files(body.session_id)}


@app.post("/api/files/delete")
def files_delete(body: PathIn):
    return {**workspace.delete_file(body.session_id, body.path),
            "files": workspace.list_files(body.session_id)}


# ---- Subagents: the permission the assistant cannot give itself ----

@app.get("/api/subagents")
def subagents_get():
    return {"allowance": subagents.allowance(),
            "requests": subagents.pending_requests(),
            "max_grant": subagents.MAX_GRANT,
            "kinds": list(subagents.KINDS)}


class GrantIn(BaseModel):
    kind: str
    count: int


@app.post("/api/subagents/grant")
def subagents_grant(body: GrantIn):
    """The only route that can increase an allowance, and it is reachable
    only from the UI. Nothing the model emits arrives here."""
    result = subagents.grant(body.kind, body.count)
    return {**result, "requests": subagents.pending_requests()}


@app.post("/api/subagents/revoke")
def subagents_revoke():
    return {**subagents.revoke_all(),
            "requests": subagents.pending_requests()}


@app.get("/api/subagents/archives")
def subagent_archives():
    """Finished deep-subagent runs kept for inspection.

    Forensics only: nothing here is reachable by the assistant's recall. What
    it kept of a subagent is the report it absorbed as testimony — see
    subagents.py for why the working-out is deliberately not retrievable."""
    return {"archives": subagents.list_archives(),
            "root": subagents.ARCHIVE_ROOT,
            "keep_runs": subagents.ARCHIVE_KEEP_RUNS}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
