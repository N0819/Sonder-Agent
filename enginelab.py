"""An engine of its own to break: scratch databases that run real test stories.

WHY THIS EXISTS. Reading a broken story out of `engine.db` (see `storydb`) ends
at a theory, and a theory about interactive fiction is cheap. The engine's
defects live in what agents do to each other's output across a turn, and the
only instrument that settles those is another turn — run against the changed
code, watched from the inside. Until now the assistant could edit
`Sonder_Engine_working` and could not run it, so every proposed repair was
argued rather than observed. That is the exact failure this repository names
first: a fix for a defect never observed failing cannot afterwards be told
apart from a fix for nothing.

WHAT A LAB IS. A directory with its own SQLite database, bootstrapped by the
engine's own `db.init()` so the schema is never a second implementation that
can drift, seeded with a small cast and a scenario, and driven one turn at a
time through `agents.runtime.run_pipeline` — the same entry point the server
uses. The live `engine.db` is opened read-only and only to copy configuration
out of. Nothing here can write to it.

WHY A SUBPROCESS, ALWAYS. Both projects have a top-level module named `db`.
This one imports it as `from db import q`, and so does the engine. Importing
the engine into this process would bind whichever was imported first and make
every subsequent query go to the wrong database — silently, because the
statements are similar enough to run. The isolation is not caution about
crashes; it is the only way both can exist at once.

WHY THE KEY NEVER COMES BACK. A lab needs real credentials to reach a model,
so `provision` copies the provider rows verbatim — inside the child process,
which writes them straight to the lab database and returns a count. The parent
never holds the string, so it cannot land in a return value, a trace, an
evidence excerpt or this public repository. `refdb` redacts the same columns on
the way out for the same reason. Two doors, one rule, neither relying on being
remembered.

WHY A RUN IS ASYNCHRONOUS. A turn is a dozen model calls and takes minutes to
tens of minutes. A fetch verb that blocks that long makes the assistant's own
turn look hung, and a timeout in the middle leaves a half-written database with
no record of why. `play` starts a detached run and returns a handle; `runs`
reports on it. The lab is on disk, so a run started in one turn is still there
to be read in the next — which is the same shape as an experiment, deliberately.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

import refdb

# Configuration copied out of the live engine. NOT `SELECT *`: `settings` holds
# the host password hash, its salt, the session secret and a third-party API
# key, and none of those have anything to do with running a story. An allowlist
# because the opposite — a denylist — silently admits every setting added
# later, and the one added later is the one nobody audits.
COPIED_SETTINGS = (
    "agent_models",          # which model plays which role. Without it, nothing runs.
    "active_preset",
    "prompt_presets",
    "max_output_tokens",
    "reasoning_effort",
    "openrouter_routing",
    "nsfw_enabled",
    "ambience_enabled",
    "backdrops_enabled",
)

# Written by the child on its last line so a result can be told apart from the
# engine's logging, which is voluminous, JSON-shaped, and on the same stream.
# Parsing "the last line that looks like JSON" would pick up a log record.
SENTINEL = "@@PONDER_LAB_RESULT@@"

MAX_LOG_TAIL = 6000
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,48}$")

_ROOT = ""
_SOURCE = ""
_ENGINE_DB = ""


def configure(root=None, source=None, engine_db=None):
    """Where labs live, which engine source runs them, which engine to copy from.

    Defaults come from the environment (`PONDER_LAB_ROOT`,
    `PONDER_ENGINE_SOURCE`, `PONDER_ENGINE_DB`).

    THE SOURCE DEFAULTS TO THE WORKING COPY, not to the installed engine. A lab
    exists to run code the assistant has just changed; pointing it at pristine
    source by default would make the common case — edit, then test — the one
    that silently measures something else.
    """
    global _ROOT, _SOURCE, _ENGINE_DB
    _ROOT = str(root or os.environ.get("PONDER_LAB_ROOT")
                or os.path.join(os.getcwd(), "workspaces", "labs"))
    _SOURCE = str(source or os.environ.get("PONDER_ENGINE_SOURCE") or "")
    _ENGINE_DB = str(engine_db or os.environ.get("PONDER_ENGINE_DB") or "")
    return {"root": _ROOT, "source": _SOURCE, "engine_db": _ENGINE_DB}


def _ensure():
    if not _ROOT:
        configure()
    return _ROOT


def _lab_dir(name):
    return os.path.join(_ensure(), str(name))


def _bad_name(name):
    if not NAME_OK.match(str(name or "")):
        return ("a lab name must be letters, digits, dot, dash or underscore "
                f"(1-49 chars) — {name!r} is not one. The name becomes a "
                "directory, so anything else is a path escape")
    return ""


def _meta(name, update=None):
    path = os.path.join(_lab_dir(name), "lab.json")
    data = {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    if update is not None:
        data.update(update)
        os.makedirs(_lab_dir(name), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)
    return data


def _source_for(name, override=None):
    """Which engine source tree this lab runs, and why it is that one."""
    if override:
        return str(override)
    meta = _meta(name)
    if meta.get("source"):
        return str(meta["source"])
    if _SOURCE:
        return _SOURCE
    for candidate in (os.path.join(os.getcwd(), "workspaces", "workspace",
                                   "Sonder_Engine_working"),):
        if os.path.isfile(os.path.join(candidate, "db.py")):
            return candidate
    return ""


def _engine_db_for(override=None):
    if override:
        return str(override)
    if _ENGINE_DB:
        return _ENGINE_DB
    # The reference database is already named and already resolved; asking
    # `refdb` rather than hardcoding a path means the two lanes cannot come to
    # disagree about which file is the engine.
    for entry in refdb.databases():
        if entry.get("name") == "engine" and entry.get("ok"):
            return refdb._DATABASES.get("engine", "")
    return ""


def _run_child(name, script, args, *, timeout, detach=False, label="run"):
    """Execute one generated script against the lab, in its own interpreter.

    Returns the parsed sentinel result, or an explicit failure carrying the tail
    of the log. THE LOG TAIL IS THE POINT: a child that dies during import says
    so in a traceback nobody would otherwise see, and "no result" alone is
    indistinguishable from "the engine refused the story".
    """
    problem = _bad_name(name)
    if problem:
        return {"ok": False, "error": problem}
    lab = _lab_dir(name)
    source = args.get("source") or ""
    if not source or not os.path.isfile(os.path.join(source, "db.py")):
        return {"ok": False, "error": f"no engine source at {source!r} — "
                                      f"there is no db.py there. Set it with "
                                      f"configure(source=...) or pass source="}
    runs_dir = os.path.join(lab, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    seq = 1 + max([int(m.group(1)) for m in
                   (re.match(r"(\d+)\.", f) for f in os.listdir(runs_dir))
                   if m] or [0])
    stem = os.path.join(runs_dir, f"{seq:04d}")
    with open(stem + ".script.py", "w") as fh:
        fh.write(script)
    with open(stem + ".args.json", "w") as fh:
        json.dump(args, fh, indent=1)
    log = open(stem + ".log", "w")
    env = dict(os.environ)
    env["ENGINE_DB"] = args["db"]
    env["PYTHONUNBUFFERED"] = "1"
    # The engine imports itself by top-level name from its own root, exactly as
    # this project does. Running from anywhere else finds this project's `db`.
    env["PYTHONPATH"] = source
    cmd = [sys.executable, stem + ".script.py", stem + ".args.json"]
    if detach:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        _meta(name, {f"{label}_pid": proc.pid, f"{label}_seq": seq})
        with open(stem + ".pid", "w") as fh:
            fh.write(str(proc.pid))
        return {"ok": True, "started": True, "run": f"{seq:04d}",
                "pid": proc.pid, "log": stem + ".log"}
    try:
        subprocess.run(cmd, cwd=source, env=env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        log.close()
        return {"ok": False, "run": f"{seq:04d}", "timed_out": True,
                "error": f"the child ran past {timeout}s and was killed",
                "log_tail": _tail(stem + ".log")}
    finally:
        if not log.closed:
            log.close()
    return _read_result(stem)


def _tail(path, limit=MAX_LOG_TAIL):
    try:
        with open(path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    # The engine logs one JSON record per line at INFO. Dropping them leaves
    # the tracebacks, which are the only lines anybody reads a failure log for.
    lines = [ln for ln in text.splitlines()
             if '"level": "INFO"' not in ln and not ln.startswith(SENTINEL)]
    tail = "\n".join(lines)
    return tail[-limit:] if len(tail) > limit else tail


def _read_result(stem):
    try:
        with open(stem + ".log", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return {"ok": False, "error": "the child left no log at all"}
    for line in reversed(lines):
        if line.startswith(SENTINEL):
            try:
                out = json.loads(line[len(SENTINEL):].strip())
            except ValueError as exc:
                return {"ok": False, "error": f"the child's result was not "
                                              f"JSON: {exc}",
                        "log_tail": _tail(stem + ".log")}
            out.setdefault("ok", True)
            out["run"] = os.path.basename(stem)
            if not out.get("ok"):
                out["log_tail"] = _tail(stem + ".log")
            return out
    return {"ok": False, "run": os.path.basename(stem),
            "error": "the child finished without writing a result — it died "
                     "before it got there. The log tail says where.",
            "log_tail": _tail(stem + ".log")}


# --------------------------------------------------------------------------
# The generated children. Each is a whole program: it takes one JSON file,
# binds the engine, does one thing, and prints one sentinel line.
# --------------------------------------------------------------------------

_PROVISION = '''\
import json, os, sqlite3, sys
A = json.load(open(sys.argv[1]))
os.environ["ENGINE_DB"] = A["db"]
sys.path.insert(0, A["source"])
out = {"ok": False}
try:
    import db
    db.configure(A["db"])
    db.init()
    from db import q, qi
    src = sqlite3.connect("file:%s?mode=ro" % A["engine_db"], uri=True)
    src.row_factory = sqlite3.Row
    provs = []
    for row in src.execute("SELECT * FROM providers"):
        key = row["api_key"] or ""
        qi("INSERT OR REPLACE INTO providers(id,name,kind,base_url,api_key,"
           "enabled) VALUES(?,?,?,?,?,?)",
           (row["id"], row["name"], row["kind"], row["base_url"], key,
            row["enabled"]))
        # The key is written and counted. It is never put in `provs`, which is
        # the only thing that leaves this process.
        provs.append({"id": row["id"], "name": row["name"],
                      "kind": row["kind"], "base_url": row["base_url"],
                      "enabled": bool(row["enabled"]),
                      "key_present": bool(key), "key_chars": len(key)})
    copied, missing = [], []
    for key in A["settings"]:
        r = src.execute("SELECT value FROM settings WHERE key=?",
                        (key,)).fetchone()
        if r is None:
            missing.append(key)
            continue
        qi("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)",
           (key, r["value"]))
        copied.append(key)
    src.close()
    ver = q("SELECT value FROM schema_meta WHERE key='version'", one=True)
    out = {"ok": True, "providers": provs, "settings_copied": copied,
           "settings_absent": missing,
           "schema_version": (ver or {})["value"] if ver else None,
           "tables": len(q("SELECT name FROM sqlite_master WHERE type='table'")
                         or [])}
except Exception as exc:
    import traceback
    out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
           "traceback": traceback.format_exc()[-2000:]}
print("''' + SENTINEL + ''' " + json.dumps(out))
'''

_SEED = '''\
import json, os, sys, time
A = json.load(open(sys.argv[1]))
os.environ["ENGINE_DB"] = A["db"]
sys.path.insert(0, A["source"])
out = {"ok": False}
try:
    import db
    db.configure(A["db"])
    db.init()
    from db import q, qi, wset
    from character_schema import normalize_character_data, character_name
    S = A["story"]
    pid = None
    if S.get("persona"):
        sheet = normalize_character_data(S["persona"])
        pid = qi("INSERT INTO personas(name,sheet,source) VALUES(?,?,?)",
                 (character_name(sheet), json.dumps(sheet), "{}"))
    cid = qi("INSERT INTO chats(name,persona_id,lorebook_id,scenario,created) "
             "VALUES(?,?,?,?,?)",
             (S.get("name") or "lab story", pid, None,
              S.get("scenario") or "", time.time()))
    cast = []
    for raw in (S.get("characters") or []):
        sheet = normalize_character_data(raw)
        chid = qi("INSERT INTO characters(name,sheet,source,created) "
                  "VALUES(?,?,?,?)",
                  (character_name(sheet), json.dumps(sheet), "{}", time.time()))
        qi("INSERT INTO chat_chars(chat_id,char_id,status,state) "
           "VALUES(?,?,?,?)", (cid, chid, "active", "{}"))
        cast.append(character_name(sheet))
    if pid is not None:
        qi("INSERT INTO chat_personas(chat_id,persona_id) VALUES(?,?)",
           (cid, pid))
    for key, value in (S.get("world") or {}).items():
        wset(cid, key, value)
    out = {"ok": True, "chat_id": cid, "persona_id": pid, "cast": cast,
           "world_keys": sorted((S.get("world") or {}).keys())}
except Exception as exc:
    import traceback
    out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
           "traceback": traceback.format_exc()[-2000:]}
print("''' + SENTINEL + ''' " + json.dumps(out))
'''

_PLAY = '''\
import json, os, sys, time
A = json.load(open(sys.argv[1]))
os.environ["ENGINE_DB"] = A["db"]
sys.path.insert(0, A["source"])
out = {"ok": False}
t0 = time.time()
try:
    import db
    db.configure(A["db"])
    from db import q, qi, transaction
    from agents.runtime import run_pipeline
    CID = A["chat_id"]
    with transaction():
        row = q("SELECT MAX(idx) AS m FROM turns WHERE chat_id=?", (CID,),
                one=True)
        idx = (row["m"] + 1) if row and row["m"] is not None else 0
        tid = qi("INSERT INTO turns(chat_id,idx,player_input,created,frame_id) "
                 "VALUES(?,?,?,?,?)", (CID, idx, A["text"], time.time(), None))
    try:
        from checkpoints import ensure_checkpoint
        ensure_checkpoint(CID, idx)
    except Exception:
        # A missing checkpoint costs rollback, not the turn. Failing the whole
        # run here would report a checkpoint defect as a pipeline defect.
        pass
    err = None
    try:
        for _ in run_pipeline(CID, tid):
            pass
    except Exception as exc:
        import traceback
        err = "%s: %s" % (type(exc).__name__, exc)
        out["traceback"] = traceback.format_exc()[-3000:]
    steps = []
    for s in q("SELECT id,key,label,ord,stale FROM steps WHERE turn_id=? "
               "ORDER BY ord", (tid,)) or []:
        v = q("SELECT content FROM variants WHERE step_id=? AND active=1",
              (s["id"],), one=True)
        steps.append({"key": s["key"], "ord": s["ord"],
                      "stale": bool(s["stale"]),
                      "chars": len(v["content"]) if v else 0,
                      "no_active_variant": v is None})
    narr = None
    r = q("SELECT v.content AS c FROM steps s JOIN variants v "
          "ON v.step_id=s.id AND v.active=1 WHERE s.turn_id=? AND s.key=?",
          (tid, "narrator"), one=True)
    if r:
        try:
            narr = (json.loads(r["c"]) or {}).get("prose")
        except ValueError:
            narr = None
    out.update({"ok": err is None, "chat_id": CID, "idx": idx,
                "turn_row_id": tid, "seconds": round(time.time() - t0, 1),
                "error": err, "steps": steps,
                "steps_run": len(steps),
                "prose": (narr or "")[:A.get("prose_chars", 4000)],
                "memories": (q("SELECT COUNT(*) AS n FROM memories "
                               "WHERE chat_id=? AND turn_idx=?", (CID, idx),
                               one=True) or {"n": 0})["n"]})
except Exception as exc:
    import traceback
    out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
           "traceback": traceback.format_exc()[-2000:],
           "seconds": round(time.time() - t0, 1)}
print("''' + SENTINEL + ''' " + json.dumps(out))
'''


# --------------------------------------------------------------------------
# The lane.
# --------------------------------------------------------------------------

def _turns_in(db_path):
    """How many turns the lab has actually played, asked of the lab itself."""
    if not os.path.isfile(db_path):
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        # A lab provisioned but never migrated has no `turns` table. That is
        # zero turns, not an error worth failing the whole listing over.
        return 0


def labs():
    """Every lab on disk, with whether it has been seeded and what it runs."""
    root = _ensure()
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        meta = _meta(name)
        db_path = os.path.join(path, "run.db")
        out.append({"name": name,
                    "provisioned": os.path.isfile(db_path),
                    "bytes": (os.path.getsize(db_path)
                              if os.path.isfile(db_path) else 0),
                    "chat_id": meta.get("chat_id"),
                    "story": meta.get("story"),
                    "source": meta.get("source"),
                    # COUNTED FROM THE DATABASE, not from a counter this lane
                    # increments. The counter only advanced on a blocking run,
                    # so a lab with two detached turns in it reported zero —
                    # and "no turns yet" is the one reading that makes an
                    # assistant re-seed a story it has already played.
                    "turns_played": _turns_in(db_path),
                    # A RUN IN FLIGHT IS THE STATE MOST WORTH SEEING HERE.
                    # Runs are detached and outlive the turn that started
                    # them, so without this the listing shows a lab that looks
                    # idle while a turn is halfway through writing it — and
                    # the obvious next move, starting another, is the one
                    # thing that makes the trace impossible to untangle.
                    **({"running": _running(name)}
                       if _running(name) else {})})
    return out


def provision(name, source=None, engine_db=None, reset=False, timeout=180):
    """Create a lab: fresh database, engine schema, engine's own configuration.

    `reset` deletes an existing one first. Without it, provisioning over a lab
    that already exists is refused rather than merged — a half-reseeded database
    is the kind of state that produces a result nobody can attribute.
    """
    problem = _bad_name(name)
    if problem:
        return {"ok": False, "error": problem}
    lab = _lab_dir(name)
    db_path = os.path.join(lab, "run.db")
    if os.path.isfile(db_path) and not reset:
        return {"ok": False, "error": f"lab {name!r} already exists with a "
                                      f"database. Pass reset=True to destroy "
                                      f"and rebuild it, or play into it as is",
                "lab": lab}
    if reset and os.path.isdir(lab):
        shutil.rmtree(lab, ignore_errors=True)
    os.makedirs(lab, exist_ok=True)
    src = _source_for(name, source)
    eng = _engine_db_for(engine_db)
    if not eng or not os.path.isfile(eng):
        return {"ok": False, "error": f"no engine database to copy settings "
                                      f"from at {eng!r} — a lab with no "
                                      f"provider cannot reach a model"}
    result = _run_child(name, _PROVISION,
                        {"db": db_path, "source": src, "engine_db": eng,
                         "settings": list(COPIED_SETTINGS)},
                        timeout=timeout, label="provision")
    if result.get("ok"):
        _meta(name, {"source": src, "engine_db": eng, "db": db_path,
                     "created": time.time(), "turns_played": 0})
        result["lab"] = lab
        result["db"] = db_path
        result["source"] = src
        result["note"] = ("provider credentials were copied into the lab "
                          "database by the child process and are not "
                          "returned here — key_chars is the whole of what "
                          "this lane will say about them")
    return result


def seed(name, story, timeout=180):
    """Put a story in the lab: persona, cast, scenario, per-chat world settings.

    `story` is {name, scenario, persona, characters[], world{}}. Sheets go
    through the engine's own `normalize_character_data`, so a sheet this
    accepts is one the engine accepts — writing a second normaliser here would
    be a second thing to keep in step with the first.
    """
    if not isinstance(story, dict):
        return {"ok": False, "error": "story must be an object with at least "
                                      "a scenario and one character"}
    if not str(story.get("scenario") or "").strip():
        return {"ok": False, "error": "a story needs a scenario — it is the "
                                      "opening situation every agent reads"}
    lab = _lab_dir(name)
    db_path = os.path.join(lab, "run.db")
    if not os.path.isfile(db_path):
        return {"ok": False, "error": f"lab {name!r} is not provisioned yet"}
    result = _run_child(name, _SEED,
                        {"db": db_path, "source": _source_for(name),
                         "story": story}, timeout=timeout, label="seed")
    if result.get("ok"):
        _meta(name, {"chat_id": result.get("chat_id"),
                     "story": story.get("name") or "lab story"})
    return result


def play(name, text="", chat_id=None, source=None, wait=False, timeout=2400):
    """Run one turn of the lab's story and record what every agent did.

    Detached by default: returns a handle immediately, and `runs(name)` reports
    on it. A turn is many model calls and routinely runs for minutes.

    `wait=True` blocks — for a test whose whole point is the result, and only
    where the caller can afford `timeout` seconds of nothing.
    """
    lab = _lab_dir(name)
    db_path = os.path.join(lab, "run.db")
    if not os.path.isfile(db_path):
        return {"ok": False, "error": f"lab {name!r} is not provisioned yet"}
    meta = _meta(name)
    cid = chat_id if chat_id is not None else meta.get("chat_id")
    if cid is None:
        return {"ok": False, "error": f"lab {name!r} has no story in it — "
                                      f"seed one first"}
    active = _running(name)
    if active:
        # Two pipelines writing one SQLite file interleave their turns, and the
        # trace afterwards cannot be untangled into two runs.
        return {"ok": False, "error": f"run {active['run']} is still going "
                                      f"(pid {active['pid']}) — one turn at a "
                                      f"time per lab", "running": active}
    result = _run_child(name, _PLAY,
                        {"db": db_path, "source": _source_for(name, source),
                         "chat_id": cid, "text": str(text or "")},
                        timeout=timeout, detach=not wait, label="play")
    if result.get("ok") and wait:
        _meta(name, {"turns_played": int(meta.get("turns_played", 0)) + 1})
    if result.get("ok") and not wait:
        result["note"] = ("started, not finished. Poll with runs(); a turn is "
                          "typically minutes.")
    return result


def _running(name):
    """The detached run still alive in this lab, if there is one."""
    runs_dir = os.path.join(_lab_dir(name), "runs")
    try:
        stems = sorted(f[:-4] for f in os.listdir(runs_dir)
                       if f.endswith(".pid"))
    except OSError:
        return None
    for stem in reversed(stems):
        try:
            with open(os.path.join(runs_dir, stem + ".pid")) as fh:
                pid = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        return {"run": stem, "pid": pid}
    return None


def runs(name, run=None, limit=8):
    """What the lab's runs did — finished, still going, or dead without a word.

    A RUN WITH NO RESULT AND NO PROCESS IS A FAILURE, and it is reported as
    one. The alternative is an entry that reads as pending forever, which is
    how a crashed child gets mistaken for a slow one.
    """
    runs_dir = os.path.join(_lab_dir(name), "runs")
    if not os.path.isdir(runs_dir):
        return {"ok": False, "error": f"lab {name!r} has no runs directory — "
                                      f"nothing has been run in it"}
    stems = sorted({f.split(".")[0] for f in os.listdir(runs_dir)
                    if re.match(r"^\d+\.", f)})
    if run:
        stems = [s for s in stems if s == str(run)]
        if not stems:
            return {"ok": False, "error": f"no run {run!r} in lab {name!r}"}
    out = []
    for stem in stems[-int(limit or 8):]:
        path = os.path.join(runs_dir, stem)
        entry = {"run": stem}
        pid = None
        try:
            with open(path + ".pid") as fh:
                pid = int(fh.read().strip())
        except (OSError, ValueError):
            pid = None
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        result = _read_result(path)
        if alive and not result.get("ok"):
            entry.update({"state": "running", "pid": pid,
                          "log_tail": _tail(path + ".log", 1200)})
        elif result.get("ok"):
            entry.update({"state": "done", **result})
        else:
            entry.update({"state": "failed", **result})
        try:
            entry["log_bytes"] = os.path.getsize(path + ".log")
        except OSError:
            entry["log_bytes"] = 0
        out.append(entry)
    return {"ok": True, "lab": name, "runs": out}


def stop(name):
    """Kill the lab's running turn. Returns what was killed, or says nothing was."""
    active = _running(name)
    if not active:
        return {"ok": True, "stopped": False, "note": "nothing was running"}
    try:
        os.killpg(os.getpgid(active["pid"]), signal.SIGTERM)
    except OSError as exc:
        return {"ok": False, "error": f"could not stop {active['pid']}: {exc}"}
    return {"ok": True, "stopped": True, **active}


def lab_query(name, sql, **kw):
    """Read the lab's own database with the same statement gate and caps.

    Registered with `refdb` under `lab:<name>` on the way in, so a lab is read
    through exactly the machinery that reads the engine — including the
    credential redaction, which matters more here than there: this database has
    a copied key in it.
    """
    db_path = os.path.join(_lab_dir(name), "run.db")
    if not os.path.isfile(db_path):
        return {"ok": False, "error": f"lab {name!r} is not provisioned yet"}
    key = f"lab:{name}"
    known = dict(refdb._DATABASES or {})
    if known.get(key) != db_path:
        known[key] = db_path
        refdb.configure(known)
    return refdb.query(key, sql, **kw)


def destroy(name):
    """Delete a lab and everything in it, after stopping whatever it is doing."""
    problem = _bad_name(name)
    if problem:
        return {"ok": False, "error": problem}
    lab = _lab_dir(name)
    if not os.path.isdir(lab):
        return {"ok": False, "error": f"no lab named {name!r}"}
    stop(name)
    shutil.rmtree(lab, ignore_errors=True)
    return {"ok": True, "destroyed": name}
