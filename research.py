# research.py — questions become hypotheses; searching gathers evidence;
# evidence moves confidence; contradiction becomes a dispute, never an
# average; and the final answer must cite the evidence it actually used.
#
# This is the engine's dispute/citation discipline promoted from a rarely-
# occasioned corner (fictional deception) to the centre of the product. The
# pieces and their lineage:
#
#   hypothesis      an `active_hypothesis` with a confidence, engine-style —
#                   the field the model reads is `i_suspect`, so its own
#                   conjecture can never read back as settled fact.
#   evidence row    cites a real URL, and is ALSO minted as a `read`-
#                   provenance memory, so a source consulted last month is
#                   retrievable later and can dispute a newer claim.
#   confidence      moved deterministically by the ENGINE from stances, with
#                   bounded convex steps (affect.py's bounded-update
#                   machinery in spirit: clamped step, blend toward evidence,
#                   never a jump to the extreme on one data point).
#   dispute         two contradictory sources are held side by side with the
#                   confidence pinned in the middle band. Averaging two
#                   incompatible claims produces a number that represents
#                   neither; the engine's record_dispute keeps both readings,
#                   and so does this.
#   grounding       a conclusion may cite only delivered evidence/memory
#                   refs. Ungrounded citations are dropped with a warning —
#                   and a conclusion that loses ALL grounding is rejected,
#                   which sends the loop back to work.
#
# The research loop runs AUTOMATED internal rounds (search → fetch → ponder →
# re-weigh) until the model produces a grounded conclusion at adequate
# confidence or the budget runs out — at which point the engine assembles a
# deterministic, explicitly hedged answer from the evidence table rather than
# letting the model improvise a confident one. "Satisfying" is a checkable
# predicate, not a feeling: grounded citations + confidence over the bar.

import hashlib
import json
import time
import urllib.parse

import memory
import tools_web
from db import q, qi

# Bounded-update constants. One supporting source closes 35% of the distance
# to certainty; one contradicting source explains away up to 45% of what is
# there. Both bounded (engine _MAX_STEP spirit) so no single page can settle
# or demolish a question — revision needs repeated evidence.
_SUPPORT_PLASTICITY = 0.35
_CONTRADICT_SUPPRESSION = 0.45
_MAX_STEP = 0.4

# A dispute pins confidence into the middle band: the honest number for "two
# live sources disagree" is one that neither side would accept as victory.
_DISPUTE_BAND = (0.35, 0.65)

# The loop's budget and bar. Rounds are model calls; the cap is what makes
# "ponder strategically until satisfied" terminate. CONCLUDE_CONFIDENCE is
# the bar a conclusion must clear to end the loop early — below it the loop
# keeps working until budget, then hedges.
# SIZED FOR THE JOB, NOT FOR A CHARACTER'S PATIENCE.
#
# Six was a reasonable number of beats to spend in a story. It is a poor number
# of sources to settle a real question on, and the loop's own termination
# predicate already stops early the moment the answer is grounded and confident
# -- so a higher ceiling costs nothing on easy questions and is the whole
# difference on hard ones. What bounds this is the user's time and the token
# budget, not a claim about how long a mind can concentrate.
RESEARCH_MAX_ROUNDS = 16
CONCLUDE_CONFIDENCE = 0.6


# Where every hypothesis starts. Named rather than spelled 0.3 inline,
# because a confidence is only readable against the prior it moved from: two
# hypotheses at 0.545 read as a shared default until you can see that both
# opened at 0.3 and each took one supporting step. The payload carries this
# number for exactly that reason.
PRIOR_CONFIDENCE = 0.3


def _clamp(v, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def open_hypothesis(question, turn_idx, session_id=None):
    """A question under investigation starts at PRIOR_CONFIDENCE, not 0.5: an
    unresearched hypothesis leans toward "I do not know yet", and the
    asymmetry means supporting evidence must actually arrive before the
    statement reads as likely."""
    hid = qi("INSERT INTO hypotheses(session_id,question,statement,confidence,"
             "status,created_turn,updated_turn) VALUES(?,?,?,?,?,?,?)",
             (session_id, str(question or "").strip(), "", PRIOR_CONFIDENCE,
              "open", turn_idx, turn_idx))
    return get_hypothesis(hid)


def get_hypothesis(hid):
    row = q("SELECT * FROM hypotheses WHERE id=?", (hid,), one=True)
    if not row:
        return None
    out = dict(row)
    try:
        out["dispute"] = json.loads(row["dispute"]) if row["dispute"] else None
    except (TypeError, ValueError):
        out["dispute"] = None
    return out


def list_hypotheses(status=None, limit=50):
    sql = "SELECT * FROM hypotheses"
    args = ()
    if status:
        sql += " WHERE status=?"
        args = (status,)
    rows = q(sql + " ORDER BY updated_turn DESC, id DESC LIMIT ?",
             args + (limit,))
    return [get_hypothesis(r["id"]) for r in rows]


def evidence_tally(hid):
    """How many rows of each stance are behind a hypothesis's confidence.

    A NUMBER WITHOUT ITS DENOMINATOR IS UNREADABLE. Four open hypotheses were
    once shown at confidence 0.545 apiece and read, reasonably, as four
    questions sharing an untouched default — the one thing that would have
    settled it, that each had received exactly one supporting row, was not in
    the payload. 0.3 + 0.35 * (1 - 0.3) is 0.545: identical values from
    identical starting points and identical arithmetic, which is the
    mechanism working, and indistinguishable from the mechanism never having
    fired. `evidence: {}` and `evidence: {"supports": 1}` are the difference,
    and both are worth knowing."""
    return {r["stance"]: r["c"] for r in q(
        "SELECT stance, COUNT(*) AS c FROM evidence WHERE hypothesis_id=? "
        "GROUP BY stance", (hid,))}


def _evidence_ref(row_id):
    return f"ev:{row_id}"


def evidence_for(hid):
    rows = q("SELECT * FROM evidence WHERE hypothesis_id=? ORDER BY id",
             (hid,))
    return [{**dict(r), "ref": _evidence_ref(r["id"])} for r in rows]


# One source, one spelling.
#
# INHERITED THE HARD WAY. Sonder's costliest recurring defect was that a single
# being routinely carried two names at once -- a display name and an entity id
# -- and every comparison written as `==` between them silently answered False.
# Five separate defects in one investigation were that one comparison, and one
# of them was a firewall failing OPEN. The lesson recorded there is blunt: an
# engine in which one thing can carry two names cannot use `==` to decide what
# anything is, and a guard that must be REMEMBERED is a guard that will be
# forgotten -- so fold the data instead of adding a helper people must call.
#
# A URL is that same shape. These are six spellings of one page:
#
#     http://example.com/a      https://example.com/a
#     https://www.example.com/a https://example.com/a/
#     https://example.com/a?utm_source=x     https://example.com/a#part2
#
# Evidence idempotency keyed on the raw string, so each spelling opened its own
# row and each fresh row moved confidence by up to _SUPPORT_PLASTICITY. Six
# fetches of one page could carry a hypothesis from 0.30 to past 0.90 on a
# single source -- which is exactly the "repetition wearing corroboration"
# failure the idempotency check was written to prevent, arriving through the
# door the check left open.
#
# Canonicalised on the way IN, and the canonical form is what is stored, so
# plain equality is correct on stored rows because there is nothing left for it
# to be wrong about.

# Tracking parameters carry no information about WHICH page this is. Stripped
# rather than kept, because two links to one article differing only in campaign
# tags are one source and must count once.
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "ref",
    "ref_src", "igshid", "spm", "_hsenc", "_hsmi", "yclid", "s_cid",
})


_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def _normalize_escapes(text):
    """RFC 3986 percent-encoding normalisation: hex digits uppercased, and
    escapes of unreserved characters decoded.

    `%2f` and `%2F` are the same octet and `%7Euser` is `~user` — the RFC
    says so normatively, so these cannot possibly be two documents. They were
    two spellings to the idempotency check, which is the whole failure mode
    `canonical_url` exists to close: four fetches of two real pages carried a
    hypothesis to 0.875 where the honest number was 0.704."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "%" and i + 2 < len(text):
            hexpair = text[i + 1:i + 3]
            try:
                byte = int(hexpair, 16)
            except ValueError:
                out.append(ch)
                i += 1
                continue
            char = chr(byte)
            out.append(char if char in _UNRESERVED
                       else "%" + hexpair.upper())
            i += 3
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def canonical_url(raw):
    """One page's many spellings reduced to the one this project stores.

    Deliberately conservative. It folds only differences that CANNOT change
    which document is served -- scheme case, host case, a `www.` prefix, the
    default port, tracking parameters, a fragment, a trailing slash on a path
    -- and leaves everything else alone. Query parameters that are not
    tracking are kept and sorted: `?page=2` is a different page and folding it
    would merge two sources into one, which is the worse error in both
    directions (a lost source AND an inflated count).

    http vs https is folded to https. They can in principle serve different
    documents; in practice one redirects to the other, and treating a redirect
    pair as two corroborating sources is the failure this exists to stop.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    # Non-web refs pass through untouched. `experiment:<digest>` and any other
    # internal scheme is an opaque identifier, not a location, and there is no
    # such thing as two spellings of it -- caught by the suite's own first run,
    # which is the intended way to find this class of thing.
    scheme = text.split(":", 1)[0].lower() if ":" in text else ""
    if scheme and scheme not in ("http", "https"):
        return text
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urllib.parse.urlsplit(text)
        port = parts.port
    except ValueError:
        return text
    host = (parts.hostname or "").lower()
    # A trailing dot is the explicit form of the same absolute name.
    host = host.rstrip(".")
    # IDN and punycode are the same host BY DEFINITION -- a search backend
    # returns `xn--bcher-kva.example`, a model retypes `bücher.example`, and
    # a bare string comparison called them two sources. Same class of defect
    # as the six URL spellings, same fix: fold on the way in.
    try:
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        pass
    if host.startswith("www."):
        host = host[4:]
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    path = _normalize_escapes(parts.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query,
                                                      keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS]
    query = urllib.parse.urlencode(sorted(kept))
    # No fragment: it selects a place WITHIN a document, not a document.
    return urllib.parse.urlunsplit(("https", host, path, query, ""))


def same_source(left, right):
    """Do these two links name the same page? Never a bare `==`."""
    left, right = canonical_url(left), canonical_url(right)
    return bool(left) and left == right


def record_evidence(hid, *, url, title, excerpt, stance, turn_idx,
                    claim_hint=""):
    """One evidence row + its `read` memory + the deterministic confidence
    move. `stance` is the model's judgement (supports/contradicts/context) —
    a judgement call the engine cannot make lexically — but what the stance
    DOES to confidence is engine arithmetic, and the URL is recorded verbatim
    so the judgement stays auditable against its source."""
    hyp = get_hypothesis(hid)
    if hyp is None:
        return None
    stance = stance if stance in ("supports", "contradicts", "context") \
        else "context"
    # Canonical BEFORE anything keys on it -- the idempotency check below, the
    # event_key, and the memory row all have to agree about what "the same
    # page" means, and they can only agree if there is one spelling.
    url = canonical_url(url)
    # THE EXCERPT SAYS WHEN IT IS A PARTIAL COPY — the same scar as
    # `source_chars`, one table over, and it cost two turns of an audit before
    # anyone saw it. An experiment's observation is written through here, so a
    # long stdout was clipped at 600 characters with nothing anywhere saying
    # so: the numbers being measured looked complete and were not.
    #
    # Marked in the TEXT as well as counted in the column, because the text is
    # what the model reads back and a column it is never shown cannot warn it.
    excerpt = " ".join(str(excerpt or "").split())
    excerpt_chars = len(excerpt)
    if excerpt_chars > 600:
        marker = f" …[cut: {excerpt_chars} chars total]"
        excerpt = excerpt[:600 - len(marker)] + marker
    # Idempotency: the same page cited for the same hypothesis is one row.
    # Without this, a loop that re-fetches a good source pumps confidence by
    # repetition — the popularity loop again, wearing "corroboration".
    digest = hashlib.sha1(f"{hid}|{url}".encode()).hexdigest()[:16]
    event_key = f"evidence:{digest}"
    existing = q("SELECT id, stance, excerpt FROM evidence "
                 "WHERE hypothesis_id=? AND url=?", (hid, url), one=True)
    flipped = False
    if existing:
        # A RE-READ THAT REVERSES ITSELF IS A DISPUTE, NOT AN EDIT.
        #
        # This branch used to overwrite stance and excerpt in place and skip
        # `_apply_stance` entirely (it ran only when `fresh`), so the same
        # source read again with the opposite stance destroyed the earlier
        # reading and left the confidence the OLD stance had already bought.
        # The coding suite drives exactly this path: a re-run experiment
        # hashes to the same `experiment:<digest>` url, so "passed, then
        # failed" — the single most informative thing a flaky test can tell
        # you — silently became "failed", at a confidence earned by passing.
        # Both readings are kept and both are cited, which is the invariant.
        prior_stance = str(existing["stance"] or "")
        flipped = (prior_stance != stance
                   and {prior_stance, stance} <= {"supports", "contradicts"})
        if flipped:
            excerpt = (f"[reading reversed] now {stance}: {excerpt}"
                       f" || previously {prior_stance}: "
                       f"{str(existing['excerpt'] or '')[:200]}")[:600]
        qi("UPDATE evidence SET excerpt=?, excerpt_chars=?, stance=?, "
           "title=? WHERE id=?",
           (excerpt, excerpt_chars, stance, str(title or "")[:200],
            existing["id"]))
        eid = existing["id"]
        fresh = False
    else:
        eid = qi("INSERT INTO evidence(hypothesis_id,url,title,excerpt,"
                 "excerpt_chars,stance,event_key,fetched_turn,created) "
                 "VALUES(?,?,?,?,?,?,?,?,?)",
                 (hid, url, str(title or "")[:200], excerpt, excerpt_chars,
                  stance, event_key, turn_idx, time.time()))
        fresh = True
    # The memory: provenance `read`, salience from how decisive the stance
    # is. Retrievable forever by the ordinary machinery; this is what lets
    # an old source resurface and dispute a new claim.
    memory.add_memory(
        "semantic", "read",
        0.6 if stance == "context" else 0.7,
        f"Regarding \"{hyp['question']}\": {excerpt or title or url}",
        source_url=url, turn_idx=turn_idx, event_key=event_key,
        confidence=0.8)
    if fresh:
        _apply_stance(hid, stance)
    elif flipped:
        # One source that has now said both things. Repetition is not
        # corroboration, so no confidence step is taken — but the
        # contradiction is real and has to be visible, and the band pin is
        # what makes the answer downstream say "unsettled".
        record_dispute_note(
            hid, f"the same source ({url}) supported this on an earlier read "
                 f"and contradicts it now")
    if claim_hint:
        qi("UPDATE hypotheses SET statement=? WHERE id=? AND statement=''",
           (str(claim_hint)[:400], hid))
    qi("UPDATE hypotheses SET updated_turn=? WHERE id=?", (turn_idx, hid))
    row = q("SELECT * FROM evidence WHERE id=?", (eid,), one=True)
    return {**dict(row), "ref": _evidence_ref(eid)}


def record_dispute_note(hid, note):
    """A finding ABOUT a hypothesis that is not evidence for or against it.

    Non-determinism is the case this exists for: the same experiment reaching
    two different outcomes says nothing about whether the hypothesis is true,
    and everything about whether the thing under test is a function of its
    inputs. Moving confidence on it would be wrong in either direction, so the
    note is recorded, the hypothesis is flagged disputed, and the confidence is
    pinned into the middle band -- the same treatment two contradicting sources
    get, for the same reason: an honest number for "this is unsettled" is one
    neither side would call a win.
    """
    hyp = get_hypothesis(hid)
    if hyp is None:
        return None
    conf = float(hyp["confidence"] or 0.3)
    low, high = _DISPUTE_BAND
    pinned = min(high, max(low, conf))
    # Merged into the same `dispute` blob _detect_dispute writes, under its own
    # key: a source disagreement and a non-reproducible experiment are both
    # "this is unsettled", and splitting them across two fields would let a
    # reader see one and miss the other.
    existing = hyp.get("dispute") if isinstance(hyp.get("dispute"), dict) else {}
    notes = list(existing.get("notes") or [])
    if note not in notes:
        notes.append(note)
    blob = {**existing, "notes": notes}
    # `status='disputed'` was in the docstring and not in the UPDATE, so the
    # pin did not hold and nothing downstream ever learned about it: the
    # conclude bar's disputed branch never fired for a note-only dispute,
    # `_hedged_conclusion`'s "Sources disagree" line never appeared, and two
    # later supporting sources walked the confidence straight back out of the
    # band to 0.81 and concluded "answered" — with a non-determinism note
    # sitting in the row, invisible to every consumer.
    qi("UPDATE hypotheses SET dispute=?, confidence=?, status='disputed' "
       "WHERE id=?", (json.dumps(blob), round(pinned, 4), hid))
    memory.add_memory(
        "semantic", "inferred", 0.75,
        f"Unsettled, regarding \"{hyp['question']}\": {note}",
        turn_idx=int(hyp["updated_turn"] or 0), confidence=0.6,
        event_key=f"dispute-note:{hid}:{hashlib.sha1(note.encode()).hexdigest()[:8]}")
    return {"hypothesis_id": hid, "note": note, "confidence": pinned}


def _apply_stance(hid, stance):
    hyp = get_hypothesis(hid)
    conf = float(hyp["confidence"] or 0.3)
    if stance == "supports":
        step = min(_MAX_STEP, _SUPPORT_PLASTICITY * (1.0 - conf))
        conf = _clamp(conf + step)
    elif stance == "contradicts":
        step = min(_MAX_STEP, _CONTRADICT_SUPPRESSION * conf)
        conf = _clamp(conf - step)
    # `context` moves nothing: background reading is not evidence for or
    # against, and counting it either way would let volume masquerade as
    # support.
    # A hypothesis already flagged disputed stays pinned. `_detect_dispute`
    # only re-pins when BOTH stances exist as rows, so a dispute raised as a
    # note (non-determinism, a source that reversed itself) could be walked
    # out of the band by any later supporting source — the pin has to survive
    # the thing it exists to survive.
    if str(hyp["status"] or "") == "disputed":
        conf = _clamp(conf, *_DISPUTE_BAND)
    qi("UPDATE hypotheses SET confidence=? WHERE id=?", (round(conf, 4), hid))
    _detect_dispute(hid)


def _detect_dispute(hid):
    """Two live contradictory sources are a DISPUTE, not an average.

    The failure this prevents: source A says X, source B says not-X, and a
    confidence that drifts to 0.5 reads exactly like "moderately likely" —
    the number silently claims a state of knowledge nobody has. A dispute
    pins the confidence into the middle band AND records both sides, so the
    answer downstream is forced to say "sources disagree" with citations for
    each, which is the only honest sentence available."""
    rows = evidence_for(hid)
    supporting = [r for r in rows if r["stance"] == "supports"]
    contradicting = [r for r in rows if r["stance"] == "contradicts"]
    if supporting and contradicting:
        dispute = {
            "supporting": [{"ref": r["ref"], "url": r["url"]}
                           for r in supporting],
            "contradicting": [{"ref": r["ref"], "url": r["url"]}
                              for r in contradicting],
        }
        hyp = get_hypothesis(hid)
        conf = _clamp(hyp["confidence"], *_DISPUTE_BAND)
        qi("UPDATE hypotheses SET status='disputed', dispute=?, confidence=? "
           "WHERE id=?",
           (json.dumps(dispute, ensure_ascii=False), round(conf, 4), hid))


def ground_citations(citations, delivered_refs):
    """Keep only citations naming refs that were actually delivered to the
    model this loop. Returns (grounded, warnings). Never invents a citation
    the model omitted — audit metadata describes the model's reasoning, it
    does not repair it after the fact. This is the engine's
    _ground_observation_citations, which is the backbone the whole research
    feature hangs from: a claim that cannot name its evidence is dropped or
    hedged, not trusted."""
    grounded, warnings = [], []
    for ref in citations or []:
        name = str(ref or "").strip()
        if name and name in delivered_refs:
            if name not in grounded:
                grounded.append(name)
        else:
            warnings.append(f"dropped ungrounded citation {name!r}")
    return grounded, warnings


def _hedged_conclusion(hyp, rows):
    """The budget ran out. Assemble the honest answer DETERMINISTICALLY from
    the evidence table — what supports, what contradicts, what is merely
    context — rather than asking the model to sound more finished than the
    evidence is. An engine-written hedge is drier than a model's, and that is
    the point: at this moment dryness is accuracy."""
    supporting = [r for r in rows if r["stance"] == "supports"]
    contradicting = [r for r in rows if r["stance"] == "contradicts"]
    lines = []
    statement = str(hyp.get("statement") or "").strip()
    if hyp.get("status") == "disputed":
        lines.append("Sources disagree on this, so I can't give you a "
                     "settled answer.")
    elif statement and supporting:
        lines.append(f"Best supported reading so far: {statement}")
    else:
        lines.append("I couldn't establish a well-supported answer within "
                     "my research budget.")
    for r in supporting[:3]:
        lines.append(f"- supports [{r['ref']}]: {r['excerpt'][:200]} "
                     f"({r['url']})")
    for r in contradicting[:3]:
        lines.append(f"- contradicts [{r['ref']}]: {r['excerpt'][:200]} "
                     f"({r['url']})")
    if not rows:
        lines.append("No usable sources were found; this needs either "
                     "different search terms or a source you can point "
                     "me at.")
    return {
        "answer": "\n".join(lines),
        "citations": [r["ref"] for r in (*supporting[:3],
                                         *contradicting[:3])],
        "confidence": float(hyp.get("confidence") or 0.3),
        "hedged": True,
    }


def research_loop(hid, ask_model, turn_idx, *, max_rounds=RESEARCH_MAX_ROUNDS,
                  search_results=5):
    """Automated research rounds until a grounded conclusion or budget end.

    Each round the model sees the hypothesis, its evidence so far, and any
    memories it pondered up, and returns ONE action:

      {"action": "search",  "query": "..."}
      {"action": "fetch",   "url": "...", "stance": ..., "excerpt": ...}
      {"action": "ponder",  "query": "...", "why": "..."}   — own memory
      {"action": "conclude","answer": "...", "citations": [refs],
                            "statement": "..."}

    The engine executes the action; the model never touches the database or
    the network itself. Ponder is the strategic lane: before burning a web
    round on something the assistant may already know, it can ask its own
    bank — prior evidence rows from old research surface here with their
    URLs, provenance-labelled `what_i_read`.

    Termination is deterministic: a conclusion is accepted only when its
    citations ground against delivered refs AND hypothesis confidence clears
    CONCLUDE_CONFIDENCE (or the hypothesis is disputed — "sources disagree"
    is a complete answer). Otherwise the loop continues; at budget end the
    engine writes the hedged answer itself. Returns a dict with the answer,
    grounded citations, rounds used, trace, and warnings."""
    trace, warnings = [], []
    delivered = set()          # refs the model has actually been shown
    firsthand_refs = set()     # of those, the ones that are not conjecture
    pondered_payload = []
    last_search = []
    for round_no in range(1, max_rounds + 1):
        hyp = get_hypothesis(hid)
        rows = evidence_for(hid)
        for r in rows:
            delivered.add(r["ref"])
        payload = {
            "question": hyp["question"],
            "i_suspect": hyp["statement"] or "(no working statement yet)",
            "confidence": hyp["confidence"],
            "status": hyp["status"],
            **({"dispute": hyp["dispute"]} if hyp["dispute"] else {}),
            "evidence": [{"ref": r["ref"], "url": r["url"],
                          "title": r["title"], "stance": r["stance"],
                          "excerpt": r["excerpt"]} for r in rows],
            "search_results": last_search,
            "remembered": pondered_payload,
            "rounds_left": max_rounds - round_no + 1,
            "conclude_bar": CONCLUDE_CONFIDENCE,
        }
        act = ask_model(payload) or {}
        action = str(act.get("action") or "").strip()
        trace.append({"round": round_no, "action": action,
                      "detail": {k: v for k, v in act.items()
                                 if k != "action"}})
        if action == "search":
            last_search = tools_web.search(str(act.get("query") or
                                               hyp["question"]),
                                          max_results=search_results)
            if not last_search:
                warnings.append("search returned nothing")
        elif action == "fetch":
            url = str(act.get("url") or "").strip()
            page = tools_web.fetch(url)
            excerpt = str(act.get("excerpt") or "").strip() \
                or page.get("text", "")[:400]
            # A page that yields nothing readable is a FAILED fetch, not a
            # silent source. A JS-only SPA, a PDF, or a page `_strip_html`
            # eats returns HTTP 200 with no text, and the row it minted moved
            # confidence a full step on nothing at all: three of them reached
            # 0.81 and concluded "answered", citing ev:1, ev:2 and ev:3, all
            # substanceless. The user sees "Yes." and three sources.
            if page.get("error") or not excerpt:
                warnings.append(
                    f"fetch failed: {page['error']}" if page.get("error")
                    else f"fetch returned no readable text: {url}")
                # Never a bare `==` on a URL — AGENTS.md, and the reason is
                # right here: `url` is the model's spelling and `s['url']` is
                # the search backend's, so one trailing slash left the dead
                # result in the list to be re-offered and re-fetched every
                # remaining round until the budget drained.
                last_search = [s for s in last_search
                               if not same_source(s.get("url"), url)]
                continue
            ev = record_evidence(
                hid, url=url, title=page.get("title") or "",
                excerpt=excerpt,
                stance=str(act.get("stance") or "context"),
                turn_idx=turn_idx,
                claim_hint=str(act.get("statement") or ""))
            if ev:
                delivered.add(ev["ref"])
        elif action == "ponder":
            # Strategic recall over the assistant's own bank — the engine's
            # deliberate-recall lane, here spendable mid-research. Costs an
            # embedding call, not a web round.
            found = memory.search_memories(
                str(act.get("query") or hyp["question"]), k=4,
                current_turn_idx=turn_idx)
            pondered_payload = []
            for mem_row in found:
                item = memory.project_memory(mem_row, turn_idx)
                delivered.add(item["memory_ref"])
                # Which recalled rows are TESTIMONY rather than the
                # assistant's own guesswork. The memory-grounded exit below
                # needs this and cannot get it from the ref string.
                if mem_row.get("provenance") in ("read", "told", "witnessed"):
                    firsthand_refs.add(item["memory_ref"])
                pondered_payload.append(item)
            if not pondered_payload:
                warnings.append("ponder recalled nothing")
        elif action == "conclude":
            grounded, warns = ground_citations(act.get("citations"),
                                               delivered)
            warnings.extend(warns)
            hyp = get_hypothesis(hid)
            statement = str(act.get("statement") or "").strip()
            if statement:
                qi("UPDATE hypotheses SET statement=? WHERE id=?",
                   (statement[:400], hid))
            if not grounded:
                # A conclusion with no surviving citation is not an answer,
                # it is prose. Reject it and keep working — this is the
                # research-loop form of "dropped or hedged exactly the way
                # the engine drops an ungrounded citation".
                warnings.append("conclusion rejected: no grounded citations")
                continue
            # The strategic-ponder exit: no web evidence was gathered and
            # every citation is a memory ref — the assistant already knew.
            # Its stored `read`/`told` rows carry their own provenance and
            # urls, so the answer is grounded even though the hypothesis's
            # web-evidence confidence never moved. Without this exit, a
            # question fully answered by memory would burn the whole budget
            # and come back hedged.
            evidence_rows = evidence_for(hid)
            # THE MEMORY-GROUNDED EXIT REQUIRES TESTIMONY, NOT CONJECTURE.
            #
            # It used to require only that no evidence row existed and no
            # citation began with "ev:" — it never looked at provenance. But
            # pipeline mints every user_model update as an `inference` row
            # with its own event_key, so the assistant's own conjectures are
            # citable and pondered like anything else. Measured: one
            # `inferred` row reading "I suspect the tool shipped in March
            # 2022 but never checked" was pondered up, cited, and accepted —
            # hedged False, status answered, confidence 0.30 — and the
            # pipeline surfaced it to the user under a "Sources:" line. An
            # unverified i_suspect reading back as a settled answer is the
            # one thing the i_suspect key exists to prevent, and this is that
            # collapse happening inside one mind. DESIGN.md justifies the
            # exit with "its stored read/told rows carry their own
            # provenance"; now the code requires it.
            from_memory = (not evidence_rows
                           and bool(grounded)
                           and all(g in firsthand_refs for g in grounded))
            if (hyp["confidence"] >= CONCLUDE_CONFIDENCE
                    or hyp["status"] == "disputed"
                    or from_memory):
                status = ("disputed" if hyp["status"] == "disputed"
                          else "answered")
                qi("UPDATE hypotheses SET status=?, updated_turn=? "
                   "WHERE id=?", (status, turn_idx, hid))
                return {"answer": str(act.get("answer") or "").strip(),
                        "citations": grounded,
                        "confidence": hyp["confidence"],
                        "status": status, "rounds": round_no,
                        "from_memory": from_memory,
                        "trace": trace, "warnings": warnings,
                        "hedged": False}
            warnings.append(
                "conclusion below confidence bar "
                f"({hyp['confidence']:.2f} < {CONCLUDE_CONFIDENCE}); "
                "continuing research")
        else:
            warnings.append(f"unknown research action {action!r}")
    # Budget exhausted: the engine writes the hedge.
    hyp = get_hypothesis(hid)
    rows = evidence_for(hid)
    out = _hedged_conclusion(hyp, rows)
    qi("UPDATE hypotheses SET updated_turn=? WHERE id=?", (turn_idx, hid))
    out.update({"status": hyp["status"], "rounds": max_rounds,
                "trace": trace, "warnings": warnings})
    return out


def dispute_memory_against_evidence(memory_ref, reading, turn_idx):
    """New evidence has changed what an old memory means. Central here where
    it was rarely occasioned in the engine: a benchmark superseded, a doc
    page now wrong for the current version, a user correcting a stored fact.
    The memory keeps saying "I read this"; the dispute beside it says "and I
    no longer read it that way"."""
    return memory.record_dispute(reading, turn_idx, memory_ref=memory_ref)
