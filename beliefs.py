# beliefs.py — hypothesis persistence and revision. A port of Sonder Engine's
# theory_of_mind.py with the body (absorption) removed and the kind vocabulary
# retargeted from "minds in a fiction" to "an assistant's model of its user
# and its topics".
#
# In the engine this machinery modelled what one character believed about
# another. Here the same machinery IS the user model: "the user" is simply
# the most important subject the assistant holds hypotheses about, and topics
# under research are the others. What survives unchanged, because it is the
# part that was right:
#
# - Belief perseverance / primacy (Ross, Lepper & Hubbard 1975): stable-
#   character claims (trait, identity) are sticky and resist single-instance
#   revision — low plasticity, long half-life, low ceiling.
# - Source monitoring: something directly stated is strong, fast-updating
#   evidence (stated_fact: high plasticity, high cap).
# - Ebbinghaus forgetting: unreinforced beliefs fade rather than sitting at
#   peak confidence forever.
# - "Explaining away": a strong competing claim WEAKENS a prior belief
#   without erasing it. Real revision needs repeated disconfirmation, not
#   one data point — and never averaging, which is how two contradictory
#   sources become one wrong number.
#
# What was cut and why: absorption (cognitive bandwidth claimed by the body)
# and the formed-under/reappraisal cycle built on it. An assistant has no
# body monopolising its attention; the machinery's job there — making
# extremity-formed beliefs come back up for review — has no input signal
# here. If a fatigue-analog ever appears (e.g. degraded evidence quality
# under budget pressure) the engine's shape is the one to re-import.
#
# Every function is pure (no DB/network) so commit code and payload builders
# can both import it without a cycle, and tests run it in isolation.

import re

_DEFAULT_KIND = "goal"

# confidence cap: the ceiling a single claim of this kind may ever reach —
# an epistemic-effort gradient. plasticity: how far one consistent data point
# moves the belief. half_life: turns until an unreinforced belief decays by
# half; deliberately not the inverse of plasticity, because "how fast it
# updates" and "how fast it fades from disuse" are dissociable.
#
# `preference` is the assistant-specific addition: what the user likes and
# how they want things done. Capped below stated_fact (a preference is
# inferred from conduct more often than declared), more plastic than trait
# (people change tools and tastes), and long-lived (a preference unmentioned
# for months is usually still held — the cost of forgetting one is re-asking
# a question the user already answered, which is exactly the failure a
# long-memory assistant exists to avoid).
CONFIDENCE_CAPS = {
    "observation": 1.0, "stated_fact": 0.9, "preference": 0.8,
    "goal": 0.65, "trait": 0.45, "identity": 0.35,
}

PLASTICITY = {
    "observation": 0.75, "stated_fact": 0.85, "preference": 0.45,
    "goal": 0.5, "trait": 0.25, "identity": 0.2,
}

HALF_LIFE = {
    "observation": 5, "stated_fact": 18, "preference": 300,
    "goal": 45, "trait": 400, "identity": 400,
}

_SIMILARITY_THRESHOLD = 0.4
_MAX_SUPPRESSION = 0.6

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "seems", "seem", "appears", "appear", "of", "to", "about", "that",
    "this", "these", "those", "they", "them", "their", "he", "she", "him",
    "her", "his", "hers", "it", "its", "and", "or", "but", "with", "for",
    "on", "in", "at", "as", "has", "have", "had", "will", "would", "may",
    "might", "can", "could", "from", "by", "so", "not", "than", "then",
    "into", "up", "out", "if", "no", "there", "around", "near", "toward",
    "when", "while", "very", "just", "still",
}


def _clamp01(value, fallback=0.5):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, v))


def _kind_or_default(kind):
    kind = str(kind or _DEFAULT_KIND)
    return kind if kind in CONFIDENCE_CAPS else _DEFAULT_KIND


# ---- The declared kind is not taken on trust ----
#
# Every ceiling here keys off a kind the MODEL declares, and the path of
# least resistance is mislabelling: "the user is the sort who cuts corners"
# declared `observation` (cap 1.0) instead of `trait` (cap 0.45) buys
# confidence the epistemics say it cannot have. So the claim's own language
# votes too, and the STRICTER of declared and inferred wins. This is text
# matching, and text matching is not an information boundary — it is
# confidence calibration, arranged so a misfire can only make the assistant
# LESS sure, never more. Under-confidence degrades; over-confidence launders
# a guess into a fact, and those are not equally bad.
_KIND_CUES = (
    ("stated_fact", (
        r"\b(?:said|told|stated|claimed|admitted|confirmed|announced)\b",
    )),
    ("identity", (
        r"\bis really\b", r"\bis actually\b", r"\bmust be the\b",
        r"\bis the one who\b", r"\bgoes by\b",
    )),
    ("trait", (
        r"\balways\b", r"\bnever\b", r"\b(?:kind|sort|type) of person\b",
        r"\bby nature\b", r"\btends to\b",
    )),
    ("preference", (
        r"\bprefers?\b", r"\blikes?\b", r"\bdislikes?\b", r"\bhates?\b",
        r"\bfavou?rite\b", r"\brather (?:than|have|use)\b",
        r"\bwants? (?:it|things|answers|responses) \b",
    )),
    ("goal", (
        r"\bwants? to\b", r"\bintends?\b", r"\bplans? to\b",
        r"\btrying to\b", r"\bmeans to\b", r"\bhopes? to\b",
        r"\baims? to\b", r"\bis going to\b", r"\bworking (?:on|toward)\b",
    )),
)


def _inferred_kind(claim):
    low = str(claim or "").casefold()
    if not low.strip():
        return None
    for kind, cues in _KIND_CUES:
        if any(re.search(cue, low) for cue in cues):
            return kind
    return None


def effective_kind(declared, claim):
    """Whichever of the declared and inferred kind carries the LOWER ceiling.
    A model cannot buy confidence by mislabelling, and a wrong inference can
    only cost confidence, never grant it."""
    declared_kind = _kind_or_default(declared)
    inferred = _inferred_kind(claim)
    if inferred is None:
        return declared_kind
    if CONFIDENCE_CAPS[inferred] < CONFIDENCE_CAPS[declared_kind]:
        return inferred
    return declared_kind


def decayed_confidence(confidence, kind, turns_elapsed):
    """Exponential (Ebbinghaus-style) decay of an unreinforced belief."""
    conf = _clamp01(confidence, fallback=0.0)
    try:
        elapsed = max(0.0, float(turns_elapsed or 0))
    except (TypeError, ValueError):
        elapsed = 0.0
    if elapsed == 0 or conf <= 0.0:
        return conf
    half_life = HALF_LIFE.get(_kind_or_default(kind), HALF_LIFE[_DEFAULT_KIND])
    return conf * (0.5 ** (elapsed / half_life))


def _tokens(text):
    words = re.findall(r"[a-z0-9']+", str(text or "").casefold())
    filtered = [w for w in words if w not in _STOPWORDS]
    return set(filtered) if filtered else set(words)


# ---- Polarity ----
#
# The overlap matcher is blind to negation, and "not" sits in _STOPWORDS, so
# "prefers concise answers" and "does not prefer concise answers" scored 0.667
# — comfortably over the 0.4 threshold. Measured consequence in the real
# pipeline: a user CORRECTION reinforced the belief it contradicted. The
# convex blend pulled confidence toward the new "evidence", the claim text was
# overwritten with its own negation, and "explaining away" — the mechanism
# that exists precisely for a competing claim — never fired once.
#
# That is contradiction being averaged, the one thing AGENTS.md forbids
# outright, arriving through the door the stopword list left open. Removing
# "not" from the stopwords does not fix it (the shared content words still
# carry the overlap past threshold): polarity has to be read, not counted.
#
# Text matching is not an information boundary. Like effective_kind, this is
# arranged so a misfire is survivable: a false polarity split makes two
# hypotheses compete that should have merged, and competition is partial and
# self-correcting. A missed split laundered a contradiction into agreement.
_NEGATION_CUES = (
    r"\bnot\b", r"\bno longer\b", r"\bnever\b", r"\bn't\b", r"\bnone\b",
    r"\bstopped\b", r"\bceased\b", r"\bdenies\b", r"\bdenied\b",
    r"\bdoesn'?t\b", r"\bdidn'?t\b", r"\bisn'?t\b", r"\bwasn'?t\b",
    r"\bwon'?t\b", r"\bcan'?t\b", r"\bcannot\b", r"\bwithout\b",
    r"\bdis(?:likes?|agrees?|prefers?)\b", r"\bun(?:like|willing)\b",
    r"\brather than\b", r"\binstead of\b", r"\bopposed to\b",
)


def claim_polarity(claim):
    """+1 for an affirmative claim, -1 for a negated one. Double negation is
    not modelled — two cues read as affirmative, which is the ordinary English
    reading of "never not" and rare enough not to matter."""
    low = str(claim or "").casefold()
    hits = sum(1 for cue in _NEGATION_CUES if re.search(cue, low))
    return -1 if hits % 2 else 1


def claim_similarity(a, b, ignore=None):
    """How likely two claims describe the same underlying belief.

    `ignore` is the SUBJECT both claims are about, removed from both sides
    before comparing — naming the subject inside a claim is ordinary phrasing,
    but it inflated every same-subject pair toward a match in the engine
    ("Chamber 0505 has a bench" vs "Chamber 0505 is empty" merged on the
    strength of "chamber 0505" alone). What a claim is ABOUT is carried by
    the subject key; only what it SAYS decides sameness.

    Overlap coefficient plus a subset short-circuit, so a terse restatement
    of a longer claim reads as the same belief rather than a competitor. No
    embeddings: this runs per update against a small in-memory list. It will
    occasionally misread very short same-vocabulary claims — accepted,
    because reinforcement blends rather than overwrites and suppression is
    partial, so a misread self-corrects over a few turns."""
    # Opposite polarity is a COMPETITOR, never a restatement — checked before
    # any overlap arithmetic, because the words two contradictory claims share
    # are exactly the words that make them look alike.
    if claim_polarity(a) != claim_polarity(b):
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    if ignore:
        drop = _tokens(ignore)
        ta, tb = (ta - drop) or ta, (tb - drop) or tb
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    if ta <= tb or tb <= ta:
        return 1.0
    return len(ta & tb) / min(len(ta), len(tb))


def _elapsed(hypothesis, turn_idx):
    last = hypothesis.get("last_updated_turn", turn_idx)
    try:
        last = int(last)
    except (TypeError, ValueError):
        last = turn_idx
    return max(0, int(turn_idx) - last)


def _live_confidence(hypothesis, turn_idx):
    kind = _kind_or_default(hypothesis.get("kind"))
    return decayed_confidence(hypothesis.get("confidence", 0.0), kind,
                              _elapsed(hypothesis, turn_idx))


def apply_belief_updates(state, updates, turn_idx, floor=0.05,
                         max_per_subject=30):
    """Merge this turn's belief updates into persistent state.

    A claim matching an existing same-kind hypothesis for the subject
    reinforces it via a convex blend toward the new evidence, scaled by the
    kind's plasticity — never a jump to the higher value. A claim that does
    not match is a COMPETING hypothesis: kept alongside, and it partially
    suppresses ("explains away") its same-group siblings rather than erasing
    them. Every confidence is decay-adjusted before use; hypotheses (or whole
    subjects) below `floor` are pruned, which is what keeps a long history
    from accumulating unbounded tracked subjects."""
    models = state.setdefault("mind_models", {})
    for update in updates or []:
        if not isinstance(update, dict):
            continue
        about = str(update.get("about") or update.get("about_entity")
                    or "unknown").strip() or "unknown"
        claim = str(update.get("claim") or "").strip()
        if not claim:
            continue
        kind = effective_kind(update.get("kind"), claim)
        evidence_confidence = _clamp01(update.get("confidence", 0.5))
        cap = CONFIDENCE_CAPS.get(kind, 1.0)
        plasticity = PLASTICITY.get(kind, PLASTICITY[_DEFAULT_KIND])

        model = models.setdefault(about, {"hypotheses": []})
        hyps = model.setdefault("hypotheses", [])
        group = [i for i, h in enumerate(hyps)
                 if isinstance(h, dict)
                 and _kind_or_default(h.get("kind")) == kind]
        best_idx, best_sim = None, 0.0
        for i in group:
            sim = claim_similarity(claim, str(hyps[i].get("claim") or ""),
                                   ignore=about)
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_idx is not None and best_sim >= _SIMILARITY_THRESHOLD:
            existing = hyps[best_idx]
            decayed_old = _live_confidence(existing, turn_idx)
            new_conf = (decayed_old
                        + (evidence_confidence - decayed_old) * plasticity)
            merged = dict(update)
            merged["about"] = about
            merged["kind"] = kind
            merged["claim"] = claim
            # Provenance belongs to the BELIEF, not to the update that last
            # touched it — `merged` is rebuilt from the incoming update, so
            # anything not explicitly carried is silently lost. The engine
            # shipped that bug: a belief merely restated became one that had
            # always been freshly held.
            if "first_seen_turn" in existing and "first_seen_turn" not in update:
                merged["first_seen_turn"] = existing["first_seen_turn"]
            merged["confidence"] = max(
                0.0, min(max(cap, decayed_old), new_conf))
            merged["last_updated_turn"] = turn_idx
            hyps[best_idx] = merged
        else:
            new_hyp = dict(update)
            new_hyp["about"] = about
            new_hyp["kind"] = kind
            new_hyp["claim"] = claim
            new_hyp["confidence"] = max(0.0, min(cap, evidence_confidence))
            new_hyp["last_updated_turn"] = turn_idx
            new_hyp["first_seen_turn"] = turn_idx
            hyps.append(new_hyp)
            # Explain away, never erase: the competitor's live confidence is
            # scaled down by how plausible the new claim is, bounded so one
            # data point cannot delete a standing belief.
            suppression = min(_MAX_SUPPRESSION,
                              plasticity * evidence_confidence)
            for i in group:
                sib = hyps[i]
                sib["confidence"] = max(
                    0.0, _live_confidence(sib, turn_idx) * (1 - suppression))
                sib["last_updated_turn"] = turn_idx
        model["last_updated_turn"] = turn_idx

    # Sweep EVERY subject, not just those mentioned this turn, so beliefs
    # about topics that dropped out of the conversation actually fade from
    # storage instead of accumulating forever.
    for about in list(models.keys()):
        model = models.get(about) or {}
        scored = []
        for h in model.get("hypotheses") or []:
            if not isinstance(h, dict):
                continue
            live = _live_confidence(h, turn_idx)
            if live >= floor:
                scored.append((live, h))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        survivors = [h for _, h in scored[:max_per_subject]]
        if survivors:
            model["hypotheses"] = survivors
            models[about] = model
        else:
            models.pop(about, None)
    return state


def belief_credence(state, about, claim, turn_idx):
    """Current live credence in `claim` about `about`, or None when no
    surviving hypothesis expresses it. Uses the SAME matcher and threshold as
    the merge itself, so a memory and a hypothesis are judged "the same
    belief" by one rule rather than two that drift apart."""
    models = (state or {}).get("mind_models")
    if not isinstance(models, dict):
        return None
    model = models.get(str(about or "").strip())
    if not isinstance(model, dict):
        return None
    best, best_sim = None, 0.0
    for hyp in model.get("hypotheses") or []:
        if not isinstance(hyp, dict):
            continue
        # `ignore=about` is not optional here, whatever the argument's default
        # says. Without it this function was a SECOND, looser rule: the merge
        # strips the subject's own tokens before comparing, credence did not,
        # and naming the subject inside a claim (ordinary phrasing) inflated
        # every same-subject pair toward a match. Measured straddle: two
        # claims the merge files as separate competitors (sim 0.33) scored
        # 0.60 here, so reconcile_inference_confidence ranked an inference row
        # on a DIFFERENT belief's credence — an abandoned theory inheriting
        # the confidence of the theory that replaced it, which is precisely
        # the case reconciliation exists to fix. Two names for one rule, and
        # the drift failed toward over-confidence.
        sim = claim_similarity(claim, str(hyp.get("claim") or ""),
                               ignore=about)
        if sim > best_sim:
            best_sim, best = sim, hyp
    if best is None or best_sim < _SIMILARITY_THRESHOLD:
        return None
    return _clamp01(_live_confidence(best, turn_idx), fallback=0.0)


def beliefs_for_payload(mind_models, turn_idx, max_competitors=2):
    """Leading belief + live competitors per (subject, kind), decay applied
    for display without mutating storage — the assistant can see it is still
    weighing two theories, which is metacognition rather than a flat dump."""
    out = {}
    for about, model in (mind_models or {}).items():
        by_kind = {}
        for h in (model or {}).get("hypotheses") or []:
            if not isinstance(h, dict):
                continue
            kind = _kind_or_default(h.get("kind"))
            by_kind.setdefault(kind, []).append({
                "claim": h.get("claim", ""),
                "confidence": round(_live_confidence(h, turn_idx), 3),
            })
        kinds_out = {}
        for kind, entries in by_kind.items():
            entries.sort(key=lambda e: e["confidence"], reverse=True)
            kinds_out[kind] = {"leading": entries[0],
                               "competitors": entries[1:1 + max_competitors]}
        if kinds_out:
            out[about] = kinds_out
    return out


# ---- The stable hypothesis sheet ----

_SHEET_CAPACITY = 5
# An incumbent keeps its slot unless a challenger beats it by this much.
# Without hysteresis the sheet churns every turn as confidences wobble and
# stops being "what I am actively wondering about" — which is the whole point
# of it being stable.
_SHEET_INCUMBENT_MARGIN = 0.08


def hypothesis_key(about, kind, claim):
    return f"{str(about).strip()}|{_kind_or_default(kind)}|{str(claim).strip()}"


def select_active_hypotheses(mind_models, previous_keys, turn_idx,
                             capacity=_SHEET_CAPACITY):
    """The few open questions this mind is actively holding, each explicitly
    marked AS a hypothesis: the payload key is `i_suspect`, so the field
    itself carries the epistemic status. An assistant reading its own
    conjecture back as settled fact is an information-layer collapse
    happening inside one mind — the exact failure the engine polices between
    minds. Returns (entries, keys)."""
    previous = set(previous_keys or ())
    scored = []
    for about, model in (mind_models or {}).items():
        for hyp in ((model or {}).get("hypotheses") or []):
            if not isinstance(hyp, dict):
                continue
            claim = str(hyp.get("claim") or "").strip()
            if not claim:
                continue
            kind = _kind_or_default(hyp.get("kind"))
            live = _live_confidence(hyp, turn_idx)
            key = hypothesis_key(about, kind, claim)
            rank = live + (_SHEET_INCUMBENT_MARGIN if key in previous else 0.0)
            scored.append((rank, live, key, about, kind, claim, hyp))
    scored.sort(key=lambda row: (-row[0], row[2]))
    entries, keys = [], []
    for _rank, live, key, about, kind, claim, hyp in scored[:max(0, capacity)]:
        entries.append({
            "about": about, "kind": kind, "i_suspect": claim,
            "confidence": round(live, 3),
            "held_since_turn": hyp.get("first_seen_turn",
                                       hyp.get("last_updated_turn")),
        })
        keys.append(key)
    return entries, keys
