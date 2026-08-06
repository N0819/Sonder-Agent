# Ponder Engine

**Version 1.0** — 22 modules, 437 tests, offline suite.
Released as **Sonder Assistant**; renamed, same project.

A single-user chatbot with an assistant persona, real long-term memory, and
evidence-driven research over the web. It is the cognition core of
[Sonder Engine](../Sonder_Engine) — a multi-agent interactive-fiction system —
extracted into a standalone project, with the fiction and the body removed
and the epistemics kept.

The name is the engine's, one letter over, and it is also a verb this system
already had: `need_more.ponder` is how a turn asks its own memory instead of
the world. Pondering is the half that was kept.

```bash
make test   # full suite, offline, no network, no model
make run    # http://localhost:8010 (uvicorn, --reload)
```

Configure providers in the **Settings** tab, or by environment:

```bash
# The key itself always lives in the environment. Settings stores the NAME of
# the variable, never the value — see config.py for why.
export OPENROUTER_API_KEY=sk-or-...
```

**Chat and embeddings are separate providers**, because they genuinely are two
services: the Claude Code CLI composes replies and has no embeddings endpoint
at all, so an assistant using it still needs somewhere to vectorise or three
of the four ranking lanes go dark. Settings has a preset for each role.

Embeddings presets include **OpenRouter → Perplexity Embed V1 4B**
(`perplexity/pplx-embed-v1-4b`, 32K context, $0.03/M tokens) and its lighter
sibling `pplx-embed-v1-0.6b` ($0.004/M) — OpenRouter is the only way to reach
Perplexity's embedding models, since there is no direct Perplexity embeddings
API. OpenAI's `text-embedding-3-small`/`-large` are there too, direct or
through OpenRouter.

**Changing the embedding model strands the bank**, and the settings page says
so before you commit: a vector is comparable only with one from the same
model, so every existing row would score 0.0 against a new query embedding and
retrieval would quietly run on keyword match alone. **Rebuild embeddings** in
Settings re-embeds the stranded rows — incremental, re-runnable, and it
refuses rather than overwriting good vectors with fallback ones.

With no chat model configured the app still runs, records every exchange
into memory, and says plainly that it cannot compose a reply — the
deterministic floor never impersonates the model, and it distinguishes "no
provider configured" from "the provider failed this turn" rather than
blaming the wrong subsystem. With no embeddings provider, retrieval degrades
to a hashing-trick lexical fallback (`cheap_embed`) that the engine measured
at 0% recall on vocabulary-disjoint paraphrases: it works on shared
vocabulary and not at all otherwise. Every stored vector is stamped with its
model, and — this is the half that was missing — every turn now **counts**
the rows that stamp has stranded and warns, because three of the four
ranking lanes are vector lanes and recall goes on looking healthy when they
all score zero.

## What was inherited, and why

The memory system is the crown jewel, preserved mechanically rather than
sketched. Where a constant appears with a justification, the measurement
behind it is the engine's (`Sonder_Engine/docs/MEMORY.md`); nothing here
re-derived or "improved" them.

- **Memory rows with provenance** (`witnessed / told / read / inferred /
  remembered`), salience, importance, confidence, `event_key`, `turn_idx`,
  disputes. Provenance routes: three separate summary scopes (what happened
  / what I was told or read / what I concluded), because one melted summary
  let an inference come back indistinguishable from experience — belief
  laundered into knowledge inside one mind.
- **Retrieval**: RRF fusion of four rankings (semantic 1.0, cue-vector 1.15,
  BM25 1.1, exact-match 1.25) at `_RRF_SCALE = 12`, per-aspect rank lists at
  0.55 (concatenating short facets onto a long query was measured doing
  nothing: cosine 0.994 to the query alone), scalar bonuses, MMR diversity
  (0.82/0.18), chronological-neighbour padding, and a relevance gate rather
  than a fixed `k` — `RECALL_LIMIT` (64) is a payload ceiling, not a target.
  See AGENTS.md for the measured caveat: the gate is coarser than its
  documentation claims and the constant is deliberately not retuned against
  a `cheap_embed` corpus. Results are returned in chronological order because
  consumers read them as a narrative.
- **`_rank_normalized_importance`**: the salience term respaced inside the
  visible rows' own p10–p90. The engine replayed 270 recalls to choose this:
  deleting the term moved 35% of retrievals, normalising to [0,1] moved 60%
  (a 3.7x weight increase wearing the word "normalisation"), respacing moved
  15%. Ordering preserved exactly, influence budget unchanged.
- **Importance revision gated on consequence**: asymptotic steps toward a
  0.97 ceiling, `only_unrevised` closing the popularity loop structurally
  (a cited memory is lifted once, ever), disputes moving importance further
  than citations because being wrong about something is a bigger fact about
  it. `access_count` stays written-and-unread, deliberately.
- **Consolidation**: windows keyed `(scope, end_turn_idx)` — the engine's
  singleton design silently overwrote every chapter but the last — with the
  first-hand row written unconditionally because its `end_turn_idx` is the
  cursor. **Support sets** (schema v25): per-clause citations derived
  host-side by content-word overlap (floor of three shared words,
  calibrated), scoped to the clause's own epistemic class. An empty support
  set is a finding, not an error.
- **Belief revision** (`beliefs.py`, from `theory_of_mind.py`): per-kind
  confidence caps / plasticity / half-lives, convex blend on reinforcement,
  partial "explaining away" on contradiction, decay, pruning,
  `effective_kind` taking the stricter of declared and inferred kind (a
  model cannot buy confidence by mislabelling; a misfire can only make the
  assistant less sure). The stable hypothesis sheet with incumbent
  hysteresis, keyed `i_suspect` so a conjecture can never read back as fact.
- **Inference-confidence reconciliation**: mint `salience = 0.45 +
  0.3·confidence` (reconstructible because salience is never revised);
  abandoned beliefs rest at a fixed fraction of mint confidence — the
  engine measured the compounding alternative crushing 76–80% of an
  inference bank in 7–18 turns; held ≥ abandoned, always; idempotent.
- **The citation discipline**: everything the model cites is grounded
  against what was actually delivered to it; ungrounded citations are
  dropped with a warning, never repaired. A warning is the system working —
  nothing crossed.
- **Disputes** (`record_dispute`): a memory re-read against new evidence
  keeps its content untouched and carries the new reading beside it. In the
  engine this was reachable and rarely occasioned (deception is rare in
  warm stories); here it is central, because superseded docs, retracted
  claims and user corrections are everyday events for an assistant.
- **The commit discipline**: model output is provisional until deterministic
  code validates it; embedding happens before the write transaction (a
  network round trip must never hold SQLite's writer); primary turn
  mutations are atomic — the `turns` row included; consolidation runs after,
  because a reconstructible cache's failure must never roll back a valid
  turn. The connection is opened in true autocommit and every write
  transaction is `BEGIN IMMEDIATE`, because under sqlite3's legacy
  isolation mode none of the above was actually happening: see
  AGENTS.md § Invariants and `tests/test_durability.py`.
- **The read seam**: one function (`visible_memory_rows`) with required,
  defaultless invariant arguments. Forgetting the turn cutoff is a
  `TypeError`, not a leak. (The engine's earlier documentation claimed
  repeating the filters at each call site was the safety; that reasoning
  was backwards, and the seam is the correction.)

## What was cut, and why

Everything fictional and embodied: perception/sensory channels, spatial
reasoning, attire, scene/world state, the Director, the Narrator, background
presences, interaction/reaction loops, drives/wants/intentions, frames/eras,
checkpoints, and the information firewall — though **subagents have since
made that last one only half true**. The firewall was cut because one mind
has nothing to withhold from itself; a subagent is a second mind, so its
report enters as `told` testimony rather than experience, and its private
working-out is archived out of reach of recall rather than merged. The
firewall's *internal* form was always here and now carries real weight:
provenance labelling, `i_suspect`, epistemic-origin stamps, all preventing
the layer-collapse *inside* one mind that the firewall prevented between
minds. See AGENTS.md § Subagents.

**Affect** (valence/arousal/mood/stress/hedonics, `encoding_valence`,
mood-congruent recall, unbidden contrast recall, absorption) was cut as
semantics but kept as machinery: the bounded-update discipline — clamped
maximum steps, convex blends toward evidence, half-life decay, floors and
ceilings, hysteresis — reappears in the research confidence updates, the
belief store, and the hypothesis sheet. What was deliberately *not* kept:
mood-congruence (no mood axis to read), unbidden recall (its trigger was
"a character measurably stuck", a fiction-shaped signal), and the
absorption→reappraisal cycle (no body to supply the input; noted in
`beliefs.py` as the shape to re-import if a budget-pressure analog ever
appears).

## What was added

- **Persona** (`persona.py`): identity, expertise, working style, standing
  commitments, stable preferences. No emotions. The engine's authoring
  lesson is enforced: an empty field fails silently, so every write path
  warns on one.
- **The user model**: the mind-model machinery pointed at "the user" — what
  the assistant learns about you over time, every claim requiring grounded
  evidence, with a `preference` kind (long half-life, moderate plasticity)
  because re-asking an answered question is the failure a long-memory
  assistant exists to avoid.
- **Research** (`research.py`, `tools_web.py`): a question becomes an
  active hypothesis with a confidence; searching gathers evidence rows that
  cite real URLs; evidence moves confidence deterministically (bounded
  steps, same-URL idempotency so repetition is not corroboration);
  contradictory sources become a **dispute** — both sides kept and cited,
  confidence pinned to the middle band, never averaged. Every evidence row
  is also a `read`-provenance memory, so old research resurfaces through
  ordinary retrieval and can dispute newer claims.
- **The automated research loop**: internal rounds (search / fetch /
  strategic ponder over its own memory / conclude) until a conclusion
  grounds and clears the confidence bar — or the budget runs out, at which
  point the *engine* writes an explicitly hedged answer from the evidence
  table, because at that moment dryness is accuracy. A question fully
  answered from memory exits early with memory citations.
- **A minimal chat UI**: browser globals, no bundler, load order as the
  dependency graph — engine frontend conventions. Panels: Chat, Memory,
  Research, Beliefs, Files (drag-and-drop upload + archive extraction),
  Subagents (the permission surface), Settings (provider + persona).
- **Provider choice** (`config.py`): an OpenAI-compatible endpoint or the
  **Claude Code CLI**, selectable from Settings without a restart. Secrets
  are never stored: a settings row holds the NAME of an environment variable
  and the UI shows only whether it resolves, so a live credential never
  lands in `assistant.db` beside the memory bank.
- **A file workspace** (`workspace.py`): drag-and-drop uploads per session,
  readable by the sandbox, with archive extraction whose threat model is the
  ARCHIVE rather than the user — zip slip, absolute members, symlink members
  and decompression bombs are all refused from the central directory before
  a byte is written.
- **Codemaps** (`codemap.py`): an index — never a summary — of an uploaded
  project, plus any AGENTS.md / CLAUDE.md it carries, so an agent navigates
  structure instead of reading everything or guessing.
- **The coding suite, wired in**: `experiment` and `propose_fix` are turn
  side-channels now. They existed, were tested, and had no caller outside
  tests — built and unreachable is indistinguishable from absent.
- **Retirement** (`memory.retire_memories`): the assistant can set memories
  aside as no longer relevant to the current iteration of a project — they
  leave recall, consolidation and the delivered-ref set entirely. It is a
  scope judgement, not a truth one (a thing believed wrong is a dispute), it
  requires a stated reason, it is recorded as its own memory, and it is
  reversible from the Memory panel. Commitments, disputed rows and
  retirement notes are refused. Hard deletion is host-only, confirmed, and
  spares anything a summary still cites.
- **Subagents** (`subagents.py`): a `deep` type (full cognitive suite, own
  temporary database, own sandbox, archived then torn down) and a read-only
  `scout`. Spawning requires a host-held grant the assistant can read and
  request but never raise. Batches are coordinated — scopes must not overlap,
  and a change outside an agent's scope is refused at validation. Siblings
  collaborate through the parent, which is the only topology available to a
  subprocess, and only when the message is job-relevant by a rule in code.

## Layout

```
db.py          schema + q/qi/transaction helpers (SQLite, WAL)
providers.py   chat + embeddings, offline degradation, test stubs
memory.py      the memory system (mint, seam, retrieval, consolidation,
               reconciliation, disputes, importance)
beliefs.py     hypothesis persistence/revision; the user model
research.py    hypotheses, evidence, disputes, the automated loop
tools_web.py   search + fetch (stdlib, stubbable)
persona.py     the assistant sheet
prompts.py     the three system prompts
pipeline.py    one chat turn, stage by stage
app.py         FastAPI routes
static/        UI (browser globals: utils.js → chat.js → app.js)
config.py      which provider, configured how (secrets stay in the env)
workspace.py   uploaded files, safe extraction, the sandbox working set
codemap.py     how an agent navigates code it has never seen
subagents.py   delegation, the grant ledger, coordination, collaboration
subagent_runner.py  one deep subagent's whole life, in a subprocess
tests/         187 regression tests, engine-style docstrings, fully offline
```

See `DESIGN.md` for the pipeline written out stage by stage, how
research/hypotheses/evidence/memory interlock, and the theorycraft register
for further assistant-oriented tuning.
