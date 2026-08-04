# Design

How the pieces interlock, stage by stage, and where the design should go
next. Companion to `README.md` (what was inherited/cut); this document is
about how the surviving machinery fits together in an assistant.

## The central idea

Sonder Engine keeps objective truth, perception, memory, inference, belief
and narration as distinct information layers because collapsing them between
minds destroys the story. An assistant has one mind — but the same collapse
happens *inside* one mind, and it is the ordinary failure mode of chatbots:
a guess restated as fact, a summary read as experience, a source remembered
as certainty, two contradictory pages averaged into confident nonsense.

So every piece of information in this system carries **how it is known**:

- memory rows carry `provenance` (witnessed/told/read/inferred/remembered);
- projections carry `epistemic_origin` and `temporal_status`;
- summaries are three separate rows by epistemic scope, with per-clause
  host-derived `support` sets;
- conjectures travel under the key `i_suspect`;
- evidence rows carry URLs and stances;
- and everything the model cites is grounded against what it was actually
  shown, with ungrounded citations dropped and warned about.

The deterministic floor never depends on the model cooperating. Model output
is provisional until commit code validates it; a leak (an invented citation,
an ungrounded belief) is an engine failure to be fixed at the seam, and a
warning means the seam worked.

## The turn pipeline

`pipeline.run_turn(user_text, session_id)`:

```
1 recall    deterministic. Global turn ordinal RESERVED from a durable
            counter (not `MAX(turn_idx)`, and no row written yet); pending
            ponder consumed; memory payload built (recent buffer ≤12, ranked
            recall relevance-gated to ≤RECALL_LIMIT via RRF+MMR, three scope
            summaries, ≤2 earlier first-hand chapters, deliberate-recall
            lane); belief payload + stable hypothesis sheet read from state;
            retrieval health measured and warned on.
2 respond   ONE model call. Persona + payload → reply plus typed side
            channels: memory_evidence_used, user_model_updates (each with
            evidence), remember marks, dispute, ponder, research request.
3 ground    deterministic. Citations filtered to delivered refs ∪ {current};
            user-model updates without surviving evidence dropped with a
            warning. Nothing is repaired into plausibility.
4 research  only when requested — the automated loop (below). Its grounded
            answer replaces the respond stage's reply.
4a experiment — the coding suite, wired in. `coding.py` and `sandbox.py`
            had no caller outside tests: a complete implementation of
            coding-as-scientific-method that no turn could reach. Built,
            tested and unreachable is indistinguishable from absent.
4b subagents — only with a grant. Spawned as a COHORT (concurrently, so
            siblings can collaborate while alive), scopes checked for
            overlap first, reports validated, file changes written back to
            the workspace, deep runs archived.
5 commit    deterministic, one transaction, embeddings prepared BEFORE it:
            INSERT the turns row (so the exchange's own record is inside the
            invariant, not beside it) → re-read `state` under the write lock
            (a stage-1 snapshot is a model call old, and stage 5 writes the
            whole blob back) → delete_turn_memories (rerun-safety) → mint
            episode / kept-fact rows / inference rows (salience = 0.45 +
            0.3·confidence) →
            apply_belief_updates → reconcile_inference_confidence (after the
            belief write, so this turn's own inferences are re-weighted by
            the store everything else now reads) → raise_importance for
            load-bearing citations (once-ever) → record_dispute → reselect
            the hypothesis sheet (hysteresis) → store pending ponder →
            update the turn row.
6 settle    post-commit: consolidation when due (≥10 turns or ≥40 rows past
            the cursor). A consolidation failure warns and never rolls back
            the turn — summaries are reconstructible caches.
```

## How research, hypotheses, evidence and memory interlock

```
user question ──► respond stage emits research{question}
                        │
                        ▼
              hypotheses row (confidence 0.3, status open)
                        │
        ┌───────────────┼─── automated rounds (budget 6) ───────────┐
        │               ▼                                           │
        │   model sees: question, i_suspect, confidence, evidence   │
        │   so far, search results, remembered rows, rounds_left    │
        │               │ one action per round                      │
        │   ponder ─────┼── search_memories over own bank; prior    │
        │               │   evidence memories surface WITH urls     │
        │   search ─────┼── tools_web.search → results next round   │
        │   fetch ──────┼── tools_web.fetch → evidence row          │
        │               │     • stance supports/contradicts/context │
        │               │     • confidence moved by BOUNDED step    │
        │               │     • same-URL idempotent (repetition is  │
        │               │       not corroboration)                  │
        │               │     • also minted as `read` MEMORY with   │
        │               │       event_key = evidence:<hash>         │
        │               │     • supports+contradicts → DISPUTE:     │
        │               │       both sides kept, confidence pinned  │
        │               │       to the middle band, never averaged  │
        │   conclude ───┴── citations grounded against delivered    │
        │                   refs; accepted iff grounded AND         │
        │                   (confidence ≥ 0.6 OR disputed OR fully  │
        │                   memory-grounded with no web evidence)   │
        └───────────────────────────────────────────────────────────┘
                        │ budget exhausted
                        ▼
              engine-written hedge, assembled deterministically from
              the evidence table (supports/contradicts, cited)
```

The loop is the mid-task requirement "automate its turns, ponder
strategically until it derives a satisfying answer" made terminating and
checkable: *satisfying* is a predicate (grounded + over the bar, or an
honest dispute, or memory-grounded), *strategic ponder* is a real action
that costs an embedding call instead of a web round and surfaces what the
assistant already read, and *until* is bounded by `RESEARCH_MAX_ROUNDS`
because autonomy is safe to grant exactly when it provably halts.

After the loop, the interlock continues in ordinary turns:

- evidence memories retrieve like any other memory (provenance `what_i_read`,
  URL attached), so an old source can inform — or dispute — a later answer;
- `reconcile_inference_confidence` keeps inference rows ranked by *current*
  credence, so an abandoned working theory stops outranking its replacement
  while remaining recallable as having been held;
- `record_dispute` writes a new reading BESIDE an old evidence memory when
  the world moved (superseded docs, corrections), and the projection carries
  both (`i_now_read_this_differently`);
- consolidation folds the era into scope-separated summaries whose clauses
  carry host-derived support refs.

## Schema (one database, `assistant.db`)

- `sessions`, `turns` — `turn_idx` is a **global** ordinal across sessions:
  memory spans sessions, so the retrieval cutoff needs one shared order.
- `memories` — the engine's table minus frames/location/affect columns.
- `memory_summaries` — keyed `(scope, end_turn_idx)`; `support` JSON.
- `state` — one JSON blob: mind models, hypothesis-sheet keys, pending
  ponder.
- `hypotheses`, `evidence` — the research tables; `evidence.event_key`
  links each row to its memory twin.
- `memory_retrieval_fts` — FTS5, `unicode61 remove_diacritics 2`.

## Theorycraft: tuning the machinery further toward an assistant

A register in the engine's `UNBUILT.md` spirit: argued, prioritised, and
explicitly not built yet. Ordered by expected value per unit of work.

**1. Wall-clock half-lives.** Every decay in `beliefs.py` counts turns
because the engine's clock was the beat. An assistant's gaps are real time:
ten turns in one evening and ten turns across three months are different
epistemic distances. The infrastructure is already shaped for it (the
engine's `_elapsed` accepts `elapsed_seconds`); the work is choosing
per-kind time constants (observation: hours; goal: weeks; preference:
months-to-years) and blending time with turn count so a marathon session
does not age beliefs artificially. This is the highest-value change because
every downstream number (credence, reconciliation, the sheet) reads decay.

**2. Evidence freshness and claim volatility.** A `read` memory about a
fast-moving fact (prices, versions, schedules) should not carry 0.8
confidence forever, while "SQLite shipped in 2000" is stable indefinitely.
Shape: a `volatility` field on evidence-minted memories (model-declared,
engine-capped like `effective_kind` — the stricter of declared and inferred
from claim language: dates/versions/prices infer volatile), feeding a
wall-clock confidence decay for `read` rows in reconciliation. The dispute
machinery then catches what decay misses: a fresh fetch contradicting a
stale memory disputes it rather than silently outvoting it.

**3. Source reliability as a belief subject.** The assistant already holds
mind models about arbitrary subjects; domains are subjects. "docs.example
was accurate the three times I checked" is a trait-kind hypothesis about a
source, earned through dispute outcomes (a source on the losing side of a
resolved dispute takes a suppression; one on the winning side a
reinforcement). Feeds a small scalar bonus/penalty on evidence stance
weight — bounded like every other bonus, so provenance reputation breaks
ties and never outvotes content. This is the engine's "outcome feedback
exists, narrowly" pattern: disputes are the one place research produces a
success signal without trusting a self-report.

**4. Session-boundary consolidation and the opening ponder.** The engine
consolidates on row pressure because stories have no natural chapter breaks;
an assistant has them: session end. Consolidating at session close (plus the
pressure fallback) makes windows align with lived chapters. Its complement
at session *open*: an automatic ponder seeded from `unresolved_threads` —
"what did we leave unfinished" — so continuity is retrieved rather than
hoped for. Cheap: both hooks exist (`maybe_consolidate`, the ponder lane);
this is plumbing, not mechanism.

**5. Commitment governance.** `commitment` rows are protected from
archiving but nothing yet *surfaces* them at the right moment. The engine's
promise ledger suggests the shape: match open commitments against the turn
query as a dedicated aspect ranking (they are few, so the cost is nil), and
let the model emit a `kept`/`released` mark — engine-verified against the
conversation before the row's status moves. The half of the engine worth
NOT importing: intentions decay into dormancy there because fictional minds
must be allowed to drift; an assistant's commitment should nag, not fade.

**6. Proactive contradiction-seeking.** The research prompt asks the model
to seek disconfirmation; nothing enforces it. A deterministic nudge: when a
hypothesis reaches the conclude bar on supporting evidence only, the engine
can spend one automatic round searching a negated query ("X" → "X wrong |
X criticism") before accepting the conclusion. Costs one round, only at the
moment of highest risk (an untested hypothesis about to become an answer),
and needs no model cooperation. The engine's lesson about bare prohibitions
inverting ("never conclude without checking" reads as an argument against
checking) says: build the check into the loop, not the prompt.

**7. Retrieval aspects from the user model.** Aspects currently carry the
hypothesis sheet, unresolved threads and standing commitments. The user
model's high-confidence `preference` beliefs are a natural fourth facet
("how the user wants things done"), so a question about formatting recalls
the exchange where the preference was set. Free at retrieval time (aspects
share one embedding batch); the risk is aspect dilution — the engine's
set-membership finding says each aspect adds ~0.1 to its members, so beyond
four or five facets the bonus band drowns the ranking band. Measure first.

**8. The relevance gate, and why its constant is not touched yet.** The gate
is documented as cutting "where the fused score falls off a cliff", keeping an
ordinary question in the 12–25 range. It does not, and the reason is
structural rather than a bad number: RRF contributes `12/(60+rank)` per lane,
so rank 60 scores 50.8% of rank 1 *within* a lane while
`_RELEVANCE_FLOOR_RATIO` is 0.45. Nothing below 0.5 can discriminate inside a
lane, so the cut is really "how many of the four lanes did this row hit" — a
four-valued signal. Measured: 55–65 rows returned per turn against a 360-row
bank, saturating `RECALL_LIMIT`.

Two candidate shapes, both needing a measurement this project cannot yet make:
(a) cut the vector lanes on *similarity* relative to the best match before
they become ranks, so lane membership is itself relevance-bounded rather than
a flat `[:60]`; (b) cut on the derivative of the fused curve — an actual knee
— instead of a ratio of a deliberately flat function. What blocks the choice
is that the only offline corpus runs under `cheap_embed`, where everything
correlates with everything; picking a constant there is the small-payload
benchmark scar exactly. `build_memory_context` now exports `recall.returned`
and `recall.at_ceiling` per turn, so the number can be earned the way the
engine's numbers were.

**9. Re-embedding after a provider change — BUILT.** Both halves are now in
place: retrieval *reports* stranded vectors (`retrieval_health`, plus a
warning on the settings page at the moment of the decision), and
`memory.rebuild_embeddings` migrates them. Incremental, re-runnable, and it
refuses when no provider is reachable rather than overwriting real vectors
with hashing-trick ones — the guard that matters, because `fallback` is true
only when a CONFIGURED provider fails and is false when none is configured at
all.

It covers **both** embedded tables, which it did not at first. `memories` was
migrated and `memory_summaries` was not, and the summary side fails harder: a
stranded memory still reaches recall by keyword, while
`search_memory_summaries` *skips* a cross-model window instead of scoring it,
so every consolidated window left retrieval outright — and the rebuild then
reported the bank fully comparable. Measure before you enrich cuts both ways:
the mechanism was firing, on one of the two places it had to. The second scar
is smaller and the same shape: each document was reconstructed from the prose
fields only, so `_memory_document` fell back to `kind: episodic` /
`source: witnessed`, and a rebuilt vector encoded text the row never had.

What is still not built: an automatic sweep. A rebuild costs money per
token and should be a decision, not a side effect of opening Settings.

**10. What to keep refusing.** No unconditional recency bonus (recency has
its own field); no retrieval-driven importance (popularity loop); no
averaging of disputed claims (the number would claim knowledge nobody has);
no ANN index until a bank is measured slow (the engine's structural
argument holds here unchanged: the turn cutoff must run before ranking, and
one user's bank at ~3–4 rows/turn is years away from the 2.2s/200k-row
mark); no model-written audit trails (support sets stay host-derived).

## Delegation

```
                 ┌─────────────── the parent ───────────────┐
 user grants ──► │ grant ledger (host-held, model-readable) │
                 │ plan_assignments: scopes must not overlap│
                 └───┬──────────────────────────┬───────────┘
                     │ spawn_cohort (threads)   │
        ┌────────────▼──────────┐   ┌───────────▼───────────┐
        │ deep (subprocess)     │   │ deep (subprocess)     │
        │ own temp DB, sandbox  │   │ own temp DB, sandbox  │
        │ owns tokenizer.py     │   │ owns parser.py        │
        └───┬───────────────┬───┘   └───┬───────────────────┘
            │ ASK PARENT    │ TELL SIBLINGS                 │
            ▼               └──────────► routed if job-relevant
     memory first, then a scout          (mailbox, never blocking)
     if memory is thin (same ledger)
            │
            ▼  report ─► validated ─► absorbed as `told`
                                  ─► file_changes written back
                                  ─► run archived (deep only)
```

Scouts are erased because there is nothing to keep: one call, no state, no
bank. Deep runs are archived because their working-out is the thing you want
when a report turns out to be wrong — and the archive is inert, read by a
human and never by recall.

## Known simplifications (recorded, not hidden)

- Retirement has no explicit "project" or "iteration" entity: the grouping is
  the retirement BATCH plus its stated reason, which is what a user actually
  reviews. A first-class project scope would let rows be retired and restored
  wholesale when switching between two live efforts; the batch is enough for
  the linear case and nothing yet measures the other one.
- Retired rows are never purged automatically. An age-based sweep would be
  easy (`purge_retired(older_than_turns=...)` already takes the argument) and
  is deliberately not wired to anything: silent destruction is the failure
  mode this whole design is arranged against.

- The research `fetch` action declares stance and excerpt in the same round
  as the fetch — the model judges from the search snippet, and the engine
  substitutes page text when no excerpt is given. A two-round read-then-
  judge shape is more honest and costs one round per source; adopt it if
  measured stance quality warrants.
- No rerun/regenerate UI, though the commit path is rerun-safe
  (`delete_turn_memories` + event keys) and the seam's cutoff assumes
  replays exist — the engine's audit found exactly that hole after
  assuming replays were hypothetical. One known gap if a replay UI lands:
  `record_dispute` increments its `count` and lifts importance on every
  call, so a replayed turn that disputed a memory would inflate the
  "re-read twice means genuinely unstable" signal without anything having
  been re-read.
- The sandbox does not confine the filesystem, block the network, or cap
  process count. It bounds wall clock, CPU, address space, file size,
  descriptors, output read into the parent, and the process tree. The
  header says so precisely; the previous wording claimed two protections
  that measurement showed were absent. `bwrap --unshare-all` is the
  one-line upgrade when it matters.
- A deep subagent's collaboration is turn-granular: a sibling's message
  arrives at the start of the recipient's next turn, so a finding can be up
  to one turn stale. Blocking RPC would remove the lag and introduce
  deadlock between two agents in one area, which is the case the feature
  exists for.
- Sibling relevance is lexical (shared paths, shared directory, content-word
  overlap). It will occasionally drop a message that mattered; it errs that
  way deliberately, because a delivered irrelevant message costs a whole
  subagent turn.
- The Claude Code provider runs with `--tools ""` and a neutral cwd: the
  assistant's tools are its own and are governed by the epistemics here, so
  the CLI is used for composition only. A provider that ran its own tool
  loop would put actions outside every guard in this repository.
- The research `fetch` action refuses non-public addresses and re-checks
  every redirect hop, but it is not a full SSRF boundary: DNS rebinding
  between the check and the connect is unaddressed, since the check and the
  socket resolve the name independently.
- Session titles, memory editing UI, and archive export are absent. The
  memory panel is read-only.
