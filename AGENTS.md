# AGENTS.md

What to touch for which change, what must stay true, and the rules that exist
because something broke.

## Edit routing

| Changing… | Touch | Also check |
|---|---|---|
| what is recalled, how it is ranked | `memory.py` | `tests/test_retrieval.py`; the relevance gate, not a `k` |
| what a memory row means, provenance | `memory.py`, `db.py` | consolidation scopes, `tests/test_consolidation.py` |
| importance, disputes | `memory.py` | `tests/test_importance_and_disputes.py` |
| how a question becomes research | `research.py` | termination predicate, `tests/test_research.py` |
| running code, sandbox limits | `sandbox.py` | `tests/test_coding_suite.py` |
| the scientific-method rules | `coding.py` | the four rules below |
| reading HTML/CSS/JS | `ui_review.py` | each check must name a defect that bites |
| what the assistant knows about the user | `beliefs.py` | evidence gating |
| who the assistant is | `persona.py` | `_FIELDS` — every field is required |
| any prompt | `prompts.py` | **variables last**; render via `prompts.render`, never `.format` at the call site |
| turn order | `pipeline.py` | commit ordering below |
| schema | `db.py` | additive migrations only, keyed on `schema_version` |
| anything that writes | `db.py` | `tests/test_durability.py` — a green suite is not evidence a write committed |
| which model composes a reply | `config.py`, `providers.py` | a key is stored (`*_key`) or named (`*_key_env`); stored wins, and `redacted_status` is the only thing keeping it off the wire |
| which model embeds | `config.py` (`embed_*`) | a SEPARATE provider; changing it strands the bank — offer the rebuild |
| uploaded files, archives | `workspace.py` | the threat model is the ARCHIVE, not the user |
| how an agent navigates code | `codemap.py` | an index, never a summary |
| delegating work | `subagents.py` | the grant ledger; `tests/test_subagents.py` |
| what a subagent may touch | `subagents.py` | scope is enforced at validation, not in a prompt |
| a mechanism you believe is live | anywhere | count it firing before you tune it; `tests/test_silent_mechanisms.py` |

## Invariants

**Model output is provisional until deterministic code validates it.** A claim
that cites nothing loses the citation, not the benefit of the doubt. This is
the single rule everything else here serves.

**Contradiction is a dispute, never an average.** Two live sources disagreeing
produce a confidence that represents neither. Both readings are kept, both are
cited, and the answer is forced to say "sources disagree". The same applies to
an experiment that disagrees with its own earlier run — that is
non-determinism, and smoothing it destroys the only interesting fact available.

**Commit ordering.** Embed before the write lock; all durable turn mutations in
one transaction — **including the `turns` row itself**; consolidation
afterwards, because a summary is a reconstructible cache and must never be able
to roll back a valid turn.

The turn row used to be INSERTed at the top of `run_turn` and autocommitted, so
the one record naming the exchange sat outside the invariant that names it: an
exception mid-commit rolled the memories back and left a turn with a user
message, no reply, and a consumed ordinal. The ordinal is still *reserved*
early (stage 1's cutoff needs it) but through a durable counter in `meta`, not
by writing a row.

**A transaction only exists if the connection lets it.** `sqlite3`'s legacy
`isolation_level` opens an implicit transaction before every write. Under it
`qi`'s "commit when not in a transaction" guard could never fire, `transaction()`
mistook itself for nested and skipped its COMMIT, and **nothing this application
wrote ever reached the disk** — a whole session lived in one open write
transaction that `close()` rolled back. All 80 tests passed throughout, because
a test is one thread on one connection and a connection reads its own
uncommitted writes. `isolation_level = None` and `BEGIN IMMEDIATE` are load
bearing; so is `tests/test_durability.py`, which asserts from a *second*
connection.

**Reserve the ordinal, do not compute it.** `SELECT MAX(turn_idx)` then INSERT is
a read-modify-write with no lock between the halves; two concurrent turns both
claimed the same ordinal and the second's `event_key` upsert silently
overwrote the first turn's episode. Two browser tabs was enough.

**One thing, one spelling.** Where two spellings of one thing can exist, fold
on the way in. `research.canonical_url` folds six spellings of a URL into one,
because evidence idempotency keyed on the raw string meant six fetches of one
page could carry a hypothesis from 0.30 to past 0.90 — repetition wearing
corroboration, arriving through the door the idempotency check left open.

**Recall is bounded by relevance, not by an attention budget.** The engine
capped recall at 16 because a character has to be a person. This does not. The
cut is made on the fused score; `RECALL_LIMIT` survives only as a payload
ceiling.

Measured caveat, recorded because the claim above is stronger than the code:
RRF contributes `12/(60+rank)` per lane, so rank 60 scores **50.8%** of rank 1
*within a lane* — and `_RELEVANCE_FLOOR_RATIO` is 0.45, below that. No ratio
under 0.5 can discriminate inside a lane at all, so what the gate actually cuts
on is how many of the four lanes a row appeared in: a coarse four-valued
signal, not a cliff. Over 120 synthetic turns against a 360-row bank, recall
returned 55–65 rows and saturated the ceiling. That measurement ran under
`cheap_embed`, which the README records at 0% recall on vocabulary-disjoint
paraphrases — the worst possible corpus to tune against — so the constant is
**not** retuned. `build_memory_context` now exports `recall.returned` /
`recall.at_ceiling` in the turn trace so the decision can be made on real
traffic. See DESIGN.md § Theorycraft.

## The four rules of the coding suite

1. **A prediction before a run, or it is not an experiment.** `expect` is
   required and graded mechanically, outside the thing being graded. Running
   code and then deciding what it proved is how a model talks itself into a
   fix.
2. **Reproduce before you fix.** `propose_fix` refuses a fix for a defect never
   observed failing. Afterwards, nobody can tell such a fix from a fix for
   nothing.
3. **A failure is data, never an error.** Crashes, timeouts and missing
   interpreters come back as observations. The moment a harness raises on its
   own negative results, the loop starts avoiding them.
4. **Inconclusive is not a soft refutation.** "The interpreter was missing"
   says nothing about the hypothesis. Folding it into `contradicts` would move
   confidence on the strength of a broken harness.

## Inherited scars

Each of these cost the engine something real. They are listed because the
cheapest way to pay for them again is to not know about them.

- **An empty authored field fails silently.** Rich prose in three fields and an
  empty fourth reads as complete. The symptom arrives fifty exchanges later and
  looks like a model problem. `persona_warnings` flags every empty field; the
  drive is checked like the rest and is never optional.
- **A drive must be unsatisfiable.** Put motivation in goals and it decays with
  them — the engine watched a courier walk sixteen optimal rooms to his
  destination and turn away, because nothing underneath the spent goals wanted
  it. "Assist the user" is the right shape: it cannot be completed and cannot
  be traded away.
- **Bare prohibitions invert.** A clause that only forbids reads as a
  suggestion. Name the concrete occasions instead: the engine's dispute
  mechanism fired zero times in 181 eligible beats against a prohibition-only
  clause, beside a sibling naming four occasions that fired 89%.
- **Measure against opportunities, not rows.** A rate computed against every
  row reads catastrophic for a field that did not exist for most of the corpus,
  and means nothing. Against the beats that could have carried one, the same
  number is a diagnosis.
- **`==` on a thing that carries two names fails open and silently.** Five
  separate defects in one engine investigation were that comparison, and one of
  them was a firewall failing open.
- **A small-payload benchmark does not predict real behaviour.** Every pair the
  engine re-tested against its own prompt and validator inverted, one by 12×.
  Measure the real contract or do not claim a ranking.
- **Cacheability is a prefix property.** `{name}` at byte 32 of a 55kB prompt
  left an 8-token cacheable prefix; relocating it recovered 98–99% across a
  14-model sweep. Invariant text first, always. (Verified here: RESPOND is
  70.4% cacheable, RESEARCH 65.4%, and both are byte-identical across turns
  while the persona is unchanged.)

## Scars earned in this project

Each was found by counting, not by reading. None raised, warned, or failed a
test.

- **A mechanism nothing reads fails harder than an empty field.** `persona.py`
  argues at length that the drive is the field whose absence costs most — and
  `persona_prompt` never rendered it. `persona_warnings` stayed silent because
  the field was *full*. Where an empty field fails silently, an unread one
  defeats the check built for it.
- **Whether a template has a variable is not a safe thing for a call site to
  know.** Two prompts were `.format`ed and one was not, so the third's escaped
  `{{` braces reached the model literally and the consolidator's JSON failed to
  parse — the entire summary layer dying every turn behind a warning. Every
  prompt now goes through `prompts.render`.
- **Two constants that were never compared to each other.**
  `_ARCHIVE_KEEP_RECENT` (12) exceeded `CONSOLIDATE_EVERY_TURNS` (10), so the
  archive cutoff always clamped to the window's own start and nothing in a
  window was ever older than that. 0 rows archived in 40 turns. Neither number
  is wrong alone.
- **Negation is invisible to an overlap matcher, and "not" was a stopword.**
  "prefers X" and "does not prefer X" scored 0.667 against a 0.4 threshold, so
  a user CORRECTION reinforced the belief it contradicted and explaining-away
  never fired. Contradiction being averaged, arriving through the stopword
  list.
- **A docstring claiming "the same rule" is not the same rule.**
  `belief_credence` promised the merge's matcher and passed different
  arguments to it; the drift failed toward over-confidence.
- **"Refuted" means the prediction was wrong, not that anything failed.** A
  typo'd `expect` opened the reproduce-before-you-fix gate on a run that
  exited 0.
- **A checker that cries wolf gets switched off.** `ui_review` returned 42
  findings against this project's own `static/`, 40 of them false — including
  a "selector" that was a sentence from a CSS comment. Against that noise a
  true positive is unfindable. Precision is a correctness property here, and
  `tests/test_ui_review_precision.py` pins it.
- **A cap applied after the fact is not a cap.** The sandbox truncated output
  to 20,000 characters *after* buffering the whole stream, so model-written
  code took the host process from 12 MB to 3,181 MB. Bound the read, not the
  result.

## Retiring memories

The one path reachable from model output that REMOVES information rather than
annotating it. Everything else here exists to stop exactly that, so the line
matters.

**It is a claim about scope, never about truth.** "Irrelevant to the current
iteration" means superseded — the Postgres schema you replaced with SQLite.
A thing believed WRONG is a `dispute`, which keeps both readings. Recall is a
scarce resource and superseded project context taxes every answer; `archived`
does not help, because an archived row has only left the rolling consolidation
window and is still fully recallable, deliberately.

**Retire, don't delete.** `retired` is a column, exactly like `disputed`, for
the same reason: the row is still the evidence. Retired rows leave recall
entirely, do not reach consolidation, and are not delivered as refs — so
retirement is total, not cosmetic — and they still exist and come back.
Scope judgements are fallible in a specific way: the Postgres decision becomes
relevant again the moment somebody asks why you moved. That should cost a
restore, not the information.

**Refused, and why each one:** `commitment` rows (an open promise must nag,
not fade — a commitment the assistant can retire when it feels stale is not
one), disputed rows (the dispute IS the record that something was unstable),
and retirement notes themselves (an assistant that can forget that it forgot
will later tell you confidently that it never knew).

**A reason is required, and the act is remembered.** The retirement mints its
own `witnessed` row naming what went and why. Refs are grounded against
delivered refs like every other citation: a ref the model was not shown is not
a ref it may discard.

**`purge` is host-only and there is no side channel for it.** Retiring is a
reversible relevance judgement; destroying the record is a different act and
belongs to the person whose records they are. Purge also spares rows a summary
still cites — those refs are the audit trail for a clause the assistant keeps
asserting. `tests/test_retirement.py` asserts the absence of a model path.

## Subagents, and the firewall this project thought it did not need

README says the engine's information firewall was cut because "an assistant
has no second mind to be kept out of". **Subagents make that sentence false.**
The firewall is still not reintroduced as a mechanism, but its internal form —
provenance labelling — is now load-bearing at a seam where it was previously
decorative.

**A report is testimony.** A child's report enters the parent as
`provenance='told'`, attributed to the child, its claims kept only where their
citations name evidence the child actually filed. Absorbing it as first-hand
would be the layer collapse this project exists to prevent, at the one place a
second mind genuinely exists.

**The allowance is host-held.** The assistant READS its allowance (delivered in
every turn payload, so consulting it is free) and may REQUEST more. It cannot
grant itself anything: `spawn` refuses, the decrement happens in `subagents.py`,
and no prompt text is load-bearing anywhere in that sequence. `grant` and
`record_request` are two functions rather than one with a flag, so the path the
model can reach and the path that raises a budget do not overlap.

**A failed spawn still spends its grant.** Conservative on purpose: a crashed
child that cost nothing is a retry loop against the user's budget.

**Coordination is checked, not prompted.** Two agents assigned one file is a
property of the ASSIGNMENT, so `plan_assignments` drops the second claim and
`validate_report` refuses changes outside an agent's scope. Told-not-to is not
the mechanism; cannot is.

**Siblings talk through the parent because there is no other topology.** A child
is a subprocess holding one pipe, to its parent. Messages are queued and
delivered at the start of the recipient's next turn — never a blocking call,
because two agents working the same area is exactly when child-to-child RPC
deadlocks. Relevance is decided in code (`job_relevant`): shared paths, shared
directory, or content overlap with the recipient's brief. An irrelevant message
costs a sibling a whole turn, so it is dropped with a reason.

**Deep runs are archived; scouts are not, and that is structural.** A scout has
no working directory at all — no database, no sandbox, no bank — so there is
nothing of it that outlives its answer. A deep run leaves a scratch database
worth opening when a report later looks wrong. The archive is **forensics for a
human and is never reachable by recall**: a subagent's bank holds its own
episodes and half-formed inferences, and making those retrievable would mean
the parent recalling a dead agent's private working-out as its own experience.

**A deep subagent's drive is completion, and that is the one place the
unsatisfiable-drive rule is suspended.** The courier scar requires a mind that
OUTLIVES its goals; this one does not — task done and agent gone are the same
event. See the block in `subagent_runner.py` before changing it back.

**A child's file changes must be written back or the work is lost.** The child
edits inside a directory deleted the moment it reports. `apply_changes` is what
makes "it fixed the tokenizer" a fact about the workspace rather than a
sentence.

## What is deliberately not here

No perception, no spatial model, no attire, no director or narrator, no
interaction loops, no drives-plural, no affect, no somatic state, and no
information firewall. The firewall in particular is absent on purpose: it
exists to keep fictional minds out of each other's heads, and an assistant has
no second mind to be kept out of. Do not reintroduce a mechanism from the
engine without saying which question here it answers.
