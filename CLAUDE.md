# CLAUDE.md

Guidance for Claude Code working in this repository.

## Start here

This project is the cognition core of [Sonder Engine](../Sonder_Engine)
extracted and re-aimed. Read these before any non-trivial change:

1. [`AGENTS.md`](AGENTS.md) — edit routing, the invariants, and the rules
   inherited from Sonder that exist because something broke. **Read this first
   for any behavioural change.**
2. [`DESIGN.md`](DESIGN.md) — architecture, the turn pipeline stage by stage,
   how research/hypotheses/evidence/memory interlock, and the register of what
   is argued but unbuilt.
3. [`README.md`](README.md) — what was inherited from the engine and what was
   cut, with the reasoning.

When a constant here carries a justification, the measurement behind it is
usually the engine's. Do not re-derive or "improve" one without a measurement
of your own — see § Measure before you enrich.

## Commands

```bash
make test    # full suite, offline, no network, no model
make run     # uvicorn on :8010
```

Single test: `python3 -m pytest tests/test_retrieval.py::test_name -q`

Run from the repository root — top-level imports (`from db import q`), not an
installed package. Python 3.11+. The database is `assistant.db`; override with
`ASSISTANT_DB`.

## Architecture

A single-user assistant with real long-term memory and evidence-driven
research. One turn runs: **recall → research (if the question needs it) →
respond → commit**. Everything durable is written by the commit path, and model
output is provisional until deterministic code validates it.

- `memory.py` — the crown jewel. Provenance-typed rows, RRF fusion over vector
  + lexical + aspect rankings, MMR diversity, relevance-gated recall,
  importance revision, disputes, consolidation with per-clause support sets.
- `research.py` — questions become hypotheses; evidence moves confidence
  through bounded arithmetic; contradiction becomes a **dispute, never an
  average**; conclusions must cite evidence actually delivered.
- `coding.py` + `sandbox.py` — coding as the scientific method. A code run is
  an experiment and its result is evidence, so it reuses the research
  epistemics wholesale rather than inventing a second set.
- `ui_review.py` — deterministic structural claims about HTML/CSS/JS, because
  an assistant cannot see a screen and most UI defects are invisible in a
  screenshot anyway.
- `beliefs.py` — the engine's mind-model machinery aimed at the user.
- `persona.py` — the sheet. One **drive**, and it is unsatisfiable by design.
- `pipeline.py`, `app.py`, `static/` — orchestration, HTTP, browser-globals UI.

## Working in this repo

- **Reproduce before you fix.** `coding.propose_fix` enforces this for the
  assistant; the same rule applies to you. A fix for a defect never observed
  failing cannot afterwards be told apart from a fix for nothing.
- **Fix the earliest stage where the data first becomes wrong**, not the last
  stage where it becomes visible.
- **A guard that must be remembered will be forgotten.** Where two spellings of
  one thing can exist, fold the data on the way in rather than adding a helper
  everyone must call — `research.canonical_url` is the worked example, and the
  engine's five-defect identity investigation is why.
- **Measure before you enrich.** The engine's costliest recurring discovery was
  that mechanisms assumed live were not running at all: disputes fired 0 times
  in 181 eligible beats, an entire tier of psychology had never run in 14
  story banks, and a mechanism argued dead turned out to be the most active
  thing in retrieval. This project inherited exactly those mechanisms. Before
  tuning any of them, find out whether it fires — and measure against the
  opportunities it had, never against every row in the table.
- **An empty field fails silently.** `persona_warnings` exists because a sheet
  with three rich fields and one empty reads as complete and is not. The
  failure surfaces much later, looking like a model problem.
- **Bare prohibitions invert.** A prompt clause that only forbids gets read as
  a suggestion of the thing it forbids. Name concrete occasions instead.
- **Cacheability is a property of the prefix.** Invariant contract first,
  `{persona}` last. A variable near the top of a prompt costs the whole prompt
  — measured at 98–99% of a 14k-token prefix in the engine.
- Prefer a deterministic answer over asking a model. Where the engine can
  decide, it should: model output is judged, never trusted.
- Every test docstring says what broke or what is being prevented, not what is
  asserted. The assertion is already in the code.
- Never commit `assistant.db*`, `__pycache__/`, or anything under a sandbox
  workspace.
