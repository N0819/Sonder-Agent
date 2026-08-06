# Ponder Engine — run and test targets. Mirrors the engine's Makefile
# discipline: `make test` is the full tier and is cheap enough to run freely;
# there is no "fast" tier to hide behind at this size.

PY ?= python3

.PHONY: run serve test compile check

run:
	$(PY) -m uvicorn app:app --reload --port 8010

serve:
	$(PY) -m uvicorn app:app --port 8010

test:
	$(PY) -m pytest tests/ -q

compile:
	$(PY) -m compileall -q .

check: compile test
