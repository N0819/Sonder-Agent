# conftest.py — every test runs against a temp database and offline
# providers. Engine rule carried over: the deterministic floor must not
# depend on a model cooperating or a network existing, and the test tier is
# where that claim is proved rather than assumed. cheap_embed is the
# embeddings provider under test by construction (no env config), which also
# exercises the exact degraded mode a fresh install runs in.

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import providers  # noqa: E402
import tools_web  # noqa: E402


@pytest.fixture
def temp_db(tmp_path):
    db.configure(str(tmp_path / "test.db"))
    yield
    db.close()


@pytest.fixture(autouse=True)
def _clean_stubs():
    """Stubs are process-global; a test that forgets to clear one would make
    its neighbour's 'no model configured' path silently take the stubbed
    path instead. Clearing them around every test makes that impossible."""
    yield
    providers.set_chat_stub(None)
    tools_web.set_search_stub(None)
    tools_web.set_fetch_stub(None)
    # Ensure no env leakage turns a later test's providers 'configured'.
    for var in ("ASSISTANT_CHAT_BASE", "ASSISTANT_CHAT_MODEL",
                "ASSISTANT_EMBED_BASE", "ASSISTANT_EMBED_MODEL"):
        os.environ.pop(var, None)
