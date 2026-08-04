# config.py — one place that answers "which provider, configured how".
#
# Configuration used to live entirely in the process environment, which meant
# the only way to point the assistant at a model was to restart it with
# different exports. That is fine for a daemon and wrong for something with a
# browser UI: the settings tab needs to read and write this, and the pipeline
# needs to see the change on the next turn without a restart.
#
# SECRETS: TWO PATHS, AND WHY THE WEAKER ONE IS THE DEFAULT ONE.
#
# This file used to store only the NAME of an environment variable, never a
# key. The argument was sound and is unchanged: a key written to `settings`
# sits in plaintext inside `assistant.db` — the same file the memory panel
# reads, the same file that travels with every backup and every "can you look
# at my database" copy. A memory bank is a thing you share; a credential is
# not, and the moment they live in one file the safe handling of both becomes
# the unsafe handling of one.
#
# What that argument left out is the cost it charges. Env-only means the key
# is entered by `export` in the shell that launches the process, and a running
# process cannot see an export made after it launched — so every key change
# is an export plus a RESTART, and getting it wrong produces a 401 against
# configuration that looks perfectly correct. For a single-user app with a
# settings page, that is a bad trade, and it produced exactly one confusing
# failure per attempt for the person who owns the machine, the database and
# the key.
#
# So both paths exist. `*_key_env` names a variable and remains the better
# option for anything deployed; `*_key` (KEY_VALUE_FIELDS) stores the key
# outright and is what the settings page offers, because "paste it and press
# save" is what a user of their own tool should get. A stored key WINS over
# the variable, so what you typed is what runs.
#
# The original argument survives as three enforced consequences, not as a
# prohibition: `redacted_status` never returns a stored key to the browser
# (only whether one exists and where it came from), `db._restrict_permissions`
# chmods the database 0600 because it is now a credential store, and clearing
# a stored key is a first-class action rather than an omission.

import os
import re

from db import setting_get, setting_put

# provider ids the chat role understands.
PROVIDER_HTTP = "openai-compatible"
PROVIDER_CLAUDE_CODE = "claude-code"
CHAT_PROVIDERS = (PROVIDER_HTTP, PROVIDER_CLAUDE_CODE)

_SETTINGS_KEY = "providers"

# Every field, its default, and — for key fields — the fact that the value is
# an env var NAME. Defaults come from the environment so an existing
# env-configured install keeps working with no settings row at all.
_DEFAULTS = {
    "chat_provider": lambda: (PROVIDER_HTTP
                              if os.environ.get("ASSISTANT_CHAT_BASE")
                              else PROVIDER_HTTP),
    "chat_base": lambda: os.environ.get("ASSISTANT_CHAT_BASE", ""),
    "chat_model": lambda: os.environ.get("ASSISTANT_CHAT_MODEL", ""),
    "chat_key_env": lambda: "ASSISTANT_CHAT_KEY",
    # Claude Code CLI
    "claude_binary": lambda: "claude",
    "claude_model": lambda: "",          # "" = whatever the CLI defaults to
    "claude_timeout": lambda: 180.0,
    # EMBEDDINGS ARE THEIR OWN PROVIDER, and always HTTP.
    #
    # Not a convenience — a necessity. The Claude Code CLI has no embeddings
    # endpoint, so an assistant composing replies through it still needs
    # somewhere to vectorise, or three of the four ranking lanes go dark and
    # recall silently falls back to keyword match. The two roles are
    # configured separately because they genuinely are two services, and the
    # chat side is the one with a non-HTTP option.
    "embed_base": lambda: os.environ.get("ASSISTANT_EMBED_BASE", ""),
    "embed_model": lambda: os.environ.get("ASSISTANT_EMBED_MODEL", ""),
    "embed_key_env": lambda: "ASSISTANT_EMBED_KEY",
    # The credentials themselves, when the user chooses to store them rather
    # than export them. See KEY_VALUE_FIELDS for what that costs and what is
    # enforced in exchange. Default empty: an install that uses the
    # environment never acquires one.
    "chat_key": lambda: "",
    "embed_key": lambda: "",
}

# Known endpoints, so a working setup is a dropdown rather than a URL nobody
# remembers. Presets fill the fields; they are not a separate code path, and
# anything OpenAI-compatible still works by typing a base URL.
#
# OpenRouter standardised an OpenAI-shaped `/embeddings` endpoint, so it can
# serve either role. Perplexity's embedding models are only reachable through
# it — there is no direct Perplexity embeddings API — which is precisely the
# case that makes a separate embeddings provider worth having.
CHAT_PRESETS = {
    "openrouter": {"label": "OpenRouter",
                   "base": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY",
                   "example_model": "anthropic/claude-sonnet-4.5"},
    "openai": {"label": "OpenAI",
               "base": "https://api.openai.com/v1",
               "key_env": "OPENAI_API_KEY",
               "example_model": "gpt-4o"},
    "local": {"label": "Local (llama.cpp / Ollama / vLLM)",
              "base": "http://localhost:11434/v1",
              "key_env": "ASSISTANT_CHAT_KEY",
              "example_model": ""},
}

EMBED_PRESETS = {
    "openrouter-pplx-4b": {
        "label": "OpenRouter — Perplexity Embed V1 4B",
        "base": "https://openrouter.ai/api/v1",
        "model": "perplexity/pplx-embed-v1-4b",
        "key_env": "OPENROUTER_API_KEY",
        "note": "32K context, $0.03/M tokens. Built for web-scale retrieval; "
                "the larger of Perplexity's two embedding models.",
    },
    "openrouter-pplx-0.6b": {
        "label": "OpenRouter — Perplexity Embed V1 0.6B",
        "base": "https://openrouter.ai/api/v1",
        "model": "perplexity/pplx-embed-v1-0.6b",
        "key_env": "OPENROUTER_API_KEY",
        "note": "32K context, $0.004/M tokens. Lighter and cheaper; the one "
                "to try if 4B is more than this bank needs.",
    },
    "openrouter-openai-3-small": {
        "label": "OpenRouter — OpenAI text-embedding-3-small",
        "base": "https://openrouter.ai/api/v1",
        "model": "openai/text-embedding-3-small",
        "key_env": "OPENROUTER_API_KEY",
        "note": "The familiar baseline, through the same key.",
    },
    "openai-3-small": {
        "label": "OpenAI — text-embedding-3-small",
        "base": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "key_env": "OPENAI_API_KEY",
        "note": "Direct, if you already hold an OpenAI key.",
    },
    "openai-3-large": {
        "label": "OpenAI — text-embedding-3-large",
        "base": "https://api.openai.com/v1",
        "model": "text-embedding-3-large",
        "key_env": "OPENAI_API_KEY",
        "note": "Higher quality, 3072 dimensions, more storage per row.",
    },
}

# Which fields name an environment variable rather than carrying a value.
KEY_FIELDS = frozenset({"chat_key_env", "embed_key_env"})

# Which fields carry a credential OUTRIGHT, stored in `assistant.db`.
#
# THE MODULE HEADER ARGUES AGAINST THIS, and the argument still stands — it is
# why every one of these fields is paired with a `_key_env` alternative that
# remains the better option for anyone deploying rather than using this. It is
# stored anyway because the header's conclusion ("keep it in the shell") is a
# real cost paid on every restart by a single-user app with a settings page,
# and an owner who wants to type a key into their own tool is entitled to.
#
# What the header's reasoning DOES still buy, and what is therefore enforced:
#   - `redacted_status` never returns these to the browser, only presence.
#     The settings page can report a stored key, never re-display it.
#   - `db.connect` chmods the database 0600 on open, because "the same file
#     that travels with every backup" is now also a credential store.
#   - a key here beats the environment (below), so what you typed is what
#     runs — a stale export silently shadowing it is the failure this whole
#     episode was made of.
KEY_VALUE_FIELDS = frozenset({"chat_key", "embed_key"})

# name field -> the value field that overrides it.
_KEY_VALUE_FOR = {"chat_key_env": "chat_key", "embed_key_env": "embed_key"}

# Sent by the UI's "Forget stored key" control. An empty string cannot mean
# "clear" because it is what an untouched password field submits, and a
# credential you cannot delete through the interface that stored it is a
# worse position than never having stored it.
CLEAR_SECRET = "__clear__"


def get_config():
    """The effective configuration: stored settings over environment
    defaults.

    MAY CONTAIN A SECRET — `KEY_VALUE_FIELDS` hold credentials outright. This
    is the internal view; `redacted_status` is the boundary that strips them,
    and it is the only shape a route may return."""
    stored = setting_get(_SETTINGS_KEY)
    stored = stored if isinstance(stored, dict) else {}
    out = {}
    for field, default in _DEFAULTS.items():
        value = stored.get(field)
        out[field] = default() if value in (None, "") else value
    if out["chat_provider"] not in CHAT_PROVIDERS:
        out["chat_provider"] = PROVIDER_HTTP
    return out


# The provider layer appends the route to the base — `/embeddings`,
# `/chat/completions`. A base URL that ALREADY ends in its route therefore
# doubles it into a 404, and the message the UI shows for that is
# "HTTP Error 404", which names neither the cause nor the field.
#
# Copying a URL from a provider's docs gives you the full endpoint, so this is
# the likely input rather than a careless one — and it is exactly the identity
# case AGENTS.md settles by folding on the way in. Two spellings of one base
# become one at the point of entry, instead of a guard in the request path
# that everybody has to remember.
_ROUTE_SUFFIXES = ("/chat/completions", "/completions", "/embeddings")


def _fold_base_url(value):
    """Strip a trailing API route from a base URL. Idempotent."""
    value = str(value or "").strip().rstrip("/")
    changed = True
    while changed:
        changed = False
        for suffix in _ROUTE_SUFFIXES:
            if value.lower().endswith(suffix):
                value = value[:-len(suffix)].rstrip("/")
                changed = True
    return value


def config_default(field):
    """The value a field falls back to with nothing stored."""
    factory = _DEFAULTS.get(field)
    return factory() if factory else ""


def _is_env_var_name(value):
    """A key field holds the NAME of an environment variable. A name is an
    identifier; a credential is not — every key format in circulation carries
    hyphens or dots, and none is a valid shell identifier.

    Deciding by "is this a well-formed name" rather than by "does this look
    like a secret" is the deterministic direction: it needs no list of vendor
    prefixes to stay current, and the failure it prevents is the serious one
    (see save_config)."""
    value = str(value or "").strip()
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)) and len(value) <= 64


def save_config(values):
    """Write the settings row. Unknown fields are dropped rather than stored,
    so a stray key from a future UI cannot become configuration nobody reads.
    Returns the effective config plus warnings.

    A credential pasted into the VARIABLE-NAME field is not refused and not
    stored where it was typed — it is routed to the paired value field, which
    is where a key belongs now that keys can be stored at all. Refusing was
    correct only while there was nowhere to put it. Folding the mistake into
    the right shape on the way in is the same rule `_fold_base_url` follows,
    and it means the obvious action (paste key, press save) simply works
    wherever the user does it."""
    values = values if isinstance(values, dict) else {}
    clean, warnings = {}, []
    for field in _DEFAULTS:
        if field not in values:
            continue
        value = values[field]
        if field == "claude_timeout":
            try:
                clean[field] = max(10.0, min(float(value), 900.0))
            except (TypeError, ValueError):
                continue
        elif field in KEY_VALUE_FIELDS:
            value = str(value or "").strip()
            # EMPTY MEANS LEAVE ALONE. The page cannot re-display a stored key
            # (redacted_status will not hand it over), so the field it draws
            # is necessarily blank — and a blank submit that cleared the key
            # would delete the credential every time any other setting was
            # saved. Clearing is therefore explicit.
            if value == CLEAR_SECRET:
                clean[field] = ""
            elif value:
                clean[field] = value
        elif field in KEY_FIELDS:
            value = str(value or "").strip()
            if value and not _is_env_var_name(value):
                clean[_KEY_VALUE_FOR[field]] = value
                warnings.append(
                    f"that looks like a key, not a variable name, so it was "
                    f"saved as the stored {_KEY_VALUE_FOR[field]} and is in "
                    "use now. It lives in assistant.db — keep that file out "
                    "of backups you share, or clear it here and export "
                    f"{config_default(field)!r} in the shell instead.")
                continue
            clean[field] = value
        elif field in ("chat_base", "embed_base"):
            clean[field] = _fold_base_url(value)
        else:
            clean[field] = str(value or "").strip()
    # MERGE, never replace. This wrote `clean` as the entire settings row, so
    # a save that mentioned some fields silently reset every field it did not
    # — and `apply_preset` is precisely such a save. Choosing an embeddings
    # preset sends three embed_* fields and used to wipe chat_base,
    # chat_model and the claude_* block back to their environment defaults:
    # configuring one provider destroyed the other, with the page reporting
    # success. A partial update is the normal case, not the exception.
    stored = setting_get(_SETTINGS_KEY)
    stored = stored if isinstance(stored, dict) else {}
    setting_put(_SETTINGS_KEY, {**stored, **clean})
    config = get_config()
    return config, warnings + config_warnings(config)


def secret_for(field):
    """The credential for a key field. The only function that hands back an
    actual secret, and it hands it to the provider layer, never to a route.

    A STORED KEY WINS OVER THE ENVIRONMENT. The reverse order would let an
    export made months ago silently shadow the key just typed into Settings,
    and the symptom of that is a 401 with correct-looking configuration — the
    exact failure this pair of fields exists to end. Falling back to the
    variable keeps every environment-configured install working untouched."""
    config = get_config()
    stored = str(config.get(_KEY_VALUE_FOR.get(field, ""), "") or "").strip()
    if stored:
        return stored
    name = str(config.get(field) or "").strip()
    if not name:
        return ""
    return os.environ.get(name, "")


def secret_source(field):
    """Where `secret_for` got it: 'stored', 'environment', or ''. The settings
    page needs to distinguish "no key" from "a key you cannot see"."""
    config = get_config()
    if str(config.get(_KEY_VALUE_FOR.get(field, ""), "") or "").strip():
        return "stored"
    name = str(config.get(field) or "").strip()
    return "environment" if name and os.environ.get(name, "") else ""


def apply_preset(role, preset_id):
    """Fill the fields for a known endpoint. Returns (config, warnings).

    A preset sets the base, the model and the NAME of the key variable — it
    never sets a key, because this module never holds one."""
    table = CHAT_PRESETS if role == "chat" else EMBED_PRESETS
    preset = table.get(preset_id)
    if not preset:
        return get_config(), [f"unknown {role} preset {preset_id!r}"]
    if role == "chat":
        return save_config({"chat_provider": PROVIDER_HTTP,
                            "chat_base": preset["base"],
                            "chat_key_env": preset["key_env"],
                            **({"chat_model": preset["example_model"]}
                               if preset.get("example_model") else {})})
    return save_config({"embed_base": preset["base"],
                        "embed_model": preset["model"],
                        "embed_key_env": preset["key_env"]})


def embedding_identity(config=None):
    """The stamp `providers.embed_texts_meta` will write on new vectors.

    Exposed so the settings page can compare it against what is already in
    the bank — see `config_warnings`."""
    config = config or get_config()
    model = str(config.get("embed_model") or "")
    if not (config.get("embed_base") and model):
        return "cheap:crc32:256"
    return f"api:{model}"


def config_warnings(config=None):
    """What is not going to work, said now rather than as a confusing
    behaviour later — the persona_warnings pattern applied to configuration.
    An unset key is the single most common cause of "it just stopped
    replying", and it is invisible unless something says so."""
    config = config or get_config()
    warnings = []
    if config["chat_provider"] == PROVIDER_HTTP:
        if not config["chat_base"] or not config["chat_model"]:
            warnings.append(
                "chat is unconfigured: set a base URL and a model, or switch "
                "the provider to the Claude Code CLI")
        elif not secret_for("chat_key_env"):
            warnings.append(
                "no chat key: paste one into the chat key field, or set the "
                f"environment variable {config['chat_key_env']!r} — requests "
                "will go out unauthenticated until then")
    if config["embed_base"] and not config["embed_model"]:
        warnings.append("embeddings have a base URL but no model; retrieval "
                        "will stay on the lexical fallback")
    elif config["embed_base"] and not secret_for("embed_key_env"):
        warnings.append(
            "no embeddings key: paste one into the embeddings key field, or "
            f"set the environment variable {config['embed_key_env']!r} — "
            "embedding calls will fail with 401 and retrieval will fall back "
            "to keyword match until then")
    elif not config["embed_base"]:
        warnings.append(
            "no embeddings provider: retrieval is running on the local "
            "hashing fallback, which was measured at 0% recall on "
            "paraphrases that share no vocabulary")
    # CHANGING THE EMBEDDING MODEL STRANDS THE BANK, and it does so silently
    # unless something says this here. A vector can only be compared with one
    # from the same model, so every existing row scores 0.0 against a query
    # embedded by a different one — retrieval keeps working, on keyword match
    # alone, and looks fine. Said at the moment of the decision rather than
    # discovered later from bad answers.
    try:
        from db import q
        wanted = embedding_identity(config)
        rows = q("SELECT embedding_model AS m, COUNT(*) AS c FROM memories "
                 "WHERE embedding IS NOT NULL GROUP BY embedding_model")
        stale = sum(r["c"] for r in rows if r["m"] != wanted)
        # Summary windows carry their own vectors and are counted separately
        # because they fail HARDER: a stranded memory still reaches recall by
        # keyword, but search_memory_summaries skips a cross-model window
        # outright, so it leaves retrieval entirely.
        windows = q("SELECT embedding_model AS m, COUNT(*) AS c FROM "
                    "memory_summaries WHERE embedding IS NOT NULL AND "
                    "TRIM(summary)<>'' GROUP BY embedding_model")
        stale_windows = sum(r["c"] for r in windows if r["m"] != wanted)
        if stale:
            warnings.append(
                f"{stale} stored memories were embedded by a different model "
                f"and cannot be compared with {wanted!r}. They are reachable "
                "by keyword only until you rebuild the embeddings.")
        if stale_windows:
            warnings.append(
                f"{stale_windows} consolidated summary windows were embedded "
                f"by a different model and cannot be compared with {wanted!r}. "
                "Unlike memories these have no keyword fallback — they are out "
                "of retrieval completely until you rebuild the embeddings.")
    except Exception:
        pass
    return warnings


def redacted_status():
    """What the settings UI is allowed to see: the configuration, plus
    whether each key resolves and from where — never the key itself.

    THE STRIP IS THE POINT. `get_config` now returns stored credentials, and
    this is the only thing standing between them and the browser, so it is
    written as a whitelist-by-removal over KEY_VALUE_FIELDS rather than a list
    of fields to send: a credential field added later is redacted by default
    instead of shipped by omission."""
    config = get_config()
    safe = {field: ("" if field in KEY_VALUE_FIELDS else value)
            for field, value in config.items()}
    return {
        "config": safe,
        "secrets": {field: {"env": config[field],
                            "present": bool(secret_for(field)),
                            "source": secret_source(field)}
                    for field in KEY_FIELDS},
        "warnings": config_warnings(config),
        "providers": list(CHAT_PROVIDERS),
        "chat_presets": CHAT_PRESETS,
        "embed_presets": EMBED_PRESETS,
        "embedding_identity": embedding_identity(config),
    }
