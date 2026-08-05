# tools_web.py — web search and page fetch, kept deliberately small.
#
# Stdlib only (urllib + regex HTML stripping): the point of this module is
# not crawling prowess, it is producing EVIDENCE rows that cite real URLs.
# Both entry points take test stubs, for the same reason providers.py does —
# everything outside the network call must be deterministic and provable
# offline, and a research test that needs the live web is a test that fails
# on the train.
#
# The default search backend is DuckDuckGo's HTML endpoint because it needs
# no API key; it is a best-effort scrape and is expected to rot eventually.
# When it does, set_search_backend swaps it without touching the research
# machinery — the contract is just [(title, url, snippet)].

import html
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

_UA = {"User-Agent": "SonderAssistant/0.1 (research; local single-user)"}
_TIMEOUT = 20
# Search bodies were read unbounded while fetch capped at 1.5 MB. A hostile
# or merely broken endpoint could exhaust memory through the half that had no
# limit; there is no reason for the two to differ.
_MAX_BYTES = 1_500_000


# ---- Where the fetcher is allowed to point ----
#
# The url reaching `fetch` is chosen by a model reasoning over search results
# it did not write — untrusted input by this project's own doctrine — and the
# body it returns becomes an evidence row AND a durable `read` memory with the
# url attached. With only a `^https?://` check, `http://127.0.0.1:8010/` and
# `http://169.254.169.254/latest/meta-data/` were both fetchable, and because
# urllib follows redirects by default a perfectly ordinary public page could
# 302 the fetcher onto either. That is a confused deputy: the assistant's own
# network position spent on somebody else's instruction, with the result
# written into long-term memory.
#
# Resolve first, check every address the name resolves to, and re-check on
# every redirect hop — a name that answers publicly once and privately the
# second time is the standard way this check gets walked around.

def _address_is_public(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False, f"cannot resolve host {host!r}"
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False, f"unparseable address for {host!r}"
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast
                or addr.is_unspecified):
            return False, (f"refusing to fetch {host!r}: resolves to the "
                           f"non-public address {addr}")
    return True, ""


def _check_fetchable(url):
    """(ok, error). Scheme, then every address the host resolves to."""
    if not re.match(r"^https?://", str(url or "")):
        return False, "not an http(s) url"
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        return False, "url has no host"
    return _address_is_public(host)


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-run the address check on each hop. A public page redirecting into
    the internal network is the interesting case, not the exotic one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        ok, why = _check_fetchable(newurl)
        if not ok:
            raise urllib.error.URLError(f"blocked redirect: {why}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_GuardedRedirects)


def _read_text(response, limit=_MAX_BYTES):
    """Decode a response body using the charset it declares.

    Hardcoding utf-8 turned an ISO-8859-1 page into 'caf trs dsol' — every
    accented character silently deleted, then quoted as an excerpt — and let
    an `application/pdf` body through as '%PDF-1.4 <binary>' with no error at
    all, ready to become evidence."""
    ctype = (response.headers.get_content_type() or "").lower()
    if ctype and not (ctype.startswith("text/")
                      or ctype in ("application/xhtml+xml",
                                   "application/xml", "application/json")):
        return None, f"not a text document ({ctype})"
    charset = response.headers.get_content_charset() or "utf-8"
    try:
        return response.read(limit).decode(charset, "replace"), ""
    except LookupError:
        return response.read(limit).decode("utf-8", "replace"), ""

_search_stub = None
_fetch_stub = None


def set_search_stub(fn):
    """fn(query, max_results) -> [{title, url, snippet}] | None to clear."""
    global _search_stub
    _search_stub = fn


def set_fetch_stub(fn):
    """fn(url) -> {url, title, text} | None to clear."""
    global _fetch_stub
    _fetch_stub = fn


def _strip_html(markup):
    # Comments go FIRST. `<[^>]+>` cannot see them, so a comment's contents
    # survived into "readable text" — measured leaking an internal note
    # ("revenue > forecast, DO NOT SHIP") into an excerpt. The content least
    # likely to be a quotable source was the content most likely to reach a
    # citation.
    markup = re.sub(r"(?s)<!--.*?-->", " ", markup)
    markup = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)"
                    r"[^>]*>.*?</\1>", " ", markup)
    markup = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n",
                    markup)
    # Quoted attribute values are skipped rather than scanned for `>`, so
    # `<a title="a > b">` no longer ends the tag early and spill the rest of
    # the markup into the text.
    text = re.sub(r"(?s)<[^>\"']*(?:\"[^\"]*\"|'[^']*'[^>\"']*)*[^>]*>",
                  " ", markup)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# Mojeek, replacing DuckDuckGo's HTML endpoint. The module predicted its own
# rot at the top of this file and the prediction came true: DDG now answers
# that endpoint with HTTP 202 and an anti-bot challenge page ("Please complete
# the following challenge to confirm this search was made by a human"), which
# carries no result markup at all. Every search returned zero, including a
# bare "BBC News" control, and the lane had been dead for an unknown number of
# turns while reporting only "search returned nothing".
#
# Mojeek is an independent index that serves this request to the honest
# user-agent above — no key, no browser spoofing, and explicit <!--rs-->
# result delimiters, which give the title/url/snippet grouping this parser
# used to have to reconstruct by splitting on anchors.
_SEARCH_BASE = "https://www.mojeek.com/search?q="

# A backend that has started refusing us does not say so in a status code —
# DDG returned 202, which is a success. The cues are what distinguishes
# "blocked" from "this query genuinely has no results", and those two need
# opposite responses: fix the lane, versus ask a different question.
_BLOCK_CUES = ("captcha", "unusual traffic", "are you a robot",
               "confirm this search was made by a human",
               "complete the following challenge")


def _parse_results(page, max_results):
    """Result blocks → [{title, url, snippet}].

    ONE PASS OVER WHOLE RESULT BLOCKS, not two independent passes zipped by
    position. Snippets used to be collected separately and paired by index, so
    a single result rendered without a snippet shifted every later snippet
    onto the wrong url — proved on a crafted page: result 1 was handed result
    2's text. The model judges `stance` from the snippet in the same round it
    fetches, so a shifted snippet means a stance judged about a page other
    than the one cited."""
    out = []
    for block in page.split("<!--rs-->")[1:]:
        block = block.split("<!--re-->")[0]
        m = re.search(r'<a[^>]+class="title"[^>]*href="([^"]+)"[^>]*>'
                      r'(.*?)</a>', block, re.S)
        if not m:
            continue
        sm = re.search(r'<p class="s">(.*?)</p>', block, re.S)
        out.append({"title": _strip_html(m.group(2))[:200],
                    "url": m.group(1),
                    "snippet": _strip_html(sm.group(1))[:400] if sm else ""})
        if len(out) >= max_results:
            break
    return out


_search_backend = None


def set_search_backend(fn):
    """Swap the search backend. `fn(query, max_results)` returns either a list
    of {title, url, snippet} or a full {results, status, detail} dict.

    THE ESCAPE HATCH THIS MODULE HAS PROMISED SINCE IT WAS WRITTEN. The
    comment at the top says "when it does [rot], set_search_backend swaps it
    without touching the research machinery" — and no such function existed.
    The one documented recovery path for the failure the module correctly
    predicted was never built, so when the rot arrived the only options were
    editing this file or doing without a web lane.

    Distinct from `set_search_stub`, which exists so tests never touch the
    network. This is for production: a keyed provider, a different index, a
    local cache. Nothing above it needs to know which."""
    global _search_backend
    _search_backend = fn


def search_detail(query, max_results=5):
    """Web search, WITH THE REASON IT RETURNED NOTHING.

    `search` collapses every failure to an empty list, so a backend serving a
    CAPTCHA, a transport error, and a query that genuinely has no results are
    one indistinguishable outcome. That cost real money: the search lane died
    when DuckDuckGo started challenging the endpoint, and the only signal
    anywhere was "search returned nothing" — so the assistant spent a deep
    research run, two of its own deliberation rounds and a subagent budget
    establishing by control query what this function already knew.

    Returns {results, status, detail} where status is one of ok, empty,
    blocked, error. The list stays the contract for `search`; this is the
    channel the caller needs to tell a dead lane from a quiet one."""
    if _search_stub is not None:
        rows = list(_search_stub(query, max_results) or [])
        return {"results": rows, "status": "ok" if rows else "empty",
                "detail": ""}
    if _search_backend is not None:
        try:
            got = _search_backend(query, max_results)
        except Exception as exc:
            return {"results": [], "status": "error",
                    "detail": f"search backend raised "
                              f"{type(exc).__name__}: {str(exc)[:200]}"}
        if isinstance(got, dict):
            rows = list(got.get("results") or [])
            return {"results": rows,
                    "status": str(got.get("status")
                                  or ("ok" if rows else "empty")),
                    "detail": str(got.get("detail") or "")}
        rows = list(got or [])
        return {"results": rows, "status": "ok" if rows else "empty",
                "detail": ""}
    try:
        url = _SEARCH_BASE + urllib.parse.quote_plus(str(query or ""))
        req = urllib.request.Request(url, headers=_UA)
        with _opener.open(req, timeout=_TIMEOUT) as r:
            page, err = _read_text(r)
        if err:
            return {"results": [], "status": "error", "detail": err}
    except Exception as exc:
        return {"results": [], "status": "error",
                "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}
    results = _parse_results(page, max_results)
    if not results and any(c in page.lower() for c in _BLOCK_CUES):
        return {"results": [], "status": "blocked",
                "detail": "the search backend served an anti-bot challenge "
                          "instead of results; the lane is down, not the "
                          "query"}
    return {"results": results, "status": "ok" if results else "empty",
            "detail": ""}


def search(query, max_results=5):
    """Web search: [{title, url, snippet}]. Empty list on any failure — the
    research loop treats "search returned nothing" as a finding to route
    around, never as an exception that kills the turn. Callers that need to
    know WHY it was empty want `search_detail`."""
    return search_detail(query, max_results)["results"]


def fetch(url, max_chars=8000):
    """Fetch one page as readable text: {url, title, text}. `text` is
    truncated — evidence is an excerpt with a citation, not an archive.
    Returns an `error` key instead of raising, same reasoning as search."""
    if _fetch_stub is not None:
        return dict(_fetch_stub(url) or {"url": url, "title": "",
                                         "text": "", "error": "stub miss"})
    ok, why = _check_fetchable(url)
    if not ok:
        return {"url": url, "title": "", "text": "", "error": why}
    try:
        req = urllib.request.Request(url, headers=_UA)
        with _opener.open(req, timeout=_TIMEOUT) as r:
            page, err = _read_text(r)
        if err:
            return {"url": url, "title": "", "text": "", "error": err}
    except Exception as exc:
        return {"url": url, "title": "", "text": "", "error": str(exc)[:200]}
    title = ""
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    if m:
        title = _strip_html(m.group(1))[:200]
    return {"url": url, "title": title, "text": _strip_html(page)[:max_chars]}
