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


def search(query, max_results=5):
    """Web search: [{title, url, snippet}]. Empty list on any failure — the
    research loop treats "search returned nothing" as a finding to route
    around, never as an exception that kills the turn."""
    if _search_stub is not None:
        return list(_search_stub(query, max_results) or [])
    try:
        url = ("https://html.duckduckgo.com/html/?q="
               + urllib.parse.quote_plus(str(query or "")))
        req = urllib.request.Request(url, headers=_UA)
        with _opener.open(req, timeout=_TIMEOUT) as r:
            page, err = _read_text(r)
        if err:
            return []
    except Exception:
        return []
    out = []
    # ONE PASS OVER WHOLE RESULT BLOCKS, not two independent passes zipped by
    # position. Snippets used to be collected separately and paired by index,
    # so a single result rendered without a snippet shifted every later
    # snippet onto the wrong url — proved on a crafted page: result 1 was
    # handed result 2's text. The model judges `stance` from the snippet in
    # the same round it fetches, so a shifted snippet means a stance judged
    # about a page other than the one cited. Cutting the result block at the
    # next result anchor keeps title, href and snippet provably together.
    blocks = re.split(r'(?=<a[^>]+class="result__a")', page)
    for block in blocks:
        m = re.search(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>'
                      r'(.*?)</a>', block, re.S)
        if not m:
            continue
        href, title = m.group(1), _strip_html(m.group(2))
        # DDG wraps result urls in a redirect; unwrap to cite the REAL url —
        # an evidence row pointing at a redirector is not a citation.
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        real = qs.get("uddg", [href])[0]
        sm = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                       block, re.S)
        out.append({"title": title[:200], "url": real,
                    "snippet": _strip_html(sm.group(1))[:400] if sm else ""})
        if len(out) >= max_results:
            break
    return out


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
