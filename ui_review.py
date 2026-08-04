# ui_review.py — making sense of interface code without eyes.
#
# THE PROBLEM. An assistant asked to change a UI cannot see the UI. The
# tempting answer is to render a screenshot and look, which needs a browser, a
# renderer and a vision model, and still answers the wrong question: most UI
# defects are not visible in a screenshot of the happy path. They are a
# duplicate id that makes `getElementById` return the wrong node, a selector
# whose specificity quietly loses to one three files away, a focus state that
# only a keyboard user would ever discover, a handler bound to an element that
# no longer exists.
#
# THE APPROACH, taken from how Sonder's own frontend is maintained. It has no
# bundler and no browser tests; what it has is `tests/test_frontend_state_
# guards.py`, which makes ASSERTIONS ABOUT THE SOURCE — that a sequence guard
# is present, that a fetch result is dropped when the navigation moved on,
# that arrow keys yield to a text field. Those checks caught real defects that
# no screenshot would have shown, and they run in milliseconds.
#
# So the model here is: UI is understood as a set of checkable structural
# claims, and the assistant's job is to turn "does this look right" into
# questions that have answers. This module supplies the deterministic half —
# the findings a machine can be certain of — so the model spends its judgement
# on the part that genuinely needs judgement.
#
# It is not a linter and not a validator. Every check earns its place by
# corresponding to a defect that actually bites, and each one says what breaks
# rather than which rule was violated.

import re
from collections import Counter
from html.parser import HTMLParser

_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                   "link", "meta", "param", "source", "track", "wbr"})

_INTERACTIVE = frozenset({"button", "a", "input", "select", "textarea"})

# Elements whose end tag is OPTIONAL in HTML: a parent close legitimately
# ends them, so an implicit close is correct markup, not a missing tag. Every
# other element popped by an ancestor's close really was left open -- and the
# pop-until-match loop used to swallow all of them silently, so a phantom
# <span> inside a <div> was reported as nothing at all.
_OPTIONAL_END = frozenset({
    "p", "li", "dt", "dd", "option", "optgroup", "thead", "tbody", "tfoot",
    "tr", "td", "th", "rt", "rp", "colgroup", "caption",
})


def _finding(kind, detail, why, severity="warn"):
    return {"kind": kind, "detail": detail, "why": why, "severity": severity}


class _Scanner(HTMLParser):
    """A real parser, because a regex scanner is a defect factory here.

    Every one of these was an `error`-severity finding on CORRECT markup:

      <!-- <div class="old"> was here --><p>hi</p>   -> "unclosed <div>"
      <div><script>s="</div>";</script></div>        -> two "stray-close"
      <a href=/foo/>home</a>                         -> "stray-close </a>"
      <li data-id="row"></li><li data-id="row"></li> -> "duplicate-id 'row'"

    The last is the worst: `\\bid\\s*=` matches `data-id=` because `-` is a
    word boundary, and `data-id` on list rows is one of the most common
    patterns in UI code. A checker that cries wolf on ordinary markup gets
    switched off, and then it protects nothing.

    html.parser handles comments, CDATA/script raw text, attribute quoting
    and void elements, and hands back a real attribute dict — which kills the
    `data-id` bug for free rather than by another regex."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.events = []          # ("start"|"end", name, attrs dict)

    def handle_starttag(self, tag, attrs):
        self.events.append(("start", tag.lower(), dict(attrs)))

    def handle_startendtag(self, tag, attrs):
        self.events.append(("void", tag.lower(), dict(attrs)))

    def handle_endtag(self, tag):
        self.events.append(("end", tag.lower(), {}))


def _scan(html):
    scanner = _Scanner()
    try:
        scanner.feed(html or "")
        scanner.close()
    except Exception:
        pass
    return scanner.events


def review_html(html):
    """Structural claims about one HTML document that a machine can settle."""
    out = []
    events = _scan(html)

    # DUPLICATE IDS. `getElementById` returns the FIRST, so the second element
    # is unreachable and every handler bound by id silently drives the wrong
    # node. Invisible in a screenshot; instant here.
    ids = [attrs["id"] for _kind, _name, attrs in events
           if attrs.get("id")]
    for value, count in Counter(ids).items():
        if count > 1:
            out.append(_finding(
                "duplicate-id", f"id={value!r} appears {count} times",
                "getElementById returns the first match, so every later "
                "element with this id is unreachable and any handler bound "
                "by it drives the wrong node", "error"))

    # UNCLOSED ELEMENTS. A missing </div> does not error; it reparents
    # everything after it, which is why the symptom is usually "the layout
    # broke somewhere else entirely".
    stack = []
    for kind, name, _attrs in events:
        if name in _VOID or kind == "void":
            continue
        if kind == "end":
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                # Everything skipped over on the way down was left open.
                while stack:
                    popped = stack.pop()
                    if popped == name:
                        break
                    if popped not in _OPTIONAL_END:
                        out.append(_finding(
                            "unclosed", f"<{popped}> never closed",
                            f"</{name}> closed it implicitly, so the tree you "
                            "get is not the tree you wrote and the symptom "
                            "appears somewhere else entirely", "error"))
            else:
                out.append(_finding(
                    "stray-close", f"</{name}> with nothing open",
                    "the parser discards it, and the tree you get is not the "
                    "tree you wrote", "error"))
        else:
            stack.append(name)
    for name in stack:
        out.append(_finding(
            "unclosed", f"<{name}> never closed",
            "everything after it is reparented, so the visible symptom "
            "usually appears somewhere else entirely", "error"))

    # ACCESSIBLE NAMES. An icon button whose only content is a glyph is a
    # button with no name: a screen reader announces "button", and so does
    # every automated check that later tries to find it.
    for match in re.finditer(r"<button\b([^>]*)>(.*?)</button>", html or "",
                             re.S | re.I):
        raw_attrs, inner_markup = match.group(1), match.group(2)
        inner = re.sub(r"<[^>]+>", "", inner_markup)
        # A PRESENT attribute is not a NAME: `aria-label=""` announced as
        # named and is not. And an <img alt="Save"> inside the button names
        # it perfectly well — that was flagged too.
        named = bool(
            re.search(r'aria-(?:label|labelledby)\s*=\s*(["\'])\s*(?!\1)\S',
                      raw_attrs, re.I)
            or re.search(r'\balt\s*=\s*(["\'])\s*(?!\1)\S',
                         inner_markup, re.I))
        # Any two adjacent letters in ANY script. `[A-Za-z]{2,}` flagged
        # <button>保存</button> — a correctly named button in every language
        # that does not use the Latin alphabet.
        wordy = bool(re.search(r"\w{2,}", inner, re.UNICODE)
                     and re.search(r"[^\W\d_]{2,}", inner, re.UNICODE))
        if not named and not wordy:
            out.append(_finding(
                "unnamed-control", f"<button>{inner.strip()[:20]}</button>",
                "its only content is a glyph, so it has no accessible name — "
                "a screen reader announces 'button' and nothing else"))

    # A HANDLER ON SOMETHING UNFOCUSABLE. A click handler on a div is
    # unreachable by keyboard, which is a whole class of user locked out.
    for match in re.finditer(r"<(\w+)\b([^>]*\bonclick\b[^>]*)>", html or "",
                             re.I):
        tag, attrs = match.group(1).lower(), match.group(2)
        if tag in _INTERACTIVE:
            continue
        if "tabindex" in attrs.lower() and "role" in attrs.lower():
            continue
        out.append(_finding(
            "keyboard-unreachable", f"<{tag}> carries onclick",
            "a non-interactive element takes no keyboard focus, so this "
            "action does not exist for anyone not using a mouse"))
    return out


def _key_subject(selector):
    """The right-most compound of a selector — the element it actually
    styles. Two rules can only fight over one element if they agree here."""
    last = re.split(r"[\s>+~]+", selector.strip())[-1]
    return re.sub(r"::?[\w-]+(\([^)]*\))?$", "", last) or last


def review_css(css):
    """Cascade claims. Specificity collisions are the defect that most often
    produces 'I changed it and nothing happened'."""
    out = []
    # Comments FIRST. Without this the rule regex read prose out of a `/* */`
    # block as a selector — against this project's own style.css it reported
    # the selector `'not a product surface. */ *'`, which is a sentence.
    css = re.sub(r"/\*.*?\*/", " ", css or "", flags=re.S)
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    by_property = {}
    for selector_group, body in rules:
        for selector in selector_group.split(","):
            selector = " ".join(selector.split())
            if not selector or selector.startswith("@"):
                continue
            # Counting CHARACTERS was wrong in both directions: `#` inside
            # an attribute value (`a[href="#top"]`) counted as an id, and a
            # `:` inside `url(...)` or a pseudo-element counted as a class.
            # Strip strings first, then count real components.
            bare = re.sub(r'"[^"]*"|\'[^\']*\'', "", selector)
            spec = (len(re.findall(r"#[\w-]+", bare)),
                    len(re.findall(r"\.[\w-]+", bare))
                    + len(re.findall(r"\[[^\]]*\]", bare))
                    + len(re.findall(r"(?<!:):[\w-]+", bare)),
                    len(re.findall(r"(?:^|[\s>+~])([a-zA-Z][\w-]*)", bare)))
            for declaration in body.split(";"):
                if ":" not in declaration:
                    continue
                prop = declaration.split(":", 1)[0].strip().lower()
                if not prop:
                    continue
                # `!important` wins regardless of specificity, so a rule
                # carrying it is not "losing the cascade" and reporting it as
                # such is simply false.
                if "!important" in declaration.lower():
                    continue
                by_property.setdefault(prop, []).append((selector, spec, body))

    for prop, entries in by_property.items():
        if len(entries) < 2:
            continue
        # Two rules setting the same property where the LATER one is weaker:
        # the edit reads as taking effect and does not.
        for i in range(1, len(entries)):
            later_sel, later_spec, _ = entries[i]
            for earlier_sel, earlier_spec, _ in entries[:i]:
                # Only when the two selectors could actually match the SAME
                # element. Without this the check compared every pair of
                # rules touching a property regardless of what they select,
                # so `#header .logo` and `.footer-note` were reported as a
                # cascade conflict. Measured against this project's own
                # static/: 40 findings, every one of them false. A check that
                # cannot be right is noise, and noise buries the one finding
                # that would have mattered. Same right-most compound is a
                # conservative proxy for overlap: it under-reports rather
                # than inventing.
                if _key_subject(earlier_sel) != _key_subject(later_sel):
                    continue
                if earlier_spec > later_spec and earlier_sel != later_sel:
                    out.append(_finding(
                        "specificity-loss",
                        f"{later_sel!r} sets {prop} but {earlier_sel!r} is "
                        f"more specific",
                        "the later rule loses the cascade, so editing it "
                        "changes nothing and the change looks like it was "
                        "not applied"))
                    break

    if re.search(r":focus\s*\{[^}]*outline\s*:\s*(none|0)\s*[;}]", css,
                 re.I):
        # A replacement is any focus rule declaring a VISIBLE outline or
        # box-shadow. Matching on the whole value rather than its first
        # character, because `box-shadow: 0 0 0 2px blue` is a perfectly
        # visible ring that begins with a zero -- the first version of this
        # check rejected it and reported a defect that had been fixed.
        replaced = False
        for match in re.finditer(
                r":focus(?:-visible)?\s*\{([^}]*)\}", css or "", re.I):
            for declaration in match.group(1).split(";"):
                if ":" not in declaration:
                    continue
                prop, value = declaration.split(":", 1)
                if prop.strip().lower() not in ("outline", "box-shadow"):
                    continue
                if value.strip().lower() not in ("none", "0", ""):
                    replaced = True
        if not replaced:
            out.append(_finding(
                "focus-removed", "outline:none on :focus with no replacement",
                "keyboard users lose all indication of where they are, which "
                "no screenshot of the page will ever show", "error"))
    return out


def review_js(js, html=""):
    """Claims about the seam between script and document — where the two get
    out of step is where the runtime error lives."""
    out = []
    ids_present = {attrs["id"] for _k, _n, attrs in _scan(html)
                   if attrs.get("id")}
    if html:
        # `getElementById("x")` ONLY, with the closing paren required so the
        # literal is the whole argument. The old pattern also accepted `$(`,
        # so `$("div").hide()` and `$("form").submit()` were reported as two
        # `error` findings for missing #div and #form -- jQuery TAG selectors
        # read as ids. And without the closing paren,
        # `getElementById('panel-' + btn.dataset.panel)` (this project's own
        # app.js) was reported as a lookup of the literal id "panel-".
        for match in re.finditer(
                r'getElementById\(\s*["\']([\w-]+)["\']\s*\)', js or ""):
            name = match.group(1)
            if name not in ids_present:
                out.append(_finding(
                    "missing-node", f"script looks up #{name}",
                    "no element in the document has that id, so this is null "
                    "at runtime and the next property access throws", "error"))

    added = len(re.findall(r"addEventListener\(", js or ""))
    removed = len(re.findall(r"removeEventListener\(", js or ""))
    if added and not removed and added >= 3:
        out.append(_finding(
            "listener-accumulation",
            f"{added} addEventListener calls, no removeEventListener",
            "if any of this runs more than once — a re-render, a reopened "
            "panel — handlers stack and each event fires N times"))

    # Template literals too: `el.innerHTML = \`<b>${name}</b>\`` is the
    # modern shape of exactly this defect and the concatenation-only pattern
    # never saw it.
    if (re.search(r"\.innerHTML\s*=\s*[^;]*['\"][^;]*\+", js or "")
            or re.search(r"\.innerHTML\s*=\s*`[^`]*\$\{", js or "")):
        out.append(_finding(
            "innerhtml-concat", "innerHTML assigned from concatenation",
            "any interpolated value is parsed as markup, so a stray angle "
            "bracket in ordinary text silently restructures the page",
            "error"))
    return out


def review(files):
    """{path: contents} -> findings, most severe first.

    Routed by extension, so the caller hands over whatever it has.
    """
    html = "\n".join(v for k, v in files.items()
                     if k.endswith((".html", ".htm")))
    findings = []
    for path, contents in (files or {}).items():
        if path.endswith((".html", ".htm")):
            found = review_html(contents)
        elif path.endswith(".css"):
            found = review_css(contents)
        elif path.endswith((".js", ".mjs")):
            found = review_js(contents, html)
        else:
            continue
        findings += [{**f, "file": path} for f in found]
    order = {"error": 0, "warn": 1}
    return sorted(findings, key=lambda f: order.get(f["severity"], 2))
