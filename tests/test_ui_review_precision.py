# ui_review's own rule is that "each check must name a defect that bites".
# Measured against this project's OWN static/ directory, it returned 42
# findings of which 40 were false — including one naming the selector
# 'not a product surface. */ *', which is a sentence out of a CSS comment.
#
# A checker that cries wolf on ordinary markup gets switched off, and then it
# protects nothing. So these tests come in pairs: the false positive that must
# stay quiet, and the true positive next to it that must still fire.

import pathlib

import ui_review


# ---- HTML: the regex scanner was a defect factory ----

def test_a_comment_is_not_an_unclosed_tag():
    """`<!-- <div class="old"> was here -->` reported `unclosed <div>` at
    error severity. `<[^>]+>` cannot see comments."""
    assert ui_review.review_html(
        '<!-- <div class="old"> was here --><p>hi</p>') == []


def test_script_contents_are_not_markup():
    """`<div><script>s="</div>";</script></div>` reported two `stray-close`
    errors. Script is raw text, not a place to look for tags."""
    assert ui_review.review_html(
        '<div><script>s="</div>";</script></div>') == []


def test_an_unquoted_url_ending_in_a_slash_is_not_self_closing():
    """`<a href=/foo/>home</a>` reported `stray-close </a>`: the `(/?)` group
    read the URL's trailing slash as a self-closing marker."""
    assert ui_review.review_html('<a href=/foo/>home</a>') == []


def test_data_id_is_not_an_id():
    """`\\bid\\s*=` matched `data-id=` because `-` is a word boundary, so two
    list rows sharing a data-id were reported as a duplicate-id ERROR. This is
    one of the most common attribute patterns in list UIs."""
    assert ui_review.review_html(
        '<li data-id="row"></li><li data-id="row"></li>') == []


def test_a_real_duplicate_id_still_fires():
    """The check has to survive its own fix."""
    kinds = [f["kind"] for f in ui_review.review_html(
        '<div id="a"></div><div id="a"></div>')]
    assert kinds == ["duplicate-id"]


def test_a_genuinely_unclosed_element_still_fires():
    """The pop-until-match loop silently absorbed everything it skipped, so a
    phantom element inside a closed parent was reported as nothing at all."""
    kinds = [f["kind"] for f in ui_review.review_html('<div><span>a</div>')]
    assert "unclosed" in kinds


def test_an_optional_end_tag_is_not_a_missing_one():
    """`</p>` and `</li>` are optional in HTML: a parent close legitimately
    ends them, so reporting those would be reporting correct markup."""
    assert ui_review.review_html('<div><p>a</div>') == []
    assert ui_review.review_html('<ul><li>a<li>b</ul>') == []


def test_a_button_named_by_alt_text_is_named():
    """`<button><img alt="Save"></button>` was flagged unnamed though the alt
    text names it perfectly well."""
    assert ui_review.review_html('<button><img alt="Save"></button>') == []


def test_a_button_named_in_a_non_latin_script_is_named():
    """`wordy` was `[A-Za-z]{2,}`, so `<button>保存</button>` was flagged — a
    correctly named button in every language not using the Latin alphabet."""
    assert ui_review.review_html('<button>保存</button>') == []


def test_an_empty_aria_label_is_not_a_name():
    """Presence is not a name: `aria-label=""` passed as named because the
    check tested for the attribute rather than for its value."""
    kinds = [f["kind"] for f in ui_review.review_html(
        '<button aria-label="">✕</button>')]
    assert "unnamed-control" in kinds


# ---- CSS: 40 of the 42 findings came from here ----

def test_css_comments_are_not_selectors():
    """The rule regex read prose out of a `/* */` block as a selector and
    reported the specificity of a sentence."""
    findings = ui_review.review_css(
        "/* this is not a product surface. */ * { color: red }")
    assert not [f for f in findings if "not a product" in f["detail"]]


def test_rules_that_cannot_match_the_same_element_do_not_collide():
    """The check compared every pair of rules touching one property with no
    test that they could ever select the same element, so `#header .logo` and
    `.footer-note` were reported as a cascade conflict."""
    assert ui_review.review_css(
        "#header .logo{display:block}\n.footer-note{display:none}") == []


def test_important_wins_regardless_of_specificity():
    """`!important` was ignored entirely, so a rule that actually wins the
    cascade was reported as losing it."""
    assert ui_review.review_css(
        "#panel .row{color:red}\n.row{color:blue !important}") == []


def test_a_hash_inside_an_attribute_value_is_not_an_id():
    """Specificity counted CHARACTERS, so `a[href="#top"]` counted as
    carrying an id selector."""
    assert ui_review.review_css(
        'a[href="#top"]{color:red}\n#nav a{color:blue}') == []


def test_a_genuine_specificity_loss_still_fires():
    """Same subject, later rule weaker: the edit reads as taking effect and
    does not. This is the defect the check exists for."""
    kinds = [f["kind"] for f in ui_review.review_css(
        "#nav a{color:red}\na{color:blue}")]
    assert "specificity-loss" in kinds


# ---- JS ----

def test_jquery_tag_selectors_are_not_id_lookups():
    """`$("div").hide(); $("form").submit();` yielded two ERROR findings for
    missing #div and #form — jQuery TAG selectors read as ids."""
    assert ui_review.review_js('$("div").hide(); $("form").submit();',
                               '<div id="here"></div>') == []


def test_an_id_built_at_runtime_is_not_a_literal_lookup():
    """`getElementById('panel-' + btn.dataset.panel)` — this project's own
    app.js — was reported as looking up the literal id "panel-"."""
    assert ui_review.review_js(
        "document.getElementById('panel-' + x)",
        '<div id="panel-a"></div>') == []


def test_a_real_missing_node_still_fires():
    kinds = [f["kind"] for f in ui_review.review_js(
        "document.getElementById('nope')", '<div id="here"></div>')]
    assert "missing-node" in kinds


def test_a_template_literal_innerhtml_is_caught():
    """`el.innerHTML = `<b>${name}</b>`` is the modern shape of exactly this
    defect, and the concatenation-only pattern never saw it."""
    kinds = [f["kind"] for f in ui_review.review_js(
        "el.innerHTML = `<b>${name}</b>`")]
    assert "innerhtml-concat" in kinds


# ---- the whole thing, against this repository ----

def test_reviewing_this_projects_own_frontend_is_quiet():
    """The measurement that started all of this: 42 findings on static/, 40 of
    them false, including 40 specificity-loss on a 48-line stylesheet. Against
    that noise a true positive is unfindable. This test is the standing
    version of that measurement — if a future check starts flagging correct
    code, it fails here first."""
    root = pathlib.Path(__file__).resolve().parent.parent / "static"
    html = (root / "index.html").read_text()
    css = (root / "style.css").read_text()
    js = "\n".join(p.read_text() for p in sorted((root / "js").glob("*.js")))
    findings = (ui_review.review_html(html) + ui_review.review_css(css)
                + ui_review.review_js(js, html))
    errors = [f for f in findings if f["severity"] == "error"]
    assert errors == [], errors
    assert len(findings) <= 3, findings
