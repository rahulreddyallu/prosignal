"""The interface and the payload it binds to must agree.

This file exists because the old table rendered `undefined` in its rank column
for a whole release. The column was real, the value was computed, and the API
simply never serialised it -- nothing in the test suite compared the two sides,
so nothing failed. The same class of gap is now cheaper to hit, not more
expensive: the interface reads a nested view model rather than a flat row.

So the contract is asserted from both ends. Every field the frontend reads off
a pick must be produced by build_view, and the states a user can actually land
in -- empty store, no scan yet, nothing qualified, a failed run -- must each
have real markup behind them. Every one of those has been a blank screen in
this project at some point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from prosignal.presentation.viewmodel import build_view

UI = Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static" / "index.html"


def _html() -> str:
    return UI.read_text(encoding="utf-8")


def _sample_view() -> dict:
    card = {
        "ticker": "RELIANCE", "model_rank": 1, "rank": 1, "score": 0.9,
        "percentile": 100.0, "last_close": 1482.2, "strength": "High",
        "decision": "BUY CANDIDATE", "sector": "Energy",
        "entry_zone": [1470.0, 1495.0], "stop": 1380.0, "invalidation": 1300.0,
        "target_1": 1620.0, "target_2": 1750.0,
        "holding_period": "15-63 sessions", "cost_note": "Round-trip cost 88 bps.",
        "why": ["resid_mom: +1.55 sd"], "against": ["low_volume_breakout: x (-0.10)"],
        "cleared": ["gap_signal"], "not_testable": ["promoter_pledging"],
        "exits": ["1. thesis_invalidation (Rs 1,300.00) -- gone"],
        "research_basis": ["Jegadeesh & Titman (1993)"], "warning": "Unvalidated.",
        "factors": {
            "resid_mom": {"standardised": 1.5, "weight": 0.02, "raw": 0.1, "available": True},
            "deliv_pct": {"standardised": 0.9, "weight": 0.03, "raw": 0.5, "available": True},
            "downside_vol": {"standardised": -1.2, "weight": 0.01, "raw": 0.2, "available": True},
        },
    }
    return build_view(
        {"recommendations": [card], "watchlist": [],
         "as_of_date": "2026-08-21", "generated_at": "2026-08-21T16:02:00",
         "regime": {"compatibility": "Favorable", "trend": "Uptrend",
                    "breadth_state": "Strong", "breadth_pct": 65.8,
                    "volatility": "Low/stable", "allow_new_entries": True},
         "funnel": {"universe_considered": 750, "buys": 1},
         "disclaimer": "Not advice.", "probability_note": "Not a probability."},
        company_names={"RELIANCE": "Reliance Industries Limited"},
    )


#: Every `p.<field>` the interface reads off a pick. Extracted from the source
#: rather than listed by hand, so a field added to the interface without a
#: backing value fails here instead of rendering as "undefined".
_PICK_FIELDS = {
    "position", "ticker", "company", "sector", "status", "price", "strength",
    "rank", "confirmation", "highlights", "thesis", "what_would_change",
    "evidence", "levels", "holding_period", "cost_note", "technical",
}


def test_every_field_the_interface_reads_is_produced_by_the_view():
    view = _sample_view()
    pick = view["picks"][0]
    missing = sorted(f for f in _PICK_FIELDS if f not in pick)
    assert not missing, f"the interface reads fields the view never sets: {missing}"


def test_the_interface_reads_no_pick_field_the_view_does_not_set():
    """The other direction. `p.foo` in the source with no `foo` in the payload
    is exactly the bug that shipped `undefined` to users."""
    html = _html()
    body = html[html.index("function panelHTML"):]
    read = set(re.findall(r"\bp\.([a-z_][a-z0-9_]*)\b", body))
    known = set(_sample_view()["picks"][0].keys())
    unknown = sorted(read - known)
    assert not unknown, f"interface reads unknown pick fields: {unknown}"


def test_the_view_is_json_serialisable():
    """It crosses the wire. A dataclass that survived the conversion would
    raise inside the response rather than in a test."""
    json.dumps(_sample_view())


@pytest.mark.parametrize("state", [
    "Building the market data",       # store short of what the model needs
    "No scan has run yet",            # ingested, never run
    "Nothing met the bar today",      # the designed common outcome
    "No recommendation was produced", # withheld on bad data
    "could not be completed",         # a failed run must not imply a trade
    "Checks that could not run",      # NOT_TESTABLE is not a pass
    "What would move this to Buy",    # the near misses stay actionable
    # "No results yet" was the empty History page. It is now "The record opens
    # here", because the page is scoped to the running configuration and the
    # honest statement is not "there are no results" -- there are 128 of them --
    # but "none of them were decided by this engine".
    "The record opens here",
    "superseded configuration",       # and what is being left out, in one line
    "New market data",       # store moved on, results did not
    "What this configuration has done",  # history is scoped, and says so
    # "Not followed yet" belonged to `outcomeRow`, which was removed: nothing
    # called it and every verdict it could render ("Target reached", "Stop
    # touched") described an exit this engine has disarmed.
    "Clear the run history?",         # destructive action is confirmed
])
def test_every_reachable_state_has_markup(state):
    assert state in _html(), f"no markup for the {state!r} state"


@pytest.mark.parametrize("term", [
    "composite_score", "model_rank", "resid_mom", "deliv_pct", "prox_52w",
    "signal_generator", "analyse_run", "stage6", "pipeline",
])
def test_no_engine_vocabulary_is_rendered_as_a_label(term):
    """Engine names may appear in the advanced drawer's raw block, which is
    fed from the payload at runtime. None may be authored into the markup as
    something a reader sees."""
    html = _html()
    body = html[html.index("<body>"):]
    # Strip the JS field accessors -- `p.technical` is code, not a label.
    labels = re.findall(r">([^<>{}]{3,80})<", body)
    for label in labels:
        assert term not in label, f"engine term {term!r} rendered as: {label.strip()!r}"


def test_the_score_is_not_presented_as_a_probability():
    html = _html().lower()
    for phrase in ("probability of", "% chance", "likelihood of profit",
                   "confidence:", "win rate"):
        assert phrase not in html


def test_the_interface_makes_no_third_party_request():
    """A CDN font or script would put a network dependency between the user and
    their own analysis, and leak that they are running it.

    An <a href> is not that. It is navigation the reader chooses, it fetches
    nothing on load, and the footer's LinkedIn link is the whole point of the
    attribution. The test targets what the PAGE fetches -- src, stylesheet
    links, @import, url() -- not where it can send someone.
    """
    html = _html()
    fetching = re.findall(
        r'(?:src\s*=\s*["\']|<link[^>]+href\s*=\s*["\']|@import\s+["\']|url\()\s*([^"\')\s]+)',
        html, re.I)
    external = [u for u in fetching
                if u.startswith(("http://", "https://", "//"))
                and "localhost" not in u]
    assert not external, f"the page fetches from off-host: {external[:3]}"


def test_every_external_link_is_a_deliberate_navigation():
    """Only links a reader clicks may leave the origin, and each must open
    safely -- an unguarded target=_blank hands the opener to the destination."""
    html = _html()
    for url, attrs in re.findall(r'<a\s+href="(https?://[^"]+)"([^>]*)>', html):
        assert "linkedin.com/in/rahulreddyallu" in url, f"unexpected link: {url}"
        assert 'target="_blank"' in attrs
        assert "noopener" in attrs and "noreferrer" in attrs


def test_both_themes_define_every_colour_token():
    """A token defined only inside a media query renders one theme's text on
    the other theme's ground."""
    html = _html()
    root = html[html.index(":root {"):html.index("@media (prefers-color-scheme: dark)")]
    dark = html[html.index(':root[data-theme="dark"]'):]
    dark = dark[:dark.index("}")]
    base = set(re.findall(r"(--[a-z0-9-]+):", root))
    darkset = set(re.findall(r"(--[a-z0-9-]+):", dark))
    colours = {t for t in base
               if not t.startswith(("--r-", "--sp-", "--font", "--mono", "--maxw"))}
    missing = sorted(colours - darkset)
    assert not missing, f"tokens with no dark-theme value: {missing}"


#: Names that come from the browser, the DOM, or a builtin -- everything else
#: called from this file has to be defined in it.
_AMBIENT = {
    "if", "for", "while", "switch", "catch", "function", "return", "typeof",
    "new", "Promise", "Number", "String", "Date", "Array", "Object", "JSON",
    "Math", "fetch", "setTimeout", "parseInt", "parseFloat", "isNaN",
    "isFinite", "encodeURIComponent", "Error", "addEventListener",
    "confirm", "alert", "Map", "Set", "RegExp", "Boolean",
    "setInterval", "clearInterval", "clearTimeout",
    "removeEventListener", "querySelector", "querySelectorAll",
    "getElementById", "getItem", "setItem", "setAttribute", "removeAttribute",
    "click", "focus", "blur", "contains", "forEach", "from", "map", "join",
    "push", "find", "filter", "replace", "toFixed", "toLocaleString",
    "toLocaleDateString", "resolve", "json", "stopPropagation", "min", "max",
    "sort", "slice", "split", "trim", "includes", "scrollIntoView",
    "bind", "call", "apply", "select", "stringify", "parse", "then", "catch",
    "round", "floor", "ceil", "abs", "pow",
    "requestAnimationFrame", "cancelAnimationFrame", "matchMedia", "now",
    "concat", "isArray", "keys", "reverse", "toUpperCase", "casefold",
    "reduce",
    # Date instance methods, used to count sessions a name has been held
    "getDate", "setDate", "getDay",
    # CSS function names picked up by the same scan
    "var", "rgba", "rect", "minmax", "clamp", "repeat", "translateX",
    "translateY", "scaleX", "rotate", "brightness", "saturate", "blur",
    "gradient", "mix", "not", "bezier", "step", "add", "remove",
}


def test_every_function_the_interface_calls_is_defined():
    """Deleting a block of view code has twice taken a neighbouring helper with
    it -- `row()` once, then `openPanel`, `closePanel` and `panelHTML`. Both
    times the page loaded, the suite passed, and the failure was a dead click
    in a details panel that only appeared by opening it.
    """
    body = _html()[_html().index("<script>"):]
    defined = set(re.findall(r"(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", body))
    # Anything bound to a const is a definition -- an arrow function, or a
    # method captured off another object (`const rawFetch = fetch.bind(...)`).
    defined |= set(re.findall(
        r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", body))
    # No whitespace before the paren: a real call never has one, and allowing
    # it matched prose like "returned an error (" inside a string literal.
    # And NOT PRECEDED BY A DOT. `\b` matches immediately after ".", so
    # `map.get(k)` was read as a call to a bare `get`, and every method name in
    # the file had to be added to _AMBIENT to keep this quiet -- which is why
    # that set is forty browser methods long. A method call is not a reference
    # to a function this file has to define, so it is excluded at the pattern
    # rather than allowlisted afterwards.
    called = set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\(", body))
    # Handlers are passed by REFERENCE and never called, so the paren scan
    # cannot see them. confirmWipe was deleted with a neighbouring block and
    # this test passed anyway -- the Clear button threw on click.
    called |= set(re.findall(r"addEventListener\(\s*\"[^\"]+\"\s*,\s*([A-Za-z_$][\w$]*)\s*\)", body))
    missing = sorted(called - defined - _AMBIENT)
    assert not missing, f"called but never defined: {missing}"


def test_the_interface_has_one_results_tab():
    """Watchlist and Research were separate screens showing the same run from
    two more angles. Near misses already appear on the one screen whenever
    fewer than five names qualify, so a second list of them was a second place
    to look for something already in front of the reader."""
    html = _html()
    tabs = re.findall(r'data-view="([a-z]+)"', html)
    assert tabs == ["overview", "history"], tabs
    # Settings is a drawer, not a peer of the two screens you actually read.
    assert "viewMonitored" not in html and "viewResearch" not in html
    assert "viewMethod" not in html


def test_the_card_layout_does_not_depend_on_the_company_name():
    """One grid for the whole header. Two stacked columns could not keep the
    rows level: the left had three items and the right had two or three, so
    what sat beside the company name depended on how many lines it took."""
    html = _html()
    block = html[html.index(".card-top {"):html.index(".card-top .rank")]
    assert "grid-template-columns: minmax(0, 1fr) auto" in block
    assert "flex-wrap" not in block
    # Auto-placement carries the alignment, so source order is load-bearing:
    # rank and pill fill row 1, name and price row 2, ticker row 3.
    card = html[html.index("function cardHTML"):]
    card = card[:card.index("\nfunction ", 20)]
    head = card[card.index('class="card-top"'):card.index("'<div class=\"figs\">'")]
    at = {k: head.index(v) for k, v in (
        ("rank", 'class="rank"'), ("pill", 'class="pill '),
        ("name", 'class="name"'), ("price", 'class="p num"'),
    )}
    assert at["rank"] < at["pill"] < at["name"] < at["price"], at
    # The ticker joined the position label rather than taking a row of its own,
    # which was mostly empty; that space now carries the factor arithmetic.
    assert 'class="tk"' in head


def test_the_card_carries_no_verdict_words_at_all():
    """The pills read "Momentum strong / Trend Position strong" on nearly
    every name, which is what a shortlist ranked on momentum will always say.
    A label that never varies is not information, so the card carries numbers
    and one measured sentence instead."""
    html = _html()
    card = html[html.index("function cardHTML"):]
    card = card[:card.index("\nfunction ", 20)]
    assert "p.strength" not in card
    assert "highlights" not in card, "the pills are gone, not relocated"
    assert "whyLine(p)" in card, "something measured has to replace them"


def test_the_arithmetic_is_in_the_panel_and_covers_every_factor():
    """A quant needs z, the fitted coefficient and their product against the
    model's own identifier. Four of seventeen on the card answered neither
    audience -- too much for a glance, too little to check. It is all in the
    panel, and the panel is what "View analysis" opens."""
    html = _html()
    panel = html[html.index("function panelHTML"):]
    panel = panel[:panel.index("\nfunction ", 20)]
    assert "r.z.toFixed" in panel, "the standardised loading"
    assert "r.coefficient.toFixed" in panel, "the fitted coefficient"
    assert "r.contribution.toFixed" in panel, "their product"
    # The theme's LABEL leads and its column key follows it. `esc(r.factor)`
    # alone put <code>mom</code> and <code>delivery</code> in the heading of a
    # block whose summary table two sections above calls the same things
    # "Medium-term momentum" and "Delivery-backed participation" -- the same
    # theme under two names in one panel. The key is still rendered, beside the
    # members it belongs to, so a quant can still cross-reference it.
    assert "r.label || r.factor" in panel, "the theme's label, with its key kept"
    assert "esc(m.name)" in panel, "the model's own identifier, on the members"
    assert "rows.map(" in panel, "every theme, not a top slice"
    vm = UI.parents[1] / "presentation" / "viewmodel.py"
    assert "top: Optional[int] = None" in vm.read_text(encoding="utf-8"), \
        "the payload must carry every factor for that table to exist"


def test_the_advanced_section_does_not_repeat_the_summary():
    """It rendered `rows` twice -- the same five themes the table above already
    showed, with two more columns. The reader who opened it for the detail got
    the summary again.

    What is actually underneath is the MEMBERS: one coefficient is fitted per
    theme over the average of its members' ranks, so "lottery -1.81 sd" is a
    summary of four separate measurements and the panel never said which one
    moved."""
    html = _html()
    panel = html[html.index("function panelHTML"):]
    panel = panel[:panel.index("\nfunction ", 20)]
    adv = panel[panel.index("const advanced"):]
    assert "r.members" in panel, "the advanced section must show the members"
    # The wording has changed twice; what must not change is that the summary
    # COUNTS the measurements and names the themes they sit under, so a reader
    # can tell a five-row summary from the twenty rows underneath it.
    assert "members +" in adv and "rows.length" in adv, (
        "the summary must say how many measurements sit behind how many themes"
    )
    # The old duplicate: a second full table built from `rows` with z/coef/
    # contrib columns. The theme header carries those now, one line each.
    assert adv.count("rows.map(") == 0, (
        "the advanced section must not rebuild the drivers table"
    )


def test_the_panel_leads_with_the_model_that_ordered_the_book():
    """THE BUG THIS PINS. `factors` carries the v3 themes that ORDER the book
    and the fitted 26-factor themes that only watch it, in one flat dict. The
    panel rendered all eleven rows under one heading -- "the fitted model's
    separate reading" -- and summed their members together, which is how a
    22-factor model came to advertise "12 measured factors" on its own
    analysis screen.

    Two models, two sections, and the one that decides comes first."""
    html = _html()
    panel = html[html.index("function panelHTML"):]
    panel = panel[:panel.index("\nfunction ", 20)]

    assert "p.themes" in panel, "the panel must read the v3 themes"
    assert "p.model_reading" in panel, "and the secondary model separately"
    assert "What ordered this name" in panel
    assert "chose nothing" in panel, "the secondary model must be labelled as such"

    # Order matters: what decided is above what only watched.
    ret = panel[panel.rindex("return key + lead"):]
    assert ret.index("themeTable") < ret.index("drivers"), (
        "the composite that ordered the book must render above the model that "
        "did not"
    )


def test_the_panel_never_recomputes_a_contribution():
    """v3 renormalises its weights over the themes a name actually has, so
    `z x weight` is not the contribution -- measured on a live card it read 19%
    low on every theme, uniformly. The engine serialises the real number; the
    panel prints it."""
    html = _html()
    panel = html[html.index("function panelHTML"):]
    panel = panel[:panel.index("\nfunction ", 20)]
    theme_block = panel[panel.index("const themeTable"):panel.index("const themeMembers")]
    assert "r.contribution" in theme_block
    assert "r.z * r.coefficient" not in theme_block.replace(" ", "")
    assert "r.z*r.coefficient" not in theme_block.replace(" ", "")


def test_the_shortlist_says_why_it_has_no_return_column():
    """People keep looking for a % return column. The reason it is absent is
    that the engine estimates no per-name return -- the only return figures it
    has are population base rates identical on every row. Leaving that as a
    hole invites the reader to assume the number was omitted by accident."""
    html = _html()
    assert "function expectancyHTML" in html
    block = html[html.index("function expectancyHTML"):]
    block = block[:block.index("\nfunction ", 20)]
    assert "identical on every name above" in block
    assert "not a forecast" in block
    assert "no return column" in block


def test_the_hold_shown_is_not_a_bare_policy_constant():
    """"up to 63 sessions" is the backstop that force-closes a position, the
    same on every name in every run, and it read as a holding period. It is
    named as a backstop and the study's measured hold is shown beside it."""
    html = _html()
    assert "-session backstop" in html
    assert "sessions in the study" in html


def test_the_panel_says_what_was_computed_and_not_priced():
    """"26 factors" and "5 themes" are both true and neither alone is honest.
    The panel showed five and said nothing about the other twenty-one."""
    vm = (UI.parents[1] / "presentation" / "viewmodel.py").read_text(encoding="utf-8")
    assert "_unscored_note" in vm
    assert '"unscored"' in vm, "the note has to reach the payload"
    html = _html()
    assert "p.technical && p.technical.unscored" in html or "unscored" in html, \
        "and the panel has to render it"


def test_the_market_labels_carry_the_measurements_behind_them():
    """"Uptrend" on its own is an assertion. Stage 2 computes the slope, the
    distance from the 200-session average, the VIX level and its percentile,
    and none of them were being serialised."""
    from prosignal.presentation.viewmodel import _trend_evidence, _vol_evidence

    trend = _trend_evidence({"trend_slope_annualised": 0.182,
                             "index_vs_slow_ma_pct": 0.8})
    assert "+18.2%" in trend and "200-session" in trend
    vol = _vol_evidence({"vix_level": 11.2, "vix_percentile": 23.0})
    assert "11.2" in vol and "23rd" in vol


def test_the_method_note_admits_the_windows_are_not_fitted():
    """The note is no longer rendered -- the panel holding it was removed --
    but it stays in the payload, and if it is ever shown again it must not
    imply the 50/200 windows were tuned. They carry status UNVALIDATED."""
    from prosignal.presentation.viewmodel import _METHOD_NOTE

    assert "conventional defaults" in _METHOD_NOTE
    assert "not values fitted" in _METHOD_NOTE


def test_the_footer_carries_ownership_and_stops_there():
    """The owner asked for the footer to state who built it and nothing else.
    The descriptor duplicated the masthead, which already reads
    "ProSignal EQUITY RESEARCH", and the run disclaimer was a third line of
    small print under a single short copyright."""
    html = _html()
    assert "Built &amp; owned by" in html
    assert "Rahul Reddy Allu" in html
    assert "&copy; 2026 ProSignal" in html
    # Removed deliberately, not by accident.
    assert "Independent quantitative market" not in html
    assert "confidence_note" not in html


def test_the_run_still_records_what_it_could_not_check():
    """The notes panel went, then the header dot that replaced it went with
    the redundant date. The signal must not go with either -- an incomplete
    run says so under the page title, in words rather than a tooltip."""
    html = _html()
    # Third home for this signal: a folded panel, then a header dot that went
    # with the redundant date, and now the Market environment card. One of
    # the notes is the Stage 3 pledging gate reporting NOT_TESTABLE, which
    # the engine states rather than implies -- so it has to live somewhere.
    # The on-screen note was removed at the owner's request. The flags stay
    # on the run payload and in the ledger, and a check that could not run
    # still surfaces per name in the analysis panel -- which is the place it
    # actually bears on a decision.
    assert "Checks that could not run" in html
    assert "not_testable" in html
    assert "rests on partial evidence" in html


def test_a_completed_scan_invalidates_the_cached_history():
    """History loads once and is then held. Clearing it, scanning, and opening
    the tab returned the empty result cached at the moment of the clear -- the
    run had been recorded and the screen never asked again."""
    body = _html()
    scan = body[body.index("async function scan()"):body.index("async function readErr")]
    assert "state.history = undefined" in scan, (
        "a completed scan does not invalidate the history cache"
    )


def test_the_card_is_the_same_on_mobile_and_desktop():
    """Only padding and gap may change below 640px. Anything that repositions
    the price or the status badge would make the two read as different
    components -- which is what a flex row that reflowed on long names did.
    """
    html = _html()
    mobile = html[html.index("@media (max-width: 640px)"):]
    mobile = mobile[:mobile.index("\n}")]

    allowed = {"padding", "padding-left", "padding-right", "padding-top",
               "padding-bottom", "gap", "margin", "width", "justify-content",
               "border-left"}
    for rule in re.findall(r"([^{}]+)\{([^{}]*)\}", mobile):
        selector, body = rule[0].strip(), rule[1]
        if ".card" not in selector:
            continue
        props = {d.split(":")[0].strip() for d in body.split(";") if ":" in d}
        offending = sorted(props - allowed)
        assert not offending, (
            f"mobile restyles {selector!r} beyond spacing: {offending}"
        )


def test_every_card_uses_one_type_scale():
    """There was a larger 'featured' variant for the top name. Five cards that
    differ in size read as five different components."""
    assert "featured" not in _html()


def test_a_stale_result_is_never_shown_as_todays():
    """A previous session's shortlist under today's heading is not stale data,
    it is the wrong answer presented confidently. The screen compares the run's
    date with the newest session the store holds and refuses to render."""
    html = _html()
    assert "function isCurrent" in html
    body = html[html.index("function viewOverview"):html.index("function marketBlock")]
    assert "isCurrent(v)" in body, "the results view does not gate on freshness"


def test_anything_toggled_by_hidden_has_a_hidden_rule():
    """`display` on a class beats the user-agent's `[hidden] { display: none }`.
    Setting `.hidden` on such an element changes the property and nothing on
    screen -- the scan button reported `hidden === true` with a computed
    display of `flex`, and stayed visible on every tab.
    """
    html = _html()
    css = html[html.index("<style>"):html.index("</style>")]
    body = html[html.index("<script>"):]

    assert re.search(r"\.hidden\s*=", body), (
        "nothing is toggled by .hidden any more; this guard is now stale"
    )
    # Every class in the stylesheet that declares its own display and is also
    # referenced from the JS needs the rule spelled out.
    assert re.search(r"\.btn\[hidden\]\s*\{\s*display:\s*none", css), (
        ".btn sets display and is toggled with .hidden, but has no "
        "[hidden] rule -- the toggle would be silently inert"
    )


def test_clearing_history_asks_first():
    """Not with window.confirm. Embedded and sandboxed browser contexts
    suppress it, and a suppressed confirm returns false -- which is exactly
    what made the clear button do nothing at all when it was pressed.
    """
    html = _html()
    assert "window.confirm" not in html, (
        "native confirm is suppressed in embedded contexts and returns false"
    )
    assert "Clear the run history?" in html
    assert 'id="ask"' in html and "function ask(" in html


def test_clearing_history_says_the_record_is_kept():
    """The ledger backs the deflated-Sharpe trial count. A user clearing a
    screen must not be silently invalidating that."""
    html = _html()
    block = html[html.index("async function confirmWipe"):]
    block = block[:block.index("\n}")]
    assert "research record" in block


def test_the_card_numbers_its_position_not_the_models_rank():
    """Model ranks skip: on a live run the admitted names were 1, 2, 4, 5, 6
    because rank 3 was removed by a risk check and appears in neither list.
    Printing that on the card asked the reader to explain a gap that is about
    the engine's internals. Position in the shortlist has no gaps; the model
    rank moves into the analysis panel where there is room to say what it is.
    """
    html = _html()
    card = html[html.index("function cardHTML"):html.index("function viewHistory")]
    assert "p.position" in card
    assert "Rank " not in card, "the card still prints the model's rank"
    panel = html[html.index("function openPanel"):]
    assert "ranked #" in panel, "the model rank was dropped rather than moved"


def test_the_five_slot_rule_is_not_reimplemented_in_the_interface():
    """Selection is tested backend-side. A second implementation here would
    drift from it silently."""
    html = _html()
    assert "slice(0, 5)" not in html and "slice(0,5)" not in html
    assert ".picks" in html, "the interface must render what the backend selected"


def _contrast(fg: str, bg: str) -> float:
    def lum(colour: str) -> float:
        rgb = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        chan = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in rgb]
        return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]
    a, b = lum(fg), lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _tokens(block: str) -> dict:
    return dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})", block))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_muted_text_meets_aa_for_small_text(theme):
    """`--ink-3` carries the factor table's z and coefficient columns at
    11.5px. That is small text, so AA wants 4.5:1 and not the 3:1 that large
    text can get away with -- it was at 3.52 in light and 3.85 in dark, which
    reads as decoration rather than as data a quant is meant to check.
    """
    html = _html()
    if theme == "light":
        block = html[html.index(":root {"):html.index("@media (prefers-color-scheme: dark)")]
    else:
        block = html[html.index(':root[data-theme="dark"]'):]
        block = block[:block.index("}")]
    tok = _tokens(block)
    for surface in ("--surface", "--ground"):
        ratio = _contrast(tok["--ink-3"], tok[surface])
        assert ratio >= 4.5, (
            f"{theme}: ink-3 on {surface} is {ratio:.2f}:1, below AA for small text"
        )


# ------------------------------------------------------------------- brand
def test_the_product_is_called_prosignal_everywhere_it_is_the_brand():
    """The wordmark and the tab title are the only two places the product
    names itself. Every other 'signal' in this file is the ordinary word --
    signal strength, false signal -- and must not be touched."""
    html = _html()
    assert "<title>ProSignal</title>" in html
    assert '<div class="brand">ProSignal' in html
    assert ">Signal <span>" not in html, "the old wordmark survived"


def test_ownership_is_attributed_once_not_repeated():
    """Naming the owner twice in a footer this small is personal branding
    rather than quiet attribution."""
    html = _html()
    foot = html[html.index('<footer class="foot">'):html.index("</footer>")]
    assert foot.count("Rahul Reddy Allu") == 1


def test_the_owner_links_are_correct_and_open_safely():
    html = _html()
    assert 'href="https://www.linkedin.com/in/rahulreddyallu/"' in html
    assert 'href="mailto:rahulallu.career@gmail.com"' in html
    foot = html[html.index('<footer class="foot">'):html.index("</footer>")]
    assert 'target="_blank"' in foot and 'rel="noopener noreferrer"' in foot


def test_the_arrows_are_hidden_from_screen_readers():
    """A decorative glyph read aloud as "north east arrow" after every link is
    noise."""
    foot = _html()
    foot = foot[foot.index('<footer class="foot">'):foot.index("</footer>")]
    assert foot.count('aria-hidden="true"') >= 2


def test_the_copyright_names_the_product_not_the_person():
    """The product is the intellectual property; the person owns the product."""
    html = _html()
    assert "&copy; 2026 ProSignal. All rights reserved." in html


def test_the_page_never_hardcodes_a_disclaimer_of_its_own():
    """The footer no longer renders the run's disclaimer -- the owner removed
    it for space. What must not happen is the page inventing a replacement in
    markup, which would let it state something the engine never said. The
    engine still emits `disclaimer` on the payload; nothing on the page
    fabricates one."""
    html = _html()
    assert 'id="foot-note"' not in html
    assert "Not financial advice" not in html
    assert "No trades are placed" not in html


def test_the_footer_invents_no_legal_or_performance_claims():
    html = _html()
    foot = html[html.index('<footer class="foot">'):html.index("</footer>")]
    for invented in ("SEBI", "Pvt", "Ltd", "LLP", "registered", "GST",
                     "Privacy Policy", "Terms", "alpha", "beat the market",
                     "guaranteed", "institutional-grade"):
        assert invented.lower() not in foot.lower(), f"invented: {invented}"


def test_the_footer_uses_the_existing_tokens_and_no_new_colours():
    """A footer with its own palette is a second design system."""
    html = _html()
    # Only rules whose selector is the footer's. Slicing a range picked up
    # whatever happened to be declared next -- the drawer scrim's rgba and
    # the switch knob's #fff are correct where they are.
    style = html[html.index("<style>"):html.index("</style>")]
    rules = re.findall(r"^(\.foot[^{]*)\{([^}]*)\}", style, re.M)
    assert rules, "no footer rules found"
    for selector, body in rules:
        for literal in ("#", "rgb(", "rgba("):
            assert literal not in body, \
                f"hardcoded colour in {selector.strip()}: {literal}"


def test_the_ownership_line_does_not_wrap_mid_name_on_mobile():
    html = _html()
    assert "text-wrap: balance" in html


# ===================================================================
# Settings, the build tile, and the performance page
# ===================================================================

def test_settings_is_a_drawer_that_comes_from_the_side():
    """A peer slot next to the two screens you read implied it was somewhere
    to spend time. It slides from the side at every width -- a bottom sheet
    put it under the thumb and in front of the content at the same time."""
    html = _html()
    assert 'id="settings"' in html and 'class="drawer"' in html
    assert 'data-view="settings"' not in html
    assert 'state.tab === "settings"' not in html
    assert '$("#gear").addEventListener("click", openSettings)' in html
    css = html[html.index("<style>"):html.index("</style>")]
    assert "translateX(100%)" in css
    assert "translateY(100%)" not in css, "no bottom sheet at any width"


def test_the_drawer_does_not_open_behind_a_throttled_animation_frame():
    """requestAnimationFrame is throttled in a background tab. A drawer that
    never receives .in stays parked off-screen at translateX(100%) while
    reporting itself open -- which is how it was found."""
    html = _html()
    body = html[html.index("async function openSettings"):
                html.index("function closeSettings")]
    assert "requestAnimationFrame(" not in body
    assert "offsetWidth" in body


def test_one_switch_runs_the_engine_and_the_measurement():
    """Two switches for one idea. Nobody wants a measurement period running
    over days the engine did not record, or the engine recording into no
    period at all -- so the schedule switch moves both."""
    html = _html()
    assert '"sw-cron"' in html
    assert 'id="meas"' not in html, "the separate measurement control is gone"
    assert "toggleMeasurement" not in html
    src = (UI.parents[1] / "api.py").read_text(encoding="utf-8")
    resume = src[src.index("def operations_resume"):src.index("def _measurement_state")
                 if "def _measurement_state" in src[src.index("def operations_resume"):]
                 else len(src)]
    assert "_m.start" in src[src.index("def operations_resume"):][:1400]
    assert "_m.stop" in src[src.index("def operations_pause"):][:900]


def test_turning_it_back_on_after_a_change_opens_a_new_period():
    """That is the re-registration: the evidence from before a change never
    joins the evidence from after."""
    src = (UI.parents[1] / "api.py").read_text(encoding="utf-8")
    body = src[src.index("def operations_resume"):][:1400]
    assert 'status == "DRIFTED"' in body or "DRIFTED" in body


def test_the_schedule_switch_does_not_apologise_for_how_it_works():
    """It said pausing could not stop cron, which read as "this button does
    not work". The run does not happen; that is what off means."""
    html = _html()
    assert "does not stop cron" not in html
    assert "needs root" not in html
    assert "Nothing is recorded" in html


def test_the_one_click_erase_is_gone_but_the_endpoint_stays_guarded():
    html = _html()
    assert "/admin/reset/everything" not in html
    assert "confirmEraseEverything" not in html
    src = (UI.parents[1] / "api.py").read_text(encoding="utf-8")
    assert 'confirm") != "ERASE"' in src


def test_rebuilding_and_clearing_are_both_reachable_and_say_what_they_keep():
    html = _html()
    assert "/admin/reset/market-data" in html
    assert 'id="rebuild"' in html and 'id="wipe"' in html
    assert "Clears the History page" in html
    assert "Your results are" in html


# ===================================================================
# History -- the graph and the two lists, and nothing else
# ===================================================================

def test_history_is_a_graph_and_two_lists():
    """It answers which calls paid off and which did not, with the total for
    each. Everything that was not that answer was clutter around it."""
    html = _html()
    assert "function viewHistory" in html
    assert "Paid off" in html and "Went against" in html
    assert "function curveSVG" in html
    for gone in ("verdictCard", "tickerTable", "livePanel", "Following the shortlist",
                 "Cost the most", "Paid the most", "Uncorrected this reads"):
        assert gone not in html, gone


def test_the_two_lists_are_split_by_sign_and_neither_sums_its_column():
    """The split stays. The COLUMN TOTAL does not.

    Each column header carried `sum(items)` -- the sum of per-name total
    returns. Those names were held at the same time in a six-slot book, so
    adding them is a return on more capital than the strategy ever had; the
    same arithmetic put "-234.53%" at the top of this page as though it were a
    result. A count is what a list of names can honestly report about itself,
    and the return figures live per row, where they mean something.
    """
    html = _html()
    start = html.index("function viewHistory")
    body = html[start:html.index("\nfunction ", start + 20)]
    assert "total_return > 0" in body
    assert "total_return <= 0" in body
    assert "sum(items)" not in body, (
        "a summed column of overlapping positions is a return on capital the "
        "book never had"
    )
    assert "items.length" in body, "the column reports how many, not how much"


def test_a_name_row_says_what_it_returned_and_how_it_ended():
    html = _html()
    assert "function nameRow" in html
    assert "stopped out" in html and "hit its target" in html
    assert "paid off" in html
    assert 'data-stock="' in html, "each name still opens its own history"


def test_the_curve_is_summed_not_compounded():
    """The book holds several names at once. Compounding overlapping holds
    would draw a line the strategy never earned."""
    html = _html()
    assert "summed, not compounded" in html


def test_an_empty_history_explains_itself():
    """An empty page must say WHY it is empty.

    It used to say "No results yet ... the day the market takes it to its
    target or its stop" -- two exits this engine has disarmed, on a page that
    was not actually empty. Scoped to the running configuration it genuinely is
    empty, and the honest statement is that nothing has closed UNDER THIS
    CONFIGURATION, with the count of what was left out beside it.
    """
    html = _html()
    assert "The record opens here" in html
    assert "closed under this configuration" in html
    assert "superseded configuration" in html, \
        "an empty page must say what it excluded, or it reads as broken"
    assert "the day it closes" in html
    assert "target or a stop" not in html, \
        "the target and the stop-as-exit are not how a position ends any more"


def test_the_history_is_scoped_by_configuration_and_not_by_measurement_period():
    """Two different scopings, and only one of them is right.

    MEASUREMENT PERIOD, off by default. A period is an operator's clock; it can
    be started for any reason, and scoping by it meant turning the daily run on
    emptied a history of 136 closed trades that were perfectly valid.

    CONFIGURATION, on. Trades decided by a superseded configuration describe a
    different engine -- a different universe, a different sizer, a different
    exit rule -- and averaging them with current ones reports two engines as
    one. That is the failure `exit_model` and `epoch_id` already guard against
    elsewhere, and this endpoint was exempt from it: its headline was the sum
    of 97 baseline-v1 trades from an epoch recorded CLOSED VOID.
    """
    src = (UI.parents[1] / "api.py").read_text(encoding="utf-8")
    assert 'def performance_report(period: str = "all")' in src, \
        "the measurement period must still default to off"
    assert 'r.get("config_version") or ""' in src, \
        "the statistics must be scoped to the configuration that produced them"
    assert '"excluded_closed"' in src, \
        "and the page must be able to say how much it left out"


def test_the_icons_are_drawn_not_typed():
    """A glyph gear renders at whatever size the platform font decides; on
    iOS it was a speck inside its box."""
    html = _html()
    assert "&#9881;" not in html and "&#10005;" not in html
    gear = html[html.index('id="gear"'):]
    assert "<svg" in gear[:200]
    css = html[html.index("<style>"):html.index("</style>")]
    assert ".iconbtn svg" in css


def test_the_classes_that_carry_typography_are_all_styled():
    """`.fine` was deleted with the settings block it happened to live in,
    and three callers rendered small print at body size. Nothing errored."""
    html = _html()
    css = html[html.index("<style>"):html.index("</style>")]
    for cls in ("fine", "reading", "rs", "rl", "grp-h", "nr-t", "nr-h", "nr-v",
                "col-h", "chart-v", "chart-s"):
        assert "." + cls + " " in css or "." + cls + "{" in css, \
            f".{cls} is used for text but has no rule"


def test_every_control_the_drawer_renders_is_bound_where_it_is_rendered():
    """wire() belongs to render(); the drawer is painted by paintSettings.
    A control that moved into the drawer kept its binding in wire() and so
    was never bound at all -- Clear looked entirely normal and did nothing."""
    html = _html()
    render = html[html.index("function renderSettings"):
                  html.index("/* Opening and closing.")]
    paint = html[html.index("function paintSettings"):
                 html.index("async function loadMeasurement")]
    ids = set(re.findall(r'id="([a-z0-9-]+)"', render))
    ids |= set(re.findall(r'toggle\("([a-z0-9-]+)"', render))
    unbound = sorted(i for i in ids if '$("#' + i + '")' not in paint)
    assert not unbound, f"rendered by the drawer but never bound: {unbound}"


def test_history_does_not_need_a_scan_to_exist():
    """viewHistory reads resolved outcomes. Testing state.view first sent it
    down the "no scan yet" branch, so opening History on a fresh deployment
    rendered Today's empty state under the History tab."""
    html = _html()
    start = html.index("function render()")
    body = html[start:html.index("\nfunction ", start + 20)]
    assert body.index('state.tab === "history"') < body.index("if (!v)"), \
        "the history branch must be reached before the no-scan branch"


def test_the_empty_today_screen_is_one_card_not_two():
    """The build tile carries its own explanation and its own button, and the
    masthead already carries Scan Market."""
    html = _html()
    assert "narrows to the setups that clear every risk check" not in html
    assert "Rank the market to see today" in html


def test_the_drawer_has_no_second_way_to_run_a_scan():
    """Update now started the same job the Scan Market button does."""
    html = _html()
    assert "/admin/run-now" not in html
    assert "Update now" not in html


def test_clearing_results_is_applied_where_results_are_read():
    """The outcomes file is derived and rebuilt on every request, so deleting
    it clears the screen only until the next one. The watermark has to be
    applied at read time -- otherwise Clear silently does nothing."""
    src = (UI.parents[1] / "api.py").read_text(encoding="utf-8")
    # Both readers now share one resolution, and the mark is applied inside
    # it -- so it cannot be applied to one screen and forgotten on the other.
    body = src[src.index("def _resolved_rows"):src.index("@app.get(\"/stock/")]
    assert "_apply_clear_mark" in body
    for reader in ("def performance_report", "def stock_calls"):
        r = src[src.index(reader):]
        assert "_resolved_rows()" in r[:1600], reader


def test_history_loads_on_what_it_reads_not_on_a_neighbouring_variable():
    """History sat empty after toggling Daily signals.

    viewHistory reads state.perf. The loader was gated on state.history, and
    the two go stale separately: toggleSchedule clears perf and only reloads
    it when History is the open tab. Toggle it from Today -- which is where
    the Settings drawer opens from -- and nothing ever loaded perf back, so
    the page kept its skeleton and read as an empty history rather than a
    stuck one.
    """
    html = _html()
    gate = html[html.index("function goTab"):html.index("\n}", html.index("function goTab"))]
    assert "state.perf === undefined" in gate, \
        "the loader must fire on the variable the screen actually reads"


def test_a_failed_load_says_so_instead_of_showing_a_skeleton():
    """A skeleton that never resolves is indistinguishable from a history
    with nothing in it."""
    html = _html()
    loader = html[html.index("async function loadPerformance"):]
    loader = loader[:loader.index("\n}")]
    assert "failed" in loader
    view = html[html.index("function viewHistory"):]
    view = view[:view.index("\nfunction ", 20)]
    assert "perf.failed" in view
    assert "Could not read the results" in html


def test_clearing_perf_anywhere_is_recoverable():
    """Every path that invalidates perf must either reload it or leave a
    gate that will."""
    html = _html()
    # The gate is the safety net for the paths that do not reload inline.
    gate = html[html.index("function goTab"):html.index("\n}", html.index("function goTab"))]
    assert "loadHistory()" in gate


# ===================================================================
# A running scan belongs to Today
# ===================================================================

def test_a_running_scan_does_not_paint_itself_into_history():
    """render() returned early for EVERY tab while busy, and renderProgress
    wrote #view on a timer without knowing which tab was open. Switching to
    History mid-scan left the progress card sitting inside it."""
    html = _html()
    start = html.index("function render()")
    body = html[start:html.index("\nfunction ", start + 20)]
    assert 'state.busy && state.tab === "overview"' in body, \
        "busy must only own Today"
    prog = html[html.index("function renderProgress"):]
    prog = prog[:prog.index("\n}")]
    assert 'state.tab !== "overview"' in prog, \
        "the poller must not write into whatever tab is open"


def test_a_scan_invalidates_the_thing_history_reads():
    """It cleared state.history, which viewHistory does not read. After a
    clear-then-scan the page kept the empty result cached at the clear."""
    html = _html()
    scan = html[html.index("async function scan()"):]
    scan = scan[:scan.index("/* ------")]
    assert "state.perf = undefined" in scan


def test_a_running_scan_can_be_cancelled():
    """Abandoning the poll would leave the run finishing invisibly and
    writing a ledger row for a scan the reader believes they cancelled."""
    html = _html()
    assert "function cancelScan" in html
    assert 'id="cancel-scan"' in html
    cancel = html[html.index("async function cancelScan"):]
    cancel = cancel[:cancel.index("\n}")]
    assert "/cancel" in cancel and 'method: "POST"' in cancel


def test_the_open_tab_survives_a_refresh():
    """Refreshing on History landed on Today, because the tab lived only in
    memory."""
    html = _html()
    assert "signal.tab" in html
    assert "function startingTab" in html
    # A stale or hand-edited value must not strand the app somewhere it
    # cannot render.
    st = html[html.index("function startingTab"):]
    st = st[:st.index("\n}")]
    assert '"history"' in st and '"overview"' in st


def test_boot_loads_the_tab_it_restored():
    """boot() only ever fetched the Today view, so a refresh onto History
    rendered its skeleton and waited for a load nobody had started."""
    html = _html()
    boot = html[html.index("async function boot()"):]
    boot = boot[:boot.index("\n}")]
    assert 'state.tab === "history"' in boot and "loadHistory()" in boot


def test_open_calls_are_shown_not_just_counted():
    """"14 calls are still open" and nothing else is the least useful form
    the information has. A position moves every session, and that movement
    is the only thing this page has until the first call closes."""
    html = _html()
    assert "function openHTML" in html
    body = html[html.index("function openHTML"):html.index("function viewHistory")]
    assert "entry_price" in body and "last_price" in body
    assert "unrealised" in body
    assert "sessions_held" in body
    # And it must be reachable with nothing closed at all. Asserted on the
    # RETURN EXPRESSION rather than on where the strings happen to sit in the
    # source: the closed-trade copy is built into a variable above the return,
    # so a source-position comparison reads the order backwards.
    view = html[html.index("function viewHistory"):]
    view = view[:view.index("\nfunction ", 20)]
    ret = re.search(r"return chart \+ ([^;]+);", view)
    assert ret, "viewHistory must end in one composed return"
    order = ret.group(1)
    assert order.index("openHTML(open)") < order.index("closed"), \
        "the open marks must render above the closed record, not after it"


def test_an_open_mark_is_never_presented_as_a_result():
    html = _html()
    body = html[html.index("function openHTML"):html.index("function viewHistory")]
    assert "not yet a result" in body


def test_the_interface_script_actually_parses():
    """Every other test in this file matches strings, so a page that does not
    parse at all passes all of them. One shipped: a multi-line comment with
    "//" on only the first line, which made lines two and three bare code.
    The whole interface was a blank screen and the suite was green."""
    import shutil
    import subprocess

    node = shutil.which("node")
    js = _html()
    js = js[js.index("<script>") + len("<script>"):js.rindex("</script>")]
    if node:
        r = subprocess.run([node, "--check", "-"], input=js,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"index.html does not parse:\n{r.stderr[:600]}"
        return
    # No node: catch the specific shape that caused it -- a line that reads
    # like prose sitting where a statement belongs.
    pytest.skip("node not available to parse-check the interface")
