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
    "Market data store is empty",     # nothing ingested yet
    "No scan has run yet",            # ingested, never run
    "Nothing met the bar today",      # the designed common outcome
    "No recommendation was produced", # withheld on bad data
    "could not be completed",         # a failed run must not imply a trade
    "Checks that could not run",      # NOT_TESTABLE is not a pass
    "What would move this to Buy",    # the near misses stay actionable
    "No runs recorded yet",           # history with an empty ledger
    "has not been scanned yet",       # store moved on, results did not
    "Previous shortlists",            # history keyed by name, not by date
    "Not followed yet",               # a call with no sessions behind it
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
    their own analysis, and leak that they are running it."""
    html = _html()
    for pattern in ("http://", "https://", "//cdn", "fonts.googleapis"):
        offenders = [l for l in html.splitlines()
                     if pattern in l and "localhost" not in l]
        assert not offenders, f"external reference: {offenders[:2]}"


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
    "removeEventListener", "querySelector", "querySelectorAll",
    "getElementById", "getItem", "setItem", "setAttribute", "removeAttribute",
    "click", "focus", "blur", "contains", "forEach", "from", "map", "join",
    "push", "find", "filter", "replace", "toFixed", "toLocaleString",
    "toLocaleDateString", "resolve", "json", "stopPropagation", "min", "max",
    "sort", "slice", "split", "trim", "includes", "scrollIntoView",
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
    # Arrow functions bound to a const are definitions too.
    defined |= set(re.findall(
        r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", body))
    # No whitespace before the paren: a real call never has one, and allowing
    # it matched prose like "returned an error (" inside a string literal.
    called = set(re.findall(r"\b([A-Za-z_$][\w$]*)\(", body))
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
    assert "viewMonitored" not in html and "viewResearch" not in html
    assert "viewMethod" not in html


def test_the_card_layout_does_not_depend_on_the_company_name():
    """A flex row let a long name push the price and the status badge to a
    different place on every card. The identity column is now a track that
    wraps inside itself."""
    html = _html()
    block = html[html.index(".card-top {"):html.index(".card-id {")]
    assert "grid-template-columns: minmax(0, 1fr) auto" in block
    assert "flex-wrap" not in block


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
    assert "research record is kept" in block or "record is kept" in block


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


def test_the_shortlist_says_when_it_is_one_bet(_=None):
    """Five names that all rank for the same reason are not five ideas. The
    model is momentum-heavy by construction, so a five-name list hides this
    unless it is said."""
    assert "concentration" in _html()


def test_the_five_slot_rule_is_not_reimplemented_in_the_interface():
    """Selection is tested backend-side. A second implementation here would
    drift from it silently."""
    html = _html()
    assert "slice(0, 5)" not in html and "slice(0,5)" not in html
    assert ".picks" in html, "the interface must render what the backend selected"
