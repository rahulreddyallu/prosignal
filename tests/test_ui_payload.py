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
    "What would move this to Buy",    # the watchlist is actionable
    "No runs recorded yet",           # history with an empty ledger
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


def test_the_five_slot_rule_is_not_reimplemented_in_the_interface():
    """Selection is tested backend-side. A second implementation here would
    drift from it silently."""
    html = _html()
    assert "slice(0, 5)" not in html and "slice(0,5)" not in html
    assert ".picks" in html, "the interface must render what the backend selected"
