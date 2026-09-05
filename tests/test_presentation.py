"""The reader-facing layer: what the card says, and whether it can say it.

The completeness tests here are the ones that matter. This layer failed
totally for a release -- every category "Not available", `confirmation_count`
returning (0, 0), the card printing "0 of 0 areas support this" on every name
-- because `FACTOR_MAP` was keyed on individual factors after the model moved
to fitting families. Nothing raised, and the test that was supposed to catch it
asserted a hardcoded literal tuple against the table rather than asking the
engine what it fits. So the assertions below are written against
`crossmodel.FAMILIES` and against the key shape the engine actually emits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosignal.features.crossmodel import FAMILIES, FAMILY_COLUMNS, _bare
from prosignal.presentation.evidence import (
    COMPOSITE_KEYS, EVIDENCE_CATEGORIES, FACTOR_MAP, FAMILY_MAP, MODEL_KEYS,
    build_evidence, confirmation_count,
)
from prosignal.presentation.narrative import build_narrative
from prosignal.presentation.viewmodel import _scorer_used, build_view


def factor(sd: float, *, weight: float = 0.02, raw: float = 1.0,
           available: bool = True) -> dict:
    """One theme as the engine serialises it.

    ``weight`` carries a SIGN for a family: it is the fitted coefficient, and
    it is what orients the reading. A positive coefficient means a high value
    raises the score.
    """
    return {"standardised": sd, "weight": weight, "raw": raw,
            "available": available}


# ------------------------------------------------------------- completeness
def test_every_theme_the_engine_fits_has_a_label():
    """The check that was missing. Asserted against the engine's own family
    registry, so adding a family without a label fails here rather than
    silently vanishing from the screen."""
    missing = sorted(set(FAMILIES) - set(FAMILY_MAP))
    assert not missing, f"families with no reader label: {missing}"


def test_no_label_exists_for_a_theme_the_engine_does_not_fit():
    """The other direction. A label for a family that no longer exists is a
    category that can never populate, which is how a panel goes quietly
    blank."""
    stale = sorted(set(FAMILY_MAP) - set(FAMILIES))
    assert not stale, f"labels for families the engine does not fit: {stale}"


def test_the_keys_the_card_actually_carries_are_all_mappable():
    """The integration check. Stage 4 keys the card's factor block by
    `_bare(column)` for every fitted column, so THAT is the shape this layer
    receives -- not the factor names, and not anything a test author typed."""
    emitted = {_bare(c) for c in FAMILY_COLUMNS}
    unmapped = sorted(emitted - set(FAMILY_MAP))
    assert not unmapped, f"the card emits keys this layer cannot read: {unmapped}"


def test_a_live_shaped_factor_block_populates_the_panel():
    """End to end on the key shape the engine emits: at least one category has
    to come back available. This is the assertion whose absence let the whole
    panel report 'Not available' on every run."""
    block = {_bare(c): factor(1.2, weight=0.03) for c in FAMILY_COLUMNS}
    cats = build_evidence(block)
    assert any(c.available for c in cats), "no category populated from live keys"
    agree, judged = confirmation_count(cats)
    assert judged > 0, "confirmation_count reported nothing judgeable"


def test_the_scorer_report_does_not_guess_at_a_fork_that_no_longer_exists():
    """`_scorer_used` used to tell the fitted model and the hand-weighted
    composite apart by their factor keys. Both are deleted; the detector then
    reported "cross-sectional / validated" on every healthy run, which is the
    misrepresentation it existed to prevent, inverted a second time."""
    from prosignal.features import engine

    got = _scorer_used([{"factors": {t: {} for t in engine.THEMES}}])
    assert got["model"] == "engine"
    assert got["alert"] is False
    # No key set can produce any other model, because there is no other model.
    assert _scorer_used([{"factors": {"something_else": {}}}])["model"] == "engine"


def test_no_category_claims_an_indicator_this_engine_does_not_compute():
    """This is a cross-sectional ranking model. It has no RSI, no MACD, no
    moving-average crossover. A layout that showed those would be fabricating
    evidence to fill a familiar shape."""
    labels = " ".join(l for _, l in EVIDENCE_CATEGORIES).lower()
    names = " ".join(v[1] for v in FAMILY_MAP.values()).lower()
    names += " " + " ".join(v[1] for v in FACTOR_MAP.values()).lower()
    for invented in ("rsi", "macd", "ema", "stochastic", "bollinger"):
        assert invented not in labels and invented not in names


# ----------------------------------------------------------------- verdicts
def test_orientation_follows_the_fitted_coefficient_not_a_literature_prior():
    """THE test for this layer. `reversal` carries a positive literature prior
    and a NEGATIVE fitted coefficient. A table that hardcodes "high is good"
    would tell a reader the name is strong on an axis that is lowering its
    score."""
    priced_up = {c.key: c for c in build_evidence(
        {"reversal": factor(2.0, weight=+0.05)})}
    priced_down = {c.key: c for c in build_evidence(
        {"reversal": factor(2.0, weight=-0.05)})}
    assert priced_up["reversal"].verdict == "Strong"
    assert priced_down["reversal"].verdict == "Weak"


def test_a_gated_theme_reads_as_not_used_rather_than_neutral():
    """The estimator writes exactly zero for a theme it could not measure past
    its significance floor. That is a refusal to weight, not a finding of
    neutrality, and averaging the zero into a verdict says the opposite."""
    cats = {c.key: c for c in build_evidence({"mom": factor(1.8, weight=0.0)})}
    assert cats["momentum"].available is False
    assert cats["momentum"].verdict == "Not used"
    assert "significance floor" in cats["momentum"].detail


def test_a_gated_theme_is_excluded_from_the_confirmation_count():
    """A theme the model declined to use is not evidence for or against."""
    cats = build_evidence({
        "mom": factor(1.8, weight=0.0),           # gated
        "delivery": factor(1.2, weight=+0.03),    # priced, raises
        "lottery": factor(1.5, weight=-0.08),     # priced, lowers
    })
    agree, judged = confirmation_count(cats)
    assert judged == 2, "a gated theme was counted as judgeable"
    assert agree == 1


def test_a_low_lottery_reading_is_contained_not_weak():
    """Lottery demand is priced negatively, so a LOW reading helps the name.
    Orienting it naively would call the safest name in the market its worst."""
    cats = {c.key: c for c in build_evidence(
        {"lottery": factor(-1.5, weight=-0.08)})}
    assert cats["lottery"].verdict == "Contained"


def test_high_lottery_reads_as_elevated():
    cats = {c.key: c for c in build_evidence(
        {"lottery": factor(1.8, weight=-0.08)})}
    assert cats["lottery"].verdict == "Elevated"


def test_a_name_near_the_average_is_called_neutral():
    cats = {c.key: c for c in build_evidence({"mom": factor(0.1, weight=0.05)})}
    assert cats["momentum"].verdict == "Neutral"


def test_the_category_is_weighted_by_how_much_each_theme_moves_the_score():
    """An equal average lets a theme the model barely uses outvote the one
    carrying the block. Exercised on the composite path, which is the one that
    puts two inputs in a single category."""
    cats = {c.key: c for c in build_evidence({
        "momentum_12_1": factor(2.0, weight=0.05),
        "sector_relative_strength": factor(-2.0, weight=0.001),
    })}
    assert cats["momentum"].verdict == "Strong"


def test_an_unavailable_theme_does_not_count_as_evidence():
    cats = {c.key: c for c in build_evidence({
        "value": factor(2.0, weight=0.02, available=False),
    })}
    assert cats["valuation"].available is False
    assert cats["valuation"].verdict == "Not available"


def test_a_theme_with_no_coefficient_at_all_is_not_read():
    """No weight means no orientation, and a reading with an unknown direction
    is worse than no reading."""
    cats = {c.key: c for c in build_evidence(
        {"delivery": {"standardised": 1.5, "available": True, "raw": 1.0}})}
    assert cats["participation"].available is False


def test_missing_valuation_says_why_rather_than_going_blank():
    cats = {c.key: c for c in build_evidence({})}
    assert "financials" in cats["valuation"].detail


# ---------------------------------------------------------------- narrative
def _card(ticker: str, rank: int, factors: dict, against=None) -> dict:
    return {"ticker": ticker, "model_rank": rank, "factors": factors,
            "against": against or [], "exits": []}


def test_two_different_names_do_not_produce_the_same_paragraph():
    a = _card("AAA", 1, {"mom": factor(2.0, weight=0.05),
                         "lottery": factor(-1.5, weight=-0.08)})
    b = _card("BBB", 2, {"value": factor(2.0, weight=0.04),
                         "lottery": factor(1.8, weight=-0.08)})
    na = build_narrative(a, build_evidence(a["factors"]), status="BUY",
                         entry_rank=8, exit_rank=16)
    nb = build_narrative(b, build_evidence(b["factors"]), status="BUY",
                         entry_rank=8, exit_rank=16)
    assert na["thesis"] != nb["thesis"]


def test_the_thesis_names_the_area_that_actually_leads():
    card = _card("AAA", 1, {"value": factor(2.4, weight=0.04),
                            "mom": factor(0.05, weight=0.05)})
    n = build_narrative(card, build_evidence(card["factors"]), status="BUY",
                        entry_rank=8, exit_rank=16)
    assert "Valuation" in n["thesis"]


def test_a_watch_name_is_told_how_far_it_has_to_climb():
    card = _card("AAA", 11, {"mom": factor(1.2, weight=0.05)})
    n = build_narrative(card, build_evidence(card["factors"]), status="WATCH",
                        entry_rank=8, exit_rank=16)
    assert "climb 3 places" in n["what_would_change"]


def test_a_buy_name_is_told_what_would_close_it():
    card = _card("AAA", 2, {"mom": factor(1.2, weight=0.05)})
    n = build_narrative(card, build_evidence(card["factors"]), status="BUY",
                        entry_rank=8, exit_rank=16)
    assert "top 16" in n["what_would_change"]


def test_the_penalty_is_shown_as_a_reason_not_as_a_score_delta():
    """The -0.10 is what reordered the old table. It means nothing without the
    whole scale, and the measurement behind it is the part that helps."""
    card = _card("AAA", 1, {"mom": factor(1.2, weight=0.05)}, against=[
        "low_volume_breakout: latest session volume is 0.33x its 20-session "
        "average, below the 1.50x participation this check requires (-0.10)"
    ])
    n = build_narrative(card, build_evidence(card["factors"]), status="BUY",
                        entry_rank=8, exit_rank=16)
    assert "0.33x" in n["thesis"]
    assert "-0.10" not in n["thesis"]
    assert "low_volume_breakout" not in n["thesis"]


def test_no_engine_vocabulary_reaches_the_prose():
    card = _card("AAA", 1, {
        "mom": factor(1.5, weight=0.05), "delivery": factor(1.2, weight=0.03),
        "reversal": factor(1.1, weight=-0.06), "risk": factor(-0.9, weight=0.02),
    })
    n = build_narrative(card, build_evidence(card["factors"]), status="BUY",
                        entry_rank=8, exit_rank=16)
    blob = json.dumps(n)
    for token in ("resid_mom", "deliv_pct", "prox_52w", "amihud", "max_dd_120",
                  "mom_f", "reversal_f", "lottery_f", "delivery_f",
                  "composite_score", "model_rank", " sd", "coefficient"):
        assert token not in blob, f"engine vocabulary leaked: {token!r}"


# ---------------------------------------------------------------- view model
def test_the_view_never_presents_the_score_as_a_probability():
    view = build_view({"recommendations": [], "watchlist": []})
    blob = json.dumps(view).lower()
    assert "probability of" not in blob
    assert "% chance" not in blob


def test_a_company_name_is_joined_when_the_engine_left_it_blank():
    """The engine sets this on 0 of 52 rows while the store holds 2,565."""
    payload = {"recommendations": [_card("RELIANCE", 1, {})], "watchlist": []}
    view = build_view(payload, company_names={"RELIANCE": "Reliance Industries Limited"})
    assert view["picks"][0]["company"] == "Reliance Industries"


def test_an_unknown_company_falls_back_to_the_ticker_not_to_nothing():
    payload = {"recommendations": [_card("NEWCO", 1, {})], "watchlist": []}
    view = build_view(payload, company_names={})
    assert view["picks"][0]["company"] == "NEWCO"


def test_the_journey_uses_reader_labels_not_funnel_keys():
    payload = {"recommendations": [], "watchlist": [],
               "funnel": {"universe_considered": 750, "passed_eligibility": 617,
                          "survived_defense": 52, "buys": 8}}
    view = build_view(payload)
    labels = [s["label"] for s in view["journey"]]
    assert "Stocks evaluated" in labels
    assert not any("_" in l for l in labels)


def test_the_technical_detail_survives_behind_the_card():
    """Progressive disclosure moves the detail; it must not delete it."""
    card = _card("AAA", 1, {"mom": factor(1.5, weight=0.05)})
    card["why"] = ["mom: +1.55 sd ... (coefficient +0.02036)"]
    card["score"] = 0.9
    view = build_view({"recommendations": [card], "watchlist": []})
    tech = view["picks"][0]["technical"]
    assert tech["score"] == 0.9
    assert tech["why_raw"] == card["why"]
