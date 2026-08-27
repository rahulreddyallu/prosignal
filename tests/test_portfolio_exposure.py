"""Aggregate theme exposure across the final basket.

Pairwise correlation and the sector cap both look at names two at a time. Five
names can clear every pair and every sector and still be the same bet: all
loading the same theme, differently labelled.

WHY THESE TESTS CHANGED. The themes were a hardcoded tuple of individual factor
names, and the fit moved to families -- so `_aggregate_exposure` looked up
`mom_6_1` in a dict keyed `mom` and returned {} on every call for a release.
The check could not fire. The old tests did not catch it because they built
their baskets FROM the same tuple they then asserted against, so the test and
the code shared one wrong assumption and agreed with each other.

They are now written against the key shape the engine emits, and the theme list
is recovered from the card rather than declared here.
"""
from __future__ import annotations

import pytest

from prosignal.core.contracts import FactorScore, StockScore
from prosignal.features.crossmodel import FAMILY_COLUMNS, _bare
from prosignal.stages.stage8_final_signal import (
    _EXPOSURE_LIMIT, _aggregate_exposure, _exposure_themes,
)

#: What Stage 4 actually keys the card's factor block by.
LIVE_THEMES = tuple(_bare(c) for c in FAMILY_COLUMNS)


def _score(ticker, loadings, weight=0.01):
    return StockScore(
        ticker=ticker,
        factors={
            name: FactorScore(name=name, raw_value=0.0, standardised=value,
                              weight=weight, evidence_tier="model")
            for name, value in loadings.items()
        },
    )


# ------------------------------------------------------- theme discovery
def test_the_themes_are_recovered_from_the_card_not_declared_here():
    """The assertion whose absence let the check die silently."""
    basket = [_score("A", {t: 0.5 for t in LIVE_THEMES})]
    assert set(_exposure_themes(basket)) == set(LIVE_THEMES)


def test_a_gated_theme_is_not_an_exposure():
    """A theme the estimator set to exactly zero moves no score, so a basket
    cannot be concentrated in it."""
    basket = [_score("A", {"mom": 0.9, "delivery": 0.9})]
    basket[0].factors["mom"].weight = 0.0
    assert set(_exposure_themes(basket)) == {"delivery"}


def test_an_empty_basket_yields_no_themes():
    assert _exposure_themes([]) == ()


# ------------------------------------------------------------- exposure
def test_a_diversified_basket_is_under_the_limit():
    basket = [
        _score("A", {t: 0.9 for t in LIVE_THEMES}),
        _score("B", {t: -0.8 for t in LIVE_THEMES}),
        _score("C", {t: 0.1 for t in LIVE_THEMES}),
    ]
    exposure = _aggregate_exposure(basket, _exposure_themes(basket))
    assert exposure, "a populated basket produced no exposure at all"
    assert all(abs(v) < _EXPOSURE_LIMIT for v in exposure.values())


def test_five_names_that_pass_every_pair_can_still_be_one_bet():
    """The case the existing checks cannot see."""
    basket = [_score(t, {f: 1.2 for f in LIVE_THEMES}) for t in "ABCDE"]
    exposure = _aggregate_exposure(basket, _exposure_themes(basket))
    breached = [n for n, v in exposure.items() if abs(v) >= _EXPOSURE_LIMIT]
    assert set(breached) == set(LIVE_THEMES)


def test_a_basket_concentrated_the_other_way_also_breaches():
    """The limit is on magnitude. Five names all a full sd BELOW the universe
    on one theme are as much one position as five above it."""
    basket = [_score(t, {"lottery": -1.4}) for t in "ABCDE"]
    exposure = _aggregate_exposure(basket, _exposure_themes(basket))
    assert abs(exposure["lottery"]) >= _EXPOSURE_LIMIT


def test_exposure_is_the_mean_loading():
    basket = [_score("A", {"risk": 1.0}), _score("B", {"risk": 0.0})]
    assert _aggregate_exposure(basket, ("risk",))["risk"] == pytest.approx(0.5)


def test_a_missing_theme_is_skipped_rather_than_counted_as_zero():
    """Counting an absent loading as zero would dilute a real concentration
    and quietly let the basket through."""
    basket = [_score("A", {"risk": 1.0}), _score("B", {})]
    assert _aggregate_exposure(basket, ("risk",))["risk"] == pytest.approx(1.0)


def test_an_unknown_theme_produces_no_entry():
    basket = [_score("A", {"risk": 1.0})]
    assert _aggregate_exposure(basket, ("not_a_theme",)) == {}


def test_an_empty_basket_is_not_an_exposure():
    assert _aggregate_exposure([], LIVE_THEMES) == {}


def test_aggregate_exposure_reports_and_never_decides():
    """It annotates. It can neither promote nor reject.

    It used to downgrade to WATCHLIST. Measured under rank admission that
    rejected 5 of 8 names and cut the selection-period book to 3.9, taking
    Sharpe from +0.86 to +0.45 while drawdown barely moved -- a constraint
    fighting the model, since the top names load high on whatever the fit
    prices by construction. What survives is the reporting.
    """
    import inspect

    from prosignal.stages import stage8_final_signal as s8
    source = inspect.getsource(s8.run)
    block = source[source.index("_aggregate_exposure"):]
    block = block[:block.index("if len(buys)")]
    assert "buys.append" not in block, "the check must never promote a name"
    assert "Decision.WATCHLIST" not in block, "the check must never reject a name"
    assert "why_this_signal_exists.append" in block, "it must still say so"
