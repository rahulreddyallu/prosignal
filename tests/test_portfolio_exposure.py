"""Aggregate factor exposure across the final basket.

Pairwise correlation and the sector cap both look at names two at a time. Five
names can clear every pair and every sector and still be the same bet: all
high-momentum, all high-beta, differently labelled.
"""
from __future__ import annotations

import pytest

from prosignal.core.contracts import FactorScore, StockScore
from prosignal.stages.stage8_final_signal import (
    _EXPOSURE_FACTORS, _EXPOSURE_LIMIT, _aggregate_exposure,
)


def _score(ticker, loadings):
    return StockScore(
        ticker=ticker,
        factors={
            name: FactorScore(name=name, raw_value=0.0, standardised=value,
                              weight=0.01, evidence_tier="model")
            for name, value in loadings.items()
        },
    )


def test_a_diversified_basket_is_under_the_limit():
    basket = [
        _score("A", {f: 0.9 for f in _EXPOSURE_FACTORS}),
        _score("B", {f: -0.8 for f in _EXPOSURE_FACTORS}),
        _score("C", {f: 0.1 for f in _EXPOSURE_FACTORS}),
    ]
    exposure = _aggregate_exposure(basket, _EXPOSURE_FACTORS)
    assert all(v < _EXPOSURE_LIMIT for v in exposure.values())


def test_five_names_that_pass_every_pair_can_still_be_one_bet():
    """The case the existing checks cannot see."""
    basket = [_score(t, {f: 0.95 for f in _EXPOSURE_FACTORS}) for t in "ABCDE"]
    exposure = _aggregate_exposure(basket, _EXPOSURE_FACTORS)
    breached = [n for n, v in exposure.items() if v >= _EXPOSURE_LIMIT]
    assert set(breached) == set(_EXPOSURE_FACTORS)


def test_exposure_is_the_mean_loading():
    basket = [_score("A", {"beta_120": 1.0}), _score("B", {"beta_120": 0.0})]
    assert _aggregate_exposure(basket, ("beta_120",))["beta_120"] == pytest.approx(0.5)


def test_a_missing_factor_is_skipped_rather_than_counted_as_zero():
    """Counting an absent loading as zero would dilute a real concentration
    and quietly let the basket through."""
    basket = [_score("A", {"beta_120": 1.0}), _score("B", {})]
    assert _aggregate_exposure(basket, ("beta_120",))["beta_120"] == pytest.approx(1.0)


def test_an_unknown_factor_produces_no_entry():
    basket = [_score("A", {"beta_120": 1.0})]
    assert _aggregate_exposure(basket, ("not_a_factor",)) == {}


def test_an_empty_basket_is_not_an_exposure():
    assert _aggregate_exposure([], _EXPOSURE_FACTORS) == {}


def test_the_gate_only_ever_removes_names():
    """It downgrades to WATCHLIST. It can never promote or loosen."""
    import inspect

    from prosignal.stages import stage8_final_signal as s8
    source = inspect.getsource(s8.run)
    block = source[source.index("_aggregate_exposure"):]
    block = block[:block.index("if len(buys)")]
    assert "Decision.WATCHLIST" in block
    assert "buys.append" not in block
