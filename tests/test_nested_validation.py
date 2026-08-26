"""Nested validation must price selection, and must refuse to pretend.

A parameter chosen on the same data that reports the result is a fitted value,
and the number it produces is in-sample however many folds surround it. These
tests pin the construction rather than the Stage 6 result, so the infrastructure
stays honest when it is pointed at the next parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.harness import nested_band_search
from prosignal.validation.portfolio_sim import PortfolioParams

#: A realistic cross-section. Twenty-four names is below
#: `famamacbeth.MIN_CROSS_SECTION`, so every date was skipped as too thin to
#: support a cross-sectional regression and the search silently returned no
#: rows at all. The floor is right; the fixture was not.
SY = [f"S{i:02d}" for i in range(60)]


def _prices(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.011, n))) for s in SY}, index=idx)
    return {"close": close, "high": close * 1.01, "low": close * 0.99,
            "open": close.shift(1).fillna(close.iloc[0]),
            "atr": pd.DataFrame(np.full((n, len(SY)), 2.0), index=idx, columns=SY),
            "ma": close.rolling(50, min_periods=1).mean(),
            "adtv": pd.DataFrame(np.full((n, len(SY)), 5e9), index=idx, columns=SY)}


def _panel(prices):
    rng = np.random.default_rng(2)
    idx = list(prices["close"].index)
    rows = []
    for i in range(120, len(idx) - 70, 21):
        sig = rng.normal(size=len(SY))
        lab = 0.5 * sig + rng.normal(scale=1.1, size=len(SY))
        rows.append(pd.DataFrame({
            "date": idx[i], "symbol": SY, "signal_r": sig,
            "label": lab * 0.02,
            "label_rank": pd.Series(lab).rank(pct=True).to_numpy() * 2 - 1}))
    return pd.concat(rows, ignore_index=True)


def _make(entry, exit_):
    return PortfolioParams(
        capital=1e6, max_positions=8, risk_per_trade_pct=1.0,
        max_participation_of_adtv=0.01, stop_atr_multiple=2.5,
        min_stop_distance_pct=2.0, max_stop_distance_pct=15.0,
        invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
        horizon_sessions=63, entry_rank=entry, exit_rank=exit_,
        cost_bps_round_trip=86.0)


KW = dict(step_sessions=21, alpha=100.0, n_groups=6, n_test_groups=2,
          purge_sessions=63, embargo_sessions=21, min_train_rows=200)
GRID = [(5, 10), (8, 16), (10, 20)]


def test_it_reports_a_choice_and_an_outer_score_for_each_split():
    p = _prices()
    r = nested_band_search(_panel(p), ["signal_r"], p, _make, GRID, **KW)
    assert r.rows, "no outer split produced a result"
    for row in r.rows:
        assert (row["entry_rank"], row["exit_rank"]) in GRID
        assert "inner_sharpe" in row and "sharpe" in row


def test_the_inner_and_outer_scores_are_separate_numbers():
    """If they were the same, selection would be free and the whole
    construction pointless."""
    p = _prices()
    r = nested_band_search(_panel(p), ["signal_r"], p, _make, GRID, **KW)
    inner = np.array([x["inner_sharpe"] for x in r.rows])
    outer = np.array([x["sharpe"] for x in r.rows])
    assert not np.allclose(inner, outer), (
        "inner and outer Sharpe are identical, which means the outer block is "
        "not actually held out from the selection"
    )


def test_an_empty_grid_is_refused():
    p = _prices()
    with pytest.raises(ValueError, match="grid is empty"):
        nested_band_search(_panel(p), ["signal_r"], p, _make, [], **KW)


def test_a_negative_purge_is_refused_here_too():
    p = _prices()
    with pytest.raises(ValueError, match="non-negative"):
        nested_band_search(_panel(p), ["signal_r"], p, _make, GRID,
                           **{**KW, "purge_sessions": -1})


def test_chosen_counts_expose_a_scattered_winner():
    """The Stage 6 diagnostic: a scattered winner means the inner loop is
    reading noise, and the parameter does not matter."""
    p = _prices()
    r = nested_band_search(_panel(p), ["signal_r"], p, _make, GRID, **KW)
    counts = r.chosen_counts("entry_rank")
    assert sum(counts.values()) == len(r.rows)
    assert set(counts) <= {5, 8, 10}
