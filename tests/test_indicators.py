"""Indicator library.

Expected values here are hand-computed or derived from a property that must
hold, never "whatever the code returned the first time". A test that asserts
the current output is not a test, it is a snapshot, and it will happily lock in
a lookahead bug forever.

The lookahead tests at the bottom are the important ones. They assert the
property directly: truncating the series must not change any value that was
already computable. Any indicator that peeks fails it immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.indicators import (
    annualised_log_slope,
    atr,
    distance_from_ma_atr,
    distance_from_ma_pct,
    ema,
    golden_cross_state,
    is_above,
    log_returns,
    max_drawdown,
    momentum_skip,
    ols_slope,
    percentile_of_last,
    rank_to_unit_interval,
    rate_of_change_pct,
    realised_volatility,
    robust_zscore,
    rolling_annualised_slope,
    rolling_percentile,
    sector_neutralise,
    sigma_move,
    simple_returns,
    sma,
    sma_atr,
    spearman_pairs,
    standardise,
    trailing_return,
    trend_quality,
    true_range,
    wilder_atr,
    wilder_ma,
    winsorise,
    zscore,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-01", periods=n)


def _series(values) -> pd.Series:
    values = list(values)
    return pd.Series(values, index=_dates(len(values)), dtype="float64")


# =============================================================================
# returns
# =============================================================================


def test_simple_returns_hand_computed():
    s = _series([100, 110, 99])
    out = simple_returns(s)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.10)
    assert out.iloc[2] == pytest.approx(-0.10)


def test_log_returns_hand_computed():
    s = _series([100, 100 * np.e])
    out = log_returns(s)
    assert out.iloc[1] == pytest.approx(1.0)


def test_log_returns_reject_non_positive_prices_as_nan():
    """A zero price is bad data. -inf would poison every downstream mean."""
    s = _series([100, 0, 50])
    out = log_returns(s)
    assert np.isnan(out.iloc[1])
    assert not np.isinf(out).any()


def test_trailing_return_returns_none_not_zero_when_short():
    s = _series([100, 101, 102])
    assert trailing_return(s, sessions=10) is None
    assert trailing_return(s, sessions=2) == pytest.approx(0.02)


def test_momentum_skip_is_the_12_1_construction():
    """252-session return measured to 21 sessions ago, not to today."""
    n = 300
    prices = _series(np.linspace(100, 400, n))
    lookback, skip = 252, 21

    got = momentum_skip(prices, lookback, skip)

    end = float(prices.iloc[n - 1 - skip])
    start = float(prices.iloc[n - 1 - skip - lookback])
    assert got == pytest.approx(end / start - 1.0)


def test_momentum_skip_actually_excludes_the_recent_month():
    """The last 21 sessions must not influence the value at all."""
    n = 300
    base = np.linspace(100, 200, n)
    calm = _series(base)

    spiked = base.copy()
    spiked[-21:] *= 3.0  # violent recent month
    spiked_series = _series(spiked)

    assert momentum_skip(calm, 252, 21) == pytest.approx(
        momentum_skip(spiked_series, 252, 21)
    )


def test_momentum_skip_needs_lookback_plus_skip_plus_one():
    assert momentum_skip(_series(np.arange(273, dtype=float) + 1), 252, 21) is None
    assert momentum_skip(_series(np.arange(274, dtype=float) + 1), 252, 21) is not None


def test_realised_volatility_annualisation():
    rng = np.random.default_rng(0)
    prices = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    daily = realised_volatility(prices, window=252, annualise=False)
    annual = realised_volatility(prices, window=252, annualise=True)
    assert annual.iloc[-1] == pytest.approx(daily.iloc[-1] * np.sqrt(252))
    assert annual.iloc[-1] == pytest.approx(0.01 * np.sqrt(252), rel=0.2)


def test_max_drawdown_hand_computed():
    s = _series([100, 120, 60, 80])
    assert max_drawdown(s) == pytest.approx(-0.5)  # 120 -> 60
    assert max_drawdown(_series([100])) is None


# =============================================================================
# moving averages
# =============================================================================


def test_sma_is_nan_until_the_window_is_full():
    s = _series([1, 2, 3, 4, 5])
    out = sma(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_uses_recursive_form_not_adjusted():
    """adjust=False: EMA_t = a*x_t + (1-a)*EMA_{t-1}, seeded on the SMA-free path."""
    s = _series([10, 20, 30, 40, 50])
    span = 3
    alpha = 2.0 / (span + 1)
    out = ema(s, span=span, min_periods=1)

    expected = 10.0
    for value in [20, 30, 40, 50]:
        expected = alpha * value + (1 - alpha) * expected
    assert out.iloc[-1] == pytest.approx(expected)


def test_wilder_ma_differs_from_same_period_ema():
    """Wilder(n) == EMA(span=2n-1). Confusing them mis-sizes every stop."""
    s = _series(np.linspace(10, 100, 60))
    n = 14
    wilder = wilder_ma(s, n)
    equivalent = ema(s, span=2 * n - 1, min_periods=n)
    assert wilder.iloc[-1] == pytest.approx(equivalent.iloc[-1], rel=1e-9)
    assert wilder.iloc[-1] != pytest.approx(ema(s, span=n).iloc[-1])


def test_distance_from_ma_pct_hand_computed():
    price = _series([110.0])
    ma = _series([100.0])
    assert distance_from_ma_pct(price, ma).iloc[0] == pytest.approx(10.0)


def test_distance_from_ma_atr_normalises_by_volatility():
    """Same 10% extension is 5 ATRs for a calm stock, 1 ATR for a volatile one."""
    price, ma = _series([110.0]), _series([100.0])
    calm = distance_from_ma_atr(price, ma, _series([2.0]))
    wild = distance_from_ma_atr(price, ma, _series([10.0]))
    assert calm.iloc[0] == pytest.approx(5.0)
    assert wild.iloc[0] == pytest.approx(1.0)


def test_is_above_treats_unknown_as_not_above():
    price = _series([100, 100, 100])
    ma = pd.Series([np.nan, 90.0, 110.0], index=price.index)
    out = is_above(price, ma)
    assert out.tolist() == [False, True, False]
    assert out.dtype == bool


def test_golden_cross_state_is_a_state_not_an_event():
    fast = _series([1, 3, 3, 1])
    slow = _series([2, 2, 2, 2])
    assert golden_cross_state(fast, slow).tolist() == [-1, 1, 1, -1]


# =============================================================================
# ATR
# =============================================================================


def test_true_range_first_bar_is_nan_not_the_bare_range():
    high, low, close = _series([12, 14]), _series([8, 11]), _series([10, 13])
    tr = true_range(high, low, close)
    assert np.isnan(tr.iloc[0])


def test_true_range_captures_a_gap_up():
    """H-L is 2, but the gap from the previous close makes the true range 5."""
    high = _series([10, 15])
    low = _series([9, 13])
    close = _series([10, 14])
    tr = true_range(high, low, close)
    assert tr.iloc[1] == pytest.approx(5.0)  # |15 - 10| beats 15-13 = 2


def test_true_range_captures_a_gap_down():
    high = _series([10, 7])
    low = _series([9, 5])
    close = _series([10, 6])
    tr = true_range(high, low, close)
    assert tr.iloc[1] == pytest.approx(5.0)  # |5 - 10|


def test_sma_atr_hand_computed():
    high = _series([10, 11, 12, 13])
    low = _series([9, 10, 11, 12])
    close = _series([10, 11, 12, 13])
    # Each bar: range = H-L = 1, and |H - prev_close| = 1 too, so TR = 1.
    # TR = [nan, 1, 1, 1]  ->  2-period SMA = 1.0
    out = sma_atr(high, low, close, period=2)
    assert out.iloc[-1] == pytest.approx(1.0)

    # Now widen the gap so the previous close, not the range, sets the TR.
    gapped_high = _series([10, 20, 21, 22])
    gapped_low = _series([9, 19, 20, 21])
    gapped_close = _series([10, 20, 21, 22])
    # TR[1] = |20 - 10| = 10 (range is only 1)
    assert true_range(gapped_high, gapped_low, gapped_close).iloc[1] == pytest.approx(10.0)


def test_wilder_and_sma_atr_differ_but_agree_in_scale():
    rng = np.random.default_rng(1)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 200)))
    high = close + 1.0
    low = close - 1.0
    w = wilder_atr(high, low, close, 14).iloc[-1]
    s = sma_atr(high, low, close, 14).iloc[-1]
    assert w != pytest.approx(s)
    assert w == pytest.approx(s, rel=0.5)


def test_atr_dispatch_rejects_unknown_method():
    high, low, close = _series([2, 3]), _series([1, 2]), _series([1.5, 2.5])
    with pytest.raises(ValueError, match="unknown ATR method"):
        atr(high, low, close, method="exponential-ish")


# =============================================================================
# trend
# =============================================================================


def test_ols_slope_on_a_perfect_line():
    slope, r2 = ols_slope(np.array([0.0, 1.0, 2.0, 3.0]))
    assert slope == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_ols_slope_of_flat_series_is_zero_with_undefined_fit():
    slope, r2 = ols_slope(np.array([5.0, 5.0, 5.0]))
    assert slope == pytest.approx(0.0)
    assert np.isnan(r2)


def test_annualised_log_slope_recovers_a_known_growth_rate():
    """A series compounding at exactly 20%/yr must report 0.20."""
    n = 252
    target = 0.20
    daily = target / 252
    prices = _series(100 * np.exp(daily * np.arange(n)))
    assert annualised_log_slope(prices, window=n) == pytest.approx(target, rel=1e-6)


def test_annualised_log_slope_is_scale_invariant():
    """Log slope must not depend on whether the stock trades at 150 or 15,000."""
    n = 100
    base = np.exp(np.linspace(0, 0.5, n))
    cheap = annualised_log_slope(_series(150 * base), window=n)
    dear = annualised_log_slope(_series(15000 * base), window=n)
    assert cheap == pytest.approx(dear)


def test_annualised_log_slope_returns_none_when_window_not_full():
    assert annualised_log_slope(_series([1, 2, 3]), window=10) is None


def test_trend_quality_separates_clean_from_noisy_trends():
    n = 120
    rng = np.random.default_rng(2)
    drift = np.linspace(0, 0.5, n)
    clean = _series(100 * np.exp(drift))
    noisy = _series(100 * np.exp(drift + rng.normal(0, 0.15, n)))
    assert trend_quality(clean, n) == pytest.approx(1.0, abs=1e-9)
    assert trend_quality(noisy, n) < 0.9


def test_rolling_slope_matches_the_scalar_version_at_the_end():
    n = 120
    prices = _series(100 * np.exp(np.linspace(0, 0.4, n)))
    rolling = rolling_annualised_slope(prices, window=63)
    scalar = annualised_log_slope(prices, window=63)
    assert rolling.iloc[-1] == pytest.approx(scalar)


# =============================================================================
# rolling statistics
# =============================================================================


def test_rolling_percentile_extremes():
    """Midpoint tie convention: an all-time high is 90, not 100, for n=5."""
    s = _series([1, 2, 3, 4, 5])
    out = rolling_percentile(s, window=5)
    # 4 below, 1 equal (itself) -> (4 + 0.5) / 5 = 90%
    assert out.iloc[-1] == pytest.approx(90.0)
    falling = _series([5, 4, 3, 2, 1])
    # 0 below, 1 equal -> 0.5 / 5 = 10%
    assert rolling_percentile(falling, window=5).iloc[-1] == pytest.approx(10.0)


def test_flat_series_percentile_is_50_not_100():
    """Regression: the bug that made a dead-flat India VIX read as maximum vol.

    Under naive "count values <= current" ranking every element of a constant
    series ties, so the rank is 100 and the regime engine would sit in its most
    defensive volatility bucket during the calmest possible market. The
    midpoint convention returns 50: today is exactly typical of its own history.
    """
    flat = _series([12.0] * 300)
    assert rolling_percentile(flat, window=252).iloc[-1] == pytest.approx(50.0)
    assert percentile_of_last(flat, 252) == pytest.approx(50.0)


def test_percentile_midpoint_handles_partial_ties():
    s = _series([1, 1, 1, 2, 1])
    # window of 5, current = 1: 0 below, 4 equal -> (0 + 2.0)/5 = 40%
    assert rolling_percentile(s, window=5).iloc[-1] == pytest.approx(40.0)


def test_rolling_percentile_is_nan_until_window_full():
    out = rolling_percentile(_series([1, 2, 3]), window=3)
    assert out.iloc[:2].isna().all()
    assert not np.isnan(out.iloc[2])


def test_percentile_of_last_matches_rolling_version():
    rng = np.random.default_rng(3)
    s = _series(rng.normal(20, 5, 300))
    assert percentile_of_last(s, 252) == pytest.approx(
        rolling_percentile(s, 252).iloc[-1]
    )


def test_percentile_of_last_returns_none_when_short():
    assert percentile_of_last(_series([1, 2, 3]), window=252) is None


def test_sigma_move_excludes_the_observation_being_judged():
    """A huge tick must not inflate the sigma it is measured against."""
    values = [100.0] * 60 + [100.5] * 60  # tiny variation, then one huge jump
    s = _series(values + [500.0])
    result = sigma_move(s, window=100)
    assert result is not None
    assert result > 10  # unmistakably an outlier


def test_sigma_move_returns_none_on_zero_variance():
    s = _series([100.0] * 50 + [100.0])
    assert sigma_move(s, window=30) is None


def test_rate_of_change_pct_hand_computed():
    s = _series([100, 105, 110])
    assert rate_of_change_pct(s, periods=2) == pytest.approx(10.0)
    assert rate_of_change_pct(s, periods=99) is None


# =============================================================================
# cross-section
# =============================================================================


def test_winsorise_clips_without_dropping():
    s = pd.Series([1, 2, 3, 4, 1000], dtype="float64")
    # np.percentile interpolates; with 5 points the 75th percentile lands
    # exactly on the 4th element, so the clip bound is an unambiguous 4.0.
    out = winsorise(s, 0, 75)
    assert out.size == 5, "winsorising clips, it never drops names"
    assert out.iloc[-1] == pytest.approx(4.0)
    assert out.iloc[0] == pytest.approx(1.0)


def test_winsorising_before_zscore_stops_one_stock_dominating():
    """The core reason order matters in the cross-sectional pipeline."""
    s = pd.Series([1, 2, 3, 4, 5, 1000], dtype="float64")
    raw = zscore(s)
    tamed = zscore(winsorise(s, 1, 99))
    # Raw: the outlier crushes everyone else toward a single value.
    assert raw.iloc[:5].std() < 0.1
    assert tamed.iloc[:5].std() > raw.iloc[:5].std()


def test_zscore_of_constant_cross_section_is_zero_not_inf():
    out = zscore(pd.Series([5.0, 5.0, 5.0]))
    assert (out == 0).all()
    assert np.isfinite(out).all()


def test_robust_zscore_resists_an_outlier():
    s = pd.Series([1, 2, 3, 4, 5, 1000], dtype="float64")
    assert abs(robust_zscore(s).iloc[0]) > abs(zscore(s).iloc[0])


def test_rank_to_unit_interval_spans_exactly_zero_to_one():
    out = rank_to_unit_interval(pd.Series([10, 20, 30, 40], dtype="float64"))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_rank_of_single_name_is_half():
    out = rank_to_unit_interval(pd.Series([42.0]))
    assert out.iloc[0] == pytest.approx(0.5)


def test_rank_handles_ties_by_averaging():
    out = rank_to_unit_interval(pd.Series([1, 1, 2], dtype="float64"))
    assert out.iloc[0] == pytest.approx(out.iloc[1])


def test_standardise_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown standardisation"):
        standardise(pd.Series([1.0, 2.0]), method="vibes")


def test_sector_neutralise_removes_a_pure_sector_bet():
    """Momentum concentrated in one sector must not survive demeaning."""
    values = pd.Series(
        {"A": 10.0, "B": 11.0, "C": 12.0, "X": 1.0, "Y": 2.0, "Z": 3.0}
    )
    sectors = {"A": "Metals", "B": "Metals", "C": "Metals",
               "X": "FMCG", "Y": "FMCG", "Z": "FMCG"}
    out = sector_neutralise(values, sectors)
    # Within each sector the values now centre on zero.
    assert out[["A", "B", "C"]].mean() == pytest.approx(0.0)
    assert out[["X", "Y", "Z"]].mean() == pytest.approx(0.0)
    # And the previously-dominant Metals names no longer outrank FMCG.
    assert out["C"] == pytest.approx(out["Z"])


def test_sector_neutralise_skips_sectors_that_are_too_small():
    """Demeaning a 2-stock sector manufactures a signal from arithmetic."""
    values = pd.Series({"A": 10.0, "B": 20.0, "P": 1.0, "Q": 2.0, "R": 3.0})
    sectors = {"A": "Tiny", "B": "Tiny", "P": "Big", "Q": "Big", "R": "Big"}
    out = sector_neutralise(values, sectors, min_sector_size=3)
    assert out["A"] == pytest.approx(10.0)  # untouched
    assert out["P"] == pytest.approx(-1.0)  # demeaned


def test_spearman_pairs_detects_a_monotone_relationship():
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 4, 3, 2, 1]})
    pairs = spearman_pairs(frame)
    assert pairs["a|b"] == pytest.approx(1.0)
    assert pairs["a|c"] == pytest.approx(-1.0)


def test_spearman_pair_keys_are_order_independent():
    pairs = spearman_pairs(
        pd.DataFrame({"z": [1, 2, 3, 4, 5], "a": [5, 4, 3, 2, 1]})
    )
    assert "a|z" in pairs
    assert pairs["a|z"] == pytest.approx(-1.0)


# =============================================================================
# THE IMPORTANT ONES: no lookahead, anywhere
# =============================================================================


@pytest.mark.parametrize(
    "name,fn",
    [
        ("sma", lambda s: sma(s, 20)),
        ("ema", lambda s: ema(s, 20)),
        ("wilder_ma", lambda s: wilder_ma(s, 14)),
        ("simple_returns", lambda s: simple_returns(s)),
        ("log_returns", lambda s: log_returns(s)),
        ("realised_volatility", lambda s: realised_volatility(s, 20)),
        ("rolling_percentile", lambda s: rolling_percentile(s, 20)),
        ("rolling_slope", lambda s: rolling_annualised_slope(s, 20)),
    ],
)
def test_indicator_does_not_peek_into_the_future(name, fn):
    """Truncating the series must not change any already-computable value.

    This is the property that matters. An indicator using future data produces
    different values for the same date depending on how much history follows
    it -- which is invisible in a backtest and fatal in live trading.
    """
    rng = np.random.default_rng(7)
    full = _series(100 + np.cumsum(rng.normal(0, 1, 200)))
    truncated = full.iloc[:150]

    full_out = fn(full).iloc[:150]
    trunc_out = fn(truncated)

    pd.testing.assert_series_equal(
        full_out, trunc_out, check_names=False, rtol=1e-12, atol=1e-12
    )


def test_atr_does_not_peek_into_the_future():
    rng = np.random.default_rng(8)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 200)))
    high, low = close + 1.5, close - 1.5

    full = wilder_atr(high, low, close, 14).iloc[:150]
    trunc = wilder_atr(high.iloc[:150], low.iloc[:150], close.iloc[:150], 14)
    pd.testing.assert_series_equal(full, trunc, check_names=False, rtol=1e-12)


def test_scalar_helpers_respect_an_as_of_cutoff():
    """as_of must behave exactly like truncating the series."""
    rng = np.random.default_rng(9)
    s = _series(100 + np.cumsum(rng.normal(0, 1, 400)))
    cutoff = s.index[300]

    assert trailing_return(s, 21, as_of=cutoff) == pytest.approx(
        trailing_return(s.loc[:cutoff], 21)
    )
    assert momentum_skip(s, 252, 21, as_of=cutoff) == pytest.approx(
        momentum_skip(s.loc[:cutoff], 252, 21)
    )
    assert percentile_of_last(s, 252, as_of=cutoff) == pytest.approx(
        percentile_of_last(s.loc[:cutoff], 252)
    )
    assert annualised_log_slope(s, 63, as_of=cutoff) == pytest.approx(
        annualised_log_slope(s.loc[:cutoff], 63)
    )
