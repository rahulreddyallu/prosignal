"""The three factors added after the survivorship audit.

Each replaces information the liquidity family was only appearing to supply:
on a point-in-time universe amihud and turnover_ratio carry no signal, while
these three survive with the sign their source paper reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.crosssec import FEATURES, _features_at


@pytest.fixture
def frames():
    n, m = 320, 12
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, size=(n, m)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(m)],
    )
    turnover = pd.DataFrame(rng.uniform(1e7, 5e7, size=(n, m)), index=idx, columns=close.columns)
    bench = close.mean(axis=1).pct_change().to_numpy("float64")
    return close, turnover, bench


def test_the_three_factors_are_registered():
    for name in ("prox_52w", "max5_21", "resid_mom"):
        assert name in FEATURES


def test_prox_52w_is_zero_at_the_high_and_negative_below_it(frames):
    close, turnover, bench = frames
    close = close.copy()
    close.iloc[-1, 0] = close.iloc[-252:, 0].max() * 1.5   # new high
    close.iloc[-1, 1] = close.iloc[-252:, 1].max() * 0.5   # well below
    f = _features_at(close, turnover, len(close) - 1, bench)
    assert f.loc["S0", "prox_52w"] == pytest.approx(0.0, abs=1e-12)
    assert f.loc["S1", "prox_52w"] < -0.3


def test_max5_21_rises_with_a_lottery_like_spike(frames):
    close, turnover, bench = frames
    base = _features_at(close, turnover, len(close) - 1, bench).loc["S2", "max5_21"]
    spiked = close.copy()
    spiked.iloc[-3, 2] *= 1.18
    after = _features_at(spiked, turnover, len(spiked) - 1, bench).loc["S2", "max5_21"]
    assert after > base


def test_resid_mom_strips_the_market_component(frames):
    """A stock that is pure beta with no residual should score near zero even
    when raw momentum is large."""
    close, turnover, _ = frames
    idx = close.index
    rng = np.random.default_rng(5)
    # One trending market path with genuine dispersion, and every stock a pure
    # multiple of it: raw momentum is large, residual momentum should be ~0.
    steps = rng.normal(0.0015, 0.011, size=len(idx))
    mkt = 100.0 * np.exp(np.cumsum(steps))
    close = pd.DataFrame({c: mkt * (1.0 + 0.05 * k) for k, c in enumerate(close.columns)}, index=idx)
    bench = close.mean(axis=1).pct_change().to_numpy("float64")
    f = _features_at(close, turnover, len(close) - 1, bench)
    assert (close.iloc[-22] / close.iloc[-253] - 1.0).mean() > 0.15
    assert f["resid_mom"].abs().max() < 0.05


def test_resid_mom_is_present_even_when_beta_is_undefined(frames):
    """A market with no dispersion leaves the column NaN, never missing."""
    close, turnover, _ = frames
    flat = np.zeros(len(close))
    f = _features_at(close, turnover, len(close) - 1, flat)
    assert "resid_mom" in f.columns
    assert f["resid_mom"].isna().all()


def test_no_factor_reads_beyond_the_decision_bar(frames):
    """Perturbing prices strictly after bar i must not move any feature at i."""
    close, turnover, bench = frames
    i = len(close) - 40
    before = _features_at(close, turnover, i, bench[: i + 1])
    tampered = close.copy()
    tampered.iloc[i + 1 :] *= 3.0
    after = _features_at(tampered, turnover, i, bench[: i + 1])
    for name in ("prox_52w", "max5_21", "resid_mom"):
        pd.testing.assert_series_equal(before[name], after[name], check_names=False)
