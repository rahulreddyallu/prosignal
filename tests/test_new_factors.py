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
    bench = close.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")
    return close, turnover, bench


def test_the_three_factors_are_registered():
    for name in ("prox_52w", "max5_21", "resid_mom"):
        assert name in FEATURES


def test_prox_52w_is_zero_at_the_high_and_negative_below_it(frames):
    """Measured AS OF 21 SESSIONS BACK, not today.

    This test used to set today's close to a new high and expect 0.0. That
    encoded the George & Hwang (2004) definition, which reads the current
    price. The factor now skips the most recent 21 sessions -- the window
    `resid_reversal` prices with the opposite sign -- so the reference bar is
    `hist.iloc[-22]` and the window is the 252 sessions ending there.
    """
    close, turnover, bench = frames
    close = close.copy()
    window = close.iloc[-273:-21]
    close.iloc[-22, 0] = window.iloc[:, 0].max() * 1.5    # new high, 21 back
    close.iloc[-22, 1] = window.iloc[:, 1].max() * 0.5    # well below, 21 back
    f = _features_at(close, turnover, len(close) - 1, bench)
    assert f.loc["S0", "prox_52w"] == pytest.approx(0.0, abs=1e-12)
    assert f.loc["S1", "prox_52w"] < -0.3


def test_prox_52w_ignores_the_last_21_sessions(frames):
    """The property the skip exists for, pinned directly.

    Using today's close put the most recent month inside prox_52w, which is
    exactly the window `resid_reversal` measures with the opposite sign, so
    momentum and reversal partially cancelled through this one factor: measured
    within date the two correlated +0.378 as shipped and -0.029 with the skip.

    A move inside the skipped window must therefore not move the factor at all.
    """
    close, turnover, bench = frames
    base = _features_at(close, turnover, len(close) - 1, bench)["prox_52w"]

    spiked = close.copy()
    # Triple every price across the whole skipped window, including today.
    spiked.iloc[-21:, :] *= 3.0
    after = _features_at(spiked, turnover, len(spiked) - 1, bench)["prox_52w"]

    pd.testing.assert_series_equal(base, after)


def test_max5_21_rises_with_a_lottery_like_spike(frames):
    close, turnover, bench = frames
    base = _features_at(close, turnover, len(close) - 1, bench).loc["S2", "max5_21"]
    spiked = close.copy()
    spiked.iloc[-3, 2] *= 1.18
    after = _features_at(spiked, turnover, len(spiked) - 1, bench).loc["S2", "max5_21"]
    assert after > base


def _beta_panel(idx, cols, seed, betas, drifts, drift_window=None):
    """A trending market plus a per-name residual, with per-name beta and drift.

    Two things this fixture has to respect, both learned by getting them wrong:

    * The benchmark is the equal-weight mean of these same names, so a drift
      given to EVERY name is MARKET drift and never reaches the residual. Only
      cross-sectional differences are residual.
    * The regression carries an intercept, so a drift constant across the whole
      estimation window is absorbed into alpha and leaves no residual momentum.
      That is the construction working as intended -- residual momentum prices
      RECENT residual out-performance against the name's own long-run norm --
      so a drift meant to register must be confined to the formation window.
      ``drift_window`` is the number of trailing sessions the drift applies to.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0020, 0.010, size=len(idx))
    n = len(idx)
    data = {}
    for k, c in enumerate(cols):
        eps = rng.normal(0.0, 0.008, size=n)
        if drifts[k]:
            lo = 0 if drift_window is None else max(0, n - int(drift_window))
            eps[lo:] += drifts[k]
        data[c] = 100.0 * np.exp(np.cumsum(betas[k] * steps + eps))
    return pd.DataFrame(data, index=idx)


def test_resid_mom_strips_the_market_component(frames):
    """The claim, tested: in a trending market, RAW momentum tracks beta and
    residual momentum does not.

    REWRITTEN, because the old version never tested this. It built every stock
    as an exact multiple of the index, which makes the residual identically
    zero -- so the old unstandardised factor returned ~0 for any input and the
    assertion `abs().max() < 0.05` could not fail for the stated reason. Under
    the corrected construction that case is 0/0 and is pinned separately.
    """
    close, turnover, _ = frames
    cols = list(close.columns)
    n = len(cols)
    betas = np.linspace(0.5, 1.6, n)
    px = _beta_panel(close.index, cols, seed=5, betas=betas, drifts=np.zeros(n))
    bench = px.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")
    f = _features_at(px, turnover, len(px) - 1, bench)

    raw = (px.iloc[-22] / px.iloc[-253] - 1.0).reindex(f.index)
    assert raw.mean() > 0.15, "the market must actually trend for this to bite"
    c_raw = float(pd.Series(betas, index=cols).reindex(f.index).corr(raw, method="spearman"))
    c_res = float(pd.Series(betas, index=cols).reindex(f.index)
                  .corr(f["resid_mom"], method="spearman"))
    assert c_raw > 0.8, f"raw momentum should track beta in a trending market; got {c_raw:+.2f}"
    assert abs(c_res) < 0.5, (
        f"residual momentum still tracks beta at rho {c_res:+.2f}; the market "
        f"component is not being stripped")


def test_resid_mom_detects_residual_drift(frames):
    """The other half. A guard that only checks the null is not a guard.

    PAIRED, deliberately. A standardised cumulative residual is a t-like
    statistic whose cross-sectional spread over a 231-session formation window
    is O(sqrt(231)) ~ 15, so an unpaired "the drifting name should top the
    field" assertion fights that noise and tests the seed rather than the
    factor. Two panels identical but for one name's formation-window drift
    isolate the effect exactly.
    """
    close, turnover, _ = frames
    cols = list(close.columns)
    n = len(cols)

    def build(drift):
        d = np.zeros(n)
        d[0] = drift
        # Confined to the formation window: a drift spanning the whole
        # estimation window is the name's alpha and the intercept removes it.
        px = _beta_panel(close.index, cols, seed=5, betas=np.ones(n), drifts=d,
                         drift_window=252)
        bench = px.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")
        return _features_at(px, turnover, len(px) - 1, bench)["resid_mom"]

    flat, up = build(0.0), build(0.0015)
    moved = float(up.loc[cols[0]] - flat.loc[cols[0]])
    others = (up.drop(cols[0]) - flat.drop(cols[0])).abs().max()
    assert moved > 5.0, (
        f"residual out-performance of 0.15%/session over the formation window "
        f"moved resid_mom by only {moved:+.2f}; the factor is not detecting it")
    assert moved > 3.0 * float(others), (
        f"the drifting name moved {moved:+.2f} while the untouched names moved "
        f"up to {others:.2f}; the response is not specific to the name that "
        f"out-performed")


def test_resid_mom_is_undefined_for_an_exact_multiple_of_the_index(frames):
    """0/0 is not a number. A name that is exactly a linear function of the
    market has no residual to accumulate and no dispersion to divide by."""
    close, turnover, _ = frames
    idx = close.index
    rng = np.random.default_rng(5)
    mkt = 100.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.011, size=len(idx))))
    px = pd.DataFrame({c: mkt * (1.0 + 0.05 * k)
                       for k, c in enumerate(close.columns)}, index=idx)
    bench = px.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")
    f = _features_at(px, turnover, len(px) - 1, bench)
    assert f["resid_mom"].isna().all(), (
        f"expected NaN for a degenerate residual; got "
        f"{f['resid_mom'].abs().max():.3g}")


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
