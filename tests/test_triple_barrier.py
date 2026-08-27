"""The label should encode the trade the engine would actually take.

The engine promises a stop and a holding period, then fitted against the return
h sessions later as though neither existed. That label is blind to the path: a
name that fell 20% and recovered by day 63 scored the same as one that drifted
up quietly, and the engine was stopped out of the first in week two. Fitting
against it teaches the model to like trades it would have closed at a loss.

Demonstrated on a constructed case:

    name        old horizon label     triple-barrier
    ROUNDTRIP        +2.00%           -13.31%, stopped on day 17
    UP              +28.49%           +16.01%, target on day 54
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.labels import (
    BarrierSpec, average_uniqueness, concurrency, triple_barrier,
)

IDX = pd.bdate_range("2024-01-01", periods=220)


def _series(values):
    return pd.DataFrame(values, index=IDX)


def _walk(seed, drift=0.0, vol=0.012, n=220):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(drift, vol, n))


# ------------------------------------------------------------- the barriers
def test_a_round_trip_is_a_stop_not_a_winner():
    """The case the horizon return hides."""
    up = _walk(1, drift=0.0)
    round_trip = _walk(2, drift=0.0)
    round_trip[71:95] *= np.linspace(1.0, 0.72, 24)
    round_trip[95:] *= 0.72
    round_trip[125:] = round_trip[70] * 1.02          # recovers by the horizon
    close = pd.DataFrame({"UP": up, "ROUNDTRIP": round_trip}, index=IDX)

    out = triple_barrier(close, 70, BarrierSpec(1.0, 0.75, 63, 60))
    assert out.loc["ROUNDTRIP", "side"] == -1.0
    assert out.loc["ROUNDTRIP", "ret"] < 0
    assert out.loc["ROUNDTRIP", "held"] < 63

    horizon_return = close.iloc[70 + 63]["ROUNDTRIP"] / close.iloc[70]["ROUNDTRIP"] - 1
    assert horizon_return > 0, "the old label would have called this a winner"


def test_a_bar_touching_both_barriers_counts_as_the_stop():
    """Daily bars cannot order intraday events, and assuming the favourable one
    inflates everything built on the label. Same convention as the backtest."""
    close = pd.DataFrame({"X": _walk(3)}, index=IDX)
    high = close * 5.0                     # every bar touches the upper
    low = close * 0.2                      # and the lower
    out = triple_barrier(close, 70, BarrierSpec(1.0, 0.75, 63, 60),
                         high=high, low=low)
    assert out.loc["X", "side"] == -1.0


def test_intraday_extremes_are_used_when_given():
    """A stop is not a close-only instrument. Testing on closes understates how
    often one is hit and flatters the label."""
    close = pd.DataFrame({"X": np.full(220, 100.0) + _walk(4) * 0.0 + _walk(4)},
                         index=IDX)
    spec = BarrierSpec(1.0, 0.75, 63, 60)
    closes_only = triple_barrier(close, 70, spec)
    with_wicks = triple_barrier(close, 70, spec,
                                high=close * 1.02, low=close * 0.98)
    # A wider range can only reach a barrier sooner, never later.
    a, b = closes_only.loc["X", "held"], with_wicks.loc["X", "held"]
    assert b <= a


def test_barriers_are_scaled_by_the_names_own_volatility():
    """A flat 8% means different things to a 1.2%-sigma large cap and a
    4%-sigma midcap."""
    calm = pd.DataFrame({"CALM": _walk(5, vol=0.004)}, index=IDX)
    wild = pd.DataFrame({"WILD": _walk(5, vol=0.030)}, index=IDX)
    spec = BarrierSpec(1.0, 0.75, 63, 60)
    c = triple_barrier(calm, 70, spec).loc["CALM", "ret"]
    w = triple_barrier(wild, 70, spec).loc["WILD", "ret"]
    if np.isfinite(c) and np.isfinite(w):
        assert abs(w) > abs(c), "the volatile name's barriers must sit wider"


def test_a_name_with_no_measured_volatility_is_refused_not_labelled():
    """Zero dispersion collapses both barriers onto the entry price, every bar
    touches both, and the both-touched rule books a stop at exactly zero -- a
    label manufactured out of a degenerate estimate."""
    close = pd.DataFrame({"FLAT": np.full(220, 100.0)}, index=IDX)
    out = triple_barrier(close, 70, BarrierSpec(1.0, 0.75, 63, 60))
    assert not np.isfinite(out.loc["FLAT", "ret"])


def test_the_label_never_reads_past_the_time_barrier():
    close = pd.DataFrame({"X": _walk(6)}, index=IDX)
    spec = BarrierSpec(1.0, 0.75, 63, 60)
    full = triple_barrier(close, 70, spec)
    # Truncating everything after the time barrier cannot change the answer.
    cut = triple_barrier(close.iloc[: 70 + 63 + 1], 70, spec)
    assert full.loc["X", "side"] == cut.loc["X", "side"]
    assert full.loc["X", "ret"] == pytest.approx(cut.loc["X", "ret"], abs=1e-12)


def test_the_label_never_reads_the_entry_bar_or_before():
    """Reading row i for the outcome is a full session of foresight."""
    import inspect

    from prosignal.features import labels

    src = inspect.getsource(labels.triple_barrier)
    assert "iloc[i + 1: end + 1]" in src


def test_a_time_barrier_that_cannot_fit_returns_nothing():
    close = pd.DataFrame({"X": _walk(7, n=80)}, index=IDX[:80])
    out = triple_barrier(close, 79, BarrierSpec(1.0, 0.75, 63, 60))
    assert out.empty or not np.isfinite(out.loc["X", "ret"])


# ----------------------------------------------------------- the weights
def test_two_identical_spans_each_weigh_half():
    u = average_uniqueness(np.array([0, 0, 50]), np.array([9, 9, 59]), 60)
    assert u[0] == pytest.approx(0.5)
    assert u[1] == pytest.approx(0.5)
    assert u[2] == pytest.approx(1.0)


def test_the_engines_own_geometry_is_about_forty_percent_unique():
    """A 63-session label sampled every 21 shares two thirds of its window."""
    t0 = np.arange(0, 210, 21)
    u = average_uniqueness(t0, t0 + 63, 300)
    assert 0.35 < u.mean() < 0.45


def test_concurrency_counts_live_labels_per_bar():
    c = concurrency(np.array([0, 2]), np.array([3, 5]), 6)
    assert c[0] == 1 and c[2] == 2 and c[5] == 1


def test_a_reversed_span_is_refused_rather_than_counted():
    u = average_uniqueness(np.array([5]), np.array([2]), 10)
    assert not np.isfinite(u[0])


# ------------------------------------------------------------- the fit
def test_uniform_weights_reproduce_the_unweighted_fit():
    from prosignal.features.linear import ridge_fit

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = x @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.3, size=300)
    a = ridge_fit(x, y, alpha=1.0)
    b = ridge_fit(x, y, alpha=1.0, weights=np.ones(300))
    assert np.allclose(a["coef"], b["coef"], atol=1e-10)


def test_weights_are_rescaled_so_the_penalty_keeps_its_meaning():
    """Down-weighting the sample would otherwise silently strengthen alpha."""
    from prosignal.features.linear import ridge_fit

    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 3))
    y = x @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.3, size=300)
    a = ridge_fit(x, y, alpha=1.0, weights=np.ones(300))
    b = ridge_fit(x, y, alpha=1.0, weights=np.full(300, 0.1))
    assert np.allclose(a["coef"], b["coef"], atol=1e-10)


def test_a_degenerate_weight_vector_falls_back_rather_than_dividing_by_zero():
    from prosignal.features.linear import ridge_fit

    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    out = ridge_fit(x, y, alpha=1.0, weights=np.zeros(100))
    assert np.isfinite(out["coef"]).all()


# ----------------------------------------------------------- the panel
def test_the_panel_carries_the_barrier_outcome_and_the_weights():
    from prosignal.features.crosssec import build_panel

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=600)
    cols = [f"S{i:02d}" for i in range(20)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.014, (600, 20)), axis=0),
        index=idx, columns=cols)
    tno = pd.DataFrame(1e8, index=idx, columns=cols)
    panel = build_panel(close, tno, horizon=63, step=21, min_names=5,
                        barriers=BarrierSpec(1.0, 0.75, 63, 60),
                        high=close * 1.008, low=close * 0.992)
    assert {"barrier_side", "held", "uniqueness", "t0", "t1"} <= set(panel.columns)
    assert panel["uniqueness"].between(0, 1).all()
    assert set(panel["barrier_side"].dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_uniqueness_is_measured_within_a_symbol_not_across_the_universe():
    """Thirty names on one date are thirty correlated observations, not a
    thirtieth of one. Pooling them into the concurrency count returned 0.014
    and would have thrown away almost the whole panel."""
    import inspect

    from prosignal.features import crosssec

    src = inspect.getsource(crosssec.build_panel)
    assert 'groupby("symbol"' in src


def test_without_a_barrier_spec_the_panel_keeps_the_horizon_label():
    """The old behaviour stays reachable, so the change is opt-in per caller."""
    from prosignal.features.crosssec import build_panel

    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2019-01-01", periods=600)
    cols = [f"S{i:02d}" for i in range(20)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.014, (600, 20)), axis=0),
        index=idx, columns=cols)
    tno = pd.DataFrame(1e8, index=idx, columns=cols)
    panel = build_panel(close, tno, horizon=63, step=21, min_names=5)
    assert panel["barrier_side"].isna().all()
    assert (panel["held"] == 63).all()


# --------------------------------------------- the geometry is NOT the label
def test_the_barrier_label_is_not_what_the_ranker_is_fitted_on():
    """The barrier geometry was calibrated, measured, and then REMOVED from the
    training label. This assertion previously read `is True`; it encoded a
    decision that has since been reversed, and the reversal is the point.

    The calibration was real. Measured on the real universe over 91 panel dates:

        upper/lower   target   stop  timeout   uniqueness   median hold
          2.0/1.5       14%     10%     76%      0.404          63
          1.0/0.75      37%     36%     27%      0.576          34   <- chosen

    At 2.0/1.5 three quarters of labels time out and the label collapses back to
    the horizon return it exists to replace. That table is why 1.0/0.75 was
    chosen and it is still true.

    What the calibration could not see is that the label's MAGNITUDE is
    3 x stop width for a winner and -1 x stop width for a loser, and the stop is
    2.5 x ATR -- so for 91% of rows the label's size IS the name's volatility.
    Fitting a cross-sectional ranker on it prices volatility and de-prices
    momentum: mom fell to t -0.14 while lottery reached t -6.3. Against the real
    63-session forward return the shipped configuration scored rank IC +0.0262
    (t +1.00) with a top-decile excess of -0.92%; with the label switched off,
    +0.0668 (t +4.15) and +1.00%.

    In Lopez de Prado (2018) the barrier label is a CLASSIFICATION target and
    its volatility scaling exists to make that classification comparable across
    names. Using the scaled magnitude as a ranking target puts the
    heteroskedasticity back.

    The geometry itself is untouched and still enforced where it belongs:
    stage7_risk places the stop, stage6_entry holds the exit band, and
    exits.resolve_exits scores the outcome. The sigma parameters stay in the
    config, inert, documenting what was tried.
    """
    from prosignal.config.loader import load_config

    lab = load_config().params.stage4_core_score.labels
    assert lab.triple_barrier is False
    # Inert while triple_barrier is false, and kept so the calibrated geometry
    # stays on the record rather than being silently dropped.
    assert lab.upper_sigma == pytest.approx(1.0)
    assert lab.lower_sigma == pytest.approx(0.75)
    assert lab.uniqueness_weighting is True


def test_wider_barriers_time_out_more_often():
    """The property behind the calibration table, not just the numbers in it."""
    rng = np.random.default_rng(11)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"S{i:02d}" for i in range(60)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0, 0.014, (n, 60)), axis=0),
        index=idx, columns=cols)
    tight = triple_barrier(close, 200, BarrierSpec(0.8, 0.6, 63, 60))
    wide = triple_barrier(close, 200, BarrierSpec(2.5, 2.0, 63, 60))
    assert (wide["side"] == 0).mean() > (tight["side"] == 0).mean()


def test_the_labeller_is_fast_enough_to_refit_with():
    """The per-symbol Python loop this replaced ran ~3,500 symbols x ~90 dates
    and did not finish in twelve minutes."""
    import time

    rng = np.random.default_rng(12)
    n, k = 400, 800
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.014, (n, k)), axis=0),
        index=idx, columns=[f"S{i:04d}" for i in range(k)])
    start = time.time()
    triple_barrier(close, 200, BarrierSpec(1.0, 0.75, 63, 60),
                   high=close * 1.008, low=close * 0.992)
    assert time.time() - start < 1.0, "one date must cost milliseconds, not seconds"
