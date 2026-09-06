"""The headline performance statistic must not move with a sizing knob.

WHAT THIS PINS. `mean_excess` is `mean(book - benchmark)` against a benchmark
that is fully invested, while the book is not: sizing is
`risk_budget / risk_per_share`, so the deployed fraction is a pure function of
the risk budget and the stop width. Measured on the shipped configuration over
87 periods, deployed capital is 0.218 and the equal-weight eligible universe
returned +22.1% a year, so 17.3 of the 18.6 points of reported
"underperformance" are the cash the book is not holding.

That made every ablation ever selected on `mean_excess` unsafe. Holding the
ranking, the names, the cost model and every other setting fixed and moving
ONLY `risk_per_trade_pct` from 1% to 4.6%:

    risk/trade   deployed   mean_excess   alpha_ann   alpha_on_deployed
        1%         0.218      -20.98%      -1.32%          -6.05%
        2%         0.434      -18.73%      -3.05%          -7.03%
        3%         0.633      -16.51%      -4.60%          -7.27%
      4.6%         0.824      -11.49%      -3.81%          -4.62%

`mean_excess` improves by nine points for a reason that has nothing to do with
the signal. Note that the RAW alpha is not invariant either: writing
`r = dep * r_d` gives `beta = dep*beta_d` and `alpha = dep*alpha_d`, so alpha
merely loses the additive cash-drag term while staying proportional to
deployment. The figure that does not move is `alpha / dep` -- the alpha of the
capital actually at risk -- and that is the headline these tests pin.

So the tests below assert the ORDERING PROPERTY rather than any particular
number: raw excess must respond to leverage and the headline must not. A future
edit that reinstates the raw figure as the headline fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.portfolio_sim import (
    PortfolioParams,
    _benchmark_stats,
    simulate,
)

SYMBOLS = [f"S{i:02d}" for i in range(24)]


def _params(**over) -> PortfolioParams:
    base = dict(
        capital=1_000_000.0, max_positions=6, risk_per_trade_pct=1.0,
        max_participation_of_adtv=0.01, stop_atr_multiple=8.0,
        min_stop_distance_pct=2.0, max_stop_distance_pct=35.0,
        invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
        horizon_sessions=63, entry_rank=6, exit_rank=18,
        cost_bps_round_trip=70.0, use_target=False, use_invalidation=False,
    )
    base.update(over)
    return PortfolioParams(**base)


def _prices(n: int = 2600, drift: float = 0.0008, seed: int = 7):
    """A rising market -- the regime in which cash drag is most visible."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.012, n))) for s in SYMBOLS},
        index=idx,
    )
    high, low = close * 1.012, close * 0.988
    open_ = close.shift(1).bfill()
    atr = pd.DataFrame(np.full((n, len(SYMBOLS)), 2.5), index=idx, columns=SYMBOLS)
    ma = close.rolling(50, min_periods=1).mean()
    adtv = pd.DataFrame(np.full((n, len(SYMBOLS)), 5e9), index=idx, columns=SYMBOLS)
    bench = (1.0 + close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)).cumprod()
    return {"close": close, "high": high, "low": low, "open": open_, "atr": atr,
            "ma": ma, "adtv": adtv, "benchmark": bench}


def _rankings(prices, every: int = 21):
    idx = list(prices["close"].index)
    rng = np.random.default_rng(3)
    out = []
    for i in range(200, len(idx) - 70, every):
        s = pd.Series(rng.permutation(len(SYMBOLS)).astype(float), index=SYMBOLS)
        out.append((idx[i], s.sort_values(ascending=False)))
    return out


def _run(p):
    prices = _prices()
    r = simulate(_rankings(prices), prices, p, phase=0, step_sessions=21)
    assert not r.empty
    return r.metrics(periods_per_year=252.0 / p.horizon_sessions)


# =============================================================================
# the invariant
# =============================================================================


def test_the_headline_metric_is_the_leverage_invariant_one():
    m = _run(_params())
    assert m["headline_metric"] == "alpha_on_deployed_ann"
    assert "mean_excess" in m["leverage_confounded"]
    assert "information_ratio" in m["leverage_confounded"]
    # raw alpha is better than raw excess and still not invariant; it must be
    # labelled as such so nobody promotes it a second time.
    assert "alpha_per_period" in m["leverage_proportional"]
    assert "beta_to_benchmark" in m["leverage_proportional"]


def test_alpha_on_deployed_is_alpha_divided_by_deployment():
    m = _run(_params())
    assert m["alpha_on_deployed"] == pytest.approx(
        m["alpha_per_period"] / m["deployed_frac"], rel=1e-9)


def test_raw_excess_moves_with_the_risk_budget_and_alpha_does_not():
    """The whole finding, as one assertion.

    Raising the risk budget raises deployed capital, which mechanically raises
    the raw excess against a fully-invested benchmark. If `alpha_per_period`
    moved by anything like as much, it would be as confounded as the figure it
    replaces and there would be no honest headline available.
    """
    low = _run(_params(risk_per_trade_pct=0.5))
    high = _run(_params(risk_per_trade_pct=4.0))

    assert high["deployed_frac"] > low["deployed_frac"] * 1.5, (
        "the risk budget must drive deployed capital; if it does not, this "
        "fixture no longer reproduces the mechanism under audit"
    )
    raw_move = abs(high["mean_excess"] - low["mean_excess"])
    head_move = abs(high["alpha_on_deployed"] - low["alpha_on_deployed"])
    assert raw_move > head_move, (
        f"raw excess moved {raw_move:.5f} and the headline moved "
        f"{head_move:.5f}. The headline must be LESS sensitive to leverage "
        f"than the figure it replaced, or promoting it achieved nothing."
    )


def test_cash_drag_reconciles_raw_excess_to_the_levmatch_figure():
    """levmatch_excess = mean_excess + (1 - deployed) * mean(bench), exactly."""
    m = _run(_params())
    assert m["levmatch_excess"] == pytest.approx(
        m["mean_excess"] + m["cash_drag_per_period"], rel=1e-9, abs=1e-12
    )
    assert m["cash_drag_per_period"] == pytest.approx(
        (1.0 - m["deployed_frac"]) * m["bench_mean_return"], rel=1e-9, abs=1e-12
    )


def test_alpha_is_exactly_the_regression_intercept():
    """No second definition. alpha = mean(r) - beta * mean(b)."""
    r = np.array([0.05, -0.02, 0.03, 0.01, 0.04, -0.01, 0.02, 0.06])
    b = np.array([0.04, -0.03, 0.02, 0.02, 0.03, -0.02, 0.01, 0.05])
    s = _benchmark_stats(r, b, periods_per_year=4.0,
                         deployed=np.full(len(r), 0.25))
    beta = np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1)
    assert s["beta_to_benchmark"] == pytest.approx(beta)
    assert s["alpha_per_period"] == pytest.approx(r.mean() - beta * b.mean())
    assert s["cash_drag_per_period"] == pytest.approx(0.75 * b.mean())


def test_alpha_carries_a_t_statistic():
    """A headline quoted without an error bar is how -1.3% got read as a fact.

    The t is scale-free: alpha and se(alpha) both carry the factor `dep`, so
    it is the significance of `alpha_on_deployed` as well as of `alpha`.
    """
    m = _run(_params())
    assert np.isfinite(m["alpha_t"]), "the headline must arrive with its own t"


def test_a_fully_deployed_book_has_no_cash_drag():
    """The boundary: at deployed == 1 the raw and matched figures coincide."""
    r = np.array([0.05, -0.02, 0.03, 0.01, 0.04, -0.01])
    b = np.array([0.04, -0.03, 0.02, 0.02, 0.03, -0.02])
    s = _benchmark_stats(r, b, periods_per_year=4.0, deployed=np.ones(len(r)))
    assert s["cash_drag_per_period"] == pytest.approx(0.0, abs=1e-12)
    assert s["levmatch_excess"] == pytest.approx(s["mean_excess"])


def test_missing_deployment_does_not_fabricate_a_matched_figure():
    """Unknown deployment is NaN, never silently 1.0.

    Reporting `levmatch == mean_excess` for a caller that could not supply the
    deployed fraction would assert a full-investment claim nobody checked --
    the same silent-default failure the liquidity gate exists to prevent.
    """
    r = np.array([0.05, -0.02, 0.03, 0.01, 0.04, -0.01])
    b = np.array([0.04, -0.03, 0.02, 0.02, 0.03, -0.02])
    s = _benchmark_stats(r, b, periods_per_year=4.0, deployed=None)
    assert np.isnan(s["deployed_frac"])
    assert np.isnan(s["levmatch_excess"])
    assert np.isnan(s["alpha_on_deployed"]), (
        "without a deployment figure the headline is UNKNOWN. Falling back to "
        "the raw alpha here would quietly assert dep == 1."
    )
    assert np.isfinite(s["alpha_per_period"]), (
        "the raw regression alpha needs no deployment figure and is still "
        "reported -- it is simply not the headline"
    )
