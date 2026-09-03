"""The corporate-action table must explain the large price jumps in the store.

WHY THIS EXISTS. Until 2026-09-03 the shipped table (yfinance-sourced) covered 34.6%
of split-like price moves in 2010-2017 and 83.1% in 2018-2026. HDFC's 1:5, Tata
Motors' 1:5, Infibeam's 1:10, Bajaj Finance's 10x and Vedanta's demerger were all
absent. An unadjusted 1:10 split reads as a -90% session and corrupts momentum,
volatility, kurtosis and drawdown for that name across EVERY window spanning the
ex-date -- silently, and with every other test still passing.

The test is written against the property that matters (are the jumps explained?)
rather than against a row count, because a row count says nothing about whether the
rows are the right ones.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from prosignal.data.store import DataStore

# A one-day move this large in a liquid name is not a market move.
JUMP = 0.35
LIQUID_TURNOVER = 1e6
# Simple split/bonus ratios. A real crash can land near 1/2 or 2/3 by chance, so a
# floor on coverage rather than a demand for perfection.
SIMPLE = np.array([1/2, 1/3, 1/4, 1/5, 1/10, 2/3, 3/4, 2/5, 1/20])
MIN_COVERAGE = 0.85


@pytest.fixture(scope="module")
def store(live_cfg):
    """The real curated store, read UNADJUSTED -- the point is to see the raw jumps."""
    return DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots,
                     adjust_prices=False)


def _split_like(store):
    px = store.read_prices(columns=["date", "symbol", "series", "close", "turnover"])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["series"] == "EQ"].sort_values(["symbol", "date"])
    g = px.groupby("symbol", sort=False)
    px["pc"] = g["close"].shift(1)
    px["gap"] = g["date"].diff().dt.days
    liq = px[(px["gap"] <= 5) & px["pc"].notna() & (px["turnover"] >= LIQUID_TURNOVER)]
    ratio = liq["close"] / liq["pc"]
    cand = liq[(ratio - 1.0).abs() > JUMP]
    r = (cand["close"] / cand["pc"]).to_numpy()
    near = np.abs(r[:, None] / SIMPLE[None, :] - 1.0).min(1) < 0.06
    return cand[near]


def test_large_price_jumps_are_explained_by_a_corporate_action(store):
    """Split-like jumps in liquid names must appear in the action table."""
    sl = _split_like(store)
    if sl.empty:
        pytest.skip("no split-like moves in this store")
    acts = store.read_corporate_actions()
    acts = acts[acts["action_type"] != "dividend"]
    acts["ex_date"] = pd.to_datetime(acts["ex_date"])
    by = {}
    for s, d in zip(acts["symbol"], acts["ex_date"]):
        by.setdefault(s, []).append(d)
    covered = [
        any(abs((d - x).days) <= 4 for x in by.get(s, ()))
        for s, d in zip(sl["symbol"], sl["date"])
    ]
    rate = float(np.mean(covered))
    missed = sl[[not c for c in covered]]
    assert rate >= MIN_COVERAGE, (
        f"corporate-action coverage of split-like moves is {rate:.1%}, "
        f"below the {MIN_COVERAGE:.0%} floor. {len(missed)} unexplained jumps, "
        f"worst: "
        + ", ".join(
            f"{r.symbol} {r.date.date()} {r.pc:.1f}->{r.close:.1f}"
            for r in missed.nlargest(5, "turnover").itertuples()
        )
    )


def test_coverage_does_not_differ_between_eras(store):
    """Train and holdout must be equally well adjusted.

    A data-quality asymmetry between the two windows would confound every
    out-of-sample comparison made across them -- the model would look worse on
    whichever era had more unadjusted splits, for a reason that is not the model.
    """
    sl = _split_like(store)
    if sl.empty:
        pytest.skip("no split-like moves in this store")
    acts = store.read_corporate_actions()
    acts = acts[acts["action_type"] != "dividend"]
    acts["ex_date"] = pd.to_datetime(acts["ex_date"])
    by = {}
    for s, d in zip(acts["symbol"], acts["ex_date"]):
        by.setdefault(s, []).append(d)

    def cov(frame):
        if frame.empty:
            return None
        return float(np.mean([
            any(abs((d - x).days) <= 4 for x in by.get(s, ()))
            for s, d in zip(frame["symbol"], frame["date"])
        ]))

    early = cov(sl[sl["date"] < "2018-01-01"])
    late = cov(sl[sl["date"] >= "2018-01-01"])
    if early is None or late is None:
        pytest.skip("only one era present in this store")
    assert abs(early - late) < 0.20, (
        f"corporate-action coverage differs by era: {early:.1%} before 2018 "
        f"against {late:.1%} after. Any measurement compared across that boundary "
        f"is measuring the data, not the model."
    )
