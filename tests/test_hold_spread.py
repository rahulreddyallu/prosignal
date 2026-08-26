"""The buy/hold spread has to be priced, not assumed.

The engine enters at rank 8 and holds until rank 16. That gap is the whole of
its turnover control and it had never been measured, because the simulator
netted cost into the return and discarded the parts -- so the saving a wider
band exists to capture was invisible in every number the engine reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.portfolio_sim import PortfolioParams, phase_summary, simulate

SY = [f"S{i:02d}" for i in range(40)]


def _prices(n=900, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))) for s in SY},
        index=idx)
    return {"close": close, "high": close * 1.01, "low": close * 0.985,
            "open": close.shift(1).fillna(close.iloc[0]),
            "atr": pd.DataFrame(np.full((n, len(SY)), 2.0), index=idx, columns=SY),
            "ma": close.rolling(50, min_periods=1).mean(),
            "adtv": pd.DataFrame(np.full((n, len(SY)), 5e9), index=idx, columns=SY)}


def _rankings(prices, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(120, len(prices["close"]) - 70, 21):
        s = pd.Series(rng.normal(size=len(SY)), index=SY).sort_values(ascending=False)
        out.append((prices["close"].index[i], s))
    return out


def _params(entry=8, exit_=16, cost=70.0):
    return PortfolioParams(
        capital=1e6, max_positions=8, risk_per_trade_pct=1.0,
        max_participation_of_adtv=0.02, stop_atr_multiple=2.5,
        min_stop_distance_pct=2.0, max_stop_distance_pct=15.0,
        invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
        horizon_sessions=63, entry_rank=entry, exit_rank=exit_,
        cost_bps_round_trip=cost)


class TestCostIsVisible:
    def test_gross_and_cost_are_reported_separately(self):
        p = _prices()
        r = simulate(_rankings(p), p, _params())
        assert not r.empty
        assert {"gross_ret", "cost_ret"} <= set(r.periods.columns)

    def test_net_is_exactly_gross_less_cost(self):
        p = _prices()
        r = simulate(_rankings(p), p, _params())
        gap = r.periods["gross_ret"] - r.periods["cost_ret"] - r.periods["ret"]
        assert float(gap.abs().max()) < 1e-12

    def test_a_costlier_book_hands_over_more_of_its_gross(self):
        p = _prices()
        cheap = simulate(_rankings(p), p, _params(cost=20.0)).metrics()
        dear = simulate(_rankings(p), p, _params(cost=200.0)).metrics()
        assert dear["mean_cost"] > cheap["mean_cost"]
        assert dear["mean_return"] < cheap["mean_return"]

    def test_the_cost_share_survives_phase_pooling(self):
        """Reported per phase but thrown away when phases were pooled, which is
        the only figure `research spread` actually prints."""
        p = _prices()
        m = phase_summary(_rankings(p), p, _params(), step_sessions=21)
        assert np.isfinite(m["mean_gross"]) and np.isfinite(m["mean_cost"])
        assert m["mean_gross"] == pytest.approx(
            m["mean_return"] + m["mean_cost"], abs=1e-12)


class TestTheBandDoesSomething:
    def test_a_wider_exit_band_turns_the_book_over_less(self):
        p = _prices()
        rk = _rankings(p)
        tight = phase_summary(rk, p, _params(8, 8), step_sessions=21)
        wide = phase_summary(rk, p, _params(8, 25), step_sessions=21)
        assert wide["avg_new"] < tight["avg_new"]

    def test_and_therefore_pays_less_entry_cost(self):
        p = _prices()
        rk = _rankings(p)
        tight = phase_summary(rk, p, _params(8, 8), step_sessions=21)
        wide = phase_summary(rk, p, _params(8, 25), step_sessions=21)
        assert wide["mean_cost"] < tight["mean_cost"]

    def test_an_entry_rank_beyond_the_slot_count_changes_nothing(self):
        """The book has 8 slots, so the 10th-best candidate is never reached.
        `research spread` says so rather than printing duplicate rows as though
        they were different configurations."""
        p = _prices()
        rk = _rankings(p)
        a = phase_summary(rk, p, _params(8, 16), step_sessions=21)
        b = phase_summary(rk, p, _params(10, 16), step_sessions=21)
        assert a["mean_return"] == pytest.approx(b["mean_return"])
        assert a["avg_new"] == pytest.approx(b["avg_new"])

    def test_the_exit_band_cannot_be_tighter_than_the_entry(self):
        """A name would be sold the moment it was bought."""
        p = _prices()
        rk = _rankings(p)
        m = phase_summary(rk, p, _params(8, 4), step_sessions=21)
        tight = phase_summary(rk, p, _params(8, 8), step_sessions=21)
        # It still runs, but it holds nothing across rebalances: every name is
        # new every period.
        if m:
            assert m["avg_new"] >= tight["avg_new"] - 1e-9


class TestPairedComparison:
    def test_bands_are_paired_by_phase_and_date_not_date_alone(self):
        """Two phases produce periods on different dates from the same schedule.
        Keying on date alone silently pairs one phase against another."""
        from prosignal.cli import _band_periods

        p = _prices()
        rk = _rankings(p)
        keys = list(_band_periods(rk, p, _params(), 8, 16).keys())
        assert keys and all(isinstance(k, tuple) and len(k) == 2 for k in keys)
        assert len({k[0] for k in keys}) > 1, "only one phase was walked"

    def test_the_same_band_compared_with_itself_is_exactly_zero(self):
        from prosignal.cli import _band_periods

        p = _prices()
        rk = _rankings(p)
        a = _band_periods(rk, p, _params(), 8, 16)
        b = _band_periods(rk, p, _params(), 8, 16)
        assert set(a) == set(b)
        assert max(abs(a[k][0] - b[k][0]) for k in a) == 0.0
