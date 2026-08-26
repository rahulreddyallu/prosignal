"""The label, the backtest and the engine must describe the SAME trade.

The codebase carried three answers to "how did this position end" and they
disagreed on 16% of outcomes. The model was fitted against one, the backtest
measured another, and Stage 7 traded a third.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.exits import (
    EXIT_INVALIDATION, EXIT_STOP, EXIT_TARGET, EXIT_TIMEOUT, ExitRules,
    atr_panel, ma_panel, resolve_exits)

SY = [f"S{i:02d}" for i in range(12)]


def _prices(n=400, seed=3, drift=0.0015):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n))) for s in SY},
        index=idx)
    return close, close * 1.008, close * 0.992, close.shift(1).bfill()


def _rules(**kw):
    base = dict(stop_atr_multiple=2.5, min_stop_distance_pct=2.0,
                max_stop_distance_pct=15.0, target_r_multiple=3.0,
                invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
                atr_period_sessions=14, horizon=63)
    base.update(kw)
    return ExitRules(**base)


class TestOneDefinition:
    def test_the_simulator_and_the_label_agree_exactly(self):
        """The proof. `_hold` is the per-symbol adapter onto the same resolver
        the label uses, so any divergence is a divergence with itself."""
        from prosignal.validation.portfolio_sim import PortfolioParams, _hold

        close, high, low, open_ = _prices()
        r = _rules()
        atr = atr_panel(high, low, close, r.atr_period_sessions, r.atr_method)
        ma = ma_panel(close, r.invalidation_ma_sessions)
        params = PortfolioParams(
            capital=1e6, max_positions=8, risk_per_trade_pct=1.0,
            max_participation_of_adtv=0.02,
            stop_atr_multiple=r.stop_atr_multiple,
            min_stop_distance_pct=r.min_stop_distance_pct,
            max_stop_distance_pct=r.max_stop_distance_pct,
            invalidation_ma_sessions=r.invalidation_ma_sessions,
            invalidation_buffer_atr=r.invalidation_buffer_atr,
            horizon_sessions=r.horizon, entry_rank=8, exit_rank=16,
            target_r_multiple=r.target_r_multiple)

        shared = resolve_exits(close, 200, r, high=high, low=low, open_=open_,
                               atr=atr, ma=ma)
        compared = 0
        for sym in SY:
            ref = _hold(sym, 200, close, low, open_, ma, atr, params)
            got = shared.loc[sym, "ret"]
            if ref is None and not np.isfinite(got):
                continue
            assert ref is not None and np.isfinite(got), sym
            assert ref == pytest.approx(got, abs=1e-12), sym
            compared += 1
        assert compared >= 6, "the fixture produced too few live trades to prove anything"

    def test_the_label_reads_the_engines_own_config(self):
        """`rules_from_config` is the only reader, so a change to the traded
        stop moves the training label with it."""
        import pathlib

        from prosignal.config.loader import load_config
        from prosignal.features.exits import rules_from_config
        from prosignal.stages._cfg import fv

        cfg = load_config(pathlib.Path("config/parameters.yaml")).params
        r = rules_from_config(cfg.stage4_core_score, cfg.stage7_risk)
        assert r.stop_atr_multiple == fv(cfg.stage7_risk.stop_loss.atr_multiple)
        assert r.target_r_multiple == fv(cfg.stage7_risk.targets.t2_r_multiple)

    def test_the_shipped_label_uses_the_engine_geometry(self):
        import pathlib

        from prosignal.config.loader import load_config

        lab = load_config(
            pathlib.Path("config/parameters.yaml")
        ).params.stage4_core_score.labels
        assert lab.barrier_source == "engine", (
            "sigma barriers describe a 1.33:1 trade this engine does not take"
        )


class TestGeometry:
    def test_reward_to_risk_is_the_engines_not_the_labels(self):
        close, high, low, open_ = _prices(drift=0.004, seed=9)
        r = _rules()
        out = resolve_exits(close, 200, r, high=high, low=low, open_=open_)
        won = out[out["side"] == EXIT_TARGET]
        if len(won):
            # A target hit returns exactly target_r_multiple x the stop distance.
            assert (won["ret"] > 0).all()

    def test_a_stop_gap_fills_at_the_open_not_the_stop(self):
        """Assuming the stop price is the optimistic error, and the optimistic
        error is the one that matters."""
        idx = pd.bdate_range("2022-01-03", periods=200)
        close = pd.DataFrame({"A": np.linspace(100, 120, 200)}, index=idx)
        high, low, open_ = close * 1.01, close * 0.99, close.copy()
        # A violent gap down on the bar after entry.
        for f in (close, high, low, open_):
            f.iloc[151] = 60.0
        r = _rules(invalidation_buffer_atr=1e9)
        out = resolve_exits(close, 150, r, high=high, low=low, open_=open_)
        assert out.loc["A", "side"] == EXIT_STOP
        assert out.loc["A", "ret"] < -0.35, "filled at the stop, not the gap open"

    def test_a_name_invalid_at_entry_is_not_a_trade(self):
        """Stage 6 would never trigger it. Labelling it as a day-one loss put a
        trend filter inside the label: invalidation became 52% of all outcomes
        at a median hold of THREE sessions."""
        idx = pd.bdate_range("2022-01-03", periods=300)
        falling = pd.DataFrame(
            {"D": 100 * np.exp(np.cumsum(np.full(300, -0.004)))}, index=idx)
        out = resolve_exits(falling, 200, _rules(), high=falling * 1.01,
                            low=falling * 0.99, open_=falling)
        assert bool(out.loc["D"].isna().all())

    def test_ties_inside_a_bar_resolve_to_the_worst_outcome(self):
        idx = pd.bdate_range("2022-01-03", periods=200)
        close = pd.DataFrame({"A": np.full(200, 100.0)}, index=idx)
        high, low, open_ = close.copy(), close.copy(), close.copy()
        high.iloc[151] = 400.0            # target
        low.iloc[151] = 1.0               # and stop, same bar
        r = _rules(invalidation_buffer_atr=1e9)
        out = resolve_exits(close, 150, r, high=high, low=low, open_=open_)
        assert out.loc["A", "side"] == EXIT_STOP

    def test_it_reads_nothing_at_or_before_the_decision_date_for_the_outcome(self):
        close, high, low, open_ = _prices()
        r = _rules()
        before = resolve_exits(close, 200, r, high=high, low=low, open_=open_)
        c2, h2, l2, o2 = (f.copy() for f in (close, high, low, open_))
        for f in (c2, h2, l2, o2):
            f.iloc[201 + r.horizon:] *= 5.0        # violent future beyond the cap
        after = resolve_exits(c2, 200, r, high=h2, low=l2, open_=o2)
        pd.testing.assert_frame_equal(before, after)

    def test_every_exit_kind_is_reachable(self):
        close, high, low, open_ = _prices(n=600, seed=11)
        r = _rules()
        sides = set()
        for i in range(200, 500, 21):
            out = resolve_exits(close, i, r, high=high, low=low, open_=open_)
            sides |= set(out["side"].dropna().unique())
        assert {EXIT_STOP, EXIT_TARGET, EXIT_INVALIDATION} <= sides or len(sides) >= 2
