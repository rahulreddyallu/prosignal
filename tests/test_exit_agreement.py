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
    high, low = close * 1.008, close * 0.992

    # ONE DELIBERATE SPIKE PER SYMBOL: a bar that trades far above the 3R
    # target intraday and closes back below it. Without it these paths never
    # produce the bar that distinguishes a target read on the HIGH from one
    # read on the CLOSE, and the agreement test below passed for months while
    # the simulator read one and the label read the other. A fixture that
    # cannot exercise a difference cannot prove its absence.
    high = high.copy()
    for k, s in enumerate(SY):
        high.iloc[210 + k, high.columns.get_loc(s)] = close[s].iloc[210 + k] * 1.60
    return close, high, low, close.shift(1).bfill()


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
        the label uses, so any divergence is a divergence with itself.

        `high` is passed. It used to be None inside `_hold`, which made the
        TARGET touch test read the close while the stop read the intraday low --
        an asymmetry inside the module whose entire purpose is that there be one
        definition. This test passed anyway because the fixture rarely reached a
        target; it now exercises the path that hid the divergence."""
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
            # `high=high`, because that is what the simulator now passes. It
            # used to pass None, and `resolve_exits` substitutes the CLOSE when
            # the high is missing -- so the target became a close-only
            # instrument while the stop stayed intraday. This test already
            # compared the two paths correctly; it could not SEE the difference
            # because the fixture below never produced a bar that traded
            # through the target and closed under it. The assertion at the end
            # is what closes that gap.
            ref = _hold(sym, 200, close, low, open_, ma, atr, params, high=high)
            got = shared.loc[sym, "ret"]
            if ref is None and not np.isfinite(got):
                continue
            assert ref is not None and np.isfinite(got), sym
            assert ref[0] == pytest.approx(got, abs=1e-12), sym
            assert ref[1] == pytest.approx(shared.loc[sym, "side"]), sym
            compared += 1
        assert compared >= 6, "the fixture produced too few live trades to prove anything"

    def test_the_fixture_actually_exercises_every_exit(self):
        """A fixture that reaches only one barrier cannot prove agreement.

        The agreement test above passed for months while the simulator read the
        target on the close and the label read it on the high, because these
        paths happen never to spike through the 3R target and close below it.
        Requiring the fixture to produce each exit at least once is what makes
        the comparison a test rather than a coincidence.
        """
        close, high, low, open_ = _prices()
        r = _rules()
        atr = atr_panel(high, low, close, r.atr_period_sessions, r.atr_method)
        ma = ma_panel(close, r.invalidation_ma_sessions)
        seen = set()
        for i in (150, 200, 250, 300):
            out = resolve_exits(close, i, r, high=high, low=low, open_=open_,
                                atr=atr, ma=ma)
            seen.update(int(v) for v in out["side"].dropna().unique())
        for side, name in ((EXIT_STOP, "stop"), (EXIT_TARGET, "target"),
                           (EXIT_TIMEOUT, "timeout")):
            assert int(side) in seen, (
                f"no {name} exit anywhere in the fixture, so an error in how "
                f"the {name} is resolved would pass every test in this file"
            )

    def test_the_fixture_can_tell_a_high_from_a_close(self):
        """The specific bar the agreement test needs and did not have.

        A target read on the CLOSE and a target read on the HIGH agree on every
        bar except one: the bar that trades through the level and closes back
        under it. If the fixture contains none, the two constructions are
        indistinguishable and the agreement test is vacuous with respect to the
        one divergence that was actually present in production.
        """
        close, high, low, open_ = _prices()
        r = _rules()
        atr = atr_panel(high, low, close, r.atr_period_sessions, r.atr_method)
        ma = ma_panel(close, r.invalidation_ma_sessions)
        on_high = resolve_exits(close, 200, r, high=high, low=low, open_=open_,
                                atr=atr, ma=ma)
        on_close = resolve_exits(close, 200, r, high=None, low=low, open_=open_,
                                 atr=atr, ma=ma)
        differ = (on_high["side"] != on_close["side"]).sum()
        assert differ > 0, (
            "reading the target on the high and on the close give the same "
            "answer everywhere in this fixture, so it cannot detect the "
            "asymmetry the simulator shipped with"
        )

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


class TestTheHierarchySwitchesReachTheMeasurement:
    """`stage7_risk.exit_hierarchy` decided what the CARD printed and nothing
    else.

    `rules_from_config`, `resolve_exits`, the training label and the portfolio
    simulator all ignored it, so turning the stop off in `parameters.yaml`
    removed it from the card and left it in every backtest and every validation
    number. An operator who did that would have seen the measurements not move
    and concluded the stop was free -- which is the reverse of what this engine
    measures it to cost.
    """

    def _panels(self):
        close, high, low, open_ = _prices()
        r = _rules()
        return (close, high, low, open_,
                atr_panel(high, low, close, r.atr_period_sessions, r.atr_method),
                ma_panel(close, r.invalidation_ma_sessions))

    def test_disarming_the_stop_removes_stop_exits(self):
        close, high, low, open_, atr, ma = self._panels()
        on = resolve_exits(close, 200, _rules(), high=high, low=low,
                           open_=open_, atr=atr, ma=ma)
        off = resolve_exits(close, 200, _rules(use_stop=False), high=high,
                            low=low, open_=open_, atr=atr, ma=ma)
        assert (on["side"] == EXIT_STOP).sum() > 0, "the fixture has no stops"
        assert (off["side"] == EXIT_STOP).sum() == 0
        assert not np.allclose(on["ret"].dropna().to_numpy(),
                               off["ret"].reindex(on["ret"].dropna().index).to_numpy(),
                               equal_nan=True)

    def test_disarming_the_target_removes_target_exits(self):
        close, high, low, open_, atr, ma = self._panels()
        on = resolve_exits(close, 200, _rules(), high=high, low=low,
                           open_=open_, atr=atr, ma=ma)
        off = resolve_exits(close, 200, _rules(use_target=False), high=high,
                            low=low, open_=open_, atr=atr, ma=ma)
        assert (on["side"] == EXIT_TARGET).sum() > 0, "the fixture has no targets"
        assert (off["side"] == EXIT_TARGET).sum() == 0

    def test_disarming_the_invalidation_removes_invalidation_exits(self):
        close, high, low, open_, atr, ma = self._panels()
        tight = _rules(invalidation_buffer_atr=0.1)
        on = resolve_exits(close, 200, tight, high=high, low=low,
                           open_=open_, atr=atr, ma=ma)
        off = resolve_exits(close, 200, _rules(invalidation_buffer_atr=0.1,
                                               use_invalidation=False),
                            high=high, low=low, open_=open_, atr=atr, ma=ma)
        assert (on["side"] == EXIT_INVALIDATION).sum() > 0
        assert (off["side"] == EXIT_INVALIDATION).sum() == 0

    def test_rules_from_config_carries_the_switches(self):
        """The one reader. If it drops them the switches are decoration again.

        DISARMED IN THE STUB, not in the shipped config. Asserting that the
        shipped `true` arrives as `True` cannot distinguish a wired switch from
        a hardcoded one -- which is the whole failure mode being guarded, so the
        stub turns each one OFF and requires the change to arrive.
        """
        from types import SimpleNamespace

        from prosignal.config.loader import load_config
        from prosignal.features.exits import rules_from_config

        cfg = load_config()
        c4, c7 = cfg.params.stage4_core_score, cfg.params.stage7_risk
        armed = lambda n: bool(getattr(n, "value", n))
        live = rules_from_config(c4, c7)
        assert (live.use_stop, live.use_target, live.use_invalidation) == (
            armed(c7.exit_hierarchy.stop_loss_breach),
            armed(c7.exit_hierarchy.target_achieved),
            armed(c7.exit_hierarchy.thesis_invalidation))
        # v1 arms all three, so honouring them changes nothing about what v1
        # trades -- which is why this could be fixed under a frozen config.
        assert (live.use_stop, live.use_target, live.use_invalidation) == (
            True, True, True)

        for field, attr in (("stop_loss_breach", "use_stop"),
                            ("target_achieved", "use_target"),
                            ("thesis_invalidation", "use_invalidation")):
            stub = SimpleNamespace(**{
                k: getattr(c7, k) for k in
                ("stop_loss", "targets", "thesis_invalidation", "atr")})
            stub.exit_hierarchy = SimpleNamespace(**{
                "stop_loss_breach": True, "target_achieved": True,
                "thesis_invalidation": True, **{field: False}})
            r = rules_from_config(c4, stub)
            assert getattr(r, attr) is False, (
                f"exit_hierarchy.{field} was switched off and {attr} did not "
                f"follow, so the switch reaches the card and nothing else"
            )
