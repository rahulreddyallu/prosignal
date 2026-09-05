"""Guards for the production build: what orders the book, when it buys, and
what it records about the trade it is issuing.

These are the three mechanisms the tuning added, and each of them fails in a
way that is invisible rather than loud, which is why each gets a test that can
tell the mechanism working from the mechanism absent:

  * the RANKING POLICY fails by falling back to a scorer that lost to an
    equal-weight benchmark in all 144 of its configurations, on a run that looks
    completely normal;
  * the ENTRY CLOCK fails by being stuck open (the old behaviour, which is at
    least a strategy that was measured) or stuck closed (a book that silently
    never buys again);
  * the TRADE PLAN fails by recording today's config against a trade issued
    months ago, which is how a paper-trading record stops being a record.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from types import SimpleNamespace
import pytest

from prosignal import cadence
from prosignal.core.contracts import RiskPlan
from prosignal.features import engine
from prosignal.stages.stage4_core_score import (
    RankingUnavailable,
    _rank_by_engine,
)
from prosignal.tradeplan import build_trade_plan


# =============================================================================
# the entry clock
# =============================================================================
class TestTheEntryClock:
    """The engine runs every session and buys on a cadence. Those are two
    different things and the whole point of this module is that they can be."""

    @staticmethod
    def _sessions(n=200, start="2026-09-01"):
        return [d.date() for d in pd.bdate_range(start, periods=n)]

    def test_cadence_one_opens_every_session(self):
        s = self._sessions()
        for d in s[:10]:
            assert cadence.resolve(s, d, cadence_sessions=1,
                                   anchor=s[0]).is_entry_date

    def test_only_every_nth_session_opens(self):
        s = self._sessions()
        opens = [d for d in s[:64]
                 if cadence.resolve(s, d, cadence_sessions=21,
                                    anchor=s[0]).is_entry_date]
        assert opens == [s[0], s[21], s[42], s[63]], (
            f"got {opens}: the clock must open on exact multiples of the "
            f"cadence counted in SESSIONS from the anchor")

    def test_it_counts_sessions_not_calendar_days(self):
        """The failure this prevents: a holiday shifting every subsequent entry.

        Two calendars covering the same span, one with a week missing. Counting
        days, the 21st entry would land on different dates; counting sessions it
        lands on the 21st SESSION of each, which is what a backtest replays.
        """
        full = self._sessions(60)
        holed = [d for i, d in enumerate(full) if not (10 <= i < 15)]
        a = [i for i, d in enumerate(full)
             if cadence.resolve(full, d, cadence_sessions=21, anchor=full[0]).is_entry_date]
        b = [i for i, d in enumerate(holed)
             if cadence.resolve(holed, d, cadence_sessions=21, anchor=holed[0]).is_entry_date]
        assert a == b == [0, 21, 42], (
            "the entry phase must depend on the session count alone, so a "
            "market holiday cannot re-phase the schedule")

    def test_it_reports_the_next_entry_date(self):
        s = self._sessions()
        c = cadence.resolve(s, s[1], cadence_sessions=21, anchor=s[0])
        assert not c.is_entry_date
        assert c.next_entry_date == s[21]
        assert "next entry date" in c.reason

    def test_an_unresolvable_clock_fails_OPEN_and_says_so(self):
        """A clock that fails CLOSED stops the book silently and forever.

        Both failure directions are wrong, but they are not equally wrong. Stuck
        open is the previous behaviour -- a strategy that was measured, just not
        the one intended. Stuck closed is a book that never trades again and
        whose output is indistinguishable from a market with no candidates.
        """
        s = self._sessions()
        beyond = cadence.resolve(s, s[5], cadence_sessions=21,
                                 anchor=dt.date(2099, 1, 1))
        assert beyond.is_entry_date
        assert "could not be resolved" in beyond.reason

        off_calendar = cadence.resolve(s, dt.date(2026, 1, 1),
                                       cadence_sessions=21, anchor=s[0])
        assert off_calendar.is_entry_date

    def test_the_anchor_resolves_FORWARD_to_a_real_session(self):
        """An anchor on a holiday must resolve deterministically, and forward.

        Backward would reach into sessions already traded, which would re-phase
        a schedule that has live positions on it.
        """
        s = self._sessions()
        weekend = s[0] - dt.timedelta(days=1)
        c = cadence.resolve(s, s[0], cadence_sessions=21, anchor=weekend)
        assert c.is_entry_date and c.sessions_since_anchor == 0

    def test_blocked_reason_is_none_when_open(self):
        s = self._sessions()
        assert cadence.resolve(s, s[0], cadence_sessions=21,
                               anchor=s[0]).blocked_reason() is None
        assert cadence.resolve(s, s[1], cadence_sessions=21,
                               anchor=s[0]).blocked_reason()

    def test_the_shipped_config_resolves(self, cfg):
        s = self._sessions(400)
        c = cadence.clock_from_config(cfg, s, s[-1])
        assert c.cadence_sessions == int(
            cfg.params.stage6_entry.admission.entry_cadence_sessions.value)


# =============================================================================
# the ranking, of which there is one
# =============================================================================
class TestTheRankingIsSingular:
    """This class used to be `TestTheRankingPolicy` and exercised a six-way
    switch: v3_composite, v9r_core, a v2 composite, a fitted cross-sectional
    model, measured_factor and family_average. Five of those never ran. The
    switch and the alternatives were deleted on 2026-09-05, and the tests that
    proved the switch worked went with them -- a test suite that still checks a
    branch nobody can reach is how dead code survives a cleanup.

    What is worth pinning is what replaced it: exactly one scorer, no fallback,
    and a loud failure when it cannot build."""

    class _Cfg:
        """Nested classes cannot see each other's names at class-body scope, so
        this builds the shape in __init__ rather than by nesting."""

        def __init__(self, min_themes: int = 3):
            self.ranking = SimpleNamespace(
                min_themes=SimpleNamespace(value=min_themes))

    @staticmethod
    def _scored(n=60, missing=0):
        idx = [f"S{i:03d}" for i in range(n)]
        score = pd.Series(np.linspace(-1, 1, n), index=idx)
        if missing:
            score.iloc[:missing] = np.nan
        return pd.DataFrame({"score": score,
                             "n_themes": pd.Series(5, index=idx)})

    def test_the_engine_ranks_and_says_so(self):
        scored = self._scored()
        composite = pd.Series(0.0, index=scored.index)
        notes = []
        out = _rank_by_engine(composite, self._Cfg(), notes, scored)
        pd.testing.assert_series_equal(out.sort_index(),
                                       scored["score"].sort_index(),
                                       check_names=False)
        joined = " ".join(notes)
        assert "Book ordered by the composite" in joined
        assert "15 factors in 5 themes" in joined

    def test_the_note_reports_effective_weights_not_declared_ones(self):
        """The blend renormalises over the themes a name has, so the declared
        vector is not what ranked anything."""
        scored = self._scored()
        for t in engine.THEMES:
            scored[t + "_sub"] = 0.0
        notes = []
        _rank_by_engine(pd.Series(0.0, index=scored.index), self._Cfg(),
                        notes, scored)
        assert "declared -> effective" in " ".join(notes)

    def test_the_removed_seven_are_named_on_every_run(self):
        """An absence nobody is told about gets re-added by the next person."""
        scored = self._scored()
        notes = []
        _rank_by_engine(pd.Series(0.0, index=scored.index), self._Cfg(),
                        notes, scored)
        joined = " ".join(notes)
        for f in engine.REMOVED_2026_09:
            assert f in joined

    def test_a_scorer_that_did_not_build_stops_the_run(self):
        """There is no second scorer to fall back to, and falling back to one
        would issue signals from a model that was not the one measured."""
        composite = pd.Series(0.0, index=["A", "B"])
        with pytest.raises(RankingUnavailable):
            _rank_by_engine(composite, self._Cfg(), [], None)
        with pytest.raises(RankingUnavailable):
            _rank_by_engine(composite, self._Cfg(), [], pd.DataFrame())

    def test_a_ranking_over_a_minority_of_the_universe_is_refused(self):
        """A ranking built on a minority of the universe is a ranking of that
        minority, whatever the header says."""
        scored = self._scored(n=60, missing=45)
        composite = pd.Series(0.0, index=scored.index)
        with pytest.raises(RankingUnavailable):
            _rank_by_engine(composite, self._Cfg(), [], scored)

# =============================================================================
# the trade plan
# =============================================================================
class TestTheTradePlan:
    """Every issued trade records what it is. Without it a resolved outcome can
    be compared with the market and never with the engine's own claim."""

    def test_it_records_the_cadence_and_the_planned_hold(self, cfg):
        tp = build_trade_plan(cfg, None)
        adm = cfg.params.stage6_entry.admission
        hold = cfg.params.stage7_risk.holding_period
        assert tp.cadence_sessions == int(adm.entry_cadence_sessions.value)
        assert tp.planned_hold_sessions == int(hold.max_holding_sessions.value)

    def test_the_frequencies_come_from_the_config_study(self, cfg):
        tp = build_trade_plan(cfg, None)
        e = cfg.params.expectancy
        assert tp.probability_of_profit == pytest.approx(e.probability_of_profit)
        assert tp.expected_return_pct == pytest.approx(e.expected_return_pct)
        assert tp.basis_trades == int(e.sample_trades)
        assert e.study in tp.basis

    def test_the_mean_and_the_median_are_both_carried(self, cfg):
        """Quoting only the mean describes a typical trade that does not exist.

        The distribution is right-skewed -- two thirds of the return comes from
        the 39% of positions that reach the time limit -- so mean +7.09% and
        median +3.65% are both true and neither alone is honest.
        """
        tp = build_trade_plan(cfg, None)
        assert tp.median_return_pct < tp.expected_return_pct

    def test_beating_the_benchmark_is_reported_separately_from_making_money(self, cfg):
        """The gap is the whole finding.

        58% of these trades make money and 51% beat the universe: most of the
        profit is the market, and quoting the first as though it were the second
        would be the most misleading number the card could print.
        """
        tp = build_trade_plan(cfg, None)
        assert tp.probability_of_beating_benchmark < tp.probability_of_profit

    def test_the_caveat_is_never_empty(self, cfg):
        tp = build_trade_plan(cfg, None)
        assert "not a forecast" in tp.caveat.lower()

    def test_the_risk_at_the_floor_is_sized_from_the_plan(self, cfg):
        plan = RiskPlan(ticker="X", reference_price=100.0, stop_price=65.0,
                        position_size_shares=200)
        tp = build_trade_plan(cfg, plan)
        assert tp.risk_at_stop_inr == pytest.approx(7000.0)
        book = float(cfg.params.capital.total_capital_inr.value)
        assert tp.risk_at_stop_pct_of_book == pytest.approx(700000.0 / book)

    def test_no_size_means_no_invented_risk_number(self, cfg):
        plan = RiskPlan(ticker="X", reference_price=100.0, stop_price=65.0)
        tp = build_trade_plan(cfg, plan)
        assert tp.risk_at_stop_inr is None

    def test_disabling_the_study_leaves_the_geometry_and_drops_the_claims(self, cfg):
        """A deployment that has not run its own study must record no
        expectation rather than somebody else's."""
        cfg.params.expectancy.enabled = False
        try:
            tp = build_trade_plan(cfg, None)
            assert tp.cadence_sessions and tp.planned_hold_sessions
            assert tp.probability_of_profit is None
            assert tp.expected_return_pct is None
            assert tp.basis is None
        finally:
            cfg.params.expectancy.enabled = True


# =============================================================================
# the shipped exit geometry
# =============================================================================
class TestTheShippedExitGeometry:
    """Four of eight rungs are off, each on its own measurement. Pinned here so
    a revert is a failing test rather than a quiet change in what gets sold."""

    def test_the_two_switches_that_control_one_rule_must_agree(self, cfg):
        c7 = cfg.params.stage7_risk
        assert (bool(c7.thesis_invalidation.enabled_as_exit)
                is bool(c7.exit_hierarchy.thesis_invalidation))
        assert (bool(c7.trailing_stop.enabled)
                is bool(c7.exit_hierarchy.trailing_stop))

    def test_the_shipped_rungs_are_what_was_measured(self, cfg):
        h = cfg.params.stage7_risk.exit_hierarchy
        assert h.stop_loss_breach and h.signal_reversal and h.time_expiration
        assert h.new_hard_rejection and h.severe_regime_change
        assert not h.thesis_invalidation, (
            "measured alone this cost 15.6 points of per-trade win probability "
            "and 14.3 points of annual alpha")
        assert not h.target_achieved, "booking at 3R cost 0.9 points of alpha"
        assert not h.trailing_stop

    def test_the_stop_is_a_disaster_floor_not_a_trading_stop(self, cfg):
        sl = cfg.params.stage7_risk.stop_loss
        assert float(sl.atr_multiple.value) >= 6.0, (
            "below about 6 ATR the paired test shows the stop costing return "
            "rather than adding it")
        assert float(sl.max_stop_distance_pct.value) >= 30.0, (
            "a clip under 30% converts the floor back into a trading stop, "
            "which is the setting measured as costly")

    def test_the_invalidation_level_still_gates_ENTRY(self, cfg):
        """Removing the exit must not widen the population the model is fitted
        on. The two used the same number and only one of them was wrong."""
        assert bool(cfg.params.stage6_entry.admission.require_above_invalidation.value)

    def test_the_portfolio_simulator_is_handed_the_same_armed_exits(self, cfg):
        """The simulator has its own parameter object, and it did not carry the
        arming switches at all.

        `PortfolioParams.target_r_multiple` defaulted to 3.0 and there was no
        notion of a disarmed rung, so `research portfolio` measured a book that
        takes profit at 3R and sells on invalidation. That was harmless while
        every rung was armed -- and became a measurement of a strategy the
        engine does not run the moment two of them were not. It is the same
        failure `rules_from_config` was written to end, in the one place that
        builds its geometry separately.
        """
        from prosignal.cli import _portfolio_params

        p = _portfolio_params(cfg)
        h = cfg.params.stage7_risk.exit_hierarchy
        armed = lambda n: bool(getattr(n, "value", n))
        assert p.use_stop is armed(h.stop_loss_breach)
        assert p.use_target is armed(h.target_achieved)
        assert p.use_invalidation is armed(h.thesis_invalidation)
        assert p.target_r_multiple == pytest.approx(
            float(cfg.params.stage7_risk.targets.t2_r_multiple.value)), (
            "the R multiple must come from the config too; a default of 3.0 "
            "here is a second definition of the target")
        # And the geometry must be the shipped geometry, not a stale default.
        assert p.stop_atr_multiple == pytest.approx(
            float(cfg.params.stage7_risk.stop_loss.atr_multiple.value))
        assert p.max_positions == int(cfg.params.capital.max_open_positions.value)
        assert p.exit_rank == int(
            cfg.params.stage6_entry.admission.exit_rank.value)

    def test_the_staleness_gate_reads_its_own_config(self, cfg):
        """`feeds:` declared a max age the pipeline hardcoded.

        The two agreed on the shipped values -- 1/1/1/2 in both places -- which
        is exactly why it went unnoticed for so long: the defect was invisible
        until someone tried to change the policy and found the run unchanged.
        """
        import inspect

        from prosignal import pipeline

        src = inspect.getsource(pipeline._manifest_from_store)
        assert "_max_age(" in src, (
            "the staleness limits must come from config.params.feeds, not from "
            "literals in this function")
        for feed in ("equity_ohlcv", "index_ohlcv", "india_vix", "delivery_data"):
            assert f'_max_age("{feed}"' in src, feed

    def test_the_book_size_and_the_entry_band_agree(self, cfg):
        assert (int(cfg.params.capital.max_open_positions.value)
                == int(cfg.params.stage6_entry.admission.entry_rank.value))
        assert (int(cfg.params.stage6_entry.admission.exit_rank.value)
                > int(cfg.params.stage6_entry.admission.entry_rank.value))
