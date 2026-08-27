"""Stages 3-8, the cost model, and the end-to-end pipeline.

The tests that matter here assert the system's *discipline*, not its output:
that it refuses to emit a probability, that liquidity can override conviction,
that a missing check is never upgraded to a pass, and that NO TRADE is a
first-class result rather than an empty list.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.core.calendar import TradingCalendar
from prosignal.core.contracts import DataQualityReport, StockDataFlags
from prosignal.core.enums import EntryStatus, TriggerType, GateResult, RejectionReason
from prosignal.costs import CostModel
from prosignal.data.store import DataStore
from prosignal.data.types import DATE, SYMBOL
from prosignal.data.universe import UniverseSnapshot
from prosignal.stages import stage3_eligibility, stage4_core_score, stage6_entry, stage7_risk

N = 340


def _dates(n=N):
    return pd.bdate_range("2024-06-03", periods=n)


def _prices(symbols, dates, start=500.0, drift=0.0008, vol=0.012, turnover=2e8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, s in enumerate(symbols):
        path = start * np.exp(np.cumsum(rng.normal(drift, vol, len(dates))))
        for d, px in zip(dates, path):
            rows.append({
                DATE: d, SYMBOL: s, "series": "EQ",
                "open": px, "high": px * 1.012, "low": px * 0.988, "close": px,
                "volume": 400_000.0, "turnover": turnover, "trades": 5_000.0,
                "isin": f"INE{i:04d}A01", "source": "test",
            })
    return pd.DataFrame(rows)


def _setup(tmp_path, symbols, dates, **kw):
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_prices(_prices(symbols, dates, **kw))
    cal = TradingCalendar([d.date() for d in dates])
    uni = UniverseSnapshot("NIFTY 200", dates[-1].date(), list(symbols),
                           {s: "TestSector" for s in symbols})
    return store, cal, uni


def _clean_quality():
    return DataQualityReport(run_status=GateResult.PASS)


# =============================================================================
# cost model
# =============================================================================


def _composite_only(config):
    """Opt these tests into the hand-weighted composite.

    They exercise composite behaviour -- factor ranking, the quality drop, the
    redundancy report -- on fixtures with nowhere near enough history to fit
    the cross-sectional model. Stage 4 refuses to fall back silently, which is
    the point of allow_composite_fallback, so a composite test has to say so.
    """
    config.params.stage4_core_score.allow_composite_fallback = True
    return config


def test_delivery_stt_is_charged_on_both_legs(cfg):
    cb = CostModel(cfg).round_trip(1000.0, 100)
    stt_cfg = cfg.params.costs
    expected = (100_000 * float(stt_cfg.stt_delivery_buy_pct.value) / 100
                + 100_000 * float(stt_cfg.stt_delivery_sell_pct.value) / 100)
    assert cb.stt_inr == pytest.approx(expected)


def test_gst_excludes_stt_and_stamp_duty(cfg):
    """GST applies to brokerage + exchange + SEBI only, never to the taxes."""
    cb = CostModel(cfg).round_trip(1000.0, 100)
    base = cb.brokerage_inr + cb.exchange_txn_inr + cb.sebi_fee_inr
    rate = float(cfg.params.costs.gst_pct_on_charges.value) / 100
    assert cb.gst_inr == pytest.approx(base * rate)
    assert cb.gst_inr < (base + cb.stt_inr + cb.stamp_duty_inr) * rate


def test_stamp_duty_is_buy_side_only(cfg):
    cb = CostModel(cfg).round_trip(1000.0, 100)
    expected = 100_000 * float(cfg.params.costs.stamp_duty_buy_pct.value) / 100
    assert cb.stamp_duty_inr == pytest.approx(expected)


def test_impact_grows_with_participation(cfg):
    m = CostModel(cfg)
    small = m.impact_bps(1_00_000, 10_00_00_000)
    large = m.impact_bps(1_00_00_000, 10_00_00_000)
    assert large > small


def test_missing_adtv_does_not_assume_zero_impact(cfg):
    """The optimistic error is the dangerous one."""
    m = CostModel(cfg)
    assert m.impact_bps(1_000_000, None) > 0
    assert m.impact_bps(1_000_000, 0) > 0


def test_stress_costs_exceed_base_costs(cfg):
    cb = CostModel(cfg).round_trip(1000.0, 100, adtv_inr=5e7)
    assert cb.stressed_total_inr > cb.total_inr
    assert cb.stressed_bps_of_buy > cb.total_bps_of_buy


def test_breakeven_move_is_positive_and_finite(cfg):
    be = CostModel(cfg).breakeven_move_pct(1000.0, 100, adtv_inr=5e7)
    assert 0 < be < 10


# =============================================================================
# stage 3 -- eligibility
# =============================================================================


def test_stage1_failures_are_rejected_before_scoring(tmp_path, cfg):
    dates = _dates()
    syms = ["AAA", "BBB"]
    store, cal, uni = _setup(tmp_path, syms, dates)
    q = DataQualityReport(
        run_status=GateResult.PASS,
        per_stock_flags={"BBB": StockDataFlags(status=GateResult.FAIL,
                                               failed_checks=["outlier_return"])},
    )
    rep = stage3_eligibility.run(uni, store, cal, q, cfg, as_of=dates[-1].date())
    assert rep.rejected["BBB"] is RejectionReason.DATA_QUALITY
    assert "BBB" not in rep.eligible_universe


def test_illiquid_names_are_rejected(tmp_path, cfg):
    dates = _dates()
    store, cal, uni = _setup(tmp_path, ["THIN"], dates, turnover=1e6)
    rep = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg,
                                 as_of=dates[-1].date())
    assert rep.rejected["THIN"] is RejectionReason.ILLIQUID


def test_participation_gate_is_unreachable_at_the_shipped_capital(cfg):
    """FINDING: the ADTV floor dominates the participation gate entirely.

    position_value / max_participation_of_adtv = the ADTV a name needs before
    the participation gate would bite. With the shipped Rs 10L book that is
    Rs 1.25 Cr -- but `min_adtv_inr` already rejects anything under Rs 5 Cr, so
    the gate can never be the binding constraint.

    This matters because `max_participation_of_adtv` is Tier-A search parameter
    #1, described as the most important in the system. At this capital,
    searching it changes nothing at the ELIGIBILITY stage; it only binds inside
    Stage 7 position sizing, and only becomes an eligibility constraint at a
    much larger book. Recorded rather than silently "fixed" by moving a
    threshold.
    """
    p = cfg.params
    need = p.capital.position_value_inr() / float(p.capital.max_participation_of_adtv.value)
    floor = float(p.stage3_eligibility.liquidity.min_adtv_inr.value)
    assert floor > need, (
        "if this ever fails, the participation gate has become reachable at "
        "the eligibility stage and this test should be replaced by a real "
        "behavioural assertion"
    )


def test_insufficient_history_is_rejected_not_approximated(tmp_path, cfg):
    dates = _dates(120)
    store, cal, uni = _setup(tmp_path, ["NEW"], dates)
    rep = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg,
                                 as_of=dates[-1].date())
    assert rep.rejected["NEW"] is RejectionReason.INSUFFICIENT_HISTORY


def test_absent_pledging_is_not_testable_never_a_pass(tmp_path, cfg):
    dates = _dates()
    store, cal, uni = _setup(tmp_path, ["AAA"], dates, turnover=1e10)
    rep = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg,
                                 as_of=dates[-1].date())
    assert "AAA" in rep.eligible_universe
    assert "promoter_pledging" in rep.not_testable["AAA"]


# =============================================================================
# stage 4 -- scoring
# =============================================================================


def _score(tmp_path, cfg, symbols, seeds, regime):
    dates = _dates()
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    frames = [_prices([s], dates, drift=d, seed=i, turnover=1e10)
              for i, (s, d) in enumerate(zip(symbols, seeds))]
    store.write_prices(pd.concat(frames, ignore_index=True))
    store.write_indices(pd.DataFrame([
        {DATE: d, "index_name": "NIFTY 200", "open": 100.0, "high": 100.0,
         "low": 100.0, "close": 100.0 * (1 + 0.0002 * i), "volume": 0.0, "source": "t"}
        for i, d in enumerate(dates)
    ]))
    cal = TradingCalendar([d.date() for d in dates])
    uni = UniverseSnapshot("NIFTY 200", dates[-1].date(), symbols,
                           {s: "TestSector" for s in symbols})
    el = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg,
                                as_of=dates[-1].date())
    return stage4_core_score.run(el, store, cal, regime, cfg, as_of=dates[-1].date()), el


def _momentum_universe(tmp_path, cfg, drifts):
    from prosignal.stages import stage2_regime
    dates = _dates()
    syms = list(drifts)
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_prices(pd.concat(
        [_prices([s], dates, drift=d, vol=0.005, seed=i + 1, turnover=1e10)
         for i, (s, d) in enumerate(drifts.items())], ignore_index=True))
    store.write_indices(pd.DataFrame([
        {DATE: d, "index_name": "NIFTY 200", "open": 100.0, "high": 100.0,
         "low": 100.0, "close": 100.0, "volume": 0.0, "source": "t"} for d in dates
    ]))
    cal = TradingCalendar([d.date() for d in dates])
    uni = UniverseSnapshot("NIFTY 200", dates[-1].date(), syms, {s: "S" for s in syms})
    el = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg,
                                as_of=dates[-1].date())
    regime = stage2_regime.run(store, cal, el.eligible_universe, cfg,
                               as_of=dates[-1].date())
    sc = stage4_core_score.run(el, store, cal, regime, cfg, as_of=dates[-1].date())
    return sc, el


def test_higher_momentum_ranks_higher(tmp_path, cfg, monkeypatch):
    """Every name here is inside the model's fitted domain -- all three are
    above their invalidation level -- so the ordering is the thing under test
    rather than the domain filter."""
    _composite_only(cfg)
    sc, _ = _momentum_universe(tmp_path, cfg, {
        "STRONG": 0.0020, "MID": 0.0012, "WEAK": 0.0005})
    order = [s.ticker for s in sc.ranked_scores]
    assert order.index("STRONG") < order.index("WEAK")
    assert sc.ranked_scores[0].composite_score >= sc.ranked_scores[-1].composite_score


def test_a_name_in_a_decline_is_never_ranked_at_all(tmp_path, cfg):
    """The model's coefficients are estimated on a panel that EXCLUDES names
    below their thesis-invalidation level -- `resolve_exits` gives them a NaN
    label and `build_panel` drops the row. Ranking one anyway extrapolates the
    reversal coefficient past where it was measured, which is how the five
    highest-ranked names on 2026-08-25 all came to be in that state.

    The universe is therefore restricted BEFORE the ranking, so that every
    percentile, the dispersion floor and min_universe_percentile describe a
    population the engine can actually buy."""
    from prosignal.core.enums import RejectionReason

    _composite_only(cfg)
    # A decline steep enough that the outcome does not turn on the noise seed:
    # -0.4%/session puts the close well clear of MA(50) - 1.5 ATR.
    sc, el = _momentum_universe(tmp_path, cfg, {
        "RISING": 0.0020, "FALLING": -0.0040})

    assert "FALLING" not in [s.ticker for s in sc.ranked_scores]
    assert el.rejected.get("FALLING") is RejectionReason.OUTSIDE_MODEL_DOMAIN
    assert "invalidation level" in el.rejection_details["FALLING"]
    assert "RISING" in el.eligible_universe


def test_quality_is_dropped_without_point_in_time_fundamentals(tmp_path, cfg):
    _composite_only(cfg)
    from prosignal.stages import stage2_regime
    dates = _dates()
    store, cal, uni = _setup(tmp_path, ["AAA", "BBB"], dates, turnover=1e10)
    store.write_indices(pd.DataFrame([
        {DATE: d, "index_name": "NIFTY 200", "open": 100.0, "high": 100.0,
         "low": 100.0, "close": 100.0, "volume": 0.0, "source": "t"} for d in dates
    ]))
    el = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg, as_of=dates[-1].date())
    regime = stage2_regime.run(store, cal, el.eligible_universe, cfg, as_of=dates[-1].date())
    sc = stage4_core_score.run(el, store, cal, regime, cfg, as_of=dates[-1].date())

    assert "quality" in sc.dropped_factors
    assert "lookahead" in sc.dropped_factors["quality"]
    assert "quality" not in sc.effective_weights
    assert sum(sc.effective_weights.values()) == pytest.approx(1.0)


def test_redundancy_is_measured_not_assumed(tmp_path, cfg):
    _composite_only(cfg)
    from prosignal.stages import stage2_regime
    dates = _dates()
    syms = [f"S{i}" for i in range(6)]
    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_prices(pd.concat(
        [_prices([s], dates, drift=0.0004 * i, seed=i, turnover=1e10)
         for i, s in enumerate(syms)], ignore_index=True))
    store.write_indices(pd.DataFrame([
        {DATE: d, "index_name": "NIFTY 200", "open": 100.0, "high": 100.0,
         "low": 100.0, "close": 100.0, "volume": 0.0, "source": "t"} for d in dates
    ]))
    cal = TradingCalendar([d.date() for d in dates])
    uni = UniverseSnapshot("NIFTY 200", dates[-1].date(), syms, {s: "S" for s in syms})
    el = stage3_eligibility.run(uni, store, cal, _clean_quality(), cfg, as_of=dates[-1].date())
    regime = stage2_regime.run(store, cal, el.eligible_universe, cfg, as_of=dates[-1].date())
    sc = stage4_core_score.run(el, store, cal, regime, cfg, as_of=dates[-1].date())

    assert sc.redundancy.pairwise_spearman, "correlation must be measured"
    for rho in sc.redundancy.pairwise_spearman.values():
        assert -1.0 <= rho <= 1.0


# =============================================================================
# stage 7 -- risk
# =============================================================================


def _frame(vol=0.02, price=1000.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    path = price * np.exp(np.cumsum(rng.normal(0.0005, vol, n)))
    dates = _dates(n)
    return pd.DataFrame({
        DATE: dates, "open": path, "high": path * (1 + vol),
        "low": path * (1 - vol), "close": path, "volume": 500_000.0,
        "turnover": 5e8,
    })


def test_stop_scales_with_volatility_not_a_flat_percent(cfg):
    costs = CostModel(cfg)
    calm = stage7_risk.build_plan("CALM", _frame(vol=0.005), 1000.0, 0.8, 5e8, cfg, costs)
    wild = stage7_risk.build_plan("WILD", _frame(vol=0.05, seed=1), 1000.0, 0.8, 5e8, cfg, costs)
    assert wild.stop_distance_pct > calm.stop_distance_pct


def test_stop_respects_the_configured_floor_and_ceiling(cfg):
    costs = CostModel(cfg)
    lo = float(cfg.params.stage7_risk.stop_loss.min_stop_distance_pct.value)
    hi = float(cfg.params.stage7_risk.stop_loss.max_stop_distance_pct.value)
    for v_, seed in [(0.0005, 2), (0.12, 3)]:
        plan = stage7_risk.build_plan("X", _frame(vol=v_, seed=seed), 1000.0, 0.8, 5e8, cfg, costs)
        assert lo - 1e-9 <= plan.stop_distance_pct <= hi + 1e-9


def test_invalidation_is_distinct_from_the_stop(cfg):
    plan = stage7_risk.build_plan("X", _frame(), 1000.0, 0.8, 5e8, cfg, CostModel(cfg))
    assert plan.invalidation_level is not None
    assert plan.invalidation_level != plan.stop_price
    assert "structure" in (plan.invalidation_basis or "")


def test_liquidity_can_be_the_binding_position_constraint(cfg):
    """The engine must be able to say: good trade, but you cannot size into it."""
    costs = CostModel(cfg)
    plan = stage7_risk.build_plan("TINY", _frame(), 1000.0, 0.9, 1e6, cfg, costs)
    assert plan.risk_category_inputs["qty_by_liquidity"] <= plan.risk_category_inputs["qty_by_slot"]
    assert any("Liquidity is the binding constraint" in n for n in plan.notes)


def test_targets_are_r_multiples_of_the_actual_stop(cfg):
    plan = stage7_risk.build_plan("X", _frame(), 1000.0, 0.8, 5e8, cfg, CostModel(cfg))
    risk = plan.reference_price - plan.stop_price
    t2_r = float(cfg.params.stage7_risk.targets.t2_r_multiple.value)
    assert plan.target_2 == pytest.approx(plan.reference_price + t2_r * risk, abs=0.1)
    assert plan.reward_to_risk_t2 == pytest.approx(t2_r, abs=0.05)


def test_exit_hierarchy_puts_thesis_invalidation_first(cfg):
    plan = stage7_risk.build_plan("X", _frame(), 1000.0, 0.8, 5e8, cfg, CostModel(cfg))
    assert plan.exit_conditions[0].reason.value == "thesis_invalidation"
    assert plan.exit_conditions[0].priority == 1


def test_cost_estimate_is_attached_to_the_plan(cfg):
    plan = stage7_risk.build_plan("X", _frame(), 1000.0, 0.8, 5e8, cfg, CostModel(cfg))
    assert plan.estimated_round_trip_cost_bps and plan.estimated_round_trip_cost_bps > 0


# =============================================================================
# stage 6 -- entry
# =============================================================================


def test_no_trigger_yields_watchlist_not_buy(cfg):
    """A good stock at a bad price is a bad trade."""
    frame = _frame(n=200)
    frame["volume"] = 100.0  # no volume confirmation possible
    rep = stage6_entry.run(["X"], {"X": frame}, cfg, dt.date(2026, 1, 1))
    assert rep.decisions["X"].status is not EntryStatus.TRIGGERED


def test_breakout_trigger_fires_on_a_real_breakout(cfg):
    n = 200
    dates = _dates(n)
    base = np.full(n, 100.0)
    base[-1] = 118.0  # decisive break of the range
    vols = np.full(n, 100_000.0)
    vols[-1] = 900_000.0
    frame = pd.DataFrame({
        DATE: dates, "open": base, "high": base * 1.001, "low": base * 0.999,
        "close": base, "volume": vols, "turnover": base * vols,
    })
    # Admission is by rank now; the trigger describes the price structure and
    # sets the entry zone but no longer decides. Rank 1 is inside any band.
    rep = stage6_entry.run(["B"], {"B": frame}, cfg, dates[-1].date(),
                           ranks={"B": 1}, held=[])
    d = rep.decisions["B"]
    assert d.status is EntryStatus.TRIGGERED
    # Some structure is recognised. Which one is an ordering detail -- the
    # triggers are tried pullback, reclaim, breakout and the first match wins.
    assert d.trigger_type is not TriggerType.NONE
    assert d.entry_zone is not None and d.entry_zone[0] < d.entry_zone[1]


def test_a_breakout_outside_the_entry_band_is_not_bought(cfg):
    """The trigger firing is not a reason to buy. Rank is."""
    n = 200
    dates = _dates(n)
    base = np.full(n, 100.0)
    base[-1] = 118.0
    vols = np.full(n, 100_000.0)
    vols[-1] = 900_000.0
    frame = pd.DataFrame({
        DATE: dates, "open": base, "high": base * 1.001, "low": base * 0.999,
        "close": base, "volume": vols, "turnover": base * vols,
    })
    entry = int(cfg.params.stage6_entry.admission.entry_rank.value)
    rep = stage6_entry.run(["B"], {"B": frame}, cfg, dates[-1].date(),
                           ranks={"B": entry + 5}, held=[])
    d = rep.decisions["B"]
    assert d.status is EntryStatus.WATCHLIST
    assert d.trigger_type is not TriggerType.NONE   # still detected, still reported


def test_a_held_name_survives_between_the_two_bands(cfg):
    """Hysteresis: inside the exit band a held name stays, an unheld one does not."""
    n = 200
    dates = _dates(n)
    base = np.full(n, 100.0) * (1 + np.arange(n) * 0.001)
    frame = pd.DataFrame({
        DATE: dates, "open": base, "high": base * 1.001, "low": base * 0.999,
        "close": base, "volume": np.full(n, 100_000.0), "turnover": base * 100_000.0,
    })
    adm = cfg.params.stage6_entry.admission
    between = (int(adm.entry_rank.value) + int(adm.exit_rank.value)) // 2

    held = stage6_entry.run(["B"], {"B": frame}, cfg, dates[-1].date(),
                            ranks={"B": between}, held=["B"])
    fresh = stage6_entry.run(["B"], {"B": frame}, cfg, dates[-1].date(),
                             ranks={"B": between}, held=[])
    assert held.decisions["B"].status is EntryStatus.TRIGGERED
    assert fresh.decisions["B"].status is EntryStatus.WATCHLIST, (
        "without hysteresis a name drifting across one boundary is bought and "
        "sold at every rebalance for no change in view"
    )


# =============================================================================
# end-to-end
# =============================================================================


def test_pipeline_runs_on_the_real_store_and_is_reproducible(live_cfg):
    """The full eight-stage run against real ingested NSE data."""
    from prosignal.pipeline import run_analysis

    a = run_analysis(live_cfg)
    b = run_analysis(live_cfg)

    assert a.output.as_of_date == b.output.as_of_date
    assert a.funnel == b.funnel, "same inputs must produce the same funnel"
    assert [r.ticker for r in a.output.recommendations] == [
        r.ticker for r in b.output.recommendations
    ]
    assert a.output.config_version == live_cfg.version
    assert set(a.output.stage_timings_ms) >= {
        "stage1_data_quality", "stage2_regime", "stage3_eligibility",
        "stage4_core_score", "stage5_false_signal", "stage6_entry",
        "stage7_risk", "stage8_final_signal",
    }


def test_no_trade_reports_the_funnel_not_an_empty_list(live_cfg):
    from prosignal.pipeline import run_analysis

    out = run_analysis(live_cfg).output
    if out.no_trade is None:
        pytest.skip("a BUY was generated today; funnel assertion covered elsewhere")
    nt = out.no_trade
    assert nt.reason
    assert nt.gate_summary["universe_considered"] > 0
    assert "passed_eligibility" in nt.gate_summary
    for c in nt.closest_candidates:
        assert c.gate_failed and c.detail


def test_engine_never_emits_a_probability(live_cfg):
    """Section 23: a weighted score is not a probability.

    Nothing here has been calibrated against realised outcomes, so no field may
    present itself as a likelihood. This test walks the serialised output and
    fails if any key implies one.
    """
    from prosignal.pipeline import run_analysis
    from prosignal.stages.stage8_final_signal import PROBABILITY_UNAVAILABLE

    payload = run_analysis(live_cfg).output.model_dump(mode="json")

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v_ in node.items():
                assert "probabilit" not in k.lower(), f"probability field at {path}.{k}"
                assert k.lower() not in {"confidence", "win_rate", "success_rate"}, (
                    f"calibration-implying field at {path}.{k}"
                )
                walk(v_, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v_ in enumerate(node):
                walk(v_, f"{path}[{i}]")

    walk(payload)
    assert "unavailable" in PROBABILITY_UNAVAILABLE.lower()


def test_every_recommendation_carries_contrarian_evidence(live_cfg):
    """Section 42: the engine must argue against its own candidates."""
    from prosignal.pipeline import run_analysis

    out = run_analysis(live_cfg).output
    cards = out.recommendations + out.watchlist
    if not cards:
        pytest.skip("no cards produced today")
    for rec in cards:
        assert rec.why_this_signal_exists, f"{rec.ticker} has no supporting evidence"
        contrarian = rec.false_signal_flagged or rec.market_regime
        assert contrarian, f"{rec.ticker} has no disconfirming evidence"
        assert rec.unvalidated_parameter_warning


def test_untestable_checks_are_never_reported_as_passed(live_cfg):
    from prosignal.pipeline import run_analysis

    out = run_analysis(live_cfg).output
    cards = out.recommendations + out.watchlist
    if not cards:
        pytest.skip("no cards produced today")
    for rec in cards:
        overlap = set(rec.false_signal_cleared) & set(rec.false_signal_not_testable)
        assert not overlap, f"{rec.ticker}: {overlap} counted as both passed and untestable"
