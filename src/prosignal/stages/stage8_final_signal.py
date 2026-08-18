"""Stage 8 -- Final decision, portfolio constraints, and the NO-TRADE report.

Two things this stage will not do.

**It will not lower a threshold to produce a signal.** If nothing clears the
gates, the answer is NO TRADE, and that is a successful outcome rather than a
failure to find something.

**It will not emit a probability.** Nothing in this engine has been calibrated
against realised outcomes, so any number presented as "72% chance" would be a
weighted factor score wearing a statistical costume. The contract carries a
signal STRENGTH BAND instead, and the card says explicitly that a probability is
unavailable and why. When a backtest with out-of-sample calibration exists, and
only then, this is where a real probability would attach.

Every rejection is counted, so the NO-TRADE output can show the funnel rather
than shrugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ._cfg import fv, iv
from ..core.contracts import (
    ClosestCandidate,
    CoreScoreReport,
    EligibilityReport,
    EntryReport,
    FalseSignalReport,
    NoTradeReport,
    Recommendation,
    RegimeState,
    RiskPlan,
)
from ..core.enums import Decision, EntryStatus, StrengthBand
from ..core.logging import get_logger

__all__ = ["run", "STAGE_NAME", "PROBABILITY_UNAVAILABLE"]

STAGE_NAME = "stage8_final_signal"
log = get_logger(__name__)

#: Printed wherever a probability would otherwise go.
PROBABILITY_UNAVAILABLE = (
    "Probability estimate unavailable: no out-of-sample calibration exists yet. "
    "The composite score below is a RANK within today's eligible universe, not a "
    "likelihood of profit, and it must not be read as one."
)


def run(
    regime: RegimeState,
    eligibility: EligibilityReport,
    scores: CoreScoreReport,
    defense: FalseSignalReport,
    entries: EntryReport,
    plans: Dict[str, RiskPlan],
    closes: pd.DataFrame,
    config,
    company_names: Optional[Dict[str, str]] = None,
) -> Tuple[List[Recommendation], List[Recommendation], Optional[NoTradeReport]]:
    p = config.params
    cfg = p.stage8_final_signal
    names = company_names or {}

    gate_counts: Dict[str, int] = {
        "universe_considered": eligibility.universe_considered,
        "passed_eligibility": len(eligibility.eligible_universe),
        "scored": len(scores.ranked_scores),
        "defended": len(defense.per_stock),
        "survived_defense": 0,
        "triggered": 0,
        "passed_score_threshold": 0,
        "passed_regime_gate": 0,
        "passed_portfolio_limits": 0,
    }

    # -- market-wide blocks -------------------------------------------------
    if not regime.allow_new_entries:
        return [], [], _no_trade(
            f"Market regime '{regime.regime_bucket}' blocks new entries. "
            f"{regime.block_reason or ''}".strip(),
            scores, gate_counts, cfg,
        )
    if defense.market_halt:
        return [], [], _no_trade(
            f"Market-wide defense halt: {defense.market_halt_reason}",
            scores, gate_counts, cfg,
        )

    min_score = fv(cfg.scarcity.min_composite_score)
    min_pct = fv(cfg.scarcity.min_universe_percentile)

    survivors: List[str] = []
    for sym, res in defense.per_stock.items():
        if res.final_status != "REJECTED":
            survivors.append(sym)
    gate_counts["survived_defense"] = len(survivors)

    by_ticker = {s.ticker: s for s in scores.ranked_scores}
    buys: List[Recommendation] = []
    watch: List[Recommendation] = []

    sector_used: Dict[str, int] = {}
    max_per_sector = iv(cfg.portfolio.max_signals_per_sector)
    max_signals = iv(cfg.portfolio.max_signals_per_run)
    max_corr = fv(cfg.portfolio.max_pairwise_correlation)
    corr_lb = iv(cfg.portfolio.correlation_lookback_sessions)
    accepted_symbols: List[str] = []

    for sym in survivors:
        score = by_ticker.get(sym)
        decision = entries.decisions.get(sym)
        if score is None or decision is None:
            continue

        defense_res = defense.per_stock[sym]
        final_score = defense_res.score_after

        if decision.status is EntryStatus.TRIGGERED:
            gate_counts["triggered"] += 1

        # score / percentile gate
        if final_score < min_score or score.percentile < min_pct:
            continue
        gate_counts["passed_score_threshold"] += 1
        gate_counts["passed_regime_gate"] += 1

        plan = plans.get(sym)
        rec = _card(sym, names.get(sym), score, defense_res, decision, plan,
                    regime, eligibility, scores, final_score, cfg)

        if decision.status is not EntryStatus.TRIGGERED:
            watch.append(rec)
            continue

        # portfolio limits apply only to actual BUYs
        sector = score.sector or "Unknown"
        if sector_used.get(sector, 0) >= max_per_sector:
            rec.decision = Decision.WATCHLIST
            rec.why_this_signal_exists.append(
                f"Downgraded to WATCH: already {max_per_sector} signal(s) in {sector}. "
                f"Two names in one sector are not two independent bets."
            )
            watch.append(rec)
            continue

        corr = _max_correlation(sym, accepted_symbols, closes, corr_lb)
        if corr is not None and corr > max_corr:
            rec.decision = Decision.WATCHLIST
            rec.why_this_signal_exists.append(
                f"Downgraded to WATCH: {corr:.2f} correlation with an accepted "
                f"signal, above the {max_corr:.2f} limit."
            )
            watch.append(rec)
            continue

        if len(buys) >= max_signals:
            rec.decision = Decision.WATCHLIST
            watch.append(rec)
            continue

        gate_counts["passed_portfolio_limits"] += 1
        sector_used[sector] = sector_used.get(sector, 0) + 1
        accepted_symbols.append(sym)
        buys.append(rec)

    if buys:
        log.info("stage 8 complete", extra={"buys": len(buys), "watch": len(watch)})
        return buys, watch, None

    return [], watch, _no_trade(
        "No candidate cleared every gate. This is the designed outcome when the "
        "evidence does not justify risking capital.",
        scores, gate_counts, cfg, defense=defense, entries=entries,
    )


# =============================================================================
def _band(score: float, cfg) -> StrengthBand:
    if score >= fv(cfg.strength_bands.high_min):
        return StrengthBand.HIGH
    if score >= fv(cfg.strength_bands.medium_min):
        return StrengthBand.MEDIUM
    return StrengthBand.LOW


def _card(sym, name, score, defense_res, decision, plan, regime, eligibility,
          scores, final_score, cfg) -> Recommendation:
    """Build the recommendation, including the evidence AGAINST it."""
    why: List[str] = []
    for fname, f in score.factors.items():
        if f.raw_value is None:
            continue
        why.append(
            f"{fname}: raw {f.raw_value:+.2%}, universe rank "
            f"{score.percentile:.0f}th, weight {f.weight:.0%} [{f.evidence_tier}] "
            f"({f.citation})"
        )
    why.append(
        f"Composite {final_score:.3f} ranks #{score.rank} of "
        f"{scores.universe_size} eligible names."
    )

    cleared = [c.check for c in defense_res.passed()]
    flagged = [
        f"{c.check}: {c.reason} (-{c.penalty:.2f})" for c in defense_res.penalised()
    ]
    untestable = [f"{c.check}: {c.reason}" for c in defense_res.not_testable()]

    # Contrarian evidence is mandatory, not decorative.
    against: List[str] = list(flagged)
    if regime.transition_flag:
        against.append(
            f"Regime is in transition ({', '.join(regime.transition_components)}); "
            f"historical relationships are least reliable now."
        )
    if regime.breadth_state.value == "Weak":
        against.append(
            f"Market breadth is weak ({regime.breadth_pct_above_ma:.0f}% above the "
            f"long-term average) -- a narrow advance is a poor base for momentum."
        )
    if regime.breadth_divergence_flag:
        against.append("Breadth divergence: index at highs while participation falls.")
    if defense_res.total_penalty > 0:
        against.append(
            f"Cumulative defense penalty {defense_res.total_penalty:.2f} reduced the "
            f"score from {defense_res.score_before:.3f} to {defense_res.score_after:.3f}."
        )
    if untestable:
        against.append(
            f"{len(untestable)} defense check(s) could not run at all, so this "
            f"signal rests on partial evidence."
        )
    if not against:
        against.append(
            "No disconfirming evidence was found among the checks that could run. "
            "That is not the same as no risk."
        )

    sell: List[str] = []
    if plan:
        for e in plan.exit_conditions:
            lvl = f" (Rs {e.level:,.2f})" if e.level else ""
            sell.append(f"{e.priority}. {e.reason.value}{lvl} -- {e.description}")

    return Recommendation(
        ticker=sym,
        company_name=name,
        sector=score.sector,
        decision=Decision.BUY_CANDIDATE if decision.status is EntryStatus.TRIGGERED
        else Decision.WATCHLIST,
        signal_strength_band=_band(final_score, cfg),
        regime_compatibility=regime.compatibility(),
        expected_holding_period=(
            f"{plan.expected_holding_sessions[0]}-{plan.expected_holding_sessions[1]} sessions"
            if plan else "unknown"
        ),
        entry_zone=decision.entry_zone,
        invalidation_level=plan.invalidation_level if plan else None,
        initial_stop=plan.stop_price if plan else None,
        target_1=plan.target_1 if plan else None,
        target_2=plan.target_2 if plan else None,
        position_risk_category=plan.risk_category if plan else None,
        last_close=decision.reference_price,
        composite_score=round(final_score, 4),
        universe_percentile=round(score.percentile, 1),
        rank=score.rank,
        why_this_signal_exists=why,
        market_regime=[
            f"Bucket {regime.regime_bucket}; trend {regime.trend_regime.value}; "
            f"volatility {regime.vol_tercile.value}/{regime.vol_context.value}",
            f"Breadth {regime.breadth_pct_above_ma:.0f}% above long-term average"
            if regime.breadth_pct_above_ma is not None else "Breadth not measurable",
        ],
        sector_state=[f"Sector: {score.sector or 'Unknown'}"],
        confirmation=decision.confirmations_passed,
        false_signal_cleared=cleared,
        false_signal_flagged=flagged,
        false_signal_not_testable=untestable + decision.confirmations_not_testable,
        sell_conditions=sell,
        research_basis=[
            f.citation for f in score.factors.values() if f.citation
        ],
        data_quality_note=(
            [f"Not testable: {', '.join(eligibility.not_testable.get(sym, []))}"]
            if eligibility.not_testable.get(sym) else []
        ),
        factor_detail=dict(score.factors),
        cost_note=(
            f"Round-trip cost {plan.estimated_round_trip_cost_bps:.0f} bps, of which "
            f"~{plan.estimated_impact_bps:.0f} bps is modelled impact."
            if plan and plan.estimated_round_trip_cost_bps else None
        ),
    )


def _max_correlation(sym, accepted, closes, lookback) -> Optional[float]:
    if not accepted or closes is None or closes.empty or sym not in closes.columns:
        return None
    rets = closes.tail(lookback + 1).pct_change(fill_method=None).dropna()
    if sym not in rets.columns or len(rets) < 20:
        return None
    best = None
    for other in accepted:
        if other not in rets.columns:
            continue
        c = rets[sym].corr(rets[other])
        if pd.notna(c):
            best = c if best is None else max(best, float(c))
    return best


def _no_trade(reason, scores, gate_counts, cfg, defense=None, entries=None) -> NoTradeReport:
    """The funnel, not a shrug."""
    n = iv(cfg.no_trade.show_closest_n)
    closest: List[ClosestCandidate] = []
    for s in scores.ranked_scores[:n]:
        gate = "score/percentile threshold"
        detail = (
            f"composite {s.composite_score:.3f} vs required "
            f"{fv(cfg.scarcity.min_composite_score):.2f}; percentile "
            f"{s.percentile:.0f} vs required {fv(cfg.scarcity.min_universe_percentile):.0f}"
        )
        if defense and s.ticker in defense.per_stock:
            d = defense.per_stock[s.ticker]
            if d.final_status == "REJECTED":
                hard = d.rejected()
                gate = "false-signal defense"
                detail = hard[0].reason if hard else f"penalty {d.total_penalty:.2f}"
        if entries and s.ticker in entries.decisions:
            dec = entries.decisions[s.ticker]
            if dec.status is not EntryStatus.TRIGGERED and gate == "score/percentile threshold":
                gate = "no entry trigger"
                detail = dec.reason or "no trigger active"
        closest.append(ClosestCandidate(
            ticker=s.ticker, composite_score=round(s.composite_score, 4),
            rank=s.rank, gate_failed=gate, detail=detail,
        ))

    return NoTradeReport(
        reason=reason,
        closest_candidates=closest,
        eligible_universe_size=gate_counts.get("passed_eligibility", 0),
        scored_count=gate_counts.get("scored", 0),
        survived_defense_count=gate_counts.get("survived_defense", 0),
        triggered_count=gate_counts.get("triggered", 0),
        gate_summary=gate_counts,
    )
