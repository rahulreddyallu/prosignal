"""Stage 8 -- Final decision, portfolio constraints, and the NO-TRADE report.

Thresholds are never lowered to produce a signal. If nothing clears the gates
the answer is NO TRADE.

No probability is emitted. Nothing here has been calibrated against realised
outcomes, so a "72% chance" would be a weighted factor score presented as a
statistic. The contract carries a signal strength band instead, and the card
states that a probability is unavailable and why. A calibrated out-of-sample
backtest is what would let a real probability attach here.

Every rejection is counted so the NO-TRADE output can show the funnel.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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
    held: Optional[Sequence[str]] = None,
) -> Tuple[List[Recommendation], List[Recommendation], Optional[NoTradeReport]]:
    p = config.params
    cfg = p.stage8_final_signal
    names = company_names or {}
    # The book Stage 6 was asked to maintain. Everything below distinguishes
    # adding a position from keeping one, and the two obey different rules.
    open_book = {str(t) for t in (held or ())}

    gate_counts: Dict[str, int] = {
        "universe_considered": eligibility.universe_considered,
        "passed_eligibility": len(eligibility.eligible_universe),
        "scored": len(scores.ranked_scores),
        "defended": len(defense.per_stock),
        "survived_defense": 0,
        "passed_score_threshold": 0,
        "triggered": 0,
        "passed_portfolio_limits": 0,
    }

    # -- market-wide blocks -------------------------------------------------
    # These stop NEW entries. They do NOT close the book, and the difference is
    # the whole of this stage's contract with Stage 6.
    #
    # Returning an empty `buys` list here used to be how a risk-off regime was
    # expressed. But `buys` IS the book: the ledger records it as
    # `signals_generated`, and the next run rebuilds Stage 6's `held` set from
    # that record. So one blocked session did not pause the strategy, it
    # liquidated it -- every open position was dropped without an exit, the
    # hysteresis state was destroyed, and the run after the block started from
    # an empty book. Measured on the recorded ledger: 10 of 63 runs blocked,
    # and book turnover of 89.3% against a strategy validated at a median hold
    # far longer than one session.
    #
    # A regime that forbids opening positions says nothing about closing them.
    # What closes a position is Stage 6's exit band, and only that.
    blocked_reason: Optional[str] = None
    if not regime.allow_new_entries:
        blocked_reason = (
            f"Market regime '{regime.regime_bucket}' blocks new entries. "
            f"{regime.block_reason or ''}"
        ).strip()
    elif defense.market_halt:
        blocked_reason = f"Market-wide defense halt: {defense.market_halt_reason}"

    min_score = fv(cfg.scarcity.min_composite_score)
    min_pct = fv(cfg.scarcity.min_universe_percentile)

    # A flat day. The percentile gate cannot express one: the score is a rank,
    # so its distribution is uniform every session and `min_universe_percentile`
    # admits the top 10% whether or not the top 10% is any better than the
    # middle. This reads the model's RAW spread instead, and a day where it
    # ordered the universe without distinguishing it produces no signal rather
    # than a shortlist of noise.
    #
    # It does NOT close the book. A day with no view is a day to add nothing,
    # not a day to liquidate -- what closes a position is the Stage 6 exit band.
    min_ratio = fv(cfg.scarcity.min_dispersion_ratio)
    dispersion = scores.prediction_dispersion
    typical = scores.typical_dispersion
    if (blocked_reason is None and min_ratio > 0
            and dispersion is not None and typical):
        ratio = dispersion / typical
        if ratio < min_ratio:
            blocked_reason = (
                f"The model separated the universe by {dispersion:.4f} between "
                f"its top decile and its median, against the {typical:.4f} it "
                f"normally manages -- {ratio:.0%} of its usual spread, below the "
                f"{min_ratio:.0%} floor. It ranked the names without "
                f"distinguishing them, so today's ordering carries no view."
            )

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
    accepted: List[object] = []

    # Stage 5 can demote a name, and the score it produces is what every gate
    # below compares against. Ordering by the pre-defense rank left a penalised
    # candidate sitting above names that scored higher after the argument
    # against it was heard, so the column and the position disagreed. Only the
    # defended set is reordered: names outside it were never tested, and an
    # untested name has not earned a place above a tested one.
    def _final_of(sym: str) -> float:
        res = defense.per_stock.get(sym)
        return res.score_after if res is not None else float("-inf")

    survivors = sorted(survivors, key=_final_of, reverse=True)
    #: Display position among the defended set. Fixed here from the single
    #: score ordering so that it does not depend on which pass admits a name.
    positions = {sym: i for i, sym in enumerate(survivors, start=1)}

    def _build(sym: str) -> Optional[Recommendation]:
        score = by_ticker.get(sym)
        decision = entries.decisions.get(sym)
        if score is None or decision is None:
            return None
        defense_res = defense.per_stock[sym]
        return _card(sym, names.get(sym), score, defense_res, decision,
                     plans.get(sym), regime, eligibility, scores,
                     defense_res.score_after, cfg, position=positions[sym])

    # -- the score gate, applied once, in score order ------------------------
    # Score gate first, then the entry trigger. Counting the trigger before
    # the score meant the two lines measured different populations and the
    # funnel ran backwards: triggered=1 followed by passed_score=8. Counted
    # in decision order it is monotonic and reads as what it is.
    # These read as two independent gates and are one. percentile is
    # rank_to_unit_interval(score) * 100, so min_composite_score = 0.60 is
    # arithmetically percentile >= 60 and is already implied by
    # min_universe_percentile = 90. Measured over 27,478 scored names in the
    # holdout: 8,244 pass the score gate and fail the percentile gate, and
    # zero pass the percentile gate and fail the score gate. Both are kept
    # -- a Stage 5 penalty lands on final_score and can pull it below 0.60
    # while the pre-defence percentile stands -- but only one of them
    # selects, and tuning the other has no effect.
    qualified: List[str] = []
    for sym in survivors:
        score = by_ticker.get(sym)
        decision = entries.decisions.get(sym)
        if score is None or decision is None:
            continue
        if defense.per_stock[sym].score_after < min_score or score.percentile < min_pct:
            continue
        gate_counts["passed_score_threshold"] += 1
        qualified.append(sym)
        if decision.status is EntryStatus.TRIGGERED:
            gate_counts["triggered"] += 1

    def _accept(rec: Recommendation, sym: str, sector: str) -> None:
        gate_counts["passed_portfolio_limits"] += 1
        sector_used[sector] = sector_used.get(sector, 0) + 1
        accepted_symbols.append(sym)
        accepted.append(by_ticker[sym])
        buys.append(rec)

    def _sector_of(sym: str) -> str:
        # An unknown sector is not evidence that two names share one. Pooling
        # every unclassified name into a single "Unknown" bucket would cap the
        # whole run at max_per_sector for a reason that is a gap in reference
        # data, not concentration. Each unclassified name gets its own key; the
        # pairwise correlation cap below still applies to all of them, and it is
        # computed from prices, so it never depends on a membership file.
        sector = by_ticker[sym].sector or ""
        if not sector or sector == "Unknown":
            return f"Unclassified:{sym}"
        return sector

    # ======================================================================
    # PASS 1 -- the existing book.
    #
    # Every constraint below this stage owns is an ENTRY constraint: the sector
    # cap, the pairwise correlation cap and max_signals_per_run all answer "may
    # this position be opened?". They were being applied to names the engine
    # had already opened, in score order, as though each session were the first.
    # A held name that drifted below a fresher one in its sector was demoted to
    # WATCH -- and because the ledger's `signals_generated` is the only record
    # of the book, that demotion deleted the position with no exit recorded and
    # no way for the next run to know it had ever existed.
    #
    # Measured on the recorded ledger, restricted to ADJACENT sessions so that
    # backfill runs months apart are not counted as a position being held: of
    # 54 held-name transitions only 12 stayed in the book, and 19 were demoted
    # this way. Stage 6's hysteresis was correct and this stage threw it away.
    #
    # What closes a position is Stage 6's exit band. If Stage 6 still admits the
    # name, it is held here, and it seeds the constraint state so that new
    # entries are measured against the book that actually exists.
    # ======================================================================
    # Every name pass 1 disposes of, whether it accepted it or set it aside.
    # Tracking only the accepted ones let a held name the cap dropped fall
    # through to pass 2 and be appended to `watch` a second time.
    settled: set = set()
    for sym in qualified:
        if sym not in open_book:
            continue
        if entries.decisions[sym].status is not EntryStatus.TRIGGERED:
            # Stage 6 closed it: the rank left the exit band. It falls through
            # to pass 2 and is reported as monitored, which is what it now is.
            continue
        rec = _build(sym)
        if rec is None:
            continue
        if len(buys) >= max_signals:
            # Only reachable when max_signals_per_run is lowered below a book
            # that is already larger -- in steady state the entry band bounds
            # the book at or under this cap, so this never fires. When it does,
            # the book shrinks from the bottom and says so, rather than growing
            # without limit or being cut by whichever name happened to sort
            # last.
            rec.decision = Decision.WATCHLIST
            rec.why_this_signal_exists.append(
                f"Closed: the book is capped at {max_signals} and this was the "
                f"lowest-scoring position held. The cap was reduced below the "
                f"size of the existing book."
            )
            watch.append(rec)
            settled.add(sym)
            continue
        reason = (entries.decisions[sym].reason or "").strip()
        rec.why_this_signal_exists.append(
            " ".join(filter(None, [
                "Held from a previous run.",
                reason if not reason.endswith(".") else reason,
                "The sector, correlation and book-size limits govern what may "
                "be opened; they do not close a position that is already open.",
            ]))
        )
        _accept(rec, sym, _sector_of(sym))
        settled.add(sym)

    # ======================================================================
    # PASS 2 -- new entries, measured against the book above.
    # ======================================================================
    for sym in qualified:
        if sym in settled:
            continue
        rec = _build(sym)
        if rec is None:
            continue
        decision = entries.decisions[sym]
        score = by_ticker[sym]

        if decision.status is not EntryStatus.TRIGGERED:
            rec.decision = Decision.WATCHLIST
            watch.append(rec)
            continue

        if blocked_reason is not None:
            rec.decision = Decision.WATCHLIST
            rec.why_this_signal_exists.append(
                f"Not opened: {blocked_reason} Positions already held are "
                f"unaffected -- what closes one is the Stage 6 exit band."
            )
            watch.append(rec)
            continue

        # portfolio limits apply only to actual BUYs
        sector = _sector_of(sym)
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

        # Pairwise correlation and the sector cap both look at names two at a
        # time. Five names can clear every pair and every sector and still be
        # the same bet: all high-momentum, all high-beta, differently labelled.
        # This looks at the basket as a whole.
        # Aggregate exposure is REPORTED, not gated. It used to reject, and
        # measured under rank admission that rejected 5 of 8 names and cut the
        # book to 3.9 on the selection period:
        #
        #                        selection SR   holdout SR   selection maxDD
        #   no gate                  +0.86         +1.56        -10.00%
        #   gate at 0.75 (was)       +0.45         +1.45         -9.76%
        #
        # Drawdown barely moved at any limit tested, so the constraint bought no
        # risk reduction and paid for it in breadth -- Grinold again, on a book
        # the gate cut nearly in half. The deeper problem is that it was fighting
        # the model: momentum is 41% of this model's IC, so the top names load
        # high on momentum by construction, and capping mean momentum loading
        # rejects the basket for being what the model is designed to produce.
        # The operator is still told, which is the part that had value.
        exposure = _aggregate_exposure(accepted + [score], _EXPOSURE_FACTORS)
        breached = [n for n, v in exposure.items() if v >= _EXPOSURE_LIMIT]
        if breached:
            rec.why_this_signal_exists.append(
                f"Concentration note: with this name the basket's mean "
                f"{', '.join(sorted(breached))} loading reaches "
                f"{_EXPOSURE_LIMIT:+.2f} or more. Every pair passes and every "
                f"sector passes; taken together they are one macro position. "
                f"This is reported, not blocked -- blocking it was measured at "
                f"selection Sharpe +0.45 against +0.86."
            )

        if len(buys) >= max_signals:
            rec.decision = Decision.WATCHLIST
            watch.append(rec)
            continue

        _accept(rec, sym, sector)


    # The funnel's last line. It used to be added by the pipeline only on the
    # path where no_trade was absent, so a run that reported a block carried a
    # funnel that stopped one step short -- and a blocked regime can now hold a
    # live book, which makes that the exact case where the count matters.
    gate_counts["buys"] = len(buys)

    if buys:
        log.info("stage 8 complete",
                 extra={"buys": len(buys), "watch": len(watch),
                        "held": len(open_book & set(accepted_symbols)),
                        "blocked": blocked_reason is not None})
        # A live book with new entries blocked is not NO TRADE -- it is a book
        # that is not being added to. Saying "no trade" over five open positions
        # would be false, and saying nothing would hide the block.
        if blocked_reason is not None:
            return buys, watch, _no_trade(
                f"{blocked_reason} No new positions were opened. The "
                f"{len(buys)} position(s) already held are unchanged: what "
                f"closes one is the Stage 6 exit band, not the regime.",
                scores, gate_counts, cfg, defense=defense, entries=entries,
            )
        return buys, watch, None

    if blocked_reason is not None:
        return [], watch, _no_trade(
            f"{blocked_reason} No position was open to carry.",
            scores, gate_counts, cfg, defense=defense, entries=entries,
        )

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
          scores, final_score, cfg, position: int = 0) -> Recommendation:
    """Build the recommendation, including the evidence AGAINST it."""
    why: List[str] = []
    model_tier = [f for f in score.factors.values() if f.evidence_tier == "model"]
    if model_tier:
        # Attribution from the fit. raw_value is this factor's contribution to
        # the score and standardised is the z-score it came from, so the reader
        # can see both how unusual the name is and how much that mattered. Only
        # the terms that moved it are listed; the rest are near zero and would
        # bury the ones that did.
        # A theme the estimator gated out carries coefficient EXACTLY zero. It
        # did not contribute nothing; it was not used. Printing it as
        # "+ lottery: -0.82 sd, raises the score by 0.0000" next to a citation
        # reads as though the model consulted it and found it neutral, which is
        # the opposite of what happened -- the significance floor removed it.
        priced = [f for f in score.factors.values()
                  if f.evidence_tier == "model" and f.raw_value is not None
                  and abs(f.weight or 0.0) > 1e-12]
        gated = [f for f in score.factors.values()
                 if f.evidence_tier == "model" and abs(f.weight or 0.0) <= 1e-12]
        shown = sorted(priced, key=lambda f: -abs(f.raw_value))[:6]
        for f in shown:
            direction = "raises" if f.raw_value >= 0 else "lowers"
            z = f"{f.standardised:+.2f} sd" if f.standardised is not None else "n/a"
            why.append(
                f"{f.name}: {z} vs the universe, {direction} the score by "
                f"{abs(f.raw_value):.4f} (coefficient {f.weight:+.5f}) ({f.citation})"
            )
        total = sum(abs(f.raw_value) for f in priced)
        listed = sum(abs(f.raw_value) for f in shown)
        if total > 0 and len(shown) < len(priced):
            why.append(
                f"These {len(shown)} of {len(priced)} priced factors carry "
                f"{listed / total:.0%} of the movement; the rest are near zero."
            )
        if gated:
            names = ", ".join(sorted(f.name for f in gated))
            why.append(
                f"Not used in this fit: {names}. The estimator could not measure "
                f"them past its significance floor on their own training window, "
                f"so they were set to zero rather than given a weight the data "
                f"did not support."
            )
    else:
        for fname, f in score.factors.items():
            if f.raw_value is None:
                continue
            why.append(
                f"{fname}: raw {f.raw_value:+.2%}, universe rank "
                f"{score.percentile:.0f}th, weight {f.weight:.0%} [{f.evidence_tier}] "
                f"({f.citation})"
            )
    if abs(final_score - defense_res.score_before) > 1e-9:
        why.append(
            f"Score {final_score:.3f} after Stage 5, from {defense_res.score_before:.3f} "
            f"before. The model placed it #{score.rank} of {scores.universe_size} "
            f"eligible names; among the defended candidates it now sits #{position}."
        )
    else:
        why.append(
            f"Score {final_score:.3f} ranks #{score.rank} of "
            f"{scores.universe_size} eligible names, unchanged by Stage 5."
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
        rank=position or score.rank,
        model_rank=score.rank,
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


#: Factors whose combined loading defines a macro bet. Momentum and beta are
#: the two that concentrate without showing up pairwise: two names can correlate
#: at 0.4 and still both sit in the top decile of each.
_EXPOSURE_FACTORS = ("mom_6_1", "resid_mom", "prox_52w", "beta_120")

#: Mean standardised loading at which the basket stops being diversified. Ranks
#: run -1 to +1, so 0.75 means the average name sits in roughly the top eighth
#: of the universe on that factor.
_EXPOSURE_LIMIT = 0.75


def _aggregate_exposure(scores, factors) -> Dict[str, float]:
    """Mean loading across a candidate basket, per factor.

    Reads the standardised value the model itself used, so the check cannot
    drift from the scoring. A factor absent from the attribution is skipped
    rather than counted as zero, which would dilute a real concentration.
    """
    out: Dict[str, float] = {}
    if not scores:
        return out
    for name in factors:
        values = []
        for score in scores:
            factor = (getattr(score, "factors", None) or {}).get(name)
            if factor is None:
                continue
            loading = getattr(factor, "standardised", None)
            if loading is not None and pd.notna(loading):
                values.append(float(loading))
        if values:
            out[name] = float(np.mean(values))
    return out


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
