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
from ..tradeplan import build_trade_plan

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
) -> Tuple[List[Recommendation], List[Recommendation], Optional[NoTradeReport],
             Dict[str, int]]:
    """Returns (buys, watchlist, no_trade, gate_counts).

    `gate_counts` is returned on EVERY path, not only the no-trade one. The
    pipeline used to rebuild the funnel by hand when a trade was produced --
    reading `entries.triggered()` for the trigger count, which is the population
    BEFORE the score gate. That is the exact non-monotonic funnel this stage
    documents having fixed, still being displayed on the path that matters most.
    """
    p = config.params
    cfg = p.stage8_final_signal
    names = company_names or {}
    # The book Stage 6 was asked to maintain. Everything below distinguishes
    # adding a position from keeping one, and the two obey different rules.
    open_book = {str(t) for t in (held or ())}

    # In NARROWING ORDER. This dict is the funnel the interface renders, and a
    # key appended later lands at the bottom -- which is how a gate that runs in
    # the middle of the loop ends up displayed below the ones it precedes,
    # making the funnel look non-monotonic when it is not.
    gate_counts: Dict[str, int] = {
        "universe_considered": eligibility.universe_considered,
        "passed_eligibility": len(eligibility.eligible_universe),
        "scored": len(scores.ranked_scores),
        "defended": len(defense.per_stock),
        "survived_defense": 0,
        "passed_score_threshold": 0,
    }
    if fv(cfg.scarcity.min_win_probability) > 0:
        gate_counts["passed_meta_label"] = 0
    gate_counts["triggered"] = 0
    gate_counts["passed_portfolio_limits"] = 0

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
                     defense_res.score_after, cfg, position=positions[sym],
                     config=config)

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
    # THE NO TRADE VETO. A second model, fitted only on the trades this engine
    # would have taken, predicting whether one reaches its profit barrier before
    # its stop. It has no long side -- it cannot propose a name the primary did
    # not -- so its only power is to refuse. Disabled by default; the reason is
    # measured and recorded in MetaLabelConfig.
    #
    # It applies to NEW entries only. A held position is governed by the Stage 6
    # exit band, and letting a freshly refitted classifier close an open trade
    # would put the book at the mercy of a model that changes every 21 sessions.
    win_probs = scores.win_probability or {}
    min_win = fv(cfg.scarcity.min_win_probability)
    vetoed = 0

    # An ENABLED veto that produced no probabilities has not approved anybody
    # and has not refused anybody -- it did not run. Skipping every candidate
    # one at a time turns that into a silent empty book with a full funnel
    # above it, which reads exactly like a day the market offered nothing. It
    # is a stated refusal instead.
    if min_win > 0 and not win_probs and blocked_reason is None:
        blocked_reason = (
            "The NO TRADE veto is switched on and could not score today's "
            "candidates"
            + (f": {scores.win_probability_unavailable}"
               if scores.win_probability_unavailable else ".")
            + " No new position is opened on a gate that did not run."
        )

    for sym in survivors:
        score = by_ticker.get(sym)
        decision = entries.decisions.get(sym)
        if score is None or decision is None:
            continue
        # `percentile` is the stage-4 rank, measured BEFORE Stage 5 ran, and it
        # stays that way: it is a UNIVERSE POSITION test, and a penalty is not a
        # change of position. Gating it post-penalty was tried and is wrong --
        # penalties are denominated in rank units, so subtracting one and then
        # testing against a rank threshold double-counts it. Measured live it
        # cut 45 of 47 defended names, turning `min_universe_percentile` into
        # "no penalty allowed".
        #
        # `min_composite_score` is the post-penalty test, and it is the one that
        # does not currently bind: a real buy candidate sits near composite 0.99
        # and would need a penalty of 0.39 to fail it, which the 0.35
        # auto-reject forecloses. So for the names that actually get bought the
        # graduated penalties only ever REORDER them -- which is a real effect,
        # since `_score_of` sorts on `score_after` -- and never remove one.
        # Raising the floor so it bites is a live-behaviour decision and is left
        # to the config rather than made here.
        if defense.per_stock[sym].score_after < min_score or score.percentile < min_pct:
            continue
        # Counted BEFORE the veto, so the funnel shows what each gate removed
        # rather than attributing both cuts to whichever ran last.
        gate_counts["passed_score_threshold"] += 1
        if min_win > 0 and sym not in open_book:
            prob = win_probs.get(sym)
            # An absent probability is NOT an approval. A name the veto could
            # not score has not been cleared by it, and treating a gap as a pass
            # is how a gate quietly stops gating.
            if prob is None or prob < min_win:
                vetoed += 1
                continue
        qualified.append(sym)
        if decision.status is EntryStatus.TRIGGERED:
            gate_counts["triggered"] += 1

    # Reported as SURVIVORS, because that is what every other rung of this
    # funnel counts -- a removal count sitting among them reads as a survivor
    # count and inverts the narrowing. Present only when the veto actually ran,
    # so a disabled gate does not add a row that always repeats the one above.
    # A SURVIVOR count, like every other rung. The number vetoed is the drop
    # from the line above and does not need a row that inverts the narrowing.
    if min_win > 0:
        gate_counts["passed_meta_label"] = len(qualified)
        log.info("meta-label veto applied",
                 extra={"floor": min_win, "vetoed": vetoed,
                        "passed": len(qualified)})

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
        # (That measurement was taken when momentum carried the fit. It no
        # longer does -- the live coefficients price reversal, lottery and
        # delivery -- so the REASON has moved even though the decision stands.
        # `_exposure_themes` now reads whatever is priced rather than naming
        # momentum, so the check follows the model instead of a snapshot.)
        # The operator is still told, which is the part that had value.
        basket = accepted + [score]
        exposure = _aggregate_exposure(basket, _exposure_themes(basket))
        breached = [n for n, v in exposure.items() if abs(v) >= _EXPOSURE_LIMIT]
        if breached:
            rec.why_this_signal_exists.append(
                f"Concentration note: with this name the basket's mean loading "
                f"on {', '.join(sorted(breached))} reaches "
                f"{max(abs(exposure[n]) for n in breached):.2f} sd, at or past "
                f"the {_EXPOSURE_LIMIT:.2f} sd mark. Every pair passes and "
                f"every sector passes; taken together they are one macro "
                f"position. This is reported, not blocked -- blocking it was "
                f"measured at selection Sharpe +0.45 against +0.86."
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

    # SLOTS THE BOOK COULD NOT FILL, and why. The entry band is `model_rank <=
    # entry_rank`, so a name inside the band that Stage 5 hard-rejects is not
    # replaced by the next one down: the band is a rank threshold, not a queue.
    # The book then runs one name light and the cash sits idle.
    #
    # DISCLOSED RATHER THAN CHANGED. Backfilling from rank K+1 is a strategy
    # change -- it is what the trade-level study simulated, since that study had
    # no false-signal defense to reject anybody -- and it has not been measured
    # against the defense that actually runs. The honest position is to say the
    # book is short and which rank went missing, so the drag is visible in the
    # record rather than showing up later as an unexplained gap between the
    # study's 96.5% fill and the live one.
    want = iv(cfg.portfolio.max_signals_per_run)
    if buys and len(buys) < want:
        taken = {r.model_rank for r in buys}
        missing = [r for r in range(1, want + 1) if r not in taken]
        if missing:
            notes_short = ", ".join(f"#{r}" for r in missing)
            for r in buys:
                r.data_quality_note.append(
                    f"The book is holding {len(buys)} of {want} slots: rank(s) "
                    f"{notes_short} were inside the entry band and did not "
                    f"survive to a card. The band is a rank threshold, not a "
                    f"queue, so the next name down does not take the slot -- "
                    f"that capital stays in cash for this cycle.")

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
            ), gate_counts
        return buys, watch, None, gate_counts

    if blocked_reason is not None:
        return [], watch, _no_trade(
            f"{blocked_reason} No position was open to carry.",
            scores, gate_counts, cfg, defense=defense, entries=entries,
        ), gate_counts

    # THE CALENDAR IS NOT A VERDICT ON THE EVIDENCE. On a session the entry
    # clock has closed, every candidate is held back by the schedule rather
    # than by a gate, and reporting "the evidence does not justify risking
    # capital" would be false in the most misleading direction available: an
    # operator reading it three sessions running would conclude the engine had
    # found nothing worth buying, when what it found was a day it does not buy.
    if not getattr(entries, "entries_open", True):
        return [], watch, _no_trade(
            (entries.entries_closed_reason
             or "New entries are closed on this session by the entry cadence.")
            + " This is the schedule, not a judgement about the candidates: "
              "they cleared the ranking and are held on the watchlist for the "
              "next entry date.",
            scores, gate_counts, cfg, defense=defense, entries=entries,
        ), gate_counts

    return [], watch, _no_trade(
        "No candidate cleared every gate. This is the designed outcome when the "
        "evidence does not justify risking capital.",
        scores, gate_counts, cfg, defense=defense, entries=entries,
    ), gate_counts


# =============================================================================
def _band(score: float, cfg) -> StrengthBand:
    if score >= fv(cfg.strength_bands.high_min):
        return StrengthBand.HIGH
    if score >= fv(cfg.strength_bands.medium_min):
        return StrengthBand.MEDIUM
    return StrengthBand.LOW


def _card(sym, name, score, defense_res, decision, plan, regime, eligibility,
          scores, final_score, cfg, position: int = 0,
          config=None) -> Recommendation:
    """Build the recommendation, including the evidence AGAINST it."""
    why: List[str] = []
    # WHAT PUT THIS NAME HERE, first, before any theme attribution. Under
    # `ranking.source: measured_factor` the book is ordered by one column and
    # the fitted composite explains the THEMES behind a name without deciding
    # anything. Leading with the theme attribution -- as this card did -- reads
    # as "the model liked it for these reasons", which is no longer what
    # happened, and a reader has no way to tell from a list of coefficients
    # that the coefficients did not choose.
    rank_cfg = (getattr(config.params.stage4_core_score, "ranking", None)
                if config is not None else None)
    source = str(rank_cfg.source) if rank_cfg is not None else "fitted_composite"
    if source == "v2_composite":
        why.append(
            f"Ranked #{score.rank} of {scores.universe_size} eligible names by the "
            f"v2 composite -- ten equal-weighted sector-neutral factor ranks, "
            f"listed below with what each contributed. On a sealed 17-month "
            f"holdout this ranking's quintile spread was +1.65% per 42 sessions "
            f"(t 2.56); the top ten names were NOT better than the ranking as a "
            f"whole over that window, so read the position as a shortlist, not "
            f"as an ordering you can trust between #1 and #10.")
    elif rank_cfg is not None and source != "fitted_composite":
        why.append(
            f"Ranked #{score.rank} of the eligible universe by {rank_cfg.column} "
            f"-- sector-neutral 6-1 momentum, the single column that orders this "
            f"book. The themes below are the fitted model's separate reading of "
            f"the same name; they are recorded and monitored, and they did not "
            f"choose it.")
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
    elif source == "v2_composite":
        # THIS NAME'S OWN RANK PER FACTOR, not the composite percentile. The
        # first version printed `score.percentile` on every line, so all ten
        # factors read "98th" whatever they measured -- a table that looks like
        # an attribution and carries none.
        #
        # And the raw value is printed WITHOUT a percent sign. Four of the ten
        # are ratios or moments (a vol-adjusted return, a kurtosis, a delivery
        # z-score); "+3363.09%" is a vol-adjusted return of 33.6 misread as a
        # percentage, which is the kind of number a reader either dismisses or,
        # worse, believes.
        ranked = sorted((f for f in score.factors.values()
                         if f.contribution is not None),
                        key=lambda f: -abs(f.contribution))
        for f in ranked:
            pct = (f.standardised + 1.0) / 2.0 * 100.0 if f.standardised is not None else None
            pos = f"{pct:.0f}th pct" if pct is not None else "unranked"
            why.append(
                f"{f.name}: {pos} in its sector, weight {f.weight:+.2f}, "
                f"contributes {f.contribution:+.4f} to a composite of "
                f"{score.composite_raw:+.4f} ({f.citation})"
            )
        missing = [f.name for f in score.factors.values() if not f.available]
        if missing:
            why.append(
                f"Ranked neutral for want of an input: {', '.join(sorted(missing))}. "
                f"The remaining weights renormalise, so this name is scored on "
                f"{len(score.factors) - len(missing)} of {len(score.factors)} factors."
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
        # Numbered by DISPLAY POSITION, not by the rung's stable priority. Four
        # rungs ship disarmed, so the priorities that survive are 2, 3, 4, 5 and
        # 8, and a card that printed those verbatim would read as though five
        # conditions had gone missing. The rung id is kept in the text so the
        # two numbering schemes can still be reconciled against the config.
        for i, e in enumerate(plan.exit_conditions, start=1):
            lvl = f" (Rs {e.level:,.2f})" if e.level else ""
            sell.append(f"{i}. {e.reason.value}{lvl} -- {e.description} "
                        f"[rung {e.priority}]")
        # WHAT WILL NOT CLOSE THIS POSITION, said out loud. Someone holding a
        # name needs to know that reaching the target does not sell it and that
        # losing MA(50) does not sell it, and the absence of a line cannot
        # communicate that.
        # `cfg` here is stage8's own block; the exit rules live in stage 7.
        c7 = config.params.stage7_risk if config is not None else None
        if c7 is not None:
            h = c7.exit_hierarchy
            off = [name for flag, name in (
                (h.thesis_invalidation, "thesis invalidation"),
                (h.trailing_stop, "trailing stop"),
                (h.target_achieved, "profit target"),
            ) if not bool(getattr(flag, "value", flag))]
            if off:
                sell.append(
                    f"NOT an exit: {', '.join(off)}. Each was measured alone "
                    f"against this configuration and each cost return -- the "
                    f"invalidation exit most of all, at 15.6 points of per-trade "
                    f"win probability. The position ends on the rank band, the "
                    f"{iv(c7.holding_period.max_holding_sessions)}-session limit, "
                    f"the disaster stop, or loss of eligibility.")

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
        # WHAT THIS TRADE IS, recorded with it. The cadence it belongs to, the
        # hold it is planned for, and what the configuration's own 258 study
        # trades did -- so a resolved outcome months from now can be compared
        # against the engine's claim and not only against the market.
        trade_plan=(build_trade_plan(config, plan) if config is not None else None),
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


#: Themes whose combined loading defines a macro bet: two names can correlate at
#: 0.4 and still both sit in the top decile of the same theme, so this catches
#: what the pairwise and sector caps cannot.
#:
#: EVERY PRICED THEME, read from the model rather than listed by hand. This was
#: a hardcoded tuple of individual factor names -- `mom_6_1`, `resid_mom`,
#: `prox_52w`, `beta_120` -- and the family refactor made the card's keys
#: `mom`, `reversal`, `lottery`, `risk`, `delivery`. The intersection was
#: empty, so `_aggregate_exposure` returned {} on every call and the
#: concentration note could never fire. A hand-maintained list of the model's
#: own column names is a thing that goes stale silently; asking the card which
#: themes moved this name cannot.
def _exposure_themes(scores) -> Tuple[str, ...]:
    """The themes the fit actually priced, from the attribution itself."""
    for score in scores or ():
        factors = getattr(score, "factors", None) or {}
        priced = tuple(
            name for name, f in factors.items()
            if getattr(f, "evidence_tier", None) == "model"
            and abs(getattr(f, "weight", 0.0) or 0.0) > 1e-12
        )
        if priced:
            return priced
    return ()

#: Mean loading at which the basket stops being diversified, in STANDARD
#: DEVIATIONS. The previous value of 0.75 was documented as "ranks run -1 to +1,
#: so 0.75 is roughly the top eighth" -- but `FactorScore.standardised` on the
#: model path is `crossmodel.standardised_features`, a z-score, not a rank. On
#: that scale 0.75 sd is about the 77th percentile for ONE name, and requiring
#: a whole basket to average it is a much higher bar than the comment claimed.
#:
#: 1.00 sd of MEAN loading across the basket is the equivalent statement: with
#: five names that is a basket sitting a full standard deviation above the
#: universe on one theme, which is one position wearing five tickers. Reported,
#: never gated -- see the measurement below.
_EXPOSURE_LIMIT = 1.00


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
