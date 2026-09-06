"""Stage 9 -- Conviction. The 0-2 decision.

WHERE THIS SITS. After Stage 7, because the economic gate needs the modelled
round-trip cost of the actual position, and that only exists once a risk plan
has been built. Before Stage 8, because Stage 8 builds the cards and this stage
decides which names get one.

WHAT IT TOOK OVER. Admission used to be `model_rank <= entry_rank` plus the
21-session entry clock. Both are now silent on whether a name is bought:

  * THE CLOCK HAS NO VOTE. A name that was not considered yesterday is fully
    eligible today; a name bought yesterday may vanish today. There is no
    schedule, no cadence and no carried registration in the entry decision. The
    open book is still tracked and still reported -- for outcome attribution
    and research -- but it does not decide what is discovered.

  * RANK IS NECESSARY AND NOT SUFFICIENT. Being first is not evidence of being
    ahead. On the live cross-section the top-ranked name was REFUSED: its
    fourteen supporting factors spanned 1.53 independent directions with 91% of
    the evidence in one of them, and the name that was bought sat at rank 2.

WHAT IT CANNOT DO. It cannot promote a name Stage 5 rejected, cannot rescue a
name that failed eligibility, and cannot emit more than `max_buys`. It has no
long side of its own: every candidate it sees was already ranked and defended.
Its only power is to refuse.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ._cfg import bv, fv, iv, v
from ..conviction import evidence as _ev
from ..conviction import gate as _gate
from ..core.logging import get_logger

__all__ = ["run", "STAGE_NAME", "thresholds_from_config"]

STAGE_NAME = "stage9_conviction"
log = get_logger(__name__)


def thresholds_from_config(config) -> _gate.Thresholds:
    """Read the gate's bar out of `stage9_conviction`."""
    c = config.params.stage9_conviction
    return _gate.Thresholds(
        shortlist=iv(c.shortlist),
        min_independent_evidence=fv(c.min_independent_evidence),
        max_evidence_concentration=fv(c.max_evidence_concentration),
        min_gap_to_median=fv(c.min_gap_to_median),
        min_gap_to_next=fv(c.min_gap_to_next),
        min_robustness=fv(c.min_robustness),
        max_cost_burden=fv(c.max_cost_burden),
        max_residual_correlation=fv(c.max_residual_correlation),
        max_evidence_similarity=fv(c.max_evidence_similarity),
        min_basket_enb=fv(c.min_basket_enb),
        max_buys=iv(c.max_buys),
    )


def _horizon(config) -> int:
    """The holding horizon the economic gate prices against.

    Read from the risk stage rather than hardcoded: the measured edge is
    horizon-dependent (+0.64% at 21 sessions, +1.76% at 63), so pricing a
    63-session trade against a 21-session edge would understate it by nearly
    three times, and the reverse would manufacture one.
    """
    try:
        return int(v(config.params.stage7_risk.holding_period.max_holding_sessions))
    except Exception:
        return 63


def run(scores, defense, plans: Dict[str, object], regime,
        closes: pd.DataFrame, config,
        as_of: Optional[dt.date] = None) -> _gate.Verdict:
    """Evaluate the shortlist and fill 0, 1 or 2 slots."""
    th = thresholds_from_config(config)

    if not bool(config.params.stage9_conviction.enabled):
        return _gate.Verdict(
            cause=_gate.NoTradeCause.DATA,
            reason="stage9_conviction is disabled; no conviction decision was made")

    ranked = list(scores.ranked_scores)
    if len(ranked) < 8:
        return _gate.Verdict(
            cause=_gate.NoTradeCause.UNIVERSE,
            reason=(f"only {len(ranked)} names were scored. A cross-sectional "
                    f"ranking needs a cross-section; separation cannot be "
                    f"measured against a field this small."))

    if defense.market_halt:
        return _gate.Verdict(
            cause=_gate.NoTradeCause.HALT,
            reason=f"market-wide defense halt: {defense.market_halt_reason}")

    # The population is exactly Stage 5's survivors, ordered by the score that
    # survived the argument against them.
    survivors = [s for s in ranked
                 if (defense.per_stock.get(s.ticker) is not None
                     and defense.per_stock[s.ticker].final_status != "REJECTED")]
    if not survivors:
        return _gate.Verdict(
            cause=_gate.NoTradeCause.NO_CANDIDATE,
            reason="every defended name was rejected by the false-signal defense")

    survivors.sort(key=lambda s: defense.per_stock[s.ticker].score_after,
                   reverse=True)
    shortlist = survivors[:max(th.shortlist, 1)]

    # Built ONCE for the whole cross-section. Per candidate this is O(universe)
    # and doing it fifteen times is the difference between one pass and fifteen.
    theme_panel = _ev.theme_matrix(ranked)
    member_panel = _ev.member_matrix(ranked)
    themes = theme_panel[1]

    horizon = _horizon(config)
    candidates: List[_gate.Candidate] = []
    for s in shortlist:
        candidates.append(_gate.evaluate(
            s, ranked, plans.get(s.ticker), regime, horizon, th,
            theme_panel=theme_panel, member_panel=member_panel, themes=themes))

    verdict = _gate.select_at_most_two(candidates, closes, th, themes=themes)

    log.info("stage 9 complete", extra={
        "shortlisted": len(candidates),
        "cleared": sum(1 for c in candidates if c.clears),
        "buys": len(verdict.buys),
        "cause": verdict.cause,
    })
    return verdict
