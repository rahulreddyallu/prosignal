"""Stamp every issued trade with what it is and what its kind has done.

One function, called once per recommendation, so the plan on the card, the plan
in the ledger and the plan the outcome is later scored against are the same
object built in one place. The previous arrangement -- the card computing a
holding-period string, the ledger recording levels, and nothing recording an
expectation at all -- meant a resolved outcome could be compared with reality
but never with the engine's own claim.

WHAT IS AND IS NOT PER-NAME. The stop, the target, the risk in rupees and the
planned hold are THIS name's: they come from its own price and ATR. The
probabilities and mean returns are the CONFIGURATION's, read from the frozen
study in `expectancy`, identical on every card. That asymmetry is deliberate and
it is the honest one -- this engine ranks names and has never been shown to
estimate how much any individual one will return, so a per-name expected return
would be a number invented to fill a field.
"""

from __future__ import annotations

from typing import Optional

from .core.contracts import RiskPlan, TradePlan
from .core.logging import get_logger

__all__ = ["build_trade_plan"]

log = get_logger(__name__)


def build_trade_plan(config, plan: Optional[RiskPlan]) -> Optional[TradePlan]:
    """Build the plan recorded with one recommendation.

    Returns None when the study is switched off, so a deployment that has not
    run its own study records no expectation rather than someone else's. Returns
    a plan with null frequencies and real geometry when `plan` is None -- a
    watchlist name still has a cadence and a planned hold, and those are the two
    fields that make the run's schedule reconstructable from the ledger alone.
    """
    # Imported here, not at module scope. `prosignal.stages.__init__` imports
    # stage 8, stage 8 imports this module, and a top-level `from .stages._cfg
    # import ...` closes that loop into a circular import. Every other module
    # that needs the accessors from inside a stage does the same.
    from .stages._cfg import fv, iv

    exp = getattr(config.params, "expectancy", None)
    adm = config.params.stage6_entry.admission
    hold = config.params.stage7_risk.holding_period
    cadence = iv(adm.entry_cadence_sessions)
    planned = iv(hold.max_holding_sessions)

    if exp is None or not bool(exp.enabled):
        return TradePlan(cadence_sessions=cadence, planned_hold_sessions=planned)

    risk_inr = None
    risk_pct = None
    if plan is not None:
        # What the position loses if the disaster floor fills exactly at the
        # stop. "Exactly" is doing work: a floor eight ATRs out is reached by a
        # name in collapse, and a name in collapse gaps. The realised loss on
        # the nine trades that hit it averaged -31.0% against a stop placed at
        # most 35% away, so this figure is the OPTIMISTIC end of what a stop
        # costs, and the card says so rather than presenting it as a bound.
        try:
            entry = float(plan.reference_price) if plan.reference_price else None
            stop = float(plan.stop_price) if plan.stop_price else None
            qty = getattr(plan, "position_size_shares", None)
            book = fv(config.params.capital.total_capital_inr)
            if entry and stop and entry > stop:
                per_share = entry - stop
                if qty:
                    risk_inr = per_share * float(qty)
                    if book:
                        risk_pct = 100.0 * risk_inr / float(book)
        except (TypeError, ValueError, AttributeError) as exc:
            log.warning("could not size the risk on this plan",
                        extra={"ticker": getattr(plan, "ticker", "?"),
                               "error": str(exc)})

    return TradePlan(
        cadence_sessions=cadence,
        planned_hold_sessions=planned,
        expected_hold_sessions=float(exp.expected_hold_sessions),
        expected_return_pct=float(exp.expected_return_pct),
        median_return_pct=float(exp.median_return_pct),
        expected_excess_pct=float(exp.expected_excess_pct),
        median_excess_pct=float(exp.median_excess_pct),
        probability_of_profit=float(exp.probability_of_profit),
        probability_of_beating_benchmark=float(exp.probability_of_beating_benchmark),
        assumed_cost_bps=float(exp.assumed_cost_bps),
        # THE WHOLE TABLE, not just the scenario in force. An operator deciding
        # whether to act on a card needs to know how much of the result is
        # riding on the cost assumption, and the answer -- very little -- is
        # only visible if the alternatives travel with it.
        cost_sensitivity={
            sc.name: {"cost_bps": float(sc.cost_bps), "p_win": float(sc.p_win),
                      "p_beat": float(sc.p_beat),
                      "mean_net_pct": float(sc.mean_net_pct),
                      "median_net_pct": float(sc.median_net_pct)}
            for sc in (exp.by_cost or [])},
        risk_at_stop_inr=risk_inr,
        risk_at_stop_pct_of_book=risk_pct,
        basis=f"{exp.study} (measured {exp.measured_on.isoformat()})",
        basis_trades=int(exp.sample_trades),
        basis_period=str(exp.sample_period),
    )
