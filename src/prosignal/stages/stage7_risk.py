"""Stage 7 -- Risk, stop, target, and position size.

Stop distance derives from the stock's own volatility rather than a flat
percent. A 5% stop is loose on an FMCG major and inside the daily noise of a
smallcap that swings 4%, so a fixed number stops out of the volatile names
where the premium sits.

Two exit levels, kept separate:

    stop_price          where the position is closed on price
    invalidation_level  where the original thesis is dead

A stock can hold above its stop while the reason for buying it has gone, which
is an exit even though the stop never fired.

Position size binds three constraints -- risk budget, capital and liquidity --
and the tightest wins. A trade whose executable size would move the price more
than the modelled edge is not a trade.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


from ..core.contracts import ExitCondition, RiskPlan
from ..core.enums import ExitReason, RiskCategory
from ..core.logging import get_logger
from ..costs import CostModel
from ._cfg import bv, fv, iv, v
from ..indicators import atr, realised_volatility

__all__ = ["build_plan", "STAGE_NAME"]

STAGE_NAME = "stage7_risk"
log = get_logger(__name__)


def build_plan(
    ticker: str,
    frame: pd.DataFrame,
    reference_price: float,
    composite_score: float,
    adtv_inr: Optional[float],
    config,
    costs: CostModel,
) -> RiskPlan:
    p = config.params
    cfg = p.stage7_risk
    notes: List[str] = []

    a_series = atr(
        frame["high"], frame["low"], frame["close"],
        iv(cfg.atr.period_sessions), str(v(cfg.atr.method)),
    ).dropna()
    if a_series.empty:
        return RiskPlan(ticker=ticker, reference_price=reference_price,
                        notes=["ATR not computable; no risk plan"])
    atr_value = float(a_series.iloc[-1])
    atr_pct = atr_value / reference_price * 100.0

    # ---- stop -------------------------------------------------------------
    mult = fv(cfg.stop_loss.atr_multiple)
    raw_stop = reference_price - mult * atr_value
    dist_pct = (reference_price - raw_stop) / reference_price * 100.0

    min_pct = fv(cfg.stop_loss.min_stop_distance_pct)
    max_pct = fv(cfg.stop_loss.max_stop_distance_pct)
    basis = f"{mult:g} x ATR({iv(cfg.atr.period_sessions)})"
    if dist_pct < min_pct:
        raw_stop = reference_price * (1 - min_pct / 100.0)
        dist_pct = min_pct
        basis += f", widened to the {min_pct:g}% floor"
        notes.append(
            f"ATR stop was {mult:g}x = {mult*atr_pct:.1f}%, inside the {min_pct:g}% "
            f"floor. A stop inside the stock's daily noise is a guaranteed exit."
        )
    elif dist_pct > max_pct:
        raw_stop = reference_price * (1 - max_pct / 100.0)
        dist_pct = max_pct
        basis += f", capped at the {max_pct:g}% ceiling"
        notes.append(
            f"ATR stop was {mult*atr_pct:.1f}%, beyond the {max_pct:g}% cap. Risk per "
            f"share is capped, which reduces position size rather than accepting the loss."
        )

    stop_price = _round_paise(raw_stop, p)
    risk_per_share = reference_price - stop_price

    # ---- targets ----------------------------------------------------------
    t1_r = fv(cfg.targets.t1_r_multiple)
    t2_r = fv(cfg.targets.t2_r_multiple)
    t1 = _round_paise(reference_price + t1_r * risk_per_share, p)
    t2 = _round_paise(reference_price + t2_r * risk_per_share, p)
    target_basis = f"{t1_r:g}R and {t2_r:g}R from a {dist_pct:.1f}% stop"

    resistance = _resistance(frame, iv(cfg.targets.resistance_lookback_sessions), reference_price)
    snap = fv(cfg.targets.snap_to_resistance_within_pct)
    if resistance is not None and 0 < (resistance - t1) / t1 * 100.0 <= snap:
        t1 = _round_paise(resistance * 0.995, p)
        target_basis += f"; T1 pulled below resistance at Rs {resistance:,.2f}"
        notes.append(
            f"T1 sits just under prior resistance at Rs {resistance:,.2f}. Targeting "
            f"through overhead supply is how a winning trade becomes a round trip."
        )

    rr1 = (t1 - reference_price) / risk_per_share if risk_per_share > 0 else None
    rr2 = (t2 - reference_price) / risk_per_share if risk_per_share > 0 else None

    # ---- invalidation (distinct from the stop) ----------------------------
    inval_ma = iv(cfg.thesis_invalidation.structure_ma_sessions)
    buf = fv(cfg.thesis_invalidation.structure_buffer_atr)
    closes = pd.Series(frame["close"].to_numpy(dtype="float64"))
    ma = closes.rolling(inval_ma, min_periods=inval_ma).mean()
    invalidation = None
    inval_basis = None
    if not ma.dropna().empty:
        invalidation = _round_paise(float(ma.iloc[-1]) - buf * atr_value, p)
        inval_basis = (
            f"close below the {inval_ma}-session average less {buf:g} ATR -- the "
            f"structure the entry relied on is gone, whether or not the stop fired"
        )

    # ---- risk category ----------------------------------------------------
    rel_vol = _relative_vol(frame, iv(cfg.risk_category.relative_vol_lookback_sessions))
    category = _category(composite_score, rel_vol, cfg)

    # ---- position size: risk budget vs capital vs LIQUIDITY ---------------
    qty, size_notes, size_inputs = _position_size(
        reference_price, risk_per_share, adtv_inr, category, p, costs
    )
    notes.extend(size_notes)

    cb = costs.round_trip(reference_price, qty, adtv_inr=adtv_inr) if qty > 0 else None
    cost_bps = cb.total_bps_of_buy if cb else None
    impact_bps = costs.impact_bps(reference_price * qty, adtv_inr) if qty > 0 else None

    if cb and rr1:
        gross_t1_pct = (t1 - reference_price) / reference_price * 100.0
        net_t1_pct = gross_t1_pct - cb.total_bps_of_buy / 100.0
        notes.append(
            f"T1 is {gross_t1_pct:.2f}% gross, {net_t1_pct:.2f}% after "
            f"{cb.total_bps_of_buy:.0f} bps of round-trip cost."
        )

    exits = _exit_hierarchy(cfg, stop_price, invalidation, t1, t2)
    hold_lo = iv(cfg.holding_period.min_holding_sessions)
    hold_hi = iv(cfg.holding_period.max_holding_sessions)

    return RiskPlan(
        ticker=ticker,
        reference_price=reference_price,
        atr=round(atr_value, 4),
        atr_pct_of_price=round(atr_pct, 3),
        stop_price=stop_price,
        stop_distance_pct=round(dist_pct, 3),
        stop_basis=basis,
        invalidation_level=invalidation,
        invalidation_basis=inval_basis,
        trailing_stop_rule=(
            f"{str(v(cfg.trailing_stop.style))} at {fv(cfg.trailing_stop.atr_multiple):g} ATR, "
            f"active after {fv(cfg.trailing_stop.activate_after_r):g}R"
            if bv(cfg.trailing_stop.enabled) else None
        ),
        target_1=t1,
        target_2=t2,
        target_basis=target_basis,
        reward_to_risk_t1=round(rr1, 2) if rr1 else None,
        reward_to_risk_t2=round(rr2, 2) if rr2 else None,
        risk_category=category,
        risk_category_inputs={**size_inputs, "composite_score": round(composite_score, 4),
                              "relative_volatility": round(rel_vol, 3) if rel_vol else 0.0},
        expected_holding_sessions=(hold_lo, hold_hi),
        expected_holding_weeks=(max(hold_lo // 5, 1), max(hold_hi // 5, 1)),
        exit_conditions=exits,
        estimated_round_trip_cost_bps=round(cost_bps, 1) if cost_bps else None,
        estimated_impact_bps=round(impact_bps, 1) if impact_bps else None,
        notes=notes,
    )


# =============================================================================
def _round_paise(value: float, params) -> float:
    step = iv(params.stage6_entry.entry_zone.round_to_paise) / 100.0
    return round(round(value / step) * step, 2) if step > 0 else round(value, 2)


def _resistance(frame, lookback: int, reference: float) -> Optional[float]:
    highs = pd.to_numeric(frame.get("high"), errors="coerce").dropna().tail(lookback)
    if highs.empty:
        return None
    above = highs[highs > reference]
    return float(above.min()) if not above.empty else None


def _relative_vol(frame, lookback: int) -> Optional[float]:
    closes = pd.Series(frame["close"].to_numpy(dtype="float64"))
    sigma = realised_volatility(closes, window=min(lookback, max(len(closes) - 1, 2)))
    s = sigma.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _category(score: float, rel_vol: Optional[float], cfg) -> RiskCategory:
    if score >= fv(cfg.risk_category.standard_min_score):
        return RiskCategory.STANDARD
    if score >= fv(cfg.risk_category.reduced_min_score):
        return RiskCategory.REDUCED
    return RiskCategory.MINIMUM


_CATEGORY_FRACTION = {
    RiskCategory.STANDARD: 1.0,
    RiskCategory.REDUCED: 0.6,
    RiskCategory.MINIMUM: 0.3,
}


def _position_size(price, risk_per_share, adtv, category, params, costs) -> Tuple[int, List[str], Dict[str, float]]:
    """Smallest of: risk budget, capital slot, liquidity cap."""
    notes: List[str] = []
    capital = fv(params.capital.total_capital_inr)
    slot = float(params.capital.position_value_inr())
    risk_pct = fv(params.capital.risk_per_trade_pct)

    frac = _CATEGORY_FRACTION[category]
    risk_budget = capital * (risk_pct / 100.0) * frac
    qty_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    qty_slot = int((slot * frac) / price) if price > 0 else 0

    cap_pct = fv(params.capital.max_participation_of_adtv)
    qty_liq = int((adtv * cap_pct) / price) if adtv and price > 0 else qty_slot

    qty = max(min(qty_risk, qty_slot, qty_liq), 0)
    binding = min(
        [("risk budget", qty_risk), ("capital slot", qty_slot), ("liquidity cap", qty_liq)],
        key=lambda x: x[1],
    )[0]
    notes.append(
        f"Size {qty:,} shares (Rs {qty*price:,.0f}). Binding constraint: {binding}. "
        f"Risk category {category.value} scales the budget to {frac:.0%}."
    )
    if binding == "liquidity cap":
        notes.append(
            "Liquidity is the binding constraint, not conviction. The statistically "
            "attractive size is not executable here without moving the price."
        )
    return qty, notes, {
        "qty_by_risk": float(qty_risk), "qty_by_slot": float(qty_slot),
        "qty_by_liquidity": float(qty_liq), "risk_budget_inr": round(risk_budget, 2),
    }


def _exit_hierarchy(cfg, stop, invalidation, t1, t2) -> List[ExitCondition]:
    """Ordered. Thesis invalidation outranks the stop deliberately."""
    h = cfg.exit_hierarchy
    out: List[ExitCondition] = []
    spec = [
        (h.thesis_invalidation, ExitReason.THESIS_INVALIDATION, 1,
         "the reason for the trade is gone, regardless of price", invalidation),
        (h.stop_loss_breach, ExitReason.STOP_LOSS_BREACH, 2, "protective stop breached", stop),
        (h.new_hard_rejection, ExitReason.NEW_HARD_REJECTION, 3,
         "a Stage 3/5 hard gate now fails", None),
        (h.severe_regime_change, ExitReason.SEVERE_REGIME_CHANGE, 4,
         "regime moved to a no-new-entry bucket", None),
        (h.signal_reversal, ExitReason.SIGNAL_REVERSAL, 5,
         "composite rank fell below the exit percentile", None),
        (h.trailing_stop, ExitReason.TRAILING_STOP, 6, "trailing stop hit", None),
        (h.target_achieved, ExitReason.TARGET_ACHIEVED, 7, "target reached", t2),
        (h.time_expiration, ExitReason.TIME_EXPIRATION, 8,
         "maximum holding period reached -- a backstop, never the primary exit", None),
    ]
    for enabled, reason, prio, desc, level in spec:
        if bv(enabled):
            out.append(ExitCondition(reason=reason, priority=prio, description=desc, level=level))
    return out
