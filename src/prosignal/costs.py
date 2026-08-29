"""Indian equity transaction costs and market impact.

Rates are config values rather than literals: SEBI and the exchanges change
them. `parameters.yaml` tags them STATUTORY with the date each was verified.

Delivery-segment round trip as modelled:

    buy  : brokerage + STT + exchange txn + SEBI fee + GST + stamp duty
    sell : brokerage + STT + exchange txn + SEBI fee + GST + DP charge

GST applies to brokerage, exchange charges and the SEBI fee, not to STT or
stamp duty, which are taxes in their own right. Applying it to the wrong base
shifts total cost by roughly 18% of that base.

Impact is modelled separately from fees: fees are linear in turnover, impact
grows with the square root of participation. The square-root form is standard
(Almgren et al.); the coefficient here is UNVALIDATED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _v(node: Any) -> Any:
    """Unwrap a Tunable, or pass a bare scalar through.

    parameters.yaml mixes both shapes: research parameters carry provenance
    metadata, plain switches do not. Neither is wrong, so cost code reads
    through this rather than assuming.
    """
    return node.value if hasattr(node, "value") else node


def _fv(node: Any) -> float:
    return float(_v(node))

__all__ = ["CostModel", "CostBreakdown"]


@dataclass
class CostBreakdown:
    """Every component, in rupees, for one round trip."""

    turnover_buy_inr: float
    turnover_sell_inr: float
    brokerage_inr: float
    stt_inr: float
    exchange_txn_inr: float
    sebi_fee_inr: float
    gst_inr: float
    stamp_duty_inr: float
    dp_charge_inr: float
    fees_total_inr: float
    impact_inr: float
    spread_inr: float
    total_inr: float
    total_bps_of_buy: float
    stressed_total_inr: float
    stressed_bps_of_buy: float

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}

    def summary_line(self) -> str:
        return (
            f"round-trip cost Rs {self.total_inr:,.0f} "
            f"({self.total_bps_of_buy:.0f} bps of position); "
            f"stressed Rs {self.stressed_total_inr:,.0f} "
            f"({self.stressed_bps_of_buy:.0f} bps)"
        )


class CostModel:
    """Config-driven Indian cash-equity cost model."""

    def __init__(self, config) -> None:
        self.c = config.params.costs

    # -- fees ---------------------------------------------------------------
    def _fees(self, buy_value: float, sell_value: float) -> Dict[str, float]:
        c = self.c
        turnover = buy_value + sell_value

        brokerage = 0.0
        for value in (buy_value, sell_value):
            if value <= 0:
                continue
            pct_component = value * _fv(c.brokerage_pct_of_turnover) / 100.0
            flat = _fv(c.brokerage_flat_per_order_inr)
            per_order = pct_component if pct_component > 0 else flat
            cap = _fv(c.brokerage_cap_per_order_inr)
            brokerage += min(per_order, cap) if cap > 0 else per_order

        # Delivery STT is charged on BOTH legs; intraday only on the sell.
        if str(_v(c.segment)).strip().lower() == "delivery":
            stt = (
                buy_value * _fv(c.stt_delivery_buy_pct) / 100.0
                + sell_value * _fv(c.stt_delivery_sell_pct) / 100.0
            )
        else:
            stt = sell_value * _fv(c.stt_intraday_sell_pct) / 100.0

        exchange = turnover * _fv(c.exchange_transaction_charge_pct) / 100.0
        sebi = turnover * _fv(c.sebi_turnover_fee_pct) / 100.0
        # Stamp duty is buy-side only.
        stamp = buy_value * _fv(c.stamp_duty_buy_pct) / 100.0
        dp = _fv(c.dp_charge_per_scrip_sell_inr) if sell_value > 0 else 0.0

        # GST applies to brokerage + exchange + SEBI only.
        gst = (brokerage + exchange + sebi) * _fv(c.gst_pct_on_charges) / 100.0

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange": exchange,
            "sebi": sebi,
            "gst": gst,
            "stamp": stamp,
            "dp": dp,
        }

    # -- impact -------------------------------------------------------------
    def impact_bps(self, position_value_inr: float, adtv_inr: Optional[float]) -> float:
        """Square-root market impact, in basis points of the traded value.

        ``impact = coefficient * (participation ** exponent)``, expressed in bps.

        UNKNOWN LIQUIDITY IS THE MOST EXPENSIVE CASE, NOT THE CHEAPEST. This
        used to return the half-spread alone when ADTV was missing, with a
        comment reasoning that assuming zero impact would be the optimistic
        error. The half-spread alone IS assuming zero impact: it is the cheapest
        number the model can produce, and it was being handed to exactly the
        names whose liquidity could not be measured -- while the sizer, for the
        same names, took the largest position the slot allowed.

        A name with no measurable liquidity should not be traded at all
        (`liquidity.assess`, and `_position` refuses it). This branch remains
        because a cost model must answer every question it is asked, so it
        answers with the stressed figure: the impact of participating at the
        model's own cap. Optimism here can only ever flatter a trade the rest
        of the engine has already refused.
        """
        m = self.c.impact_model
        half_spread = _fv(m.assumed_half_spread_bps)
        coeff = _fv(m.coefficient)
        expo = _fv(m.exponent)
        if position_value_inr <= 0:
            return half_spread
        if not adtv_inr or adtv_inr <= 0:
            participation = float(_fv(m.unknown_liquidity_participation))
        else:
            participation = position_value_inr / adtv_inr
        # coefficient is expressed as a fraction of price; convert to bps.
        impact = coeff * (participation ** expo) * 10_000.0
        return impact + half_spread

    # -- public -------------------------------------------------------------
    def round_trip(
        self,
        entry_price: float,
        quantity: int,
        exit_price: Optional[float] = None,
        adtv_inr: Optional[float] = None,
    ) -> CostBreakdown:
        """Full round-trip cost. ``exit_price`` defaults to a flat exit."""
        exit_price = entry_price if exit_price is None else exit_price
        buy_value = max(entry_price * quantity, 0.0)
        sell_value = max(exit_price * quantity, 0.0)

        f = self._fees(buy_value, sell_value)
        fees_total = sum(f.values())

        # Impact and spread are paid on BOTH legs.
        impact_bps_one_leg = self.impact_bps(buy_value, adtv_inr)
        half_spread_bps = _fv(self.c.impact_model.assumed_half_spread_bps)
        pure_impact_bps = max(impact_bps_one_leg - half_spread_bps, 0.0)

        impact_inr = (pure_impact_bps / 10_000.0) * (buy_value + sell_value)
        spread_inr = (half_spread_bps / 10_000.0) * (buy_value + sell_value)

        total = fees_total + impact_inr + spread_inr
        bps = (total / buy_value * 10_000.0) if buy_value > 0 else 0.0

        s = self.c.stress_tests
        stressed = (
            fees_total * _fv(s.cost_multiplier)
            + (impact_inr + spread_inr) * _fv(s.impact_multiplier)
        )
        stressed_bps = (stressed / buy_value * 10_000.0) if buy_value > 0 else 0.0

        return CostBreakdown(
            turnover_buy_inr=buy_value,
            turnover_sell_inr=sell_value,
            brokerage_inr=f["brokerage"],
            stt_inr=f["stt"],
            exchange_txn_inr=f["exchange"],
            sebi_fee_inr=f["sebi"],
            gst_inr=f["gst"],
            stamp_duty_inr=f["stamp"],
            dp_charge_inr=f["dp"],
            fees_total_inr=fees_total,
            impact_inr=impact_inr,
            spread_inr=spread_inr,
            total_inr=total,
            total_bps_of_buy=bps,
            stressed_total_inr=stressed,
            stressed_bps_of_buy=stressed_bps,
        )

    def breakeven_move_pct(
        self, entry_price: float, quantity: int, adtv_inr: Optional[float] = None
    ) -> float:
        """How far the stock must move just to cover the round trip."""
        cb = self.round_trip(entry_price, quantity, adtv_inr=adtv_inr)
        return (cb.total_inr / cb.turnover_buy_inr * 100.0) if cb.turnover_buy_inr else 0.0
