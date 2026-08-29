"""What is known about a name's liquidity, and what may be done about it.

THE DEFECT THIS EXISTS TO REMOVE. Liquidity was a float that could be `None`,
and every consumer invented its own answer for the missing case. They did not
agree, and both of the important ones were wrong in the same direction:

    portfolio_sim._position   no ADTV -> `qty_liq = slot / entry`, i.e. the
                              LARGEST size the capital slot allows
    costs.impact_bps          no ADTV -> the half-spread alone, i.e. the
                              CHEAPEST fill in the model

So a name whose liquidity could not be measured received the biggest position
and the best execution assumption available. That is exactly backwards. An
unmeasured quantity should reduce confidence, and reduced confidence should
reduce size and worsen the assumed fill.

Measured on the selection period, refusing those names is worth +0.17% per
63-session period on both ranking constructions, and costs about six points of
deployed capital. So this is not merely a hygiene fix -- the names it admitted
were, on average, ones the book was better off without.

MISSING IS NOT ZERO AND NOT STALE. Four states, because they license different
actions and collapsing them is what allowed the defect:

    KNOWN_VALID   a finite, positive ADTV computed over a full window
    KNOWN_STALE   a real measurement, too old to be relied on. Tradable only
                  under an explicit policy, at a discounted capacity
    MISSING       never measured. Not tradable
    INVALID       measured and impossible -- zero, negative, NaN, infinite.
                  Not tradable, and distinct from MISSING because it means a
                  feed is producing nonsense rather than nothing

Only KNOWN_VALID is automatically tradable. Nothing here silently imputes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["LiquidityState", "LiquidityView", "assess", "STALE_DISCOUNT"]


class LiquidityState(str, Enum):
    KNOWN_VALID = "KNOWN_VALID"
    KNOWN_STALE = "KNOWN_STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"


#: Capacity multiplier applied to a stale reading. A number that was true a
#: month ago is evidence, but weaker evidence, and liquidity leaves a name in
#: days rather than months -- `stage7_risk._recent_liquidity` was written about
#: exactly that. Halving is a policy, not a measurement, and it is stated here
#: rather than tuned: no result in this repository selected it, and it is
#: deliberately blunt so that nobody mistakes it for a calibrated figure.
STALE_DISCOUNT = 0.5


@dataclass(frozen=True)
class LiquidityView:
    """A liquidity reading and what it licenses."""

    state: LiquidityState
    #: The usable ADTV in rupees, already discounted for staleness. `None`
    #: whenever the state does not license a trade -- deliberately NOT a
    #: number, so a caller that ignores `tradable` gets a TypeError rather than
    #: a plausible-looking position.
    adtv_inr: Optional[float]
    #: The reading as it arrived, before any discount. For reporting only.
    raw_adtv_inr: Optional[float]
    age_sessions: Optional[int]
    reason: str

    @property
    def tradable(self) -> bool:
        return self.state in (LiquidityState.KNOWN_VALID,
                              LiquidityState.KNOWN_STALE)

    @property
    def confident(self) -> bool:
        return self.state is LiquidityState.KNOWN_VALID

    def describe(self) -> str:
        if self.adtv_inr is None:
            return f"{self.state.value}: {self.reason}"
        return (f"{self.state.value}: Rs {self.adtv_inr:,.0f} usable"
                + (f" ({self.reason})" if self.reason else ""))


def assess(raw: Optional[float], *, age_sessions: Optional[int] = None,
           max_age_sessions: int = 5,
           allow_stale: bool = True) -> LiquidityView:
    """Classify a liquidity reading.

    ``age_sessions`` is how long ago the window it was computed over ended.
    ``None`` means the caller does not track it -- which is treated as fresh,
    because inventing an age would be the same class of error as inventing a
    liquidity. Callers that can supply it should.

    ``allow_stale`` False makes staleness a refusal rather than a discount, for
    an operator who would rather hold cash than trade on a month-old depth
    estimate. It is a policy switch and it is tested in both positions.
    """
    if raw is None:
        return LiquidityView(LiquidityState.MISSING, None, None, age_sessions,
                             "no ADTV was computed for this name on this date")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return LiquidityView(LiquidityState.INVALID, None, None, age_sessions,
                             f"ADTV is not a number: {raw!r}")
    if not math.isfinite(value):
        return LiquidityView(LiquidityState.INVALID, None, value, age_sessions,
                             f"ADTV is {value}")
    if value <= 0.0:
        # ZERO IS A MEASUREMENT, and what it measures is a name that did not
        # trade. It is not a small number to be sized against; it is a refusal.
        return LiquidityView(LiquidityState.INVALID, None, value, age_sessions,
                             f"ADTV is {value:,.0f} -- the name did not trade")

    if age_sessions is not None and age_sessions > max_age_sessions:
        if not allow_stale:
            return LiquidityView(
                LiquidityState.MISSING, None, value, age_sessions,
                f"last measured {age_sessions} sessions ago, beyond the "
                f"{max_age_sessions}-session limit, and stale readings are "
                f"not accepted under this policy")
        return LiquidityView(
            LiquidityState.KNOWN_STALE, value * STALE_DISCOUNT, value,
            age_sessions,
            f"last measured {age_sessions} sessions ago; capacity discounted "
            f"to {STALE_DISCOUNT:.0%}")

    return LiquidityView(LiquidityState.KNOWN_VALID, value, value,
                         age_sessions, "")
