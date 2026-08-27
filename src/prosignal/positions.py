"""Events that overtake an open position.

Stage 3 decides what may be entered. Nothing decides what happens to a name
already held when the facts that admitted it stop being true: it is dropped
from the index, suspended from trading, or delisted. Entry-time gates do not
govern open positions, so today such a name simply stops appearing in the
universe and the position falls out of tracking without a recorded exit.

Each case gets one explicit rule, because the alternative is not "no rule" --
it is a different, undocumented rule per code path.

  reconstitution  hold and flag. Leaving an index changes who must own a stock,
                  not whether it is tradeable. Forcing an exit here would sell
                  into the reconstitution flow, which is the worst available
                  price and a cost the thesis never accounted for.
  suspension      hold and flag. There is no price to exit at. Recording one
                  would invent it.
  delisting       force exit at the last tradeable price, flagged. The position
                  is going to zero as a listed instrument whatever we decide;
                  the only question is whether the record says so.

None of this places an order. It marks the position and states what the
operator has to do.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

__all__ = ["UniverseEvent", "PositionAction", "PositionDirective", "review_open_position"]


class UniverseEvent(str, Enum):
    NONE = "none"
    RECONSTITUTION = "index_removal"
    SUSPENSION = "trading_suspension"
    DELISTING = "delisting"


class PositionAction(str, Enum):
    HOLD = "hold"
    HOLD_AND_FLAG = "hold_and_flag"
    FORCE_EXIT = "force_exit"


@dataclass(frozen=True)
class PositionDirective:
    """What to do about one open position, and why."""

    ticker: str
    event: UniverseEvent
    action: PositionAction
    reason: str
    last_tradeable_price: Optional[float] = None
    last_tradeable_date: Optional[dt.date] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "event": self.event.value,
            "action": self.action.value,
            "reason": self.reason,
            "last_tradeable_price": self.last_tradeable_price,
            "last_tradeable_date": (
                self.last_tradeable_date.isoformat() if self.last_tradeable_date else None
            ),
        }


#: Sessions without a print after which a name is treated as suspended rather
#: than merely quiet. Long enough that a holiday cluster or a thin week does not
#: trigger it.
SUSPENSION_SESSIONS = 5

#: Sessions without a print after which suspension is treated as delisting.
DELISTING_SESSIONS = 30


def review_open_position(
    ticker: str,
    frame: Optional[pd.DataFrame],
    as_of: dt.date,
    in_universe: bool,
    sessions: Sequence[dt.date],
    delisted: bool = False,
    excluded_because: Optional[str] = None,
) -> PositionDirective:
    """Classify one open position against today's facts.

    ``sessions`` is the exchange calendar, so a gap is measured in trading days
    rather than calendar days -- otherwise a long weekend reads as a halt.
    """
    last_price: Optional[float] = None
    last_date: Optional[dt.date] = None
    gap = 0

    if frame is not None and not frame.empty and "close" in frame.columns:
        traded = frame.dropna(subset=["close"])
        if "volume" in traded.columns:
            traded = traded[pd.to_numeric(traded["volume"], errors="coerce").fillna(0) > 0]
        if not traded.empty:
            row = traded.iloc[-1]
            last_price = float(row["close"])
            last_date = pd.Timestamp(row["date"]).date() if "date" in traded.columns else None
            if last_date is not None:
                gap = sum(1 for s in sessions if last_date < s <= as_of)

    if delisted or gap >= DELISTING_SESSIONS:
        return PositionDirective(
            ticker, UniverseEvent.DELISTING, PositionAction.FORCE_EXIT,
            reason=(
                f"no print for {gap} sessions; treated as delisted. The exit is "
                f"recorded at the last price that traded, which is the last "
                f"price that existed."
            ),
            last_tradeable_price=last_price, last_tradeable_date=last_date,
        )

    if gap >= SUSPENSION_SESSIONS:
        return PositionDirective(
            ticker, UniverseEvent.SUSPENSION, PositionAction.HOLD_AND_FLAG,
            reason=(
                f"no print for {gap} sessions; trading appears suspended. The "
                f"position is held because there is no price to exit at -- "
                f"recording one would invent it."
            ),
            last_tradeable_price=last_price, last_tradeable_date=last_date,
        )

    if not in_universe:
        return PositionDirective(
            ticker, UniverseEvent.RECONSTITUTION, PositionAction.HOLD_AND_FLAG,
            reason=(
                "no longer in the tradeable universe. Held rather than exited: "
                "leaving an index changes who must own a stock, not whether it "
                "can be sold, and exiting into reconstitution flow pays the "
                "worst price available for a reason the thesis never priced."
            ),
            last_tradeable_price=last_price, last_tradeable_date=last_date,
        )

    if excluded_because:
        # "Trading normally" was FALSE here. A held name reaches this review
        # only because the run produced no card for it, and the commonest cause
        # is an eligibility gate that rejected it with a stated reason --
        # earnings proximity, a data-quality failure, illiquidity. `in_universe`
        # is tested against the RAW universe, so all of those still read as "in
        # universe", and the operator was told nothing was wrong about a
        # position the engine had explicitly refused to evaluate.
        #
        # The ACTION is unchanged and deliberate: entry-time gates do not
        # govern open positions, and exiting into one pays the worst price
        # available for a reason the thesis never priced. Only the reason is
        # corrected, because a flag nobody can act on is worse than no flag.
        return PositionDirective(
            ticker, UniverseEvent.NONE, PositionAction.HOLD,
            reason=f"still trading, but the run set it aside: {excluded_because}",
            last_tradeable_price=last_price, last_tradeable_date=last_date,
        )
    return PositionDirective(
        ticker, UniverseEvent.NONE, PositionAction.HOLD,
        reason="in universe and trading normally",
        last_tradeable_price=last_price, last_tradeable_date=last_date,
    )
