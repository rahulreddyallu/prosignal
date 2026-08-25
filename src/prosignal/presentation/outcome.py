"""What happened after the engine said it.

The ledger has recorded each run's names with the close at the time, the stop
and the targets since October 2023, so this is arithmetic over a record that
already exists rather than a backfill. Nothing here re-runs the model or
re-derives a level: it reads what was written down that day and compares it
with what the market did afterwards.

TWO HONESTIES. A level that was touched intraday is reported as touched, using
the high and the low rather than the close, because a stop is not a
close-only instrument and pretending otherwise flatters every result. And the
elapsed count is trading sessions taken from the price store, not calendar
days -- the strategy's holding period is quoted in sessions, so a comparison
against it has to use the same unit.

This is a record of what the engine said and what followed. It is not a
backtest: the names are whatever the run put on the screen, there is no
position sizing, no cost, and no portfolio. The distinction is stated on the
screen for the same reason it is stated here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class Outcome:
    ticker: str
    signal_price: Optional[float]
    last_price: Optional[float]
    change_pct: Optional[float]
    sessions: int
    high_since: Optional[float]
    low_since: Optional[float]
    peak_gain_pct: Optional[float]
    worst_drop_pct: Optional[float]
    target_1: Optional[float]
    stop: Optional[float]
    target_hit: Optional[bool]
    stop_hit: Optional[bool]
    #: "target" | "stop" | "both" | "open" | "pending"
    #: `pending` means there is nothing to report yet -- no session has closed
    #: since the signal. It is NOT the same as an ambiguous result, and
    #: collapsing the two made a name signalled this morning read as though
    #: its target and its stop had both already been hit.
    resolved: str
    note: Optional[str] = None


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


#: Ratio bounds outside which the record is broken rather than merely adjusted.
#: A 1:10 split gives 0.1 and a 10:1 reverse gives 10.
_BASIS_MIN, _BASIS_MAX = 0.001, 1000.0


def _basis_factor(frame, ticker: str, cutoff, signal_price: float) -> Optional[float]:
    """Stored close on the signal session over the close recorded that day.

    That ratio IS the cumulative price adjustment applied since the call was
    made, so multiplying the recorded levels by it puts them in the basis these
    bars are quoted in. It needs no corporate-action lookup and it corrects any
    adjustment, including one the actions table is missing.

    Uses the last session at or before the signal date, so a signal dated on a
    holiday still reconciles.

    Returns 1.0 when the window holds no session at or before the signal date.
    That is missing information about adjustment, not evidence of it, and the
    unadjusted case is overwhelmingly the common one -- refusing there would
    blank every outcome for any caller that passes a forward-only window. The
    production caller reads prices from the first run date onward, so the
    factor is always computable where it matters.

    Returns None only when a factor CAN be computed and is not credible, which
    is a broken record rather than a corporate action.
    """
    rows = frame[(frame["symbol"] == ticker) & (frame["date"] <= cutoff)]
    if rows.empty:
        return 1.0
    stored = _f(rows.sort_values("date")["close"].iloc[-1])
    if stored is None or stored <= 0 or signal_price <= 0:
        return 1.0
    factor = stored / signal_price
    if not (_BASIS_MIN <= factor <= _BASIS_MAX):
        return None
    return factor


def outcomes_for(
    picks: Sequence[Dict[str, Any]],
    as_of: dt.date,
    prices: pd.DataFrame,
) -> List[Outcome]:
    """Track each pick from the session it was issued to the latest stored one.

    `prices` must carry date, symbol, high, low and close. The window starts
    the session AFTER `as_of`: the signal was formed on that day's close, so
    including it would credit the setup with a move that had already happened
    before anything could be acted on.
    """
    out: List[Outcome] = []
    if prices is None or prices.empty:
        return [_blank(p, "No price history was available to follow this up.")
                for p in picks]

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    cutoff = pd.Timestamp(as_of)
    forward = frame[frame["date"] > cutoff]

    for pick in picks:
        ticker = str(pick.get("ticker") or "")
        signal_price = _f(pick.get("signal_price"))
        rows = forward[forward["symbol"] == ticker].sort_values("date")
        if rows.empty or signal_price is None or signal_price <= 0:
            out.append(_blank(
                pick,
                "No session has closed since this was flagged."
                if signal_price else "No price was recorded with the signal.",
            ))
            continue

        # The recorded price and levels are in the basis that existed on the
        # signal date; these bars are adjusted to today. A corporate action in
        # between makes them different currencies, and comparing them produces
        # nonsense of exactly one shape: the level sits far outside every
        # subsequent bar, so it reads as instantly hit or never reachable.
        # BAJFINANCE's stop of 8195.05 against a re-based close of 886.25 is
        # the case this was found on.
        factor = _basis_factor(frame, ticker, cutoff, signal_price)
        if factor is None:
            out.append(_blank(
                pick,
                "The price basis for this signal could not be reconciled with "
                "the stored history, so no change is reported. A corporate "
                "action has re-based the prices since the call was made.",
            ))
            continue
        signal_price *= factor
        target = _f(pick.get("target_1"))
        stop = _f(pick.get("stop"))
        target = target * factor if target is not None else None
        stop = stop * factor if stop is not None else None

        last = _f(rows["close"].iloc[-1])
        high = _f(rows["high"].max())
        low = _f(rows["low"].min())

        target_hit = (high is not None and target is not None and high >= target)
        stop_hit = (low is not None and stop is not None and low <= stop)
        # Both levels inside one window cannot be ordered from daily bars --
        # the sequence within a session is not recorded. Saying which came
        # first would be a guess, so it is not said.
        if target_hit and stop_hit:
            resolved = "both"
        elif target_hit:
            resolved = "target"
        elif stop_hit:
            resolved = "stop"
        else:
            resolved = "open"

        out.append(Outcome(
            ticker=ticker,
            signal_price=signal_price,
            last_price=last,
            change_pct=_pct(signal_price, last),
            sessions=int(len(rows)),
            high_since=high,
            low_since=low,
            peak_gain_pct=_pct(signal_price, high),
            worst_drop_pct=_pct(signal_price, low),
            target_1=target,
            stop=stop,
            target_hit=target_hit if target is not None else None,
            stop_hit=stop_hit if stop is not None else None,
            resolved=resolved,
            note=("Both the target and the stop were touched in this window; "
                  "daily bars do not record which came first."
                  if resolved == "both" else None),
        ))
    return out


def _pct(base: Optional[float], value: Optional[float]) -> Optional[float]:
    if base is None or value is None or base <= 0:
        return None
    return round((value - base) / base * 100.0, 2)


def _blank(pick: Dict[str, Any], note: str) -> Outcome:
    return Outcome(
        ticker=str(pick.get("ticker") or ""),
        signal_price=_f(pick.get("signal_price")),
        last_price=None, change_pct=None, sessions=0,
        high_since=None, low_since=None, peak_gain_pct=None,
        worst_drop_pct=None,
        target_1=_f(pick.get("target_1")), stop=_f(pick.get("stop")),
        target_hit=None, stop_hit=None, resolved="pending", note=note,
    )


def summarise(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """A short account of the day, with the sample size attached.

    Five names is not a result. The count travels with every figure so the
    screen cannot imply otherwise.
    """
    tracked = [o for o in outcomes if o.change_pct is not None]
    if not tracked:
        return {
            "tracked": 0,
            "text": ("This run is the most recent one. Nothing can be measured "
                     "until the market trades again."),
        }

    gains = [o for o in tracked if o.change_pct > 0]
    avg = round(sum(o.change_pct for o in tracked) / len(tracked), 2)
    sessions = max(o.sessions for o in tracked)
    return {
        "tracked": len(tracked),
        "advancing": len(gains),
        "average_change_pct": avg,
        "sessions": sessions,
        "targets_hit": sum(1 for o in tracked if o.target_hit),
        "stops_hit": sum(1 for o in tracked if o.stop_hit),
        "text": (
            f"{len(gains)} of {len(tracked)} are above the price they were "
            f"flagged at, {sessions} session{'s' if sessions != 1 else ''} on."
        ),
    }
