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
    resolved: str          # "target", "stop", "open", "unknown"
    note: Optional[str] = None


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


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
                "This name has no sessions after the signal yet."
                if signal_price else "No price was recorded with the signal.",
            ))
            continue

        last = _f(rows["close"].iloc[-1])
        high = _f(rows["high"].max())
        low = _f(rows["low"].min())
        target = _f(pick.get("target_1"))
        stop = _f(pick.get("stop"))

        target_hit = (high is not None and target is not None and high >= target)
        stop_hit = (low is not None and stop is not None and low <= stop)
        # Both levels inside one window cannot be ordered from daily bars --
        # the sequence within a session is not recorded. Saying which came
        # first would be a guess, so it is not said.
        if target_hit and stop_hit:
            resolved = "unknown"
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
                  if resolved == "unknown" else None),
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
        target_hit=None, stop_hit=None, resolved="unknown", note=note,
    )


def summarise(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """A short account of the day, with the sample size attached.

    Five names is not a result. The count travels with every figure so the
    screen cannot imply otherwise.
    """
    tracked = [o for o in outcomes if o.change_pct is not None]
    if not tracked:
        return {"tracked": 0, "text": "No follow-up prices are available yet."}

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
