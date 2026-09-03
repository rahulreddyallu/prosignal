"""Earnings proximity, as a RISK DISCLOSURE. Not a factor.

WHAT THIS IS NOT. It is not alpha. Post-earnings drift is a real and documented
effect and this store cannot test it: the earnings calendar is dense for 179
symbols and has a median of two rows for everybody else, so a cross-sectional
factor built on it would be measuring 24% of the universe and guessing at the
rest. Nothing here predicts a return, and nothing here gates an entry -- gating
would change a traded number and both sealed windows are spent.

WHAT IT IS. The engine sizes every position off an ATR stop and prints the
result as "risk at the floor: Rs 9,994 (1.00% of the book)". A stop is a LEVEL,
not a fill. An overnight gap opens through it and the trade closes wherever the
market reopens, so that number is an understatement exactly when the name is
about to report -- and the card said nothing about when that was.

MEASURED ON THIS STORE, not asserted. 179 symbols with a real calendar
(>= 8 announcements), 246,437 sessions, 2017-09 to 2026-08, each name compared
against ITSELF outside its earnings windows rather than against the wider
universe -- the naive comparison flatters the result, because the names with
calendars are large caps and calmer than the universe they sit in:

                              non-earnings      within 3 days      ratio
    daily return sd                 2.12%             3.23%        1.79x
    |overnight gap| p90             1.23%             2.35%        1.91x
    |overnight gap| p99             3.74%             6.15%        1.64x
    P(gap worse than -5%)           0.20%             0.96%        4.94x
    P(gap worse than -8%)           0.07%             0.19%        2.83x
    1st-percentile session         -5.09%            -7.93%

So the window is not wildly more volatile on a typical day -- the medians are
0.39% against 0.59% -- and it is roughly FIVE TIMES more likely to produce the
one move a stop cannot protect against. That asymmetry is the whole point, and
it is why this reports a tail probability rather than a volatility multiple.

ABSENCE OF A DATE IS NOT ABSENCE OF EARNINGS. The forward calendar comes from
NSE board-meeting notices and covers 173 of 750 live names. A name with no
confirmed date may still report next week; the disclosure says which case it is
and never implies the second is safe.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence

import pandas as pd

__all__ = ["EARNINGS_RISK", "NEAR_SESSIONS", "earnings_dates",
           "sessions_until_next", "days_since_last", "risk_note"]

#: Measured on 179 symbols with a real calendar, 246,437 sessions, 2017-2026,
#: each name against itself outside its own earnings windows. Regenerate with
#: the v3 earnings-gap study (removed 2026-09-03) if the store changes.
EARNINGS_RISK = {
    "window_days": 3,
    "symbols": 179,
    "sessions": 246437,
    "sd_ratio": 1.79,
    "p_gap_below_5pct": 0.0096,
    "p_gap_below_5pct_baseline": 0.0020,
    "p_gap_below_5pct_ratio": 4.94,
    "p_gap_below_8pct": 0.0019,
    "p_gap_below_8pct_ratio": 2.83,
    "session_p01": -0.0793,
    "session_p01_baseline": -0.0509,
}

#: Sessions ahead of a scheduled announcement at which the card starts warning.
#: Matched to the median hold: a 20-session hold opened here spans the event.
NEAR_SESSIONS = 10


def earnings_dates(store) -> pd.DataFrame:
    """symbol / earnings_date, normalised. Empty frame when the feed is absent."""
    try:
        e = store.read_earnings_calendar()
    except Exception:
        return pd.DataFrame(columns=["symbol", "earnings_date"])
    if e is None or e.empty or "earnings_date" not in e.columns:
        return pd.DataFrame(columns=["symbol", "earnings_date"])
    e = e[["symbol", "earnings_date"]].dropna().copy()
    e["earnings_date"] = pd.to_datetime(e["earnings_date"], errors="coerce").dt.normalize()
    return e.dropna().drop_duplicates().sort_values(["symbol", "earnings_date"])


def sessions_until_next(cal: pd.DataFrame, symbols: Sequence[str], as_of: dt.date,
                        sessions: Sequence[dt.date]) -> Dict[str, Optional[int]]:
    """Trading sessions from `as_of` to each symbol's next scheduled date.

    Counted in SESSIONS, not calendar days, because a hold is measured in
    sessions and a weekend is not a day of risk. None where no future date is on
    file -- which means unknown, not none.
    """
    if cal is None or cal.empty:
        return {s: None for s in symbols}
    today = pd.Timestamp(as_of).normalize()
    fwd = [d for d in pd.to_datetime(pd.Series(list(sessions))).dt.normalize()
           if d >= today]
    pos = {d: i for i, d in enumerate(fwd)}
    nxt = (cal[cal["earnings_date"] >= today]
           .groupby("symbol")["earnings_date"].min())
    out: Dict[str, Optional[int]] = {}
    for s in symbols:
        d = nxt.get(s)
        if d is None or pd.isna(d):
            out[s] = None
            continue
        if d in pos:
            out[s] = int(pos[d])
        else:
            # announced for a non-trading day: count the sessions before it
            out[s] = int(sum(1 for x in fwd if x < d))
    return out


def days_since_last(cal: pd.DataFrame, symbols: Sequence[str],
                    as_of: dt.date) -> Dict[str, Optional[int]]:
    if cal is None or cal.empty:
        return {s: None for s in symbols}
    today = pd.Timestamp(as_of).normalize()
    prev = (cal[cal["earnings_date"] <= today]
            .groupby("symbol")["earnings_date"].max())
    out: Dict[str, Optional[int]] = {}
    for s in symbols:
        d = prev.get(s)
        out[s] = None if d is None or pd.isna(d) else int((today - d).days)
    return out


def risk_note(symbol: str, until: Optional[int], since: Optional[int],
              near: int = NEAR_SESSIONS) -> Optional[str]:
    """One line for the card, or None when there is nothing worth saying.

    Returns a line ONLY when a scheduled date is close, or when nothing is on
    file at all. A name that reported last week and has no date for months is
    the ordinary case and does not need a sentence.
    """
    r = EARNINGS_RISK
    if until is not None and until <= int(near):
        when = "today" if until == 0 else f"in {until} session{'s' if until != 1 else ''}"
        return (
            f"Reports {when}. Earnings windows carry "
            f"{r['p_gap_below_5pct_ratio']:.0f}x the usual chance of an "
            f"overnight gap past -5%. A gap opens through the stop, so the "
            f"risk shown is a floor, not a cap.")
    if until is None:
        # NOT SHOWN ON THE CARD, and it should not be: three names in four have
        # no scheduled date, so a line about it would appear on almost every
        # card and say nothing actionable. Returned for anything that wants it.
        return None
    return None
