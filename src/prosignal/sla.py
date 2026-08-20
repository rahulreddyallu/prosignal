"""Pipeline deadline against the next session's open.

Every price in a run assumes execution at the next session's open. That
assumption expires: a decision produced after the open it was meant to act on
describes a fill that already happened at a price nobody read. The run does not
become wrong so much as unusable, and nothing currently notices.

So the run carries a deadline. Past it the correct output is no decision and a
stated reason, not a decision produced in a hurry -- there is no version of
"late" that improves by rushing the last two stages.

The deadline is expressed as minutes before the next session opens, so it moves
with the calendar rather than assuming every session starts at 09:15. NSE runs
special Saturday sessions (muhurat, live drills) and occasional truncated days,
and a hard-coded weekday assumption silently skips them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["SlaVerdict", "NSE_OPEN", "check_sla", "next_session"]

#: NSE continuous trading opens at 09:15 IST. Special sessions publish their own
#: timings; where one is known it should be passed in rather than assumed.
NSE_OPEN = dt.time(9, 15)

#: Minutes before the open by which a decision must exist to be actionable.
DEFAULT_MARGIN_MINUTES = 30


@dataclass(frozen=True)
class SlaVerdict:
    """Whether a decision produced now can still be acted on."""

    actionable: bool
    reason: str
    next_session_date: Optional[dt.date] = None
    minutes_to_open: Optional[float] = None


def next_session(as_of: dt.date, sessions: Sequence[dt.date]) -> Optional[dt.date]:
    """The first calendar session strictly after ``as_of``.

    Read from the exchange calendar rather than derived from the weekday, so a
    Saturday session is a session and a Monday holiday is not.
    """
    for session in sessions:
        if session > as_of:
            return session
    return None


def check_sla(
    as_of: dt.date,
    now: dt.datetime,
    sessions: Sequence[dt.date],
    margin_minutes: int = DEFAULT_MARGIN_MINUTES,
    session_open: dt.time = NSE_OPEN,
) -> SlaVerdict:
    """Is there still time to act on a decision for ``as_of``?"""
    nxt = next_session(as_of, sessions)
    if nxt is None:
        return SlaVerdict(
            actionable=False,
            reason=(
                "no session after the decision date in the calendar, so the "
                "entry price this run assumes does not exist yet"
            ),
        )

    opens_at = dt.datetime.combine(nxt, session_open)
    minutes = (opens_at - now).total_seconds() / 60.0

    if minutes < 0:
        return SlaVerdict(
            actionable=False,
            reason=(
                f"the {nxt} open was {abs(minutes):.0f} minutes ago. Every price "
                f"in this run assumes a fill at that open, so the run is skipped "
                f"rather than issued against a price that has already traded."
            ),
            next_session_date=nxt, minutes_to_open=minutes,
        )

    if minutes < margin_minutes:
        return SlaVerdict(
            actionable=False,
            reason=(
                f"{minutes:.0f} minutes to the {nxt} open, inside the "
                f"{margin_minutes}-minute margin. Skipped deliberately: the "
                f"remaining stages are not faster under time pressure, they are "
                f"only less reviewed."
            ),
            next_session_date=nxt, minutes_to_open=minutes,
        )

    return SlaVerdict(
        actionable=True,
        reason=f"{minutes:.0f} minutes to the {nxt} open",
        next_session_date=nxt, minutes_to_open=minutes,
    )
