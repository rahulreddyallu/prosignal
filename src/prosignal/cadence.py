"""When the book is allowed to open a position.

THE ENGINE RUNS EVERY SESSION AND THAT DOES NOT CHANGE. The disaster floor is
checked on every bar, the open book is re-ranked, eligibility is re-tested, and
outcomes are resolved. What this module decides is narrower: whether TODAY is a
session on which a NEW position may be opened.

WHY THE TWO ARE DIFFERENT. A daily entry clock and a 21-session entry clock are
different strategies, not the same strategy sampled differently. Measured across
378 (cadence, book size, hold, band, floor) cells over six calendar years, the
21-session stem is the only one whose entire floor ladder is positive in every
year; the daily and 10-session clocks trade twice as often for less. The engine
had no way to express that, because "run" and "may buy" were the same event.

WHY SESSIONS, NOT DAYS. Counting in calendar days makes the schedule depend on
where the holidays fell, so two machines asked "is today an entry date?" could
disagree after any exchange holiday, and a backtest could never reproduce the
live schedule. Sessions are counted against the exchange calendar the store
already keeps, from a fixed anchor, so the answer is a pure function of
(calendar, anchor, cadence) and is the same everywhere.

WHY AN ANCHOR RATHER THAN "EVERY 21ST SESSION OF THE YEAR". An anchor makes the
phase explicit and auditable. Moving it re-phases every entry date, so it is a
recorded decision rather than an emergent property of when the year started.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["EntryClock", "clock_from_config"]


@dataclass(frozen=True)
class EntryClock:
    """The entry schedule, resolved against a concrete session calendar."""

    #: Sessions between entry opportunities. 1 means every session.
    cadence_sessions: int
    #: Cadence date zero is the first session on or after this date.
    anchor: dt.date
    #: True when the run date is an entry date.
    is_entry_date: bool
    #: Sessions since the anchor session; None when the calendar cannot place
    #: the run date (a date before the anchor, or absent from the calendar).
    sessions_since_anchor: Optional[int]
    #: The next session on which entries open, when today is not one.
    next_entry_date: Optional[dt.date]
    #: Human-readable, and recorded on the run.
    reason: str

    def blocked_reason(self) -> Optional[str]:
        """The string the run records when entries are closed today."""
        if self.is_entry_date:
            return None
        return self.reason


def _index_of(sessions: Sequence[dt.date], when: dt.date) -> Optional[int]:
    """Position of ``when`` in the session calendar, or None.

    A run date absent from the calendar is not an error here: a rerun asked for
    a Sunday, or a session the store has not ingested, and the caller decides
    what that means. It is not silently rounded to a neighbour, because rounding
    would move the entry phase.
    """
    try:
        return sessions.index(when)              # list
    except (ValueError, AttributeError):
        pass
    for i, s in enumerate(sessions):             # any sequence
        if s == when:
            return i
    return None


def _anchor_index(sessions: Sequence[dt.date], anchor: dt.date) -> Optional[int]:
    """First session on or after the anchor.

    ON OR AFTER, not "nearest": an anchor set to a holiday or a weekend must
    resolve forward deterministically, and forward is the direction that cannot
    reach back into sessions that have already been traded.
    """
    for i, s in enumerate(sessions):
        if s >= anchor:
            return i
    return None


def resolve(sessions: Sequence[dt.date], as_of: dt.date, *,
            cadence_sessions: int, anchor: dt.date) -> EntryClock:
    """Decide whether ``as_of`` is an entry date.

    An unknown answer is an OPEN one. If the calendar cannot place the run date
    or the anchor -- a store that has not ingested far enough, an anchor beyond
    the end of history -- the clock reports open and says why. Closing the book
    on a bookkeeping failure would look identical to a market with no
    candidates, and the failure mode of a stuck-closed clock (no trades, ever,
    silently) is worse than that of a stuck-open one (the previous behaviour).
    """
    cadence = max(int(cadence_sessions), 1)
    if cadence == 1:
        return EntryClock(cadence, anchor, True, None, None,
                          "entry cadence is 1: every session is an entry date")

    ai = _anchor_index(sessions, anchor)
    ti = _index_of(sessions, as_of)
    if ai is None or ti is None:
        return EntryClock(
            cadence, anchor, True, None, None,
            f"entry cadence {cadence} could not be resolved -- "
            f"{'the anchor ' + anchor.isoformat() + ' is beyond the session calendar' if ai is None else 'the run date ' + as_of.isoformat() + ' is not a session in the calendar'}. "
            f"Entries are left OPEN: a clock that fails closed would stop the "
            f"book silently and look exactly like a market with no candidates.")
    if ti < ai:
        return EntryClock(
            cadence, anchor, True, None, None,
            f"the run date {as_of.isoformat()} is before the cadence anchor "
            f"{anchor.isoformat()}, so the schedule has not started. Entries "
            f"are open.")

    since = ti - ai
    phase = since % cadence
    if phase == 0:
        return EntryClock(
            cadence, anchor, True, since, as_of,
            f"session {since} since the anchor {anchor.isoformat()} is an exact "
            f"multiple of the {cadence}-session entry cadence")
    ahead = cadence - phase
    nxt = sessions[ti + ahead] if ti + ahead < len(sessions) else None
    return EntryClock(
        cadence, anchor, False, since, nxt,
        f"new entries are closed today: session {since} since the anchor "
        f"{anchor.isoformat()} is {phase} of {cadence} into the entry cycle. "
        f"The next entry date is "
        f"{nxt.isoformat() if nxt else f'{ahead} sessions from now'}. Held "
        f"positions are unaffected -- the rank band, the disaster floor and the "
        f"time limit are checked every session.")


def clock_from_config(config, sessions: Sequence[dt.date],
                      as_of: dt.date) -> EntryClock:
    """Read the cadence from `stage6_entry.admission` and resolve it."""
    from .stages._cfg import iv

    adm = config.params.stage6_entry.admission
    raw = getattr(adm.entry_cadence_anchor, "value", adm.entry_cadence_anchor)
    anchor = raw if isinstance(raw, dt.date) else dt.date.fromisoformat(str(raw))
    return resolve(list(sessions), as_of,
                   cadence_sessions=iv(adm.entry_cadence_sessions),
                   anchor=anchor)
