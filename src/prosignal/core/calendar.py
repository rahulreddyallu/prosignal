"""NSE trading calendar.

Design decision worth stating plainly: **the calendar is derived from data that
actually exists, not from a hardcoded holiday table.** A shipped holiday list
goes stale every year and, worse, fails silently -- the engine would happily
compute a "21-session lookback" that spans a different number of real sessions
than intended, and nothing would complain.

So:

* The authoritative session list is the set of dates for which NSE actually
  published an index file. That is ground truth by construction.
* ``STATIC_CLOSURE_HINTS`` exists only to avoid pointlessly probing dates that
  are almost certainly closed. A wrong hint costs one wasted HTTP request, not
  a wrong answer, because a hinted-closed date is still probed if it would
  otherwise be the resolved decision date.

Every lookback in this engine is expressed in *sessions*, never calendar days,
so a Diwali week or a long weekend can never quietly change a window length.
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left, bisect_right
from typing import Iterable, List, Optional, Sequence, Set

from .errors import DataError

__all__ = ["TradingCalendar", "STATIC_CLOSURE_HINTS", "is_probably_closed"]


#: Fixed-date national holidays on which NSE is reliably shut. Deliberately
#: limited to dates that do not move year to year -- lunar-calendar holidays
#: (Diwali, Holi, Eid, Muhurat sessions) are discovered from the data instead
#: of guessed here, because guessing them wrong is worse than not guessing.
STATIC_CLOSURE_HINTS = {
    (1, 26),   # Republic Day
    (5, 1),    # Maharashtra Day
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (12, 25),  # Christmas
}


def is_probably_closed(day: dt.date) -> bool:
    """Cheap pre-filter. Never authoritative -- only used to skip HTTP probes."""
    if day.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    return (day.month, day.day) in STATIC_CLOSURE_HINTS


class TradingCalendar:
    """An immutable, sorted list of confirmed NSE trading sessions."""

    __slots__ = ("_sessions", "_set", "_approximate")

    def __init__(self, sessions: Iterable[dt.date], approximate: bool = False) -> None:
        self._approximate = approximate
        cleaned: Set[dt.date] = set()
        for s in sessions:
            if isinstance(s, dt.datetime):
                s = s.date()
            if not isinstance(s, dt.date):
                raise TypeError(f"session must be a date, got {type(s).__name__}")
            cleaned.add(s)
        self._sessions: List[dt.date] = sorted(cleaned)
        self._set = cleaned

    # -- construction -------------------------------------------------------
    @classmethod
    def from_series_index(cls, index: Sequence) -> "TradingCalendar":
        """Build from a pandas DatetimeIndex (or any date-like sequence)."""
        out: List[dt.date] = []
        for x in index:
            if hasattr(x, "date"):
                out.append(x.date())
            elif isinstance(x, dt.date):
                out.append(x)
            else:
                out.append(dt.date.fromisoformat(str(x)[:10]))
        return cls(out)

    @classmethod
    def weekday_fallback(cls, start: dt.date, end: dt.date) -> "TradingCalendar":
        """Weekdays minus static hints. ONLY for bootstrapping an empty store.

        Any calendar built this way is approximate. Callers that care (the
        backtest harness, in particular) must check :attr:`is_approximate`.
        """
        days: List[dt.date] = []
        cur = start
        while cur <= end:
            if not is_probably_closed(cur):
                days.append(cur)
            cur += dt.timedelta(days=1)
        return cls(days, approximate=True)

    @property
    def is_approximate(self) -> bool:
        return self._approximate

    # -- basics -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, day: object) -> bool:
        return day in self._set

    def __iter__(self):
        return iter(self._sessions)

    @property
    def sessions(self) -> List[dt.date]:
        return list(self._sessions)

    @property
    def first(self) -> dt.date:
        self._require_nonempty()
        return self._sessions[0]

    @property
    def last(self) -> dt.date:
        self._require_nonempty()
        return self._sessions[-1]

    def _require_nonempty(self) -> None:
        if not self._sessions:
            raise DataError(
                "Trading calendar is empty. Run `prosignal data ingest` first -- "
                "the calendar is derived from sessions NSE actually published, "
                "not from a hardcoded holiday table."
            )

    def is_session(self, day: dt.date) -> bool:
        return day in self._set

    # -- navigation ---------------------------------------------------------
    def last_session_on_or_before(self, day: dt.date) -> Optional[dt.date]:
        i = bisect_right(self._sessions, day)
        return self._sessions[i - 1] if i > 0 else None

    def first_session_on_or_after(self, day: dt.date) -> Optional[dt.date]:
        i = bisect_left(self._sessions, day)
        return self._sessions[i] if i < len(self._sessions) else None

    def previous_session(self, day: dt.date, n: int = 1) -> Optional[dt.date]:
        """The n-th session strictly before ``day``."""
        if n < 1:
            raise ValueError("n must be >= 1")
        i = bisect_left(self._sessions, day)
        j = i - n
        return self._sessions[j] if j >= 0 else None

    def next_session(self, day: dt.date, n: int = 1) -> Optional[dt.date]:
        """The n-th session strictly after ``day``."""
        if n < 1:
            raise ValueError("n must be >= 1")
        i = bisect_right(self._sessions, day)
        j = i + n - 1
        return self._sessions[j] if j < len(self._sessions) else None

    def shift(self, day: dt.date, sessions: int) -> Optional[dt.date]:
        """Move ``sessions`` sessions from ``day`` (negative = backwards).

        ``day`` itself must be a session; use :meth:`last_session_on_or_before`
        first if it might not be.
        """
        if day not in self._set:
            raise DataError(
                f"{day} is not a known trading session; anchor to a real session "
                f"before shifting.",
            )
        i = bisect_left(self._sessions, day)
        j = i + sessions
        if 0 <= j < len(self._sessions):
            return self._sessions[j]
        return None

    def sessions_between(
        self, start: dt.date, end: dt.date, inclusive: bool = True
    ) -> List[dt.date]:
        lo = bisect_left(self._sessions, start)
        hi = bisect_right(self._sessions, end) if inclusive else bisect_left(self._sessions, end)
        return self._sessions[lo:hi]

    def count_between(self, start: dt.date, end: dt.date, inclusive: bool = True) -> int:
        return len(self.sessions_between(start, end, inclusive=inclusive))

    def trailing_window(self, end: dt.date, sessions: int) -> List[dt.date]:
        """The ``sessions`` sessions ending at (and including) ``end``."""
        if sessions < 1:
            raise ValueError("sessions must be >= 1")
        i = bisect_right(self._sessions, end)
        lo = max(0, i - sessions)
        return self._sessions[lo:i]

    def age_in_sessions(self, timestamp: dt.date, as_of: dt.date) -> int:
        """How many sessions old a feed is -- the Stage 1 staleness metric.

        Measured in sessions, not calendar days, so a three-day weekend never
        spuriously trips the staleness gate.
        """
        if timestamp > as_of:
            return 0
        return max(0, self.count_between(timestamp, as_of, inclusive=True) - 1)

    def has_coverage(self, end: dt.date, sessions: int) -> bool:
        """True when the calendar holds at least ``sessions`` sessions up to ``end``."""
        return len(self.trailing_window(end, sessions)) >= sessions

    def merged_with(self, other: Iterable[dt.date]) -> "TradingCalendar":
        return TradingCalendar(list(self._sessions) + list(other))

    def __repr__(self) -> str:  # pragma: no cover - display only
        if not self._sessions:
            return "TradingCalendar(empty)"
        approx = ", approximate" if self.is_approximate else ""
        return (
            f"TradingCalendar({len(self._sessions)} sessions, "
            f"{self.first}..{self.last}{approx})"
        )
