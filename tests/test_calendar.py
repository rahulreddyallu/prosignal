"""Trading-calendar navigation.

Every lookback in the engine is measured in SESSIONS. If ``shift`` or
``age_in_sessions`` is off by one, a "12-1 month momentum" is quietly not
12-1 month momentum any more, and nothing else in the system would notice.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.core.calendar import TradingCalendar, is_probably_closed
from prosignal.core.errors import DataError

from .conftest import make_sessions


@pytest.fixture
def cal() -> TradingCalendar:
    return TradingCalendar(make_sessions(60, end=dt.date(2026, 8, 14)))


def test_weekends_are_probably_closed():
    assert is_probably_closed(dt.date(2026, 8, 15))  # Saturday + Independence Day
    assert is_probably_closed(dt.date(2026, 8, 16))  # Sunday
    assert not is_probably_closed(dt.date(2026, 8, 14))  # Friday


def test_static_hints_cover_fixed_date_holidays():
    assert is_probably_closed(dt.date(2026, 1, 26))
    assert is_probably_closed(dt.date(2026, 10, 2))
    assert is_probably_closed(dt.date(2026, 12, 25))


def test_membership_and_bounds(cal):
    assert cal.last == dt.date(2026, 8, 14)
    assert len(cal) == 60
    assert cal.is_session(dt.date(2026, 8, 14))
    assert not cal.is_session(dt.date(2026, 8, 15))


def test_last_session_on_or_before_snaps_backwards(cal):
    # Sunday 16 Aug resolves back to Friday 14 Aug.
    assert cal.last_session_on_or_before(dt.date(2026, 8, 16)) == dt.date(2026, 8, 14)
    assert cal.last_session_on_or_before(dt.date(2026, 8, 14)) == dt.date(2026, 8, 14)


def test_previous_and_next_are_strict(cal):
    friday = dt.date(2026, 8, 14)
    thursday = cal.previous_session(friday)
    assert thursday == dt.date(2026, 8, 13)
    assert cal.previous_session(friday, 5) == dt.date(2026, 8, 7)
    assert cal.next_session(thursday) == friday
    assert cal.next_session(friday) is None


def test_shift_requires_an_anchor_that_is_a_session(cal):
    with pytest.raises(DataError):
        cal.shift(dt.date(2026, 8, 15), -1)


def test_shift_roundtrip(cal):
    day = dt.date(2026, 8, 14)
    back = cal.shift(day, -21)
    assert cal.shift(back, 21) == day


def test_trailing_window_length_and_inclusivity(cal):
    window = cal.trailing_window(dt.date(2026, 8, 14), 21)
    assert len(window) == 21
    assert window[-1] == dt.date(2026, 8, 14)
    assert window == sorted(window)


def test_age_in_sessions_ignores_weekends(cal):
    """A long weekend must not make a fresh feed look stale."""
    # Friday data read on the following Friday is 5 sessions old, not 7 days.
    assert cal.age_in_sessions(dt.date(2026, 8, 7), dt.date(2026, 8, 14)) == 5
    assert cal.age_in_sessions(dt.date(2026, 8, 14), dt.date(2026, 8, 14)) == 0
    # A timestamp ahead of the decision date is never "aged".
    assert cal.age_in_sessions(dt.date(2026, 9, 1), dt.date(2026, 8, 14)) == 0


def test_count_between_is_inclusive_by_default(cal):
    n = cal.count_between(dt.date(2026, 8, 10), dt.date(2026, 8, 14))
    assert n == 5


def test_has_coverage(cal):
    assert cal.has_coverage(dt.date(2026, 8, 14), 60)
    assert not cal.has_coverage(dt.date(2026, 8, 14), 61)


def test_empty_calendar_refuses_to_answer():
    empty = TradingCalendar([])
    with pytest.raises(DataError):
        _ = empty.last


def test_weekday_fallback_is_flagged_approximate():
    cal = TradingCalendar.weekday_fallback(dt.date(2026, 1, 1), dt.date(2026, 3, 31))
    assert cal.is_approximate
    assert not TradingCalendar(make_sessions(5)).is_approximate


def test_merge_deduplicates():
    a = TradingCalendar(make_sessions(10))
    merged = a.merged_with(make_sessions(15))
    assert len(merged) == 15
