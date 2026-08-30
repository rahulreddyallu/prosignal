"""Earnings proximity as a risk disclosure.

The engine sizes every position off an ATR stop and prints the result as the
risk. A stop is a LEVEL, not a fill: an overnight gap opens through it. Measured
on this store an earnings window carries 1.8x the daily volatility and 4.9x the
chance of a gap worse than -5%, so the printed risk is a floor on the loss and
not a cap -- and the card never said when a name was about to report.

These tests pin the two things that make the disclosure honest: that it is a
disclosure and not a gate, and that a missing date reads as UNKNOWN rather than
as clear.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.features import earnings as E


def _cal(rows):
    return pd.DataFrame(
        [{"symbol": s, "earnings_date": pd.Timestamp(d)} for s, d in rows])


def _sessions(start="2026-09-01", n=40):
    return [d.date() for d in pd.bdate_range(start, periods=n)]


# ----------------------------------------------------------------- counting
def test_sessions_are_counted_in_sessions_not_calendar_days():
    """A hold is measured in sessions and a weekend is not a day of risk.
    2026-09-04 is a Friday and 2026-09-07 the next Monday: three calendar days
    apart, one session."""
    s = _sessions("2026-09-04", 10)
    cal = _cal([("AAA", "2026-09-07")])
    out = E.sessions_until_next(cal, ["AAA"], dt.date(2026, 9, 4), s)
    assert out["AAA"] == 1


def test_an_announcement_today_reads_zero():
    s = _sessions()
    cal = _cal([("AAA", "2026-09-01")])
    assert E.sessions_until_next(cal, ["AAA"], dt.date(2026, 9, 1), s)["AAA"] == 0


def test_a_past_announcement_is_not_a_future_one():
    s = _sessions()
    cal = _cal([("AAA", "2026-08-01")])
    assert E.sessions_until_next(cal, ["AAA"], dt.date(2026, 9, 1), s)["AAA"] is None
    assert E.days_since_last(cal, ["AAA"], dt.date(2026, 9, 1))["AAA"] == 31


def test_the_nearest_future_date_wins_not_the_last_one_listed():
    s = _sessions()
    cal = _cal([("AAA", "2026-12-01"), ("AAA", "2026-09-03")])
    assert E.sessions_until_next(cal, ["AAA"], dt.date(2026, 9, 1), s)["AAA"] == 2


def test_an_announcement_on_a_non_trading_day_still_counts_the_sessions_before_it():
    s = _sessions("2026-09-01", 10)
    cal = _cal([("AAA", "2026-09-05")])          # a Saturday
    n = E.sessions_until_next(cal, ["AAA"], dt.date(2026, 9, 1), s)["AAA"]
    assert n == 4, "Tue-Fri are four sessions before that Saturday"


def test_an_absent_feed_returns_unknown_for_everything_and_never_raises():
    empty = pd.DataFrame(columns=["symbol", "earnings_date"])
    assert E.sessions_until_next(empty, ["A", "B"], dt.date(2026, 9, 1),
                                 _sessions()) == {"A": None, "B": None}
    assert E.days_since_last(empty, ["A"], dt.date(2026, 9, 1)) == {"A": None}


# ----------------------------------------------------------------- the note
def test_a_near_announcement_warns_and_names_the_tail_not_just_the_vol():
    """The window is barely worse on a TYPICAL day -- 0.59% against 0.39%
    median gap. What matters is that it is ~5x more likely to produce the one
    move a stop cannot protect against, so the line leads with the tail."""
    note = E.risk_note("AAA", 2, 80)
    assert note and "Reports in 2 sessions" in note
    assert "5x" in note and "-5%" in note
    assert "gap opens through the stop" in note
    assert "floor, not a cap" in note
    assert len(note) < 260, "this sits on a card; it has to stay one glance"


def test_a_missing_date_says_nothing_at_all():
    """Three names in four have no scheduled date. A line saying so would
    appear on almost every card and carry nothing to act on -- that is the
    definition of the noise this panel is being cleared of. The panel and the
    card both simply omit it."""
    assert E.risk_note("AAA", None, 40) is None
    assert E.risk_note("AAA", None, None) is None


def test_a_distant_announcement_says_nothing():
    """A name reporting in three months during a twenty-session hold is the
    ordinary case. A line on every card is a line nobody reads."""
    assert E.risk_note("AAA", E.NEAR_SESSIONS + 1, 30) is None
    assert E.risk_note("AAA", 60, 30) is None
    assert E.risk_note("AAA", E.NEAR_SESSIONS, 30) is not None


# ----------------------------------------------------- the measured constants
def test_the_published_risk_numbers_are_the_measured_ones():
    """These come from `work/v3/earnings_gap.py` over 179 symbols and 246,437
    sessions, each name compared against ITSELF outside its earnings windows.

    The uncontrolled version of that comparison -- earnings sessions against
    every session in the store -- put the ratio at 1.6x, because the names with
    calendars are large caps and calmer than the universe around them. It
    understated the risk threefold, in the direction that makes holding through
    an earnings print look safe. If someone regenerates these, they have to
    regenerate them controlled."""
    r = E.EARNINGS_RISK
    assert r["symbols"] == 179 and r["sessions"] == 246437
    assert r["sd_ratio"] == pytest.approx(1.79)
    assert r["p_gap_below_5pct_ratio"] == pytest.approx(4.94)
    assert r["p_gap_below_5pct"] > r["p_gap_below_5pct_baseline"]
    assert r["session_p01"] < r["session_p01_baseline"] < 0
    assert r["p_gap_below_8pct_ratio"] > 1.0


def test_earnings_proximity_is_not_a_factor():
    """It predicts no return and enters no composite. If it ever appears in
    ALL_FACTORS somebody has turned a risk disclosure into an alpha claim on
    data that cannot support one -- the calendar is dense for 179 names and has
    a median of two rows for everybody else."""
    from prosignal.features import v3
    names = " ".join(v3.ALL_FACTORS).lower()
    assert "earn" not in names
    assert not hasattr(E, "factor_frame")
