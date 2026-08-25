"""What the market did after the engine spoke.

The ledger has recorded each run's names with the close at the time, the stop
and the targets since October 2023, so following a past run forward is
arithmetic over an existing record. Two rules carry the weight.

The window starts the session AFTER the signal date. The signal is formed on
that day's close, so including it would credit the setup with a move that had
already happened before anyone could act on it -- the same look-ahead the rest
of this engine is built to avoid, reintroduced at the last step.

And levels are tested against the high and the low, not the close. A stop is
not a close-only instrument, and checking it against the close alone reports
stops as unhit and flatters every result on the screen.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.presentation.outcome import outcomes_for, summarise


def bars(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "high": h, "low": l, "close": c}
         for d, s, h, l, c in rows]
    )


def pick(ticker="AAA", price=100.0, stop=90.0, target=120.0, status="BUY"):
    return {"ticker": ticker, "signal_price": price, "stop": stop,
            "target_1": target, "status": status, "position": 1}


SIG = dt.date(2026, 1, 5)


def test_the_signal_session_itself_is_excluded():
    """The signal forms on that close. Counting the same day's move would
    credit the setup with something that already happened."""
    frame = bars([
        ("2026-01-05", "AAA", 200.0, 100.0, 100.0),   # the signal session
        ("2026-01-06", "AAA", 105.0, 99.0, 104.0),
    ])
    out = outcomes_for([pick()], SIG, frame)[0]
    assert out.sessions == 1
    assert out.high_since == 105.0, "the signal session leaked into the window"


def test_change_is_measured_from_the_price_recorded_with_the_signal():
    frame = bars([("2026-01-06", "AAA", 112.0, 108.0, 110.0)])
    out = outcomes_for([pick(price=100.0)], SIG, frame)[0]
    assert out.change_pct == 10.0


def test_a_stop_touched_intraday_counts_as_touched():
    """Checking the close alone reports this as unhit and flatters the run."""
    frame = bars([("2026-01-06", "AAA", 101.0, 89.0, 99.0)])
    out = outcomes_for([pick(stop=90.0)], SIG, frame)[0]
    assert out.stop_hit is True
    assert out.resolved == "stop"


def test_a_target_touched_intraday_counts_as_touched():
    frame = bars([("2026-01-06", "AAA", 121.0, 99.0, 100.0)])
    out = outcomes_for([pick(target=120.0)], SIG, frame)[0]
    assert out.target_hit is True and out.resolved == "target"


def test_both_levels_touched_is_reported_as_both_not_guessed():
    """Daily bars do not record the sequence within a session. Choosing one
    would be inventing an ordering the data does not contain."""
    frame = bars([("2026-01-06", "AAA", 125.0, 85.0, 100.0)])
    out = outcomes_for([pick()], SIG, frame)[0]
    assert out.resolved == "both"
    assert "which came first" in out.note


def test_nothing_to_report_yet_is_not_the_same_as_an_ambiguous_result():
    """These were both `unknown`, so a name flagged this morning rendered as
    "Both touched" -- a claim that its target and its stop had already been
    hit, about a name that had not traded since."""
    pending = outcomes_for([pick()], SIG, bars([]))[0]
    ambiguous = outcomes_for(
        [pick()], SIG, bars([("2026-01-06", "AAA", 125.0, 85.0, 100.0)]))[0]
    assert pending.resolved == "pending"
    assert ambiguous.resolved == "both"
    assert pending.resolved != ambiguous.resolved


def test_an_untouched_setup_stays_open():
    frame = bars([("2026-01-06", "AAA", 110.0, 95.0, 105.0)])
    assert outcomes_for([pick()], SIG, frame)[0].resolved == "open"


def test_peak_and_trough_come_from_the_extremes_not_the_closes():
    frame = bars([
        ("2026-01-06", "AAA", 118.0, 92.0, 100.0),
        ("2026-01-07", "AAA", 104.0, 96.0, 101.0),
    ])
    out = outcomes_for([pick(price=100.0)], SIG, frame)[0]
    assert out.peak_gain_pct == 18.0
    assert out.worst_drop_pct == -8.0


def test_a_name_with_no_sessions_yet_is_reported_not_dropped():
    out = outcomes_for([pick()], SIG, bars([]))[0]
    assert out.change_pct is None and out.sessions == 0
    assert out.note


def test_a_signal_with_no_recorded_price_is_not_divided_by():
    frame = bars([("2026-01-06", "AAA", 110.0, 95.0, 105.0)])
    out = outcomes_for([pick(price=None)], SIG, frame)[0]
    assert out.change_pct is None


def test_sessions_are_counted_from_the_price_store_not_the_calendar():
    """The strategy quotes its holding period in sessions, so a comparison
    against it has to use the same unit -- a weekend is not a session."""
    frame = bars([
        ("2026-01-06", "AAA", 101.0, 99.0, 100.0),
        ("2026-01-07", "AAA", 101.0, 99.0, 100.0),
        ("2026-01-12", "AAA", 101.0, 99.0, 100.0),   # a week later
    ])
    assert outcomes_for([pick()], SIG, frame)[0].sessions == 3


# ------------------------------------------------------------------ summary
def test_the_summary_carries_the_sample_size():
    """Five names is not a result. The count travels with every figure."""
    frame = bars([("2026-01-06", "AAA", 110.0, 105.0, 108.0),
                  ("2026-01-06", "BBB", 96.0, 90.0, 95.0)])
    s = summarise(outcomes_for([pick("AAA"), pick("BBB")], SIG, frame))
    assert s["tracked"] == 2 and s["advancing"] == 1
    assert "of 2" in s["text"]


def test_a_run_with_no_follow_up_says_so_rather_than_reporting_zero():
    """Three cards reading "0 of 0" say nothing. The screen suppresses the
    tally entirely and explains why instead."""
    s = summarise(outcomes_for([pick()], SIG, bars([])))
    assert s["tracked"] == 0
    assert "until the market trades again" in s["text"]


# ------------------------------------------------------------ price basis
def test_levels_recorded_before_a_split_are_rebased_against_adjusted_bars():
    """The recorded price and levels are in the basis that existed on the
    signal date; the store re-adjusts its whole history when a corporate action
    lands. BAJFINANCE was signalled on 2025-05-02 with a stop of 8195.05 against
    a close of 8862.50, and a 4:1 bonus with a 2:1 face split on 2025-06-16 left
    the store serving that session at 886.25. Compared raw, the stop sits ten
    times above every subsequent low and the call reads as instantly stopped.
    """
    frame = bars([
        ("2026-01-05", "AAA", 10.2, 9.8, 10.0),      # signal session, post-split
        ("2026-01-06", "AAA", 10.5, 9.9, 10.4),
    ])
    # Recorded that day in the pre-split basis: 10x these numbers.
    out = outcomes_for([pick(price=100.0, stop=90.0, target=120.0)], SIG, frame)[0]
    assert out.stop == pytest.approx(9.0)
    assert out.target_1 == pytest.approx(12.0)
    assert out.stop_hit is False, "the raw stop of 90 would sit above every bar"
    assert out.change_pct == pytest.approx(4.0)


def test_an_unadjusted_call_is_left_alone():
    frame = bars([
        ("2026-01-05", "AAA", 101.0, 99.0, 100.0),
        ("2026-01-06", "AAA", 112.0, 108.0, 110.0),
    ])
    out = outcomes_for([pick(price=100.0, stop=90.0, target=120.0)], SIG, frame)[0]
    assert out.stop == pytest.approx(90.0)
    assert out.target_1 == pytest.approx(120.0)
    assert out.change_pct == pytest.approx(10.0)


def test_a_window_with_no_signal_session_assumes_the_same_basis():
    """Absence of the signal bar is missing information about adjustment, not
    evidence of it, and refusing there would blank every outcome for a caller
    that passes a forward-only window."""
    frame = bars([("2026-01-06", "AAA", 112.0, 108.0, 110.0)])
    out = outcomes_for([pick(price=100.0)], SIG, frame)[0]
    assert out.change_pct == pytest.approx(10.0)


def test_an_incredible_basis_ratio_reports_nothing_rather_than_a_wrong_number():
    frame = bars([
        ("2026-01-05", "AAA", 1.0, 1.0, 1.0),
        ("2026-01-06", "AAA", 1.0, 1.0, 1.0),
    ])
    out = outcomes_for([pick(price=1e9)], SIG, frame)[0]
    assert out.change_pct is None
    assert "could not be reconciled" in (out.note or "")
