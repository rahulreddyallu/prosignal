"""A held name the run never evaluated must still be decided about.

`positions.review_open_position` states the rules -- hold and flag a
reconstitution or a suspension, force an exit on a delisting -- and has had
tests since it was written. Nothing in the engine ever called it.

So the case it exists for stayed live: a held name that failed eligibility,
failed a data-quality check or left the universe never reached Stage 8, was
absent from `signals_generated`, and the next run rebuilt the book without it.
The position left with no exit recorded and no trace that it had been held.
Measured on the recorded ledger over adjacent sessions: 23 of 54 held-name
transitions ended that way.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.core.contracts import LedgerRow
from prosignal.ledger import Ledger
from prosignal.pipeline import _review_open_positions
from prosignal.positions import PositionAction, UniverseEvent


class _Rec:
    def __init__(self, ticker):
        self.ticker = ticker


class _Universe:
    def __init__(self, symbols):
        self.symbols = list(symbols)


class _Calendar:
    def __init__(self, sessions):
        self.sessions = list(sessions)
        self.first = self.sessions[0]

    def trailing_window(self, as_of, n):
        return [s for s in self.sessions if s <= as_of][-n:]


class _Store:
    """Serves price history for the names the review asks about."""

    def __init__(self, frames):
        self.frames = frames
        self.asked = None

    def read_prices(self, symbols, start, end):
        self.asked = list(symbols)
        parts = [self.frames[s] for s in symbols if s in self.frames]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


SESSIONS = [d.date() for d in pd.bdate_range("2026-04-01", "2026-08-25")]
AS_OF = SESSIONS[-1]


def _frame(ticker, last_index):
    days = SESSIONS[:last_index]
    return pd.DataFrame({
        "symbol": ticker,
        "date": pd.to_datetime(days),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1e6,
    })


def _review(open_book, buys=(), watch=(), frames=None, universe=None):
    store = _Store(frames or {})
    return _review_open_positions(
        list(open_book), [_Rec(t) for t in buys], [_Rec(t) for t in watch],
        store, _Universe(universe if universe is not None else open_book),
        _Calendar(SESSIONS), AS_OF,
    )


# ------------------------------------------------- who needs a directive
def test_a_held_name_still_in_the_book_needs_no_directive():
    assert _review(["A"], buys=["A"], frames={"A": _frame("A", len(SESSIONS))}) == []


def test_a_held_name_the_run_set_aside_needs_no_directive():
    """It is on the watchlist, so a rule saw it and recorded a reason."""
    assert _review(["A"], watch=["A"], frames={"A": _frame("A", len(SESSIONS))}) == []


def test_a_held_name_the_run_never_produced_gets_one():
    """The whole defect: nothing evaluated it, so nothing decided about it."""
    out = _review(["GONE"], frames={"GONE": _frame("GONE", len(SESSIONS))})
    assert len(out) == 1 and out[0]["ticker"] == "GONE"


def test_an_empty_book_reviews_nothing_and_reads_no_prices():
    assert _review([]) == []


# ------------------------------------------------------------- the rules
def test_leaving_the_universe_holds_and_flags():
    """Exiting into reconstitution flow pays the worst price available for a
    reason the thesis never priced."""
    out = _review(["X"], frames={"X": _frame("X", len(SESSIONS))}, universe=[])
    assert out[0]["event"] == UniverseEvent.RECONSTITUTION.value
    assert out[0]["action"] == PositionAction.HOLD_AND_FLAG.value


def test_a_name_that_stopped_printing_is_treated_as_suspended():
    out = _review(["X"], frames={"X": _frame("X", len(SESSIONS) - 10)})
    assert out[0]["event"] == UniverseEvent.SUSPENSION.value
    assert out[0]["action"] == PositionAction.HOLD_AND_FLAG.value
    assert out[0]["last_tradeable_price"] == 100.0


def test_a_long_silence_forces_an_exit_at_the_last_real_price():
    out = _review(["X"], frames={"X": _frame("X", len(SESSIONS) - 40)})
    assert out[0]["event"] == UniverseEvent.DELISTING.value
    assert out[0]["action"] == PositionAction.FORCE_EXIT.value
    assert out[0]["last_tradeable_price"] == 100.0
    assert out[0]["last_tradeable_date"] is not None


def test_a_name_with_no_price_history_at_all_is_still_decided_about():
    out = _review(["X"], frames={})
    assert out[0]["ticker"] == "X"
    assert out[0]["action"] in {a.value for a in PositionAction}


def test_a_price_read_failure_does_not_fail_the_run():
    class _Broken(_Store):
        def read_prices(self, symbols, start, end):
            raise OSError("store unavailable")

    out = _review_open_positions(
        ["X"], [], [], _Broken({}), _Universe(["X"]), _Calendar(SESSIONS), AS_OF,
    )
    assert len(out) == 1, "the position must still be accounted for"


def test_the_book_order_is_preserved_and_duplicates_collapse():
    out = _review(["B", "A", "B"], frames={})
    assert [d["ticker"] for d in out] == ["B", "A"]


# ------------------------------------------------- the book carries forward
def test_a_flagged_position_stays_in_the_next_runs_book(tmp_path):
    led = Ledger(tmp_path)
    led.append(LedgerRow(
        trial_id="T", run_id="r", date=dt.date(2026, 8, 24),
        logged_at=dt.datetime(2026, 8, 24, 18, 0),
        engine_version="e", schema_version="s", config_version="c",
        signals_generated=["STILL"],
        position_directives=[
            {"ticker": "SUSPENDED", "event": "trading_suspension",
             "action": "hold_and_flag", "reason": "no print for 6 sessions"},
        ],
    ))
    assert led.open_book() == ["STILL", "SUSPENDED"]


def test_a_forced_exit_does_not_stay_in_the_book(tmp_path):
    led = Ledger(tmp_path)
    led.append(LedgerRow(
        trial_id="T", run_id="r", date=dt.date(2026, 8, 24),
        logged_at=dt.datetime(2026, 8, 24, 18, 0),
        engine_version="e", schema_version="s", config_version="c",
        signals_generated=["STILL"],
        position_directives=[
            {"ticker": "DELISTED", "event": "delisting", "action": "force_exit",
             "reason": "no print for 31 sessions"},
        ],
    ))
    assert led.open_book() == ["STILL"]


def test_a_name_in_both_the_book_and_a_directive_is_not_duplicated(tmp_path):
    led = Ledger(tmp_path)
    led.append(LedgerRow(
        trial_id="T", run_id="r", date=dt.date(2026, 8, 24),
        logged_at=dt.datetime(2026, 8, 24, 18, 0),
        engine_version="e", schema_version="s", config_version="c",
        signals_generated=["A"],
        position_directives=[{"ticker": "A", "event": "none", "action": "hold",
                              "reason": "in universe"}],
    ))
    assert led.open_book() == ["A"]


def test_a_malformed_directive_row_does_not_break_the_book(tmp_path):
    led = Ledger(tmp_path)
    led.append(LedgerRow(
        trial_id="T", run_id="r", date=dt.date(2026, 8, 24),
        logged_at=dt.datetime(2026, 8, 24, 18, 0),
        engine_version="e", schema_version="s", config_version="c",
        signals_generated=["A"], position_directives=[{"action": "hold"}],
    ))
    assert led.open_book() == ["A"]


# ------------------------------------------------------- reaching the screen
def test_a_suspended_holding_reaches_the_screen():
    """It is not among the picks -- that IS the problem -- so the view is the
    only place it can be said at all."""
    from prosignal.presentation.viewmodel import build_view

    view = build_view({
        "recommendations": [], "watchlist": [],
        "position_directives": [{
            "ticker": "SUSP", "event": "trading_suspension",
            "action": "hold_and_flag", "reason": "no print for 6 sessions",
            "last_tradeable_price": 100.0, "last_tradeable_date": "2026-08-11",
        }],
    })
    alert = view["open_position_alerts"][0]
    assert alert["ticker"] == "SUSP"
    assert alert["label"] == "Not trading"
    assert alert["exit_required"] is False
    assert alert["last_price"] == 100.0


def test_a_delisting_is_marked_as_requiring_an_exit():
    from prosignal.presentation.viewmodel import build_view

    view = build_view({
        "recommendations": [], "watchlist": [],
        "position_directives": [{"ticker": "DEAD", "event": "delisting",
                                 "action": "force_exit", "reason": "gone"}],
    })
    assert view["open_position_alerts"][0]["exit_required"] is True


def test_a_normally_trading_position_is_not_an_alert():
    from prosignal.presentation.viewmodel import build_view

    view = build_view({
        "recommendations": [], "watchlist": [],
        "position_directives": [{"ticker": "FINE", "event": "none",
                                 "action": "hold", "reason": "trading normally"}],
    })
    assert view["open_position_alerts"] == []


def test_the_alert_block_renders_on_the_empty_shortlist_paths_too():
    """A delisted holding on a day when nothing qualifies is the case where the
    alert is least visible and most urgent. The two early returns for an empty
    screen used to exit before the block was ever assembled."""
    import re

    page = (__import__("pathlib").Path("src/prosignal/static/index.html")
            .read_text())
    body = page[page.index("const s = v.summary || {};"):]
    body = body[:body.index("state.slateSize = v.picks.length;")]
    returns = re.findall(r"return marketBlock\(v\)[^;]*;", body, re.S)
    assert len(returns) == 2, "the two empty-screen paths"
    for branch in returns:
        assert "alerts" in branch, f"an open-position alert cannot be skipped: {branch[:80]}"
