"""A position ends when the ENGINE ends it, not only at a stop or a target.

Stage 6 admits at rank <= entry_rank and holds while the name stays inside
exit_rank. The book is the exit rule; the stop and the target are the two ways
a position can end early. Outcome resolution modelled the stop, the target and
the holding-period limit, and not the book -- so a position that the engine had
already closed kept running in the record until a level happened to be touched.

Measured on the recorded record before this was added: the simulation held past
the engine's own exit in **94% of trades**, by a median of **14 sessions**.
Median simulated hold was 15 sessions against a book that had let the name go
after 1. Every figure the History page showed was computed over those phantom
sessions.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.outcomes import (
    EXIT_MODEL, _REFUSED, _closed_by_engine, _resolve_one, book_by_date,
    load_outcomes,
)

SESSIONS = [d.date() for d in pd.bdate_range("2026-06-01", "2026-09-30")]


class _Costs:
    def round_trip(self, entry_price, qty, exit_price=None):
        class _B:
            total_inr = 0.0
        return _B()


class _Cfg:
    class params:
        class capital:
            @staticmethod
            def position_value_inr():
                return 100_000.0


def _bars(prices):
    """prices: list of (date, open, high, low, close)."""
    return pd.DataFrame({
        "date": pd.to_datetime([p[0] for p in prices]),
        "open": [p[1] for p in prices],
        "high": [p[2] for p in prices],
        "low": [p[3] for p in prices],
        "close": [p[4] for p in prices],
    })


def _flat(n=40, price=100.0, start=0):
    """A series that never touches a stop or a target."""
    return [(SESSIONS[start + i], price, price + 0.5, price - 0.5, price)
            for i in range(n)]


def _resolve(bars, book, *, stop=90.0, t1=112.0, signal_idx=0, max_hold=63,
             last_close=100.0):
    item = {
        "run_id": "r1",
        "date": SESSIONS[signal_idx].isoformat(),
        "rec": {"ticker": "AAA", "last_close": last_close, "stop": stop,
                "target_1": t1, "target_2": 120.0},
        "config_version": "c", "engine_version": "e",
    }
    return _resolve_one(item, {"AAA": bars}, max_hold, _Costs(), _Cfg,
                        SESSIONS[-1], book)


# ------------------------------------------------------------- the book exit
def test_the_position_closes_when_the_engine_stops_holding_it():
    bars = _bars(_flat(20))
    # Held on the entry session and the one after; dropped at SESSIONS[2]'s close.
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"},
            SESSIONS[2]: {"BBB"}, SESSIONS[3]: {"BBB"}}
    out = _resolve(bars, book)
    assert out["exit_reason"] == "book_exit"
    # Dropped at SESSIONS[2]'s close -> filled at SESSIONS[3]'s open.
    assert out["exit_date"] == str(SESSIONS[3])


def test_it_fills_at_the_next_open_not_the_close_that_decided_it():
    prices = _flat(20)
    prices[3] = (SESSIONS[3], 104.0, 104.5, 103.5, 104.0)   # gap up on the exit bar
    bars = _bars(prices)
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"}, SESSIONS[2]: set()}
    out = _resolve(bars, book)
    assert out["exit_price"] == 104.0, "the engine decides at a close and trades the next open"


def test_a_stop_on_the_same_bar_takes_precedence():
    """Daily bars cannot order a stop against a decision taken at the previous
    close, and the stop is the worse outcome."""
    prices = _flat(20)
    prices[3] = (SESSIONS[3], 100.0, 100.5, 85.0, 88.0)     # stop is breached
    bars = _bars(prices)
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"}, SESSIONS[2]: set()}
    out = _resolve(bars, book)
    assert out["exit_reason"] == "stop"
    assert out["exit_price"] == 90.0


def test_a_target_reached_before_the_engine_exits_still_wins():
    prices = _flat(20)
    prices[2] = (SESSIONS[2], 100.0, 113.0, 99.5, 112.5)    # target on the way
    bars = _bars(prices)
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"}, SESSIONS[2]: set()}
    out = _resolve(bars, book)
    assert out["exit_reason"] == "target_1"


def test_a_session_with_no_recorded_run_is_not_an_exit():
    """Absent information is not a decision. Treating a day the engine simply
    was not run as an exit would close every position on the first gap in the
    record."""
    bars = _bars(_flat(20))
    book = {SESSIONS[0]: {"AAA"}}          # nothing recorded after that
    out = _resolve(bars, book)
    assert out is None or out["exit_reason"] != "book_exit"


def test_a_name_the_engine_keeps_holding_runs_to_its_levels():
    prices = _flat(20)
    prices[10] = (SESSIONS[10], 100.0, 113.0, 99.5, 112.5)
    bars = _bars(prices)
    book = {s: {"AAA"} for s in SESSIONS[:20]}
    out = _resolve(bars, prices and book)
    assert out["exit_reason"] == "target_1"
    assert out["sessions_held"] == 9


def test_no_book_at_all_behaves_like_the_old_model():
    """A caller with no ledger to offer must not have every position closed."""
    prices = _flat(20)
    prices[10] = (SESSIONS[10], 100.0, 113.0, 99.5, 112.5)
    out = _resolve(_bars(prices), {})
    assert out["exit_reason"] == "target_1"


# ----------------------------------------------------------- the book reader
def test_the_last_run_recorded_for_a_date_is_the_one_that_stands():
    rows = [
        {"date": "2026-06-01", "logged_at": "2026-06-01T10:00", "signals_generated": ["OLD"]},
        {"date": "2026-06-01", "logged_at": "2026-06-01T18:00", "signals_generated": ["NEW"]},
    ]
    assert book_by_date(rows)[dt.date(2026, 6, 1)] == {"NEW"}


def test_a_failed_run_does_not_become_the_book():
    rows = [
        {"date": "2026-06-01", "logged_at": "2026-06-01T10:00", "signals_generated": ["GOOD"]},
        {"date": "2026-06-01", "logged_at": "2026-06-01T18:00", "error": "boom",
         "signals_generated": []},
    ]
    assert book_by_date(rows)[dt.date(2026, 6, 1)] == {"GOOD"}


def test_a_date_with_no_run_is_absent_rather_than_empty():
    book = book_by_date([{"date": "2026-06-01", "logged_at": "x",
                          "signals_generated": ["AAA"]}])
    assert dt.date(2026, 6, 2) not in book
    assert _closed_by_engine(book, dt.date(2026, 6, 2), "AAA") is False
    assert _closed_by_engine(book, dt.date(2026, 6, 1), "AAA") is False


def test_a_run_that_did_not_name_the_ticker_is_an_exit():
    book = book_by_date([{"date": "2026-06-01", "logged_at": "x",
                          "signals_generated": ["OTHER"]}])
    assert _closed_by_engine(book, dt.date(2026, 6, 1), "AAA") is True


def test_a_run_that_held_nothing_still_closes_the_position():
    """An empty book is a decision -- the engine ran and holds nothing."""
    book = book_by_date([{"date": "2026-06-01", "logged_at": "x",
                          "signals_generated": []}])
    assert _closed_by_engine(book, dt.date(2026, 6, 1), "AAA") is True


# --------------------------------------------------------- model versioning
def test_every_resolved_row_carries_the_model_that_produced_it():
    bars = _bars(_flat(20))
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"}, SESSIONS[2]: set()}
    assert _resolve(bars, book)["exit_model"] == EXIT_MODEL


def test_rows_from_an_older_exit_rule_are_not_served(tmp_path):
    """Two strategies in one file averaged together is exactly the failure this
    module exists to detect elsewhere."""
    import json

    path = tmp_path / "outcomes.jsonl"
    path.write_text(
        json.dumps({"ticker": "OLD", "net_return": 0.5}) + "\n"          # unstamped
        + json.dumps({"ticker": "STALE", "net_return": 0.5,
                      "exit_model": "stop-target-v1"}) + "\n"
        + json.dumps({"ticker": "CURRENT", "net_return": 0.1,
                      "exit_model": EXIT_MODEL}) + "\n"
    )
    assert [r["ticker"] for r in load_outcomes(path)] == ["CURRENT"]
    assert len(load_outcomes(path, model=None)) == 3, "the record is still readable"


# ------------------------------------------------------- the price basis
def _bars_with_close(prices):
    df = _bars(prices)
    return df


def test_levels_recorded_before_a_split_are_rebased_not_compared_raw():
    """The defect, stated as a test.

    BAJFINANCE was signalled on 2025-05-02 with a stop of 8195.05 against a
    close of 8862.50. A 4:1 bonus with a 2:1 face split landed on 2025-06-16 and
    the store now serves that session at a close of 886.25. The stop then sat
    ten times above every subsequent low, so the position "stopped out" on its
    first bar at 8195.05 against an entry of 887.75 -- a loss recorded as
    **+823%**. Twenty-nine trades cleared +50% this way.
    """
    # Store prices are post-adjustment (a tenth of the recorded basis).
    prices = [(SESSIONS[0], 88.6, 89.0, 88.0, 88.6)] + _flat(20, price=88.0, start=1)
    bars = _bars(prices)
    item = {
        "run_id": "r1", "date": SESSIONS[0].isoformat(),
        "rec": {"ticker": "AAA", "last_close": 886.0,      # pre-split basis
                "stop": 820.0, "target_1": 990.0, "target_2": 1100.0},
        "config_version": "c", "engine_version": "e",
    }
    # A book exit so the position closes and can be scored at all.
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"}, SESSIONS[2]: set()}
    out = _resolve_one(item, {"AAA": bars}, 63, _Costs(), _Cfg, SESSIONS[-1], book)
    assert out is not None
    assert out["price_basis_factor"] == pytest.approx(0.1, rel=1e-3)
    # 82.0 is the rebased stop; a flat series at 88 never reaches it.
    assert out["exit_reason"] != "stop", "the raw stop would have fired instantly"
    assert abs(float(out["gross_return"])) < 0.5, "no 800% loss"


def test_an_unadjusted_run_needs_no_correction():
    prices = _flat(20, price=100.0)
    book = {SESSIONS[0]: {"AAA"}, SESSIONS[1]: {"AAA"}, SESSIONS[2]: set()}
    out = _resolve(_bars(prices), book)
    assert out["price_basis_factor"] == pytest.approx(1.0)


def test_a_trade_whose_basis_cannot_be_established_is_refused():
    """A trade that cannot be priced honestly is not evidence. Scoring it on
    whichever basis happened to be handy is how +823% got into the record."""
    prices = _flat(20, price=100.0)
    item = {
        "run_id": "r1", "date": SESSIONS[0].isoformat(),
        "rec": {"ticker": "AAA", "stop": 90.0, "target_1": 112.0},  # no last_close
        "config_version": "c", "engine_version": "e",
    }
    assert _resolve_one(item, {"AAA": _bars(prices)}, 63, _Costs(), _Cfg,
                        SESSIONS[-1], {}) is _REFUSED


def test_an_absurd_basis_ratio_is_refused_rather_than_applied():
    prices = _flat(20, price=100.0)
    item = {
        "run_id": "r1", "date": SESSIONS[0].isoformat(),
        "rec": {"ticker": "AAA", "last_close": 1e-9, "stop": 90.0,
                "target_1": 112.0},
        "config_version": "c", "engine_version": "e",
    }
    assert _resolve_one(item, {"AAA": _bars(prices)}, 63, _Costs(), _Cfg,
                        SESSIONS[-1], {}) is _REFUSED


def test_the_rebased_levels_still_fire_when_the_price_reaches_them():
    """Rebasing must correct the scale, not disable the stop."""
    prices = _flat(20, price=88.0)
    prices[3] = (SESSIONS[3], 88.0, 88.5, 80.0, 81.0)   # through the rebased 82.0
    item = {
        "run_id": "r1", "date": SESSIONS[0].isoformat(),
        "rec": {"ticker": "AAA", "last_close": 880.0, "stop": 820.0,
                "target_1": 990.0, "target_2": 1100.0},
        "config_version": "c", "engine_version": "e",
    }
    out = _resolve_one(item, {"AAA": _bars(prices)}, 63, _Costs(), _Cfg,
                       SESSIONS[-1], {})
    assert out["exit_reason"] == "stop"
    assert out["exit_price"] == pytest.approx(82.0)


def test_a_refusal_is_counted_separately_from_a_running_position(tmp_path):
    """A refused trade is not running, it is unscoreable. Collapsing the two
    would report a shrinking sample as patience."""
    import json

    import pandas as pd

    from prosignal import outcomes as O

    class _Store:
        def __init__(self, frame):
            self._f = frame
        def price_sessions(self):
            return [d.date() for d in self._f["date"]]
        def read_prices(self, symbols=None, start=None, end=None):
            return self._f

    days = pd.bdate_range("2024-01-01", periods=80)
    frame = pd.DataFrame({
        "symbol": "X", "date": days, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0,
    })
    led, out = tmp_path / "ledger", tmp_path / "outcomes.jsonl"
    led.mkdir()
    (led / "runs-2024.jsonl").write_text(json.dumps({
        "run_id": "r1", "date": "2024-01-01", "signals_generated": ["X"],
        # no last_close -> the basis cannot be established
        "stocks_scored": [{"ticker": "X", "stop": 90.0, "target_1": 110.0}],
    }) + "\n")

    from prosignal.config.loader import load_config
    stats = O.resolve_pending(_Store(frame), led, out, load_config(),
                              as_of=days[-1].date())
    assert stats["refused"] == 1
    assert stats["resolved"] == 0
    assert stats["still_open"] == 0, "a refusal must not read as a live position"


def test_a_merged_api_row_carries_one_price_basis():
    """`outcomes_for` re-bases the recorded levels; the raw record still holds
    the originals. Merging them verbatim shipped both -- a stop of 8195.05
    beside a re-based 819.51 for the same call -- and left the interface free to
    render either."""
    from prosignal.api import _in_outcome_basis

    class _Out:
        signal_price = 886.25
        stop = 819.51
        target_1 = 986.37

    raw = {"ticker": "AAA", "signal_price": 8862.5, "stop": 8195.05,
           "target_1": 9863.7, "status": "BUY"}
    merged = _in_outcome_basis(raw, _Out())
    assert merged["signal_price"] == 886.25
    assert merged["stop"] == 819.51
    assert merged["target_1"] == 986.37
    assert merged["status"] == "BUY", "the rest of the record is untouched"
    assert raw["stop"] == 8195.05, "the input must not be mutated"
