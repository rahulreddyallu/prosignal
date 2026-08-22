"""Past runs, and what moved between them.

The ledger has recorded every completed run all along, so this is a read over
an existing record rather than new storage. Two properties carry the weight.

A date is run many times -- the live store held 98 rows for its newest date --
and only the last run for a date reflects what the engine finally said.

And the comparison is deliberately narrow. The ledger keeps which names were
selected, not the factor values behind them, so a change log cannot honestly
say "momentum improved": the evidence was never written down. Position is not
compared either, because runs logged before the interface rebuild ordered their
lists by the penalised score and runs after it order by model rank -- comparing
across that boundary would manufacture movement out of a change in sorting.
"""

from __future__ import annotations

import pytest

from prosignal.presentation.history import (
    BUY, WATCH, build_history, changes, load_days,
)


def row(date, buys, watch, *, logged_at="2026-01-01T00:00:00", run_id="r",
        trend="Uptrend", vol="Low", allow=True, error=None):
    return {
        "date": date, "signals_generated": list(buys),
        "watchlist_generated": list(watch), "logged_at": logged_at,
        "run_id": run_id, "error": error,
        "regime_state": {"trend": trend, "vol_tercile": vol,
                         "allow_new_entries": allow},
        "gate_counts": {"universe_considered": 750},
    }


# ------------------------------------------------------------------- days
def test_the_last_run_for_a_date_is_the_one_that_counts():
    """A date is re-run many times. Only the final one is what the engine said."""
    days = load_days([
        row("2026-08-21", ["OLD"], [], logged_at="2026-08-21T09:00:00"),
        row("2026-08-21", ["NEW"], [], logged_at="2026-08-21T17:00:00"),
    ])
    assert len(days) == 1
    assert days[0].buys == ["NEW"]


def test_rows_appended_out_of_order_still_resolve_to_the_latest():
    days = load_days([
        row("2026-08-21", ["NEW"], [], logged_at="2026-08-21T17:00:00"),
        row("2026-08-21", ["OLD"], [], logged_at="2026-08-21T09:00:00"),
    ])
    assert days[0].buys == ["NEW"]


def test_days_come_back_newest_first():
    days = load_days([row("2026-08-13", [], []), row("2026-08-21", [], []),
                      row("2026-08-17", [], [])])
    assert [d.date for d in days] == ["2026-08-21", "2026-08-17", "2026-08-13"]


def test_a_failed_run_is_not_history():
    days = load_days([row("2026-08-21", ["X"], [], error="pipeline blocked")])
    assert days == []


def test_a_row_without_a_date_is_skipped():
    assert load_days([row(None, ["X"], [])]) == []


# ---------------------------------------------------------------- changes
def _day(date, buys, watch):
    return load_days([row(date, buys, watch)])[0]


def test_a_promotion_is_reported():
    today = _day("2026-08-21", ["AAA"], [])
    prev = _day("2026-08-20", [], ["AAA"])
    d = changes(today, prev)
    assert d["promoted"] == ["AAA"]
    assert "watch to buy" in d["summary"]


def test_a_demotion_is_reported():
    today = _day("2026-08-21", [], ["AAA"])
    prev = _day("2026-08-20", ["AAA"], [])
    assert changes(today, prev)["demoted"] == ["AAA"]


def test_entering_and_leaving_are_reported():
    today = _day("2026-08-21", ["NEW"], [])
    prev = _day("2026-08-20", ["GONE"], [])
    d = changes(today, prev)
    assert d["entered"] == ["NEW"] and d["left"] == ["GONE"]


def test_an_unchanged_shortlist_says_so():
    today = _day("2026-08-21", ["AAA", "BBB"], [])
    prev = _day("2026-08-20", ["AAA", "BBB"], [])
    d = changes(today, prev)
    assert d["entered"] == [] and d["left"] == []
    assert "unchanged" in d["summary"]


def test_only_names_that_reached_a_screen_are_compared():
    """The engine monitors dozens. A name drifting from rank 38 to 41 never
    appeared and still does not, and is not a change worth reporting."""
    today = _day("2026-08-21", ["A", "B", "C", "D", "E"], ["DEEP"])
    prev = _day("2026-08-20", ["A", "B", "C", "D", "E"], ["OTHER"])
    d = changes(today, prev)
    assert d["entered"] == [] and d["left"] == []


def test_the_earliest_run_has_nothing_to_compare_against():
    d = changes(_day("2026-08-21", ["A"], []), None)
    assert d["available"] is False
    assert "earliest run" in d["reason"]


def test_the_change_log_does_not_claim_a_reason_it_cannot_know():
    """Factor values are not retained per name, so 'momentum improved' would be
    invention. The absence is stated rather than left to be inferred."""
    d = changes(_day("2026-08-21", ["AAA"], []), _day("2026-08-20", [], ["AAA"]))
    blob = " ".join(str(v) for v in d.values()).lower()
    for invented in ("momentum improved", "volume strengthened",
                     "relative strength improved"):
        assert invented not in blob
    assert "not recoverable" in d["reason_note"]


# ------------------------------------------------------- slate reconstruction
def test_a_past_slate_is_reconstructed_buys_first_then_near_misses():
    today = _day("2026-08-21", ["B1", "B2"], ["W1", "W2", "W3", "W4"])
    d = build_history([row("2026-08-21", ["B1", "B2"], ["W1", "W2", "W3", "W4"])])
    shown = d["days"][0]["shown"]
    assert [x["ticker"] for x in shown] == ["B1", "B2", "W1", "W2", "W3"]
    assert [x["status"] for x in shown] == [BUY, BUY, WATCH, WATCH, WATCH]


def test_a_thin_day_is_not_padded_in_history_either():
    """2026-08-14 on the live ledger had two candidates in total."""
    d = build_history([row("2026-08-14", [], ["W1", "W2"])])
    assert len(d["days"][0]["shown"]) == 2


# --------------------------------------------------------------- view shape
def test_company_names_are_joined_for_the_history_view():
    d = build_history([row("2026-08-21", ["RELIANCE"], [])],
                      company_names={"RELIANCE": "Reliance Industries Limited"})
    assert d["days"][0]["shown"][0]["company"] == "Reliance Industries"


def test_an_empty_ledger_is_not_an_error():
    d = build_history([])
    assert d["days"] == [] and d["latest_changes"] is None
    assert d["note"]
