"""The open book must come from a PRIOR run, never the current date.

Stage 6's exit band needs to know what the previous run committed to, and the
engine holds no live position state, so it reads the ledger. The pipeline also
WRITES to the ledger at the end of every run -- so without a date filter, a
second run on the same day would read the first run's output as its own open
book and hold names on hysteresis it had not actually held. Re-running an
analysis is something an operator does routinely.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.ledger import Ledger, LedgerRow


def _row(when: dt.date, signals, run_id: str) -> LedgerRow:
    return LedgerRow(
        trial_id=f"T-{run_id}", run_id=run_id, date=when,
        logged_at=dt.datetime.combine(when, dt.time(18, 0)),
        engine_version="0.1.0", schema_version="1", config_version="test",
        mode="live", regime_state={}, eligible_universe_size=100,
        universe_considered=120, stocks_scored=[], signals_generated=list(signals),
        watchlist_generated=[], no_trade=False, gate_counts={},
        data_quality_flags=[], survivorship_risk=False, stage_timings_ms={},
        duration_ms=1.0,
    )


def test_the_open_book_is_the_last_recorded_run(tmp_path):
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 14), ["AAA", "BBB"], "r1"))
    led.append(_row(dt.date(2026, 8, 17), ["BBB", "CCC"], "r2"))
    assert led.open_book() == ["BBB", "CCC"]


def test_a_rerun_on_the_same_date_does_not_read_its_own_output(tmp_path):
    """The defect this guards: run twice, and the second run inherits a book
    the operator never held."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "prior"))
    led.append(_row(dt.date(2026, 8, 18), ["XXX", "YYY", "ZZZ"], "today"))

    same_day = led.open_book(before=dt.date(2026, 8, 18))
    assert same_day == ["AAA"], (
        "a second run on 2026-08-18 must see the 17th's book, not the one the "
        "first run of the same day just wrote"
    )


def test_an_empty_ledger_is_an_empty_book_not_an_error(tmp_path):
    """A first run holds nothing. That is a state, not a failure."""
    assert Ledger(tmp_path).open_book() == []


def test_a_run_that_bought_nothing_leaves_an_empty_book(tmp_path):
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "r1"))
    led.append(_row(dt.date(2026, 8, 18), [], "r2"))
    assert led.open_book() == []


def test_a_malformed_date_is_skipped_rather_than_crashing_the_run(tmp_path):
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "r1"))
    (tmp_path / "runs-2026.jsonl").open("a").write(
        '{"date":"not-a-date","signals_generated":["JUNK"]}\n'
    )
    assert led.open_book() == ["AAA"], "a bad row must not become the book"
