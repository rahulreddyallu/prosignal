"""The open book must be a fact, not a race with file-append order.

`Ledger.previous_run` is the engine's ENTIRE position memory: Stage 3's held
bypass, Stage 6's hysteresis, Stage 8's portfolio limits and the orphan review
all read the open book out of the one row it returns. Two defects lived in that
one selection.

FILE ORDER DECIDED IT. The rule was `if latest_date is None or when >=
latest_date`, so within a date the last line in the file won. Measured on the
shipped ledger: 5 market dates carry runs that disagree about the book, and
2026-08-18 carries 676 runs across 14 configuration versions recording SEVEN
different books. Which one the engine believed was settled by whichever process
happened to flush last.

LINEAGE WAS NOT TRACKED. `mode` has been on the row since v1 and every path
wrote the literal "live", so a `--date` backfill appended after a live run
became that run's successor and the next live session inherited its book from a
reconstruction.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.ledger import AmbiguousLedgerHistory, Ledger, LedgerRow


def _row(when: dt.date, signals, run_id: str, *, mode: str = "live",
         logged_at: dt.datetime = None, slate=None) -> LedgerRow:
    return LedgerRow(
        trial_id=f"T-{run_id}", run_id=run_id, date=when,
        logged_at=logged_at or dt.datetime.combine(when, dt.time(18, 0)),
        engine_version="0.1.0", schema_version="1", config_version="test",
        mode=mode, regime_state={}, eligible_universe_size=100,
        universe_considered=120, stocks_scored=[], signals_generated=list(signals),
        watchlist_generated=[],
        slate_shown=[{"ticker": t, "position": i + 1}
                     for i, t in enumerate(slate or signals)],
        no_trade=False, gate_counts={}, data_quality_flags=[],
        survivorship_risk=False, stage_timings_ms={}, duration_ms=1.0,
    )


# =============================================================================
# ambiguity
# =============================================================================

def test_two_runs_on_one_date_that_disagree_about_the_book_are_fatal(tmp_path):
    """The 2026-08-18 case, minimised. There is no fact of the matter about
    what was held, and every available fallback invents one."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "r1",
                    logged_at=dt.datetime(2026, 8, 17, 18, 0)))
    led.append(_row(dt.date(2026, 8, 17), ["BBB", "CCC"], "r2",
                    logged_at=dt.datetime(2026, 8, 17, 19, 0)))

    with pytest.raises(AmbiguousLedgerHistory) as caught:
        led.previous_run(before=dt.date(2026, 8, 18))

    message = str(caught.value)
    assert "2026-08-17" in message
    assert "AAA" in message and "BBB" in message, (
        "the operator has to be told WHICH books disagree, or the error is "
        "not actionable"
    )


def test_reruns_that_agree_are_not_a_conflict(tmp_path):
    """Pressing SCAN twice is ordinary. Only DISAGREEMENT is fatal -- a rule
    that refused every repeated date would block the normal operator action."""
    led = Ledger(tmp_path)
    for i in range(4):
        led.append(_row(dt.date(2026, 8, 17), ["AAA", "BBB"], f"r{i}",
                        logged_at=dt.datetime(2026, 8, 17, 18 + i, 0)))
    row = led.previous_run(before=dt.date(2026, 8, 18))
    assert row["signals_generated"] == ["AAA", "BBB"]
    assert row["run_id"] == "r3", "the newest agreeing row is the one to serve"


def test_a_differing_slate_is_a_conflict_even_when_the_book_matches(tmp_path):
    """`previous_slate` drives the screen and comes out of the same row. Two
    runs holding the same names while showing different ones still describe two
    incompatible states."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "r1", slate=["AAA", "XXX"],
                    logged_at=dt.datetime(2026, 8, 17, 18, 0)))
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "r2", slate=["AAA", "YYY"],
                    logged_at=dt.datetime(2026, 8, 17, 19, 0)))
    with pytest.raises(AmbiguousLedgerHistory):
        led.previous_run(before=dt.date(2026, 8, 18))


def test_conflicting_dates_reports_before_a_run_walks_into_it(tmp_path):
    """The diagnostic: which days need cleaning, without failing an analysis
    to find out."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 14), ["AAA"], "ok1"))
    led.append(_row(dt.date(2026, 8, 17), ["AAA"], "bad1",
                    logged_at=dt.datetime(2026, 8, 17, 18, 0)))
    led.append(_row(dt.date(2026, 8, 17), ["ZZZ"], "bad2",
                    logged_at=dt.datetime(2026, 8, 17, 19, 0)))

    found = led.conflicting_dates()
    assert [f["date"] for f in found] == ["2026-08-17"]
    assert found[0]["runs"] == 2
    assert found[0]["distinct_books"] == 2


def test_a_clean_ledger_reports_no_conflicts(tmp_path):
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 14), ["AAA"], "r1"))
    led.append(_row(dt.date(2026, 8, 17), ["BBB"], "r2"))
    assert led.conflicting_dates() == []


# =============================================================================
# lineage
# =============================================================================

def test_a_live_run_does_not_inherit_a_replay_book(tmp_path):
    """The defect: backfill a past date, and the next live session reads its
    open book out of the reconstruction."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 14), ["LIVE1", "LIVE2"], "live1"))
    # A backfill of the SAME date, run later, recording a different book.
    led.append(_row(dt.date(2026, 8, 14), ["REPLAY1"], "replay1", mode="replay",
                    logged_at=dt.datetime(2026, 8, 20, 9, 0)))

    assert led.open_book(before=dt.date(2026, 8, 17)) == ["LIVE1", "LIVE2"]
    assert led.open_book(before=dt.date(2026, 8, 17), mode="replay") == ["REPLAY1"]


def test_a_replay_chains_to_replays_not_to_live_history(tmp_path):
    """A backfill sequence is its own experiment. It must reconstruct from what
    it itself produced, or each date in the sequence restarts from live."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 14), ["LIVE1"], "live1"))
    led.append(_row(dt.date(2026, 8, 17), ["R1"], "rep1", mode="replay"))
    assert led.open_book(before=dt.date(2026, 8, 18), mode="replay") == ["R1"]


def test_mode_none_reads_across_every_lineage(tmp_path):
    """A reporting caller wants the whole history; a pipeline caller never
    does."""
    led = Ledger(tmp_path)
    led.append(_row(dt.date(2026, 8, 14), ["LIVE1"], "live1"))
    led.append(_row(dt.date(2026, 8, 17), ["R1"], "rep1", mode="replay"))
    row = led.previous_run(before=dt.date(2026, 8, 18), mode=None)
    assert row["run_id"] == "rep1"


def test_a_row_written_before_mode_existed_counts_as_live(tmp_path):
    """Every one of the 1,942 shipped rows says "live". A missing or empty
    value must not silently drop the whole history out of the live lineage."""
    led = Ledger(tmp_path)
    path = led._path(dt.date(2026, 8, 14))
    path.write_text(
        '{"trial_id":"T-old","run_id":"old","date":"2026-08-14",'
        '"logged_at":"2026-08-14T18:00:00","engine_version":"0.1.0",'
        '"schema_version":"1","config_version":"test",'
        '"signals_generated":["AAA"]}\n', encoding="utf-8")
    assert led.open_book(before=dt.date(2026, 8, 17)) == ["AAA"]


# =============================================================================
# determinism
# =============================================================================

def test_selection_does_not_depend_on_position_in_the_file(tmp_path):
    """Two clones of one ledger must resolve the same book. The rule orders by
    (logged_at, run_id), not by whichever process flushed last."""
    forward = Ledger(tmp_path / "forward")
    backward = Ledger(tmp_path / "backward")
    rows = [
        _row(dt.date(2026, 8, 17), ["AAA", "BBB"], "r1",
             logged_at=dt.datetime(2026, 8, 17, 18, 0)),
        _row(dt.date(2026, 8, 17), ["AAA", "BBB"], "r2",
             logged_at=dt.datetime(2026, 8, 17, 19, 0)),
        _row(dt.date(2026, 8, 17), ["AAA", "BBB"], "r3",
             logged_at=dt.datetime(2026, 8, 17, 20, 0)),
    ]
    for r in rows:
        forward.append(r)
    for r in reversed(rows):          # same records, opposite file order
        backward.append(r)

    a = forward.previous_run(before=dt.date(2026, 8, 18))
    b = backward.previous_run(before=dt.date(2026, 8, 18))
    assert a["run_id"] == b["run_id"] == "r3"
