"""Clearing the history screen must not destroy the research record.

The ledger is where every run is written, `fail_run_if_unwritable` is true by
design, and the comment on that setting says why: an unlogged run corrupts the
deflated-Sharpe trial count. Deleting rows to tidy a screen would silently
invalidate the statistic that decides whether this strategy is
distinguishable from luck, so clearing sets a watermark and the rows stay.
"""

from __future__ import annotations

import datetime as dt

from prosignal.presentation.clearmark import clear_mark, read_mark, set_mark
from prosignal.presentation.history import build_history, load_days


def row(date, stamp, buys=("AAA",)):
    return {"date": date, "logged_at": stamp, "signals_generated": list(buys),
            "watchlist_generated": [], "run_id": "r", "error": None,
            "regime_state": {"trend": "Uptrend", "vol_tercile": "Low"},
            "gate_counts": {}, "stocks_scored": []}


def test_no_watermark_shows_everything(tmp_path):
    assert read_mark(tmp_path) is None
    assert len(load_days([row("2026-01-05", "2026-01-05T17:00:00")])) == 1


def test_a_watermark_hides_what_came_before_it(tmp_path):
    stamp = set_mark(tmp_path, dt.datetime(2026, 1, 6, 12, 0, 0))
    days = load_days([row("2026-01-05", "2026-01-05T17:00:00")], since=stamp)
    assert days == []


def test_runs_after_the_watermark_still_appear(tmp_path):
    stamp = set_mark(tmp_path, dt.datetime(2026, 1, 6, 12, 0, 0))
    days = load_days([row("2026-01-05", "2026-01-05T17:00:00"),
                      row("2026-01-07", "2026-01-07T17:00:00")], since=stamp)
    assert [d.date for d in days] == ["2026-01-07"]


def test_clearing_is_reversible(tmp_path):
    set_mark(tmp_path)
    assert read_mark(tmp_path) is not None
    clear_mark(tmp_path)
    assert read_mark(tmp_path) is None


def test_an_unreadable_watermark_fails_open(tmp_path):
    """Failing closed would hide every run with no way to tell why."""
    set_mark(tmp_path)
    (tmp_path / ".history-cleared").write_text("{not json", encoding="utf-8")
    assert read_mark(tmp_path) is None


def test_a_cleared_history_explains_itself(tmp_path):
    stamp = set_mark(tmp_path, dt.datetime(2026, 1, 6, 12, 0, 0))
    out = build_history([row("2026-01-05", "2026-01-05T17:00:00")], since=stamp)
    assert out["days"] == []
    assert "cleared" in out["note"].lower()
    assert out["cleared_at"] == stamp


def test_the_watermark_lands_atomically(tmp_path):
    """A half-written watermark would be unreadable, and an unreadable one
    fails open -- so a torn write silently un-clears the history."""
    set_mark(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_a_clear_does_not_hide_the_runs_logged_after_it(tmp_path):
    """THE BUG THAT EMPTIED THE PAGE. The open book filtered on `date` -- the
    market session a run scored -- against a watermark that is a wall-clock
    instant. Cleared on Sunday the 30th, the newest run that can exist scored
    Friday the 28th, so every row ever written failed `28 >= 30` and History
    read "No calls yet" through any number of fresh scans. It would not have
    recovered on its own until a session date overtook the afternoon someone
    pressed Clear."""
    from prosignal.presentation.clearmark import kept
    stamp = set_mark(tmp_path, dt.datetime(2026, 8, 30, 19, 55, 9))
    scanned_after_the_clear = row("2026-08-28", "2026-08-30T20:10:00")
    assert kept([scanned_after_the_clear], stamp) == [scanned_after_the_clear]


def test_a_clear_still_hides_the_runs_logged_before_it(tmp_path):
    """The watermark has to keep working, or the fix is just a removal."""
    from prosignal.presentation.clearmark import kept
    stamp = set_mark(tmp_path, dt.datetime(2026, 8, 30, 19, 55, 9))
    assert kept([row("2026-08-28", "2026-08-29T20:50:07")], stamp) == []


def test_a_row_with_no_timestamp_is_kept(tmp_path):
    """Failing open shows more than intended. Failing closed shows nothing,
    which is the failure being fixed."""
    from prosignal.presentation.clearmark import kept
    stamp = set_mark(tmp_path, dt.datetime(2026, 8, 30, 19, 55, 9))
    old = {"date": "2026-08-28", "run_id": "r"}
    assert kept([old], stamp) == [old]


def test_nothing_cleared_keeps_everything(tmp_path):
    from prosignal.presentation.clearmark import kept
    rows = [row("2026-08-28", "2026-08-29T20:50:07")]
    assert kept(rows, None) == rows


def test_the_open_book_reads_the_watermark_the_same_way_the_days_list_does(tmp_path):
    """One watermark, two readers. `load_days` compared `logged_at` to the
    whole stamp and was right; the open book compared `date` to its first ten
    characters and was wrong, so the same clear hid different things on two
    halves of one page."""
    import inspect
    from prosignal import api as A
    src = inspect.getsource(A.create_app)
    body = src[src.index("def _ledger_after_clear"):]
    body = body[:body.index("def _runs_after_clear")]
    assert "clearmark import kept" in body
    assert 'r.get("date")' not in body, (
        "the session a run scored is not comparable to a wall-clock watermark"
    )
