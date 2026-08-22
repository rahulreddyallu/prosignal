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
