"""Grouping by a categorical symbol must yield only the groups that exist.

SYMBOL is stored as a category over the whole equity master -- 2,880 levels on
this store. pandas' current groupby default emits a group for every level, so a
frame holding five names produces 2,880 groups of which 2,875 are empty. That
is wasted work everywhere and a defect wherever the loop body either records
the key or reduces over the rows:

  * Stage 1's suspect map returned a key for every symbol in the market, each
    mapping to an empty list, instead of the handful with unexplained jumps.
  * Stage 3's earnings map calls min() over each group, which raises on an
    empty one.

pandas is changing this default. These tests pin the intent so the behaviour
does not move when it does.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.data.types import DATE, SYMBOL

CATS = [f"SYM{i:04d}" for i in range(300)]


def _categorical(symbols, **cols):
    frame = pd.DataFrame({SYMBOL: pd.Categorical(symbols, categories=CATS)})
    for k, v in cols.items():
        frame[k] = v
    return frame


def test_stage1_suspect_map_only_lists_symbols_that_have_suspects():
    from prosignal.stages.stage1_data_quality import _suspect_map

    suspects = _categorical(
        ["SYM0001", "SYM0001", "SYM0007"],
        **{DATE: pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])},
    )
    out = _suspect_map(suspects)
    assert set(out) == {"SYM0001", "SYM0007"}, (
        "a key for every category means every symbol in the market reports as "
        "having unexplained action dates"
    )
    assert len(out["SYM0001"]) == 2
    assert "SYM0042" not in out


def test_stage3_earnings_map_survives_a_categorical_symbol():
    """min() over an empty group raises; the default produces one per absent name."""
    from prosignal.core.calendar import TradingCalendar
    from prosignal.stages.stage3_eligibility import _earnings_map

    sessions = [dt.date(2026, 1, d) for d in range(5, 26)]
    calendar = TradingCalendar(sessions)
    frame = _categorical(
        ["SYM0001", "SYM0007"],
        earnings_date=[dt.date(2026, 1, 20), dt.date(2026, 1, 22)],
    )

    class _Store:
        def read_earnings_calendar(self):
            return frame

    out = _earnings_map(_Store(), calendar, dt.date(2026, 1, 6), None)
    assert set(out) == {"SYM0001", "SYM0007"}


def test_a_categorical_groupby_would_otherwise_explode():
    """Documents the scale of the thing being guarded against."""
    frame = _categorical(["SYM0001"] * 3 + ["SYM0007"] * 2)
    assert len(list(frame.groupby(SYMBOL, observed=False))) == len(CATS)
    assert len(list(frame.groupby(SYMBOL, observed=True))) == 2
