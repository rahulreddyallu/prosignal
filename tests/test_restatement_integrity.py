"""Restatements must not rewrite history.

If a later restatement overwrites the figure a quarter was originally filed
with, every retrain sees a number the market did not have at the time, and the
leak is invisible: the filing date still looks correct.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.data.types import SYMBOL
from prosignal.features.fundamentals import compute_features


def _filing(filing_date, period_end, net_profit):
    return pd.DataFrame([{
        SYMBOL: "X",
        "filing_date": pd.Timestamp(filing_date),
        "period_end": pd.Timestamp(period_end),
        "period_start": pd.Timestamp(period_end) - pd.Timedelta(days=90),
        "revenue": 1000.0, "net_profit": net_profit,
        "profit_before_tax": net_profit * 1.3, "finance_costs": 10.0,
        "paid_up_capital": 1000.0, "face_value": 10.0, "shares_outstanding": 100.0,
    }])


#: Four quarters, because earnings yield is a trailing-twelve-month figure.
_QUARTERS = [
    ("2024-08-01", "2024-06-30", 50.0),
    ("2024-11-01", "2024-09-30", 50.0),
    ("2025-02-01", "2024-12-31", 50.0),
    ("2025-05-01", "2025-03-31", 100.0),
]
HISTORY = pd.concat([_filing(f, p, n) for f, p, n in _QUARTERS], ignore_index=True)

ORIGINAL = _filing("2025-05-01", "2025-03-31", 100.0)
#: The March quarter revised down, filed six months after the original.
RESTATED = _filing("2025-11-01", "2025-03-31", 40.0)
BOTH = pd.concat([HISTORY, RESTATED], ignore_index=True)

#: market cap is 100 shares at Rs 50.
_MCAP = 5000.0


def test_the_store_keeps_both_versions(tmp_path):
    """Keyed on filing date, so a restatement is a new record rather than an
    overwrite of the original."""
    from prosignal.data.store import DataStore

    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_fundamentals(ORIGINAL)
    store.write_fundamentals(RESTATED)
    stored = store.read_fundamentals()
    assert len(stored) == 2
    assert sorted(stored["net_profit"].tolist()) == [40.0, 100.0]


def test_a_date_before_the_restatement_sees_the_original_figure():
    """The rule the whole thing rests on. TTM = 50 + 50 + 50 + 100."""
    feats = compute_features(BOTH, {"X": 50.0}, dt.date(2025, 8, 1))
    assert feats.iloc[0]["earnings_yield"] == pytest.approx(250.0 / _MCAP)


def test_a_date_after_the_restatement_sees_the_revised_figure():
    """TTM = 50 + 50 + 50 + 40 once the revision is public."""
    feats = compute_features(BOTH, {"X": 50.0}, dt.date(2026, 1, 1))
    assert feats.iloc[0]["earnings_yield"] == pytest.approx(190.0 / _MCAP)


def test_the_restatement_is_invisible_on_its_own_filing_eve():
    """One day before it was filed, the revision does not exist."""
    feats = compute_features(BOTH, {"X": 50.0}, dt.date(2025, 10, 31))
    assert feats.iloc[0]["earnings_yield"] == pytest.approx(250.0 / _MCAP)


def test_a_same_day_correction_replaces_rather_than_duplicates(tmp_path):
    """Two figures filed the same day for the same quarter is a correction, not
    a restatement; keeping both would leave the reader to guess."""
    from prosignal.data.store import DataStore

    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_fundamentals(ORIGINAL)
    store.write_fundamentals(_filing("2025-05-01", "2025-03-31", 95.0))
    stored = store.read_fundamentals()
    assert len(stored) == 1
    assert stored.iloc[0]["net_profit"] == pytest.approx(95.0)
