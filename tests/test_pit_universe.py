"""The point-in-time liquidity universe.

Index membership cannot be made point-in-time from what NSE publishes: only the
current constituent list is served, so any historical date is scored against
today's members. A trailing-turnover screen needs no membership history and is
survivorship-free by construction.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.core.errors import IntegrityError
from prosignal.data.universe import UniverseResolver


class _Store:
    def __init__(self, prices: pd.DataFrame, master: pd.DataFrame):
        self._prices, self._master = prices, master

    def price_sessions(self):
        return sorted({d.date() for d in pd.to_datetime(self._prices["date"])})

    def read_prices(self, start=None, end=None, columns=None, symbols=None):
        p = self._prices.copy()
        p["date"] = pd.to_datetime(p["date"])
        if start is not None:
            p = p[p["date"] >= pd.Timestamp(start)]
        if end is not None:
            p = p[p["date"] <= pd.Timestamp(end)]
        return p

    def read_equity_master(self):
        return self._master


def _store(turnovers, listing="2019-01-01", n=80):
    idx = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for sym, tno in turnovers.items():
        for d in idx:
            rows.append({"date": d, "symbol": sym, "close": 150.0, "turnover": tno})
    master = pd.DataFrame(
        [{"symbol": s, "listing_date": pd.Timestamp(listing)} for s in turnovers]
    )
    return _Store(pd.DataFrame(rows), master), idx[-1].date()


def _resolve(store, as_of, **kw):
    args = dict(
        as_of=as_of, min_adtv_inr=5e7, lookback_sessions=60, max_names=750,
        min_history_sessions=10, min_price_inr=20.0,
    )
    args.update(kw)
    return UniverseResolver(store, object()).resolve_liquidity_pit(**args)


def test_the_screen_admits_liquid_names_and_rejects_thin_ones():
    store, as_of = _store({"LIQUID": 9e7, "THIN": 1e6})
    snap = _resolve(store, as_of)
    assert "LIQUID" in snap.symbols
    assert "THIN" not in snap.symbols


def test_it_never_reports_survivorship_risk():
    """The whole point: no membership list is consulted, so there is nothing to
    be biased by."""
    store, as_of = _store({"A": 9e7, "B": 8e7})
    snap = _resolve(store, as_of)
    assert snap.survivorship_risk is False
    assert snap.index_name == "LIQUIDITY-PIT"


def test_names_are_ranked_by_turnover_and_capped():
    store, as_of = _store({"BIG": 9e8, "MID": 5e8, "SMALL": 9e7})
    snap = _resolve(store, as_of, max_names=2)
    assert snap.symbols == sorted(["BIG", "MID"])


def test_a_recent_listing_is_excluded_for_want_of_history():
    store, as_of = _store({"OLD": 9e7, "NEW": 9e7})
    store._master = pd.DataFrame([
        {"symbol": "OLD", "listing_date": pd.Timestamp("2019-01-01")},
        {"symbol": "NEW", "listing_date": pd.Timestamp("2025-06-01")},
    ])
    snap = _resolve(store, as_of, min_history_sessions=40)
    assert "OLD" in snap.symbols
    assert "NEW" not in snap.symbols


def test_an_empty_screen_raises_rather_than_returning_nothing():
    store, as_of = _store({"THIN": 1e5})
    with pytest.raises(IntegrityError):
        _resolve(store, as_of)


def test_sectors_are_attached_where_known_and_flagged_where_not():
    store, as_of = _store({"A": 9e7, "B": 8e7})
    snap = _resolve(store, as_of, sector_map={"A": "Financial Services"})
    assert snap.sector_of("A") == "Financial Services"
    assert snap.sector_of("B") == "Unknown"
    assert "sector known for 1" in (snap.note or "")


def test_the_screen_reads_no_price_after_the_decision_date():
    store, as_of = _store({"A": 9e7, "B": 8e7})
    future = store._prices.copy()
    future.loc[:, "date"] = pd.to_datetime(future["date"])
    extra = future[future["symbol"] == "B"].tail(1).copy()
    extra["date"] = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    extra["turnover"] = 1e12          # would dominate the ranking if it leaked
    store._prices = pd.concat([future, extra], ignore_index=True)
    snap = _resolve(store, as_of, max_names=1)
    assert snap.symbols == ["A"], "a post-decision session changed the ranking"
