"""Frame contracts and the point-in-time store.

The invariants under test are the ones whose violation is silent:
duplicate rows (which double-count volume), forward-filled gaps (which leak),
and non-idempotent appends (which corrupt a store over repeated runs).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.core.errors import DataError, IntegrityError
from prosignal.data.store import DataStore
from prosignal.data.types import (
    DATE,
    OHLCV_COLUMNS,
    SYMBOL,
    coerce_ohlcv,
    from_wide,
    normalise_symbol,
    to_wide,
    validate_ohlcv,
)
from prosignal.data.universe import UniverseResolver

from .conftest import make_sessions, synthetic_prices


# =============================================================================
# symbols & coercion
# =============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("reliance", "RELIANCE"),
        (" TCS ", "TCS"),
        ("INFY.NS", "INFY"),
        ("HDFCBANK.BO", "HDFCBANK"),
        ("m&m", "M&M"),
    ],
)
def test_normalise_symbol(raw, expected):
    assert normalise_symbol(raw) == expected


def test_coerce_fills_missing_columns_with_nan_not_zero():
    df = pd.DataFrame(
        {
            DATE: ["2026-08-14"],
            SYMBOL: ["aaa"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        }
    )
    out = coerce_ohlcv(df, source="test")
    assert list(out.columns) == OHLCV_COLUMNS
    assert out.loc[0, SYMBOL] == "AAA"
    # A missing delivery figure is unknown, not zero.
    assert np.isnan(out.loc[0, "deliv_pct"])
    assert out.loc[0, "adj_factor"] == 1.0
    assert out.loc[0, "source"] == "test"


def test_coerce_drops_rows_without_a_date_or_symbol():
    df = pd.DataFrame(
        {
            DATE: ["2026-08-14", None],
            SYMBOL: ["AAA", "BBB"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    assert len(coerce_ohlcv(df)) == 1


# =============================================================================
# validation
# =============================================================================


def test_validate_accepts_clean_frame(prices):
    assert validate_ohlcv(prices, "clean") is prices


def test_validate_rejects_duplicates(prices):
    dupe = pd.concat([prices, prices.head(1)], ignore_index=True)
    with pytest.raises(DataError) as exc:
        validate_ohlcv(dupe, "dupe")
    assert "duplicate" in str(exc.value)


def test_validate_rejects_high_below_low(prices):
    broken = prices.copy()
    broken.loc[0, "high"] = broken.loc[0, "low"] - 1.0
    with pytest.raises(DataError):
        validate_ohlcv(broken, "hl")


def test_validate_rejects_close_outside_range(prices):
    broken = prices.copy()
    broken.loc[0, "close"] = broken.loc[0, "high"] + 5.0
    with pytest.raises(DataError):
        validate_ohlcv(broken, "bracket")


def test_validate_rejects_negative_prices(prices):
    broken = prices.copy()
    broken.loc[0, "low"] = -1.0
    with pytest.raises(DataError):
        validate_ohlcv(broken, "neg")


def test_validate_missing_required_column():
    with pytest.raises(DataError):
        validate_ohlcv(pd.DataFrame({DATE: [], SYMBOL: []}), "empty")


# =============================================================================
# reshaping
# =============================================================================


def test_to_wide_shape_and_no_forward_fill():
    sessions = make_sessions(5, end=dt.date(2026, 8, 14))
    df = synthetic_prices(["AAA", "BBB"], sessions)
    # Drop one observation to create a genuine gap.
    df = df.drop(df[(df[SYMBOL] == "BBB") & (df[DATE] == pd.Timestamp(sessions[2]))].index)

    wide = to_wide(df, "close")
    assert wide.shape == (5, 2)
    assert list(wide.columns) == ["AAA", "BBB"]
    assert np.isnan(wide.loc[pd.Timestamp(sessions[2]), "BBB"]), (
        "a gap must survive the pivot -- forward-filling across sessions is an "
        "explicit leakage source"
    )


def test_to_wide_reindexes_to_requested_symbols(prices):
    wide = to_wide(prices, "close", symbols=["CCC", "ZZZ"])
    assert list(wide.columns) == ["CCC", "ZZZ"]
    assert wide["ZZZ"].isna().all()


def test_wide_roundtrip(prices):
    wide = to_wide(prices, "close")
    back = from_wide(wide, "close")
    assert len(back) == len(prices)


# =============================================================================
# store
# =============================================================================


@pytest.fixture
def store(tmp_path) -> DataStore:
    return DataStore(tmp_path / "curated", tmp_path / "snapshots")


def test_store_roundtrip(store, prices):
    store.write_prices(prices)
    out = store.read_prices()
    assert len(out) == len(prices)
    assert set(out[SYMBOL]) == {"AAA", "BBB", "CCC"}


def test_store_write_is_idempotent(store, prices):
    store.write_prices(prices)
    store.write_prices(prices)
    assert len(store.read_prices()) == len(prices)
    store.validate_no_duplicates()


def test_store_rewrite_supersedes_earlier_rows(store, prices):
    store.write_prices(prices)
    amended = prices.head(1).copy()
    amended.loc[amended.index[0], "close"] = 999.0
    store.write_prices(amended)
    out = store.read_prices(symbols=[amended.iloc[0][SYMBOL]])
    row = out[out[DATE] == amended.iloc[0][DATE]].iloc[0]
    assert row["close"] == 999.0
    assert len(store.read_prices()) == len(prices)


def test_store_filters_by_symbol_and_date(store, prices):
    store.write_prices(prices)
    sessions = sorted(prices[DATE].dt.date.unique())
    subset = store.read_prices(symbols=["AAA"], start=sessions[-10], end=sessions[-1])
    assert set(subset[SYMBOL]) == {"AAA"}
    assert len(subset) == 10


def test_store_spans_year_boundaries(store):
    sessions = [dt.date(2025, 12, 30), dt.date(2025, 12, 31), dt.date(2026, 1, 1)]
    df = synthetic_prices(["AAA"], sessions)
    store.write_prices(df)
    assert store.prices.years() == [2025, 2026]
    assert len(store.read_prices()) == 3
    assert store.prices.max_date() == dt.date(2026, 1, 1)


def test_known_sessions_comes_from_the_index_table(store):
    sessions = make_sessions(4, end=dt.date(2026, 8, 14))
    frame = pd.DataFrame(
        {
            DATE: [pd.Timestamp(d) for d in sessions],
            "index_name": ["Nifty 50"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    store.write_indices(frame)
    assert store.known_sessions() == sessions
    series = store.index_series("Nifty 50")
    assert len(series) == 4
    assert series.iloc[-1] == 103.0


def test_feed_state_roundtrip(store):
    store.update_feed_state("equity_ohlcv", dt.date(2026, 8, 14), "nse_archives", 3464)
    state = store.feed_state("equity_ohlcv")
    assert state["last_timestamp"] == "2026-08-14"
    assert state["row_count"] == 3464


def test_validate_no_duplicates_raises_on_corruption(store, prices, tmp_path):
    store.write_prices(prices)
    path = store.prices._path(sorted(prices[DATE].dt.year.unique())[0])
    corrupt = pd.concat([pd.read_parquet(path), pd.read_parquet(path).head(1)])
    corrupt.to_parquet(path, index=False)
    with pytest.raises(DataError):
        store.validate_no_duplicates()


# =============================================================================
# universe resolution
# =============================================================================


def _constituents(symbols):
    return pd.DataFrame(
        {
            SYMBOL: symbols,
            "company_name": [f"{s} Ltd" for s in symbols],
            "sector": ["Financial Services"] * len(symbols),
            "series": ["EQ"] * len(symbols),
            "isin": [f"INE{i:06d}01010" for i in range(len(symbols))],
        }
    )


def test_universe_prefers_snapshot_on_or_before_the_date(store):
    resolver = UniverseResolver(store, config=None)
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 6, 1), _constituents(["AAA", "BBB"]))
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 8, 14), _constituents(["AAA", "CCC"]))

    snap = resolver.resolve("NIFTY 200", dt.date(2026, 7, 1))
    assert snap.symbols == ["AAA", "BBB"]
    assert not snap.survivorship_risk
    assert "2026-06-01" in snap.source


def test_universe_flags_survivorship_when_only_later_snapshots_exist(store):
    resolver = UniverseResolver(store, config=None)
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 8, 14), _constituents(["AAA"]))
    snap = resolver.resolve("NIFTY 200", dt.date(2024, 1, 1), pre_snapshot_policy="flag")
    assert snap.survivorship_risk
    assert "SURVIVORSHIP RISK" in (snap.note or "")


def test_universe_halt_policy_refuses_a_biased_run(store):
    resolver = UniverseResolver(store, config=None)
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 8, 14), _constituents(["AAA"]))
    with pytest.raises(IntegrityError):
        resolver.resolve("NIFTY 200", dt.date(2024, 1, 1), pre_snapshot_policy="halt")


def test_universe_excludes_names_not_yet_listed(store):
    resolver = UniverseResolver(store, config=None)
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 8, 14), _constituents(["AAA", "NEWCO"]))
    store.write_equity_master(
        pd.DataFrame(
            {
                SYMBOL: ["AAA", "NEWCO"],
                "company_name": ["A", "N"],
                "series": ["EQ", "EQ"],
                "listing_date": [pd.Timestamp("2010-01-01"), pd.Timestamp("2026-08-01")],
                "isin": ["INE1", "INE2"],
            }
        )
    )
    snap = resolver.resolve("NIFTY 200", dt.date(2026, 7, 1), pre_snapshot_policy="flag")
    assert "NEWCO" not in snap.symbols
    assert snap.excluded_not_yet_listed == ["NEWCO"]


def test_universe_applies_manual_exclusions(store):
    resolver = UniverseResolver(store, config=None)
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 8, 14), _constituents(["AAA", "BBB"]))
    snap = resolver.resolve("NIFTY 200", dt.date(2026, 8, 14), manual_exclusions=["bbb"])
    assert snap.symbols == ["AAA"]
    assert snap.excluded_manual == ["BBB"]


def test_membership_csv_wins_when_it_covers_the_date(store):
    resolver = UniverseResolver(store, config=None)
    store.write_universe_snapshot("NIFTY 200", dt.date(2026, 8, 14), _constituents(["AAA", "BBB"]))
    membership = pd.DataFrame(
        {
            "index_name": ["NIFTY 200", "NIFTY 200"],
            SYMBOL: ["OLDCO", "AAA"],
            "effective_from": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "effective_to": [pd.Timestamp("2026-01-01"), pd.NaT],
        }
    )
    snap = resolver.resolve("NIFTY 200", dt.date(2024, 6, 1), membership_csv=membership)
    assert set(snap.symbols) == {"OLDCO", "AAA"}
    assert not snap.survivorship_risk
    assert snap.source == "index_membership.csv"

    later = resolver.resolve("NIFTY 200", dt.date(2026, 8, 14), membership_csv=membership)
    assert later.symbols == ["AAA"]
