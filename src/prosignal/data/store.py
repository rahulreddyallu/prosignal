"""On-disk point-in-time data store (parquet + JSON state).

Layout under ``data/``::

    curated/prices/year=2025.parquet        tidy OHLCV, one file per year
    curated/indices/year=2025.parquet       tidy index OHLC incl. India VIX
    curated/delivery/year=2025.parquet      delivery qty / %
    curated/open_interest/year=2025.parquet stock-futures OI
    curated/equity_master.parquet           symbol -> listing date, ISIN
    curated/corporate_actions.parquet       ex-date-stamped adjustment ratios
    curated/earnings_calendar.parquet       scheduled results dates
    snapshots/universe/<INDEX>/<date>.parquet   dated membership snapshots
    curated/_state.json                     per-feed last-update bookkeeping

Three properties this module guarantees:

**Idempotent appends.** Writing the same session twice is a no-op, not a
duplicate. Duplicate ``(date, symbol)`` rows would double-count volume and
corrupt every cross-sectional rank, so dedup happens on write, not on read.

**Atomic writes.** Every file lands via ``.tmp`` + ``os.replace``, so an
interrupted ingest can never leave a half-written parquet that reads as
plausible-but-wrong data.

**No forward-fill, anywhere.** Gaps stay gaps. Filling them is a leakage source
the research program's section 7 checklist names explicitly, and a store that
did it silently would defeat Stage 1's continuity check.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..core.errors import DataError
from ..core.logging import get_logger
from .types import (
    CORPORATE_ACTION_COLUMNS,
    DATE,
    SYMBOL,
    empty_index_frame,
    empty_ohlcv,
    normalise_symbol,
)

__all__ = ["DataStore"]

log = get_logger(__name__)

_STATE_FILE = "_state.json"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover
                pass


def _atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover
                pass


class _PartitionedTable:
    """A tidy frame stored as one parquet file per calendar year."""

    def __init__(self, root: Path, name: str, key_columns: Sequence[str]) -> None:
        self.dir = root / name
        self.name = name
        self.key_columns = list(key_columns)

    def _path(self, year: int) -> Path:
        return self.dir / f"year={year}.parquet"

    def years(self) -> List[int]:
        if not self.dir.is_dir():
            return []
        out: List[int] = []
        for p in self.dir.glob("year=*.parquet"):
            try:
                out.append(int(p.stem.split("=", 1)[1]))
            except (IndexError, ValueError):
                continue
        return sorted(out)

    def write(self, df: pd.DataFrame) -> int:
        """Merge ``df`` into the store, deduplicating on the key columns."""
        if df is None or df.empty:
            return 0
        frame = df.copy()
        frame[DATE] = pd.to_datetime(frame[DATE]).dt.normalize()
        written = 0
        for year, chunk in frame.groupby(frame[DATE].dt.year):
            path = self._path(int(year))
            if path.is_file():
                existing = pd.read_parquet(path, engine="pyarrow")
                existing[DATE] = pd.to_datetime(existing[DATE]).dt.normalize()
                # `chunk` last so a re-fetch supersedes a stale earlier row.
                combined = pd.concat([existing, chunk], ignore_index=True)
            else:
                combined = chunk
            combined = (
                combined.drop_duplicates(subset=self.key_columns, keep="last")
                .sort_values(self.key_columns)
                .reset_index(drop=True)
            )
            _atomic_write_parquet(combined, path)
            written += len(chunk)
        return written

    def read(
        self,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        symbols: Optional[Iterable[str]] = None,
        symbol_column: str = SYMBOL,
    ) -> pd.DataFrame:
        years = self.years()
        if not years:
            return pd.DataFrame()
        if start is not None:
            years = [y for y in years if y >= start.year]
        if end is not None:
            years = [y for y in years if y <= end.year]
        frames: List[pd.DataFrame] = []
        wanted = (
            {normalise_symbol(s) for s in symbols} if symbols is not None else None
        )
        for year in years:
            path = self._path(year)
            if not path.is_file():
                continue
            chunk = pd.read_parquet(path, engine="pyarrow")
            if chunk.empty:
                continue
            chunk[DATE] = pd.to_datetime(chunk[DATE]).dt.normalize()
            if wanted is not None and symbol_column in chunk.columns:
                chunk = chunk[chunk[symbol_column].isin(wanted)]
            frames.append(chunk)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if start is not None:
            out = out[out[DATE] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out[DATE] <= pd.Timestamp(end)]
        return out.sort_values(self.key_columns).reset_index(drop=True)

    def max_date(self) -> Optional[dt.date]:
        years = self.years()
        if not years:
            return None
        path = self._path(years[-1])
        if not path.is_file():
            return None
        chunk = pd.read_parquet(path, columns=[DATE], engine="pyarrow")
        if chunk.empty:
            return None
        return pd.to_datetime(chunk[DATE]).max().date()

    def distinct_dates(self) -> List[dt.date]:
        out: List[dt.date] = []
        for year in self.years():
            path = self._path(year)
            if not path.is_file():
                continue
            chunk = pd.read_parquet(path, columns=[DATE], engine="pyarrow")
            if chunk.empty:
                continue
            out.extend(pd.to_datetime(chunk[DATE]).dt.normalize().dt.date.unique().tolist())
        return sorted(set(out))


class DataStore:
    """Everything the engine has persisted, addressed by feed."""

    def __init__(self, curated_dir: Path, snapshot_dir: Path) -> None:
        self.curated = Path(curated_dir)
        self.snapshots = Path(snapshot_dir)
        self.curated.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)

        self.prices = _PartitionedTable(self.curated, "prices", [SYMBOL, DATE])
        self.indices = _PartitionedTable(self.curated, "indices", ["index_name", DATE])
        self.delivery = _PartitionedTable(self.curated, "delivery", [SYMBOL, DATE])
        self.open_interest = _PartitionedTable(self.curated, "open_interest", [SYMBOL, DATE])

        self._state_path = self.curated / _STATE_FILE

    # =====================================================================
    # prices
    # =====================================================================
    def write_prices(self, df: pd.DataFrame) -> int:
        return self.prices.write(df)

    def read_prices(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        out = self.prices.read(start=start, end=end, symbols=symbols)
        return out if not out.empty else empty_ohlcv()

    # =====================================================================
    # indices
    # =====================================================================
    def write_indices(self, df: pd.DataFrame) -> int:
        return self.indices.write(df)

    def read_indices(
        self,
        names: Optional[Iterable[str]] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        out = self.indices.read(start=start, end=end)
        if out.empty:
            return empty_index_frame()
        if names is not None:
            wanted = {str(n).strip() for n in names}
            out = out[out["index_name"].isin(wanted)]
        return out.reset_index(drop=True)

    def index_series(
        self,
        index_name: str,
        field: str = "close",
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.Series:
        """A single index's field as a date-indexed Series."""
        frame = self.read_indices(names=[index_name], start=start, end=end)
        if frame.empty:
            return pd.Series(dtype="float64", name=index_name)
        frame = frame.sort_values(DATE)
        series = pd.Series(
            frame[field].to_numpy(),
            index=pd.DatetimeIndex(frame[DATE]),
            name=index_name,
            dtype="float64",
        )
        return series[~series.index.duplicated(keep="last")]

    def available_index_names(self) -> List[str]:
        frame = self.indices.read()
        if frame.empty:
            return []
        return sorted(frame["index_name"].dropna().unique().tolist())

    # =====================================================================
    # delivery / open interest
    # =====================================================================
    def write_delivery(self, df: pd.DataFrame) -> int:
        return self.delivery.write(df)

    def read_delivery(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        return self.delivery.read(start=start, end=end, symbols=symbols)

    def write_open_interest(self, df: pd.DataFrame) -> int:
        return self.open_interest.write(df)

    def read_open_interest(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        return self.open_interest.read(start=start, end=end, symbols=symbols)

    # =====================================================================
    # flat reference tables
    # =====================================================================
    def _flat_path(self, name: str) -> Path:
        return self.curated / f"{name}.parquet"

    def write_table(self, name: str, df: pd.DataFrame, key_columns: Sequence[str]) -> int:
        if df is None or df.empty:
            return 0
        path = self._flat_path(name)
        if path.is_file():
            existing = pd.read_parquet(path, engine="pyarrow")
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df.copy()
        keys = [c for c in key_columns if c in combined.columns]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last")
            combined = combined.sort_values(keys)
        _atomic_write_parquet(combined.reset_index(drop=True), path)
        return len(df)

    def read_table(self, name: str) -> pd.DataFrame:
        path = self._flat_path(name)
        if not path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(path, engine="pyarrow")

    def replace_table(self, name: str, df: pd.DataFrame) -> int:
        _atomic_write_parquet(df.reset_index(drop=True), self._flat_path(name))
        return len(df)

    # convenience wrappers -------------------------------------------------
    def write_equity_master(self, df: pd.DataFrame) -> int:
        return self.replace_table("equity_master", df)

    def read_equity_master(self) -> pd.DataFrame:
        return self.read_table("equity_master")

    def write_corporate_actions(self, df: pd.DataFrame) -> int:
        return self.write_table("corporate_actions", df, [SYMBOL, "ex_date", "action_type"])

    def read_corporate_actions(self) -> pd.DataFrame:
        out = self.read_table("corporate_actions")
        if out.empty:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
        out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.normalize()
        return out

    def write_earnings_calendar(self, df: pd.DataFrame) -> int:
        return self.write_table("earnings_calendar", df, [SYMBOL, "earnings_date"])

    def read_earnings_calendar(self) -> pd.DataFrame:
        out = self.read_table("earnings_calendar")
        if out.empty:
            return pd.DataFrame(columns=[SYMBOL, "earnings_date", "confirmed", "source"])
        out["earnings_date"] = pd.to_datetime(out["earnings_date"]).dt.normalize()
        return out

    def write_pledging(self, df: pd.DataFrame) -> int:
        return self.write_table("pledging", df, [SYMBOL, "as_of_date"])

    def read_pledging(self) -> pd.DataFrame:
        return self.read_table("pledging")

    def write_fundamentals(self, df: pd.DataFrame) -> int:
        return self.write_table("fundamentals", df, [SYMBOL, "filing_date"])

    def read_fundamentals(self) -> pd.DataFrame:
        return self.read_table("fundamentals")

    def write_regulatory_events(self, df: pd.DataFrame) -> int:
        return self.write_table("regulatory_events", df, [SYMBOL, "event_date"])

    def read_regulatory_events(self) -> pd.DataFrame:
        return self.read_table("regulatory_events")

    # =====================================================================
    # universe snapshots (point-in-time membership)
    # =====================================================================
    def _universe_dir(self, index_name: str) -> Path:
        safe = index_name.replace(" ", "_").replace("/", "-").upper()
        return self.snapshots / "universe" / safe

    def write_universe_snapshot(
        self, index_name: str, as_of: dt.date, df: pd.DataFrame
    ) -> Path:
        """Persist a DATED membership snapshot.

        NSE publishes only today's constituent list. Snapshotting it on every
        run is how genuine point-in-time membership accumulates going forward.
        It cannot retroactively fix history, and the manifest says so rather
        than pretending otherwise.
        """
        out_dir = self._universe_dir(index_name)
        path = out_dir / f"{as_of.isoformat()}.parquet"
        frame = df.copy()
        frame["snapshot_date"] = pd.Timestamp(as_of)
        frame["index_name"] = index_name
        _atomic_write_parquet(frame.reset_index(drop=True), path)
        return path

    def universe_snapshot_dates(self, index_name: str) -> List[dt.date]:
        out_dir = self._universe_dir(index_name)
        if not out_dir.is_dir():
            return []
        dates: List[dt.date] = []
        for p in out_dir.glob("*.parquet"):
            try:
                dates.append(dt.date.fromisoformat(p.stem))
            except ValueError:
                continue
        return sorted(dates)

    def read_universe_snapshot(
        self, index_name: str, as_of: dt.date
    ) -> Optional[pd.DataFrame]:
        path = self._universe_dir(index_name) / f"{as_of.isoformat()}.parquet"
        if not path.is_file():
            return None
        return pd.read_parquet(path, engine="pyarrow")

    # =====================================================================
    # calendar ground truth
    # =====================================================================
    def known_sessions(self) -> List[dt.date]:
        """Sessions NSE actually published an index file for.

        This is the authoritative trading calendar -- derived from data, never
        from a hardcoded holiday table.
        """
        return self.indices.distinct_dates()

    def price_sessions(self) -> List[dt.date]:
        return self.prices.distinct_dates()

    # =====================================================================
    # feed state
    # =====================================================================
    def load_state(self) -> Dict[str, Any]:
        if not self._state_path.is_file():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("store state file unreadable; starting fresh")
            return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        _atomic_write_json(state, self._state_path)

    def update_feed_state(
        self,
        feed: str,
        last_timestamp: Optional[dt.date],
        source: Optional[str],
        row_count: int = 0,
        note: Optional[str] = None,
    ) -> None:
        state = self.load_state()
        feeds = state.setdefault("feeds", {})
        feeds[feed] = {
            "last_timestamp": last_timestamp.isoformat() if last_timestamp else None,
            "source": source,
            "row_count": row_count,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "note": note,
        }
        self.save_state(state)

    def feed_state(self, feed: str) -> Dict[str, Any]:
        return self.load_state().get("feeds", {}).get(feed, {})

    # =====================================================================
    # maintenance
    # =====================================================================
    def summary(self) -> Dict[str, Any]:
        def _rng(table: _PartitionedTable) -> Dict[str, Any]:
            dates = table.distinct_dates()
            return {
                "sessions": len(dates),
                "first": dates[0].isoformat() if dates else None,
                "last": dates[-1].isoformat() if dates else None,
            }

        prices = self.prices.read()
        return {
            "prices": {
                **_rng(self.prices),
                "rows": len(prices),
                "symbols": int(prices[SYMBOL].nunique()) if not prices.empty else 0,
            },
            "indices": {
                **_rng(self.indices),
                "names": len(self.available_index_names()),
            },
            "delivery": _rng(self.delivery),
            "open_interest": _rng(self.open_interest),
            "equity_master_rows": len(self.read_equity_master()),
            "corporate_actions_rows": len(self.read_corporate_actions()),
            "earnings_rows": len(self.read_earnings_calendar()),
            "pledging_rows": len(self.read_pledging()),
            "fundamentals_rows": len(self.read_fundamentals()),
        }

    def wipe(self) -> None:  # pragma: no cover - destructive maintenance helper
        """Delete every curated artefact. Raw payload cache is left untouched."""
        if self.curated.is_dir():
            shutil.rmtree(self.curated)
        if (self.snapshots / "universe").is_dir():
            shutil.rmtree(self.snapshots / "universe")
        self.curated.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def validate_no_duplicates(self) -> None:
        """Belt-and-braces integrity assertion, callable from the CLI."""
        prices = self.prices.read()
        if prices.empty:
            return
        dupes = int(prices.duplicated(subset=[SYMBOL, DATE]).sum())
        if dupes:
            raise DataError(
                f"price store contains {dupes} duplicate (symbol, date) rows -- "
                f"this corrupts volume sums and cross-sectional ranks"
            )
