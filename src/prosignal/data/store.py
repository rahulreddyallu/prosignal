"""On-disk point-in-time data store (parquet + JSON state).

Layout under ``data/``::

    curated/prices/year=2025.parquet        tidy OHLCV, one file per year
    curated/indices/year=2025.parquet       tidy index OHLC incl. India VIX
    curated/delivery/year=2025.parquet      delivery qty / %
    curated/equity_master.parquet           symbol -> listing date, ISIN
    curated/corporate_actions.parquet       ex-date-stamped adjustment ratios
    curated/earnings_calendar.parquet       scheduled results dates
    snapshots/universe/<INDEX>/<date>.parquet   dated membership snapshots
    curated/_state.json                     per-feed last-update bookkeeping

Appends are idempotent: dedup happens on write, since duplicate
``(date, symbol)`` rows would double-count volume and corrupt cross-sectional
ranks.

Writes are atomic via ``.tmp`` + ``os.replace``, so an interrupted ingest
cannot leave a half-written parquet that still parses.

Gaps are never forward-filled. Filling them is a leakage source, and would
defeat Stage 1's continuity check.
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .schema import SCHEMAS, validate_feed
from ..core.errors import DataError
from ..core.errors import IntegrityError
from ..core.logging import get_logger
from ..core.memory import release_memory
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

    #: Repeated low-cardinality strings. Stored as pandas `object` these dominate
    #: memory: on a real year-file `source` has ONE unique value and still cost
    #: 180 MB, and the four string columns together were 681 MB of a 972 MB
    #: frame. As `category` they cost almost nothing.
    _CATEGORICAL = (SYMBOL, "series", "isin", "source", "index_name")

    @staticmethod
    def _wants_category(series: pd.Series) -> bool:
        """True for a string column that is not yet categorical.

        The test used to be ``dtype == object``, which was correct only while
        pandas returned parquet strings as object arrays. Pandas 3 returns them
        as the dedicated string dtype, so that test silently stopped matching:
        no column was ever converted, the 681 MB of string columns this class
        exists to compress came back, and -- worse than the memory -- the cached
        slice stopped agreeing with a fresh read about categories, which is what
        `groupby(observed=False)` iterates. A dtype check that fails OPEN like
        that is the dangerous kind: nothing raises, the numbers just drift.
        Matching on "is it a string" instead of "is it object" holds under both
        pandas 2 and 3.
        """
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return False
        return dtype == object or pd.api.types.is_string_dtype(dtype)

    def read(
        self,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        symbols: Optional[Iterable[str]] = None,
        symbol_column: str = SYMBOL,
        columns: Optional[Sequence[str]] = None,
        predicates: Optional[Sequence[tuple]] = None,
    ) -> pd.DataFrame:
        """Read a slice of the table.

        An unfiltered read of the price table measured 972 MB and exceeded a
        512 MB host, so three filters apply:

        * ``columns`` projects at the parquet level, so unread columns are
          never materialised;
        * ``symbols`` and ``predicates`` are pushed into the reader as
          row-group filters rather than loaded and discarded;
        * string columns return as ``category``.

        Measured together, these took a year-file from 182 MB to 1 MB. Floats
        stay float64: downcasting gained a further 14 percentage points and
        would raise a precision question over every price, stop and target.
        """
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
        # Push the symbol filter into the reader so non-matching row groups are
        # never decoded. Falls back to a post-read filter if the engine cannot
        # apply it (older pyarrow, or a column absent from this table).
        pq_filters: Optional[List[tuple]] = None
        if wanted is not None:
            pq_filters = [(symbol_column, "in", sorted(wanted))]
        if predicates:
            pq_filters = (pq_filters or []) + list(predicates)
        for year in years:
            path = self._path(year)
            if not path.is_file():
                continue
            try:
                chunk = pd.read_parquet(
                    path, engine="pyarrow", columns=list(columns) if columns else None,
                    filters=pq_filters,
                )
            except (ValueError, TypeError, KeyError):
                chunk = pd.read_parquet(path, engine="pyarrow")
                if wanted is not None and symbol_column in chunk.columns:
                    chunk = chunk[chunk[symbol_column].isin(wanted)]
            if chunk.empty:
                continue
            chunk[DATE] = pd.to_datetime(chunk[DATE]).dt.normalize()
            if wanted is not None and symbol_column in chunk.columns:
                chunk = chunk[chunk[symbol_column].isin(wanted)]
            for col in self._CATEGORICAL:
                if col in chunk.columns and self._wants_category(chunk[col]):
                    chunk[col] = chunk[col].astype("category")
            frames.append(chunk)
            # Trim between year-files. Measured: decoding five year-files to a
            # 33 MB result peaked at 291 MB, because each file's transient
            # decode buffers stayed in the allocator arena rather than being
            # reused. Live data was never the problem; retained arena was.
            # Trimming here bounds peak at roughly base + one file.
            if len(years) > 1:
                release_memory()
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        # Concatenating categoricals with different category sets yields object
        # again, which silently undoes the saving. Re-apply once.
        for col in self._CATEGORICAL:
            if col in out.columns and self._wants_category(out[col]):
                out[col] = out[col].astype("category")
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
        """Session dates held by this table.

        Uniqueness is resolved inside Arrow. Reading the column into pandas and
        calling ``.dt.date`` first would allocate one Python date object per row
        -- tens of millions of them across the price years -- to arrive at a list
        of roughly a thousand.
        """
        out: set = set()
        for year in self.years():
            path = self._path(year)
            if not path.is_file():
                continue
            table = pq.read_table(path, columns=[DATE])
            if table.num_rows == 0:
                continue
            uniq = pc.unique(table.column(DATE).combine_chunks())
            out.update(pd.to_datetime(uniq.to_pandas()).dt.normalize().dt.date)
            del table, uniq
        return sorted(out)


def _validate(frame: "pd.DataFrame", schema_key: str, where: str) -> None:
    """Gate every curated write on the feed's declared shape.

    Placed at the store rather than at each provider so a new ingest path
    cannot reach the curated files without passing the same checks. An empty
    frame is allowed through: writers already treat it as a no-op, and
    rejecting it here would turn a quiet day into an integrity failure.
    """
    if frame is None or len(frame) == 0:
        return
    schema = SCHEMAS.get(schema_key)
    if schema is None:
        return
    validate_feed(frame, schema, context=where)


class DataStore:
    """Everything the engine has persisted, addressed by feed."""

    def __init__(
        self,
        curated_dir: Path,
        snapshot_dir: Path,
        equity_series: Sequence[str] = ("EQ",),
        adjust_prices: bool = True,
    ) -> None:
        self.equity_series = tuple(equity_series)
        self.adjust_prices = bool(adjust_prices)
        self.curated = Path(curated_dir)
        self.snapshots = Path(snapshot_dir)
        self.curated.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)

        self._price_cache: Optional[Dict[str, Any]] = None
        self.prices = _PartitionedTable(self.curated, "prices", [SYMBOL, DATE])
        self.indices = _PartitionedTable(self.curated, "indices", ["index_name", DATE])
        self.delivery = _PartitionedTable(self.curated, "delivery", [SYMBOL, DATE])
        self._state_path = self.curated / _STATE_FILE

    # =====================================================================
    # prices
    # =====================================================================
    def write_prices(self, df: pd.DataFrame) -> int:
        _validate(df, "prices", "write_prices")
        return self.prices.write(df)

    #: What the stages actually consume. The price table has 18 columns; no
    #: stage reads more than these, and projecting them cut a year-file from
    #: 33 MB to 17 MB in measurement.
    #: VWAP IS A PRICE COLUMN AND IT IS SERVED. Every fill in this engine's
    #: research -- and therefore every cost and every excess return it reports
    #: -- is the VWAP of the session after the signal, which is the manual
    #: next-session execution the product asks of its user. This list omitted
    #: `vwap`, so the store held it (100% coverage in the parquet) and served
    #: it to nobody: `build_v3_panel` and `build_v2_panel` both fall back to
    #: open, then close, when vwap is absent, and would have quietly measured a
    #: different execution than the one the holdouts were computed on. It also
    #: left `price_vs_vwap_20` -- one of the four reversal factors -- NaN on
    #: every production row.
    PRICE_COLUMNS = [
        DATE, SYMBOL, "series", "open", "high", "low", "close", "vwap",
        "volume", "turnover", "deliv_pct",
    ]

    def prefetch_prices(
        self,
        symbols: Iterable[str],
        start: dt.date,
        end: dt.date,
    ) -> int:
        """Read one window that later reads can be served from.

        Stages ask for overlapping windows of the same table: measured on a warm
        run, seven calls decoded 255,270 rows to produce 15.4 MB of frames, and
        the widest window contained almost all of the others. Reading that
        window once and slicing it removes the repeated decode and the allocator
        churn behind it.

        Correctness is the whole risk here, so the cache is deliberately
        conservative: it serves a later read only when the symbols, the dates
        and the columns are all contained in what was fetched. Anything else
        falls through to a real read.
        """
        frame = self.read_prices(symbols=symbols, start=start, end=end)
        self._price_cache = None if frame.empty else {
            "frame": frame,
            "symbols": {normalise_symbol(s) for s in symbols},
            "start": start,
            "end": end,
            "columns": set(frame.columns),
        }
        return 0 if frame.empty else len(frame)

    def clear_price_cache(self) -> None:
        self._price_cache = None

    def _serve_from_cache(
        self,
        symbols: Optional[Iterable[str]],
        start: Optional[dt.date],
        end: Optional[dt.date],
        columns: Optional[Sequence[str]],
    ) -> Optional[pd.DataFrame]:
        cache = getattr(self, "_price_cache", None)
        if cache is None:
            return None
        # An unbounded request cannot be satisfied from a bounded window.
        if symbols is None or start is None or end is None:
            return None
        wanted = {normalise_symbol(s) for s in symbols}
        if not wanted <= cache["symbols"]:
            return None
        if start < cache["start"] or end > cache["end"]:
            return None
        cols = list(columns) if columns else self.PRICE_COLUMNS
        if not set(cols) <= cache["columns"]:
            return None
        frame = cache["frame"]
        dates = pd.to_datetime(frame[DATE])
        mask = (
            frame[SYMBOL].astype(str).isin(wanted)
            & (dates >= pd.Timestamp(start))
            & (dates <= pd.Timestamp(end))
        )
        out = frame.loc[mask, cols].reset_index(drop=True)
        # Drop categories the slice no longer contains. A fresh read would only
        # have the categories present in its own rows, and groupby with
        # observed=False iterates categories rather than values -- so a slice
        # carrying the full universe would behave differently from the read it
        # replaces.
        for col in out.columns:
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].cat.remove_unused_categories()
        return out

    def read_prices(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """OHLCV for the cash equity series only.

        NSE publishes debt instruments under the issuer's own ticker: NTPC has
        an EQ line near Rs 358 and debenture lines (ND, N7, ...) between Rs 5 and
        Rs 1,223. Bhavcopy carries them all, so an unfiltered read produces a
        close series that jumps between the equity and a bond and back. Filtering
        here rather than at each call site means no stage can accidentally price
        an equity off its issuer's debt.
        """
        cached = self._serve_from_cache(symbols, start, end, columns)
        if cached is not None:
            return cached if not cached.empty else empty_ohlcv()

        want = list(columns) if columns else self.PRICE_COLUMNS
        want = self._adjusted_columns(want)
        predicates = (
            [("series", "in", list(self.equity_series))] if self.equity_series else None
        )
        out = self.prices.read(
            start=start, end=end, symbols=symbols, columns=want, predicates=predicates,
        )
        if out.empty:
            return empty_ohlcv()
        out = self._apply_corporate_actions(out, columns)
        return out if not out.empty else empty_ohlcv()

    def _adjusted_columns(self, want: List[str]) -> List[str]:
        """Adjustment needs date and symbol even when the caller did not ask."""
        if not self.adjust_prices:
            return want
        need = list(want)
        for col in (DATE, SYMBOL):
            if col not in need:
                need.append(col)
        return need

    def _apply_corporate_actions(
        self, frame: pd.DataFrame, requested: Optional[Sequence[str]]
    ) -> pd.DataFrame:
        """Rewrite prices so returns across a split are economically real.

        Splits and bonuses are share-count changes, not returns. Left raw, a 1:10
        split reads as a -90% session: measured on this store, 80 of 100
        split/bonus events inside the window showed a drop beyond 30%, affecting
        72 of the 200 index names -- NESTLEIND, EICHERMOT, BAJFINANCE, TATASTEEL,
        KOTAKBANK among them. Any momentum or volatility computed across one of
        those dates was describing an accounting event.

        `apply_adjustments` existed for this and was never called on the read
        path, so every consumer saw unadjusted prices.

        This is not lookahead. It removes an artifact rather than adding
        information: the adjusted series is what the holder actually
        experienced, and every index and vendor reports it this way.
        """
        # A REQUEST-SHAPE CHECK, BEFORE ANYTHING ELSE -- including the
        # adjust_prices switch. It used to sit below that early return, so
        # `DataStore(adjust_prices=False)` served the 1.0 placeholder silently
        # for a caller that asked for adj_factor alone, which is the same
        # silent-placeholder failure with one more way in. `adj_factor` is
        # COMPUTED here by `apply_adjustments` from the action table; the column
        # sitting in the parquet is a write-time placeholder. Ask for it without
        # a price column and `price_cols` below is empty, this function returns
        # early, and the caller receives the placeholder -- 1.0 everywhere --
        # with nothing to distinguish it from a genuine "no actions" answer.
        #
        # This is not hypothetical. It is how the corporate-action repair was
        # first measured as having changed nothing: the verification read
        # ["date", "symbol", "adj_factor"], got 1.0 for every cell, and reported
        # zero affected names. The real figure was 4,905 (date, symbol) cells
        # recovered across 165 symbols. A silent placeholder that reads exactly
        # like a valid answer is worse than no answer, so it raises.
        if requested is not None:
            req = set(requested)
            if "adj_factor" in req and not (req & {"open", "high", "low", "close",
                                                    "vwap"}):
                raise IntegrityError(
                    "adj_factor was requested without a price column. It is "
                    "computed during adjustment, not stored; reading it alone "
                    "returns a placeholder of 1.0 that is indistinguishable "
                    "from 'no corporate actions'. Add a price column (e.g. "
                    "'close') to the read.",
                    columns=sorted(req),
                )
        if not self.adjust_prices:
            return frame
        try:
            actions = self.read_corporate_actions()
            if actions is None or actions.empty:
                return frame
            from .corporate_actions import apply_adjustments

            # vwap is adjusted with the rest. An unadjusted vwap beside an
            # adjusted close makes `price_vs_vwap_20` read a 1:10 split as a
            # -90% dislocation, and makes a VWAP fill price a post-split share
            # at its pre-split price.
            price_cols = [c for c in ("open", "high", "low", "close", "vwap")
                          if c in frame.columns]
            if not price_cols:
                return frame
            adjusted = apply_adjustments(frame, actions, price_columns=price_cols)
            if requested is not None:
                keep = [c for c in requested if c in adjusted.columns]
                adjusted = adjusted[keep]
            elif "adj_factor" in adjusted.columns:
                adjusted = adjusted.drop(columns=["adj_factor"])
            return adjusted
        except Exception as exc:
            # Unadjusted prices read a 1:10 split as a -90% session and corrupt
            # momentum, volatility, drawdown and beta for that name across every
            # window spanning the ex-date -- measured, 72 of 200 index symbols.
            # Returning them quietly produced a normal-looking watchlist with
            # nothing on the card to say the data was wrong, so this now raises.
            # The caller decides; it is not decided here by omission.
            log.error("corporate-action adjustment failed", extra={"error": str(exc)})
            raise IntegrityError(
                f"corporate-action adjustment failed ({exc}). Prices are served "
                f"unadjusted only when explicitly requested via "
                f"DataStore(adjust_prices=False); a silent fallback would put a "
                f"split artefact into every return."
            ) from exc

    # =====================================================================
    # indices
    # =====================================================================
    def write_indices(self, df: pd.DataFrame) -> int:
        _validate(df, "indices", "write_indices")
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

    def resolve_index_name(self, name: str) -> Optional[str]:
        """Match an index name case- and whitespace-insensitively.

        NSE publishes ``Nifty 200`` in ``ind_close_all`` while the config
        writes ``NIFTY 200``. An exact-match lookup returned an empty series
        and halted Stage 2 the first time it ran against real data.

        Resolving here means a change in NSE's capitalisation cannot break
        callers, while a genuinely absent index still returns ``None``.
        """
        target = " ".join(str(name).strip().casefold().split())
        for candidate in self.available_index_names():
            if " ".join(candidate.strip().casefold().split()) == target:
                return candidate
        return None

    def index_series(
        self,
        index_name: str,
        field: str = "close",
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.Series:
        """A single index's field as a date-indexed Series.

        The name is resolved case-insensitively; an unknown index yields an
        empty Series so the caller can decide whether that is fatal.
        """
        resolved = self.resolve_index_name(index_name) or index_name
        frame = self.read_indices(names=[resolved], start=start, end=end)
        if frame.empty:
            # An EMPTY DatetimeIndex, not the default RangeIndex. Callers slice
            # these series by date, and an integer index makes `series.index <=
            # Timestamp` raise an opaque TypeError. That turned a missing India
            # VIX feed -- which Stage 2 is designed to survive with a reduced-
            # confidence note -- into a crash.
            return pd.Series(
                dtype="float64",
                index=pd.DatetimeIndex([], name=DATE),
                name=index_name,
            )
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
        _validate(df, "delivery", "write_delivery")
        return self.delivery.write(df)

    def read_delivery(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        return self.delivery.read(start=start, end=end, symbols=symbols)



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

    def write_statements(self, df: pd.DataFrame) -> int:
        return self.replace_table("statements", df)

    def read_statements(self) -> pd.DataFrame:
        """Income statement, balance sheet and cash flow by period.

        Separate from `fundamentals`, which holds the NSE Ind-AS filings and
        their true filing dates. This table carries period end only, so the
        fundamental factor layer derives an availability date from the SEBI
        LODR deadline rather than pretending a filing date is known.
        """
        return self.read_table("statements")

    def write_sector_map(self, df: pd.DataFrame) -> int:
        return self.replace_table("sector_map", df)

    def read_sector_map(self) -> pd.DataFrame:
        """symbol -> sector, pooled from the NSE index constituent files.

        Current vintage: NSE publishes only today's membership. That is exactly
        right for a live decision and mildly forward-looking in a backtest, but
        it feeds the Stage 8 diversification cap alone and never the score, so
        it cannot manufacture predictability the way a survivorship-biased
        universe does.
        """
        return self.read_table("sector_map")

    def write_corporate_actions(self, df: pd.DataFrame) -> int:
        """Merge into the existing table, then collapse duplicate descriptions.

        write_table dedups on its key alone, and the key includes action_type,
        so NSE's "bonus" and yfinance's "split_or_bonus" for one event both
        survive and the ratio is applied twice. The collapse has to happen after
        the merge, on the combined table, not on the incoming frame.
        """
        from .corporate_actions import dedupe_actions

        _validate(df, "corporate_actions", "write_corporate_actions")
        self.write_table("corporate_actions", df, [SYMBOL, "ex_date", "action_type"])
        existing = self.read_table("corporate_actions")
        if existing.empty:
            return 0
        collapsed = dedupe_actions(existing)
        return self.replace_table("corporate_actions", collapsed)

    def read_corporate_actions(self) -> pd.DataFrame:
        out = self.read_table("corporate_actions")
        if out.empty:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
        out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.normalize()
        return out

    def write_results_calendar(self, df: pd.DataFrame) -> int:
        """Every quarterly filing NSE has a record of, with its availability date.

        Keyed on (symbol, period_end, consolidated) because NSE publishes a
        Consolidated and a Non-Consolidated row for the same quarter and both
        are real filings with their own timestamps. Collapsing them would
        silently drop one, and which one survived would depend on row order.

        Distinct from `earnings_calendar`, which holds SCHEDULED dates from
        yfinance. This one is what was actually filed and when it became public.
        """
        return self.write_table("results_calendar", df,
                                [SYMBOL, "period_end", "consolidated"])

    def read_results_calendar(self) -> pd.DataFrame:
        return self.read_table("results_calendar")

    def write_shareholding(self, df: pd.DataFrame) -> int:
        """Quarterly shareholding patterns -- promoter, public, free float.

        Keyed on (symbol, period_end): one pattern per quarter. A revised
        filing for the same quarter replaces the original, which is correct
        here and is the opposite of the rule for RESULTS -- a restated income
        statement must not backdate itself onto the date the original was
        published, but a corrected shareholding pattern supersedes a wrong one
        and both carry their own broadcast date.
        """
        return self.write_table("shareholding", df, [SYMBOL, "period_end"])

    def read_shareholding(self) -> pd.DataFrame:
        return self.read_table("shareholding")

    def write_security_list(self, df: pd.DataFrame) -> int:
        """Surveillance state per security, as DATED SNAPSHOTS.

        Keyed on (symbol, snapshot_date) rather than symbol alone, because NSE
        publishes only the current list: membership accumulates going forward
        and cannot be reconstructed backwards. Keeping every snapshot is what
        makes it point-in-time from the first one onward. Overwriting on symbol
        would leave one undated list that silently claims to describe every
        date, which is the survivorship error the universe screen already
        refuses for index membership.
        """
        return self.write_table("security_list", df, [SYMBOL, "snapshot_date"])

    def read_security_list(self) -> pd.DataFrame:
        return self.read_table("security_list")

    def write_fo_lots(self, df: pd.DataFrame) -> int:
        """F&O eligibility and lot size, as dated snapshots. See above."""
        return self.write_table("fo_lots", df, [SYMBOL, "snapshot_date"])

    def read_fo_lots(self) -> pd.DataFrame:
        return self.read_table("fo_lots")

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

        With one correction. NSE's archive occasionally serves an index file
        for a date the market never opened; this store holds two, Sunday
        2023-06-04 and Saturday 2023-11-04, each carrying 106 index rows and no
        equity prices at all. Left in, they are counted by everything measured
        in sessions -- the 63-session purge, the 63-session forward return, every
        feature lookback -- so a window nominally spanning 63 trading days spans
        62 and one day that never traded.

        A weekend date is therefore admitted only if equity prices exist for it.
        That keeps a genuine special session (NSE has held them, including a
        Saturday disaster-recovery session) while dropping the artifacts, and it
        cannot silently discard a weekday: weekdays are admitted regardless, so
        a missing-price weekday such as 2026-02-05 stays visible as a gap to be
        refetched rather than being quietly erased from the calendar.
        """
        sessions = self.indices.distinct_dates()
        if not any(d.weekday() >= 5 for d in sessions):
            return sessions
        priced = set(self.prices.distinct_dates())
        return [d for d in sessions if d.weekday() < 5 or d in priced]

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
