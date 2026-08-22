"""Canonical column names and frame contracts for the data layer.

Providers return frames using these column names and dtypes regardless of
upstream naming, so downstream code contains no provider-specific column names
and swapping a vendor is a provider change rather than a pipeline change.

Two frame shapes:

* tidy -- long format, one row per ``(date, symbol)``; the storage format,
  partitionable and cheap to append.
* wide -- a ``date x symbol`` matrix of one field; the compute format for
  ranking, correlations and breadth.

:func:`to_wide` / :func:`from_wide` convert between them.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.errors import DataError

__all__ = [
    "DATE",
    "SYMBOL",
    "OHLCV_COLUMNS",
    "OHLCV_REQUIRED",
    "OHLCV_DTYPES",
    "INDEX_COLUMNS",
    "CORPORATE_ACTION_COLUMNS",
    "empty_ohlcv",
    "empty_index_frame",
    "validate_ohlcv",
    "coerce_ohlcv",
    "to_wide",
    "from_wide",
    "normalise_symbol",
    "session_dates",
]

DATE = "date"
SYMBOL = "symbol"

#: The canonical equity OHLCV frame.
OHLCV_COLUMNS: List[str] = [
    DATE,          # datetime64[ns], midnight, tz-naive IST trade date
    SYMBOL,        # NSE ticker, upper-case, no exchange suffix
    "series",      # EQ / BE / BZ ...
    "isin",
    "open",
    "high",
    "low",
    "close",       # unadjusted close as published
    "prev_close",
    "last",
    "vwap",        # turnover / volume -- an honest daily proxy, not tick VWAP
    "volume",      # shares
    "turnover",    # rupees
    "trades",
    "deliv_qty",
    "deliv_pct",
    "adj_factor",  # cumulative corporate-action factor; 1.0 until applied
    "source",
]

#: Columns without which the frame is unusable.
OHLCV_REQUIRED: List[str] = [DATE, SYMBOL, "open", "high", "low", "close", "volume"]

OHLCV_DTYPES: Dict[str, str] = {
    "series": "object",
    "isin": "object",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "prev_close": "float64",
    "last": "float64",
    "vwap": "float64",
    "volume": "float64",
    "turnover": "float64",
    "trades": "float64",
    "deliv_qty": "float64",
    "deliv_pct": "float64",
    "adj_factor": "float64",
    "source": "object",
}

#: The canonical index frame (Nifty 50/200/sector indices and India VIX).
INDEX_COLUMNS: List[str] = [
    DATE,
    "index_name",
    "open",
    "high",
    "low",
    "close",
    "points_change",
    "pct_change",
    "volume",
    "turnover",
    "pe",
    "pb",
    "div_yield",
    "source",
]

#: Corporate actions, timestamped to EX-DATE (never to announcement date).
CORPORATE_ACTION_COLUMNS: List[str] = [
    SYMBOL,
    "ex_date",
    "action_type",   # split | bonus | rights | dividend | other
    "ratio",         # multiplicative price factor, e.g. 0.5 for a 1:1 bonus
    "raw_details",
    "source",
]


# =============================================================================
# constructors
# =============================================================================


def empty_ohlcv() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=OHLCV_DTYPES.get(c, "object")) for c in OHLCV_COLUMNS})
    df[DATE] = pd.Series(dtype="datetime64[ns]")
    return df[OHLCV_COLUMNS]


def empty_index_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in INDEX_COLUMNS})
    df[DATE] = pd.Series(dtype="datetime64[ns]")
    df["index_name"] = pd.Series(dtype="object")
    df["source"] = pd.Series(dtype="object")
    return df[INDEX_COLUMNS]


# =============================================================================
# symbols
# =============================================================================


def normalise_symbol(sym: object) -> str:
    """Upper-case, strip whitespace, drop a Yahoo-style exchange suffix."""
    s = str(sym).strip().upper()
    for suffix in (".NS", ".BO", ".NSE", ".BSE"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


# =============================================================================
# coercion & validation
# =============================================================================


def coerce_ohlcv(df: pd.DataFrame, source: Optional[str] = None) -> pd.DataFrame:
    """Force an arbitrary frame into the canonical OHLCV contract.

    Missing optional columns are added as NaN -- never as zero. A zero close or
    a zero volume is a meaningful market fact; conflating it with "we don't
    know" is precisely the kind of quiet corruption Stage 1 exists to catch.
    """
    out = df.copy()

    for col in OHLCV_COLUMNS:
        if col not in out.columns:
            if col == "adj_factor":
                out[col] = 1.0
            elif col == "source":
                out[col] = source
            elif col in (DATE, SYMBOL, "series", "isin"):
                out[col] = pd.NA
            else:
                out[col] = np.nan

    out[DATE] = pd.to_datetime(out[DATE], errors="coerce").dt.normalize()
    out[SYMBOL] = out[SYMBOL].map(normalise_symbol)
    out["series"] = out["series"].astype("object")
    out["isin"] = out["isin"].astype("object")
    if source is not None:
        out["source"] = out["source"].fillna(source)

    for col, dtype in OHLCV_DTYPES.items():
        if dtype == "float64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out = out[OHLCV_COLUMNS]
    out = out.dropna(subset=[DATE, SYMBOL])
    out = out.sort_values([SYMBOL, DATE]).reset_index(drop=True)
    return out


def validate_ohlcv(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Assert the structural invariants an OHLCV frame must satisfy.

    Raises rather than warns: a frame that violates these is not something the
    engine should try to reason around.
    """
    where = f" [{context}]" if context else ""

    missing = [c for c in OHLCV_REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"OHLCV frame missing required columns{where}: {missing}")

    if df.empty:
        return df

    if df[DATE].isna().any():
        raise DataError(f"OHLCV frame has null dates{where}")
    if df[SYMBOL].isna().any():
        raise DataError(f"OHLCV frame has null symbols{where}")

    dupes = df.duplicated(subset=[DATE, SYMBOL]).sum()
    if dupes:
        raise DataError(
            f"OHLCV frame has {dupes} duplicate (date, symbol) rows{where}. "
            f"Duplicates silently double-count volume and break every "
            f"cross-sectional rank."
        )

    # High/low must actually bracket open/close on rows where all four exist.
    px = df[["open", "high", "low", "close"]]
    complete = px.notna().all(axis=1)
    if complete.any():
        sub = px[complete]
        bad_hl = (sub["high"] < sub["low"]).sum()
        if bad_hl:
            raise DataError(f"{bad_hl} rows have high < low{where}")
        bad_bracket = (
            (sub["high"] < sub[["open", "close"]].max(axis=1))
            | (sub["low"] > sub[["open", "close"]].min(axis=1))
        ).sum()
        if bad_bracket:
            raise DataError(
                f"{bad_bracket} rows have an open/close outside the high/low "
                f"range{where}"
            )

    neg = (df[["open", "high", "low", "close"]] < 0).any().any()
    if neg:
        raise DataError(f"OHLCV frame contains negative prices{where}")

    if (df["volume"].dropna() < 0).any():
        raise DataError(f"OHLCV frame contains negative volume{where}")

    return df


# =============================================================================
# reshaping
# =============================================================================


def to_wide(
    df: pd.DataFrame,
    field: str = "close",
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Pivot a tidy frame into a ``date x symbol`` matrix of one field.

    No forward-filling happens here, ever. A gap stays a NaN so that Stage 1's
    continuity check can see it. Forward-filling across non-trading days is an
    explicit leakage source in the research program's section 7 checklist.
    """
    if field not in df.columns:
        raise DataError(f"cannot pivot on missing field {field!r}")
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=DATE))
    wide = df.pivot_table(index=DATE, columns=SYMBOL, values=field,
                          aggfunc="last", observed=True)
    wide = wide.sort_index()
    if symbols is not None:
        wanted = [normalise_symbol(s) for s in symbols]
        wide = wide.reindex(columns=wanted)
    wide.index.name = DATE
    wide.columns.name = SYMBOL
    return wide


def from_wide(wide: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Inverse of :func:`to_wide`, dropping NaN cells."""
    try:
        stacked = wide.stack(future_stack=True).dropna()
    except TypeError:  # pandas < 2.1
        stacked = wide.stack(dropna=True)
    out = stacked.reset_index()
    out.columns = [DATE, SYMBOL, field]
    return out


def session_dates(df: pd.DataFrame) -> List[dt.date]:
    """Distinct sorted trade dates present in a tidy frame."""
    if df.empty:
        return []
    return sorted({d.date() for d in pd.to_datetime(df[DATE]).dt.normalize().unique()})


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{context}: missing columns {missing}", present=list(df.columns))
