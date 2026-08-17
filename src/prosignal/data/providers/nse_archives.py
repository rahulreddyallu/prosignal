"""NSE static-archive provider -- the engine's PRIMARY data source.

Every endpoint below was probed against the live hosts during the build and
returns HTTP 200 without authentication:

===========================  ==================================================
Bhavcopy (UDiFF, >=2024-07)  full cash-segment OHLCV + turnover + ISIN
Bhavcopy (legacy, <2024-07)  same fields, older column names
sec_bhavdata_full            adds DELIV_QTY / DELIV_PER (delivery percentage)
ind_close_all                OHLC for EVERY NSE index in one file, incl India VIX
ind_nifty<N>list             current constituents + Industry (sector) + ISIN
EQUITY_L                     every listed symbol with DATE OF LISTING and ISIN
F&O bhavcopy                 open interest, for the Stage 5 OI context check
===========================  ==================================================

``ind_close_all`` is quietly the most valuable file here: one request per
session yields Nifty 50, Nifty 200, every sector index, *and* India VIX, which
is the entire input set for Stage 2's regime engine and for sector-relative
strength in Stage 4.

A 404 means "no session that day" (weekend, holiday, or not yet published) and
is returned as ``None``. Callers use that to *discover* the trading calendar
rather than trusting a hardcoded holiday table.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ...core.errors import ProviderError
from ...core.logging import get_logger
from ..types import (
    DATE,
    SYMBOL,
    coerce_ohlcv,
    empty_index_frame,
    normalise_symbol,
)
from .http import HttpClient

__all__ = ["NseArchivesProvider", "INDIA_VIX_NAME"]

log = get_logger(__name__)

#: The exact label India VIX carries inside ind_close_all.
INDIA_VIX_NAME = "India VIX"

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _fmt(path_template: str, day: dt.date) -> str:
    return (
        path_template.replace("{yyyymmdd}", day.strftime("%Y%m%d"))
        .replace("{ddmmyyyy}", day.strftime("%d%m%Y"))
        .replace("{yyyy}", day.strftime("%Y"))
        .replace("{MON}", _MONTHS[day.month - 1])
        .replace("{ddMONyyyy}", f"{day.day:02d}{_MONTHS[day.month - 1]}{day.year}")
    )


def _num(series: pd.Series) -> pd.Series:
    """NSE writes '-' for 'not applicable'. Treat it as missing, never as zero."""
    return pd.to_numeric(
        series.astype(str).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Several NSE CSVs ship column names with leading spaces."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_date(series: pd.Series, fmt: str) -> pd.Series:
    """Parse a fixed-format NSE date column, tolerating padded values.

    NSE pads several CSV fields with a leading space (``' 14-Aug-2026'``).
    Passing that straight to ``pd.to_datetime(format=...)`` yields NaT for
    every row, which then silently drops the entire feed -- exactly the kind
    of quiet, total data loss that looks like "the source is down" rather than
    "we parsed it wrong". Stripping first is cheap insurance.
    """
    cleaned = series.astype(str).str.strip()
    parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
    if parsed.isna().all() and cleaned.notna().any():
        # Fall back to inference rather than losing the feed outright.
        parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
    return parsed


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ProviderError("nse_archives", "zip archive contained no CSV")
        with zf.open(names[0]) as fh:
            return _strip_columns(pd.read_csv(fh, low_memory=False))


class NseArchivesProvider:
    """Fetch and normalise NSE's public archive files."""

    name = "nse_archives"

    #: Logical feed keys the storage policy can name in
    #: ``storage.raw_cache.never_cache_feeds``.
    FEED_KEYS = ("cm_bhavcopy", "fo_bhavcopy", "delivery", "index_close_all")

    def __init__(
        self,
        client: HttpClient,
        cfg: "object",
        ttl_historical_s: float,
        ttl_current_s: float,
        never_cache_feeds: Optional[Iterable[str]] = None,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.ttl_historical = ttl_historical_s
        self.ttl_current = ttl_current_s
        self.never_cache_feeds = {str(f).strip() for f in (never_cache_feeds or ())}

    def _cacheable(self, feed_key: str) -> bool:
        return feed_key not in self.never_cache_feeds

    def never_cache_url_markers(self) -> List[str]:
        """Distinctive URL fragments for feeds the policy says not to cache.

        Derived from the configured path templates rather than hardcoded, so
        renaming an endpoint in parameters.yaml cannot leave the purge sweep
        silently matching nothing.
        """
        by_feed = {
            "cm_bhavcopy": self.cfg.bhavcopy_udiff_path,
            "fo_bhavcopy": self.cfg.fo_bhavcopy_path,
            "delivery": self.cfg.sec_bhavdata_full_path,
            "index_close_all": self.cfg.index_close_all_path,
        }
        markers: List[str] = []
        for feed in self.never_cache_feeds:
            template = by_feed.get(feed)
            if not template:
                continue
            # Take the stable prefix of the filename, before the first
            # date placeholder.
            filename = template.rsplit("/", 1)[-1]
            marker = filename.split("{", 1)[0].rstrip("_")
            if marker:
                markers.append(marker)
        return markers

    # -- helpers -------------------------------------------------------------
    def _ttl(self, day: dt.date) -> float:
        """Past sessions are immutable; today's files can still be republished."""
        return self.ttl_historical if day < dt.date.today() else self.ttl_current

    def _archives_url(self, path: str) -> str:
        return f"{self.cfg.base_archives.rstrip('/')}{path}"

    def _legacy_url(self, path: str) -> str:
        return f"{self.cfg.base_legacy.rstrip('/')}{path}"

    # =========================================================================
    # Cash-segment bhavcopy
    # =========================================================================
    def fetch_bhavcopy(self, day: dt.date) -> Optional[pd.DataFrame]:
        """Canonical OHLCV for every cash-segment instrument on ``day``.

        Returns ``None`` when NSE published nothing for that date.
        """
        if day >= self.cfg.bhavcopy_udiff_from:
            df = self._fetch_bhavcopy_udiff(day)
            if df is not None:
                return df
            # Fall through: very early UDiFF dates occasionally only exist in
            # the legacy tree.
        return self._fetch_bhavcopy_legacy(day)

    def _fetch_bhavcopy_udiff(self, day: dt.date) -> Optional[pd.DataFrame]:
        url = self._archives_url(_fmt(self.cfg.bhavcopy_udiff_path, day))
        res = self.client.get(
            url,
            ttl_seconds=self._ttl(day),
            context="nse_archives.bhavcopy",
            cacheable=self._cacheable("cm_bhavcopy"),
        )
        if res is None:
            return None
        raw = _read_zip_csv(res.content)

        required = {"TckrSymb", "SctySrs", "ClsPric", "TradDt"}
        missing = required - set(raw.columns)
        if missing:
            raise ProviderError(
                self.name,
                f"UDiFF bhavcopy for {day} is missing expected columns: {sorted(missing)}",
                url=url,
            )

        # Cash equities only: exclude derivatives rows that share the format.
        if "FinInstrmTp" in raw.columns:
            raw = raw[raw["FinInstrmTp"].astype(str).str.upper().isin(["STK", "EQ", ""])]

        out = pd.DataFrame(
            {
                DATE: pd.to_datetime(raw["TradDt"], errors="coerce"),
                SYMBOL: raw["TckrSymb"].map(normalise_symbol),
                "series": raw["SctySrs"].astype(str).str.strip().str.upper(),
                "isin": raw.get("ISIN", pd.Series(index=raw.index, dtype=object)),
                "open": _num(raw["OpnPric"]),
                "high": _num(raw["HghPric"]),
                "low": _num(raw["LwPric"]),
                "close": _num(raw["ClsPric"]),
                "prev_close": _num(raw["PrvsClsgPric"]),
                "last": _num(raw["LastPric"]),
                "volume": _num(raw["TtlTradgVol"]),
                "turnover": _num(raw["TtlTrfVal"]),
                "trades": _num(raw.get("TtlNbOfTxsExctd", pd.Series(index=raw.index))),
            }
        )
        # AVG_PRICE is not in UDiFF; turnover/volume is the honest daily proxy.
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vwap"] = np.where(out["volume"] > 0, out["turnover"] / out["volume"], np.nan)
        return coerce_ohlcv(out, source=self.name)

    def _fetch_bhavcopy_legacy(self, day: dt.date) -> Optional[pd.DataFrame]:
        url = self._legacy_url(_fmt(self.cfg.bhavcopy_legacy_path, day))
        res = self.client.get(
            url,
            ttl_seconds=self._ttl(day),
            context="nse_archives.bhavcopy_legacy",
            cacheable=self._cacheable("cm_bhavcopy"),
        )
        if res is None:
            return None
        raw = _read_zip_csv(res.content)
        if "SYMBOL" not in raw.columns:
            raise ProviderError(self.name, f"legacy bhavcopy for {day} has no SYMBOL column", url=url)

        out = pd.DataFrame(
            {
                DATE: _parse_date(raw["TIMESTAMP"], "%d-%b-%Y"),
                SYMBOL: raw["SYMBOL"].map(normalise_symbol),
                "series": raw["SERIES"].astype(str).str.strip().str.upper(),
                "isin": raw.get("ISIN", pd.Series(index=raw.index, dtype=object)),
                "open": _num(raw["OPEN"]),
                "high": _num(raw["HIGH"]),
                "low": _num(raw["LOW"]),
                "close": _num(raw["CLOSE"]),
                "prev_close": _num(raw["PREVCLOSE"]),
                "last": _num(raw["LAST"]),
                "volume": _num(raw["TOTTRDQTY"]),
                "turnover": _num(raw["TOTTRDVAL"]),
                "trades": _num(raw.get("TOTALTRADES", pd.Series(index=raw.index))),
            }
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vwap"] = np.where(out["volume"] > 0, out["turnover"] / out["volume"], np.nan)
        return coerce_ohlcv(out, source=self.name)

    # =========================================================================
    # Delivery data
    # =========================================================================
    def fetch_delivery(self, day: dt.date) -> Optional[pd.DataFrame]:
        """Delivery quantity and percentage -- an India-specific move-quality input.

        Practitioner-grade at best; the engine keeps it behind an off-by-default
        switch (``stage6_entry.confirmation.require_delivery_confirmation``)
        because no academic evidence was found for it.
        """
        url = self._archives_url(_fmt(self.cfg.sec_bhavdata_full_path, day))
        res = self.client.get(
            url,
            ttl_seconds=self._ttl(day),
            context="nse_archives.delivery",
            cacheable=self._cacheable("delivery"),
        )
        if res is None:
            return None
        raw = _strip_columns(pd.read_csv(io.BytesIO(res.content), low_memory=False))
        if "SYMBOL" not in raw.columns:
            return None

        out = pd.DataFrame(
            {
                DATE: _parse_date(raw["DATE1"], "%d-%b-%Y"),
                SYMBOL: raw["SYMBOL"].map(normalise_symbol),
                "series": raw["SERIES"].astype(str).str.strip().str.upper(),
                "deliv_qty": _num(raw.get("DELIV_QTY", pd.Series(index=raw.index))),
                "deliv_pct": _num(raw.get("DELIV_PER", pd.Series(index=raw.index))),
                "avg_price": _num(raw.get("AVG_PRICE", pd.Series(index=raw.index))),
            }
        )
        out = out.dropna(subset=[DATE, SYMBOL])
        return out.reset_index(drop=True)

    # =========================================================================
    # Indices (including India VIX)
    # =========================================================================
    def fetch_index_close_all(self, day: dt.date) -> Optional[pd.DataFrame]:
        """OHLC for every NSE index on ``day``, India VIX included."""
        url = self._archives_url(_fmt(self.cfg.index_close_all_path, day))
        res = self.client.get(
            url,
            ttl_seconds=self._ttl(day),
            context="nse_archives.indices",
            cacheable=self._cacheable("index_close_all"),
        )
        if res is None:
            return None
        raw = _strip_columns(pd.read_csv(io.BytesIO(res.content), low_memory=False))
        if "Index Name" not in raw.columns:
            return None

        out = pd.DataFrame(
            {
                DATE: _parse_date(raw["Index Date"], "%d-%m-%Y"),
                "index_name": raw["Index Name"].astype(str).str.strip(),
                "open": _num(raw["Open Index Value"]),
                "high": _num(raw["High Index Value"]),
                "low": _num(raw["Low Index Value"]),
                "close": _num(raw["Closing Index Value"]),
                "points_change": _num(raw.get("Points Change", pd.Series(index=raw.index))),
                "pct_change": _num(raw.get("Change(%)", pd.Series(index=raw.index))),
                "volume": _num(raw.get("Volume", pd.Series(index=raw.index))),
                # Published in crore; store rupees so every turnover in the
                # engine shares one unit.
                "turnover": _num(raw.get("Turnover (Rs. Cr.)", pd.Series(index=raw.index))) * 1e7,
                "pe": _num(raw.get("P/E", pd.Series(index=raw.index))),
                "pb": _num(raw.get("P/B", pd.Series(index=raw.index))),
                "div_yield": _num(raw.get("Div Yield", pd.Series(index=raw.index))),
            }
        )
        out["source"] = self.name
        out = out.dropna(subset=[DATE, "index_name"])
        empty = empty_index_frame()
        return out.reindex(columns=empty.columns).reset_index(drop=True)

    # =========================================================================
    # Universe / reference data
    # =========================================================================
    def fetch_index_constituents(self, index_name: str) -> pd.DataFrame:
        """Current constituents of an index, with the Industry (sector) column.

        NSE publishes only the CURRENT list. The universe module snapshots this
        with a date so that point-in-time membership accumulates going forward;
        for any date before the earliest snapshot, membership is
        survivorship-biased and the manifest says so.
        """
        files: Dict[str, str] = dict(self.cfg.index_constituent_files)
        path = files.get(index_name)
        if path is None:
            raise ProviderError(
                self.name,
                f"no constituent file configured for index {index_name!r}. Add it "
                f"under providers.nse_archives.index_constituent_files.",
                known=sorted(files),
            )
        url = self._archives_url(path)
        res = self.client.get(
            url, ttl_seconds=self.ttl_current, allow_404=False, context="nse_archives.constituents"
        )
        if res is None:  # pragma: no cover - allow_404=False raises instead
            raise ProviderError(self.name, f"constituent file missing for {index_name}", url=url)

        raw = _strip_columns(pd.read_csv(io.BytesIO(res.content)))
        colmap = {c.lower().strip(): c for c in raw.columns}

        def pick(*candidates: str) -> Optional[str]:
            for cand in candidates:
                if cand in colmap:
                    return colmap[cand]
            return None

        sym_col = pick("symbol")
        if sym_col is None:
            raise ProviderError(self.name, f"{index_name} constituent file has no Symbol column", url=url)

        out = pd.DataFrame(
            {
                SYMBOL: raw[sym_col].map(normalise_symbol),
                "company_name": raw[pick("company name", "company") or sym_col].astype(str).str.strip(),
                "sector": (
                    raw[pick("industry", "sector")].astype(str).str.strip()
                    if pick("industry", "sector")
                    else "Unknown"
                ),
                "series": (
                    raw[pick("series")].astype(str).str.strip().str.upper()
                    if pick("series")
                    else "EQ"
                ),
                "isin": (
                    raw[pick("isin code", "isin")].astype(str).str.strip()
                    if pick("isin code", "isin")
                    else None
                ),
            }
        )
        out["index_name"] = index_name
        return out.dropna(subset=[SYMBOL]).drop_duplicates(subset=[SYMBOL]).reset_index(drop=True)

    def fetch_equity_master(self) -> pd.DataFrame:
        """Every NSE-listed symbol with its listing date -- the survivorship anchor.

        Listing dates let the engine keep a name out of the universe for
        periods before it was actually tradeable, which is the mirror image of
        the delisting problem in the research program's section 7 checklist.
        """
        url = self._archives_url(self.cfg.equity_master_path)
        res = self.client.get(
            url, ttl_seconds=self.ttl_current, allow_404=False, context="nse_archives.equity_master"
        )
        raw = _strip_columns(pd.read_csv(io.BytesIO(res.content)))
        out = pd.DataFrame(
            {
                SYMBOL: raw["SYMBOL"].map(normalise_symbol),
                "company_name": raw.get("NAME OF COMPANY", pd.Series(index=raw.index)).astype(str).str.strip(),
                "series": raw.get("SERIES", pd.Series(index=raw.index)).astype(str).str.strip().str.upper(),
                "listing_date": _parse_date(
                    raw.get("DATE OF LISTING", pd.Series(index=raw.index)), "%d-%b-%Y"
                ),
                "paid_up_value": _num(raw.get("PAID UP VALUE", pd.Series(index=raw.index))),
                "face_value": _num(raw.get("FACE VALUE", pd.Series(index=raw.index))),
                "isin": raw.get("ISIN NUMBER", pd.Series(index=raw.index)).astype(str).str.strip(),
            }
        )
        return out.dropna(subset=[SYMBOL]).drop_duplicates(subset=[SYMBOL, "series"]).reset_index(drop=True)

    # =========================================================================
    # F&O open interest
    # =========================================================================
    def fetch_fo_open_interest(self, day: dt.date) -> Optional[pd.DataFrame]:
        """Stock-futures OI aggregated across expiries, per underlying.

        Feeds only the Stage 5 long-buildup / short-covering CONTEXT check.
        Per the research program's section 2.G this is practitioner-grade
        (evidence tier ●○○) and can never be a standalone signal: OI cannot
        separate hedging from directional betting, and the four-way labels are
        inferences from the price move, not independent confirmation.
        """
        url = self._archives_url(_fmt(self.cfg.fo_bhavcopy_path, day))
        # This is the single largest payload the engine downloads (~1.3 MB per
        # session) and it yields roughly 5 KB of aggregated open interest --
        # every option strike for every underlying, of which we keep only the
        # stock-futures rows. Caching it is a ~280:1 waste of disk, so by
        # default it is parsed once and discarded.
        res = self.client.get(
            url,
            ttl_seconds=self._ttl(day),
            context="nse_archives.fo",
            cacheable=self._cacheable("fo_bhavcopy"),
        )
        if res is None:
            return None
        raw = _read_zip_csv(res.content)
        if "FinInstrmTp" not in raw.columns:
            return None

        futures = raw[raw["FinInstrmTp"].astype(str).str.upper() == "STF"].copy()
        if futures.empty:
            return None

        futures[SYMBOL] = futures["TckrSymb"].map(normalise_symbol)
        futures["oi"] = _num(futures["OpnIntrst"])
        futures["oi_change"] = _num(futures["ChngInOpnIntrst"])
        futures["fut_close"] = _num(futures["ClsPric"])
        futures["fut_volume"] = _num(futures["TtlTradgVol"])

        grouped = (
            futures.groupby(SYMBOL, as_index=False)
            .agg(
                oi=("oi", "sum"),
                oi_change=("oi_change", "sum"),
                fut_volume=("fut_volume", "sum"),
                fut_close=("fut_close", "first"),
            )
        )
        grouped[DATE] = pd.to_datetime(day)
        grouped["source"] = self.name
        return grouped[[DATE, SYMBOL, "oi", "oi_change", "fut_volume", "fut_close", "source"]]

    # =========================================================================
    # Calendar discovery
    # =========================================================================
    def session_exists(self, day: dt.date) -> bool:
        """Cheap probe: did NSE publish an index file for ``day``?

        This is how the trading calendar is *discovered* rather than assumed.
        """
        url = self._archives_url(_fmt(self.cfg.index_close_all_path, day))
        try:
            res = self.client.get(url, ttl_seconds=self._ttl(day), context="nse_archives.probe")
        except ProviderError:
            return False
        return res is not None

    def discover_sessions(
        self, start: dt.date, end: dt.date, skip_probable_closures: bool = True
    ) -> List[dt.date]:
        """Probe each candidate day and return the ones that really traded."""
        from ...core.calendar import is_probably_closed

        found: List[dt.date] = []
        cur = start
        while cur <= end:
            if skip_probable_closures and is_probably_closed(cur):
                cur += dt.timedelta(days=1)
                continue
            if self.session_exists(cur):
                found.append(cur)
            cur += dt.timedelta(days=1)
        return found
