"""Quarterly shareholding patterns, and the free float they carry.

WHAT THIS UNLOCKS, and it is not a factor. `costs.impact_model` scales
participation by TRADED VALUE because the engine has no float data, and the
build plan's own note is the reason that matters: Indian promoter holdings are
high enough that two identical-capitalisation names can have floats differing
threefold, so an impact model keyed to market cap prices them the same when
they are not remotely the same to trade.

`public_val` is the free float as a percentage, published per quarter, per
symbol, by the exchange. It is not derived and not estimated.

POINT-IN-TIME, WITH THE SAME CLOCK PROBLEM AS THE RESULTS FEED. Every row
carries `broadcastDate` -- when NSE published it -- and that stamp has a TIME.
RELIANCE's June 2026 pattern was broadcast at 19:24:44, four hours after the
close. Keyed on `date` (the period end) the feed leaks about three weeks;
keyed on the broadcast DATE alone it still leaks a session on everything filed
after 15:30. So `availability_date` is derived here the same way it is for
filings, and it is the only field a factor may gate on.

Measured 2026-09-04 across 169 filings on eight symbols: period end to
broadcast is min 0, median 18, p90 21, max 115 days. The `min 0` -- a broadcast
stamped the same day as the period it describes -- is not credible as a
disclosure lag and is flagged rather than trusted.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from ...core.logging import get_logger
from ..types import SYMBOL, normalise_symbol
from .http import HttpClient, NseJsonSession

__all__ = ["NseShareholdingProvider", "SHAREHOLDING_COLUMNS"]

log = get_logger(__name__)

#: Columns written to the curated `shareholding` table.
SHAREHOLDING_COLUMNS = [
    SYMBOL, "period_end", "broadcast_ts", "availability_date",
    "promoter_pct", "public_pct", "employee_trust_pct", "free_float_pct",
    "revised", "suspect_lag", "xbrl_url",
]

#: A broadcast lag below this many days is not a disclosure lag. SEBI LODR gives
#: 21 days from quarter end for the shareholding pattern, and the measured
#: median is 18 -- so a row claiming to have been broadcast on or within a day
#: of the period it describes is a data-quality artefact, not a fast filer.
#: Flagged rather than dropped: the row may still be usable, and silently
#: discarding data is how a coverage number becomes a fiction.
MIN_CREDIBLE_LAG_DAYS = 2


def _pct(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "NA", "None", "nan"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    # Percentages only. A value outside [0, 100] is a parse failure, not a
    # holding, and admitting it would put a nonsense free float into an impact
    # model that then prices a trade on it.
    return out if 0.0 <= out <= 100.0 else None


def _ts(value) -> Optional[pd.Timestamp]:
    if not value:
        return None
    parsed = pd.to_datetime(str(value).strip(), errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed


class NseShareholdingProvider:
    """Fetch the quarterly shareholding pattern per symbol."""

    name = "nse_shareholding"

    def __init__(self, session: NseJsonSession, client: HttpClient,
                 path: str = "/api/corporate-share-holdings-master"
                             "?index=equities&symbol={symbol}") -> None:
        self.session = session
        self.client = client
        self.path = path
        self.last_error: Optional[str] = None
        self.unknown: List[str] = []

    def fetch_symbol(self, symbol: str) -> pd.DataFrame:
        from ...features.pit_fundamentals import availability_date

        sym = normalise_symbol(symbol)
        payload = self.session.get_json(self.path.format(symbol=sym),
                                        ttl_seconds=86400.0)
        if not payload:
            self.last_error = f"{sym}: shareholding endpoint returned nothing"
            return pd.DataFrame(columns=SHAREHOLDING_COLUMNS)
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=SHAREHOLDING_COLUMNS)

        out: List[Dict[str, object]] = []
        for row in rows:
            period_end = _ts(row.get("date"))
            broadcast = _ts(row.get("broadcastDate"))
            if period_end is None or broadcast is None:
                # Without both, the row cannot be placed point-in-time.
                continue
            promoter = _pct(row.get("pr_and_prgrp"))
            public = _pct(row.get("public_val"))
            trusts = _pct(row.get("employeeTrusts"))
            lag = (broadcast.normalize() - period_end.normalize()).days
            out.append({
                SYMBOL: sym,
                "period_end": period_end.normalize(),
                "broadcast_ts": broadcast,
                "promoter_pct": promoter,
                "public_pct": public,
                "employee_trust_pct": trusts,
                # THE FREE FLOAT. `public_val` is the exchange's own figure for
                # the share not held by the promoter group; it is not derived
                # here and not estimated.
                "free_float_pct": public,
                "revised": str(row.get("revisedData", "")).strip().upper() == "Y",
                "suspect_lag": bool(lag < MIN_CREDIBLE_LAG_DAYS),
                "xbrl_url": str(row.get("xbrl") or "").strip() or None,
            })
        if not out:
            return pd.DataFrame(columns=SHAREHOLDING_COLUMNS)
        frame = pd.DataFrame(out)
        frame["availability_date"] = availability_date(frame["broadcast_ts"]).to_numpy()
        return frame[SHAREHOLDING_COLUMNS]

    def fetch_universe(self, symbols: Iterable[str], progress=None) -> pd.DataFrame:
        """Every symbol. A symbol returning nothing is UNKNOWN, never "no data".

        NSE's bot shield and this endpoint both produce empty answers that mean
        "ask again". Recording those as an absence of filings would turn "could
        not check" into "check passed", which is the failure the NOT_TESTABLE
        convention exists to prevent -- so they are counted and reported.
        """
        frames: List[pd.DataFrame] = []
        symbols = list(symbols)
        unknown: List[str] = []
        for i, sym in enumerate(symbols, start=1):
            if progress:
                progress(i, len(symbols), sym)
            try:
                frame = self.fetch_symbol(sym)
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                unknown.append(sym)
                log.debug("shareholding failed",
                          extra={"symbol": sym, "error": str(exc)})
                continue
            if frame.empty:
                unknown.append(sym)
            else:
                frames.append(frame)
        self.unknown = unknown
        if unknown:
            log.warning("shareholding returned nothing for some symbols",
                        extra={"unknown": len(unknown), "of": len(symbols)})
        if not frames:
            return pd.DataFrame(columns=SHAREHOLDING_COLUMNS)
        return pd.concat(frames, ignore_index=True)
