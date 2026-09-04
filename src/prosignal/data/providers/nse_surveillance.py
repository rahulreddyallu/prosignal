"""Surveillance measures and F&O eligibility — the admission gates.

WHY THESE MATTER MORE ON THE SHORT SIDE THAN THE LONG. A bottom-decile short
screen selects disproportionately into exactly the names the exchange has put
under surveillance: 2% price bands, 100% margin, Trade-for-Trade settlement,
and in some ASM stages an explicit restriction on short selling. A backtest
that fills in those names is not modelling a trade that was available. The
long side has the same problem in weaker form, which is why these are ingested
as a hard gate for one leg and a warning flag for the other.

TWO FILES, BOTH FROM THE ARCHIVE HOST, NEITHER NEEDING A COOKIE.

  sec_list.csv    3,535 securities with SERIES, price BAND and a REMARKS field
                  that names the GSM stage outright. Series BE and BZ are
                  Trade-for-Trade; a band of 2% or 5% against the ordinary 20%
                  is the ASM/GSM signature even where the remark is blank.

  fo_mktlots.csv  the F&O-eligible list AND the LOT SIZE per expiry. The build
                  plan is explicit that the lot size must be read per symbol
                  per date rather than assumed, because a single-stock futures
                  position rounds to a lot and not to a rupee -- which is what
                  decides whether a 25-name short leg is representable at all
                  at a given book size.

THE LIMITATION, STATED UP FRONT: BOTH ARE CURRENT SNAPSHOTS. NSE does not
publish these at a dated URL, so membership accumulates going FORWARD from the
first snapshot and cannot be reconstructed backwards. Every row therefore
carries `snapshot_date`, and a consumer asking about a date before the first
snapshot must get NOT_TESTABLE rather than today's list -- projecting today's
surveillance list backwards is the same lookahead the universe screen already
refuses for index membership, and it runs in the flattering direction: names
under surveillance today were often perfectly tradable when the panel says the
book bought them, and vice versa.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd

from ...core.logging import get_logger
from ..types import SYMBOL, normalise_symbol
from .http import HttpClient

__all__ = ["NseSurveillanceProvider", "SECURITY_LIST_COLUMNS", "FO_LOT_COLUMNS",
           "ORDINARY_BAND_PCT", "T2T_SERIES"]

log = get_logger(__name__)

SECURITY_LIST_COLUMNS = [
    SYMBOL, "snapshot_date", "series", "band_pct", "gsm_stage", "asm_flag",
    "is_t2t", "restricted", "security_name",
]

FO_LOT_COLUMNS = [SYMBOL, "snapshot_date", "lot_size", "near_expiry",
                  "fo_eligible", "underlying", "is_index"]

#: The ordinary NSE price band. Anything tighter is a surveillance measure.
ORDINARY_BAND_PCT = 20.0

#: Series settled Trade-for-Trade: no intraday netting, 100% delivery both
#: ways. BZ additionally means the name is under a surveillance action.
T2T_SERIES = frozenset({"BE", "BZ"})


def _band(value) -> Optional[float]:
    text = str(value).strip()
    if not text or text.lower() in {"no band", "-", "nan", "none"}:
        # "No Band" is not a missing value -- it is the absence of a price
        # band, which is the OPPOSITE of a surveillance restriction. Returned
        # as None and read as unrestricted by `restricted` below.
        return None
    try:
        return float(text.replace("%", ""))
    except ValueError:
        return None


def _gsm_stage(remark) -> Optional[str]:
    text = str(remark or "").strip()
    if not text or text == "-":
        return None
    up = text.upper()
    return text if ("GSM" in up or "ASM" in up or "ESM" in up) else None


class NseSurveillanceProvider:
    """Current surveillance state and F&O eligibility, as dated snapshots."""

    name = "nse_surveillance"

    def __init__(self, client: HttpClient, base: str,
                 security_list_path: str = "/content/equities/sec_list.csv",
                 fo_lots_path: str = "/content/fo/fo_mktlots.csv") -> None:
        self.client = client
        self.base = base.rstrip("/")
        self.security_list_path = security_list_path
        self.fo_lots_path = fo_lots_path
        self.last_error: Optional[str] = None

    # -- surveillance -------------------------------------------------------
    def fetch_security_list(self, snapshot_date: Optional[dt.date] = None
                            ) -> pd.DataFrame:
        """Series, price band and GSM/ASM stage for every listed security."""
        snapshot = snapshot_date or dt.date.today()
        try:
            res = self.client.get(f"{self.base}{self.security_list_path}",
                                  ttl_seconds=3600.0,
                                  context="nse_surveillance.sec_list")
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the run
            self.last_error = f"security list unavailable: {exc}"
            log.warning("security list unavailable", extra={"error": str(exc)})
            return pd.DataFrame(columns=SECURITY_LIST_COLUMNS)
        if res is None:
            self.last_error = "security list: HTTP 404"
            return pd.DataFrame(columns=SECURITY_LIST_COLUMNS)

        import io
        raw = pd.read_csv(io.BytesIO(res.content))
        raw.columns = [c.strip() for c in raw.columns]
        need = {"Symbol", "Series"}
        if not need.issubset(raw.columns):
            self.last_error = f"security list columns changed: {list(raw.columns)}"
            log.warning("security list schema changed",
                        extra={"columns": list(raw.columns)})
            return pd.DataFrame(columns=SECURITY_LIST_COLUMNS)

        out = pd.DataFrame({
            SYMBOL: raw["Symbol"].astype(str).map(normalise_symbol),
            "snapshot_date": pd.Timestamp(snapshot),
            "series": raw["Series"].astype(str).str.strip().str.upper(),
            "band_pct": raw.get("Band", pd.Series(index=raw.index)).map(_band),
            "gsm_stage": raw.get("Remarks", pd.Series(index=raw.index)).map(_gsm_stage),
            "security_name": raw.get("Security Name",
                                     pd.Series(index=raw.index)).astype(str).str.strip(),
        })
        out["asm_flag"] = out["gsm_stage"].astype(str).str.upper().str.contains("ASM")
        out["is_t2t"] = out["series"].isin(T2T_SERIES)
        # RESTRICTED is the gate. A name is restricted when it settles
        # Trade-for-Trade, carries an explicit surveillance stage, or has had
        # its price band cut below the ordinary 20%. Any one is enough: they are
        # different measures for the same judgement by the exchange, and a name
        # under any of them cannot be filled at a simulated price.
        tight_band = out["band_pct"].notna() & (out["band_pct"] < ORDINARY_BAND_PCT)
        out["restricted"] = out["is_t2t"] | out["gsm_stage"].notna() | tight_band
        return out[SECURITY_LIST_COLUMNS]

    # -- F&O eligibility and lot size ---------------------------------------
    def fetch_fo_lots(self, snapshot_date: Optional[dt.date] = None
                      ) -> pd.DataFrame:
        """The F&O-eligible list with the lot size of the nearest expiry.

        The lot size is READ, not assumed. A single-stock futures position
        rounds to a lot, so the lot value decides whether a name is even
        representable in a book of a given size -- and lot values commonly sit
        in the several-lakh range, which is most of a ten-lakh paper book.
        """
        snapshot = snapshot_date or dt.date.today()
        try:
            res = self.client.get(f"{self.base}{self.fo_lots_path}",
                                  ttl_seconds=3600.0,
                                  context="nse_surveillance.fo_lots")
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"F&O lot file unavailable: {exc}"
            log.warning("F&O lot file unavailable", extra={"error": str(exc)})
            return pd.DataFrame(columns=FO_LOT_COLUMNS)
        if res is None:
            self.last_error = "F&O lot file: HTTP 404"
            return pd.DataFrame(columns=FO_LOT_COLUMNS)

        import io
        raw = pd.read_csv(io.BytesIO(res.content))
        raw.columns = [c.strip() for c in raw.columns]
        if "SYMBOL" not in raw.columns:
            self.last_error = f"F&O lot columns changed: {list(raw.columns)}"
            return pd.DataFrame(columns=FO_LOT_COLUMNS)

        # Expiry columns look like 'SEP-26'. The first one that parses is the
        # near month; its lot is the one a position would actually round to.
        expiry_cols = [c for c in raw.columns
                       if c not in ("UNDERLYING", "SYMBOL")
                       and pd.notna(pd.to_datetime(c.strip(), format="%b-%y",
                                                   errors="coerce"))]
        if not expiry_cols:
            self.last_error = "F&O lot file has no parseable expiry column"
            return pd.DataFrame(columns=FO_LOT_COLUMNS)
        near = expiry_cols[0]

        lots = pd.to_numeric(raw[near].astype(str).str.strip(), errors="coerce")
        underlying = raw.get("UNDERLYING",
                             pd.Series(index=raw.index)).astype(str).str.strip()
        out = pd.DataFrame({
            SYMBOL: raw["SYMBOL"].astype(str).map(normalise_symbol),
            "snapshot_date": pd.Timestamp(snapshot),
            "lot_size": lots,
            "near_expiry": near.strip(),
            "fo_eligible": lots.notna(),
            "underlying": underlying,
        })
        # INDEX DERIVATIVES SHARE THIS FILE WITH SINGLE STOCKS and must not
        # enter a shortable-EQUITY count -- a NIFTY future is a beta hedge, not
        # a name the cross-section can pick.
        #
        # Detected from the UNDERLYING text rather than a hardcoded list of
        # index symbols, because NSE adds index products (MIDCPNIFTY, NIFTYNXT50)
        # and a fixed list would silently start counting the new ones as
        # equities. The underlying of a stock future is a company name; the
        # underlying of an index future begins with the index family.
        u = out["underlying"].str.upper()
        out["is_index"] = u.str.startswith("NIFTY") | u.str.startswith("BANKNIFTY")
        out = out[out[SYMBOL].notna() & out["lot_size"].notna()]
        return out[FO_LOT_COLUMNS].reset_index(drop=True)
