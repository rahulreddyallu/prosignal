"""Point-in-time fundamentals from NSE quarterly results (Ind-AS XBRL).

The strongest India-specific factor evidence sits with value and profitability
(Fama-French replications on CNX 500 / NSE 500), neither computable without
fundamentals carrying a filing date.

Source choice. Point-in-time integrity requires knowing when a number became
public, not only which quarter it describes. NSE's per-symbol results endpoint
carries `filingDate` and `broadCastDate` alongside `fromDate`/`toDate`, so a
figure can be gated on when the market learned it. Measured disclosure lag is
20-45 days (DATA_SOURCES.md); a feed keyed only to period end leaks that window
into a backtest.

Coverage. The quarterly filing is an income statement, giving margins, interest
coverage, earnings growth and earnings stability. It carries no equity, assets
or borrowings, so ROE, book value and debt-to-equity are not derivable here --
Indian companies file balance sheets half-yearly at best. Those factors are
absent rather than approximated.

`PaidUpValueOfEquityShareCapital / FaceValueOfEquityShareCapital` gives shares
outstanding, which with price gives market capitalisation, unlocking earnings
yield and a real "top N by market cap" for universe construction.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ...core.logging import get_logger
from ..types import SYMBOL, normalise_symbol
from .http import HttpClient, NseJsonSession

__all__ = ["NseFundamentalsProvider", "FUNDAMENTAL_COLUMNS", "parse_indas_xbrl",
           "RESULTS_CALENDAR_COLUMNS"]

log = get_logger(__name__)

#: Columns written to the curated `fundamentals` table.
FUNDAMENTAL_COLUMNS = [
    SYMBOL, "filing_date", "filing_ts", "period_end", "period_start", "consolidated",
    "revenue", "other_income", "total_income", "expenses", "finance_costs",
    "depreciation", "profit_before_tax", "tax_expense", "net_profit",
    "paid_up_capital", "face_value", "shares_outstanding",
]

#: Columns written to the curated `results_calendar` table.
#:
#: THE METADATA IS WORTH INGESTING ON ITS OWN, separately from the line items,
#: because the two have wildly different costs. The metadata is ONE request per
#: symbol and returns every quarter the company has ever filed -- 95 to 162 rows
#: reaching back to 2005-03-31. The line items need one XBRL fetch per
#: (symbol, quarter), which for a 750-name universe over that span is roughly
#: ninety thousand requests.
#:
#: So this table answers "WHEN did each company report, and when did the market
#: learn it" for the whole universe in about five minutes, and carries the
#: `xbrl_url` so the line items can be fetched incrementally afterwards for
#: whatever subset a factor actually needs.
#:
#: It also fixes something `features/earnings.py` records as a hard limit: the
#: earnings calendar "is dense for 179 symbols and has a median of two rows for
#: everybody else", which is why earnings proximity ships as a risk disclosure
#: rather than a factor.
RESULTS_CALENDAR_COLUMNS = [
    SYMBOL, "period_end", "period_start", "filing_ts", "broadcast_ts",
    "availability_date", "consolidated", "audited", "relating_to",
    "financial_year", "is_bank", "ind_as", "isin", "xbrl_url", "seq_number",
]

#: Ind-AS element names -> our column names. Taken from live filings, not docs.
_TAGS = {
    "RevenueFromOperations": "revenue",
    "OtherIncome": "other_income",
    "Income": "total_income",
    "Expenses": "expenses",
    "FinanceCosts": "finance_costs",
    "DepreciationDepletionAndAmortisationExpense": "depreciation",
    "ProfitBeforeTax": "profit_before_tax",
    "TaxExpense": "tax_expense",
    "ProfitLossForPeriod": "net_profit",
    "PaidUpValueOfEquityShareCapital": "paid_up_capital",
    "FaceValueOfEquityShareCapital": "face_value",
}


def _num(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "")
    if not cleaned or cleaned in {"-", "NA", "nan"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_indas_xbrl(content: bytes) -> Dict[str, Optional[float]]:
    """Extract the line items we use from an Ind-AS XBRL filing.

    Deliberately regex-based rather than a full XML parse. These documents carry
    inconsistent namespace prefixes across filers and years, and every element
    of interest is a flat numeric fact -- so matching on the local element name
    is more robust here than binding to a namespace that varies.

    Where an element appears more than once (different contexts -- quarter vs
    year-to-date), the FIRST occurrence is taken, which is the primary reporting
    context in the filings inspected. This is an assumption, and it is the one
    most likely to need revisiting if a filer orders contexts differently.
    """
    text = content.decode("utf-8", "replace")
    out: Dict[str, Optional[float]] = {col: None for col in _TAGS.values()}
    for tag, column in _TAGS.items():
        m = re.search(rf"<[A-Za-z0-9\-]*:?{tag}\b[^>]*>([^<]*)<", text)
        if m:
            out[column] = _num(m.group(1))
    return out


class NseFundamentalsProvider:
    """Fetch per-symbol quarterly results and parse their XBRL."""

    name = "nse_fundamentals"

    #: NSE returns both Consolidated and Non-Consolidated rows for the same
    #: quarter. Consolidated is the economically meaningful one for a group, so
    #: it wins; standalone is the fallback for companies that file only that.
    _PREFER_CONSOLIDATED = True

    def __init__(
        self,
        session: NseJsonSession,
        client: HttpClient,
        results_path: str = "/api/corporates-financial-results?index=equities&symbol={symbol}&period=Quarterly",
        max_quarters: int = 8,
    ) -> None:
        self.session = session
        self.client = client
        self.results_path = results_path
        self.max_quarters = max_quarters
        self.last_error: Optional[str] = None

    # -- one symbol ---------------------------------------------------------
    def fetch_symbol(self, symbol: str) -> pd.DataFrame:
        """Recent quarterly filings for one symbol, each with its filing date."""
        sym = normalise_symbol(symbol)
        payload = self.session.get_json(
            self.results_path.format(symbol=sym), ttl_seconds=86400.0
        )
        if not payload:
            self.last_error = f"{sym}: results endpoint returned nothing"
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        picked = self._pick_quarters(rows)
        records: List[Dict[str, object]] = []

        for row in picked:
            xbrl = row.get("xbrl")
            if not xbrl:
                continue
            try:
                res = self.client.get(
                    xbrl, ttl_seconds=-1, context="nse_fundamentals.xbrl"
                )
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                self.last_error = f"{sym}: xbrl fetch failed: {exc}"
                log.debug("xbrl fetch failed", extra={"symbol": sym, "error": str(exc)})
                continue
            if res is None:
                continue

            parsed = parse_indas_xbrl(res.content)
            filing_ts = (_parse_ts(row.get("filingDate"))
                         or _parse_ts(row.get("broadCastDate")))
            filing = None if filing_ts is None else filing_ts.date()
            if filing is None:
                # Without a filing date the row cannot be used point-in-time,
                # and using it anyway is precisely the leakage this feed exists
                # to prevent. Drop it.
                continue

            shares = _shares_outstanding(parsed)
            records.append({
                SYMBOL: sym,
                "filing_date": filing,
                # THE TIME, CARRIED. `filing_date` is the calendar date of the
                # filing; `filing_ts` is when it actually appeared, and 59.1% of
                # this feed appears after the 15:30 close. Only the second can
                # decide which session may act on it -- see
                # `features/pit_fundamentals.availability_date`.
                "filing_ts": filing_ts,
                "period_end": _parse_date(row.get("toDate")),
                "period_start": _parse_date(row.get("fromDate")),
                "consolidated": str(row.get("consolidated", "")).strip(),
                **parsed,
                "shares_outstanding": shares,
            })

        if not records:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        frame = pd.DataFrame.from_records(records)
        for col in FUNDAMENTAL_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        return frame[FUNDAMENTAL_COLUMNS]

    def fetch_universe(
        self, symbols: Iterable[str], progress=None
    ) -> pd.DataFrame:
        """Fetch every symbol. Failures are recorded, never fatal."""
        frames: List[pd.DataFrame] = []
        symbols = list(symbols)
        failed: List[str] = []
        for i, sym in enumerate(symbols, start=1):
            if progress:
                progress(i, len(symbols), sym)
            try:
                frame = self.fetch_symbol(sym)
            except Exception as exc:  # noqa: BLE001
                failed.append(sym)
                log.debug("fundamentals failed", extra={"symbol": sym, "error": str(exc)})
                continue
            if frame.empty:
                failed.append(sym)
            else:
                frames.append(frame)

        if failed:
            log.warning(
                "fundamentals unavailable for some symbols",
                extra={"failed": len(failed), "of": len(symbols)},
            )
        if not frames:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # -- the results calendar: metadata only, every quarter ever filed ------
    def fetch_calendar_symbol(self, symbol: str) -> pd.DataFrame:
        """Every quarterly filing this symbol has ever made, WITHOUT the XBRL.

        One request. No line items, so no per-quarter XBRL fetch -- which is
        what makes the whole universe affordable. Every row carries the filing
        TIMESTAMP and the derived `availability_date`, so a consumer never has
        to re-derive when the market could have acted on it.
        """
        from ...features.pit_fundamentals import availability_date

        sym = normalise_symbol(symbol)
        payload = self.session.get_json(
            self.results_path.format(symbol=sym), ttl_seconds=86400.0)
        if not payload:
            self.last_error = f"{sym}: results endpoint returned nothing"
            return pd.DataFrame(columns=RESULTS_CALENDAR_COLUMNS)
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=RESULTS_CALENDAR_COLUMNS)

        out: List[Dict[str, object]] = []
        for row in rows:
            filing_ts = (_parse_ts(row.get("filingDate"))
                         or _parse_ts(row.get("broadCastDate")))
            period_end = _parse_date(row.get("toDate"))
            if filing_ts is None or period_end is None:
                # No filing timestamp or no period means the row cannot be
                # placed point-in-time. Dropped rather than guessed.
                continue
            out.append({
                SYMBOL: sym,
                "period_end": period_end,
                "period_start": _parse_date(row.get("fromDate")),
                "filing_ts": filing_ts,
                "broadcast_ts": _parse_ts(row.get("broadCastDate")),
                "consolidated": str(row.get("consolidated", "")).strip(),
                "audited": str(row.get("audited", "")).strip(),
                "relating_to": str(row.get("relatingTo", "")).strip(),
                "financial_year": str(row.get("financialYear", "")).strip(),
                "is_bank": str(row.get("bank", "")).strip().upper() == "Y",
                "ind_as": str(row.get("indAs", "")).strip(),
                "isin": str(row.get("isin", "")).strip(),
                "xbrl_url": str(row.get("xbrl") or "").strip() or None,
                "seq_number": str(row.get("seqNumber", "")).strip(),
            })
        if not out:
            return pd.DataFrame(columns=RESULTS_CALENDAR_COLUMNS)
        frame = pd.DataFrame(out)
        frame["availability_date"] = availability_date(frame["filing_ts"]).to_numpy()
        return frame[RESULTS_CALENDAR_COLUMNS]

    def fetch_calendar(self, symbols: Iterable[str], progress=None) -> pd.DataFrame:
        """The results calendar for every symbol. Failures are counted, never fatal.

        A symbol that returns nothing is UNKNOWN, not "never filed": NSE's bot
        shield and this endpoint's own quirks both produce empty answers that
        mean "ask again", and treating those as an absence of filings would
        convert "could not check" into "check passed".
        """
        frames: List[pd.DataFrame] = []
        symbols = list(symbols)
        empty: List[str] = []
        for i, sym in enumerate(symbols, start=1):
            if progress:
                progress(i, len(symbols), sym)
            try:
                frame = self.fetch_calendar_symbol(sym)
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                empty.append(sym)
                log.debug("results calendar failed",
                          extra={"symbol": sym, "error": str(exc)})
                continue
            if frame.empty:
                empty.append(sym)
            else:
                frames.append(frame)
        if empty:
            log.warning("results calendar returned nothing for some symbols",
                        extra={"unknown": len(empty), "of": len(symbols)})
        self.calendar_unknown = empty
        if not frames:
            return pd.DataFrame(columns=RESULTS_CALENDAR_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # -- helpers ------------------------------------------------------------
    def _pick_quarters(self, rows: List[dict]) -> List[dict]:
        """Most recent N quarters, preferring consolidated over standalone."""
        by_period: Dict[str, dict] = {}
        for row in rows:
            key = str(row.get("toDate") or "")
            if not key:
                continue
            existing = by_period.get(key)
            if existing is None:
                by_period[key] = row
                continue
            if self._PREFER_CONSOLIDATED:
                is_con = "non" not in str(row.get("consolidated", "")).lower()
                was_con = "non" not in str(existing.get("consolidated", "")).lower()
                if is_con and not was_con:
                    by_period[key] = row

        ordered = sorted(
            by_period.values(),
            key=lambda r: _parse_date(r.get("toDate")) or dt.date.min,
            reverse=True,
        )
        return ordered[: self.max_quarters]


def _shares_outstanding(parsed: Dict[str, Optional[float]]) -> Optional[float]:
    """Shares = paid-up equity capital / face value per share.

    The only route to market capitalisation from this filing, and therefore the
    only route to an earnings yield. Guarded because a zero or absent face value
    would otherwise produce an infinite share count.
    """
    paid = parsed.get("paid_up_capital")
    face = parsed.get("face_value")
    if not paid or not face or face <= 0:
        return None
    shares = paid / face
    return shares if shares > 0 else None


def _parse_date(value) -> Optional[dt.date]:
    if not value:
        return None
    parsed = pd.to_datetime(str(value).strip(), errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _parse_dt(value) -> Optional[dt.date]:
    """The filing's calendar DATE. See `_parse_ts` for why that is not enough.

    Kept because `filing_date` remains the calendar date of the filing, which is
    the honest meaning of that column name. What changed is that nothing gates
    on it directly any more -- `features/pit_fundamentals.availability_date`
    does, and it needs the time.
    """
    ts = _parse_ts(value)
    return None if ts is None else ts.date()


def _parse_ts(value) -> Optional[pd.Timestamp]:
    """The filing's full timestamp, TIME INCLUDED.

    THE TIME IS THE POINT, and discarding it was a one-session lookahead on
    most of this feed. NSE stamps filings like '16-Jan-2025 20:20' -- three
    hours after the 15:30 close. Measured 2026-09-04 over 1,204 filings across
    ten symbols, **59.1% are stamped after the close**, with the modal filing
    hour between 17:00 and 19:00.

    The previous version of this function did `str(value).split(" ")[0]` and
    returned a bare date, with a docstring saying "Date is what we gate on".
    Stored as a midnight date, an 20:20 filing became visible to the as-of join
    on the very session it was filed -- a session whose decision is taken at
    the 15:30 close, hours before the filing existed.

    A backtest cannot see that. It is invisible, it is in the flattering
    direction, and it applies to three fifths of the rows.
    """
    if not value:
        return None
    parsed = pd.to_datetime(str(value).strip(), errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed
