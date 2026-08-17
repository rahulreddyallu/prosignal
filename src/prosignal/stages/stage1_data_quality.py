"""Stage 1 -- Data Quality and Leakage Gate.

This stage exists because of an asymmetry that is easy to state and easy to
forget: a wrong number that *looks* reasonable is far more dangerous than a
missing one. A missing price stops the pipeline. An unadjusted 5:1 split reads
as a clean -80% session, sails through every downstream stage, and poisons a
12-1 momentum score for the next twelve months while looking entirely normal on
a chart.

So every check here is **binary and independent**. Nothing is blended into a
"data quality score", because a score lets three moderate problems average out
into an acceptable-looking number. Either a stock's data can be trusted or it
cannot.

Two failure levels, and the distinction is the whole design:

* **Market-wide** -- the *feed* is broken. Raises :class:`MarketWideHalt`; the
  engine refuses to form an opinion at all. Note this is emphatically NOT the
  same as NO TRADE, which means "we ran cleanly and nothing qualified".
* **Per-stock** -- this *name's* data is untrustworthy. The stock is excluded;
  the run continues.

The universe-wide failure fraction is what separates the two automatically. If
a quarter of the universe fails a stock-level check on the same session, the
overwhelmingly more likely explanation is that the feed changed format, not
that 50 companies simultaneously had bad ticks. Treating that as 50 individual
exclusions would silently shrink the universe to the handful of names whose
data happened to survive -- which is a far worse outcome than halting.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.calendar import TradingCalendar
from ..core.contracts import DataQualityReport, RawDataManifest, StockDataFlags
from ..core.enums import GateResult
from ..core.errors import MarketWideHalt
from ..core.logging import get_logger
from ..data.corporate_actions import detect_unexplained_jumps
from ..data.store import DataStore
from ..data.types import DATE, SYMBOL, normalise_symbol
from ..data.universe import UniverseSnapshot
from ..indicators import sigma_move

__all__ = ["run", "STAGE_NAME"]

STAGE_NAME = "stage1_data_quality"

log = get_logger(__name__)

#: Check identifiers. Strings appear verbatim on the recommendation card and in
#: the ledger, so they are defined once rather than typed at each call site.
CHECK_STALENESS = "feed_staleness"
CHECK_CONTINUITY = "session_continuity"
CHECK_OUTLIER = "outlier_return"
CHECK_UNEXPLAINED_ACTION = "unexplained_corporate_action"
CHECK_SOURCE_AGREEMENT = "cross_source_agreement"
CHECK_NO_PRICE = "no_price_data"
CHECK_SINGLE_SOURCE = "single_price_source"


# =============================================================================
# entry point
# =============================================================================


def run(
    manifest: RawDataManifest,
    store: DataStore,
    calendar: TradingCalendar,
    universe: UniverseSnapshot,
    config,
) -> DataQualityReport:
    """Execute the Stage 1 gate.

    Raises
    ------
    MarketWideHalt
        When a required feed is missing or stale, or when so much of the
        universe fails that the feed itself is the likely culprit.
    """
    params = config.params.stage1_data_quality
    as_of = manifest.as_of_date

    market_failures: List[str] = []
    market_soft_flags: List[str] = []

    # -- 1. required feeds present and fresh -------------------------------
    market_failures.extend(_check_required_feeds(manifest))

    # A missing required feed means the rest of this stage would be measuring
    # noise. Stop here rather than producing a confident-looking report built
    # on data we already know is absent.
    if market_failures:
        raise MarketWideHalt(market_failures, stage=STAGE_NAME)

    # -- 2. load the price window every per-stock check shares --------------
    symbols = [normalise_symbol(s) for s in universe.symbols]
    window_sessions = _required_window(params)
    window_start = _window_start(calendar, as_of, window_sessions)

    prices = store.read_prices(symbols=symbols, start=window_start, end=as_of)
    actions = store.read_corporate_actions()
    secondary = _read_secondary(store, window_start, as_of)

    if prices.empty:
        raise MarketWideHalt(
            [
                f"no price rows for any of {len(symbols)} universe symbols between "
                f"{window_start} and {as_of}"
            ],
            stage=STAGE_NAME,
        )

    prices = prices.copy()
    prices[DATE] = pd.to_datetime(prices[DATE]).dt.normalize()
    prices = prices.sort_values([SYMBOL, DATE])

    expected_sessions = _expected_sessions(calendar, as_of, params)

    # -- 3. one pass of unexplained corporate actions for the whole frame ---
    # Computed once over the whole universe rather than per symbol: the
    # detector is vectorised and this is the difference between one pass and
    # 200.
    suspects = detect_unexplained_jumps(
        prices,
        actions,
        min_ratio_gap=float(params.unexplained_split_min_ratio_gap.value),
        tolerance=float(params.unexplained_split_ratio_tolerance.value),
        lookback_sessions=int(params.continuity_window_sessions.value),
    )
    suspect_symbols = _suspect_map(suspects)

    # -- 4. per-stock checks -------------------------------------------------
    per_stock: Dict[str, StockDataFlags] = {}
    grouped = {sym: frame for sym, frame in prices.groupby(SYMBOL, sort=False)}

    for symbol in symbols:
        frame = grouped.get(symbol)
        flags = _check_one_stock(
            symbol=symbol,
            frame=frame,
            params=params,
            expected_sessions=expected_sessions,
            suspect_dates=suspect_symbols.get(symbol),
            secondary=secondary,
            as_of=as_of,
        )
        per_stock[symbol] = flags

    checked = len(per_stock)
    failed = sum(1 for f in per_stock.values() if f.status is GateResult.FAIL)

    # -- 5. universe-wide failure fraction ----------------------------------
    max_fraction = float(params.max_universe_failure_fraction.value)
    min_sample = int(params.min_universe_for_failure_fraction.value)
    if checked >= min_sample:
        fraction = failed / checked
        if fraction > max_fraction:
            reason = (
                f"{failed}/{checked} ({fraction:.1%}) of the universe failed "
                f"stock-level data checks, above the "
                f"{max_fraction:.0%} ceiling. At this rate the feed is the "
                f"likely fault, not the stocks -- halting rather than silently "
                f"shrinking the universe to whatever survived."
            )
            log.error("universe-wide data failure", extra={"fraction": round(fraction, 4)})
            raise MarketWideHalt([reason], stage=STAGE_NAME)

    # -- 6. point-in-time audit ---------------------------------------------
    pit_audit, pit_failures = _pit_audit(manifest, universe, params)
    market_soft_flags.extend(pit_failures)

    report = DataQualityReport(
        run_status=GateResult.PASS,
        market_wide_failures=[],
        market_wide_soft_flags=market_soft_flags,
        per_stock_flags=per_stock,
        pit_audit=pit_audit,
        pit_audit_failures=pit_failures,
        checked_symbols=checked,
        failed_symbols=failed,
    )

    log.info(
        "stage 1 complete",
        extra={
            "checked": checked,
            "failed": failed,
            "soft_flags": sum(1 for f in per_stock.values() if f.soft_flags),
            "pit_failures": len(pit_failures),
        },
    )
    return report


# =============================================================================
# market-wide checks
# =============================================================================


def _check_required_feeds(manifest: RawDataManifest) -> List[str]:
    """Missing or stale REQUIRED feeds. Optional feeds degrade, never halt."""
    failures: List[str] = []

    for name in manifest.missing_required():
        record = manifest.feeds[name]
        detail = record.primary_source_error or "no rows returned"
        failures.append(f"required feed '{name}' is MISSING ({detail})")

    for name in manifest.stale_required():
        record = manifest.feeds[name]
        failures.append(
            f"required feed '{name}' is STALE: last data {record.last_timestamp}, "
            f"{record.age_sessions} sessions old, limit {record.max_age_sessions}"
        )

    return failures


# =============================================================================
# per-stock checks
# =============================================================================


def _check_one_stock(
    symbol: str,
    frame: Optional[pd.DataFrame],
    params,
    expected_sessions: List[pd.Timestamp],
    suspect_dates: Optional[List[dt.date]],
    secondary: Optional[pd.DataFrame],
    as_of: dt.date,
) -> StockDataFlags:
    """Run every stock-level check. Each is binary; none is blended."""
    failed: List[str] = []
    soft: List[str] = []
    details: Dict[str, object] = {}

    if frame is None or frame.empty:
        return StockDataFlags(
            status=GateResult.FAIL,
            failed_checks=[CHECK_NO_PRICE],
            details={"reason": "no price rows in the continuity window"},
        )

    closes = pd.Series(
        frame["close"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(frame[DATE]),
        name=symbol,
    ).dropna()

    # -- continuity ---------------------------------------------------------
    gap_ok, gap_detail = _check_continuity(closes.index, expected_sessions, params)
    details["continuity"] = gap_detail
    if not gap_ok:
        failed.append(CHECK_CONTINUITY)

    # -- unexplained corporate action ---------------------------------------
    if suspect_dates:
        failed.append(CHECK_UNEXPLAINED_ACTION)
        details["unexplained_action_dates"] = [d.isoformat() for d in suspect_dates]

    # -- outlier / bad tick --------------------------------------------------
    outlier_ok, outlier_detail = _check_outlier(frame, closes, params)
    if outlier_detail:
        details["outlier"] = outlier_detail
    if not outlier_ok:
        failed.append(CHECK_OUTLIER)

    # -- cross-source agreement ----------------------------------------------
    agreement_status, agreement_detail = _check_source_agreement(
        symbol, closes, secondary, params, as_of
    )
    if agreement_detail:
        details["source_agreement"] = agreement_detail

    if agreement_status == "reject":
        failed.append(CHECK_SOURCE_AGREEMENT)
    elif agreement_status == "flag":
        soft.append(CHECK_SOURCE_AGREEMENT)
    elif agreement_status == "single_source":
        if bool(params.require_two_price_sources.value):
            failed.append(CHECK_SINGLE_SOURCE)
        else:
            soft.append(CHECK_SINGLE_SOURCE)

    status = GateResult.FAIL if failed else GateResult.PASS
    return StockDataFlags(
        status=status, failed_checks=failed, soft_flags=soft, details=details
    )


def _check_continuity(
    observed: pd.DatetimeIndex, expected_sessions: List[pd.Timestamp], params
) -> Tuple[bool, Dict[str, object]]:
    """Longest run of consecutive missing sessions inside the window.

    Counts *consecutive* gaps rather than the total. A stock missing 8 scattered
    sessions across a quarter is a data annoyance; one missing 8 in a row was
    suspended, and that is a different fact about the company.
    """
    if not expected_sessions:
        return True, {"expected_sessions": 0, "note": "calendar window empty"}

    have = set(observed.normalize())
    missing_flags = [ts not in have for ts in expected_sessions]

    longest = 0
    current = 0
    for is_missing in missing_flags:
        current = current + 1 if is_missing else 0
        longest = max(longest, current)

    limit = int(params.max_consecutive_missing_sessions.value)
    detail = {
        "expected_sessions": len(expected_sessions),
        "observed_sessions": int(sum(1 for f in missing_flags if not f)),
        "longest_consecutive_gap": longest,
        "limit": limit,
    }
    return longest <= limit, detail


def _check_outlier(
    frame: pd.DataFrame, closes: pd.Series, params
) -> Tuple[bool, Optional[Dict[str, object]]]:
    """Bad-tick detector for the most recent session.

    A move is only rejected when it is extreme **and** unexplained. Corroborating
    volume is what separates a real event from a bad print: a genuine 20% move
    on results day arrives with a volume surge, whereas a fat-fingered tick
    typically does not. Rejecting purely on size would throw away exactly the
    breakouts the engine exists to find.

    Corporate actions are handled by the separate unexplained-action check, so
    a clean split ratio never reaches here as an outlier.
    """
    if closes.size < 2:
        return True, None

    sigma_limit = float(params.outlier_return_sigma.value)
    abs_limit_pct = float(params.outlier_absolute_return_pct.value)
    lookback = int(params.outlier_sigma_lookback_sessions.value)
    volume_multiple = float(params.outlier_corroborating_volume_multiple.value)

    last_return_pct = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1.0) * 100.0

    returns = closes.pct_change(fill_method=None).dropna()
    sigma = sigma_move(returns, window=min(lookback, max(returns.size - 1, 2)))

    breaches_sigma = sigma is not None and abs(sigma) > sigma_limit
    breaches_absolute = abs(last_return_pct) > abs_limit_pct
    if not (breaches_sigma or breaches_absolute):
        return True, None

    # Extreme. Is it corroborated by volume?
    volume_ratio = _last_volume_ratio(frame, lookback)
    corroborated = volume_ratio is not None and volume_ratio >= volume_multiple

    detail = {
        "last_return_pct": round(last_return_pct, 3),
        "sigma_move": None if sigma is None else round(float(sigma), 2),
        "sigma_limit": sigma_limit,
        "absolute_limit_pct": abs_limit_pct,
        "volume_ratio": None if volume_ratio is None else round(volume_ratio, 2),
        "volume_multiple_required": volume_multiple,
        "corroborated_by_volume": bool(corroborated),
    }
    return bool(corroborated), detail


def _last_volume_ratio(frame: pd.DataFrame, lookback: int) -> Optional[float]:
    """Last session's volume as a multiple of its trailing average."""
    if "volume" not in frame.columns:
        return None
    volumes = pd.Series(frame["volume"].to_numpy(dtype="float64")).dropna()
    if volumes.size < 2:
        return None
    window = min(lookback, volumes.size - 1)
    if window < 1:
        return None
    baseline = float(volumes.iloc[-(window + 1) : -1].mean())
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    return float(volumes.iloc[-1]) / baseline


def _check_source_agreement(
    symbol: str,
    closes: pd.Series,
    secondary: Optional[pd.DataFrame],
    params,
    as_of: dt.date,
) -> Tuple[str, Optional[Dict[str, object]]]:
    """Compare the NSE close against the secondary source, in basis points.

    Disagreement is reported, never resolved by quietly preferring a source.
    Yahoo adjusts, rounds, and sometimes lags corporate actions, so a
    difference is at least as likely to be Yahoo's as NSE's -- and "the engine
    picked the other number" is not something that should happen invisibly.

    Returns one of ``ok`` / ``flag`` / ``reject`` / ``single_source`` /
    ``no_data``.
    """
    if secondary is None or secondary.empty or closes.empty:
        return "single_source", None

    rows = secondary[secondary[SYMBOL] == symbol]
    if rows.empty:
        return "single_source", None

    target = pd.Timestamp(as_of).normalize()
    same_day = rows[rows[DATE] == target]
    if same_day.empty:
        return "single_source", {"note": "no secondary row for the decision date"}

    secondary_close = float(same_day["close"].iloc[-1])
    primary_close = float(closes.iloc[-1])
    if not np.isfinite(secondary_close) or secondary_close <= 0 or primary_close <= 0:
        return "single_source", None

    diff_bps = abs(primary_close / secondary_close - 1.0) * 10_000.0
    tolerance = float(params.source_agreement_tolerance_bps.value)

    detail = {
        "primary_close": round(primary_close, 4),
        "secondary_close": round(secondary_close, 4),
        "difference_bps": round(diff_bps, 1),
        "tolerance_bps": tolerance,
    }
    if diff_bps <= tolerance:
        return "ok", detail

    action = str(params.source_disagreement_action.value).strip().lower()
    return ("reject" if action == "reject" else "flag"), detail


# =============================================================================
# point-in-time audit
# =============================================================================


def _pit_audit(
    manifest: RawDataManifest, universe: UniverseSnapshot, params
) -> Tuple[Dict[str, bool], List[str]]:
    """Record which point-in-time guarantees actually held for this run.

    A guarantee that cannot be verified is recorded as ``False`` with an
    explicit note. It is never recorded as ``True`` on the grounds that nothing
    contradicted it -- absence of evidence is exactly what survivorship bias
    looks like from the inside.
    """
    switches = params.pit_audit
    audit: Dict[str, bool] = {}
    failures: List[str] = []

    if switches.enforce_historical_membership:
        ok = not manifest.survivorship_risk
        audit["historical_membership"] = ok
        if not ok:
            note = manifest.survivorship_note or universe.note or "no dated snapshot"
            failures.append(
                f"historical index membership could not be established for "
                f"{manifest.as_of_date}: {note}. Results carry survivorship bias "
                f"of unknown size."
            )

    if switches.enforce_delisted_inclusion:
        # The universe is reconstructed from dated snapshots, which contain only
        # names that were live at snapshot time. Delisted names are absent by
        # construction, so this cannot be verified from the current data.
        ok = not manifest.survivorship_risk
        audit["delisted_inclusion"] = ok
        if not ok:
            failures.append(
                "delisted-name inclusion is unverified: the universe was "
                "reconstructed from a snapshot that by construction contains "
                "only surviving names."
            )

    if switches.enforce_fundamentals_filing_date:
        record = manifest.feeds.get("fundamentals")
        ok = bool(record and record.row_count > 0)
        audit["fundamentals_filing_date"] = ok
        if not ok:
            failures.append(
                "point-in-time fundamentals are absent, so filing-date "
                "alignment cannot be enforced. The Stage 4 quality factor is "
                "dropped rather than computed from current-vintage data."
            )

    if switches.enforce_pledging_disclosure_date:
        record = manifest.feeds.get("pledging")
        ok = bool(record and record.row_count > 0)
        audit["pledging_disclosure_date"] = ok
        if not ok:
            failures.append(
                "promoter-pledging data is absent, so disclosure-date alignment "
                "cannot be enforced. The Stage 3 pledging gate reports "
                "NOT_TESTABLE -- it does not pass."
            )

    if switches.enforce_historical_sector:
        # Sector labels come from the current constituent file. Reclassification
        # is rare but real, and pretending otherwise would be a silent lie.
        ok = not manifest.survivorship_risk
        audit["historical_sector"] = ok
        if not ok:
            failures.append(
                "sector labels are current-vintage, not as-of-date. Sector-"
                "relative strength and sector caps inherit that approximation."
            )

    if switches.forbid_forward_fill_across_sessions:
        # The store never forward-fills; the ingest writes only sessions that
        # actually published. This records the guarantee explicitly so the
        # audit trail carries it rather than relying on it being remembered.
        audit["no_forward_fill"] = True

    return audit, failures


# =============================================================================
# helpers
# =============================================================================


def _required_window(params) -> int:
    """Longest history any Stage 1 check needs, plus a small buffer."""
    return (
        max(
            int(params.continuity_window_sessions.value),
            int(params.outlier_sigma_lookback_sessions.value),
        )
        + 5
    )


def _window_start(
    calendar: TradingCalendar, as_of: dt.date, sessions: int
) -> dt.date:
    window = calendar.trailing_window(as_of, sessions)
    if window:
        return window[0]
    return calendar.first


def _expected_sessions(
    calendar: TradingCalendar, as_of: dt.date, params
) -> List[pd.Timestamp]:
    """Sessions the calendar says should exist inside the continuity window."""
    window = calendar.trailing_window(
        as_of, int(params.continuity_window_sessions.value)
    )
    return [pd.Timestamp(d).normalize() for d in window]


def _read_secondary(
    store: DataStore, start: dt.date, end: dt.date
) -> Optional[pd.DataFrame]:
    """Load the secondary price table, or ``None`` when it was never ingested."""
    frame = store.read_table("prices_secondary")
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    out[DATE] = pd.to_datetime(out[DATE]).dt.normalize()
    mask = (out[DATE] >= pd.Timestamp(start)) & (out[DATE] <= pd.Timestamp(end))
    return out.loc[mask]


def _suspect_map(suspects: pd.DataFrame) -> Dict[str, List[dt.date]]:
    """Group unexplained-action hits by symbol."""
    out: Dict[str, List[dt.date]] = {}
    if suspects is None or suspects.empty:
        return out
    for symbol, rows in suspects.groupby(SYMBOL):
        out[str(symbol)] = [
            pd.Timestamp(d).date() for d in rows[DATE].tolist()
        ]
    return out
