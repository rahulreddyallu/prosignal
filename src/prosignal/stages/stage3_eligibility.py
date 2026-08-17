"""Stage 3 -- Eligibility. Hard gates only, all binary, all before any scoring.

Order is the point: a stock excluded here can never be rescued by a high score
later. Eligibility asks "may we trade this at all", which is a different and
prior question to "is this attractive".

A gate whose data is absent reports NOT_TESTABLE and the stock is NOT rejected
on that basis -- but the untestable gate is recorded and printed on the card.
Silently upgrading "we could not check" to "passed" is the single most dangerous
thing an eligibility layer can do.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd

from ._cfg import bv, fv, iv, v
from ..core.calendar import TradingCalendar
from ..core.contracts import DataQualityReport, EligibilityReport
from ..core.enums import RejectionReason
from ..core.logging import get_logger
from ..data.store import DataStore
from ..data.types import DATE, SYMBOL, normalise_symbol
from ..data.universe import UniverseSnapshot

__all__ = ["run", "STAGE_NAME"]

STAGE_NAME = "stage3_eligibility"
log = get_logger(__name__)


def run(
    universe: UniverseSnapshot,
    store: DataStore,
    calendar: TradingCalendar,
    quality: DataQualityReport,
    config,
    as_of: Optional[dt.date] = None,
) -> EligibilityReport:
    p = config.params
    cfg = p.stage3_eligibility
    as_of = as_of or calendar.last

    symbols = [normalise_symbol(s) for s in universe.symbols]
    rejected: Dict[str, RejectionReason] = {}
    details: Dict[str, str] = {}
    not_testable: Dict[str, List[str]] = {}

    position_value = float(p.capital.position_value_inr())

    # -- window covering every lookback this stage needs --------------------
    need = max(
        iv(cfg.liquidity.adtv_lookback_sessions),
        iv(p.universe.min_history_sessions),
    ) + 5
    window = calendar.trailing_window(as_of, need)
    start = window[0] if window else calendar.first

    prices = store.read_prices(symbols=symbols, start=start, end=as_of)
    if prices.empty:
        return EligibilityReport(
            as_of_date=as_of, universe_considered=len(symbols),
            eligible_universe=[], rejected={s: RejectionReason.INSUFFICIENT_HISTORY for s in symbols},
            rejection_details={s: "no price rows" for s in symbols},
            sector_map=dict(universe.sector_map), position_value_inr=position_value,
        )

    prices = prices.copy()
    prices[DATE] = pd.to_datetime(prices[DATE]).dt.normalize()
    grouped = {s: f for s, f in prices.groupby(SYMBOL, sort=False)}

    earnings = _earnings_map(store, calendar, as_of, cfg)
    regulatory = _regulatory_map(store, calendar, as_of, cfg)
    pledging = store.read_pledging()
    pledging_available = pledging is not None and not pledging.empty

    adtv_map: Dict[str, float] = {}
    eligible: List[str] = []

    for sym in symbols:
        untestable: List[str] = []

        # 1. Stage 1 data quality -- already decided, just enforced here.
        if not quality.is_clean(sym):
            flags = quality.per_stock_flags.get(sym)
            failed = ", ".join(flags.failed_checks) if flags else "unknown"
            rejected[sym] = RejectionReason.DATA_QUALITY
            details[sym] = f"Stage 1 data-quality failure: {failed}"
            continue

        frame = grouped.get(sym)
        if frame is None or frame.empty:
            rejected[sym] = RejectionReason.INSUFFICIENT_HISTORY
            details[sym] = "no price rows in window"
            continue

        # 2. History
        min_hist = iv(p.universe.min_history_sessions)
        if len(frame) < min_hist:
            rejected[sym] = RejectionReason.INSUFFICIENT_HISTORY
            details[sym] = (
                f"{len(frame)} sessions of history, need {min_hist}. "
                f"The 12-1 momentum factor alone requires 273."
            )
            continue

        # 3. Series -- cash equity only
        series_ok = _series_allowed(frame, p)
        if not series_ok:
            rejected[sym] = RejectionReason.SERIES_NOT_ALLOWED
            details[sym] = "instrument series is not in universe.allowed_series"
            continue

        # 4. Price floor
        last_close = float(frame["close"].iloc[-1])
        floor = fv(p.universe.min_price_inr)
        if last_close < floor:
            rejected[sym] = RejectionReason.PRICE_FLOOR
            details[sym] = f"close Rs {last_close:,.2f} below floor Rs {floor:,.2f}"
            continue

        # 5. Liquidity -- ADTV and participation
        adtv = _adtv_inr(frame, iv(cfg.liquidity.adtv_lookback_sessions))
        if adtv is not None:
            adtv_map[sym] = adtv
        min_adtv = fv(cfg.liquidity.min_adtv_inr)
        if adtv is None:
            rejected[sym] = RejectionReason.ILLIQUID
            details[sym] = "turnover unavailable; ADTV not computable"
            continue
        if adtv < min_adtv:
            rejected[sym] = RejectionReason.ILLIQUID
            details[sym] = (
                f"ADTV Rs {adtv/1e7:,.2f} Cr below floor Rs {min_adtv/1e7:,.2f} Cr"
            )
            continue

        zero_vol = int((frame["volume"].tail(21).fillna(0) <= 0).sum())
        if zero_vol >= iv(cfg.liquidity.reject_on_zero_volume_sessions):
            rejected[sym] = RejectionReason.ILLIQUID
            details[sym] = f"{zero_vol} zero-volume sessions in the last 21"
            continue

        if bv(cfg.liquidity.use_participation_gate):
            participation = position_value / adtv
            cap = fv(p.capital.max_participation_of_adtv)
            if participation > cap:
                rejected[sym] = RejectionReason.ILLIQUID
                details[sym] = (
                    f"a Rs {position_value:,.0f} position is {participation:.2%} of "
                    f"ADTV, above the {cap:.2%} cap. Executable size here would "
                    f"move the price against us by more than the modelled edge."
                )
                continue

        # 6. Manual exclusion
        if sym in set(v(p.universe.manual_exclusions) or []):
            rejected[sym] = RejectionReason.MANUAL_EXCLUSION
            details[sym] = "listed in universe.manual_exclusions"
            continue

        # 7. Earnings proximity -- hard reject inside the window
        if earnings is None:
            untestable.append("earnings_proximity")
        else:
            days = earnings.get(sym)
            win = iv(cfg.earnings_proximity.holding_window_sessions)
            if days is not None and 0 <= days <= win:
                rejected[sym] = RejectionReason.EARNINGS_CONFLICT
                details[sym] = (
                    f"results due in ~{days} sessions, inside the {win}-session "
                    f"holding window. Entering ahead of a scheduled event the "
                    f"model does not price is an unmodelled bet."
                )
                continue

        # 8. Regulatory cooldown
        if regulatory is None:
            untestable.append("regulatory_cooldown")
        else:
            since = regulatory.get(sym)
            cd = iv(cfg.regulatory_cooldown.default_cooldown_sessions)
            if since is not None and since <= cd:
                rejected[sym] = RejectionReason.REGULATORY_COOLDOWN
                details[sym] = f"regulatory event {since} sessions ago, cooldown {cd}"
                continue

        # 9. Pledging -- NOT_TESTABLE when absent, never a pass
        if not pledging_available:
            untestable.append("promoter_pledging")
        else:
            pl = _pledged_pct(pledging, sym, as_of)
            if pl is None:
                untestable.append("promoter_pledging")
            elif pl > fv(cfg.pledging.max_pledged_pct_of_promoter_holding):
                rejected[sym] = RejectionReason.PLEDGING_BREACH
                details[sym] = f"promoter pledging {pl:.1f}% above cap"
                continue

        if untestable:
            not_testable[sym] = untestable
        eligible.append(sym)

    log.info(
        "stage 3 complete",
        extra={"considered": len(symbols), "eligible": len(eligible),
               "rejected": len(rejected)},
    )
    return EligibilityReport(
        as_of_date=as_of,
        universe_considered=len(symbols),
        eligible_universe=eligible,
        rejected=rejected,
        rejection_details=details,
        not_testable=not_testable,
        sector_map={s: universe.sector_of(s) for s in symbols},
        adtv_inr=adtv_map,
        position_value_inr=position_value,
    )


# =============================================================================
# helpers
# =============================================================================


def _series_allowed(frame: pd.DataFrame, params) -> bool:
    allowed = {str(s).upper() for s in (v(params.universe.allowed_series) or ["EQ"])}
    if "series" not in frame.columns:
        return True
    seen = {str(s).upper() for s in frame["series"].dropna().unique()}
    return bool(seen & allowed)


def _adtv_inr(frame: pd.DataFrame, lookback: int) -> Optional[float]:
    """Average daily traded VALUE in rupees over the lookback.

    Uses exchange-reported turnover when present. Falls back to close x volume,
    which is a proxy rather than the real thing -- it ignores intraday price
    variation, so it is close but not identical to true traded value.
    """
    tail = frame.tail(lookback)
    if "turnover" in tail.columns:
        turnover = pd.to_numeric(tail["turnover"], errors="coerce").dropna()
        turnover = turnover[turnover > 0]
        if len(turnover) >= max(lookback // 2, 5):
            return float(turnover.mean())
    close = pd.to_numeric(tail.get("close"), errors="coerce")
    volume = pd.to_numeric(tail.get("volume"), errors="coerce")
    proxy = (close * volume).dropna()
    proxy = proxy[proxy > 0]
    if proxy.empty:
        return None
    return float(proxy.mean())


def _earnings_map(store, calendar, as_of, cfg) -> Optional[Dict[str, int]]:
    """Sessions until the next scheduled results date, per symbol."""
    frame = store.read_earnings_calendar()
    if frame is None or frame.empty:
        return None
    f = frame.copy()
    f["earnings_date"] = pd.to_datetime(f["earnings_date"], errors="coerce").dt.date
    f = f.dropna(subset=["earnings_date"])
    future = f[f["earnings_date"] >= as_of]
    out: Dict[str, int] = {}
    for sym, rows in future.groupby(SYMBOL):
        nxt = min(rows["earnings_date"])
        out[str(sym)] = calendar.count_between(as_of, nxt)
    return out


def _regulatory_map(store, calendar, as_of, cfg) -> Optional[Dict[str, int]]:
    frame = store.read_regulatory_events()
    if frame is None or frame.empty:
        return None
    f = frame.copy()
    col = "event_date" if "event_date" in f.columns else DATE
    f[col] = pd.to_datetime(f[col], errors="coerce").dt.date
    f = f.dropna(subset=[col])
    past = f[f[col] <= as_of]
    out: Dict[str, int] = {}
    for sym, rows in past.groupby(SYMBOL):
        last = max(rows[col])
        out[str(sym)] = calendar.count_between(last, as_of)
    return out


def _pledged_pct(pledging: pd.DataFrame, symbol: str, as_of: dt.date) -> Optional[float]:
    rows = pledging[pledging[SYMBOL] == symbol]
    if rows.empty or "pledged_pct_of_promoter_holding" not in rows.columns:
        return None
    if "as_of_date" in rows.columns:
        rows = rows.copy()
        rows["as_of_date"] = pd.to_datetime(rows["as_of_date"], errors="coerce").dt.date
        rows = rows[rows["as_of_date"] <= as_of]
        if rows.empty:
            return None
        rows = rows.sort_values("as_of_date")
    value = pd.to_numeric(rows["pledged_pct_of_promoter_holding"], errors="coerce").dropna()
    return float(value.iloc[-1]) if not value.empty else None
