"""Stage 3 -- Eligibility. Hard gates only, binary, before any scoring.

A stock excluded here cannot be rescued by a high score later. Eligibility asks
whether the stock may be traded at all, which is prior to whether it is
attractive.

A gate whose data is absent reports NOT_TESTABLE and does not reject the stock,
but the untestable gate is recorded and printed on the card. Treating "could
not check" as "passed" is the failure mode this ordering exists to prevent.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ._cfg import bv, fv, iv, v
from ..core.calendar import TradingCalendar
from ..core.contracts import DataQualityReport, EligibilityReport
from ..indicators.circuit import band_state, is_untradeable
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
    held: Optional[Sequence[str]] = None,
) -> EligibilityReport:
    p = config.params
    cfg = p.stage3_eligibility
    as_of = as_of or calendar.last

    symbols = [normalise_symbol(s) for s in universe.symbols]
    # The book Stage 6 was asked to maintain. Only the model-domain filter
    # consults it: every other gate here is about whether the name may be
    # TRADED at all, which applies to a sale as much as to a purchase.
    open_book = {normalise_symbol(s) for s in (held or ())}
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
    grouped = {s: f for s, f in prices.groupby(SYMBOL, sort=False, observed=True)}

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

        # A name locked at its price band on the decision date cannot be
        # relied on to fill at the next session's open, which is the execution
        # assumption every downstream price in this run rests on. The band is
        # not in the feed, so this is inferred from the bar: one price all
        # session means one price was available. Rejecting rather than flagging
        # because the alternative is issuing a plan whose entry may be
        # unreachable.
        last_bar = frame.iloc[-1]
        # read_prices does not carry prev_close, so the band label is taken from
        # the prior bar in this frame. Its absence only costs the label; the
        # frozen fact comes from high == low on the bar itself.
        prior_close = float(frame["close"].iloc[-2]) if len(frame) > 1 else float("nan")
        state = band_state(
            float(last_bar.get("high", float("nan"))),
            float(last_bar.get("low", float("nan"))),
            float(last_bar.get("close", float("nan"))),
            prior_close,
            float(last_bar.get("volume", float("nan"))),
        )
        if is_untradeable(state):
            rejected[sym] = RejectionReason.ILLIQUID
            details[sym] = (
                f"session closed {state.value}: the bar offered a single price, "
                f"so an entry at the next open cannot be assumed"
            )
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

        # 5b. THE MODEL'S OWN DOMAIN.
        #
        # The coefficients are estimated on a panel that EXCLUDES names below
        # their thesis-invalidation level: `resolve_exits` returns NaN for them
        # and `build_panel` drops non-finite labels. So the fit describes
        # pullbacks WITHIN uptrends -- it has never seen a name in a decline.
        #
        # Scoring one anyway extrapolates. `reversal` carries a negative
        # coefficient, so applied outside its fitted range it ranks the
        # BIGGEST fallers highest: on 2026-08-25 the five highest-ranked names
        # were all below their invalidation level, and the engine bought two of
        # them.
        #
        # This is done HERE, before the ranking, rather than as a veto after it.
        # The score is a cross-sectional rank, so a universe containing names
        # the engine cannot buy distorts every percentile, the dispersion floor
        # and `min_universe_percentile` along with it. Filtering after ranking
        # leaves the top of the list unbuyable and silently promotes the sixth
        # name to first while the statistics still describe the old population.
        # HELD NAMES ARE EXEMPT, for the reason Stage 6 states: an entry
        # constraint must never close -- or fail to close -- a position that is
        # already open. Removing a held name from the universe here is worse
        # than admitting one: it reaches neither Stage 6's exit band nor Stage
        # 7's exit hierarchy, so the orphan review reports "hold, trading
        # normally" precisely when the position has met its first exit
        # condition. The engine stops SELLING rather than stops buying.
        if (sym not in open_book
                and bv(p.stage6_entry.admission.require_above_invalidation)):
            note = _outside_model_domain(frame, p)
            if note:
                rejected[sym] = RejectionReason.OUTSIDE_MODEL_DOMAIN
                details[sym] = note
                continue

        # 6. Manual exclusion
        if sym in set(v(p.universe.manual_exclusions) or []):
            rejected[sym] = RejectionReason.MANUAL_EXCLUSION
            details[sym] = "listed in universe.manual_exclusions"
            continue

        # 7. Earnings proximity -- hard reject inside the window
        #
        # A NAME WITH NO FORWARD DATE ON FILE IS NOT A NAME WITH NO EARNINGS.
        # This recorded NOT_TESTABLE only when the WHOLE calendar was missing;
        # a symbol absent from a present calendar produced `days is None`, fell
        # through, and was admitted with nothing recorded -- which is precisely
        # the failure this stage's contract says it exists to prevent.
        #
        # Coverage makes that the common case, not the edge one. Measured on
        # 2026-08-25: the calendar holds 8,776 rows over 2,342 symbols, but
        # only 181 carry any future date at all and just 7 of those are
        # company-confirmed. So roughly one name in seven was tested and six in
        # seven silently passed, while the card claimed no untestable gate.
        # Worse, calendar coverage tracks size and index membership, so the
        # residual was a cross-sectional tilt TOWARD names with poorer
        # reference data -- the opposite of what the gate intends.
        if earnings is None:
            untestable.append("earnings_proximity")
        else:
            days = earnings.get(sym)
            win = iv(cfg.earnings_proximity.holding_window_sessions)
            if days is None:
                untestable.append("earnings_proximity")
            elif 0 <= days <= win:
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


def _outside_model_domain(frame: pd.DataFrame, params) -> Optional[str]:
    """Why this name is outside the fitted population, or None if it is inside.

    Reads the level through `exits.tradeable_at_entry` -- the SAME predicate the
    label uses to decide which rows exist -- so the universe the model ranks and
    the panel it was fitted on cannot drift apart.
    """
    from ..features.exits import (ExitRules, invalidation_level,
                                  tradeable_at_entry)
    from ..indicators import atr as atr_fn

    c7 = params.stage7_risk
    n = iv(c7.thesis_invalidation.structure_ma_sessions)
    if len(frame) < n:
        return (f"{len(frame)} sessions, fewer than the {n} needed for the "
                f"invalidation level. A level that cannot be computed has not "
                f"been cleared.")
    rules = ExitRules(
        invalidation_ma_sessions=n,
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        atr_period_sessions=iv(c7.atr.period_sessions),
        atr_method=str(v(c7.atr.method)),
    )
    a = atr_fn(frame["high"], frame["low"], frame["close"],
               rules.atr_period_sessions, rules.atr_method).dropna()
    if a.empty:
        return "ATR is not computable, so the invalidation level cannot be checked."
    closes = pd.to_numeric(frame["close"], errors="coerce")
    ma_now = float(closes.tail(n).mean())
    atr_now = float(a.iloc[-1])
    last = float(closes.iloc[-1])
    if bool(tradeable_at_entry(last, ma_now, atr_now, rules)):
        return None
    level = invalidation_level(ma_now, atr_now, rules)
    return (
        f"Rs {last:,.2f} is below the thesis-invalidation level of "
        f"Rs {level:,.2f} ({n}-session average less "
        f"{rules.invalidation_buffer_atr:g} ATR). The model's coefficients were "
        f"fitted on a panel that excludes this state, so ranking it would "
        f"extrapolate them."
    )


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
    # observed=True: min() over an empty group raises, which is what the default
    # produces for every non-reporting name once SYMBOL is categorical.
    for sym, rows in future.groupby(SYMBOL, observed=True):
        nxt = min(rows["earnings_date"])
        out[str(sym)] = calendar.sessions_until(as_of, nxt)
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
    for sym, rows in past.groupby(SYMBOL, observed=True):
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
