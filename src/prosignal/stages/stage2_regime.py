"""Stage 2 -- Market Regime Engine.

Momentum does not merely weaken in a crash, it inverts, and it does so when a
naive screen is most confident (Daniel & Moskowitz, 2016). Regime therefore
scales the factors rather than filtering their output.

Four independent reads, kept separate so one broken input cannot swing the
whole conclusion:

Trend -- regression slope on log price and position against the 200-DMA. Both
must agree before a directional call is made; price above a falling 200-DMA, or
a rising slope beneath one, reads as Range-bound.

Volatility -- India VIX as a percentile of its own trailing year, not an
absolute level: it has printed between roughly 8 and 87, so a fixed threshold
would misread whole years. The rise/fall split follows Thenmozhi & Chandra
(2013); a VIX rising into a falling market differs from one rising into a
rally, and a drifting VIX is mostly complacency (G.C. & Kothari, 2016). The
asymmetry is carried in ``vol_signal_confidence``.

Breadth -- participation, market-level only. It never reaches a stock-level
score, since "most stocks are weak" says nothing about a particular stock.

Transition -- whether the regime is currently changing, compared against the
read N sessions ago. Historical relationships are least reliable in this state.

Output is three multipliers (momentum, quality, sector RS) from a table in
parameters.yaml. The direction of that table follows published work; the
magnitudes are tagged UNVALIDATED until CPCV promotes them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..core.calendar import TradingCalendar
from ..core.contracts import RegimeState
from ..core.enums import BreadthState, TrendRegime, VolContext, VolTercile
from ..core.errors import DataError
from ..core.logging import get_logger
from ..data.providers.nse_archives import INDIA_VIX_NAME
from ..data.store import DataStore
from ..data.types import DATE, SYMBOL, normalise_symbol
from ..indicators import (
    annualised_log_slope,
    percentile_of_last,
    rate_of_change_pct,
    sma,
)

__all__ = ["run", "STAGE_NAME", "TrendRead", "VolRead", "BreadthRead"]

STAGE_NAME = "stage2_regime"

log = get_logger(__name__)

#: Bucket key fragments. The multiplier table in parameters.yaml is keyed by
#: ``f"{trend}_{vol}"`` built from these, so they are defined once.
_TREND_KEY = {
    TrendRegime.UPTREND: "uptrend",
    TrendRegime.RANGE_BOUND: "range",
    TrendRegime.DOWNTREND: "downtrend",
}
_VOL_KEY = {
    VolTercile.LOW: "lowvol",
    VolTercile.MEDIUM: "midvol",
    VolTercile.HIGH: "highvol",
}

#: The Daniel & Moskowitz momentum-crash state, which is its own regime rather
#: than a variant of the trend read.
CRASH_BUCKET = "uptrend_highvol_rebound"


# =============================================================================
# component reads
# =============================================================================


@dataclass(frozen=True)
class TrendRead:
    regime: TrendRegime
    slope_annualised: Optional[float]
    vs_fast_ma_pct: Optional[float]
    vs_slow_ma_pct: Optional[float]
    note: Optional[str] = None

    @property
    def key(self) -> str:
        return _TREND_KEY[self.regime]


@dataclass(frozen=True)
class VolRead:
    tercile: VolTercile
    context: VolContext
    level: Optional[float]
    percentile: Optional[float]
    change_pct: Optional[float]
    confidence: float
    note: Optional[str] = None

    @property
    def key(self) -> str:
        return _VOL_KEY[self.tercile]


@dataclass(frozen=True)
class BreadthRead:
    state: BreadthState
    pct_above_ma: Optional[float]
    divergence: bool
    sample_size: int
    note: Optional[str] = None


# =============================================================================
# entry point
# =============================================================================


def run(
    store: DataStore,
    calendar: TradingCalendar,
    eligible_symbols: Sequence[str],
    config,
    as_of: Optional[dt.date] = None,
) -> RegimeState:
    """Compute the market regime for ``as_of``.

    ``eligible_symbols`` drives breadth only. Passing the post-Stage-3 universe
    keeps breadth measured over names the engine would actually consider, which
    is the population the question is about.
    """
    params = config.params.stage2_regime
    as_of = as_of or calendar.last

    benchmark = str(params.benchmark_index.value)
    index_series = store.index_series(benchmark, "close", end=as_of)
    if index_series.empty:
        raise DataError(
            f"no data for benchmark index {benchmark!r} on or before {as_of}. "
            f"Stage 2 cannot form a regime view without it.",
            index=benchmark,
            as_of=as_of.isoformat(),
        )
    vix_series = store.index_series(INDIA_VIX_NAME, "close", end=as_of)

    symbols = [normalise_symbol(s) for s in eligible_symbols]
    closes_wide = _load_breadth_frame(store, symbols, calendar, params, as_of)

    notes: List[str] = []

    # -- the three component reads, at the decision date --------------------
    trend = _trend_read(index_series, as_of, params.trend)
    vol = _vol_read(vix_series, index_series, as_of, params.volatility)
    breadth = _breadth_read(closes_wide, as_of, params.breadth)

    for read in (trend, vol, breadth):
        if read.note:
            notes.append(read.note)

    # -- transition: the same three reads, N sessions ago -------------------
    transition_flag, disagreeing = _detect_transition(
        index_series=index_series,
        vix_series=vix_series,
        closes_wide=closes_wide,
        calendar=calendar,
        as_of=as_of,
        params=params,
        current=(trend, vol, breadth),
    )

    # -- bucket, including the momentum-crash override ----------------------
    bucket, crash_fired = _resolve_bucket(
        trend=trend,
        vol=vol,
        index_series=index_series,
        as_of=as_of,
        crash_cfg=config.params.stage5_false_signal.momentum_crash,
        table=params.multipliers.table,
    )
    if crash_fired:
        notes.append(
            "Daniel & Moskowitz momentum-crash signature is active: a prior "
            "market decline, elevated volatility, and a sharp rebound in "
            "progress. This is the state in which momentum historically "
            "inverts hardest, so new entries are blocked rather than merely "
            "dampened."
        )

    # -- multipliers ---------------------------------------------------------
    momentum, quality, sector_rs = _lookup_multipliers(bucket, params.multipliers.table)

    if breadth.state is BreadthState.WEAK:
        penalty = float(params.multipliers.weak_breadth_momentum_penalty.value)
        momentum *= penalty
        notes.append(
            f"Breadth is weak ({_fmt_pct(breadth.pct_above_ma)} of the eligible "
            f"universe above its {int(params.breadth.ma_sessions.value)}-session "
            f"average), so the momentum multiplier is cut by {1 - penalty:.0%}. "
            f"A narrow advance is a worse base for a momentum signal than the "
            f"index level alone suggests."
        )

    dampener_applied = 1.0
    if transition_flag:
        dampener_applied = float(params.transition.dampener.value)
        # The dampener hits the trend-following factors, which are the ones
        # that break down when a regime turns. Quality is deliberately exempt:
        # its documented job is crash stabilisation (Asness, Frazzini &
        # Pedersen), so dampening it during a transition would remove the
        # exposure most likely to help. Tagged UNVALIDATED like the rest of the
        # multiplier design.
        momentum *= dampener_applied
        sector_rs *= dampener_applied
        notes.append(
            f"Regime transition detected: {', '.join(disagreeing)} "
            f"{'disagree' if len(disagreeing) > 1 else 'disagrees'} with the "
            f"read from {int(params.transition.lookback_sessions.value)} "
            f"sessions ago. Momentum and sector-RS multipliers dampened to "
            f"{dampener_applied:.0%}; quality left intact as the stabiliser."
        )

    if breadth.divergence:
        notes.append(
            "Breadth divergence: the index made a new high for the lookback "
            "window while participation fell. Historically a late-cycle "
            "signature -- the advance is narrowing."
        )

    # -- entry gate ----------------------------------------------------------
    no_entry_buckets = set(params.no_new_entry_buckets.value)
    allow_entries = bucket not in no_entry_buckets
    block_reason = None
    if not allow_entries:
        block_reason = (
            f"regime bucket '{bucket}' is on the no-new-entry list. This is a "
            f"hard market-wide gate, not a score penalty."
        )

    state = RegimeState(
        as_of_date=as_of,
        trend_regime=trend.regime,
        trend_slope_annualised=trend.slope_annualised,
        index_vs_fast_ma_pct=trend.vs_fast_ma_pct,
        index_vs_slow_ma_pct=trend.vs_slow_ma_pct,
        vol_tercile=vol.tercile,
        vol_context=vol.context,
        vix_level=vol.level,
        vix_percentile=vol.percentile,
        vix_change_pct=vol.change_pct,
        vol_signal_confidence=vol.confidence,
        breadth_pct_above_ma=breadth.pct_above_ma,
        breadth_state=breadth.state,
        breadth_divergence_flag=breadth.divergence,
        breadth_sample_size=breadth.sample_size,
        regime_bucket=bucket,
        transition_flag=transition_flag,
        transition_components=disagreeing,
        momentum_multiplier=round(momentum, 4),
        quality_multiplier=round(quality, 4),
        sector_rs_multiplier=round(sector_rs, 4),
        dampener_applied=dampener_applied,
        allow_new_entries=allow_entries,
        block_reason=block_reason,
        notes=notes,
    )

    log.info(
        "stage 2 complete",
        extra={
            "as_of": as_of.isoformat(),
            "bucket": bucket,
            "trend": trend.regime.value,
            "vol": vol.tercile.value,
            "vol_context": vol.context.value,
            "breadth_pct": breadth.pct_above_ma,
            "transition": transition_flag,
            "allow_entries": allow_entries,
        },
    )
    return state


# =============================================================================
# trend
# =============================================================================


def _trend_read(index_series: pd.Series, as_of: dt.date, cfg) -> TrendRead:
    """Direction from slope AND moving-average position, which must agree.

    Requiring agreement is what stops this flapping. Each measure alone
    produces a different wrong answer: the MA test calls a dead-flat market
    trending whenever price sits a hair above the line, and the slope alone
    calls every two-month bounce inside a bear market an uptrend.
    """
    series = index_series[index_series.index <= pd.Timestamp(as_of)].dropna()
    if series.empty:
        return TrendRead(
            TrendRegime.RANGE_BOUND, None, None, None,
            note="No benchmark history at the decision date; trend defaulted to Range-bound.",
        )

    fast_n = int(cfg.fast_ma_sessions.value)
    slow_n = int(cfg.slow_ma_sessions.value)
    slope_n = int(cfg.slope_lookback_sessions.value)
    flat_band = float(cfg.slope_flat_band_annualised.value)

    last = float(series.iloc[-1])
    fast_ma = sma(series, fast_n)
    slow_ma = sma(series, slow_n)

    vs_fast = _pct_above(last, fast_ma)
    vs_slow = _pct_above(last, slow_ma)
    slope = annualised_log_slope(series, window=slope_n)

    if slope is None or vs_slow is None:
        missing = "slope" if slope is None else "200-session average"
        return TrendRead(
            TrendRegime.RANGE_BOUND, slope, vs_fast, vs_slow,
            note=(
                f"Insufficient benchmark history to compute the {missing} "
                f"({series.size} sessions available). Trend reported as "
                f"Range-bound rather than guessed."
            ),
        )

    above_slow = vs_slow > 0

    if slope > flat_band and above_slow:
        return TrendRead(TrendRegime.UPTREND, slope, vs_fast, vs_slow)
    if slope < -flat_band and not above_slow:
        return TrendRead(TrendRegime.DOWNTREND, slope, vs_fast, vs_slow)

    # Everything else is genuinely ambiguous: a rising slope under a falling
    # 200-DMA is a bounce, a falling slope above it is a pullback, and a slope
    # inside the flat band is no trend at all.
    note = None
    if abs(slope) > flat_band:
        note = (
            f"Trend slope ({slope:+.1%} annualised) and the "
            f"{slow_n}-session average disagree "
            f"(index {vs_slow:+.1f}% vs it), so the regime is reported as "
            f"Range-bound rather than resolved in favour of either."
        )
    return TrendRead(TrendRegime.RANGE_BOUND, slope, vs_fast, vs_slow, note=note)


# =============================================================================
# volatility
# =============================================================================


def _vol_read(
    vix_series: pd.Series, index_series: pd.Series, as_of: dt.date, cfg
) -> VolRead:
    """India VIX tercile plus the asymmetric rising/falling split."""
    weights = cfg.asymmetric_confidence
    vix = vix_series[vix_series.index <= pd.Timestamp(as_of)].dropna()

    if vix.empty:
        return VolRead(
            VolTercile.MEDIUM, VolContext.STABLE, None, None, None,
            confidence=float(weights.falling_vix_weight),
            note=(
                "India VIX unavailable at the decision date. Volatility "
                "reported as Medium/stable with reduced confidence -- this is "
                "an absence of evidence, not evidence of calm."
            ),
        )

    lookback = int(cfg.vix_percentile_lookback_sessions.value)
    level = float(vix.iloc[-1])
    percentile = percentile_of_last(vix, lookback)

    if percentile is None:
        return VolRead(
            VolTercile.MEDIUM, VolContext.STABLE, level, None, None,
            confidence=float(weights.falling_vix_weight),
            note=(
                f"India VIX has only {vix.size} sessions of history, short of "
                f"the {lookback} needed for a percentile. Tercile reported as "
                f"Medium rather than read off a partial window."
            ),
        )

    low_cut = float(cfg.low_tercile_pct.value)
    high_cut = float(cfg.high_tercile_pct.value)
    if percentile <= low_cut:
        tercile = VolTercile.LOW
    elif percentile >= high_cut:
        tercile = VolTercile.HIGH
    else:
        tercile = VolTercile.MEDIUM

    # -- the asymmetric split ------------------------------------------------
    roc_n = int(cfg.vix_roc_lookback_sessions.value)
    vix_change = rate_of_change_pct(vix, roc_n)
    index_change = rate_of_change_pct(
        index_series[index_series.index <= pd.Timestamp(as_of)].dropna(), roc_n
    )

    rising_cut = float(cfg.vix_rising_threshold_pct.value)
    move_cut = float(cfg.market_move_threshold_pct.value)

    context = VolContext.STABLE
    if vix_change is not None:
        if vix_change >= rising_cut:
            if index_change is not None and index_change <= -move_cut:
                context = VolContext.RISING_IN_DECLINE
            elif index_change is not None and index_change >= move_cut:
                context = VolContext.RISING_IN_RALLY
            else:
                context = VolContext.RISING_IN_DECLINE
        elif vix_change <= -rising_cut:
            context = VolContext.FALLING

    # A rising VIX is informative; a falling one is mostly complacency and has
    # a much weaker relationship with what follows. A stable VIX carries no
    # directional claim of its own, so the tercile level stands on its own
    # merits and keeps full weight.
    if context is VolContext.FALLING:
        confidence = float(weights.falling_vix_weight)
    else:
        confidence = float(weights.rising_vix_weight)

    note = None
    if context is VolContext.RISING_IN_RALLY:
        note = (
            f"India VIX rose {vix_change:+.1f}% while the index gained "
            f"{index_change:+.1f}%. Volatility rising into strength is unusual "
            f"and historically sits closer to tops than to bottoms."
        )

    return VolRead(
        tercile=tercile,
        context=context,
        level=level,
        percentile=percentile,
        change_pct=vix_change,
        confidence=confidence,
        note=note,
    )


# =============================================================================
# breadth
# =============================================================================


def _breadth_read(closes_wide: pd.DataFrame, as_of: dt.date, cfg) -> BreadthRead:
    """Percentage of the eligible universe above its own long-term average."""
    if closes_wide is None or closes_wide.empty:
        return BreadthRead(
            BreadthState.NEUTRAL, None, False, 0,
            note="No constituent price data available; breadth not measured.",
        )

    ma_n = int(cfg.ma_sessions.value)
    pct, sample = _pct_above_ma(closes_wide, ma_n, as_of)

    if pct is None:
        return BreadthRead(
            BreadthState.NEUTRAL, None, False, sample,
            note=(
                f"Fewer than {ma_n} sessions of history for any constituent, "
                f"so breadth could not be measured. Reported as Neutral rather "
                f"than assumed healthy."
            ),
        )

    weak_cut = float(cfg.weak_threshold_pct.value)
    strong_cut = float(cfg.strong_threshold_pct.value)
    if pct < weak_cut:
        state = BreadthState.WEAK
    elif pct > strong_cut:
        state = BreadthState.STRONG
    else:
        state = BreadthState.NEUTRAL

    divergence = _breadth_divergence(closes_wide, as_of, cfg, current_pct=pct)

    return BreadthRead(
        state=state, pct_above_ma=pct, divergence=divergence, sample_size=sample
    )


def _pct_above_ma(
    closes_wide: pd.DataFrame, ma_sessions: int, as_of: dt.date
) -> Tuple[Optional[float], int]:
    """Share of columns trading above their own moving average at ``as_of``."""
    frame = closes_wide[closes_wide.index <= pd.Timestamp(as_of)]
    if frame.empty or frame.shape[0] < ma_sessions:
        return None, 0

    averages = frame.rolling(window=ma_sessions, min_periods=ma_sessions).mean()
    last_price = frame.iloc[-1]
    last_ma = averages.iloc[-1]

    valid = last_price.notna() & last_ma.notna()
    sample = int(valid.sum())
    if sample == 0:
        return None, 0

    above = int((last_price[valid] > last_ma[valid]).sum())
    return 100.0 * above / sample, sample


def _breadth_divergence(
    closes_wide: pd.DataFrame, as_of: dt.date, cfg, current_pct: float
) -> bool:
    """Index at a new window high while participation has fallen.

    The classic late-cycle signature, and the one that mattered on the NSE
    through 2021-22: the index kept printing highs on the strength of a few
    heavyweights while the median stock had already rolled over. A cap-weighted
    index cannot show you that; only breadth can.
    """
    lookback = int(cfg.divergence_lookback_sessions.value)
    ma_n = int(cfg.ma_sessions.value)
    min_drop = float(cfg.divergence_min_breadth_drop_pct.value)

    frame = closes_wide[closes_wide.index <= pd.Timestamp(as_of)]
    if frame.shape[0] < ma_n + lookback:
        return False

    # Equal-weighted proxy for "the index", built from the same constituents so
    # the comparison is like-for-like.
    composite = frame.mean(axis=1, skipna=True).dropna()
    if composite.size < lookback + 1:
        return False

    window = composite.tail(lookback + 1)
    at_window_high = float(window.iloc[-1]) >= float(window.max())
    if not at_window_high:
        return False

    past_date = frame.index[-(lookback + 1)]
    past_pct, _ = _pct_above_ma(frame, ma_n, past_date.date())
    if past_pct is None:
        return False

    return (past_pct - current_pct) >= min_drop


# =============================================================================
# transition
# =============================================================================


def _detect_transition(
    index_series: pd.Series,
    vix_series: pd.Series,
    closes_wide: pd.DataFrame,
    calendar: TradingCalendar,
    as_of: dt.date,
    params,
    current: Tuple[TrendRead, VolRead, BreadthRead],
) -> Tuple[bool, List[str]]:
    """Compare today's read against the read N sessions ago.

    A regime *changing* is a different and more dangerous state than any
    particular regime, because every historical relationship the engine relies
    on is estimated over periods when the regime was stable.
    """
    lookback = int(params.transition.lookback_sessions.value)
    threshold = int(params.transition.min_components_disagreeing.value)

    past_date = calendar.previous_session(as_of, lookback)
    if past_date is None:
        return False, []

    trend_now, vol_now, breadth_now = current
    trend_then = _trend_read(index_series, past_date, params.trend)
    vol_then = _vol_read(vix_series, index_series, past_date, params.volatility)
    breadth_then = _breadth_read(closes_wide, past_date, params.breadth)

    disagreeing: List[str] = []
    if trend_now.regime is not trend_then.regime:
        disagreeing.append(
            f"trend ({trend_then.regime.value} -> {trend_now.regime.value})"
        )
    if vol_now.tercile is not vol_then.tercile:
        disagreeing.append(
            f"volatility ({vol_then.tercile.value} -> {vol_now.tercile.value})"
        )
    if breadth_now.state is not breadth_then.state:
        disagreeing.append(
            f"breadth ({breadth_then.state.value} -> {breadth_now.state.value})"
        )

    return len(disagreeing) >= threshold, disagreeing


# =============================================================================
# bucket and multipliers
# =============================================================================


def _resolve_bucket(
    trend: TrendRead,
    vol: VolRead,
    index_series: pd.Series,
    as_of: dt.date,
    crash_cfg,
    table: Dict[str, List[float]],
) -> Tuple[str, bool]:
    """``{trend}_{vol}``, unless the momentum-crash signature overrides it."""
    base = f"{trend.key}_{vol.key}"

    if not getattr(crash_cfg, "enabled", True):
        return base, False
    if vol.tercile is not VolTercile.HIGH:
        return base, False
    if CRASH_BUCKET not in table:
        return base, False

    if _crash_signature(index_series, as_of, crash_cfg):
        return CRASH_BUCKET, True
    return base, False


def _crash_signature(index_series: pd.Series, as_of: dt.date, cfg) -> bool:
    """Daniel & Moskowitz: prior decline, high volatility, sharp rebound.

    The momentum crash does not happen in the decline -- it happens in the
    violent rebound off the bottom, when the most beaten-down names rally
    hardest and everything a momentum screen owns underperforms at once. March
    to June 2020 on the NSE is the textbook instance.

    Volatility is checked by the caller; this function tests the two price
    conditions.
    """
    series = index_series[index_series.index <= pd.Timestamp(as_of)].dropna()

    decline_n = int(cfg.prior_decline_lookback_sessions.value)
    decline_cut = float(cfg.prior_decline_threshold_pct.value)
    rebound_n = int(cfg.rebound_lookback_sessions.value)
    rebound_cut = float(cfg.rebound_threshold_pct.value)

    if series.size < decline_n + 1:
        return False

    # The decline is measured to the START of the rebound window, so a rebound
    # that has already erased the drawdown does not mask the setup.
    anchor = series.iloc[: series.size - rebound_n] if series.size > rebound_n else series
    if anchor.size < decline_n + 1:
        return False

    prior_change = rate_of_change_pct(anchor, decline_n)
    rebound = rate_of_change_pct(series, rebound_n)
    if prior_change is None or rebound is None:
        return False

    return prior_change <= decline_cut and rebound >= rebound_cut


def _lookup_multipliers(
    bucket: str, table: Dict[str, List[float]]
) -> Tuple[float, float, float]:
    """Fetch ``[momentum, quality, sector_rs]`` for a bucket.

    A missing bucket is an error, not a default of 1.0. Silently applying
    neutral multipliers would mean the regime engine had no effect while
    appearing to work perfectly.
    """
    row = table.get(bucket)
    if row is None:
        raise DataError(
            f"regime bucket {bucket!r} has no row in "
            f"stage2_regime.multipliers.table. Known buckets: "
            f"{sorted(table)}",
            bucket=bucket,
        )
    return float(row[0]), float(row[1]), float(row[2])


# =============================================================================
# helpers
# =============================================================================


def _load_breadth_frame(
    store: DataStore,
    symbols: Sequence[str],
    calendar: TradingCalendar,
    params,
    as_of: dt.date,
) -> pd.DataFrame:
    """Wide close-price frame (dates x symbols) covering every breadth need."""
    if not symbols:
        return pd.DataFrame()

    needed = (
        int(params.breadth.ma_sessions.value)
        + int(params.breadth.divergence_lookback_sessions.value)
        + int(params.transition.lookback_sessions.value)
        + 5
    )
    window = calendar.trailing_window(as_of, needed)
    start = window[0] if window else calendar.first

    frame = store.read_prices(symbols=list(symbols), start=start, end=as_of)
    if frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame[DATE] = pd.to_datetime(frame[DATE]).dt.normalize()
    wide = frame.pivot_table(
        index=DATE, columns=SYMBOL, values="close", aggfunc="last", observed=True
    ).sort_index()
    return wide.astype("float64")


def _pct_above(last: float, moving_average: pd.Series) -> Optional[float]:
    """Percent distance of the last price above its moving average."""
    clean = moving_average.dropna()
    if clean.empty:
        return None
    ma = float(clean.iloc[-1])
    if ma <= 0:
        return None
    return (last / ma - 1.0) * 100.0


def _fmt_pct(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.0f}%"
