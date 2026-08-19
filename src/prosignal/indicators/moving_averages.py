"""Moving averages and distance-from-average measures.

pandas' ``ewm`` defaults to ``adjust=True``, which renormalises by the weights
available so far. That is a different estimator from the recursive EMA meant by
"50 EMA" in charting packages. The two converge only after roughly ``span``
observations, and a signal built on the difference fires at the wrong times in
the interim. This module uses ``adjust=False`` and emits nothing until the
window is full.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

__all__ = [
    "sma",
    "ema",
    "wilder_ma",
    "distance_from_ma_pct",
    "distance_from_ma_atr",
    "ma_slope_pct",
    "is_above",
    "golden_cross_state",
]

Numeric = Union[pd.Series, pd.DataFrame]


def sma(series: Numeric, window: int) -> Numeric:
    """Simple moving average. ``NaN`` until ``window`` real observations exist."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.astype("float64").rolling(window=window, min_periods=window).mean()


def ema(series: Numeric, span: int, min_periods: Optional[int] = None) -> Numeric:
    """Exponential moving average, recursive form (``adjust=False``).

    ``min_periods`` defaults to ``span``, so the series stays ``NaN`` until the
    average is seeded with a full window. That is deliberate: an EMA reported
    from 5 observations of a 200-session span is not an approximation of the
    200-EMA, it is essentially the last price.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    mp = span if min_periods is None else min_periods
    values = series.astype("float64")
    out = values.ewm(span=span, adjust=False, min_periods=mp).mean()
    return out


def wilder_ma(series: Numeric, period: int) -> Numeric:
    """Wilder's smoothing: an EMA with ``alpha = 1/period``.

    Wilder's indicators (RSI, ATR, ADX) all use this, and it is NOT the same as
    a standard EMA of the same period -- a Wilder MA of period ``n`` behaves
    like a conventional EMA of span ``2n - 1``. Mixing the two is why two
    charting packages can disagree about the "14-period ATR".
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.astype("float64").ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def distance_from_ma_pct(price: Numeric, moving_average: Numeric) -> Numeric:
    """Percent distance of price above (+) or below (-) its moving average."""
    ma = moving_average.astype("float64")
    safe = ma.where(ma > 0)
    return (price.astype("float64") / safe - 1.0) * 100.0


def distance_from_ma_atr(
    price: Numeric, moving_average: Numeric, atr: Numeric
) -> Numeric:
    """Distance from the moving average measured in ATRs, not percent.

    This is the version worth using cross-sectionally. A 5% extension means
    something completely different for a low-beta FMCG name than for a smallcap
    that routinely moves 4% a session; normalising by the stock's own ATR makes
    "extended" comparable across the universe. Percent distance is what makes a
    naive screen fill up with high-volatility names every time.
    """
    ma = moving_average.astype("float64")
    atr_safe = atr.astype("float64")
    atr_safe = atr_safe.where(atr_safe > 0)
    return (price.astype("float64") - ma) / atr_safe


def ma_slope_pct(moving_average: Numeric, lookback: int) -> Numeric:
    """Percent change in the moving average over ``lookback`` sessions.

    A rising 200-DMA and a falling 200-DMA at the same price level are
    different regimes; the level alone does not say which one you are in.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    ma = moving_average.astype("float64")
    return ma.pct_change(periods=lookback, fill_method=None) * 100.0


def is_above(price: Numeric, moving_average: Numeric) -> pd.Series:
    """Boolean 'price above MA', with ``NaN`` inputs yielding ``False``.

    Returned as a plain bool Series so it can be summed for breadth directly.
    Unknown is treated as "not above" rather than propagating ``NaN``: a stock
    whose 200-DMA cannot yet be computed genuinely has no evidence of being
    above it, and counting it as participating would overstate breadth.
    """
    p = price.astype("float64")
    ma = moving_average.astype("float64")
    result = (p > ma) & p.notna() & ma.notna()
    return result.astype(bool)


def golden_cross_state(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where the fast MA is above the slow, -1 below, 0 where undefined.

    Deliberately a *state*, not an event. Cross events are rare, arrive late,
    and invite the "did it cross today?" question that turns a trend filter
    into a timing signal it was never good at.
    """
    f = fast.astype("float64")
    s = slow.astype("float64")
    state = pd.Series(0, index=f.index, dtype="int64")
    valid = f.notna() & s.notna()
    state[valid & (f > s)] = 1
    state[valid & (f < s)] = -1
    return state
