"""Trend measurement by regression on log price.

A price-versus-200-DMA test answers a weaker question: price can sit 2% above a
200-DMA that has fallen for six months and still read as an uptrend. The slope
of a regression through log price gives direction and rate, and the R-squared
indicates whether the fit means anything.

Log price rather than price: a slope on raw price is in rupees per session and
is not comparable between a stock at 150 and one at 15,000. On log price it is
a continuously-compounded growth rate, which annualises and compares across the
universe.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .returns import SESSIONS_PER_YEAR

__all__ = [
    "ols_slope",
    "annualised_log_slope",
    "rolling_annualised_slope",
    "trend_quality",
]


def ols_slope(y: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope and R-squared of ``y`` against ``0, 1, ..., n-1``.

    Closed form rather than ``np.polyfit``: this runs per-symbol per-session
    across the universe, and the explicit form avoids the fitting machinery's
    overhead entirely.
    """
    n = y.size
    if n < 2:
        return float("nan"), float("nan")

    x = np.arange(n, dtype="float64")
    x_mean = x.mean()
    y_mean = y.mean()
    dx = x - x_mean
    dy = y - y_mean

    denom = float((dx * dx).sum())
    if denom == 0:
        return float("nan"), float("nan")

    slope = float((dx * dy).sum() / denom)

    ss_tot = float((dy * dy).sum())
    if ss_tot == 0:
        # A perfectly flat series: the slope is genuinely zero and the fit is
        # exact, but R-squared is undefined (0/0). Report the slope, not a fit.
        return slope, float("nan")
    residuals = dy - slope * dx
    ss_res = float((residuals * residuals).sum())
    r_squared = 1.0 - ss_res / ss_tot
    return slope, r_squared


def annualised_log_slope(
    prices: pd.Series,
    window: int,
    sessions_per_year: int = SESSIONS_PER_YEAR,
    as_of: Optional[object] = None,
) -> Optional[float]:
    """Annualised continuously-compounded trend rate over the last ``window`` sessions.

    A return of ``0.18`` means log price is rising at 18% a year *at the
    current pace* -- an extrapolation of the fitted line, not a forecast.

    Returns ``None`` when there is not a full window. Reporting a slope fitted
    to 12 points as if it were a 63-session trend is exactly the kind of quiet
    overstatement that makes a regime engine flap.
    """
    if window < 2:
        raise ValueError("window must be >= 2 to fit a slope")

    series = pd.Series(prices).astype("float64").dropna()
    if as_of is not None:
        series = series[series.index <= as_of]
    series = series[series > 0]
    if series.size < window:
        return None

    log_price = np.log(series.tail(window).to_numpy(dtype="float64"))
    slope, _ = ols_slope(log_price)
    if not np.isfinite(slope):
        return None
    return float(slope * sessions_per_year)


def rolling_annualised_slope(
    prices: pd.Series,
    window: int,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> pd.Series:
    """``annualised_log_slope`` evaluated at every point, as a Series.

    Stage 2's transition detector needs the slope as it stood N sessions ago,
    which means the whole history, not just today's value.
    """
    if window < 2:
        raise ValueError("window must be >= 2 to fit a slope")

    series = pd.Series(prices).astype("float64")
    positive = series.where(series > 0)
    log_price = np.log(positive)

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return float("nan")
        slope, _ = ols_slope(values)
        return slope * sessions_per_year

    return log_price.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def trend_quality(
    prices: pd.Series, window: int, as_of: Optional[object] = None
) -> Optional[float]:
    """R-squared of the log-price regression: how *clean* the trend is.

    Two stocks can share an annualised slope of 40% while one grinds up in a
    straight line and the other gets there through two crashes and a spike.
    They are not the same trade. R-squared separates them, and it is the honest
    way to say "trending" versus "went up".
    """
    if window < 2:
        raise ValueError("window must be >= 2 to fit a slope")

    series = pd.Series(prices).astype("float64").dropna()
    if as_of is not None:
        series = series[series.index <= as_of]
    series = series[series > 0]
    if series.size < window:
        return None

    log_price = np.log(series.tail(window).to_numpy(dtype="float64"))
    _, r_squared = ols_slope(log_price)
    if not np.isfinite(r_squared):
        return None
    return float(r_squared)
