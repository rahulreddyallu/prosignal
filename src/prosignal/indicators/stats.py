"""Rolling distributional statistics.

India VIX has traded between roughly 8 and 87. An absolute threshold such as
"high is above 20" reads 2017 and 2020 on the same scale: a VIX of 18 was a
spike in the calm of 2017 and a relief in April 2020.

The engine asks where today sits within its own trailing distribution instead,
which adapts as the volatility regime shifts.

The window is a trade-off. Too short and the percentile saturates; too long and
it stops responding to a genuine regime change. The configured default is 252
sessions, tagged UNVALIDATED.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "rolling_percentile",
    "percentile_of_last",
    "rolling_zscore",
    "rolling_sigma",
    "sigma_move",
    "rate_of_change_pct",
]


def _percentile_rank(win: np.ndarray, midpoint: bool = True) -> float:
    """Percentile rank of the LAST element of ``win`` within ``win``.

    Ties are handled by the midpoint convention:

        rank = 100 * (count_below + 0.5 * count_equal) / n

    The naive "count values <= current" form returns 100 for a perfectly flat
    series, since every value ties the current one. A dead-flat India VIX would
    then read as maximum volatility and hold the regime engine in its most
    defensive bucket. The midpoint convention returns 50.
    """
    current = win[-1]
    if np.isnan(current):
        return float("nan")
    valid = win[~np.isnan(win)]
    if valid.size == 0:
        return float("nan")

    below = float((valid < current).sum())
    if not midpoint:
        return 100.0 * float((valid <= current).sum()) / float(valid.size)
    equal = float((valid == current).sum())
    return 100.0 * (below + 0.5 * equal) / float(valid.size)


def rolling_percentile(series: pd.Series, window: int, midpoint: bool = True) -> pd.Series:
    """Percentile rank (0-100) of each value within its own trailing window.

    The window *includes* the current observation, so the result at ``t`` uses
    only data up to and including ``t`` -- point-in-time safe.

    ``midpoint`` selects the tie convention; see :func:`_percentile_rank`. Leave
    it at the default unless you specifically want weak ranking.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    values = pd.Series(series).astype("float64")
    return values.rolling(window=window, min_periods=window).apply(
        lambda win: _percentile_rank(win, midpoint=midpoint), raw=True
    )


def percentile_of_last(
    series: pd.Series, window: int, as_of: Optional[object] = None, midpoint: bool = True
) -> Optional[float]:
    """Scalar percentile of the most recent value within its trailing window.

    Returns ``None`` when the window is not full. A tercile read off 40
    sessions when 252 were asked for is not the same statistic, and quietly
    returning it would let the regime engine claim a confidence it has not
    earned.
    """
    values = pd.Series(series).astype("float64")
    if as_of is not None:
        values = values[values.index <= as_of]
    values = values.dropna()
    if values.size < window:
        return None

    win = values.tail(window).to_numpy(dtype="float64")
    result = _percentile_rank(win, midpoint=midpoint)
    return None if np.isnan(result) else float(result)


def rolling_zscore(series: pd.Series, window: int, ddof: int = 1) -> pd.Series:
    """Z-score of each value against its own trailing window."""
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")
    values = pd.Series(series).astype("float64")
    mean = values.rolling(window=window, min_periods=window).mean()
    sigma = values.rolling(window=window, min_periods=window).std(ddof=ddof)
    return (values - mean) / sigma.where(sigma > 0)


def rolling_sigma(series: pd.Series, window: int, ddof: int = 1) -> pd.Series:
    """Rolling standard deviation with an explicit full-window requirement."""
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")
    return (
        pd.Series(series)
        .astype("float64")
        .rolling(window=window, min_periods=window)
        .std(ddof=ddof)
    )


def sigma_move(
    series: pd.Series, window: int, as_of: Optional[object] = None
) -> Optional[float]:
    """How many trailing standard deviations the LAST observation represents.

    Used by the Stage 1 bad-tick check. Deliberately excludes the observation
    being judged from the distribution it is judged against -- otherwise a
    single enormous tick inflates the sigma it is being compared to and
    conceals itself. That self-masking is the failure mode of the naive
    version of this check.
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")

    values = pd.Series(series).astype("float64")
    if as_of is not None:
        values = values[values.index <= as_of]
    values = values.dropna()
    if values.size < window + 1:
        return None

    current = float(values.iloc[-1])
    history = values.iloc[-(window + 1) : -1]
    sigma = float(history.std(ddof=1))
    mean = float(history.mean())
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return (current - mean) / sigma


def rate_of_change_pct(
    series: pd.Series, periods: int, as_of: Optional[object] = None
) -> Optional[float]:
    """Percent change of the last value versus ``periods`` sessions earlier."""
    if periods < 1:
        raise ValueError("periods must be >= 1")

    values = pd.Series(series).astype("float64")
    if as_of is not None:
        values = values[values.index <= as_of]
    values = values.dropna()
    if values.size < periods + 1:
        return None

    past = float(values.iloc[-(periods + 1)])
    current = float(values.iloc[-1])
    if past == 0:
        return None
    return (current / past - 1.0) * 100.0
