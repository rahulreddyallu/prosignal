"""Rolling distributional statistics.

The India VIX tercile is the reason this module exists, and it is worth being
precise about what "high VIX" means. India VIX has traded from roughly 8 to
roughly 87 over its life. An absolute threshold -- "high is above 20" -- is
therefore a claim that 2017 and 2020 should be read on the same scale, which
is indefensible: a VIX of 18 was an alarming spike in the calm of 2017 and a
profound relief in April 2020.

So the engine asks a relative question instead: *where does today sit inside
its own trailing distribution?* That is a rolling percentile, and it adapts as
the volatility regime itself shifts.

The window is a real trade-off, not a free parameter. Too short and the
percentile saturates -- every day is the 99th percentile of a fortnight that
has been rising. Too long and it stops responding to a genuine regime change.
252 sessions (one year) is the configured default and is tagged UNVALIDATED.
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

    This is the standard definition, and using it is not pedantry -- the naive
    "count values <= current" version returns **100 for a perfectly flat
    series**, because every value ties the current one. That single detail
    would make a dead-flat India VIX read as maximum volatility forever, and
    the regime engine would sit in its most defensive bucket during the
    calmest possible market. The midpoint convention returns 50 there, which
    is the honest answer: today is exactly typical of its own history.
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
