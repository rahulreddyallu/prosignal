"""Return and volatility primitives.

Point-in-time safe: the value at ``t`` uses only observations at or before
``t``. No ``center=True``, no ``bfill()``, no negative ``shift()``.

Windows use explicit ``min_periods`` and return ``NaN`` until fully seeded, so
a 200-session average is never computed from a fortnight of data.
200-DMA -- it is a number that looks like one, which is worse than a gap.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "SESSIONS_PER_YEAR",
    "simple_returns",
    "log_returns",
    "cumulative_return",
    "trailing_return",
    "momentum_skip",
    "realised_volatility",
    "downside_deviation",
    "max_drawdown",
]

#: NSE trades roughly 250 sessions a year once the holiday calendar is applied.
#: 252 is the convention used throughout the finance literature these factors
#: come from, so it is used here for annualisation to keep numbers comparable
#: with published research rather than drifting by a percent or two.
SESSIONS_PER_YEAR = 252

Numeric = Union[pd.Series, pd.DataFrame]


def _as_float(obj: Numeric) -> Numeric:
    return obj.astype("float64")


def simple_returns(prices: Numeric, periods: int = 1) -> Numeric:
    """Simple (arithmetic) returns over ``periods`` sessions.

    Simple returns aggregate correctly across a *portfolio* at a point in time,
    which is why position sizing and P&L use them.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")
    return _as_float(prices).pct_change(periods=periods, fill_method=None)


def log_returns(prices: Numeric, periods: int = 1) -> Numeric:
    """Log returns over ``periods`` sessions.

    Log returns aggregate correctly across *time*, which is why volatility,
    trend slope, and anything summed over sessions uses them. Non-positive
    prices become ``NaN`` rather than ``-inf`` -- a zero or negative price is
    bad data, and propagating an infinity silently poisons every downstream
    mean and standard deviation.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")
    values = _as_float(prices)
    positive = values.where(values > 0)
    return np.log(positive).diff(periods)


def cumulative_return(prices: Numeric, sessions: int) -> Numeric:
    """Total return over a trailing window of ``sessions``.

    Equivalent to ``simple_returns(prices, sessions)`` and named separately
    because the intent at the call site is different -- this is "how much did
    it make over the last N sessions", not "what was the N-session return
    series".
    """
    return simple_returns(prices, periods=sessions)


def trailing_return(prices: pd.Series, sessions: int, as_of: Optional[object] = None) -> Optional[float]:
    """Scalar total return over the last ``sessions`` sessions ending at ``as_of``.

    Returns ``None`` -- never ``0.0`` -- when the history is too short. A
    missing factor must be reported as missing so Stage 4 can renormalise the
    remaining weights; a zero would be silently read as "no momentum", which is
    a completely different and much more dangerous claim.
    """
    series = _as_float(pd.Series(prices)).dropna()
    if as_of is not None:
        series = series[series.index <= as_of]
    if series.size < sessions + 1:
        return None
    start = float(series.iloc[-(sessions + 1)])
    end = float(series.iloc[-1])
    if start <= 0:
        return None
    return end / start - 1.0


def momentum_skip(
    prices: pd.Series,
    lookback_sessions: int,
    skip_sessions: int,
    as_of: Optional[object] = None,
) -> Optional[float]:
    """The 12-1 momentum construction: an N-session return ending K sessions ago.

    With ``lookback_sessions=252`` and ``skip_sessions=21`` this is the classic
    Jegadeesh & Titman (1993) 12-1 factor: twelve months of return, measured to
    one month ago.

    The skip is the whole point and is not a detail to economise on. Short-term
    reversal dominates the most recent month -- what went up hardest last month
    tends to give some back -- so including it actively works against the
    twelve-month effect. Dropping the skip does not "use more data"; it mixes
    two opposing effects and cancels a real edge.

    Requires ``lookback + skip + 1`` observations. Returns ``None`` when short.
    """
    if lookback_sessions < 1:
        raise ValueError("lookback_sessions must be >= 1")
    if skip_sessions < 0:
        raise ValueError("skip_sessions must be >= 0")

    series = _as_float(pd.Series(prices)).dropna()
    if as_of is not None:
        series = series[series.index <= as_of]

    needed = lookback_sessions + skip_sessions + 1
    if series.size < needed:
        return None

    end_pos = series.size - 1 - skip_sessions
    start_pos = end_pos - lookback_sessions
    start = float(series.iloc[start_pos])
    end = float(series.iloc[end_pos])
    if start <= 0:
        return None
    return end / start - 1.0


def realised_volatility(
    prices: Numeric,
    window: int,
    annualise: bool = True,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> Numeric:
    """Rolling realised volatility from log returns.

    Uses ``ddof=1`` (sample standard deviation). With a 21-session window the
    difference from ``ddof=0`` is about 2.5% of the estimate -- small, but
    there is no reason to carry a known bias.
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")
    rets = log_returns(prices)
    sigma = rets.rolling(window=window, min_periods=window).std(ddof=1)
    if annualise:
        sigma = sigma * np.sqrt(sessions_per_year)
    return sigma


def downside_deviation(
    prices: Numeric,
    window: int,
    annualise: bool = True,
    sessions_per_year: int = SESSIONS_PER_YEAR,
) -> Numeric:
    """Rolling standard deviation of NEGATIVE log returns only.

    Upside volatility is not risk to a long book. Separating the two matters
    for momentum specifically, because momentum's return distribution is
    left-skewed -- the crashes are the tail that hurts (Daniel & Moskowitz,
    2016).
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")
    rets = log_returns(prices)
    downside = rets.where(rets < 0)
    sigma = downside.rolling(window=window, min_periods=2).std(ddof=1)
    if annualise:
        sigma = sigma * np.sqrt(sessions_per_year)
    return sigma


def max_drawdown(prices: pd.Series, window: Optional[int] = None) -> Optional[float]:
    """Worst peak-to-trough decline, as a negative fraction.

    ``window=None`` measures over the whole series; otherwise over the trailing
    ``window`` sessions. Returns ``None`` on insufficient data rather than 0.0,
    which would read as "never drew down".
    """
    series = _as_float(pd.Series(prices)).dropna()
    if window is not None:
        series = series.tail(window)
    if series.size < 2:
        return None
    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    return float(drawdown.min())
