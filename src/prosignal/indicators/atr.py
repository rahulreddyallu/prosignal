"""True Range and Average True Range.

ATR sets the stop distance (Stage 7), normalises distance-from-moving-average
so it is comparable across the universe (Stage 6), and feeds the risk category.

Two definitional points handled explicitly below:

Gaps. True Range is not ``high - low`` but the greater of the session range and
the distance from the previous close to today's high or low. NSE stocks gap
regularly on results, block deals and index-inclusion news, and a stop sized on
the intraday range alone is too tight on exactly those days.

Wilder vs SMA. Wilder smoothing uses ``alpha = 1/n``, a longer memory than a
conventional n-period EMA (equivalent to span ``2n-1``). The config selects the
method; Wilder is the default and tagged STRUCTURAL because it is the
definition. The SMA variant exists so a sensitivity run can confirm the choice
does not carry the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .moving_averages import wilder_ma

__all__ = ["true_range", "wilder_atr", "sma_atr", "atr", "atr_pct_of_price", "ATR_METHODS"]

#: Accepted values for the ``method`` argument / config key.
ATR_METHODS = ("wilder", "sma")


def true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """True Range: ``max(H-L, |H-C_prev|, |L-C_prev|)``.

    The first observation has no previous close and is therefore ``NaN`` --
    not ``high - low``. Seeding it with the bare range understates the first
    value and, because the ATR is a recursive average, that understatement
    persists for many sessions.
    """
    h = high.astype("float64")
    l = low.astype("float64")
    c = close.astype("float64")
    prev_close = c.shift(1)

    range_hl = h - l
    range_hc = (h - prev_close).abs()
    range_lc = (l - prev_close).abs()

    tr = pd.concat([range_hl, range_hc, range_lc], axis=1).max(axis=1)
    tr[prev_close.isna()] = np.nan
    return tr.rename("true_range")


def wilder_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """ATR using Wilder's smoothing (the standard definition)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(high, low, close)
    # Drop the leading NaN before smoothing so the recursion is seeded from a
    # real True Range, then reindex to the original frame.
    valid = tr.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=tr.index, name="atr", dtype="float64")
    smoothed = wilder_ma(valid, period)
    return smoothed.reindex(tr.index).rename("atr")


def sma_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """ATR as a simple moving average of True Range.

    Present for sensitivity testing against the Wilder default, not because it
    is better. It reacts faster and decays faster, so stops built on it are
    more jumpy after a single wide session.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(high, low, close)
    return tr.rolling(window=period, min_periods=period).mean().rename("atr")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    method: str = "wilder",
) -> pd.Series:
    """Dispatch to the configured ATR method.

    Raises on an unknown method rather than falling back to a default. A typo
    in ``parameters.yaml`` must not silently change the stop distance on every
    position in the book.
    """
    key = str(method).strip().lower()
    if key == "wilder":
        return wilder_atr(high, low, close, period)
    if key == "sma":
        return sma_atr(high, low, close, period)
    raise ValueError(
        f"unknown ATR method {method!r}; expected one of {ATR_METHODS}"
    )


def atr_pct_of_price(atr_series: pd.Series, close: pd.Series) -> pd.Series:
    """ATR expressed as a percentage of price.

    The comparable form. An ATR of 12 rupees says nothing until you know
    whether the stock trades at 150 or 15,000; the same 12 is a 8% daily range
    in one case and 0.08% in the other.
    """
    c = close.astype("float64")
    safe = c.where(c > 0)
    return (atr_series.astype("float64") / safe) * 100.0
