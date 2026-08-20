"""Point-in-time value and quality features from quarterly filings.

Every figure is gated on `filing_date <= as_of`. Measured NSE disclosure lag is
9-45 days, so a feature keyed to period end would hand the backtest up to six
weeks of foresight.

The quarterly Ind-AS filing is an income statement, supporting margins,
interest coverage, earnings growth and earnings stability. It contains no
equity, assets or borrowings, so ROE, book-to-price and debt-to-equity are
absent rather than approximated -- Indian companies file balance sheets
half-yearly at best.

`earnings_yield` is the value factor with the strongest India-specific evidence
(Fama-French replications on CNX 500 / NSE 500), computable here because shares
outstanding derive from paid-up capital over face value.

TTM rather than latest quarter throughout, since Indian earnings are seasonal
and a single quarter compares a company against its own seasonality.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..data.types import SYMBOL

__all__ = [
    "FEATURE_NAMES",
    "point_in_time_snapshot",
    "compute_features",
]

log = get_logger(__name__)

#: Features this module produces. Anything not listed is not computable from an
#: income statement and must not be faked elsewhere.
FEATURE_NAMES = [
    "earnings_yield",       # value      -- TTM net profit / market cap
    "net_margin",           # quality    -- TTM net profit / TTM revenue
    "interest_coverage",    # quality    -- (PBT + finance costs) / finance costs
    "earnings_growth",      # quality    -- TTM profit vs prior-year TTM
    "earnings_stability",   # quality    -- negative coefficient of variation
    "market_cap",           # context    -- also answers "top N by market cap"
]

#: Quarters needed for a TTM figure, and for a year-on-year TTM comparison.
_TTM_QUARTERS = 4
_GROWTH_QUARTERS = 8


def point_in_time_snapshot(
    fundamentals: pd.DataFrame, as_of: dt.date, max_age_days: Optional[int] = None
) -> pd.DataFrame:
    """Every filing publicly known on or before ``as_of``, newest first.

    The point-in-time guarantee lives here: nothing filed after ``as_of`` is
    visible.

    ``max_age_days`` additionally drops filings too old to describe current
    profitability. Gating only on ``filing_date <= as_of`` is correct against
    lookahead but says nothing about staleness, so a store that stopped
    receiving filings keeps scoring on the last one it saw -- measured at 525
    days here, used as though it were current.
    """
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    frame = fundamentals.copy()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame = frame.dropna(subset=["filing_date", "period_end"])
    if max_age_days is not None and not frame.empty:
        floor = pd.Timestamp(as_of) - pd.Timedelta(days=int(max_age_days))
        frame = frame[frame["filing_date"] >= floor]
    if frame.empty:
        # Every row lacked a usable date. Return early rather than comparing an
        # all-NaT datetime64 column against a date, which raises TypeError and
        # would take the whole pipeline down instead of simply dropping the
        # factor. A provider returning undated rows is a degradation, not a
        # crash.
        return pd.DataFrame(columns=fundamentals.columns)

    # Compare in datetime space, then hand back plain dates: mixing
    # datetime64 columns with datetime.date objects is the usual source of
    # "Invalid comparison" in this codepath.
    frame = frame[frame["filing_date"] <= pd.Timestamp(as_of)]
    if frame.empty:
        return pd.DataFrame(columns=fundamentals.columns)
    frame["filing_date"] = frame["filing_date"].dt.date
    frame["period_end"] = frame["period_end"].dt.date
    return frame.sort_values([SYMBOL, "period_end"], ascending=[True, False])


def compute_features(
    fundamentals: pd.DataFrame,
    prices: Dict[str, float],
    as_of: dt.date,
    max_age_days: Optional[int] = None,
) -> pd.DataFrame:
    """Per-symbol value/quality features, using only publicly-known filings.

    ``prices`` maps symbol -> last close, used with derived shares outstanding
    to form market capitalisation.
    """
    known = point_in_time_snapshot(fundamentals, as_of, max_age_days=max_age_days)
    if known.empty:
        return pd.DataFrame(columns=[SYMBOL] + FEATURE_NAMES)

    rows: List[Dict[str, object]] = []
    for symbol, chunk in known.groupby(SYMBOL, sort=False):
        quarters = chunk.head(_GROWTH_QUARTERS)
        feat = _one_symbol(quarters, prices.get(str(symbol)))
        if feat is not None:
            feat[SYMBOL] = str(symbol)
            rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=[SYMBOL] + FEATURE_NAMES)
    out = pd.DataFrame.from_records(rows)
    for col in FEATURE_NAMES:
        if col not in out.columns:
            out[col] = np.nan
    return out[[SYMBOL] + FEATURE_NAMES]


def _one_symbol(quarters: pd.DataFrame, price: Optional[float]) -> Optional[Dict[str, object]]:
    if quarters.empty:
        return None

    profits = _series(quarters, "net_profit")
    revenues = _series(quarters, "revenue")

    ttm_profit = _ttm(profits)
    ttm_revenue = _ttm(revenues)

    out: Dict[str, object] = {name: np.nan for name in FEATURE_NAMES}

    # -- market cap and earnings yield --------------------------------------
    shares = _latest(quarters, "shares_outstanding")
    market_cap = shares * price if (shares and price and price > 0) else None
    if market_cap and market_cap > 0:
        out["market_cap"] = market_cap
        if ttm_profit is not None:
            # Negative earnings give a negative yield, which is correct: a
            # loss-making company should rank at the bottom of a value screen,
            # not be excluded and quietly treated as neutral.
            out["earnings_yield"] = ttm_profit / market_cap

    # -- margins -------------------------------------------------------------
    if ttm_profit is not None and ttm_revenue and ttm_revenue > 0:
        out["net_margin"] = ttm_profit / ttm_revenue

    # -- interest coverage ---------------------------------------------------
    pbt = _ttm(_series(quarters, "profit_before_tax"))
    finance = _ttm(_series(quarters, "finance_costs"))
    if pbt is not None and finance is not None:
        if finance > 0:
            out["interest_coverage"] = (pbt + finance) / finance
        elif finance == 0 and pbt > 0:
            # No debt service at all is the strongest possible coverage. Capped
            # rather than infinite so it cannot dominate a cross-sectional
            # standardisation on its own.
            out["interest_coverage"] = 100.0

    # -- earnings growth: TTM vs the prior-year TTM --------------------------
    if profits.size >= _GROWTH_QUARTERS:
        recent = _ttm(profits[:_TTM_QUARTERS])
        prior = _ttm(profits[_TTM_QUARTERS:_GROWTH_QUARTERS])
        if recent is not None and prior is not None and abs(prior) > 0:
            out["earnings_growth"] = (recent - prior) / abs(prior)

    # -- earnings stability --------------------------------------------------
    # Measured on a trailing-twelve-month series, not on raw quarters.
    #
    # Indian quarterly earnings carry genuine seasonality: festive-quarter
    # retail, agri-linked cyclicality, monsoon-driven construction. A company
    # that earns the same amount every Q3 and every Q1 -- year after year, with
    # no surprises -- has a large dispersion across raw quarters and would score
    # as unstable. That penalises a predictable seasonal pattern as though it
    # were risk, which is the opposite of what this factor is for.
    #
    # Each TTM window spans four consecutive quarters, so every season appears
    # exactly once in each and the seasonal component cancels. What is left is
    # variation in the annual earning power, which is the thing worth measuring.
    # This also matches how earnings_growth is already built (TTM against the
    # prior-year TTM), so the two quality components now deseasonalise the same
    # way rather than disagreeing about what a quarter means.
    #
    # The windows overlap, which damps the estimate relative to independent
    # annual observations. Non-overlapping years would need far more history
    # than a filing feed reliably carries, and the ranking is cross-sectional:
    # every name is damped identically, so the ordering is unaffected.
    ttm_series = _rolling_ttm(profits)
    if ttm_series.size >= 3:
        mean = float(np.mean(ttm_series))
        if abs(mean) > 0:
            cv = float(np.std(ttm_series, ddof=1)) / abs(mean)
            # Negated so that higher is better, matching every other quality
            # component and removing a sign trap at the weighting step.
            out["earnings_stability"] = -cv

    return out


def _rolling_ttm(values: np.ndarray) -> np.ndarray:
    """Overlapping trailing-twelve-month sums, newest first.

    ``values`` arrives newest first, so window ``i`` covers quarters ``i`` to
    ``i + 3``. A window containing a missing quarter is dropped rather than
    summed around: three quarters plus a gap is not a year, and treating it as
    one would understate the level and overstate the variation.
    """
    if values.size < _TTM_QUARTERS:
        return np.array([], dtype="float64")
    windows = []
    for i in range(values.size - _TTM_QUARTERS + 1):
        chunk = values[i:i + _TTM_QUARTERS]
        if np.isnan(chunk).any():
            continue
        windows.append(float(np.sum(chunk)))
    return np.array(windows, dtype="float64")


def _series(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _ttm(values: np.ndarray) -> Optional[float]:
    """Trailing four quarters. Requires all four -- a partial TTM is not a TTM."""
    if values.size < _TTM_QUARTERS:
        return None
    window = values[:_TTM_QUARTERS]
    if np.isnan(window).any():
        return None
    return float(window.sum())


def _latest(frame: pd.DataFrame, column: str) -> Optional[float]:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None
