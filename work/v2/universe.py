"""Point-in-time universe screens.

Mirrors `prosignal.features.crosssec.liquidity_mask`: rolling-median turnover,
a floor on the QUOTED (unadjusted) price, a listed-history floor, then the top
`max_names` by turnover. Resolved per date, never projected backwards.
"""
from __future__ import annotations
import numpy as np, pandas as pd


def eligible_mask(close: pd.DataFrame, turnover: pd.DataFrame, adj_factor: pd.DataFrame,
                  *, min_adtv_inr: float = 5e7, lookback: int = 60,
                  max_names: int = 750, min_history: int = 300,
                  min_price_inr: float = 20.0) -> pd.DataFrame:
    adtv = turnover.rolling(lookback, min_periods=1).median()
    listed = close.notna().cummax().cumsum()
    fac = adj_factor.where(adj_factor > 0)
    price = close.divide(fac).fillna(close)          # quoted, not back-adjusted
    ok = ((adtv >= min_adtv_inr) & (price >= min_price_inr) & (listed >= min_history))
    rank = adtv.where(ok).rank(axis=1, ascending=False, method="first")
    return (ok & (rank <= max_names)).fillna(False)
