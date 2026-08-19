"""Indicator library -- pure functions over pandas, no config reads, no I/O.

Nothing in this package imports config, contracts or the store, so every
function is testable against hand-computed values and a stage cannot hide a
threshold here where it would escape the parameter inventory.

Two invariants, both enforced by tests:

1. Point-in-time safety. The value at ``t`` uses only observations at or before
   ``t``. No ``center=True``, no ``bfill``, no negative ``shift``.
2. Honest windows. A window of ``n`` returns ``NaN`` until it has ``n`` real
   observations, and scalar helpers return ``None`` rather than a number from
   too little history. A short-window value that resembles a long-window value
   is worse than a gap, because a gap is visible.
"""

from __future__ import annotations

from .atr import (
    ATR_METHODS,
    atr,
    atr_pct_of_price,
    sma_atr,
    true_range,
    wilder_atr,
)
from .crosssection import (
    STANDARDISATION_METHODS,
    rank_to_unit_interval,
    robust_zscore,
    sector_neutralise,
    spearman_matrix,
    spearman_pairs,
    standardise,
    winsorise,
    zscore,
)
from .moving_averages import (
    distance_from_ma_atr,
    distance_from_ma_pct,
    ema,
    golden_cross_state,
    is_above,
    ma_slope_pct,
    sma,
    wilder_ma,
)
from .returns import (
    SESSIONS_PER_YEAR,
    cumulative_return,
    downside_deviation,
    log_returns,
    max_drawdown,
    momentum_skip,
    realised_volatility,
    simple_returns,
    trailing_return,
)
from .stats import (
    percentile_of_last,
    rate_of_change_pct,
    rolling_percentile,
    rolling_sigma,
    rolling_zscore,
    sigma_move,
)
from .trend import (
    annualised_log_slope,
    ols_slope,
    rolling_annualised_slope,
    trend_quality,
)

__all__ = [
    # constants
    "SESSIONS_PER_YEAR",
    "ATR_METHODS",
    "STANDARDISATION_METHODS",
    # returns
    "simple_returns",
    "log_returns",
    "cumulative_return",
    "trailing_return",
    "momentum_skip",
    "realised_volatility",
    "downside_deviation",
    "max_drawdown",
    # moving averages
    "sma",
    "ema",
    "wilder_ma",
    "distance_from_ma_pct",
    "distance_from_ma_atr",
    "ma_slope_pct",
    "is_above",
    "golden_cross_state",
    # atr
    "true_range",
    "wilder_atr",
    "sma_atr",
    "atr",
    "atr_pct_of_price",
    # trend
    "ols_slope",
    "annualised_log_slope",
    "rolling_annualised_slope",
    "trend_quality",
    # stats
    "rolling_percentile",
    "percentile_of_last",
    "rolling_zscore",
    "rolling_sigma",
    "sigma_move",
    "rate_of_change_pct",
    # cross-section
    "winsorise",
    "zscore",
    "robust_zscore",
    "rank_to_unit_interval",
    "standardise",
    "sector_neutralise",
    "spearman_matrix",
    "spearman_pairs",
]
