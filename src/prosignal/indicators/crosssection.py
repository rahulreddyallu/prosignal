"""Cross-sectional transforms: winsorising, standardisation, ranking, neutralising.

These operate across symbols at one point in time, not across time for one
symbol. Stage 4 applies them in order:

    winsorise  ->  standardise  ->  neutralise  ->  weight

Winsorising comes first because a z-score's mean and standard deviation are
both destroyed by a single extreme value, and Indian midcaps can legitimately
order announcement, and left raw it will pull the entire universe's mean and
inflate the sigma so that every other stock's z-score collapses toward zero.
The result is a "diversified" factor score that is really a bet on one ticker.

Rank-based standardisation sidesteps the problem entirely and is the more
robust default. It also throws away real information -- the *distance* between
the first and second name -- which is why both are offered and the choice is a
config parameter rather than a preference baked into the code.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = [
    "winsorise",
    "zscore",
    "robust_zscore",
    "rank_to_unit_interval",
    "sector_neutralise",
    "spearman_matrix",
    "spearman_pairs",
    "standardise",
    "STANDARDISATION_METHODS",
]

#: Accepted values for the ``standardisation`` config key.
STANDARDISATION_METHODS = ("zscore", "rank", "robust_zscore")


def winsorise(
    values: pd.Series, lower_pct: float = 1.0, upper_pct: float = 99.0
) -> pd.Series:
    """Clip values to the given percentile bounds of their own distribution.

    Clipping, not dropping. A stock at the 99.8th percentile of momentum is
    still the most extreme name in the universe and should still rank first --
    it just should not be allowed to define the scale everyone else is measured
    against.
    """
    if not 0.0 <= lower_pct < upper_pct <= 100.0:
        raise ValueError(
            f"require 0 <= lower_pct < upper_pct <= 100; got {lower_pct}, {upper_pct}"
        )
    series = pd.Series(values).astype("float64")
    clean = series.dropna()
    if clean.size < 2:
        return series
    low = float(np.percentile(clean, lower_pct))
    high = float(np.percentile(clean, upper_pct))
    return series.clip(lower=low, upper=high)


def zscore(values: pd.Series, ddof: int = 1) -> pd.Series:
    """Standard z-score across the cross-section.

    A zero-variance cross-section returns all zeros rather than ``inf``: if
    every stock has the same value, no stock is above average on it, and that
    is the honest answer.
    """
    series = pd.Series(values).astype("float64")
    clean = series.dropna()
    if clean.size < 2:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    sigma = float(clean.std(ddof=ddof))
    mean = float(clean.mean())
    if not np.isfinite(sigma) or sigma <= 0:
        return pd.Series(0.0, index=series.index, dtype="float64").where(series.notna())
    return (series - mean) / sigma


def robust_zscore(values: pd.Series) -> pd.Series:
    """Median/MAD standardisation -- resistant to outliers by construction.

    The 1.4826 factor rescales the median absolute deviation so that, for
    normally distributed data, it estimates the same quantity as the standard
    deviation. Without it this score is on a different scale from ``zscore``
    and the two cannot be mixed inside one composite.
    """
    series = pd.Series(values).astype("float64")
    clean = series.dropna()
    if clean.size < 2:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(0.0, index=series.index, dtype="float64").where(series.notna())
    return (series - median) / (1.4826 * mad)


def rank_to_unit_interval(values: pd.Series, ascending: bool = True) -> pd.Series:
    """Cross-sectional rank mapped onto ``[0, 1]``.

    Ties share the average rank. The mapping uses ``(rank - 1) / (n - 1)`` so
    the worst name is exactly 0.0 and the best exactly 1.0, which makes the
    Stage 8 ``min_composite_score`` threshold mean what it appears to mean.
    With a single name the result is 0.5 -- neither best nor worst, since a
    ranking of one is not a ranking.
    """
    series = pd.Series(values).astype("float64")
    clean = series.dropna()
    n = clean.size
    if n == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    if n == 1:
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out.loc[clean.index] = 0.5
        return out

    ranks = clean.rank(method="average", ascending=ascending)
    scaled = (ranks - 1.0) / (n - 1.0)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[scaled.index] = scaled
    return out


def standardise(values: pd.Series, method: str = "zscore") -> pd.Series:
    """Dispatch to the configured standardisation method.

    Raises on an unknown method rather than silently defaulting -- a typo in
    ``parameters.yaml`` must not quietly change how every factor is scaled.
    """
    key = str(method).strip().lower()
    if key == "zscore":
        return zscore(values)
    if key == "robust_zscore":
        return robust_zscore(values)
    if key == "rank":
        return rank_to_unit_interval(values)
    raise ValueError(
        f"unknown standardisation method {method!r}; "
        f"expected one of {STANDARDISATION_METHODS}"
    )


def sector_neutralise(
    values: pd.Series,
    sectors: Dict[str, str],
    min_sector_size: int = 3,
) -> pd.Series:
    """Demean each value within its own sector.

    Without this, a momentum screen run in 2021 returns a list of PSU banks and
    metals and calls it stock selection. It is not -- it is one sector bet
    wearing ten tickers, and it will draw down as one position. Demeaning
    within sector asks the question that was actually intended: *is this stock
    strong relative to its peers?*

    Sectors with fewer than ``min_sector_size`` members are left untouched.
    Demeaning a two-stock sector forces one to +x and the other to -x by
    construction, manufacturing a signal out of arithmetic.
    """
    series = pd.Series(values).astype("float64")
    if not sectors:
        return series

    sector_series = pd.Series(
        {sym: sectors.get(sym) for sym in series.index}, dtype="object"
    ).reindex(series.index)

    out = series.copy()
    for sector, members in sector_series.groupby(sector_series).groups.items():
        if sector is None or (isinstance(sector, float) and np.isnan(sector)):
            continue
        idx = pd.Index(members)
        block = series.loc[idx].dropna()
        if block.size < min_sector_size:
            continue
        out.loc[block.index] = block - float(block.mean())
    return out


def spearman_matrix(frame: pd.DataFrame, min_observations: int = 5) -> pd.DataFrame:
    """Pairwise Spearman rank correlation between columns.

    Spearman rather than Pearson because factor scores are ordinal in spirit --
    what matters is whether two factors rank the universe the same way, not
    whether they are linearly related. Two factors can have a modest Pearson
    correlation and still produce nearly identical top-20 lists, which is the
    redundancy that actually costs you diversification.
    """
    numeric = frame.select_dtypes(include=[np.number]).astype("float64")
    if numeric.shape[1] < 2:
        return pd.DataFrame(index=numeric.columns, columns=numeric.columns, dtype="float64")
    if numeric.dropna().shape[0] < min_observations:
        return pd.DataFrame(
            np.nan, index=numeric.columns, columns=numeric.columns, dtype="float64"
        )
    return numeric.corr(method="spearman", min_periods=min_observations)


def spearman_pairs(
    frame: pd.DataFrame, min_observations: int = 5
) -> Dict[str, float]:
    """Flatten :func:`spearman_matrix` into ``{"a|b": rho}`` for the upper triangle.

    The report format Stage 4's redundancy check writes to the ledger. Keys are
    sorted so the same pair always produces the same key regardless of column
    order.
    """
    matrix = spearman_matrix(frame, min_observations=min_observations)
    out: Dict[str, float] = {}
    columns = list(matrix.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = matrix.loc[a, b]
            if pd.notna(value):
                key = "|".join(sorted((str(a), str(b))))
                out[key] = float(value)
    return out
