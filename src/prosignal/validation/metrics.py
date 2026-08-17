"""Probability of Backtest Overfitting, and the Deflated Sharpe Ratio.

Two questions, two tools.

**PBO** (Bailey, Borwein, López de Prado & Zhu, 2014) asks: *is the
configuration that looked best in-sample actually any good out-of-sample, or
did I select a fluke?* It answers with a probability, via combinatorially
symmetric cross-validation. A high PBO is not merely a bad number to report --
it is an instruction to simplify the model. It is emphatically not an
instruction to keep searching until something scores better, because that is
precisely the behaviour PBO exists to measure.

**DSR** (Bailey & López de Prado, 2014) asks: *given how many configurations I
tried, and given that my returns are not normal, how much of this Sharpe ratio
survives?* It corrects for two specific inflation sources -- selection bias
from multiple testing, and skew/kurtosis, which momentum strategies genuinely
have (see the momentum-crash literature).

Both need an honest trial count. That is why the research ledger is
append-only and why the config carries an enforced search budget: an
understated trial count silently inflates DSR, which is the most flattering
possible way to be wrong.

No scipy dependency: the two normal-distribution functions needed are
implemented here directly, which keeps the install surface small and makes the
numerics auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import erf, exp, log, sqrt
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "norm_cdf",
    "norm_ppf",
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "compute_pbo",
    "PboResult",
    "DsrResult",
]

#: Euler-Mascheroni constant, used in the expected-maximum-Sharpe formula.
EULER_MASCHERONI = 0.5772156649015329


# =============================================================================
# normal distribution
# =============================================================================


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# Acklam's rational approximation to the inverse normal CDF; |error| < 1.15e-9.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function)."""
    if not 0.0 < p < 1.0:
        if p <= 0.0:
            return float("-inf")
        return float("inf")
    if p < _P_LOW:
        q = sqrt(-2.0 * log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p > _P_HIGH:
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    )


# =============================================================================
# Sharpe ratios
# =============================================================================


def sharpe_ratio(returns: Sequence[float], periods_per_year: Optional[int] = None) -> float:
    """Sharpe ratio of a return series.

    Returns the *per-period* Sharpe unless ``periods_per_year`` is supplied.
    The PSR/DSR formulas below require the per-period figure, so annualisation
    is opt-in rather than a silent default -- feeding an annualised Sharpe into
    them is a common and badly wrong mistake.
    """
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = float(arr.mean() / sd)
    if periods_per_year:
        sr *= sqrt(periods_per_year)
    return sr


def _moments(returns: np.ndarray) -> "tuple[float, float]":
    """Sample skewness and (non-excess) kurtosis."""
    n = returns.size
    sd = returns.std(ddof=1)
    if n < 3 or sd == 0:
        return 0.0, 3.0
    centred = returns - returns.mean()
    skew = float((centred**3).mean() / sd**3)
    kurt = float((centred**4).mean() / sd**4)
    return skew, kurt


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    benchmark_sr: float = 0.0,
    observed_sr: Optional[float] = None,
) -> float:
    """PSR: probability the true Sharpe exceeds ``benchmark_sr``.

    Adjusts for track-record length, skewness and kurtosis. Both the Sharpe
    and the benchmark must be expressed per period.
    """
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return 0.0
    sr = sharpe_ratio(arr) if observed_sr is None else float(observed_sr)
    skew, kurt = _moments(arr)

    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        return 0.0
    numerator = (sr - benchmark_sr) * sqrt(n - 1)
    return norm_cdf(numerator / sqrt(denom_sq))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe under the null that no strategy has edge.

    This is the bar a *lucky* winner clears by chance alone when you try
    ``n_trials`` configurations whose Sharpes vary with ``sr_variance``. It
    rises with the number of trials, which is exactly why the search budget is
    capped rather than left open.
    """
    if n_trials < 1:
        return 0.0
    if n_trials == 1 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    term = (1.0 - g) * norm_ppf(1.0 - 1.0 / n_trials) + g * norm_ppf(
        1.0 - 1.0 / (n_trials * exp(1.0))
    )
    return sqrt(sr_variance) * term


@dataclass
class DsrResult:
    observed_sr: float
    benchmark_sr: float
    deflated_sr: float
    n_trials: int
    n_observations: int
    skew: float
    kurtosis: float
    passes: bool
    interpretation: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "observed_sr": self.observed_sr,
            "benchmark_sr_expected_max_under_null": self.benchmark_sr,
            "deflated_sr": self.deflated_sr,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "passes": self.passes,
            "interpretation": self.interpretation,
        }


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    confidence: float = 0.95,
) -> DsrResult:
    """Deflate an observed Sharpe by the multiple-testing and non-normality penalties.

    Parameters
    ----------
    returns:
        Per-period returns of the SELECTED configuration.
    n_trials:
        Honest count of configurations tried -- from the research ledger, not
        from memory. Understating it inflates the result.
    trial_sharpes:
        Sharpes of all trials, used to estimate their cross-sectional variance.
        Falls back to a conservative unit variance when unavailable, which
        makes the benchmark harder to clear rather than easier.
    """
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return DsrResult(
            0.0, 0.0, 0.0, n_trials, n, 0.0, 3.0, False,
            "insufficient observations to compute a Sharpe ratio",
        )

    observed = sharpe_ratio(arr)
    skew, kurt = _moments(arr)

    if trial_sharpes is not None and len(list(trial_sharpes)) > 1:
        sr_var = float(np.var(np.asarray(list(trial_sharpes), dtype="float64"), ddof=1))
    else:
        # Without the trial distribution the honest move is a conservative
        # assumption, not an optimistic one.
        sr_var = 1.0 / max(n - 1, 1)

    benchmark = expected_max_sharpe(n_trials, sr_var)
    dsr = probabilistic_sharpe_ratio(arr, benchmark_sr=benchmark, observed_sr=observed)
    passes = dsr >= confidence

    if passes:
        interpretation = (
            f"After charging for {n_trials} trial(s) and for skew/kurtosis, the "
            f"probability the true Sharpe exceeds what the best of {n_trials} "
            f"lucky configurations would produce is {dsr:.1%}."
        )
    else:
        interpretation = (
            f"DSR {dsr:.1%} is below the {confidence:.0%} bar. Given {n_trials} "
            f"trial(s), an observed Sharpe of {observed:.3f} is not "
            f"distinguishable from the best of that many coin flips. Simplify "
            f"the model or gather more data -- do not search further."
        )

    return DsrResult(
        observed_sr=observed,
        benchmark_sr=benchmark,
        deflated_sr=dsr,
        n_trials=n_trials,
        n_observations=n,
        skew=skew,
        kurtosis=kurt,
        passes=passes,
        interpretation=interpretation,
    )


# =============================================================================
# Probability of Backtest Overfitting
# =============================================================================


@dataclass
class PboResult:
    pbo: float
    n_combinations: int
    n_configurations: int
    logits: List[float] = field(default_factory=list)
    oos_ranks: List[float] = field(default_factory=list)
    selected_configs: List[int] = field(default_factory=list)
    median_oos_rank_of_selected: float = 0.0
    interpretation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "pbo": self.pbo,
            "n_combinations": self.n_combinations,
            "n_configurations": self.n_configurations,
            "median_oos_rank_of_selected": self.median_oos_rank_of_selected,
            "interpretation": self.interpretation,
        }


def compute_pbo(performance: np.ndarray, n_splits: int = 16) -> PboResult:
    """Probability of Backtest Overfitting via combinatorially symmetric CV.

    Parameters
    ----------
    performance:
        Matrix of shape ``(n_observations, n_configurations)`` holding each
        configuration's per-period performance (returns, typically).
    n_splits:
        Number of disjoint contiguous sub-matrices, ``S``. The procedure takes
        every way of assigning ``S/2`` of them to the in-sample half, giving
        ``C(S, S/2)`` symmetric train/test pairs. ``S`` must be even.

    Method: for each pair, pick the configuration with the best in-sample
    performance, then find its *rank* among all configurations out-of-sample.
    Convert that relative rank to a logit. PBO is the fraction of pairs where
    the logit is at or below zero -- i.e. where the in-sample winner landed
    below the out-of-sample median.
    """
    M = np.asarray(performance, dtype="float64")
    if M.ndim != 2:
        raise ValueError("performance must be a 2-D (observations x configurations) array")
    n_obs, n_cfg = M.shape
    if n_cfg < 2:
        raise ValueError("PBO needs at least 2 configurations to rank")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so the halves are symmetric")
    if n_obs < n_splits:
        raise ValueError(
            f"need at least {n_splits} observations to form {n_splits} sub-matrices"
        )

    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2

    logits: List[float] = []
    oos_ranks: List[float] = []
    selected: List[int] = []

    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = tuple(b for b in range(n_splits) if b not in is_blocks)
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_perf = _sharpe_by_column(M[is_idx, :])
        oos_perf = _sharpe_by_column(M[oos_idx, :])

        best = int(np.nanargmax(is_perf))
        selected.append(best)

        # Relative rank of the in-sample winner within the OOS distribution.
        order = np.argsort(np.argsort(oos_perf))  # 0 = worst
        rank = float(order[best] + 1) / float(n_cfg + 1)
        oos_ranks.append(rank)

        rank = min(max(rank, 1e-9), 1 - 1e-9)
        logits.append(log(rank / (1.0 - rank)))

    arr_logits = np.asarray(logits, dtype="float64")
    pbo = float(np.mean(arr_logits <= 0.0)) if arr_logits.size else 0.0
    median_rank = float(np.median(oos_ranks)) if oos_ranks else 0.0

    if pbo <= 0.2:
        verdict = (
            f"PBO {pbo:.1%}: the in-sample winner usually stays above the "
            f"out-of-sample median. Consistent with a real effect, though not "
            f"proof of one."
        )
    elif pbo <= 0.5:
        verdict = (
            f"PBO {pbo:.1%}: selection is unreliable roughly {pbo:.0%} of the "
            f"time. Treat the chosen configuration as provisional and prefer "
            f"the simpler variants within the plateau."
        )
    else:
        verdict = (
            f"PBO {pbo:.1%}: the in-sample winner lands below the out-of-sample "
            f"median more often than not, which is what selecting noise looks "
            f"like. Simplify the model -- do not keep searching for a "
            f"configuration that scores better."
        )

    return PboResult(
        pbo=pbo,
        n_combinations=len(logits),
        n_configurations=n_cfg,
        logits=logits,
        oos_ranks=oos_ranks,
        selected_configs=selected,
        median_oos_rank_of_selected=median_rank,
        interpretation=verdict,
    )


def _sharpe_by_column(block: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each column, with zero-variance columns as NaN-safe 0."""
    mean = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mean / sd, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
