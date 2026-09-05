"""Probability of Backtest Overfitting, and the Deflated Sharpe Ratio.

PBO (Bailey, Borwein, Lopez de Prado & Zhu, 2014) estimates whether the
configuration that ranked best in-sample holds up out-of-sample, via
combinatorially symmetric cross-validation. A high PBO indicates the model
should be simplified, not that the search should continue -- continued search
is the behaviour PBO measures.

DSR (Bailey & Lopez de Prado, 2014) discounts a Sharpe ratio for the number of
configurations tried and for skew and kurtosis, which momentum strategies carry
(see the momentum-crash literature).

Both require an honest trial count, which is why the research ledger is
append-only and the config enforces a search budget. An understated trial count
inflates DSR.

The two normal-distribution functions needed are implemented here rather than
pulling in scipy, keeping the install surface small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import erf, exp, log, sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
    effective_n: Optional[float] = None,
) -> float:
    """PSR: probability the true Sharpe exceeds ``benchmark_sr``.

    Adjusts for track-record length, skewness and kurtosis. Both the Sharpe
    and the benchmark must be expressed per period.

    ``effective_n`` replaces the raw length in the ``sqrt(n-1)`` term. The
    formula assumes IID returns; a series whose entries repeat (the same test
    date scored under many CPCV splits) or overlap (a 63-session label sampled
    every 21) carries fewer independent observations than it has values, and the
    statistic scales with the square root of that difference. Never larger than
    the raw length -- a caller cannot declare independence it lacks.
    """
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return 0.0
    n_eff = float(n) if effective_n is None else float(min(effective_n, n))
    # THE NULL APPROXIMATION 1/(n_eff-1) IS NOT USED, ANYWHERE, DELIBERATELY.
    # It is the right stand-in for Var[SR] only when n_eff is the INDEPENDENT
    # count; handed the raw length of an overlapping series it is the defect
    # that produced a DSR of 1.000 at 100,000 trials. `effective_n` therefore
    # moves the statistic in one direction only -- it shrinks the sqrt(n-1)
    # term and never the benchmark. A caller with a defensible variance passes
    # `sr_variance`; everyone else gets the conservative unit.
    #
    # A `declared_independence` flag used to gate that choice and outlived the
    # branch it gated, which left this comment describing a decision the code
    # no longer makes.
    if n_eff < 3:
        return 0.0
    sr = sharpe_ratio(arr) if observed_sr is None else float(observed_sr)
    skew, kurt = _moments(arr)

    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        return 0.0
    numerator = (sr - benchmark_sr) * sqrt(n_eff - 1.0)
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


#: What `sr_variance` was in the end. The DSR is more sensitive to this than to
#: anything else it is given -- on this engine's own evidence, moving it between
#: two defensible estimates moved the answer from 0.38 to 0.91 -- so a result
#: that does not say where it came from cannot be read.
SR_VAR_FROM_TRIALS = "trials"
SR_VAR_SUPPLIED = "supplied"
SR_VAR_UNIT = "unit_conservative"
SR_VAR_UNDERCOVERED = "unit_undercovered_trials"

#: Share of the CHARGED trials that must carry a recorded score before their
#: variance is used as Var[SR].
#:
#: Bailey & Lopez de Prado's expected maximum assumes Var[SR] is the dispersion
#: across the configurations that were searched. A registry holding scores for
#: some of them estimates that dispersion from whichever arms happened to be
#: scored -- and those are systematically the most SIMILAR ones, because a
#: command that sweeps eighteen buy/hold bands records eighteen near-identical
#: results while the genuinely different ideas were compared once and moved on
#: from. Measured here: 18 scored arms out of 87 charged gave Var[SR] 0.00178,
#: an expected-maximum bar of 0.105, and a comfortable PASS -- lower than the
#: unit fallback by a factor of 560 and lower than the truth by an unknown one.
#:
#: Under-covered scores are therefore INFORMATIVE, not authoritative: they are
#: reported, and the bar is set from the conservative variance until enough of
#: the search has been priced for its spread to mean anything.
MIN_TRIAL_SCORE_COVERAGE = 0.5


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
    #: Independent observations the inference actually used. Equal to
    #: ``n_observations`` for a clean series; smaller wherever the caller knows
    #: the entries repeat or overlap. Reported because a DSR read without it
    #: cannot be checked.
    effective_n: float = 0.0
    #: Cross-sectional variance of trial Sharpes used to build the benchmark,
    #: and where it came from. Carried on the result because the number is
    #: load-bearing and was previously invisible.
    sr_variance: float = 1.0
    sr_variance_source: str = SR_VAR_UNIT
    #: What the recorded trial scores actually said, whether or not it was
    #: used. Reported so an under-covered registry is visible as a number
    #: rather than as an absence.
    sr_variance_measured: float = float("nan")
    trials_scored: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "observed_sr": self.observed_sr,
            "benchmark_sr_expected_max_under_null": self.benchmark_sr,
            "deflated_sr": self.deflated_sr,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "effective_n": self.effective_n,
            "sr_variance": self.sr_variance,
            "sr_variance_source": self.sr_variance_source,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "passes": self.passes,
            "interpretation": self.interpretation,
            # `sr_variance` and `sr_variance_source` appeared twice in this
            # dict with identical values -- harmless, and a sign the method was
            # edited twice without either edit reading the other.
            "sr_variance_measured": self.sr_variance_measured,
            "trials_scored": self.trials_scored,
        }


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    confidence: float = 0.95,
    effective_n: Optional[float] = None,
    sr_variance: Optional[float] = None,
) -> DsrResult:
    """Deflate an observed Sharpe by the multiple-testing and non-normality penalties.

    Parameters
    ----------
    returns:
        Per-period returns of the SELECTED configuration. They must be
        INDEPENDENT observations. Feeding a vector that counts each period
        several times -- the pooled (split, test-date) excess a CPCV run
        produces, for instance -- inflates `n` and collapses the fallback
        variance, and the result passes whatever the strategy did.
    n_trials:
        Honest count of configurations tried -- from the research ledger, not
        from memory. Understating it inflates the result.
    trial_sharpes:
        Sharpes of the TRIALS that were searched over, used to estimate their
        cross-sectional variance. Bailey & Lopez de Prado's ``Var[SR]`` is the
        dispersion of Sharpe ratios ACROSS TRIALS -- not the sampling variance
        of one estimate, and NOT the spread of resampled paths of the single
        selected configuration. Those paths measure noise in one strategy, they
        are smaller, and using them shrinks the benchmark: measured on this
        engine, 0.0083 against 0.0455, which moves the answer from FAIL 0.38 to
        PASS 0.91.
    sr_variance:
        Supply Var[SR] directly when it is known from outside the trial list.
    effective_n:
        How many INDEPENDENT observations ``returns`` actually carries. Both the
        PSR's ``sqrt(n-1)`` term and the null fallback below scale with it, so
        passing the raw length of a series whose entries repeat or overlap
        inflates the result by the square root of the duplication.

    THE FAILURE THIS SIGNATURE EXISTS TO PREVENT. `CpcvResult.deflated` used to
    hand over the POOLED excess vector -- one entry per (split, test-date) pair.
    On the shipped geometry that is 639 entries covering 71 distinct dates, so
    every date was counted about nine times and about 28 times over against the
    number of independent 63-session windows. The docstring then claimed the
    variance fallback was "a conservative unit variance"; the code was
    ``1.0 / max(n - 1, 1)``, which at n = 639 is 1.6e-3. The two errors compound
    in the same direction and the result was a DSR of 1.000 that still passed at
    100,000 trials -- a multiple-testing defence insensitive to the multiple
    testing it exists to charge for.

    Measured on the real CPCV output, charging 44 trials:

        as shipped   (n = 639 pooled pairs, sr_var = 1/(n-1))   DSR 1.0000  pass
        n = 71 distinct panel dates                             DSR 0.4649  fail
        n = 23 independent 63-session windows                   DSR 0.1477  fail
        n = 23 windows AND sr_var from the woven path Sharpes   DSR 0.3130  fail

    THE VARIANCE LADDER, in order of preference:

      1. ``sr_variance`` supplied by the caller;
      2. the dispersion of recorded trial scores, when at least
         ``MIN_TRIAL_SCORE_COVERAGE`` of the CHARGED trials carry one;
      3. a true unit variance -- NOT ``1/(n - 1)``.

    ``effective_n`` deliberately does not reach step 3. The null approximation
    ``1/(n_eff - 1)`` is defensible arithmetic on a declared independent count,
    but wiring it to ``effective_n`` would make that argument do two opposing
    jobs at once -- shrinking the PSR's ``sqrt(n-1)`` term while shrinking the
    benchmark -- and the two can cancel, which is how an overlap correction
    stops reaching the statistic. ``effective_n`` therefore moves the answer in
    ONE direction. A caller with a defensible variance passes ``sr_variance``.
    """
    obs = np.asarray(list(returns), dtype="float64")
    obs = obs[np.isfinite(obs)]
    if obs.size < 2 or float(np.std(obs, ddof=1)) <= 0.0:
        # A degenerate series has no Sharpe to deflate. Returning a small
        # positive number invited it to be read as a weak-but-real result.
        raise ValueError(
            "deflated_sharpe_ratio needs at least two observations with "
            "non-zero variance; a constant return series has no Sharpe ratio"
        )
    arr = obs
    n = arr.size
    # The sample size the INFERENCE runs on. Never larger than what was passed:
    # a caller cannot manufacture independence it does not have.
    n_eff = float(n) if effective_n is None else float(min(effective_n, n))
    # THE NULL APPROXIMATION 1/(n_eff-1) IS NOT USED, ANYWHERE, DELIBERATELY.
    # It is the right stand-in for Var[SR] only when n_eff is the INDEPENDENT
    # count; handed the raw length of an overlapping series it is the defect
    # that produced a DSR of 1.000 at 100,000 trials. `effective_n` therefore
    # moves the statistic in one direction only -- it shrinks the sqrt(n-1)
    # term and never the benchmark. A caller with a defensible variance passes
    # `sr_variance`; everyone else gets the conservative unit.
    #
    # A `declared_independence` flag used to gate that choice and outlived the
    # branch it gated, which left this comment describing a decision the code
    # no longer makes.
    if n < 3 or n_eff < 3:
        return DsrResult(
            0.0, 0.0, 0.0, n_trials, n, 0.0, 3.0, False,
            f"insufficient independent observations to compute a Sharpe ratio "
            f"({n_eff:.1f} effective from {n} values)",
            effective_n=n_eff, sr_variance=float("nan"),
            sr_variance_source="unavailable",
        )

    observed = sharpe_ratio(arr)
    skew, kurt = _moments(arr)

    scored = list(trial_sharpes) if trial_sharpes is not None else []
    measured_var = (float(np.var(np.asarray(scored, dtype="float64"), ddof=1))
                    if len(scored) > 1 else float("nan"))
    covered = len(scored) / max(int(n_trials), 1)

    if sr_variance is not None:
        sr_var, source = float(sr_variance), SR_VAR_SUPPLIED
    elif len(scored) > 1 and covered >= MIN_TRIAL_SCORE_COVERAGE:
        sr_var, source = measured_var, SR_VAR_FROM_TRIALS
    elif len(scored) > 1:
        # Scores exist but for too few of the charged trials to describe the
        # search. Under-coverage is EVIDENCE that the search was wider than
        # what was recorded, so the bar must not fall below the conservative
        # unit. See MIN_TRIAL_SCORE_COVERAGE.
        sr_var, source = 1.0, SR_VAR_UNDERCOVERED
    else:
        # A TRUE unit variance, and `effective_n` deliberately does not reach
        # it. The null approximation 1/(n_eff-1) is defensible arithmetic, but
        # wiring it to `effective_n` makes that argument do two opposing jobs:
        # declaring fewer independent observations would shrink the PSR's
        # sqrt(n-1) term (lowering the DSR) while simultaneously shrinking
        # Var[SR] (raising it), and the two can cancel. `effective_n` therefore
        # moves the statistic in ONE direction only. A caller that wants the
        # null approximation passes it as `sr_variance` and says so.
        sr_var, source = 1.0, SR_VAR_UNIT
    if not np.isfinite(sr_var) or sr_var <= 0:
        sr_var, source = 1.0, SR_VAR_UNIT

    benchmark = expected_max_sharpe(n_trials, sr_var)
    dsr = probabilistic_sharpe_ratio(arr, benchmark_sr=benchmark,
                                     observed_sr=observed, effective_n=n_eff)
    passes = dsr >= confidence

    dup = "" if n_eff >= n else (
        f" The series carries {n} values but only {n_eff:.1f} independent "
        f"observations, and the inference uses the latter.")
    provenance = {
        SR_VAR_FROM_TRIALS: (
            f"Var[SR] {sr_var:.4g} from {len(scored)} recorded trial scores, "
            f"{covered:.0%} of those charged"),
        SR_VAR_SUPPLIED: f"Var[SR] {sr_var:.4g} supplied by the caller",
        SR_VAR_UNDERCOVERED: (
            f"Var[SR] assumed 1.0: only {len(scored)} of {n_trials} charged "
            f"trials carry a score ({covered:.0%}, below the "
            f"{MIN_TRIAL_SCORE_COVERAGE:.0%} needed), and their measured "
            f"{measured_var:.5g} describes the arms that happened to be "
            f"recorded rather than the search"),
        SR_VAR_UNIT: ("Var[SR] assumed 1.0 -- no trial Sharpes were supplied, "
                      "so the benchmark is the conservative one"),
    }[source]

    if passes:
        interpretation = (
            f"After charging for {n_trials} trial(s) and for skew/kurtosis, the "
            f"probability the true Sharpe exceeds what the best of {n_trials} "
            f"lucky configurations would produce is {dsr:.1%} "
            f"({n_eff:.1f} independent observations; {provenance}).{dup}"
        )
    else:
        interpretation = (
            f"DSR {dsr:.1%} is below the {confidence:.0%} bar. Given {n_trials} "
            f"trial(s), an observed Sharpe of {observed:.3f} over "
            f"{n_eff:.1f} independent observations is not distinguishable from "
            f"the best of that many coin flips ({provenance}). Simplify the "
            f"model or gather more data -- do not search further.{dup}"
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
        effective_n=n_eff,
        sr_variance=sr_var,
        sr_variance_source=source,
        sr_variance_measured=measured_var,
        trials_scored=len(scored),
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

    Method: for each pair, take the best in-sample configuration and find its
    rank among all configurations out-of-sample, converted to a logit. PBO is
    the fraction of pairs where that logit is at or below zero, meaning the
    in-sample winner landed below the out-of-sample median.
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


def decile_profile(
    scores: "np.ndarray",
    forward: "np.ndarray",
    buckets: int = 10,
) -> Dict[str, object]:
    """Mean forward return by score decile, and whether it climbs.

    The information coefficient is a rank correlation, and a rank correlation
    can be healthy while the top of the ranking is not the best part of it. That
    matters here specifically: the engine trades the top decile, so a chart
    whose peak sits at decile 6 says the thing being traded is not the thing the
    IC is measuring.

    Measured on the shipped configuration over the 36-period holdout, deciles
    ran +1.97% at the bottom to +3.98% at the top, a spread of +2.01%, but with
    only 6 of 9 steps increasing and decile 6 (+4.06%) above decile 9. The
    spread is real and the monotonicity is not, and both belong in the record.
    """
    import numpy as _np
    import pandas as _pd

    s = _pd.Series(_np.asarray(scores, dtype="float64"))
    f = _pd.Series(_np.asarray(forward, dtype="float64"))
    keep = s.notna() & f.notna()
    s, f = s[keep], f[keep]
    if len(s) < buckets * 2:
        return {"buckets": [], "spread": None, "monotone_steps": None,
                "reason": f"{len(s)} observations; at least {buckets * 2} needed"}

    bucket = _pd.qcut(s.rank(method="first"), buckets, labels=False)
    means = f.groupby(bucket).mean()
    values = [float(v) for v in means.to_numpy()]
    steps = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
    peak = int(_np.argmax(values))
    return {
        "buckets": values,
        "spread": float(values[-1] - values[0]),
        "monotone_steps": steps,
        "max_steps": len(values) - 1,
        "peak_bucket": peak,
        "top_is_peak": peak == len(values) - 1,
    }


# ---------------------------------------------------------------------------
# Panel metrics. These moved here from `v2_panel.py` when the v2 engine was
# retired: they are generic cross-sectional statistics that the v3 panel and
# any future scorer need, and they were only ever in a version-named module
# because that is where they happened to be written first.
# ---------------------------------------------------------------------------

def rank_ic(panel: pd.DataFrame, label: str, score: str = "score") -> Tuple[float, float, int]:
    ics = []
    for _, g in panel.groupby("date", sort=True):
        a = g[score].to_numpy("float64"); b = g[label].to_numpy("float64")
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 60:
            continue
        ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            continue
        ics.append(float(np.corrcoef(ra, rb)[0, 1]))
    ics = np.asarray(ics)
    if len(ics) < 5:
        return float("nan"), float("nan"), len(ics)
    return (float(ics.mean()),
            float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), len(ics))


def quintile_spread(panel: pd.DataFrame, label: str, q: int = 5,
                    score: str = "score") -> Tuple[float, float, int]:
    """Top-fifth minus bottom-fifth realised return, per period.

    THE STATISTIC WITH POWER, and the reason it is the headline rather than the
    book's annual excess: a permuted-label test run before the original deploy
    put a ten-name book's five-year excess almost entirely inside its own null,
    while this spread sat six standard deviations outside it. Judge the scorer
    on the number that can tell signal from noise.
    """
    sp = []
    for _, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=[score, label])
        if len(g) < 100:
            continue
        k = max(len(g) // q, 5)
        o = g.sort_values(score, ascending=False)
        sp.append(float(o[label].head(k).mean() - o[label].tail(k).mean()))
    sp = np.asarray(sp)
    if len(sp) < 5:
        return float("nan"), float("nan"), len(sp)
    return (float(sp.mean()),
            float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))), len(sp))



#: Independent label windows the re-check needs before it issues a verdict.
#: The deploy was judged on 8.6 of them; a quarterly window holds about 1.5.
MIN_INDEPENDENT_WINDOWS = 8.0

#: What the deploy earned, on 2025-03-06 to 2026-08-17, evaluated once.
DEPLOY_REFERENCE = {"ic": 0.0451, "ic_t": 2.59, "spread": 0.0165, "spread_t": 2.56,
                    "book_excess_ann": 0.0241, "book_maxdd": -0.140}
