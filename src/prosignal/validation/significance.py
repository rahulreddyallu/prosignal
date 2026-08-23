"""Statistics that survive the way this panel is actually sampled.

Every performance number in this project is a mean over observations that
OVERLAP. The label is a 63-session forward return and panel dates are 21
sessions apart, so each observation shares two thirds of its window with the
next and one third with the one after. The naive standard error of such a mean
is understated by a factor of the square root of three, and the t-statistic
built from it is overstated by the same amount.

This is not a subtlety at this sample size. The holdout's reported t of 3.20
becomes 1.85 once corrected, which is the difference between clearing the
project's stated t >= 3.0 bar and missing it.

Two corrections are provided because they answer different questions.
Newey-West is the standard parametric fix and assumes the autocorrelation dies
off within the chosen lag. The stationary bootstrap of Politis and Romano
(1994) assumes almost nothing: it resamples blocks of random length, so any
dependence shorter than the average block survives resampling, and it gives an
empirical distribution rather than a point estimate with an asymptotic
justification that 15 observations do not earn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "OverlapAdjusted", "BootstrapResult",
    "BOOTSTRAP_MIN_N", "analytic_vif", "overlap_lag", "newey_west_t",
    "stationary_bootstrap", "summarise",
]


@dataclass(frozen=True)
class OverlapAdjusted:
    mean: float
    naive_t: float
    adjusted_t: float
    #: Variance inflation factor implied by the estimated autocorrelation.
    vif: float
    #: Observations divided by the inflation factor.
    effective_n: float
    n: int
    lag: int

    @property
    def inflation(self) -> float:
        """How much the naive t overstates the adjusted one."""
        return self.naive_t / self.adjusted_t if self.adjusted_t else float("nan")


#: Below this many observations the stationary bootstrap is not calibrated on
#: overlapping data. Measured against simulated noise at the 63/21 sampling
#: scheme, its 95% interval excludes zero 25-30% of the time at n=15 for every
#: block length from 2 to 8, falling to 15% at n=30 and 8.5% at n=120. The
#: failure is sample size, not block choice: resampling blocks of an already
#: overlapping series does not reproduce the null when there are only fifteen
#: blocks to draw from.
BOOTSTRAP_MIN_N = 30


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    #: Share of resampled paths with a mean above zero.
    share_positive: float
    #: Two-sided p-value from the sign of the resampled distribution.
    p_value: float
    draws: int
    mean_block: float
    #: Set when n is below BOOTSTRAP_MIN_N. The interval is still returned --
    #: it is informative as a description of the sample -- but it must not be
    #: read as a significance test.
    uncalibrated: bool = False

    @property
    def excludes_zero(self) -> bool:
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    @property
    def is_evidence(self) -> bool:
        """Whether the interval may be used to argue significance."""
        return self.excludes_zero and not self.uncalibrated


def overlap_lag(horizon_sessions: int, step_sessions: int) -> int:
    """How many neighbouring observations a single label reaches into.

    A 63-session label sampled every 21 sessions overlaps the next two, so the
    autocorrelation is non-zero out to lag 2 by construction -- before any real
    serial dependence in returns is considered.
    """
    if horizon_sessions <= 0 or step_sessions <= 0:
        raise ValueError("horizon and step must both be positive")
    return max(int(np.ceil(horizon_sessions / step_sessions)) - 1, 0)


def analytic_vif(horizon_sessions: int, step_sessions: int, n: int) -> float:
    """Variance inflation implied by the sampling scheme alone.

    When observations are h-session sums sampled every s sessions, the
    autocorrelation of the sampled series is rho_k = (m - k) / m for k < m,
    where m = h / s -- before any real serial dependence in returns. This is
    arithmetic, not an estimate.

    It is used in preference to estimating the same quantity from the data
    because at fifteen observations the estimate is badly biased: measured
    against simulated noise, Newey-West recovers 1.74 where the true value is
    3.00, and the resulting t still rejects a true null 20% of the time at a
    nominal 5%. Estimating a quantity that is known by construction spends
    scarce degrees of freedom to get a worse answer.
    """
    m = horizon_sessions / step_sessions
    if m <= 1:
        return 1.0
    vif = 1.0
    for k in range(1, int(np.ceil(m))):
        rho = max((m - k) / m, 0.0)
        # Small-sample weighting: a lag-k covariance is estimated from n - k
        # pairs, so it cannot contribute its full asymptotic weight.
        vif += 2.0 * rho * max(1.0 - k / n, 0.0)
    return float(vif)


def newey_west_t(
    series: Sequence[float],
    *,
    lag: Optional[int] = None,
    horizon_sessions: Optional[int] = None,
    step_sessions: Optional[int] = None,
    use_analytic_vif: bool = True,
) -> OverlapAdjusted:
    """t-statistic for the mean, with a Newey-West heteroskedasticity- and
    autocorrelation-consistent standard error.

    Pass the lag directly, or pass the horizon and step and let it be derived
    from the overlap. Deriving it is preferred: the lag is then a fact about
    the sampling scheme rather than a tuning choice, and a lag chosen to make
    a result look better is exactly the kind of decision this module exists to
    prevent.
    """
    x = np.asarray([v for v in series if v is not None and np.isfinite(v)],
                   dtype="float64")
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")

    if lag is None:
        if horizon_sessions is None or step_sessions is None:
            raise ValueError("pass lag, or both horizon_sessions and step_sessions")
        lag = overlap_lag(horizon_sessions, step_sessions)
    lag = int(min(lag, n - 1))

    mean = float(x.mean())
    dev = x - mean
    gamma0 = float(dev @ dev) / n
    # Scale-relative, because an exact zero is not what a flat series produces:
    # the mean of twenty copies of 0.05 is not exactly 0.05 in binary floating
    # point, leaving a variance around 1e-36 that clears a bare `> 0` test and
    # then divides into a t-statistic of 1e17.
    scale = max(abs(mean), float(np.abs(x).max()), 1e-12)
    if gamma0 <= (scale * 1e-9) ** 2:
        raise ValueError("series has zero variance")

    # Bartlett kernel: weights fall linearly to zero at the truncation lag, so
    # the estimated long-run variance stays positive semi-definite.
    long_run = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(dev[k:] @ dev[:-k]) / n
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    # A sharply negative autocovariance can drive the estimate below zero at
    # tiny n. Falling back to the naive variance is conservative in the sense
    # that it never REPORTS more significance than Newey-West would.
    if long_run <= 0:
        long_run = gamma0

    naive_se = float(np.sqrt(gamma0 / n))
    # Prefer the inflation the sampling scheme implies over the one estimated
    # from a short, noisy series. The estimate is kept when the overlap is not
    # known, and the LARGER of the two is taken when it is -- a series with
    # real serial dependence on top of the overlap deserves the bigger penalty.
    est_vif = long_run / gamma0
    if use_analytic_vif and horizon_sessions and step_sessions:
        vif = max(est_vif, analytic_vif(horizon_sessions, step_sessions, n))
    else:
        vif = est_vif
    nw_se = float(np.sqrt(gamma0 * vif / n))
    return OverlapAdjusted(
        mean=mean,
        naive_t=mean / naive_se if naive_se else float("nan"),
        adjusted_t=mean / nw_se if nw_se else float("nan"),
        vif=float(vif),
        effective_n=float(n / vif) if vif > 0 else float(n),
        n=n,
        lag=lag,
    )


def stationary_bootstrap(
    series: Sequence[float],
    *,
    mean_block: Optional[float] = None,
    horizon_sessions: Optional[int] = None,
    step_sessions: Optional[int] = None,
    draws: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260823,
) -> BootstrapResult:
    """Politis-Romano stationary bootstrap of the mean.

    Blocks have geometrically distributed length with the given mean, and the
    series wraps, so the resampled series is stationary and any dependence
    shorter than a typical block is preserved. The default block length is the
    overlap span, which is the dependence we know exists by construction.

    Reported as a confidence interval and a share of positive paths rather
    than as a single statistic, because with fifteen observations the shape of
    the distribution is the finding.
    """
    x = np.asarray([v for v in series if v is not None and np.isfinite(v)],
                   dtype="float64")
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")

    if mean_block is None:
        if horizon_sessions is not None and step_sessions is not None:
            mean_block = float(overlap_lag(horizon_sessions, step_sessions) + 1)
        else:
            mean_block = float(max(2, round(n ** (1 / 3))))
    mean_block = float(max(1.0, min(mean_block, n)))
    p_new = 1.0 / mean_block

    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype="float64")
    for d in range(draws):
        idx = np.empty(n, dtype="int64")
        idx[0] = rng.integers(n)
        # Start a new block with probability 1/mean_block; otherwise continue
        # the current one, wrapping at the end.
        cont = rng.random(n) >= p_new
        jump = rng.integers(0, n, size=n)
        for i in range(1, n):
            idx[i] = (idx[i - 1] + 1) % n if cont[i] else jump[i]
        means[d] = x[idx].mean()

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    share_pos = float((means > 0).mean())
    # Two-sided p against the null that the mean is zero, read off the
    # resampled distribution rather than assumed normal.
    p = 2.0 * min(share_pos, 1.0 - share_pos)
    return BootstrapResult(
        mean=float(x.mean()), ci_low=lo, ci_high=hi,
        share_positive=share_pos, p_value=min(p, 1.0),
        draws=int(draws), mean_block=mean_block,
        uncalibrated=n < BOOTSTRAP_MIN_N,
    )


def summarise(
    series: Sequence[float],
    *,
    horizon_sessions: int,
    step_sessions: int,
    t_bar: float = 3.0,
    draws: int = 10_000,
) -> Tuple[OverlapAdjusted, BootstrapResult, str]:
    """Both corrections and a one-line verdict against the configured bar."""
    nw = newey_west_t(series, horizon_sessions=horizon_sessions,
                      step_sessions=step_sessions)
    bs = stationary_bootstrap(series, horizon_sessions=horizon_sessions,
                              step_sessions=step_sessions, draws=draws)
    passes = nw.adjusted_t >= t_bar
    boot = (f"bootstrap 95% CI [{bs.ci_low:+.4f}, {bs.ci_high:+.4f}]"
            + (" -- NOT CALIBRATED at this sample size, descriptive only"
               if bs.uncalibrated else
               f", {bs.share_positive:.0%} of paths positive"))
    verdict = (
        f"t {nw.naive_t:.2f} -> {nw.adjusted_t:.2f} after correcting for "
        f"{nw.lag}-lag overlap ({nw.n} observations, {nw.effective_n:.1f} "
        f"effective); {boot}. "
        f"{'CLEARS' if passes else 'MISSES'} the t >= {t_bar:.1f} bar."
    )
    return nw, bs, verdict
