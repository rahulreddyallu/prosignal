"""Fama-MacBeth estimation of theme coefficients, with hierarchical shrinkage.

WHY THIS EXISTS. The pooled ridge in `crossmodel` stacks every (symbol, date)
row into one design matrix and solves once. With ~33,000 rows it behaves as
though it has 33,000 independent observations. It does not. The panel is 70
cross-sections; within a date every name shares the same market, the same
policy news and the same flow, so the residuals are cross-sectionally
correlated and the honest sample size is closer to the number of DATES than the
number of rows. A pooled standard error divides by the square root of the wrong
number and reports significance that is not there.

Fama-MacBeth (1973) fixes this by estimating the cross-section separately on
each date and treating the resulting slope series as the sample. The slope on
date t is one observation. The mean of the series is the estimate, and its
standard error comes from the dispersion of T slopes -- so T, not N*T, sets the
confidence. On this panel that is 70, not 33,569.

The slopes are then autocorrelated, because a 63-session label sampled every 21
sessions overlaps its next two neighbours: three consecutive slopes are
estimated over windows that share most of their return path. Newey-West (1987)
with `ceil(horizon/step) - 1` lags charges for that overlap. Uncorrected, the
standard error is understated for exactly the same reason the pooled one was.

WHAT THE SHRINKAGE IS FOR. Seven themes estimated on 70 dates produce seven
noisy numbers, and the loudest of them is loud partly by luck. Jensen, Kelly &
Pedersen (2023) show that the apparent replication crisis in factor research
largely dissolves under a hierarchical prior: factors are not independent
claims, they are draws from a family, and the right estimate for any one of
them borrows strength from the others. That is empirical Bayes, and here it has
a convenient property -- when the between-theme dispersion is entirely
explained by estimation error, every weight goes to zero and the estimator
collapses exactly onto the equal-weight control. The control is not a separate
model bolted on for comparison; it is this model's own null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FMResult", "fama_macbeth", "newey_west_se", "hierarchical_shrink",
    "equal_weight_lambda", "rolling_lambda", "score_from_lambda",
    "gated_shrink", "is_degenerate",
    "THEME_PRIOR_SIGN", "MIN_CROSS_SECTION", "SIGNIFICANCE_FLOOR",
]

#: A date with fewer names than this cannot support a stable cross-sectional
#: regression on five or more themes, and its slope would enter the time series
#: as noise with the same weight as a full cross-section. Dropped instead.
MIN_CROSS_SECTION = 30

#: The sign each theme is expected to carry BEFORE any data is seen. Used only
#: to orient themes onto a common axis so the hierarchical prior can pool them
#: -- a theme priced negatively is not evidence against a theme priced
#: positively, it is the same evidence pointing the other way.
#:
#: `None` means no defensible literature prior, and such a theme shrinks toward
#: ZERO rather than toward the pooled mean. That is the conservative reading:
#: without a prior orientation there is nothing to borrow strength from.
#:
#: `risk` is None deliberately, and the reason is measured rather than assumed.
#: Its two members are -0.35 correlated within date, so the equal-weight
#: composite CANCELS the common low-risk axis and keeps the residual -- names
#: whose drawdown is shallow relative to what their beta implies. That residual
#: is what predicts here, and it is not the betting-against-beta anomaly:
#:
#:      composite                       IC       ICIR      t
#:      (beta + max_dd)/2  = risk_f   +0.0326   +0.384   +3.21
#:      (-beta + max_dd)/2 = BAB      +0.0224   +0.089   +0.75
#:      max_dd alone                  +0.0389   +0.176   +1.47
#:
#: The properly oriented low-risk composite is the WEAKEST of the three. So the
#: Frazzini-Pedersen prior cannot be claimed for this column, and setting the
#: prior from the table above would be reading the sign off the same data the
#: shrinkage is then supposed to discipline.
THEME_PRIOR_SIGN: Dict[str, Optional[int]] = {
    # Jegadeesh & Titman (1993). Winners keep winning over 3-12 months.
    "mom": +1,
    # Jegadeesh (1990), Lehmann (1990); residual variant Blitz, Huij, Lansdorp
    # & Martens (2013). A name that has run up over the last month gives some
    # back, so the raw run-up is priced NEGATIVELY.
    "reversal": -1,
    # Bali, Cakici & Whitelaw (2011). Lottery demand is paid for, so the more
    # lottery-like the name, the lower the expected return.
    "lottery": -1,
    # Frazzini & Pedersen (2014): leverage-constrained investors bid up high
    # beta, so low beta earns more per unit of risk. Agarwalla, Jacob, Varma &
    # Vasudevan (2014) find the BAB factor earns significant positive returns
    # in India and DOMINATES the size, value and momentum factors, so the prior
    # is claimable in this market specifically rather than by analogy.
    #
    # This is a prior taken from the literature BEFORE looking at the panel,
    # which is what makes it a prior. The note above explains why no prior
    # could be claimed for the old composite: (beta + max_dd)/2 was not the
    # BAB portfolio and its orientation was read off the same data the
    # shrinkage was meant to discipline. Splitting the family is what makes a
    # literature prior legitimate here.
    "beta": -1,
    # No literature analogue for a drawdown-DEPTH cross-section, and the
    # measurement disagrees with itself across labels. No prior.
    "drawdown": None,
    # Skewness preference is a real channel (Bali, Cakici & Whitelaw 2011;
    # Boyer, Mitton & Vorkink 2010) and points the same way as lottery demand:
    # investors overpay for positive skew. Prior -1, and the gate is expected
    # to zero it anyway at t -0.94.
    "skew": -1,
    # No literature analogue -- delivery percentage is an Indian market
    # microstructure disclosure with no US counterpart. The engine's thesis is
    # that delivered volume is real accumulation rather than intraday churn,
    # which is a prior about direction even without a paper behind it.
    "delivery": +1,
    # Fama & French (1992).
    "value": +1,
    # Novy-Marx (2013); members already sign-aligned via NEGATED_IN_FAMILY.
    "quality": +1,
}


@dataclass
class FMResult:
    """Per-date cross-sectional slopes and the inference that follows them."""

    features: List[str]
    slopes: pd.DataFrame                      #: dates x features
    lam: Dict[str, float] = field(default_factory=dict)
    se: Dict[str, float] = field(default_factory=dict)
    t_stat: Dict[str, float] = field(default_factory=dict)
    n_dates: int = 0
    nw_lags: int = 0
    skipped_dates: int = 0

    def significant(self, threshold: float = 2.0) -> List[str]:
        return [f for f in self.features if abs(self.t_stat.get(f, 0.0)) >= threshold]

    def summary(self) -> str:
        parts = ", ".join(
            f"{f.removesuffix('_f')} {self.lam[f]:+.4f} (t {self.t_stat[f]:+.2f})"
            for f in sorted(self.features, key=lambda k: -abs(self.t_stat.get(k, 0)))
        )
        return f"Fama-MacBeth over {self.n_dates} cross-sections, " \
               f"Newey-West {self.nw_lags} lags: {parts}"


def newey_west_se(series: np.ndarray, lags: int) -> float:
    """Standard error of the MEAN of an autocorrelated series.

    Bartlett kernel, per Newey & West (1987). With overlapping labels the slope
    series is positively autocorrelated, so the naive s/sqrt(T) understates the
    true sampling error -- consecutive slopes are partly the same observation
    seen twice.
    """
    x = np.asarray(series, dtype="float64")
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3:
        return float("nan")
    d = x - x.mean()
    gamma0 = float(d @ d) / t
    total = gamma0
    for lag in range(1, min(lags, t - 1) + 1):
        cov = float(d[lag:] @ d[:-lag]) / t
        total += 2.0 * (1.0 - lag / (lags + 1.0)) * cov
    # A Bartlett-weighted sum can go negative in small samples. The variance of
    # a mean cannot, so fall back to the uncorrected figure rather than
    # returning a nan and silently dropping the theme from inference.
    if total <= 0:
        total = gamma0
    return math.sqrt(total / t)


def _ols_slopes(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """One cross-sectional regression with an intercept. None if degenerate."""
    n, p = x.shape
    if n <= p + 1:
        return None
    design = np.column_stack([np.ones(n), x])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(beta).all():
        return None
    return beta[1:]


def fama_macbeth(
    panel: pd.DataFrame,
    features: Sequence[str],
    target: str = "label_rank",
    horizon: int = 63,
    step: int = 21,
    nw_lags: Optional[int] = None,
    window: Optional[int] = None,
    min_cross_section: int = MIN_CROSS_SECTION,
) -> Optional[FMResult]:
    """Estimate theme coefficients date by date, then average the slopes.

    ``window`` keeps only the most recent N dates, which is what makes it
    ROLLING: a coefficient estimated over the last three years is allowed to
    differ from one estimated over the last ten. Everything before the window
    is dropped rather than down-weighted, because a coefficient that needs a
    decade of history to look significant is not a coefficient this engine can
    trade on a 63-session horizon.
    """
    cols = [c for c in features if c in panel.columns]
    if not cols or target not in panel.columns or "date" not in panel.columns:
        return None

    frame = panel[["date", target] + cols].dropna()
    if frame.empty:
        return None
    dates = sorted(frame["date"].unique())
    if window is not None and len(dates) > window:
        dates = dates[-int(window):]
        frame = frame[frame["date"].isin(set(dates))]

    rows: List[np.ndarray] = []
    kept: List[object] = []
    skipped = 0
    for date, group in frame.groupby("date", sort=True, observed=True):
        if len(group) < min_cross_section:
            skipped += 1
            continue
        beta = _ols_slopes(group[cols].to_numpy("float64"),
                           group[target].to_numpy("float64"))
        if beta is None:
            skipped += 1
            continue
        rows.append(beta)
        kept.append(date)
    if len(rows) < 3:
        return None

    slopes = pd.DataFrame(rows, index=pd.Index(kept, name="date"), columns=cols)
    # A 63-session label sampled every 21 sessions overlaps its next two
    # neighbours; those are the lags that carry the induced autocorrelation.
    lags = int(nw_lags) if nw_lags is not None else max(0, math.ceil(horizon / max(step, 1)) - 1)

    result = FMResult(features=list(cols), slopes=slopes, n_dates=len(slopes),
                      nw_lags=lags, skipped_dates=skipped)
    for c in cols:
        s = slopes[c].to_numpy("float64")
        mean = float(np.nanmean(s))
        se = newey_west_se(s, lags)
        result.lam[c] = mean
        result.se[c] = se
        result.t_stat[c] = float(mean / se) if se and np.isfinite(se) and se > 0 else float("nan")
    return result


def hierarchical_shrink(
    result: FMResult,
    prior_sign: Optional[Mapping[str, Optional[int]]] = None,
    toward: str = "zero",
    tau2_from: Optional[FMResult] = None,
) -> Dict[str, float]:
    """Empirical-Bayes shrinkage of theme coefficients toward their common mean.

    Each theme is oriented by its prior sign so that the pool is over claims
    pointing the same way, then

        lambda_shrunk = mu + tau^2 / (tau^2 + se^2) * (lambda - mu)

    where ``mu`` is the oriented grand mean and ``tau^2`` is the between-theme
    variance of the TRUE coefficients, estimated as the observed dispersion of
    the estimates less the average estimation variance (Morris 1983). A theme
    measured precisely keeps most of its own estimate; a theme measured badly is
    pulled to the pool.

    When the observed dispersion is no larger than what estimation error alone
    would produce, ``tau^2`` is zero, every weight is zero, and every oriented
    theme collapses onto the grand mean -- which IS the equal-weight control.
    The 1/N benchmark is the null this estimator nests, not a rival.

    ``toward`` picks the shrinkage target, and the choice matters more than it
    looks:

    ``"prior_mean"`` is the Jensen-Kelly-Pedersen reading -- themes are
    exchangeable draws from a distribution with a positive oriented mean, so an
    imprecise theme inherits the pool's average. That is only safe when the
    orientation is trustworthy. It is not here. `lottery` carries a documented
    NEGATIVE prior and measures IC +0.0485 in this universe, and because its
    standard error is large the pool hands it nearly the full prior-oriented
    mean -- a confident coefficient built out of nothing but the assumption.
    Blended, that bet cancels momentum outright: `mom_f - lottery_f` reads IC
    +0.0031 against `mom_f` alone at +0.0481.

    ``"zero"`` is the default for that reason. An imprecisely measured theme
    goes to zero, not to the average of the themes that WERE measured. It is
    the more conservative reading of the same hierarchy, and it imposes no sign
    the data has not earned.
    """
    signs = dict(THEME_PRIOR_SIGN if prior_sign is None else prior_sign)
    if toward not in ("zero", "prior_mean"):
        raise ValueError(f"toward must be 'zero' or 'prior_mean', not {toward!r}")

    def sign_for(col: str) -> Optional[int]:
        return signs.get(col.removesuffix("_f"), None)

    var = {c: (result.se[c] ** 2) for c in result.features
           if np.isfinite(result.se.get(c, float("nan")))}
    if not var:
        return dict(result.lam)

    # BETWEEN-THEME DISPERSION IS ESTIMATED BEFORE SELECTION.
    #
    # `gated_shrink` kills every theme under the significance floor and then
    # calls this on the SURVIVORS. Estimating tau^2 from that set is the
    # winner's curse: the survivors were chosen for having large |t|, so
    # E[lambda^2] among them is inflated by the selection, tau^2 comes out too
    # big, the weight tau^2/(tau^2 + se^2) sits too close to 1, and the themes
    # that passed the gate are barely shrunk at all.
    #
    # Measured on the shipped fit, the haircut this produced was
    # delivery 1.6%, reversal 3.3%, lottery 12.1% -- an empirical-Bayes
    # discipline removing 1.6% of a coefficient is not disciplining anything,
    # and the gate was doing the entire job while the estimator took the credit.
    #
    # `tau2_from` carries the PRE-SELECTION result, so the pool is estimated
    # over every theme that was measured and only the shrinkage is applied to
    # those that survived. The point estimates and standard errors are
    # unchanged; what changes is which sample the prior is learned from.
    source = tau2_from if tau2_from is not None else result
    # STRICTLY POSITIVE, not merely finite. The pool is inverse-variance
    # weighted, so a theme enters it as 1/se^2 and a theme with se == 0 is a
    # division by zero rather than an infinitely precise measurement.
    #
    # se == 0 means the theme's slope was identical on every cross-section,
    # which is not a well-measured theme -- it is a degenerate column the panel
    # could not move. It carries no information about BETWEEN-theme dispersion,
    # so it is excluded from the tau^2 pool rather than dominating it.
    #
    # Reachable on the production fit path (fit_coefficients -> gated_shrink ->
    # here), where it would have raised ZeroDivisionError and been caught as
    # "cross-sectional model failed", taking the ranking down for the run. It
    # went unnoticed while every family had two or more members; single-member
    # themes (`reversal`, and now `skew`, `beta`, `drawdown`) make a constant
    # slope series reachable on a short or degenerate panel.
    src_var = {c: (source.se[c] ** 2) for c in source.features
               if np.isfinite(source.se.get(c, float("nan")))
               and source.se[c] ** 2 > 0.0}
    if not src_var:
        src_var = {c: v for c, v in var.items() if v > 0.0}
    if not src_var:
        return dict(result.lam)
        source = result

    # tau^2 by DerSimonian & Laird (1986), NOT by the plain moment estimator
    # mean(lambda^2) - mean(se^2). The plain version assumes the themes are
    # measured equally well, and here they are not: standard errors differ by
    # more than an order of magnitude between `mom` and `lottery`. Averaging
    # raw variances lets the single worst-measured theme drive tau^2 to zero
    # and take a theme sitting at t = 6 down with it -- which is exactly what
    # happened, zeroing an entire walk-forward fold. Inverse-variance weighting
    # gives each theme a say proportional to how well it is known.
    if toward == "prior_mean":
        pooled = [c for c in source.features
                  if sign_for(c) is not None and c in src_var]
        if len(pooled) < 2:
            return dict(result.lam)
        values = np.array([sign_for(c) * source.lam[c] for c in pooled], dtype="float64")
        w = np.array([1.0 / src_var[c] for c in pooled], dtype="float64")
        mu = float((w * values).sum() / w.sum())
        q = float((w * (values - mu) ** 2).sum())
        denom = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (q - (len(pooled) - 1)) / denom) if denom > 0 else 0.0
    else:
        # Target is zero, so E[lambda^2] = tau^2 + se^2 and the weighted moment
        # condition is sum(w * lambda^2) = tau^2 * sum(w) + k.
        cols = [c for c in source.features if c in src_var]
        values = np.array([source.lam[c] for c in cols], dtype="float64")
        w = np.array([1.0 / src_var[c] for c in cols], dtype="float64")
        mu = 0.0
        tau2 = max(0.0, float((w * values ** 2).sum() - len(cols)) / float(w.sum()))

    out: Dict[str, float] = {}
    for c in result.features:
        v = var.get(c)
        if v is None or not np.isfinite(v):
            out[c] = result.lam[c]
            continue
        weight = tau2 / (tau2 + v) if (tau2 + v) > 0 else 0.0
        s = sign_for(c)
        if toward == "zero" or s is None:
            # Nothing to borrow from, so the target is the null.
            out[c] = weight * result.lam[c]
        else:
            out[c] = s * (mu + weight * (s * result.lam[c] - mu))
    return out


def equal_weight_lambda(
    features: Sequence[str],
    prior_sign: Optional[Mapping[str, Optional[int]]] = None,
    scale: float = 1.0,
) -> Dict[str, float]:
    """The 1/N control arm: every theme weighted equally, oriented by prior.

    DeMiguel, Garlappi & Uppal (2009) found 1/N beat fourteen optimising rules
    out of sample, because estimation error in the optimiser cost more than the
    optimisation gained. Any estimated coefficient set has to beat this to earn
    the estimation. A theme with no prior orientation contributes nothing here,
    which is the honest reading of a bet whose direction is unknown.
    """
    signs = dict(THEME_PRIOR_SIGN if prior_sign is None else prior_sign)
    cols = list(features)
    n = max(len(cols), 1)
    out: Dict[str, float] = {}
    for c in cols:
        s = signs.get(c.removesuffix("_f"), None)
        out[c] = 0.0 if s is None else float(s) * scale / n
    return out


def rolling_lambda(result: FMResult, window: int) -> pd.DataFrame:
    """Trailing mean of each theme's slope. The input to the decay monitor.

    A theme whose rolling coefficient has walked to zero is not a theme with a
    small coefficient; it is a theme that used to work.
    """
    if result.slopes.empty:
        return result.slopes
    return result.slopes.rolling(int(window), min_periods=max(3, int(window) // 2)).mean()


def score_from_lambda(frame: pd.DataFrame, lam: Mapping[str, float]) -> pd.Series:
    """Linear score from a coefficient dict. Missing columns contribute nothing."""
    total = pd.Series(0.0, index=frame.index, dtype="float64")
    for col, weight in lam.items():
        if col in frame.columns and weight:
            total = total + frame[col].fillna(0.0).astype("float64") * float(weight)
    return total


#: A theme that cannot clear two standard errors on its OWN training window does
#: not steer the book. Pre-committed, and deliberately not tuned: a floor of 1.65
#: -- the one-sided 5% bar -- measured better out of sample here (IC +0.0553
#: against +0.0510, top-decile +1.35% against +1.25%), and was rejected for that
#: exact reason. Choosing the threshold that scored best on the fifty dates used
#: to evaluate it is how a backtest is manufactured. Both are counted as trials.
SIGNIFICANCE_FLOOR = 2.0

#: Curvature of the optional continuous taper, t^2 / (t^2 + c). At c = 4.0 the
#: half-weight point sits exactly at |t| = 2, so the taper SMOOTHS the shipped
#: cliff rather than replacing it with a different rule.
TAPER_C = 4.0

#: Below this the coefficient is zero outright even under the taper. A theme
#: the window genuinely cannot measure must not steer the book at a fifth
#: weight any more than at full weight.
TAPER_HARD_FLOOR = 1.0


def gated_shrink(
    result: FMResult,
    floor: float = SIGNIFICANCE_FLOOR,
    toward: str = "zero",
    prior_sign: Optional[Mapping[str, Optional[int]]] = None,
    taper: bool = False,
    taper_c: float = TAPER_C,
    taper_hard_floor: float = TAPER_HARD_FLOOR,
) -> Dict[str, float]:
    """Kill themes below the significance floor, then shrink what survives.

    Two different jobs, deliberately not merged. The floor is a RISK CONTROL:
    it refuses to let a theme the training window could not measure move real
    money, whatever its point estimate. The shrinkage is an ESTIMATOR: among
    themes that did clear the bar, it weights by precision.

    Returns every input feature, with the killed ones at exactly zero. An
    all-zero result is a legitimate answer -- it means no theme was measurable
    -- and the caller must refuse to trade rather than score a flat book.

    ``taper`` replaces the CLIFF with a continuous weight. The cliff is a real
    weakness: it makes a traded coefficient a step function of a noisy
    statistic, and `risk` sat at t +1.86 on the live fit against +2.45 on the
    rebuild -- the same theme worth nothing or nearly everything depending only
    on the window. Under the taper a theme keeps t^2 / (t^2 + c) of its shrunk
    coefficient, so |t| = 2 keeps half and |t| = 1 a fifth, and
    ``taper_hard_floor`` still zeroes anything below it outright.

    OFF by default. The argument for it is structural, but it moves live
    coefficients, which makes it an estimator change and therefore a trial --
    see `estimator.significance_taper` in the config for why that distinction
    is being enforced rather than argued.
    """
    if taper:
        keep = [c for c in result.features
                if np.isfinite(result.t_stat.get(c, float("nan")))
                and abs(result.t_stat[c]) >= float(taper_hard_floor)]
    else:
        keep = [c for c in result.features
                if np.isfinite(result.t_stat.get(c, float("nan")))
                and abs(result.t_stat[c]) >= float(floor)]
    out = {c: 0.0 for c in result.features}
    if not keep:
        return out
    sub = FMResult(
        features=keep, slopes=result.slopes[keep], n_dates=result.n_dates,
        nw_lags=result.nw_lags, skipped_dates=result.skipped_dates,
        lam={c: result.lam[c] for c in keep},
        se={c: result.se[c] for c in keep},
        t_stat={c: result.t_stat[c] for c in keep},
    )
    # The FULL result is handed over as the tau^2 source, so the pool the prior
    # is learned from is every theme that was measured -- not only the ones the
    # floor let through. See hierarchical_shrink.
    out.update(hierarchical_shrink(sub, prior_sign=prior_sign, toward=toward,
                                   tau2_from=result))
    if taper:
        # Applied AFTER the shrinkage, not instead of it. The two answer
        # different questions: hierarchical_shrink weights by how precisely a
        # theme is measured relative to the pool, and this weights by how far
        # its own t is from being indistinguishable from zero.
        c = float(taper_c)
        for col in keep:
            t = float(result.t_stat[col])
            out[col] *= (t * t) / (t * t + c)
    return out


def is_degenerate(lam: Mapping[str, float], tol: float = 1e-12) -> bool:
    """True when no theme carries weight, so the score would be flat."""
    return all(abs(float(v)) <= tol for v in lam.values())
