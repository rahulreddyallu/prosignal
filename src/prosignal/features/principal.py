"""The direction the DATA picks, rather than the one a blend asserts.

WHY THIS EXISTS. Three legitimate scorers in this repository disagree almost
completely about which name is best: measured across 380 panel dates, the
top-10 overlap between specifications ran 0.22 to 6.96 of 10, and the number of
DISTINCT names holding rank 1 across four specifications averaged 3.26 of 4.
The identity of #1 was a model choice.

The cause is not that the models carry different information. It is that the 22
factors span only about three independent directions -- measured, the first
three principal components carry 53.9% of the variance -- so any weighting is a
projection onto that low-dimensional space, and the top of the ranking is
exquisitely sensitive to which projection you pick. v3's 40/19/19/11/11 blend,
v9R's equal-risk nine, and an equal-weight blend are three arbitrary choices
among infinitely many.

RANK AGGREGATION DOES NOT FIX IT, and this was tested rather than assumed.
Mean-rank, median-rank and minimax-worst-rank consensus over four
specifications all produced top-1 excess indistinguishable from zero
(+0.136%, +0.289%, +0.012% at 21 sessions; NW t 0.16, 0.24, 0.02). Averaging
arbitrary projections yields another arbitrary projection.

WHAT DOES WORK is removing the choice. The first principal component of the
sign-corrected factor ranks is the direction of maximum variance in the
cross-section -- the one the covariance structure itself identifies. It has
NO FREE PARAMETERS. There is no weight to tune, no theme to cap, no blend to
renormalise, and therefore nothing to overfit.

MEASURED, on the 380-date research panel, 21-session forward excess over the
equal-weighted eligible universe, per date then averaged:

                       mean excess   median    hit     NW t
    v3 top-1              +0.012%   -0.967%   43.7%    0.01
    PC1 top-1             +1.973%   -0.020%   50.0%    1.72
    PC1 top-1, ENB>=2.5   +2.097%   +0.937%   56.9%    2.09
    PC1 top-3             +0.835%   +0.352%   51.0%    1.18
    PC1 top-10            +0.408%   +0.436%   54.4%    0.97

The selection curve INVERTS. Under v3 it rose with N -- breadth beat
concentration and a one-signal engine was the worst possible product. Under PC1
it falls with N, which is the shape a one-signal engine needs.

WHAT PC1 TURNS OUT TO BE, and it is not a black box. Mean loadings across every
panel date, in descending magnitude: voladj_mom_6_1 +0.354, prox_52w +0.353,
ulcer_120 +0.326, prox_52w_now +0.320, intraday_mom_126 +0.319,
voladj_mom_12_1 +0.318, mom_consist_126 +0.266, mom_3_1 +0.254. Signs are
applied from v3's own factor orientations, so a positive loading on ulcer_120
means LOW ulcer. The direction is consistent, low-drawdown trend -- quality
momentum, discovered rather than assumed. Loading signs matched the mean sign
on 100% of dates for all eight.

WHAT THIS IS NOT. t 2.09 is not a ship gate. Harvey, Liu and Zhu argue for a
bar near 3.0 in a factor-mining environment, the panel carries only about 90
independent 21-session windows, and the panel OVERLAPS the surface v3 was
selected on. PC1 was never fitted on that surface and has no parameters to
fit -- which makes it the least overfitted object tested here, not a validated
one. It is registered as a specification and made selectable. Making it the
default is a separate decision requiring a sealed window.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["FACTOR_SIGNS", "score_frame", "loadings", "PrincipalScore"]


def _v3_signs() -> Dict[str, int]:
    """v3's own factor orientations, so 'more is better' everywhere.

    Read from `v3.THEMES` rather than restated, because a sign that drifted
    apart from the scorer's would silently invert a factor inside PC1 while
    every test still passed.
    """
    from . import v3 as v3feat
    return {f: int(s) for t in v3feat.THEMES.values() for f, s in t.factors}


FACTOR_SIGNS = _v3_signs

#: A cross-section smaller than this cannot support an eigendecomposition whose
#: leading direction means anything.
MIN_NAMES = 30


class PrincipalScore:
    """The PC1 projection for one cross-section, with its loadings."""

    __slots__ = ("scores", "loadings", "variance_share", "n_names",
                 "unavailable")

    def __init__(self, scores=None, loadings=None, variance_share=None,
                 n_names=0, unavailable=None):
        self.scores: Optional[pd.Series] = scores
        self.loadings: Optional[pd.Series] = loadings
        #: Share of total variance carried by PC1 alone.
        self.variance_share: Optional[float] = variance_share
        self.n_names = n_names
        self.unavailable = unavailable

    def testable(self) -> bool:
        return self.unavailable is None and self.scores is not None

    def summary(self) -> str:
        if not self.testable():
            return f"principal direction unavailable: {self.unavailable}"
        top = self.loadings.reindex(
            self.loadings.abs().sort_values(ascending=False).index)[:5]
        bits = ", ".join(f"{k} {v:+.2f}" for k, v in top.items())
        return (f"PC1 over {self.n_names} names carries "
                f"{self.variance_share:.0%} of cross-sectional variance; "
                f"heaviest loadings {bits}")


def score_frame(raw_ranks: pd.DataFrame,
                signs: Optional[Dict[str, int]] = None,
                orient_against: Optional[pd.Series] = None) -> PrincipalScore:
    """Rank the cross-section on its own first principal direction.

    ``raw_ranks``      (names x factors) cross-sectional ranks, one column per
                       factor, UNSIGNED -- the orientation is applied here.
    ``signs``          factor -> +1/-1. Defaults to v3's own orientations.
    ``orient_against`` a reference series used to fix PC1's arbitrary sign.
                       An eigenvector is defined only up to sign, so without
                       this the ranking could invert between sessions for
                       purely numerical reasons. Defaults to the cross-sectional
                       mean of the signed factors, which is stable and needs no
                       other model present.
    """
    if raw_ranks is None or raw_ranks.empty:
        return PrincipalScore(unavailable="no factor frame supplied")

    sgn = signs if signs is not None else _v3_signs()
    cols = [c for c in raw_ranks.columns if c in sgn]
    if len(cols) < 3:
        return PrincipalScore(
            unavailable=f"only {len(cols)} oriented factors present; a principal "
                        f"direction needs at least three")

    M = raw_ranks[cols].astype(float).copy()
    for c in cols:
        M[c] = M[c] * float(sgn[c])

    # COLUMN-MEAN IMPUTATION, and it is deliberate. A name missing a factor is
    # placed at the cross-sectional average for it, which is the only fill that
    # adds no information: it neither rewards nor penalises the gap. Dropping
    # the name instead would delete most of the universe, since the fundamental
    # factors are absent on a large minority by construction.
    filled = M.fillna(M.mean())
    keep = filled.notna().all(axis=1)
    filled = filled[keep]
    if len(filled) < MIN_NAMES:
        return PrincipalScore(
            n_names=len(filled),
            unavailable=f"{len(filled)} names with complete factors, under the "
                        f"{MIN_NAMES} an eigendecomposition needs")

    X = filled.to_numpy(float)
    Xc = X - X.mean(axis=0)
    # Standardise: the factors are ranks and already share a scale, but a
    # column that happens to be degenerate on a given date would otherwise
    # contribute nothing and distort the normalisation.
    sd = Xc.std(axis=0)
    live = sd > 1e-12
    if int(live.sum()) < 3:
        return PrincipalScore(
            n_names=len(filled),
            unavailable="fewer than three factors vary across this cross-section")
    Xc = Xc[:, live] / sd[live]
    live_cols = [c for c, k in zip(cols, live) if k]

    try:
        _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        return PrincipalScore(n_names=len(filled),
                              unavailable=f"the decomposition did not converge: {exc}")

    var = sv ** 2
    share = float(var[0] / var.sum()) if var.sum() > 0 else None
    vec = Vt[0]
    proj = Xc @ vec

    # SIGN. An eigenvector is defined up to sign; without pinning it the whole
    # ranking could invert between sessions for numerical reasons alone.
    ref = (orient_against.reindex(filled.index).to_numpy(float)
           if orient_against is not None else Xc.mean(axis=1))
    ok = np.isfinite(ref) & np.isfinite(proj)
    if ok.sum() >= 8 and np.std(ref[ok]) > 0 and np.std(proj[ok]) > 0:
        if np.corrcoef(proj[ok], ref[ok])[0, 1] < 0:
            proj = -proj
            vec = -vec

    return PrincipalScore(
        scores=pd.Series(proj, index=filled.index, name="score"),
        loadings=pd.Series(vec, index=live_cols, name="loading"),
        variance_share=share,
        n_names=len(filled),
    )


def loadings(raw_ranks: pd.DataFrame, **kw) -> Optional[pd.Series]:
    """Just the loadings, for reporting what the direction turned out to be."""
    ps = score_frame(raw_ranks, **kw)
    return ps.loadings if ps.testable() else None
