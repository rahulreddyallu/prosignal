"""How many INDEPENDENT pieces of evidence stand behind a candidate.

THE PROBLEM THIS SOLVES. The shipped composite blends 22 factors in 5 themes.
A name at the top of it typically has ten or more factors pointing the same way,
and the card prints all of them. That reads as ten confirmations. It is not:
`mom_3_1`, `mom_2_0`, `voladj_mom_12_1`, `mom_consist_126` and
`intraday_mom_126` are five measurements of one phenomenon, and on the live
cross-section they correlate above 0.8. Counting them as five is the single
easiest way for this engine to talk itself into a position.

WHAT IS COMPUTED. Meucci's Effective Number of Bets, applied to evidence rather
than to capital. Given the candidate's exposure vector `w` over the supporting
factors and the cross-sectional correlation matrix `S` of those factors
measured on TODAY's eligible universe:

    diversification distribution   d_i = (E^T w)_i^2 * lambda_i / (w^T S w)
    effective independent count    ENB = exp(-sum_i d_i ln d_i)

where `S = E diag(lambda) E^T`. `d` is a probability distribution over the
uncorrelated principal directions of the evidence, and ENB is the exponential
of its Shannon entropy -- the number of equally-weighted independent bets that
would produce the same concentration. It is bounded in [1, n]: ENB = 1 means
every factor is one bet wearing n hats, ENB = n means n genuinely orthogonal
confirmations.

Meucci, "Managing Diversification" (Risk, 2009) defines the diversification
distribution and the entropy index. The torsion here is the plain PCA one
rather than minimum-torsion: PCA biases explanatory power onto the leading
components, which for THIS use is the conservative direction -- it understates
independence, so a candidate has to be genuinely broad to score well.

WHY THE CORRELATION IS MEASURED TODAY, NOT FROM THE FIT. Factor correlations
are not stable; they rise in stress, which is exactly when the engine most needs
to know its evidence has collapsed onto one direction. A correlation matrix
frozen at fit time would report yesterday's independence during today's
collapse. This reads the cross-section the run is actually ranking.

NOTHING HERE IS A GATE. It produces a number. What the number is worth is
decided in `conviction.gate`, and the threshold there is UNVALIDATED until the
selection-precision study runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "EvidenceCount",
    "effective_independent_count",
    "theme_matrix",
    "member_matrix",
    "evidence_for",
]

#: Correlations below this are treated as zero when building the evidence
#: matrix. Sampling noise on a 100-400 name cross-section puts |rho| up to
#: about 0.1 on genuinely unrelated factors, and leaving that in inflates the
#: apparent coupling of factors that are in fact independent -- which biases ENB
#: DOWN. Shrinking it to zero is the honest direction only because the
#: alternative overstates redundancy; it is a noise floor, not a model.
_RHO_FLOOR = 0.10

#: A factor must be measured on at least this fraction of the cross-section for
#: its correlations to mean anything. Below it the pairwise overlap is too thin
#: and the factor is dropped from the count rather than contributing a
#: correlation estimated on a handful of names.
_MIN_COVERAGE = 0.50


@dataclass(frozen=True)
class EvidenceCount:
    """The result of counting one candidate's evidence."""

    #: How many factors point the candidate's way at all.
    nominal: int
    #: How many INDEPENDENT directions those factors span. Always <= nominal.
    effective: float
    #: Per-direction share of the evidence. Sums to 1. Its largest entry is the
    #: share carried by the single most dominant direction.
    distribution: tuple = ()
    #: The factors that were counted, in the order of `weights`.
    names: tuple = ()
    #: The candidate's exposure to each counted factor.
    weights: tuple = ()
    #: Set when the count could not be formed. `effective` is then 0.0 and the
    #: candidate must NOT be treated as having passed -- absent evidence is not
    #: independent evidence.
    unavailable: Optional[str] = None

    @property
    def concentration(self) -> float:
        """Share of the evidence carried by its single largest direction.

        1.0 means every supporting factor is the same bet.
        """
        if not self.distribution:
            return 1.0
        return float(max(self.distribution))

    @property
    def redundancy_ratio(self) -> float:
        """nominal / effective. 1.0 is fully independent; 5.0 is five hats."""
        if self.effective <= 0:
            return float("inf")
        return float(self.nominal) / float(self.effective)

    def testable(self) -> bool:
        return self.unavailable is None and self.effective > 0.0


def _corr(matrix: np.ndarray) -> Optional[np.ndarray]:
    """Cross-sectional correlation, computed pairwise-complete.

    `np.corrcoef` drops the whole column on a single NaN, which on this panel
    removes any factor with one missing name -- and the fundamental themes are
    missing on a large minority of the universe by construction. Pairwise is
    the only version that survives the real coverage pattern.
    """
    n_cols = matrix.shape[1]
    if n_cols == 0:
        return None
    out = np.eye(n_cols, dtype=float)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            a, b = matrix[:, i], matrix[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if int(ok.sum()) < 8:
                # Too thin to estimate. Treated as INDEPENDENT, which raises
                # ENB -- so this branch is recorded by the caller dropping the
                # factor for coverage before it can be reached in bulk.
                continue
            av, bv = a[ok], b[ok]
            sa, sb = av.std(), bv.std()
            if sa <= 0 or sb <= 0:
                continue
            r = float(((av - av.mean()) * (bv - bv.mean())).mean() / (sa * sb))
            if not np.isfinite(r):
                continue
            r = max(-1.0, min(1.0, r))
            if abs(r) < _RHO_FLOOR:
                r = 0.0
            out[i, j] = out[j, i] = r
    return out


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues to zero and rescale to unit diagonal.

    A pairwise-complete correlation matrix is not guaranteed positive
    semi-definite -- different pairs are estimated on different subsets. The
    eigendecomposition below needs one that is, and a negative eigenvalue would
    make `d_i` negative and the entropy undefined.
    """
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 0.0, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    diag = np.sqrt(np.clip(np.diag(fixed), 1e-12, None))
    return fixed / np.outer(diag, diag)


def effective_independent_count(
    weights: Sequence[float],
    panel: np.ndarray,
    names: Sequence[str],
) -> EvidenceCount:
    """Meucci ENB over the supporting evidence.

    ``weights``  the candidate's exposure to each factor. Only POSITIVE entries
                 are counted: a factor pointing against the candidate is not
                 evidence for it, and including it would let disagreement raise
                 the independence score.
    ``panel``    (n_names, n_factors) cross-sectional factor values for today's
                 eligible universe. Supplies the correlation structure.
    ``names``    factor labels, aligned with ``weights`` and ``panel`` columns.
    """
    w_all = np.asarray(weights, dtype=float)
    if w_all.size == 0 or panel.size == 0:
        return EvidenceCount(0, 0.0, unavailable="no factors supplied")
    if panel.shape[1] != w_all.size:
        return EvidenceCount(
            0, 0.0, unavailable="panel and weights disagree on factor count")

    n_rows = panel.shape[0]
    coverage = np.isfinite(panel).sum(axis=0) / max(n_rows, 1)

    keep = (w_all > 0) & np.isfinite(w_all) & (coverage >= _MIN_COVERAGE)
    if not keep.any():
        return EvidenceCount(
            0, 0.0, unavailable="no supporting factor has enough coverage")

    w = w_all[keep]
    sub = panel[:, keep]
    kept_names = tuple(n for n, k in zip(names, keep) if k)
    nominal = int(keep.sum())

    if nominal == 1:
        return EvidenceCount(1, 1.0, (1.0,), kept_names, tuple(w))

    corr = _corr(sub)
    if corr is None:
        return EvidenceCount(
            nominal, 0.0, unavailable="the correlation matrix would not form")
    corr = _nearest_psd(corr)

    total = float(w @ corr @ w)
    if not np.isfinite(total) or total <= 0:
        return EvidenceCount(
            nominal, 0.0, unavailable="the evidence has no measurable variance")

    vals, vecs = np.linalg.eigh(corr)
    # d_i = (E^T w)_i^2 * lambda_i / (w^T S w). This is the diversification
    # distribution: the share of the evidence's total variance carried by the
    # i-th uncorrelated direction.
    proj = vecs.T @ w
    d = (proj ** 2) * np.clip(vals, 0.0, None) / total
    d = np.clip(d, 0.0, None)
    s = d.sum()
    if not np.isfinite(s) or s <= 0:
        return EvidenceCount(
            nominal, 0.0, unavailable="the diversification distribution is empty")
    d = d / s

    # Shannon entropy with the 0 ln 0 = 0 convention, exponentiated.
    nz = d[d > 0]
    enb = float(np.exp(-(nz * np.log(nz)).sum()))
    # Numerical guard: ENB is bounded in [1, n] by construction and only leaves
    # that range through floating-point error.
    enb = max(1.0, min(float(nominal), enb))

    return EvidenceCount(nominal, enb, tuple(float(x) for x in d),
                         kept_names, tuple(float(x) for x in w))


# =============================================================================
# Building the panel out of what Stage 4 already produced
# =============================================================================


def theme_matrix(ranked_scores) -> tuple:
    """(panel, names) of standardised THEME scores across the universe.

    Themes are the level the model actually fits a coefficient on, so this is
    the level at which "independent evidence" means what the score means.
    """
    names: List[str] = []
    for s in ranked_scores:
        for k in s.factors:
            if k not in names:
                names.append(k)
    if not names:
        return np.zeros((0, 0)), ()
    rows = []
    for s in ranked_scores:
        row = []
        for k in names:
            f = s.factors.get(k)
            v = None if f is None else f.standardised
            row.append(float(v) if v is not None and np.isfinite(v) else np.nan)
        rows.append(row)
    return np.asarray(rows, dtype=float), tuple(names)


def member_matrix(ranked_scores) -> tuple:
    """(panel, names) of individual FACTOR ranks across the universe.

    This is the level the redundancy problem actually lives at. The card lists
    members -- `mom_3_1`, `mom_2_0`, `voladj_mom_12_1` -- and a reader counts
    them. Counting them here is what makes the inflation visible.
    """
    names: List[str] = []
    for s in ranked_scores:
        for f in s.factors.values():
            for m in f.members:
                if m.name not in names:
                    names.append(m.name)
    if not names:
        return np.zeros((0, 0)), ()
    index = {n: i for i, n in enumerate(names)}
    rows = []
    for s in ranked_scores:
        row = [np.nan] * len(names)
        for f in s.factors.values():
            for m in f.members:
                if m.rank is not None and np.isfinite(m.rank):
                    row[index[m.name]] = float(m.rank)
        rows.append(row)
    return np.asarray(rows, dtype=float), tuple(names)


@dataclass(frozen=True)
class EvidenceProfile:
    """Both levels of the count for one candidate, plus what it means."""

    themes: EvidenceCount
    members: EvidenceCount

    def testable(self) -> bool:
        return self.themes.testable() and self.members.testable()

    def summary(self) -> str:
        if not self.testable():
            reason = (self.themes.unavailable or self.members.unavailable
                      or "evidence could not be counted")
            return f"Independent evidence NOT TESTABLE: {reason}"
        return (
            f"{self.members.nominal} factors in {self.themes.nominal} themes "
            f"support this name, and they span "
            f"{self.members.effective:.1f} independent directions "
            f"({self.themes.effective:.1f} at theme level). "
            f"The largest single direction carries "
            f"{self.members.concentration:.0%} of the evidence."
        )


def evidence_for(score, ranked_scores,
                 theme_panel=None, member_panel=None) -> EvidenceProfile:
    """Count one candidate's independent evidence at both levels.

    The panels may be passed in when scoring many candidates against the same
    cross-section, which is the normal case -- building them is O(universe) and
    doing it per candidate is the difference between one pass and fifteen.
    """
    tp, tnames = theme_panel if theme_panel is not None else theme_matrix(ranked_scores)
    mp, mnames = member_panel if member_panel is not None else member_matrix(ranked_scores)

    # THEME weights are the signed contributions the card already prints, so
    # the count is taken over exactly the evidence the reader is shown.
    tw = []
    for k in tnames:
        f = score.factors.get(k)
        c = None if f is None else f.contribution
        tw.append(float(c) if c is not None and np.isfinite(c) else 0.0)

    # MEMBER weights are the ranks in [-1, +1]. A rank of +0.9 is strong
    # support; -0.9 supports the opposite case and is excluded by the positive
    # filter inside the count.
    mw_index = {n: 0.0 for n in mnames}
    for f in score.factors.values():
        for m in f.members:
            if m.rank is not None and np.isfinite(m.rank):
                mw_index[m.name] = float(m.rank)
    mw = [mw_index[n] for n in mnames]

    return EvidenceProfile(
        themes=effective_independent_count(tw, tp, tnames),
        members=effective_independent_count(mw, mp, mnames),
    )
