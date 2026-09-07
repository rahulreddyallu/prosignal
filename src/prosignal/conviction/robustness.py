"""Does the candidate survive a reasonable family of model specifications?

THE PROBLEM. The shipped composite blends five theme sub-scores at fitted
weights (momentum 0.40, ownership 0.19, quality 0.19, reversal 0.11, risk 0.11).
Those weights are estimates. A name that is rank 1 under exactly one weighting
and rank 40 under a slightly different one has not been selected by the
evidence; it has been selected by the third decimal place of a coefficient.

WHAT IS PERTURBED, and why these. Every perturbation here is a change the
evidence cannot distinguish from the shipped specification:

  * DROP ONE THEME. The strongest available test, and the one that maps onto
    the ablation the research plan asks for. A name whose rank collapses when
    momentum is removed IS a momentum bet, whatever else its card lists.
  * TILT THE WEIGHTS +/-25%, one theme at a time. The fitted coefficients carry
    standard errors comfortably wider than this; the shipped vector is one draw
    from a distribution, and a conclusion that only holds at the point estimate
    is not a conclusion.
  * EQUAL-WEIGHT THE THEMES. The specification with no estimation error at all.
    A candidate that survives it is not relying on the fit being right.

WHY NOT LOOKBACK PERTURBATION. Changing a factor's lookback means recomputing
the factor from prices for the whole universe, which is a Stage 4 rerun per
perturbation. That belongs in the offline robustness study, not in a live
scan that has to finish in seconds. What is done here is exactly what can be
recomputed from the theme sub-scores already in hand -- and the theme weights
are where the estimation error actually lives, since the members inside a theme
are averaged, not fitted.

WHAT COMES OUT. The share of specifications under which the candidate stays
inside the shortlist, and how far its rank moves. Neither is a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["RankEnvelope", "envelope", "specifications"]

#: How far each theme weight is tilted, one at a time, in both directions.
_TILT = 0.25

#: A candidate is "surviving" a specification while it stays inside this rank.
#: Deliberately WIDER than the number of BUY slots: the question is whether the
#: name remains a serious candidate under the alternative, not whether it keeps
#: the exact position, and a top-2 test would score every specification as a
#: failure for the name at rank 3 under the shipped one.
DEFAULT_SURVIVE_RANK = 10


def _theme_names(ranked_scores) -> Tuple[str, ...]:
    names: List[str] = []
    for s in ranked_scores:
        for k in s.factors:
            if k not in names:
                names.append(k)
    return tuple(names)


def _panel(ranked_scores, themes: Sequence[str]):
    """(z, w, tickers) -- theme z-scores and per-name blend weights.

    The weights are PER NAME, not the frozen fitted vector: the shipped blend
    re-caps over the themes a name actually has, so a four-theme name carries
    different weights from a five-theme one. Perturbing the frozen vector and
    applying it to every name would compare the candidate against a
    specification the engine never runs.
    """
    tick = [s.ticker for s in ranked_scores]
    z = np.full((len(ranked_scores), len(themes)), np.nan)
    w = np.zeros((len(ranked_scores), len(themes)))
    for i, s in enumerate(ranked_scores):
        for j, k in enumerate(themes):
            f = s.factors.get(k)
            if f is None:
                continue
            if f.standardised is not None and np.isfinite(f.standardised):
                z[i, j] = float(f.standardised)
            if f.weight is not None and np.isfinite(f.weight):
                w[i, j] = float(f.weight)
    return z, w, tick


def specifications(themes: Sequence[str]) -> List[Tuple[str, np.ndarray]]:
    """The multiplier vectors applied to the per-name theme weights.

    Each is (label, multiplier per theme). The shipped specification is the
    vector of ones and is NOT included -- it is the baseline these are
    compared against, not one of the alternatives.
    """
    n = len(themes)
    out: List[Tuple[str, np.ndarray]] = []
    for j, name in enumerate(themes):
        drop = np.ones(n)
        drop[j] = 0.0
        out.append((f"drop {name}", drop))
        up = np.ones(n)
        up[j] = 1.0 + _TILT
        out.append((f"{name} +{_TILT:.0%}", up))
        dn = np.ones(n)
        dn[j] = 1.0 - _TILT
        out.append((f"{name} -{_TILT:.0%}", dn))
    out.append(("equal-weight themes", np.full(n, np.nan)))  # sentinel
    return out


@dataclass(frozen=True)
class RankEnvelope:
    """Where a candidate ranks across a family of specifications."""

    ticker: str
    base_rank: int
    #: Rank under each alternative, keyed by its label.
    ranks: Dict[str, int]
    #: Fraction of specifications keeping the name inside `survive_rank`.
    survival: float
    survive_rank: int
    median_rank: Optional[float] = None
    worst_rank: Optional[int] = None
    #: The specification that hurt most, and where it put the name.
    worst_spec: Optional[str] = None
    unavailable: Optional[str] = None

    def testable(self) -> bool:
        return self.unavailable is None and bool(self.ranks)

    @property
    def fragile(self) -> bool:
        """True when the conclusion depends on the exact specification."""
        return self.testable() and self.survival < 0.75

    def summary(self) -> str:
        if not self.testable():
            return f"Robustness NOT TESTABLE: {self.unavailable}"
        return (
            f"Holds a top-{self.survive_rank} place under "
            f"{self.survival:.0%} of {len(self.ranks)} alternative "
            f"specifications (median rank {self.median_rank:.0f}, worst "
            f"{self.worst_rank} under '{self.worst_spec}')."
        )


def envelope(ranked_scores, ticker: str,
             survive_rank: int = DEFAULT_SURVIVE_RANK) -> RankEnvelope:
    """Re-rank the whole universe under each alternative and find the name."""
    themes = _theme_names(ranked_scores)
    if len(themes) < 2:
        return RankEnvelope(
            ticker, 0, {}, 0.0, survive_rank,
            unavailable="fewer than two themes; nothing to perturb")

    z, w, tick = _panel(ranked_scores, themes)
    try:
        base_idx = tick.index(ticker)
    except ValueError:
        return RankEnvelope(
            ticker, 0, {}, 0.0, survive_rank,
            unavailable="the name is not in the ranked universe")

    # A missing theme contributes nothing, exactly as the shipped blend does.
    zf = np.nan_to_num(z, nan=0.0)
    present = np.isfinite(z)

    def rank_of(weights: np.ndarray) -> Optional[int]:
        # Renormalise per name over the themes that name actually has, which is
        # what the shipped blend does. Without it, dropping a theme would
        # penalise every name that HAS it relative to names that never did.
        wsum = (weights * present).sum(axis=1)
        ok = wsum > 0
        composite = np.full(len(tick), -np.inf)
        composite[ok] = (zf[ok] * weights[ok]).sum(axis=1) / wsum[ok]
        if not np.isfinite(composite[base_idx]):
            return None
        # Rank 1 = highest. Ties resolve to the better rank, which is the
        # conservative direction for a survival count.
        return int((composite > composite[base_idx]).sum()) + 1

    base_rank = rank_of(w) or 0
    ranks: Dict[str, int] = {}
    for label, mult in specifications(themes):
        if np.isnan(mult).all():
            # Equal weight across the themes each name possesses.
            alt = present.astype(float)
        else:
            alt = w * mult
        r = rank_of(alt)
        if r is not None:
            ranks[label] = r

    if not ranks:
        return RankEnvelope(
            ticker, base_rank, {}, 0.0, survive_rank,
            unavailable="no alternative specification could be scored")

    values = list(ranks.values())
    survived = sum(1 for r in values if r <= survive_rank)
    worst = max(values)
    worst_spec = next(k for k, v in ranks.items() if v == worst)
    return RankEnvelope(
        ticker=ticker,
        base_rank=base_rank,
        ranks=ranks,
        survival=survived / len(values),
        survive_rank=survive_rank,
        median_rank=float(np.median(values)),
        worst_rank=worst,
        worst_spec=worst_spec,
    )
