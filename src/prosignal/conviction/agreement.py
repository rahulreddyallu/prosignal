"""Do the engine's OTHER models agree, or does only the incumbent like it?

THE ASSET THIS USES. The repository contains three legitimate cross-sectional
scorers and ranks on one:

    fitted_composite   the Fama-MacBeth model. ~103KB of tested code in
                       `features/crossmodel.py` and `features/famamacbeth.py`,
                       FITTED ON EVERY RUN and then discarded -- the ranking
                       policy replaces its predictions and keeps only their
                       index as a population filter.
    v3_composite       the incumbent. 22 factors in 5 themes.
    v9r_core           nine factors, equal risk contribution, unneutralised.
                       The ONLY model here with an out-of-sample number:
                       +9.50% net active on a sealed 2012-2017 window at
                       Newey-West t +1.87 against a PRE-REGISTERED bar of 2.0.

WHY THIS IS EVIDENCE AND NOT A VOTE. None of these becomes the ranking. v9R in
particular FAILED its ship gate -- positive and underpowered is a failure, and
reading it as a pass would be exactly the error `docs/MODEL_v9R.md` warns
against. What the three give is a measurement the engine could never make
before: how much of the incumbent's opinion survives a change of model.

The incumbent's own ranking is a point estimate from a search over 960
configurations. A name it puts at rank 2 that the other two also place near the
top is evidenced very differently from one that only it likes -- and the second
case is exactly what a search over 960 configurations produces by construction.

WHAT IS MEASURED. Each alternative is converted to a rank (1 = best) over the
names it covers, and the candidate's rank under each is recorded. The summary
is the WORST rank across specifications, because a conviction claim is only as
strong as the model that likes the candidate least. Mean rank would let one
enthusiastic specification hide two indifferent ones.

Coverage is not agreement: a model that does not cover the name has not
endorsed it. Those are reported as absent and never counted as consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["ModelAgreement", "measure"]


@dataclass(frozen=True)
class ModelAgreement:
    """Where the candidate ranks under every specification available."""

    ticker: str
    #: source -> rank (1 = best) under that specification.
    ranks: Dict[str, int] = field(default_factory=dict)
    #: source -> how many names that specification covered.
    universe: Dict[str, int] = field(default_factory=dict)
    #: Specifications that exist but do not cover this name.
    absent: tuple = ()
    #: The rank threshold "agrees" is measured against.
    top_k: int = 10
    unavailable: Optional[str] = None

    def testable(self) -> bool:
        return self.unavailable is None and len(self.ranks) >= 2

    @property
    def worst_rank(self) -> Optional[int]:
        """The least enthusiastic specification. A claim is only this strong."""
        return max(self.ranks.values()) if self.ranks else None

    @property
    def best_rank(self) -> Optional[int]:
        return min(self.ranks.values()) if self.ranks else None

    @property
    def agreeing(self) -> int:
        """How many specifications put the name inside `top_k`."""
        return sum(1 for r in self.ranks.values() if r <= self.top_k)

    @property
    def consensus(self) -> Optional[float]:
        """Share of covering specifications that place it inside `top_k`."""
        if not self.ranks:
            return None
        return self.agreeing / len(self.ranks)

    def summary(self) -> str:
        if not self.testable():
            return (f"Model agreement NOT TESTABLE: "
                    f"{self.unavailable or 'only one specification available'}")
        parts = ", ".join(f"{k} #{v}" for k, v in sorted(self.ranks.items(),
                                                         key=lambda x: x[1]))
        tail = ""
        if self.absent:
            tail = (f" Not covered by: {', '.join(self.absent)} -- absence is "
                    f"not endorsement.")
        return (f"{self.agreeing} of {len(self.ranks)} model specifications "
                f"place it inside the top {self.top_k} ({parts}). The least "
                f"enthusiastic puts it at #{self.worst_rank}.{tail}")


def _rank_of(ticker: str, scores: Mapping[str, float]) -> Optional[int]:
    """Rank of ``ticker``, 1 = highest score. None when uncovered."""
    mine = scores.get(ticker)
    if mine is None or not np.isfinite(mine):
        return None
    return int(sum(1 for v in scores.values()
                   if v is not None and np.isfinite(v) and v > mine)) + 1


def measure(ticker: str,
            alternatives: Mapping[str, Mapping[str, float]],
            top_k: int = 10,
            min_universe: int = 8) -> ModelAgreement:
    """Rank ``ticker`` under every specification that covers it."""
    if not alternatives:
        return ModelAgreement(
            ticker, top_k=top_k,
            unavailable="the run recorded no alternative specifications")

    ranks: Dict[str, int] = {}
    universe: Dict[str, int] = {}
    absent: List[str] = []
    for source, scores in alternatives.items():
        if not scores or len(scores) < min_universe:
            # A specification that scored almost nobody has not formed a view
            # of the cross-section, and a rank inside it would be meaningless.
            absent.append(source)
            continue
        r = _rank_of(ticker, scores)
        if r is None:
            absent.append(source)
            continue
        ranks[source] = r
        universe[source] = len(scores)

    if len(ranks) < 2:
        return ModelAgreement(
            ticker, ranks, universe, tuple(absent), top_k,
            unavailable=(f"only {len(ranks)} specification(s) cover this name; "
                         f"agreement needs at least two"))
    return ModelAgreement(ticker, ranks, universe, tuple(absent), top_k)
