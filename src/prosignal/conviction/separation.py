"""Is the leader actually AHEAD, or merely first?

THE PROBLEM. `composite_score` is `(rank - 1) / (n - 1)`. Its distribution is
uniform every single session by construction, so rank 1 always scores ~1.00 and
the gap to rank 2 is always ~1/n. Read on that scale, every day looks equally
decisive and no threshold on it can ever bind -- which is exactly why Stage 8's
`min_composite_score` and `min_universe_percentile` cut zero names out of 36 on
the live run.

Separation has to be measured on the RAW predicted score, `composite_raw`,
before the rank transform flattens it. And it has to be measured in units that
are comparable across days, because raw spread varies with the ridge penalty
and with how much the model has to say. So every gap here is divided by the
cross-sectional dispersion of the raw score on the same day:

    gap_to(k)  =  (raw_1 - raw_k) / sigma_raw

That is a z-gap. "The leader is 0.8 sigma clear of the field" means the same
thing in March 2020 and in a quiet August, which "the leader scored 0.997" does
not.

WHY MAD AND NOT SD. The raw score distribution has the fat right tail the model
is built to produce, and the standard deviation of a fat-tailed cross-section is
itself dominated by the few names being measured. Median absolute deviation,
scaled to be consistent with sd under normality (x 1.4826), gives a dispersion
estimate the leader cannot inflate by being extreme.

NOTHING HERE IS A GATE. Thresholds live in `conviction.gate` and are
UNVALIDATED until the selection-precision study runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["Separation", "measure", "dispersion"]

#: Consistency constant making MAD comparable to sd for a normal sample.
_MAD_TO_SD = 1.4826


def dispersion(values: Sequence[float]) -> Optional[float]:
    """Robust cross-sectional dispersion of the raw scores.

    Returns None when the cross-section is too small or degenerate to have a
    scale -- in which case no gap can be expressed in sigma units and the
    caller must report NOT TESTABLE rather than assuming a scale.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    if arr.size < 8:
        return None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * _MAD_TO_SD
    if mad > 0 and np.isfinite(mad):
        return mad
    # A degenerate MAD (more than half the names on one value) is not
    # automatically a dead cross-section -- fall back to sd, and only give up
    # when that is dead too.
    sd = float(arr.std())
    return sd if sd > 0 and np.isfinite(sd) else None


@dataclass(frozen=True)
class Separation:
    """How far one candidate stands clear of the alternatives."""

    ticker: str
    rank: int
    #: Raw predicted score, before the rank transform.
    raw: float
    #: Cross-sectional dispersion the gaps are denominated in.
    sigma: Optional[float]
    #: (raw - raw of the next name down) / sigma. The margin over the name that
    #: would take this slot instead.
    gap_to_next: Optional[float] = None
    #: (raw - raw of rank 5) / sigma. Whether the lead survives past one name.
    gap_to_fifth: Optional[float] = None
    #: (raw - median raw) / sigma. How far above the typical name it sits.
    gap_to_median: Optional[float] = None
    unavailable: Optional[str] = None

    def testable(self) -> bool:
        return self.unavailable is None and self.sigma is not None

    def summary(self) -> str:
        if not self.testable():
            return f"Separation NOT TESTABLE: {self.unavailable}"

        def f(x):
            return "n/a" if x is None else f"{x:+.2f}"

        return (
            f"Rank {self.rank}, {f(self.gap_to_median)} sigma above the median "
            f"name, {f(self.gap_to_next)} sigma clear of the next candidate and "
            f"{f(self.gap_to_fifth)} sigma clear of rank 5."
        )


def measure(ranked_scores, ticker: str) -> Separation:
    """Measure one candidate's separation from the field.

    ``ranked_scores`` must be the FULL eligible cross-section in rank order --
    the dispersion is a property of the universe, and measuring it on a
    pre-filtered shortlist would shrink sigma toward zero and make every
    surviving gap look enormous.
    """
    raws: List[float] = []
    idx = -1
    for i, s in enumerate(ranked_scores):
        raws.append(float(s.composite_raw))
        if s.ticker == ticker:
            idx = i
    if idx < 0:
        return Separation(ticker, 0, 0.0, None,
                          unavailable="the name is not in the ranked universe")

    me = raws[idx]
    rank = int(getattr(ranked_scores[idx], "rank", idx + 1) or idx + 1)
    sigma = dispersion(raws)
    if sigma is None:
        return Separation(
            ticker, rank, me, None,
            unavailable="the cross-section has no measurable dispersion, so a "
                        "gap cannot be expressed in comparable units")

    def gap(other: Optional[float]) -> Optional[float]:
        if other is None or not np.isfinite(other):
            return None
        return float((me - other) / sigma)

    nxt = raws[idx + 1] if idx + 1 < len(raws) else None
    fifth = raws[4] if len(raws) > 4 else None
    # Measured against rank 5 as the field, EXCLUDING the candidate itself
    # when it is inside the top 5 -- otherwise a name at rank 3 is compared to
    # a list it is a member of and the gap shrinks for the wrong reason.
    if fifth is not None and idx < 5:
        fifth = raws[5] if len(raws) > 5 else None
    med = float(np.median(raws)) if raws else None

    return Separation(ticker, rank, me, sigma,
                      gap_to_next=gap(nxt),
                      gap_to_fifth=gap(fifth),
                      gap_to_median=gap(med))
