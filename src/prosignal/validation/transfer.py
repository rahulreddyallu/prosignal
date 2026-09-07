"""Can a book actually carry the IC the ranking measures?

THE GAP THIS CLOSES. This engine measures the ORDERING well -- rank IC +0.058
out of sample at h=21, overlap-corrected t +2.28 -- and builds a six-name
long-only book off the very top of it. Those are different objects and the
repository's history is largely the story of the two being confused.

Grinold's IR = TC x IC x sqrt(breadth) names the missing term. The TRANSFER
COEFFICIENT is the correlation between the positions a book actually takes and
the positions the signal implies. For a long-only book of six names out of
several hundred it is close to zero by construction: the signal has a view on
every name in the cross-section, including a strong negative view on the bottom
decile (-2.31% out of sample at h=63, the part that generalises BEST), and the
book can express none of it.

So a measured IC that does not appear in the book is not necessarily a broken
signal. It can be a book that cannot hold the signal's opinion. Those two
diagnoses call for opposite responses -- fix the model, or fix the
construction -- and nothing here could tell them apart.

WHAT THIS MODULE IS AND IS NOT. It measures what a portfolio SHAPE would have
earned, gross of the frictions that shape cannot pay. It is a construction
diagnostic, not a strategy: `shortable` defaults to nothing, no borrow cost is
modelled, and `LONG_ONLY` is the only shape this engine can trade today. India
has no retail stock-borrow market worth the name, so the short leg here is a
measurement of where the information is, not a proposal.

Every figure is per date and then averaged, never pooled. Pooling
cross-sections mixes the ordering with the drift of the cross-sectional mean,
and it is the second one that makes a long-only book look like it works in a
rising market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Below this many scored names a decile is not a decile.
MIN_NAMES = 100

LONG_ONLY = "LONG_ONLY"
LONG_SHORT = "LONG_SHORT"


@dataclass(frozen=True)
class Shape:
    """A portfolio shape, as weights over score percentile.

    `lo` and `hi` are percentile bounds in [0, 1] on the ranked score, best
    last. `weight` is the total book weight given to that band; a negative
    weight is a short.
    """

    name: str
    bands: Tuple[Tuple[float, float, float], ...]
    kind: str = LONG_ONLY

    def is_long_only(self) -> bool:
        return all(w >= 0 for _, _, w in self.bands)


def top_k_shape(n_names: int, k: int) -> Shape:
    """The shipped book: the best `k` names, equally weighted."""
    frac = k / max(n_names, 1)
    return Shape(f"top{k}", ((1.0 - frac, 1.0, 1.0),))


#: The shapes worth comparing, and why each is here.
def standard_shapes(n_names: int, k: int = 6) -> List[Shape]:
    return [
        # What ships.
        top_k_shape(n_names, k),
        # The top decile, which is what every top-decile statistic describes
        # and is NOT what a six-name book holds.
        Shape("D10", ((0.9, 1.0, 1.0),)),
        # D6-D8. Out of sample the decile profile peaks here, not at D10 --
        # see `results._decile_profile`. This is the shape that asks whether
        # that is worth anything after it stops being a table.
        Shape("D6-D8", ((0.5, 0.8, 1.0),)),
        # The top half. Nearly the whole long side of the signal's view.
        Shape("top-half", ((0.5, 1.0, 1.0),)),
        # Where the out-of-sample information actually is. NOT TRADEABLE HERE
        # -- see the module docstring on borrow -- and it is the number that
        # says whether the long-only constraint is what is binding.
        Shape("D10-D1", ((0.0, 0.1, -1.0), (0.9, 1.0, 1.0)), kind=LONG_SHORT),
        Shape("top-half minus bottom-half",
              ((0.0, 0.5, -1.0), (0.5, 1.0, 1.0)), kind=LONG_SHORT),
    ]


@dataclass
class TransferResult:
    shape: str
    kind: str
    n_dates: int
    #: Mean per-period return of the shape, in excess of the equal-weight
    #: cross-section. Gross: no cost, no borrow, no impact.
    gross_excess: float
    t_stat: float
    #: Correlation between the shape's weights and the score, per date and
    #: averaged. Grinold's transfer coefficient: 1.0 is a book that holds
    #: exactly what the signal says, 0.0 is a book that cannot express it.
    transfer_coefficient: float
    #: Names the shape holds on an average date. A shape the engine cannot
    #: staff is not a plan.
    avg_positions: float
    tradeable_long_only: bool

    def to_dict(self) -> Dict[str, object]:
        return {"shape": self.shape, "kind": self.kind,
                "n_dates": self.n_dates, "gross_excess": self.gross_excess,
                "t_stat": self.t_stat,
                "transfer_coefficient": self.transfer_coefficient,
                "avg_positions": self.avg_positions,
                "tradeable_long_only": self.tradeable_long_only}


def _weights_for(shape: Shape, n: int) -> np.ndarray:
    """Per-name weights for one cross-section, names ordered worst to best."""
    pct = (np.arange(n) + 0.5) / n
    w = np.zeros(n, dtype="float64")
    for lo, hi, weight in shape.bands:
        band = (pct >= lo) & (pct < hi) if hi < 1.0 else (pct >= lo)
        k = int(band.sum())
        if k:
            w[band] = weight / k
    return w


def evaluate(panel: pd.DataFrame, label: str, shapes: Sequence[Shape],
             score: str = "score", stride: int = 21,
             horizon: Optional[int] = None) -> List[TransferResult]:
    """Every shape, per date, averaged. Gross of cost by construction.

    The t-statistic is overlap-corrected when `horizon` is given: dates
    `stride` sessions apart against a longer label produce observations that
    are not independent, and the naive figure is inflated by about sqrt(VIF).
    """
    from . import significance as sig

    per_date: Dict[str, List[float]] = {s.name: [] for s in shapes}
    tc: Dict[str, List[float]] = {s.name: [] for s in shapes}
    held: Dict[str, List[int]] = {s.name: [] for s in shapes}

    for _, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=[score, label])
        n = len(g)
        if n < MIN_NAMES:
            continue
        order = g[score].rank(method="first").to_numpy().argsort()
        rets = g[label].to_numpy(dtype="float64")[order]
        scores = g[score].to_numpy(dtype="float64")[order]
        mean = float(rets.mean())
        for s in shapes:
            w = _weights_for(s, n)
            # EXCESS OVER THE EQUAL-WEIGHT CROSS-SECTION, so a rising market
            # does not read as skill. A long-short shape nets to zero weight
            # and its excess is its own return.
            gross = float(w @ rets)
            net_weight = float(w.sum())
            per_date[s.name].append(gross - net_weight * mean)
            if np.std(w) > 0 and np.std(scores) > 0:
                tc[s.name].append(float(np.corrcoef(w, scores)[0, 1]))
            held[s.name].append(int((w != 0).sum()))

    out: List[TransferResult] = []
    for s in shapes:
        v = np.asarray(per_date[s.name], dtype="float64")
        if v.size < 3:
            continue
        sd = float(v.std(ddof=1))
        t = float(v.mean() / sd * np.sqrt(v.size)) if sd > 0 else float("nan")
        if horizon:
            vif = sig.analytic_vif(int(horizon), int(stride), int(v.size))
            t = t / np.sqrt(vif) if np.isfinite(t) else t
        out.append(TransferResult(
            shape=s.name, kind=s.kind, n_dates=int(v.size),
            gross_excess=float(v.mean()), t_stat=t,
            transfer_coefficient=(float(np.mean(tc[s.name]))
                                  if tc[s.name] else float("nan")),
            avg_positions=float(np.mean(held[s.name])) if held[s.name] else 0.0,
            tradeable_long_only=s.is_long_only()))
    return out


def table(results: Sequence[TransferResult]) -> str:
    head = (f"{'shape':<28} {'kind':<11} {'gross excess':>13} {'t':>7} "
            f"{'transfer':>9} {'names':>7}  tradeable")
    lines = [head, "-" * len(head)]
    for r in sorted(results, key=lambda x: -x.gross_excess):
        lines.append(
            f"{r.shape:<28} {r.kind:<11} {r.gross_excess:>+13.4%} "
            f"{r.t_stat:>+7.2f} {r.transfer_coefficient:>+9.3f} "
            f"{r.avg_positions:>7.0f}  "
            + ("yes" if r.tradeable_long_only else "NO -- borrow"))
    lines += ["",
              "  GROSS. No cost, no impact, no borrow. A shape holding two "
              "hundred names",
              "  pays turnover a six-name book does not, and this table does "
              "not price it.",
              "  `transfer` is Grinold's coefficient: the correlation between "
              "the weights",
              "  a shape takes and the score it is taking them from."]
    return "\n".join(lines)
