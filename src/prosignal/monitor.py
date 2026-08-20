"""Performance monitor: a halt triggered by results rather than by data.

The existing halts all fire on inputs -- MarketWideHalt when the feed breaks,
NOT_TESTABLE when a check has nothing to check, ModelUnavailable when the fit
fails. All of them assume that if the data is sound the signals are worth
acting on. Nothing watches whether the signals are actually behaving like the
walk-forward said they would.

That is the failure this covers: every input healthy, every stage green, and a
realised hit rate that has not resembled the backtest for two months. The
system has no way to notice, because noticing requires comparing outcomes
against an expectation rather than inputs against a schema.

The test is deliberately weak. Realised performance over a few dozen trades is
noisy enough that a sensitive threshold would halt on ordinary variance, and a
halt that fires on noise gets ignored, which is worse than no halt. So this asks
only whether the realised mean is implausible under the walk-forward
expectation -- a one-sided t-test at a stated level, on a stated minimum sample.
Passing it is not evidence the edge is intact; failing it is evidence something
has changed.

It halts new signals. It never closes a position and never places an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["PerformanceVerdict", "review_performance", "MIN_TRADES", "ALERT_P"]

#: Below this many resolved outcomes the comparison is not worth making.
MIN_TRADES = 30

#: One-sided significance at which realised performance is called inconsistent
#: with the backtest. Deliberately strict: a halt that fires on noise is a halt
#: that gets switched off.
ALERT_P = 0.01


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


@dataclass
class PerformanceVerdict:
    """Whether realised performance is consistent with the expectation."""

    halt: bool
    reason: str
    n: int = 0
    realised_mean: Optional[float] = None
    expected_mean: Optional[float] = None
    t_stat: Optional[float] = None
    p_value: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "halt": self.halt, "reason": self.reason, "n": self.n,
            "realised_mean": self.realised_mean, "expected_mean": self.expected_mean,
            "t_stat": self.t_stat, "p_value": self.p_value,
        }


def review_performance(
    realised: Sequence[float],
    expected_mean: float,
    min_trades: int = MIN_TRADES,
    alert_p: float = ALERT_P,
) -> PerformanceVerdict:
    """Compare realised per-trade returns against the walk-forward expectation.

    One-sided: only underperformance halts. Beating the backtest is also a
    reason to look, but it is not a reason to stop issuing signals, and
    conflating the two would halt on a good month.
    """
    values = np.asarray([v for v in realised if v is not None and np.isfinite(v)],
                        dtype="float64")
    n = int(values.size)
    if n < min_trades:
        return PerformanceVerdict(
            halt=False,
            reason=f"{n} resolved outcomes; {min_trades} needed before the comparison means anything",
            n=n,
        )

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    # Relative, not sd <= 0. A constant series accumulates a standard deviation
    # around 1e-18 from floating point alone, which is not zero, and dividing by
    # it produces a t-statistic of 1e15 and a halt on nothing at all. A halt that
    # fires on rounding is a halt that gets switched off.
    scale = max(abs(mean), abs(float(expected_mean)), 1e-9)
    if not np.isfinite(sd) or sd <= scale * 1e-6:
        return PerformanceVerdict(
            halt=False,
            reason=(
                "realised outcomes have no dispersion to test against; a "
                "constant series cannot be inconsistent with anything"
            ),
            n=n, realised_mean=mean, expected_mean=float(expected_mean),
        )

    t = (mean - float(expected_mean)) / (sd / np.sqrt(n))
    p = _norm_cdf(t)                      # one-sided: probability of being this far BELOW

    if p < alert_p:
        return PerformanceVerdict(
            halt=True,
            reason=(
                f"realised mean {mean:+.4f} against an expected {float(expected_mean):+.4f} "
                f"over {n} trades (t {t:.2f}, p {p:.4f}). This is not consistent with "
                f"the walk-forward, so new signals are held until someone looks."
            ),
            n=n, realised_mean=mean, expected_mean=float(expected_mean),
            t_stat=float(t), p_value=float(p),
        )

    return PerformanceVerdict(
        halt=False,
        reason=(
            f"realised mean {mean:+.4f} against an expected {float(expected_mean):+.4f} "
            f"over {n} trades (t {t:.2f}, p {p:.4f}); within what the backtest allows"
        ),
        n=n, realised_mean=mean, expected_mean=float(expected_mean),
        t_stat=float(t), p_value=float(p),
    )
