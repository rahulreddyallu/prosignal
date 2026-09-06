"""Re-run an ablation on the statistic that does not move with the knob.

WHY THIS MODULE EXISTS. Every ablation recorded in `parameters.yaml` was
decided on annual alpha, excess Sharpe or raw excess -- "booking at 3R cost 0.9
points of annual alpha", "booking at 1.5R cost 4.6 points and four points of
worst-year". Those figures are not comparable across the arms that produced
them.

THE MECHANISM. Position size is `risk_budget / risk_per_share`, so anything
that moves the stop distance, the risk budget or the number of slots moves how
much capital is deployed. Writing `r = dep * r_d` for the return on deployed
capital gives `beta = dep * beta_d` and `alpha = dep * alpha_d`: raw alpha is
PROPORTIONAL to deployment, and raw excess additionally carries a cash-drag
term against a fully-invested benchmark. So a stop ablation that widens the
stop shrinks every position, and the arm looks different for a reason that has
nothing to do with whether the stop is a good rule.

Measured on the shipped configuration, moving only `risk_per_trade_pct` from
1% to 4.6% -- same ranking, same names, same costs -- moves raw excess by nine
points and alpha by 2.5. The invariant is `alpha / dep`.

SO THIS HARNESS RANKS ON `alpha_on_deployed_ann` AND REFUSES ANYTHING ELSE.
`rank_arms` raises on a confounded key rather than sorting by it, because the
failure it exists to prevent is exactly a future reader reaching for the
familiar column.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .portfolio_sim import PortfolioParams, phase_summary

#: Statistics that move with a sizing knob. Ranking arms on any of them
#: compares leverage, not the rule under test. `portfolio_sim._benchmark_stats`
#: is the definition these names come from.
CONFOUNDED = (
    "mean_excess", "mean_excess_per_period", "excess_ann", "information_ratio",
    "ir", "excess_hit_rate", "excess_sharpe",
    # Proportional rather than confounded, and still not comparable across arms.
    "alpha_per_period", "alpha_ann", "beta_to_benchmark", "mean_return",
)

#: The one an ablation may be decided on.
DECIDE_ON = "alpha_on_deployed_ann"


@dataclass
class Arm:
    """One configuration, and what it is testing."""

    name: str
    overrides: Dict[str, Any]
    note: str = ""


@dataclass
class ArmResult:
    name: str
    note: str
    metrics: Dict[str, float] = field(default_factory=dict)
    #: True when the arm produced no tradeable book.
    empty: bool = False

    def get(self, key: str, default: float = float("nan")) -> float:
        v = self.metrics.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default


def exit_rung_arms() -> List[Arm]:
    """The three exit rungs, each disarmed in turn. `parameters.yaml` records
    this comparison and decided it on annual alpha."""
    return [
        Arm("shipped", {}, "every rung as configured"),
        Arm("no_target", {"use_target": False},
            "the 3R profit target disarmed"),
        Arm("no_invalidation", {"use_invalidation": False},
            "the structure-MA invalidation disarmed"),
        Arm("no_stop", {"use_stop": False},
            "the ATR stop disarmed -- the disaster floor only"),
    ]


def risk_budget_arms(base: float) -> List[Arm]:
    """The sweep that proved the confound. Included so a reader can SEE the
    headline hold still while the raw figures move."""
    return [Arm(f"risk_{p:g}pct", {"risk_per_trade_pct": p},
                "sizing knob; the ranking and the names are identical")
            for p in (base * 0.5, base, base * 2.0, base * 4.0)]


def band_arms(base: PortfolioParams) -> List[Arm]:
    """Exit-band width. Wider keeps names longer and pays less entry cost."""
    return [Arm(f"band_{e}", {"exit_rank": e},
                f"hold while inside the best {e} names")
            for e in (base.entry_rank * 2, base.exit_rank, base.exit_rank * 2)]


def run(arms: Sequence[Arm], rankings, prices,
        base: PortfolioParams, *, step_sessions: int = 21,
        decision_sessions: Optional[int] = None,
        dates_allowed=None) -> List[ArmResult]:
    """Every arm, through the shipped simulator, at one cadence.

    The cadence is held fixed across arms on purpose: `decision_sessions`
    changes how often the book turns over and would otherwise be a second
    variable inside a one-variable comparison. See `Q6` in the findings
    register.
    """
    out: List[ArmResult] = []
    for a in arms:
        p = dataclasses.replace(base, **a.overrides)
        m = phase_summary(rankings, prices, p, step_sessions=step_sessions,
                          decision_sessions=decision_sessions,
                          dates_allowed=dates_allowed)
        out.append(ArmResult(a.name, a.note, dict(m or {}), empty=not m))
    return out


def rank_arms(results: Sequence[ArmResult], key: str = DECIDE_ON
              ) -> List[ArmResult]:
    """Best first, and only on a leverage-invariant statistic.

    Raises on a confounded key rather than sorting by it. A harness that
    quietly obeys is a harness that lets the next reader re-make the mistake
    this whole module documents.
    """
    if key in CONFOUNDED:
        raise ValueError(
            f"{key!r} moves with the sizing knob, so ranking ablation arms on "
            f"it compares leverage rather than the rule under test. Position "
            f"size is risk_budget / risk_per_share, so any arm that changes "
            f"the stop, the risk budget or the slot count changes how much "
            f"capital is deployed -- measured, nine points of raw excess "
            f"across a risk-budget sweep in which nothing else moved. Rank on "
            f"{DECIDE_ON!r}."
        )
    usable = [r for r in results if not r.empty and np.isfinite(r.get(key))]
    return sorted(usable, key=lambda r: r.get(key), reverse=True)


def table(results: Sequence[ArmResult]) -> str:
    """The comparison, with the confounded columns present and labelled.

    They are printed rather than dropped: every earlier write-up quotes them,
    and a reconciliation needs them. What they may not do is decide.
    """
    head = (f"{'arm':<18} {'alpha/dep':>10} {'excess/dep':>11} {'deployed':>9} "
            f"{'sharpe':>7} {'maxDD':>8} | {'raw excess':>11} {'IR':>7}")
    lines = [head, "-" * len(head)]
    for r in rank_arms(results):
        lines.append(
            f"{r.name:<18} {r.get('alpha_on_deployed_ann'):>+10.2%} "
            f"{r.get('excess_on_deployed_ann'):>+11.2%} "
            f"{r.get('deployed_frac'):>9.1%} {r.get('sharpe'):>+7.2f} "
            f"{r.get('worst_schedule_drawdown'):>+8.1%} | "
            f"{r.get('mean_excess') * r.get('periods_per_year'):>+11.2%} "
            f"{r.get('information_ratio'):>+7.2f}")
    empty = [r.name for r in results if r.empty]
    if empty:
        lines.append(f"  no tradeable book: {', '.join(empty)}")
    lines += ["",
              f"  Ranked on {DECIDE_ON}. The two columns right of the bar move "
              f"with the sizing knob",
              "  and are reported for reconciliation only -- see "
              "validation/ablation.py."]
    return "\n".join(lines)
