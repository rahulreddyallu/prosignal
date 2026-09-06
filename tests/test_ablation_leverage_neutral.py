"""An ablation decided on a confounded statistic decided nothing.

Every ablation recorded in `parameters.yaml` was settled on annual alpha,
excess Sharpe or raw excess -- "booking at 3R cost 0.9 points of annual alpha",
"booking at 1.5R cost 4.6 points and four points of worst-year". None of those
is comparable across the arms that produced them.

Position size is `risk_budget / risk_per_share`, so anything that moves the
stop distance, the risk budget or the slot count moves how much capital is
deployed. Write `r = dep * r_d` and `alpha = dep * alpha_d`: raw alpha is
PROPORTIONAL to deployment and raw excess additionally carries a cash-drag term
against a fully-invested benchmark. An arm that widens the stop shrinks every
position, and it then looks different for a reason that has nothing to do with
whether the stop is a good rule.

Measured on the shipped configuration, moving only `risk_per_trade_pct` from 1%
to 4.6% -- same ranking, same names, same costs -- moves raw excess nine points.

So `rank_arms` raises on a confounded key rather than sorting by it. The
failure this guards against is a future reader reaching for the familiar
column, and a harness that quietly obeys is one that lets them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_leverage_neutral import _params, _prices, _rankings   # noqa: E402

from prosignal.validation import ablation as A


def _run(arms):
    prices = _prices()
    return A.run(arms, _rankings(prices), prices, _params(), step_sessions=21)


def test_the_decision_statistic_is_the_leverage_invariant_one():
    assert A.DECIDE_ON == "alpha_on_deployed_ann"
    assert A.DECIDE_ON not in A.CONFOUNDED


def test_ranking_on_raw_excess_is_refused():
    """The finding, as one assertion."""
    res = _run(A.exit_rung_arms())
    for key in ("mean_excess", "information_ratio", "excess_sharpe",
                "alpha_ann"):
        with pytest.raises(ValueError, match="sizing knob"):
            A.rank_arms(res, key=key)


def test_the_refusal_says_what_to_use_instead():
    with pytest.raises(ValueError) as e:
        A.rank_arms(_run(A.exit_rung_arms()), key="mean_excess")
    assert A.DECIDE_ON in str(e.value)
    assert "risk_budget / risk_per_share" in str(e.value)


def test_raw_alpha_is_refused_too():
    """It is proportional rather than confounded, and that is still not
    comparable across arms -- alpha = dep * alpha_d."""
    assert "alpha_ann" in A.CONFOUNDED
    assert "alpha_per_period" in A.CONFOUNDED


def test_a_pure_sizing_sweep_moves_the_raw_figures_and_not_the_headline():
    """The confound, demonstrated inside the harness that guards against it.

    Every arm here has an identical ranking, identical names and identical
    costs. Only the risk budget moves. A statistic that separates them is
    measuring the knob.
    """
    res = _run(A.risk_budget_arms(1.0))
    dep = [r.get("deployed_frac") for r in res]
    assert max(dep) > min(dep) * 1.5, "the sweep did not move deployment"

    raw = [r.get("mean_excess") * r.get("periods_per_year") for r in res]
    head = [r.get(A.DECIDE_ON) for r in res]
    assert (max(raw) - min(raw)) > (max(head) - min(head)), (
        f"raw excess spread {max(raw) - min(raw):.4f} must exceed the headline "
        f"spread {max(head) - min(head):.4f}; if it does not, the fixture is "
        f"not exercising the confound and the guard proves nothing"
    )


def test_every_arm_is_reported_even_when_it_cannot_be_ranked():
    """An arm that produced no tradeable book is named, not dropped. A missing
    row reads as an arm that was never tried."""
    res = _run(A.exit_rung_arms())
    assert len(res) == 4
    assert {r.name for r in res} == {"shipped", "no_target", "no_invalidation",
                                     "no_stop"}


def test_the_table_prints_the_confounded_columns_and_labels_them():
    """They are kept because every earlier write-up quotes them and a
    reconciliation needs them. What they may not do is decide."""
    text = A.table(_run(A.exit_rung_arms()))
    assert "raw excess" in text
    assert "IR" in text
    assert f"Ranked on {A.DECIDE_ON}" in text
    assert "move with the sizing knob" in text


def test_the_cadence_is_held_fixed_across_arms():
    """`decision_sessions` changes how often the book turns over, so letting it
    vary inside a one-variable comparison makes it a two-variable one."""
    prices = _prices()
    rk = _rankings(prices)
    slow = A.run(A.exit_rung_arms(), rk, prices, _params(),
                 step_sessions=21, decision_sessions=63)
    fast = A.run(A.exit_rung_arms(), rk, prices, _params(),
                 step_sessions=21, decision_sessions=21)
    assert all(r.get("decision_sessions") == 63 for r in slow if not r.empty)
    assert all(r.get("decision_sessions") == 21 for r in fast if not r.empty)
