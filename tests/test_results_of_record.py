"""The generator's own arithmetic, tested without building a panel.

Building the v3 panel over the whole store takes about twenty-five minutes, so
these hold the parts that can be wrong quietly: the power statement's
denominator, the reproduction verdict, and the rule that gross, cost and net are
three numbers rather than one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prosignal.validation import results as R


# ------------------------------------------------------------ power statement
def test_the_power_statement_uses_calendar_years_not_pooled_periods():
    """The denominator bug, pinned.

    `phase_summary` pools every phase offset, so `n_periods / periods_per_year`
    counts one calendar period once per offset. On the real panel it read
    371/4 = 92.8 "years" against a span of 7.7, and every power statement built
    from it was nonsense. `ArmResult.years` is the calendar span and the
    renderer takes it from there.
    """
    text = R._power_statement(0.78, 7.7)
    assert "sqrt(7.7)" in text
    # 0.78 * sqrt(7.7) = 2.16
    assert "+2.16" in text


def test_the_power_statement_says_how_long_t_equals_two_takes():
    """At IR 0.78 it is 6.5 years, which is the whole argument about horizon."""
    text = R._power_statement(0.78, 1.5)
    assert "6.6 years" in text or "6.5 years" in text, text
    assert "+0.96" in text, "expected t after 18 months at IR 0.78 is 0.96"


def test_a_non_positive_ir_says_t_is_unreachable_rather_than_printing_a_number():
    text = R._power_statement(-0.84, 7.7)
    assert "unreachable" in text
    assert "-2.33" in text, "the expected t is still reported, negative"


def test_a_missing_ir_is_not_computable_rather_than_zero():
    assert "NOT COMPUTABLE" in R._power_statement(float("nan"), 7.7)
    assert "NOT COMPUTABLE" in R._power_statement(0.5, 0.0)


# -------------------------------------------------------- reproduction verdict
def _measured(**over):
    base = {"ir": -0.84, "mean_excess_per_period": -0.0446,
            "periods_beating_benchmark": 0.326, "alpha_per_period": 0.0009}
    base.update(over)
    return base


CLAIMED = {"mean_excess_per_period": -0.0423, "ir": -0.83,
           "alpha_per_period": -0.0067, "periods_beating_benchmark": 0.329}


def test_a_claim_that_lands_inside_tolerance_reproduces():
    status, reason, rows = R._judge_shipped(CLAIMED, _measured())
    assert status == "REPRODUCED", reason
    assert rows, "the per-figure comparison must be returned either way"


def test_a_non_headline_divergence_is_reported_but_does_not_withdraw():
    """Alpha flips sign here on a near-zero residual and is NOT a headline.

    Letting it withdraw a table whose headline reproduces to two decimals would
    be as misleading as hiding it, so it is reported in the reason and in the
    comparison table instead.
    """
    status, reason, rows = R._judge_shipped(CLAIMED, _measured())
    assert status == "REPRODUCED"
    assert "alpha" in reason.lower(), (
        "the alpha divergence was dropped from the report. A figure left out "
        "of the comparison because it was not part of the verdict is a figure "
        "hidden.")
    alpha = next(r for r in rows if r["key"] == "alpha_per_period")
    assert alpha["verdict"] == "OPPOSITE SIGN"
    assert alpha["headline"] is False


def test_a_headline_sign_flip_withdraws_the_claim():
    status, reason, _ = R._judge_shipped(CLAIMED, _measured(ir=+0.84))
    assert status == "WITHDRAWN"
    assert "OPPOSITE SIGN" in reason


def test_a_headline_magnitude_gap_withdraws_the_claim():
    status, reason, _ = R._judge_shipped(
        CLAIMED, _measured(mean_excess_per_period=-0.30))
    assert status == "WITHDRAWN"
    assert "mean excess" in reason


def test_an_unrunnable_arm_is_not_testable_and_never_a_failure():
    """NOT_TESTABLE is a distinct outcome and is never upgraded."""
    status, reason, rows = R._judge_shipped(CLAIMED, {})
    assert status == "NOT_TESTABLE"
    assert rows == []
    assert "no benchmarked result" in reason


def test_the_tuning_claim_is_judged_on_alpha_sharpe_and_book_return():
    measured = {"alpha_per_period": 0.0044, "periods_per_year": 4.0,
                "sharpe": 0.93, "book_return_ann": 0.052,
                "excess_sharpe": float("nan")}
    claimed = {"book_return_ann": 0.426, "alpha_ann": 0.203, "sharpe": 1.59,
               "excess_sharpe": 1.12}
    status, reason, rows = R._judge_tuning(claimed, measured, years=7.7)
    assert status == "WITHDRAWN"
    for bit in ("alpha", "Sharpe", "book return"):
        assert bit in reason
    assert {r["key"] for r in rows} >= {"alpha_ann", "sharpe",
                                        "book_return_ann"}


# ------------------------------------------------------ independent observations
def test_independent_observations_counts_windows_not_rows():
    """380 dates 5 sessions apart against a 63-session label is ~31 windows."""
    n = R._independent(n_dates=380, stride=5, horizon=63)
    assert 30 <= n <= 32, n
    # And it is nothing like the row count.
    assert n < 100


def test_a_shorter_label_yields_more_independent_windows():
    assert (R._independent(380, 5, 21)
            > R._independent(380, 5, 42)
            > R._independent(380, 5, 63))


def test_no_dates_is_zero_rather_than_a_division_error():
    assert R._independent(0, 5, 63) == 0.0


# --------------------------------------------------------------- the rendering
def _arm(**over):
    m = {"mean_return_per_period": 0.0083, "bench_return_per_period": 0.0529,
         "sharpe": 0.61, "bench_sharpe": 0.88,
         "mean_excess_per_period": -0.0446, "excess_ann": -0.178,
         "gross_excess_ann": -0.174, "cost_drag_ann": 0.005, "ir": -0.84,
         "alpha_per_period": 0.0009, "beta_to_benchmark": 0.14,
         "periods_beating_benchmark": 0.326,
         "worst_schedule_drawdown": -0.135, "avg_names": 4.8,
         "n_periods": 371, "periods_per_year": 4.0,
         "book_return_ann": 0.033, "bench_return_ann": 0.212}
    m.update(over)
    a = R.ArmResult(key="k", title="T", configuration="C", claimed_in="X",
                    claimed=CLAIMED, measured=m, status="REPRODUCED",
                    reason="r", is_shipped=True, years=7.68)
    a.status, a.reason, a.comparison = R._judge_shipped(CLAIMED, m)
    return a


def test_gross_cost_and_net_are_all_three_rendered():
    body = "\n".join(R._arm_block(_arm()))
    assert "gross excess over the universe" in body
    assert "cost drag" in body
    assert "**net excess**" in body


def test_the_rendered_power_line_uses_the_arms_calendar_years():
    body = "\n".join(R._arm_block(_arm()))
    assert "sqrt(7.7)" in body, (
        "the power line is not using ArmResult.years. If it reverts to "
        "n_periods/periods_per_year it will read sqrt(92.8) on a 7.7-year "
        "panel again.")
    assert "sqrt(92.8)" not in body


def test_the_status_badge_is_not_double_emboldened():
    body = "\n".join(R._arm_block(_arm()))
    assert "****" not in body, (
        "the status badge is wrapped in bold twice and renders as literal "
        "asterisks")


def test_every_published_figure_appears_in_the_comparison_table():
    body = "\n".join(R._arm_block(_arm()))
    assert "Claimed against measured" in body
    for label in ("information ratio", "mean excess / period",
                  "periods beating the benchmark", "alpha / period"):
        assert label in body, label


# ------------------------------------------------------------- the trial claim
def test_pass_zero_spent_no_trials():
    """`results.py` claims P0 stays at zero. This is that claim, checked.

    Re-running a configuration that has already been looked at, to find out
    whether its published number reproduces, is not a new look at the data:
    neither arm can be SELECTED by `research results` -- the shipped ranker is
    fixed by config and the command cannot change it -- and both are already
    inside the counts the Deflated Sharpe charges.

    If a later pass starts charging trials to P0, that is either a mistake or a
    real change in what this command does, and either way it should be argued
    for rather than absorbed.
    """
    from prosignal.config.loader import load_config
    from prosignal.validation.registry import TrialRegistry, registry_path

    cfg = load_config()
    reg = TrialRegistry(registry_path(cfg.paths.curated))
    assert reg.spent("P0") == 0, (
        f"pass P0 has spent {reg.spent('P0')} trial(s) against an allocation "
        f"of 0. Reconciliation looks at no out-of-sample score; a trial "
        f"charged here means modelling leaked into a pass designed to have "
        f"none.")
