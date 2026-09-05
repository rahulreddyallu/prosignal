"""The v10 trial budget is enforced, not merely counted.

The registry has always counted trials honestly. Counting is not the same as
spending deliberately: the Deflated Sharpe charges every configuration that was
looked at, and it already reads 0.030 against 4,877 trials, so a campaign that
discovers halfway through that it has spent thirty has already made whatever
ships less credible and nothing can give them back.

These tests hold the three properties that make the budget real: it refuses,
it refuses ATOMICALLY, and a re-run of an already-paid campaign is free.
"""

from __future__ import annotations

import json

import pytest

from prosignal.validation.registry import (
    PRE_V10_PASS, V10_BUDGET, V10_TOTAL_BUDGET, BudgetExceeded, TrialRegistry,
)


@pytest.fixture
def reg(tmp_path):
    return TrialRegistry(tmp_path / "trial_registry.jsonl")


# ------------------------------------------------------------- the allocation
def test_the_allocation_sums_to_the_declared_total():
    assert sum(V10_BUDGET.values()) == V10_TOTAL_BUDGET, (
        f"the per-pass allocation sums to {sum(V10_BUDGET.values())} but the "
        f"declared total is {V10_TOTAL_BUDGET}. A budget whose parts do not "
        f"add up to its whole is not a budget.")


def test_the_allocation_is_the_build_plans():
    assert V10_BUDGET == {"P0": 0, "P1": 0, "P2": 2, "P3": 12, "P4": 4,
                          "P5": 8, "P6": 4, "P7": 6, "P8": 4}


def test_reconciliation_and_data_passes_are_budgeted_at_zero():
    """P0 and P1 look at no out-of-sample score, so they cost nothing.

    A trial charged to either is a sign that modelling has leaked into a pass
    designed to have none, which is worth failing over.
    """
    assert V10_BUDGET["P0"] == 0
    assert V10_BUDGET["P1"] == 0


# ------------------------------------------------------------- the enforcement
def test_a_campaign_within_its_allocation_is_recorded(reg):
    assert reg.record("exp", ["a", "b"], pass_id="P2") == 2
    assert reg.spent("P2") == 2
    assert reg.remaining("P2") == 0


def test_a_campaign_that_would_exceed_its_allocation_is_refused(reg):
    reg.record("exp", ["a", "b"], pass_id="P2")
    with pytest.raises(BudgetExceeded) as exc:
        reg.record("exp", ["c"], pass_id="P2")
    e = exc.value
    assert e.pass_id == "P2" and e.allocation == 2
    assert e.spent == 2 and e.requested == 1 and e.remaining == 0
    # The arithmetic has to be ON the exception, so the caller reports it
    # rather than re-deriving it.
    assert "P2" in str(e) and "allocated 2" in str(e)


def test_a_refused_campaign_records_nothing_at_all(reg):
    """ATOMIC. Not even the prefix that would have fit.

    Recording part of a comparison charges the Deflated Sharpe for arms the
    researcher never got to compare, which is the worst of both worlds.
    """
    reg.record("exp", ["a"], pass_id="P2")
    before = reg.count()
    with pytest.raises(BudgetExceeded):
        reg.record("exp", ["b", "c", "d"], pass_id="P2")   # 1 would have fit
    assert reg.count() == before, (
        "a refused campaign wrote rows. The refusal has to be all-or-nothing.")
    assert reg.spent("P2") == 1


def test_rerunning_a_paid_campaign_is_free(reg):
    """Idempotent by (command, label): a re-run is not a new look at the data."""
    assert reg.record("exp", ["a", "b"], pass_id="P2") == 2
    assert reg.record("exp", ["a", "b"], pass_id="P2") == 0
    assert reg.spent("P2") == 2


def test_only_genuinely_new_configurations_are_charged(reg):
    """A campaign that mostly repeats itself pays only for what is new."""
    reg.record("exp", ["a"], pass_id="P2")
    assert reg.record("exp", ["a", "b"], pass_id="P2") == 1
    assert reg.spent("P2") == 2


def test_an_unknown_pass_is_refused_rather_than_given_a_default(reg):
    """Neither an implicit zero nor an implicit infinity.

    A typo'd pass silently getting no budget would refuse legitimate work; one
    silently getting unlimited budget would defeat the mechanism entirely, and
    that is the direction mistakes travel.
    """
    with pytest.raises(KeyError) as exc:
        reg.record("exp", ["a"], pass_id="P42")
    assert "P42" in str(exc.value)
    assert reg.count() == 0


def test_the_default_pass_is_never_refused(reg):
    """Existing callers keep working; their trials are simply not v10 spending."""
    for i in range(50):
        reg.record("legacy", [f"arm{i}"])
    assert reg.count() == 50
    assert reg.by_pass()[PRE_V10_PASS] == 50


def test_pre_v10_trials_are_still_charged_by_the_dsr(reg):
    """Not v10 spending is not the same as not counted."""
    reg.record("legacy", ["a", "b", "c"])
    assert reg.effective_trials(carried=20) == 23


def test_passes_do_not_share_a_budget(reg):
    reg.record("exp", ["a", "b"], pass_id="P2")
    assert reg.record("other", ["c", "d", "e"], pass_id="P3") == 3
    assert reg.spent("P2") == 2 and reg.spent("P3") == 3
    with pytest.raises(BudgetExceeded):
        reg.record("exp", ["f"], pass_id="P2")


# --------------------------------------------------------------- bookkeeping
def test_the_pass_survives_a_round_trip_through_the_file(reg):
    reg.record("exp", ["a"], pass_id="P3")
    fresh = TrialRegistry(reg.path)
    assert fresh.by_pass() == {"P3": 1}


def test_a_score_backfill_does_not_recharge_a_different_pass(reg):
    """A later run that happens to know a score does not re-spend the trial."""
    reg.record("exp", ["a"], pass_id="P3")
    reg.record("exp", ["a"], scores=[0.42], pass_id="P5")
    assert reg.spent("P3") == 1
    assert reg.spent("P5") == 0, (
        "backfilling a score re-charged the trial to the backfilling pass. A "
        "trial is spent once, by whoever spent it.")
    assert reg.recorded_scores() == [0.42]


def test_the_budget_report_covers_every_pass_and_the_legacy_bucket(reg):
    reg.record("exp", ["a"], pass_id="P3")
    reg.record("legacy", ["b"])
    rows = {r["pass"]: r for r in reg.budget_report()}
    assert set(V10_BUDGET) <= set(rows)
    assert rows["P3"] == {"pass": "P3", "allocated": 12, "spent": 1,
                          "remaining": 11, "over": 0}
    assert rows[PRE_V10_PASS]["allocated"] is None
    assert rows[PRE_V10_PASS]["spent"] == 1


def test_the_pass_id_is_persisted_only_when_it_is_a_v10_pass(reg):
    """The file stays readable and the legacy default costs no bytes."""
    reg.record("exp", ["a"], pass_id="P3")
    reg.record("legacy", ["b"])
    rows = [json.loads(l) for l in reg.path.read_text().splitlines() if l.strip()]
    by_label = {r["label"]: r for r in rows}
    assert by_label["a"]["pass_id"] == "P3"
    assert "pass_id" not in by_label["b"]
