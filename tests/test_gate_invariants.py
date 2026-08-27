"""Gates that overlap, and the invariants that keep the overlap honest.

Several gates in this engine are subsumed by others. That is not automatically
wrong -- defence in depth is a choice -- but a SUBSUMED gate is dead code that
reads as protection, and it becomes live protection again the moment the gate
above it is narrowed. These tests pin the subsumption relationships so that
either fact stays true on purpose rather than by accident.
"""
from __future__ import annotations

import pytest

from prosignal.config.loader import load_config


@pytest.fixture(scope="module")
def p():
    return load_config().params


def test_stage_5_upcoming_earnings_is_subsumed_by_stage_3(p):
    """Stage 3 rejects on ANY results date inside its holding window; Stage 5
    hard-rejects on a CONFIRMED date inside a much shorter one. While the
    Stage 3 window is the wider of the two, Stage 5's upcoming-earnings branch
    is unreachable and is defence in depth, not attrition.

    If Stage 3's window is ever narrowed below Stage 5's, this fails -- and it
    should, because the funnel then attributes the same rejection to two rungs.
    """
    s3 = int(p.stage3_eligibility.earnings_proximity.holding_window_sessions.value)
    s5 = int(p.stage5_false_signal.earnings_distortion.upcoming_earnings_sessions.value)
    assert s3 >= s5, (
        f"Stage 3 rejects at {s3} sessions and Stage 5 at {s5}. With Stage 5 "
        f"the wider of the two, both gates cut the same names and the funnel "
        f"double-counts the reason."
    )


def test_one_list_of_buckets_that_block_new_entries(p):
    """Stage 2 refuses new entries in a bucket; Stage 5 halts the market in
    one. They must not name that bucket independently -- editing the config
    then moves only one of them."""
    import inspect

    from prosignal.stages import stage5_false_signal as s5

    src = inspect.getsource(s5.run)
    assert "no_new_entry_buckets" in src, (
        "Stage 5's crash halt must read Stage 2's list rather than repeating it"
    )


def test_the_score_gate_is_subsumed_by_the_percentile_gate(p):
    """`percentile` is `rank_to_unit_interval(score) * 100`, so
    min_composite_score 0.60 is arithmetically percentile >= 60 and is already
    implied by min_universe_percentile 90. Only one of them selects; tuning the
    other has no effect on which names pass."""
    sc = p.stage8_final_signal.scarcity
    assert float(sc.min_composite_score.value) * 100.0 <= float(
        sc.min_universe_percentile.value), (
        "min_composite_score is now the binding gate, which inverts what the "
        "stage documents: the percentile test is the one that selects"
    )


def test_the_entry_band_is_tighter_than_the_scarcity_gates(p):
    """Stage 6 admits at rank <= entry_rank, which for a universe of a few
    hundred names is a far higher bar than percentile >= 90. The scarcity gates
    therefore shape the WATCHLIST, not the book -- worth knowing before either
    is tuned in the belief that it controls what gets bought."""
    entry = int(p.stage6_entry.admission.entry_rank.value)
    cap = int(p.stage8_final_signal.portfolio.max_signals_per_run.value)
    assert entry <= cap, (
        "an entry rank above the per-run cap is inert: the book fills before "
        "the band does"
    )
