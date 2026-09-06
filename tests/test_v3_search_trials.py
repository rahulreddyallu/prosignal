"""The search that chose the model must be charged for.

The trial registry held 99 configurations and every one came from a research
command written AFTER the v3 composite shipped. Not one came from the search
that chose the 22 factors, the five themes, the combination method, the weight
caps, the quality floor or the book. `cumulative_trials_logged` -- the escape
hatch for "everything before the registry existed" -- stood at 20 against a
search the record puts at 503.

It could not be reconstructed because `research/V3_SEARCH.md` was deleted in
commit f1b2a9a along with the search code. Deleting code that no longer chooses
anything is right. Deleting the record of the search removed the only evidence
of how much data-dredging the shipped model rests on, and the Deflated Sharpe
went on charging for the tuning done afterwards.

These tests hold the reconstruction to its source: every group cites a section
that exists in the restored document, the arm counts match the numbers the
document states, and a group whose grid size was never written down is marked
so the total prints as a floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosignal.validation import v3_search as V
from prosignal.validation.registry import PRE_V10_PASS, TrialRegistry

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / V.SOURCE_DOC


def test_the_search_record_is_in_the_repository():
    """It was deleted once. If it goes again the count below is unverifiable."""
    assert DOC.is_file(), (
        f"{V.SOURCE_DOC} is missing. It is the only record of the v3 factor "
        f"search and the sole evidence behind the DSR's trial count; recover "
        f"it with `git show {V.SOURCE_COMMIT}:{V.SOURCE_DOC}`."
    )


def test_every_group_cites_a_section_that_exists():
    doc = DOC.read_text(encoding="utf-8")
    for g in V.groups():
        assert f"## {g.section}." in doc, (
            f"group {g.label!r} cites section {g.section}, which is not in "
            f"{V.SOURCE_DOC}"
        )


def test_the_factor_counts_are_the_documents_own():
    """93 built across eight themes, per section 4's table."""
    doc = DOC.read_text(encoding="utf-8")
    assert sum(V.FACTORS_BUILT.values()) == 93
    assert "93 built, 33 cleared, 22 shipped" in doc
    for theme, n in V.FACTORS_BUILT.items():
        assert f"{theme} ({n})" in doc, (
            f"section 4 does not say {theme} had {n} factors built"
        )


def test_each_factor_was_a_look_at_every_horizon_it_was_screened_at():
    """A factor that cleared at h=63 and not at h=21 was four decisions, not
    one -- the record reports the verdict per horizon."""
    screens = sum(g.n for g in V.groups() if "factor screen" in g.label
                  and "dividend" not in g.label)
    assert screens == 93 * len(V.SCREEN_HORIZONS) == 372


def test_the_total_is_reported_as_a_floor_when_a_grid_size_is_unknown():
    """Two level-2 grids are described without their arm counts. Inventing
    arms is still inventing, so they enter at the documented minimum and the
    total says AT LEAST."""
    assert V.is_floor() is True
    unknown = [g.label for g in V.groups() if not g.exact]
    assert len(unknown) == 2
    assert "AT LEAST" in V.summary()
    assert "TOTAL" not in V.summary()


def test_the_total_is_the_sum_of_the_groups():
    assert V.total() == sum(g.n for g in V.groups()) == 503


def test_one_label_per_configuration_and_all_distinct():
    """The registry is content-addressed by (command, label). A single row
    reading "372 factor screens" would be idempotent with a later row saying
    the same thing about different work."""
    labels = V.labels()
    assert len(labels) == V.total()
    assert len(set(labels)) == len(labels)


def test_the_nulls_and_the_correlations_are_not_counted_as_trials():
    """A permuted-label null is a critical value, not a candidate; the
    redundancy pass's correlations are label-free. Counting either would
    inflate the charge, which is as dishonest as understating it."""
    text = V.summary()
    assert "permut" not in text.lower()
    assert "placebo" not in text.lower()
    redundancy = [g for g in V.groups() if g.section == "5"]
    assert len(redundancy) == 1 and redundancy[0].n == 1, (
        "the |rho| >= 0.80 threshold is the one choice in the redundancy pass"
    )


def test_recording_is_idempotent(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    first = reg.record(V.COMMAND, V.labels())
    second = reg.record(V.COMMAND, V.labels())
    assert first == V.total()
    assert second == 0, "re-running the reconstruction must not inflate the count"
    assert reg.count() == V.total()
    assert reg.by_command()[V.COMMAND] == V.total()


def test_the_search_is_charged_to_no_v10_pass(tmp_path):
    """It predates the budget by two generations. Back-dating it into a pass
    would report v10 spending that never happened."""
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record(V.COMMAND, V.labels())
    assert set(reg.by_pass()) == {PRE_V10_PASS}


def test_the_shipped_registry_actually_carries_the_search():
    """The one that matters. Without these rows the DSR charges the headline
    result for the configurations tried AFTER the model existed and nothing
    for the search that produced it."""
    path = ROOT / "data/curated/trial_registry.jsonl"
    if not path.is_file():
        pytest.skip("no curated registry in this checkout")
    reg = TrialRegistry(path)
    assert reg.by_command().get(V.COMMAND, 0) == V.total(), (
        "the v3 factor search is not on the shipped registry; run "
        "`prosignal research trials --register-v3-search`"
    )
    assert reg.effective_trials(20) >= 600
