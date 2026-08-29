"""The check config_version cannot make.

config_version hashes parameters.yaml. Two things change the ranking and
leave it identical: editing the model's code, and a store that grew, because
the model refits from stored history so the store IS the training set.
engine_version does not help -- it is a literal that has read 0.1.0 through
every change this project has made.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prosignal import modelprint as M


def test_the_same_source_and_depth_give_the_same_fingerprint():
    assert M.model_fingerprint(2210) == M.model_fingerprint(2210)


def test_editing_the_model_changes_it(tmp_path: Path):
    (tmp_path / "features").mkdir()
    f = tmp_path / "features" / "linear.py"
    f.write_text("alpha = 20000\n", encoding="utf-8")
    before = M.source_digest(tmp_path, ["features/linear.py"])
    f.write_text("alpha = 10\n", encoding="utf-8")
    assert M.source_digest(tmp_path, ["features/linear.py"]) != before


def test_a_store_that_grew_a_year_is_a_different_model():
    """1,900 sessions is sixteen months of training; 2,210 is nine years.
    Same code, same config, materially different coefficients."""
    assert M.model_fingerprint(1900) != M.model_fingerprint(2210)


def test_one_more_session_is_not_a_different_model():
    """The store gains a session a night. A fingerprint that changed on every
    run would report drift every day and be ignored."""
    assert M.model_fingerprint(2000) == M.model_fingerprint(2001)
    assert M.train_bucket(2000) == M.train_bucket(2249)


def test_unreadable_source_says_unknown_rather_than_guessing():
    """A fingerprint that silently changes meaning is worse than one that
    admits it does not know."""
    assert M.source_digest(Path("/nonexistent"), ["nope.py"]) == "unknown"


def test_an_unknown_depth_is_marked_not_assumed():
    assert M.model_fingerprint(None).endswith("/?")


def test_the_source_list_covers_what_decides_a_ranking():
    root = Path("src/prosignal")
    for rel in M.MODEL_SOURCES:
        assert (root / rel).is_file(), f"{rel} is listed but does not exist"
    joined = " ".join(M.MODEL_SOURCES)
    assert "crossmodel" in joined and "stage4" in joined
    # Deliberately narrow: churning on files that cannot move a coefficient
    # produces a fingerprint people learn to ignore.
    assert "api.py" not in joined and "static" not in joined


def test_the_ledger_records_it():
    src = Path("src/prosignal/ledger.py").read_text(encoding="utf-8")
    assert "model_fingerprint" in src
    contracts = Path("src/prosignal/core/contracts.py").read_text(encoding="utf-8")
    assert "model_fingerprint" in contracts


def test_the_forward_test_reports_drift_the_config_hash_cannot_see(tmp_path):
    """Registers its own test, rather than reading the production one.

    It used to call `progress(cfg.paths.ledger, ...)` against the LIVE
    registration with two hard-coded 2026-08 dates. Observations are counted by
    market date and anything before `started_on` is skipped, so re-registering
    the real forward test moved its start past those dates, the rows were
    filtered out, and a test about fingerprint drift started failing for a
    reason that had nothing to do with fingerprints.
    """
    import datetime as dt

    from prosignal.validation.forward import progress, register

    register(tmp_path, config_version="v1", engine_version="0.1.0",
             git_commit="deadbeef", started_on=dt.date(2026, 1, 1),
             unchecked_reason="fixture: about fingerprint drift, not readiness")
    rows = [{"date": "2026-01-05", "config_version": "v1",
             "model_fingerprint": "aaa/8"},
            {"date": "2026-01-06", "config_version": "v1",
             "model_fingerprint": "bbb/8"}]
    pr = progress(tmp_path, rows, today=dt.date(2026, 1, 7))
    assert any("model fingerprints" in b for b in pr.broken)
    assert sorted(pr.model_fingerprints) == ["aaa/8", "bbb/8"]


def test_one_fingerprint_across_the_window_is_not_drift(tmp_path):
    """The other half of the claim. A check that fires on everything says
    nothing, so the quiet case has to be pinned too."""
    import datetime as dt

    from prosignal.validation.forward import progress, register

    register(tmp_path, config_version="v1", engine_version="0.1.0",
             git_commit="deadbeef", started_on=dt.date(2026, 1, 1),
             unchecked_reason="fixture: about fingerprint drift, not readiness")
    rows = [{"date": "2026-01-05", "config_version": "v1",
             "model_fingerprint": "aaa/8"},
            {"date": "2026-01-06", "config_version": "v1",
             "model_fingerprint": "aaa/8"}]
    pr = progress(tmp_path, rows, today=dt.date(2026, 1, 7))
    assert not any("model fingerprints" in b for b in pr.broken)
    assert pr.broken == []


def test_the_modules_that_decide_the_shortlist_are_fingerprinted():
    """The forward test's secondary criterion is a rank IC "of the daily
    shortlist". Which names are on that shortlist is decided by
    `presentation/selection.py`, and whether a position survives an event is
    decided by `positions.py`. Neither is a stage, so a change to either used
    to alter the object under test while leaving both the config hash and this
    fingerprint identical.
    """
    from prosignal.modelprint import MODEL_SOURCES

    assert "presentation/selection.py" in MODEL_SOURCES
    assert "positions.py" in MODEL_SOURCES


def test_rendering_code_is_still_outside_the_fingerprint():
    """The line has to hold in both directions. A fingerprint that churns when
    a stylesheet changes gets ignored, and an ignored integrity check is worse
    than none."""
    from prosignal.modelprint import MODEL_SOURCES

    assert "presentation/viewmodel.py" not in MODEL_SOURCES
    assert not any(s.startswith("static/") for s in MODEL_SOURCES)
    assert "api.py" not in MODEL_SOURCES


def test_a_change_to_the_selection_rule_moves_the_fingerprint(tmp_path):
    """The property that matters, exercised rather than asserted about a list."""
    from prosignal.modelprint import MODEL_SOURCES, source_digest

    for rel in MODEL_SOURCES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    before = source_digest(tmp_path)

    (tmp_path / "presentation/selection.py").write_text("changed\n")
    assert source_digest(tmp_path) != before
