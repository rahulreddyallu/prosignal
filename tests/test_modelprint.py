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


def test_the_forward_test_reports_drift_the_config_hash_cannot_see():
    from prosignal.validation.forward import progress
    from prosignal.config.loader import load_config
    cfg = load_config()
    rows = [{"date": "2026-08-24", "config_version": "v1",
             "model_fingerprint": "aaa/8"},
            {"date": "2026-08-25", "config_version": "v1",
             "model_fingerprint": "bbb/8"}]
    pr = progress(cfg.paths.ledger, rows)
    assert any("model fingerprints" in b for b in pr.broken)
    assert sorted(pr.model_fingerprints) == ["aaa/8", "bbb/8"]
