"""The research ledger is evidence. The test suite must not write to it.

`run_analysis` appends a permanent row on every call, and the `live_cfg`
fixture handed tests the real project config -- ledger path included. So every
end-to-end pipeline test deposited a run into the production research record.

That record is not a log:

  * `Ledger.trial_count()` is the multiple-testing input to the Deflated Sharpe
    Ratio. Test runs inflated the penalty applied to every real result.
  * `validation.forward.progress()` counts ledger rows by market date, so test
    runs were counted as forward-test observations against a pre-registered
    window.
  * The next real run reads the newest row back as its open book, so a test's
    book could seed production hysteresis.

Measured on this repository: 41 rows written into the research ledger by test
runs in a single afternoon.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_live_cfg_writes_nowhere_near_the_real_ledger(live_cfg):
    real = (PROJECT_ROOT / "data" / "ledger").resolve()
    assert live_cfg.paths.ledger.resolve() != real
    assert real not in live_cfg.paths.ledger.resolve().parents


def test_live_cfg_still_reads_the_real_store(live_cfg):
    """The isolation must not turn these into tests of an empty directory."""
    assert live_cfg.paths.curated.resolve() == (
        PROJECT_ROOT / "data" / "curated").resolve()
    assert live_cfg.paths.snapshots.resolve() == (
        PROJECT_ROOT / "data" / "snapshots").resolve()


def test_a_pipeline_run_under_live_cfg_lands_in_the_sandbox(runnable_cfg):
    """The end-to-end statement: run the real pipeline, and prove the row went
    to the sandbox rather than to the research record."""
    from prosignal.ledger import Ledger
    from prosignal.pipeline import run_analysis

    before = Ledger(runnable_cfg.paths.ledger).count()
    real_before = Ledger(PROJECT_ROOT / "data" / "ledger").count()

    run_analysis(runnable_cfg)

    assert Ledger(runnable_cfg.paths.ledger).count() == before + 1
    assert Ledger(PROJECT_ROOT / "data" / "ledger").count() == real_before, (
        "a test run reached the production research ledger"
    )


def test_every_write_path_is_redirected(live_cfg):
    """Not just the ledger. A test that fills data/raw or data/cache is a test
    that changes what the next real run reads."""
    sandbox_root = live_cfg.paths.ledger.resolve().parent
    for attr in ("ledger", "logs", "raw", "cache"):
        path = getattr(live_cfg.paths, attr).resolve()
        assert sandbox_root in path.parents or path.parent == sandbox_root, (
            f"paths.{attr} still points into the project at {path}"
        )
