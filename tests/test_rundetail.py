"""The run payload the Today screen reads -- and the ordering bug it had.

`rundetail` is what makes the screen show the newest RUN rather than the newest
JOB. The nightly cron never touches the API, so no job row exists for it; the
ledger keeps a summary with no `factor_detail`, so the evidence panel cannot be
rebuilt from it. This module is the only path by which a cron-produced run
reaches the interface, and it had no tests at all.

THE BUG IT HAD. Files were named `{as_of_date}_{run_id}.json` and `load_latest`
took the last one lexically. Run ids are random hex, so for two runs on one
date -- which the ledger shows is the norm, not the exception -- the sort
picked an arbitrary one. It served the OLDEST of three and the screen showed a
stale payload while looking entirely normal.
"""
from __future__ import annotations

import datetime as dt
import json
import types

import pytest

from prosignal import rundetail


# --------------------------------------------------------------- fixtures
def _cfg(tmp_path, *, enabled=True, subdir="runs"):
    ledger = types.SimpleNamespace(write_run_detail=enabled,
                                   run_detail_subdir=subdir)
    return types.SimpleNamespace(
        paths=types.SimpleNamespace(ledger=tmp_path),
        params=types.SimpleNamespace(ledger=ledger),
    )


def _run(run_id: str, generated: dt.datetime, as_of=dt.date(2026, 8, 25)):
    """A real AnalysisRun, so `shape` is exercised rather than stubbed."""
    from prosignal.core.contracts import FinalSignalOutput, RegimeState, RunContext
    from prosignal.core.enums import TrendRegime, VolContext, VolTercile
    from prosignal.pipeline import AnalysisRun

    regime = RegimeState(
        as_of_date=as_of, trend_regime=TrendRegime.UPTREND,
        vol_tercile=VolTercile.LOW, vol_context=VolContext.STABLE,
        regime_bucket="uptrend/low", momentum_multiplier=1.0,
        quality_multiplier=1.0, sector_rs_multiplier=1.0,
    )
    output = FinalSignalOutput(
        run_id=run_id, trial_id="T", as_of_date=as_of, generated_at=generated,
        engine_version="0.1.0", config_version="cfg@aaa", regime_state=regime,
    )
    context = RunContext(
        run_id=run_id, trial_id="T", as_of_date=as_of, started_at=generated,
        engine_version="0.1.0", schema_version="1", config_version="cfg@aaa",
    )
    return AnalysisRun(output=output, context=context, timings_ms={}, funnel={})


# ------------------------------------------------------------- the ordering
def test_two_runs_on_one_date_serve_the_newest(tmp_path):
    """The defect exactly. Run ids are random hex and order nothing, so the
    generation timestamp has to be in the name."""
    cfg = _cfg(tmp_path)
    base = dt.datetime(2026, 8, 27, 8, 29, 37)
    ids = ["ffffffffffff", "000000000000", "aaaaaaaaaaaa"]
    for i, rid in enumerate(ids):
        rundetail.save(_run(rid, base + dt.timedelta(minutes=5 * i)), cfg)

    newest = rundetail.load_latest(cfg)
    assert newest["run_id"] == "aaaaaaaaaaaa", (
        "served a run that is not the most recently generated one")
    # And it is genuinely not the lexical winner among run ids.
    assert sorted(ids)[-1] != "aaaaaaaaaaaa"


def test_the_market_date_outranks_the_generation_time(tmp_path):
    """A backfill written today for a past date must not displace today's
    screen. Sorting by file mtime would do exactly that."""
    cfg = _cfg(tmp_path)
    rundetail.save(_run("todayrun0001", dt.datetime(2026, 8, 27, 8, 0),
                        as_of=dt.date(2026, 8, 25)), cfg)
    rundetail.save(_run("backfill0001", dt.datetime(2026, 8, 27, 9, 0),
                        as_of=dt.date(2026, 6, 1)), cfg)
    assert rundetail.load_latest(cfg)["as_of_date"] == "2026-08-25"


def test_the_filename_carries_all_three_keys(tmp_path):
    cfg = _cfg(tmp_path)
    rundetail.save(_run("abcdef123456", dt.datetime(2026, 8, 27, 8, 54, 26)), cfg)
    name = rundetail.available_runs(cfg)[0].name
    assert name.startswith("2026-08-25_")
    assert "20260827-085426" in name
    assert name.endswith("abcdef123456.json")


# ---------------------------------------------------------------- retention
def test_only_the_last_sixty_runs_are_kept(tmp_path):
    cfg = _cfg(tmp_path)
    base = dt.datetime(2026, 8, 27, 6, 0)
    for i in range(rundetail.KEEP_RUNS + 12):
        rundetail.save(_run(f"{i:012d}", base + dt.timedelta(minutes=i)), cfg)
    kept = rundetail.available_runs(cfg)
    assert len(kept) == rundetail.KEEP_RUNS
    assert rundetail.load_latest(cfg)["run_id"] == f"{rundetail.KEEP_RUNS + 11:012d}"


# ------------------------------------------------------------------ lookup
def test_a_run_can_be_fetched_by_id(tmp_path):
    cfg = _cfg(tmp_path)
    base = dt.datetime(2026, 8, 27, 8, 0)
    for i, rid in enumerate(["aaa000000001", "bbb000000002"]):
        rundetail.save(_run(rid, base + dt.timedelta(minutes=i)), cfg)
    assert rundetail.load(cfg, "aaa000000001")["run_id"] == "aaa000000001"
    assert rundetail.load(cfg, "nosuchrun") is None


# ------------------------------------------------------------- failure paths
def test_a_display_cache_never_fails_the_run_that_produced_it(tmp_path):
    """The ledger is the permanent record and a failure to write it fails the
    run. This is a cache; it must swallow its own errors."""
    cfg = _cfg(tmp_path)
    broken = types.SimpleNamespace(output=None, funnel={})
    assert rundetail.save(broken, cfg) is None          # does not raise


def test_a_truncated_file_is_skipped_not_fatal(tmp_path):
    cfg = _cfg(tmp_path)
    base = dt.datetime(2026, 8, 27, 8, 0)
    rundetail.save(_run("good00000001", base), cfg)
    rundetail.save(_run("torn00000002", base + dt.timedelta(minutes=1)), cfg)
    newest = rundetail.available_runs(cfg)[-1]
    newest.write_text('{"run_id": "torn0000', encoding="utf-8")
    assert rundetail.load_latest(cfg)["run_id"] == "good00000001"


def test_the_write_is_atomic(tmp_path):
    """A reader opening the file mid-write must see the previous payload or
    the new one, never half of either."""
    import inspect
    src = inspect.getsource(rundetail.save)
    assert ".tmp" in src and "os.replace" in src


def test_writing_can_be_switched_off(tmp_path):
    cfg = _cfg(tmp_path, enabled=False)
    assert rundetail.save(_run("aaa000000001", dt.datetime(2026, 8, 27, 8, 0)), cfg) is None
    assert rundetail.available_runs(cfg) == []
    assert rundetail.load_latest(cfg) is None


def test_an_empty_directory_is_not_an_error(tmp_path):
    cfg = _cfg(tmp_path)
    assert rundetail.load_latest(cfg) is None
    assert rundetail.available_runs(cfg) == []


# --------------------------------------------------------------- the payload
def test_the_payload_carries_what_the_screen_cannot_get_from_the_ledger(tmp_path):
    """The ledger keeps a summary with no `factor_detail`, which is why this
    file exists at all."""
    cfg = _cfg(tmp_path)
    rundetail.save(_run("abcdef123456", dt.datetime(2026, 8, 27, 8, 0)), cfg)
    payload = rundetail.load_latest(cfg)
    for key in ("run_id", "as_of_date", "generated_at", "regime", "funnel",
                "recommendations", "watchlist", "slate", "stage_timings_ms"):
        assert key in payload, f"the screen reads {key} and it is not persisted"
    assert payload["regime"]["bucket"] == "uptrend/low"
    # Serialisable as written -- the API returns this verbatim.
    json.loads(json.dumps(payload, default=str))
