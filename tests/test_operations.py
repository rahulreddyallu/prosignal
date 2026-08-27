"""The operator actions, and the guards that make them safe to expose.

Every one of these is reachable from a phone by anyone holding the token, so
the guard has to be in the server and not only in the dialog.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosignal import operations as O


def test_a_fresh_install_is_not_paused(tmp_path: Path):
    assert O.pause_state(tmp_path).paused is False


def test_pausing_records_when_and_why(tmp_path: Path):
    st = O.pause(tmp_path, "monitoring before I retune")
    assert st.paused and st.since
    assert st.reason == "monitoring before I retune"
    again = O.pause_state(tmp_path)
    assert again.paused and again.reason == "monitoring before I retune"


def test_an_unreadable_pause_file_still_pauses(tmp_path: Path):
    """Failing open here would silently resume a run the operator stopped."""
    (tmp_path / O.PAUSE_FILE).write_text("{ this is not json", encoding="utf-8")
    assert O.pause_state(tmp_path).paused is True


def test_resume_clears_the_flag_and_leaves_a_trail(tmp_path: Path):
    O.pause(tmp_path, "x")
    O.resume(tmp_path)
    assert O.pause_state(tmp_path).paused is False
    actions = [r["action"] for r in O.operations_log(tmp_path)]
    assert actions[:2] == ["resume", "pause"]


def test_the_run_script_declines_while_paused():
    """The flag is only worth writing if the thing it guards reads it."""
    script = Path("scripts/forward_run.sh").read_text(encoding="utf-8")
    assert "cron.paused" in script
    body = script[script.index("cron.paused"):]
    # It must exit BEFORE ingesting, not merely mention the file.
    assert "exit 0" in body[:400]
    assert body.index("exit 0") < body.index("data ingest")


class _Paths:
    def __init__(self, root: Path):
        for name in ("curated", "snapshots", "cache", "raw", "ledger"):
            p = root / name
            p.mkdir(parents=True, exist_ok=True)
            setattr(self, name, p)


def test_clearing_market_data_keeps_the_record(tmp_path: Path):
    """The store can be re-fetched from NSE. The ledger cannot be rebuilt at
    all -- it is the only record of what the engine said on a past date."""
    p = _Paths(tmp_path)
    (p.curated / "prices.parquet").write_bytes(b"x")
    (p.ledger / "runs.jsonl").write_text('{"run_id": "keep me"}\n', encoding="utf-8")

    O.reset_market_data(p)

    assert not (p.curated / "prices.parquet").exists()
    assert (p.ledger / "runs.jsonl").exists()
    assert "keep me" in (p.ledger / "runs.jsonl").read_text(encoding="utf-8")


def test_erasing_everything_removes_the_record_too(tmp_path: Path):
    p = _Paths(tmp_path)
    (p.curated / "prices.parquet").write_bytes(b"x")
    (p.ledger / "runs.jsonl").write_text('{"run_id": "gone"}\n', encoding="utf-8")

    O.erase_everything(p)

    assert not (p.curated / "prices.parquet").exists()
    assert not (p.ledger / "runs.jsonl").exists()


def test_the_note_that_an_erase_happened_survives_the_erase(tmp_path: Path):
    """An erase that deletes its own audit trail leaves a store that looks
    like it was never touched."""
    p = _Paths(tmp_path)
    O.pause(p.ledger, "before")
    O.erase_everything(p)
    actions = [r["action"] for r in O.operations_log(p.ledger)]
    assert "erase_everything" in actions
    assert "pause" in actions, "history before the erase was lost"


def test_a_torn_log_line_does_not_break_the_screen(tmp_path: Path):
    O.pause(tmp_path, "ok")
    with (tmp_path / O.OPS_LOG).open("a", encoding="utf-8") as fh:
        fh.write('{"action": "torn')            # no newline, no closing brace
    assert [r["action"] for r in O.operations_log(tmp_path)] == ["pause"]


def test_rebuilding_removes_nested_cache_files_not_just_the_top_level(tmp_path):
    """"Deleted" has to mean the bytes are gone. The raw HTTP cache nests a
    few levels deep, and a top-level unlink would leave all of it behind."""
    p = _Paths(tmp_path)
    deep = p.cache / "nse" / "bhav" / "2024"
    deep.mkdir(parents=True)
    (deep / "payload.zip").write_bytes(b"y" * 8192)
    (p.curated / "prices" / "year=2024").mkdir(parents=True)
    (p.curated / "prices" / "year=2024" / "part.parquet").write_bytes(b"x" * 4096)

    O.reset_market_data(p)

    left = [f for d in (p.curated, p.cache, p.raw)
            for f in d.rglob("*") if f.is_file()]
    assert left == [], f"still on disk: {left}"


def test_clearing_history_also_hides_the_open_calls_it_covered():
    """open_positions counts signals that have not closed, and it was handed
    every ledger row ever written while the outcomes beside it were filtered.
    A clear emptied the results and left the count intact -- one run after a
    clear reported fourteen open calls it had not made."""
    from pathlib import Path as _P
    src = _P("src/prosignal/api.py").read_text(encoding="utf-8")
    assert "_ledger_after_clear" in src
    call = src[src.index("op = _perf.open_positions("):]
    call = call[:call.index("max_hold")]
    assert "_ledger_after_clear()" in call, \
        "open positions must be counted from the rows the clear left visible"
    assert "read_all()" not in call, "unfiltered ledger rows are the bug"


# =========================================================================
# What "clear the market data" is allowed to clear
# =========================================================================
# `curated` holds the price store, which NSE serves again on request, and two
# things it does not: the trial registry -- the Deflated Sharpe's
# multiple-testing input -- and the model version archive, which is the refit
# gate's only recovery path. Both were wiped by a control labelled "Rebuild
# storage", whose own docstring said the record of what the engine SAID is
# kept.
#
# Losing the trial count does not merely forget a number. It LOWERS the bar the
# strategy has to clear, which is the direction that flatters.

import types as _types


def _paths(tmp_path):
    for name in ("curated", "snapshots", "cache", "raw", "ledger"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return _types.SimpleNamespace(
        curated=tmp_path / "curated", snapshots=tmp_path / "snapshots",
        cache=tmp_path / "cache", raw=tmp_path / "raw",
        ledger=tmp_path / "ledger")


def _stock_a_store(paths):
    from prosignal.validation.registry import FILENAME as REGISTRY
    (paths.curated / REGISTRY).write_text(
        '{"key":"k1","command":"research cpcv","label":"a","recorded_at":"x"}\n'
        '{"key":"k2","command":"research spread","label":"b","recorded_at":"y"}\n',
        encoding="utf-8")
    versions = paths.curated / "crosssec_model_versions"
    versions.mkdir()
    (versions / "crosssec_model_2026-08-25.json").write_text('{"coef":{}}', encoding="utf-8")
    (paths.curated / "prices").mkdir()
    (paths.curated / "prices" / "year=2026.parquet").write_bytes(b"not really parquet")
    (paths.curated / "crosssec_model.json").write_text('{"coef":{}}', encoding="utf-8")
    (paths.cache / "blob").write_bytes(b"cached")


def test_resetting_the_store_keeps_the_trial_registry(tmp_path):
    from prosignal.operations import reset_market_data
    from prosignal.validation.registry import FILENAME as REGISTRY, TrialRegistry
    paths = _paths(tmp_path)
    _stock_a_store(paths)
    before = TrialRegistry(paths.curated / REGISTRY).count()
    assert before == 2

    reset_market_data(paths)

    assert (paths.curated / REGISTRY).is_file(), (
        "the Deflated Sharpe's trial count cannot be re-fetched from NSE")
    assert TrialRegistry(paths.curated / REGISTRY).count() == before


def test_resetting_the_store_keeps_the_model_rollback_archive(tmp_path):
    from prosignal.operations import reset_market_data
    paths = _paths(tmp_path)
    _stock_a_store(paths)
    reset_market_data(paths)
    kept = list((paths.curated / "crosssec_model_versions").glob("*.json"))
    assert len(kept) == 1, "the refit gate's recovery path must survive"


def test_resetting_the_store_still_removes_the_market_data(tmp_path):
    """The preservation must not turn the reset into a no-op."""
    from prosignal.operations import reset_market_data
    paths = _paths(tmp_path)
    _stock_a_store(paths)
    reset_market_data(paths)
    assert not (paths.curated / "prices").exists()
    assert not (paths.curated / "crosssec_model.json").exists()
    assert not (paths.cache / "blob").exists()


def test_the_reset_reports_what_it_kept(tmp_path):
    from prosignal.operations import PRESERVE_ON_RESET, reset_market_data
    paths = _paths(tmp_path)
    _stock_a_store(paths)
    detail = reset_market_data(paths)
    for name in PRESERVE_ON_RESET:
        assert name in detail["kept"]
    assert detail["files_removed"]["curated"] >= 2


def test_erase_everything_is_still_allowed_to_erase_everything(tmp_path):
    """The other button means it. Only the operations log survives."""
    from prosignal.operations import erase_everything
    from prosignal.validation.registry import FILENAME as REGISTRY
    paths = _paths(tmp_path)
    _stock_a_store(paths)
    erase_everything(paths)
    assert not (paths.curated / REGISTRY).exists()
    assert not (paths.curated / "crosssec_model_versions").exists()
