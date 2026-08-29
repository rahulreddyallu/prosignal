"""C3/C4 -- the operating record was produced by an engine that no longer runs.

`outcomes.jsonl` is append-only and was partitioned by `exit_model` alone. Every
other thing that defines the strategy -- the universe, the sizer, the cost
model, the fitted coefficients -- could change without the record noticing, so
trades decided by two engines averaged into one win rate. That is the same
failure the `exit_model` partition already existed to prevent, applied to
everything else that moves.

TWO WRONG ANSWERS were available and both are pinned against here. Pooling
reports two engines as one. Filtering the retired rows away leaves a page
reading "no trades yet" when the truth is "the trades we have describe a
different engine" -- which is worse, because it looks like an engine that has
not been tested rather than one whose test does not apply.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosignal.outcomes import (EXIT_MODEL, PRE_EPOCH, epochs_present,
                                load_outcomes, summarise, summarise_by_epoch)

CUR = "2026-08-29-cafebabe"
OLD = "2026-01-02-deadbeef"


def _row(net, epoch, model=EXIT_MODEL, **kw):
    row = {"net_return": net, "sessions_held": 5, "exit_reason": "target",
           "exit_model": model, "epoch_id": epoch}
    row.update(kw)
    return row


@pytest.fixture
def record(tmp_path):
    """Ten losing trades from the retired engine, two winners from the current
    one -- the shape the real record has, exaggerated so pooling is visible."""
    path = tmp_path / "outcomes.jsonl"
    rows = ([_row(-0.03, None) for _ in range(10)] +
            [_row(0.06, CUR), _row(0.04, CUR)])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


# =============================================================================
# The partition
# =============================================================================


def test_an_unstamped_row_is_pre_epoch_and_not_dropped(record):
    """Rows written before epochs existed are a real record of what the engine
    did. They are just not a record of what THIS engine does."""
    counts = epochs_present(record)
    assert counts == {PRE_EPOCH: 10, CUR: 2}


def test_loading_serves_one_epoch(record):
    assert len(load_outcomes(record, epoch=CUR)) == 2
    assert len(load_outcomes(record, epoch=PRE_EPOCH)) == 10
    assert len(load_outcomes(record, epoch="*")) == 12


def test_the_epoch_and_exit_model_partitions_are_independent(record):
    """Both must bind. A row from the current epoch under an old exit rule is
    still a different strategy."""
    with record.open("a") as fh:
        fh.write(json.dumps(_row(0.5, CUR, model="stop-target-v1")) + "\n")
    assert len(load_outcomes(record, epoch=CUR)) == 2, (
        "a row resolved under a superseded exit rule was served because its "
        "epoch matched"
    )
    assert len(load_outcomes(record, model=None, epoch=CUR)) == 3


def test_pooling_is_reported_next_to_the_partition(record):
    """The number that keeps the partition from being re-collapsed.

    Per-epoch samples are small and someone will always want to add them up.
    The defence is not to forbid it -- it is to print what it costs.
    """
    rows = load_outcomes(record, epoch="*")
    out = summarise_by_epoch(rows, current=CUR)

    cur = next(e for e in out["epochs"] if e["is_current"])
    old = next(e for e in out["epochs"] if not e["is_current"])
    assert cur["n"] == 2 and old["n"] == 10
    assert cur["expectancy"] == pytest.approx(0.05)
    assert old["expectancy"] == pytest.approx(-0.03)

    assert out["pooled"]["n"] == 12
    assert out["pooled"]["expectancy"] == pytest.approx(
        (10 * -0.03 + 2 * 0.05) / 12)
    assert out["pooling_overstates_expectancy_by"] == pytest.approx(
        out["pooled"]["expectancy"] - 0.05)
    assert out["pooling_overstates_expectancy_by"] < 0, (
        "on this fixture pooling UNDERSTATES the current epoch; the sign has "
        "to survive, because a reader who sees only a magnitude cannot tell "
        "which way the mistake goes"
    )


def test_the_retired_cohort_is_labelled_rather_than_dropped(record):
    out = summarise_by_epoch(load_outcomes(record, epoch="*"), current=CUR)
    old = next(e for e in out["epochs"] if not e["is_current"])
    assert old["retired"] is True
    assert old["note"] and "not poolable" in old["note"]
    assert [e["epoch_id"] for e in out["epochs"]] == [PRE_EPOCH, CUR], (
        "the retired cohort must sort first: it is older, and a reader scans "
        "downward to the engine that is running now"
    )


def test_pooled_is_not_one_of_the_cohorts(record):
    """A caller iterating `epochs` and summing `n` must not double-count."""
    out = summarise_by_epoch(load_outcomes(record, epoch="*"), current=CUR)
    assert sum(e["n"] for e in out["epochs"]) == out["pooled"]["n"]


def test_an_epoch_with_no_trades_says_so_rather_than_borrowing_history(record):
    """The state this engine is actually in, and the one most likely to be
    papered over. An empty forward record is the honest answer for an engine
    whose universe and execution model just changed."""
    out = summarise_by_epoch(load_outcomes(record, epoch="*"),
                             current="2027-01-01-notyet")
    assert out["current_epoch_has_no_record"] is True
    assert all(e["retired"] for e in out["epochs"])
    assert out["pooled"]["n"] == 12, "the record is still there, still counted"


def test_a_single_epoch_record_does_not_claim_to_span_epochs(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(_row(0.01, CUR)) for _ in range(4)))
    out = summarise_by_epoch(load_outcomes(path, epoch="*"), current=CUR)
    assert out["spans_multiple_epochs"] is False
    assert out["pooling_overstates_expectancy_by"] == pytest.approx(0.0)


# =============================================================================
# Resolution stamps what it writes
# =============================================================================


def test_a_row_written_now_carries_the_open_epoch(live_cfg, tmp_path, monkeypatch):
    """Resolved end to end, not asserted about the source.

    Without this the partition is retrospective only: it would separate the
    history it already has and pool everything written from here, which is the
    same defect one epoch later.
    """
    import prosignal.outcomes as out_mod

    from .test_outcomes import _Store, _bars

    monkeypatch.setattr(out_mod, "_active_epoch_id", lambda *a, **k: CUR)

    n = int(live_cfg.params.stage7_risk.holding_period
            .max_holding_sessions.value) + 5
    f = _bars("AAA", "2024-01-01", n, open_=100.0, high=120.0, low=80.0,
              close=100.0)
    led, out = tmp_path / "ledger", tmp_path / "outcomes.jsonl"
    led.mkdir()
    (led / "runs-2024.jsonl").write_text(json.dumps({
        "run_id": "r1", "date": "2024-01-01", "config_version": "c",
        "engine_version": "e", "signals_generated": ["AAA"],
        "stocks_scored": [{"ticker": "AAA", "last_close": 100.0, "stop": 90.0,
                           "target_1": 110.0, "target_2": 115.0,
                           "composite_score": 0.9}],
    }) + "\n")
    out_mod.resolve_pending(_Store(f, [d.date() for d in f["date"]]), led, out,
                            live_cfg, as_of=f["date"].iloc[-1].date())

    rows = out_mod.load_outcomes(out, epoch="*")
    assert rows and all(r["epoch_id"] == CUR for r in rows), (
        "a row resolved now is unstamped, so everything written from here "
        "pools again"
    )
    assert out_mod.load_outcomes(out, epoch=PRE_EPOCH) == []


def test_a_missing_epoch_ledger_does_not_stop_resolution(tmp_path):
    """Resolution must not fail because provenance is missing. An unstamped
    row is honest about being unstamped; a crashed resolver loses evidence
    permanently, which is the failure `outcomes` exists to prevent."""
    from prosignal.outcomes import _active_epoch_id

    assert _active_epoch_id(tmp_path / "nowhere") == PRE_EPOCH


# =============================================================================
# C4 -- the interface
# =============================================================================


def test_the_endpoint_labels_retired_cohorts():
    """`/outcomes` must not serve one summary with no statement of which
    engine produced it."""
    import inspect

    from prosignal import api

    src = inspect.getsource(api.create_app)
    start = src.index("def outcomes_summary")
    body = src[start:start + 3000]
    assert 'epoch="*"' in body, (
        "the endpoint reads only the current epoch, so the retired record "
        "disappears from the page the moment an epoch opens"
    )
    assert "summarise_by_epoch" in body
    assert '"by_epoch"' in body and '"pooled"' in body


def test_the_per_name_history_is_not_filtered_by_epoch():
    """The opposite mistake, pinned.

    Statistics must not be pooled across epochs; a NAME'S OWN HISTORY must not
    be truncated by one. Serving only the current epoch there would erase
    every past call on a stock the moment an epoch opened.
    """
    import inspect

    from prosignal import api

    src = inspect.getsource(api.create_app)
    start = src.index("def _resolved_rows")
    assert 'epoch="*"' in src[start:start + 1500], (
        "the per-name history is filtered to the current epoch"
    )
