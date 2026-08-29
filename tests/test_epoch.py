"""D2 -- the research epoch: which engine produced a result.

`config_version` identified a run and covers none of the code, the data, the
universe policy, the feature schema or the execution model. Two materially
different engines could therefore produce results labelled identically, which
is how a corrected engine's numbers end up compared against an uncorrected
engine's on the assumption that both are "baseline-v1".

These tests pin the properties that make the ledger worth reading: it is
append-only, it refuses the operations that would let a record be quietly
rewritten, and it reports drift rather than acting on it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prosignal.validation.epoch import (EXECUTION_MODEL, STATUS_OPEN,
                                        STATUS_SUPERSEDED, STATUS_VOID,
                                        UNIVERSE_POLICY, Epoch, Identity,
                                        active, close_epoch, current_identity,
                                        drifted_from, load_all, open_epoch)


def _ident(**over) -> Identity:
    base = dict(code_sha="abc123", code_dirty=False, model_sources_sha="m1",
                config_version="baseline@1", data_manifest_sha="d1",
                feature_schema_sha="f1", universe_policy=UNIVERSE_POLICY,
                execution_model=EXECUTION_MODEL)
    base.update(over)
    return Identity(**base)                               # type: ignore[arg-type]


class _Cfg:
    """The three attributes `current_identity` reads."""

    def __init__(self, root: Path, version="baseline@1"):
        self.version = version

        class _P:
            pass
        self.paths = _P()
        self.paths.root = root
        self.paths.curated = root / "curated"
        self.paths.ledger = root


# =============================================================================
# Identity
# =============================================================================


def test_the_fingerprint_covers_everything_that_decides_an_answer():
    """Change any of the seven and the epoch is a different experiment."""
    base = _ident()
    for field, new in [("code_sha", "zzz"), ("model_sources_sha", "m2"),
                       ("config_version", "baseline@2"),
                       ("data_manifest_sha", "d2"),
                       ("feature_schema_sha", "f2"),
                       ("universe_policy", "anything-else"),
                       ("execution_model", "anything-else")]:
        assert replace(base, **{field: new}).fingerprint() != base.fingerprint(), (
            f"{field} moved and the fingerprint did not. A result produced "
            f"before and after that change would be labelled the same."
        )


def test_a_dirty_tree_is_recorded_but_not_fingerprinted():
    """Two clean runs of one commit must fingerprint identically.

    `code_dirty` describes the working tree an epoch was opened FROM, which is
    a fact worth recording and is not part of the model. Putting it in the
    hash would make an epoch un-reproducible from its own commit.
    """
    clean, dirty = _ident(code_dirty=False), _ident(code_dirty=True)
    assert clean.fingerprint() == dirty.fingerprint()
    assert "code_dirty" not in clean.differences(dirty), (
        "`differences` is what the drift report prints; a dirty tree is "
        "reported by the readiness gate, not as a change of engine"
    )


def test_differences_names_the_field_and_both_values():
    d = _ident(config_version="baseline@2").differences(_ident())
    assert d == ["config_version: baseline@1 -> baseline@2"], d


# =============================================================================
# The ledger
# =============================================================================


def test_closing_appends_rather_than_rewrites(tmp_path):
    """The history of what was believed, and when, has to survive.

    A ledger that rewrote the line would leave no evidence that an epoch was
    ever open -- and "this was believed for six weeks and then retired" is
    exactly the fact a reader needs.
    """
    e = open_epoch(tmp_path, _Cfg(tmp_path), label="first", identity=_ident())
    close_epoch(tmp_path, e.epoch_id, reason="superseded by the R9 refit")

    raw = (tmp_path / "epochs.jsonl").read_text().strip().splitlines()
    assert len(raw) == 2, "closing rewrote the record instead of appending"
    assert json.loads(raw[0])["status"] == STATUS_OPEN
    assert json.loads(raw[1])["status"] == STATUS_SUPERSEDED

    # And the reader collapses them to the latest state.
    eps = load_all(tmp_path)
    assert len(eps) == 1 and eps[0].status == STATUS_SUPERSEDED


def test_an_epoch_cannot_be_closed_without_a_reason(tmp_path):
    """An epoch that ends without a reason is indistinguishable from an
    experiment abandoned because it was going badly."""
    e = open_epoch(tmp_path, _Cfg(tmp_path), label="x", identity=_ident())
    for empty in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="without a reason"):
            close_epoch(tmp_path, e.epoch_id, reason=empty)
    assert active(tmp_path) is not None, "the refused close still closed it"


def test_an_epoch_closes_void_or_superseded_and_nothing_else(tmp_path):
    e = open_epoch(tmp_path, _Cfg(tmp_path), label="x", identity=_ident())
    for bad in ("CLOSED", "DONE", "OPEN", "ok"):
        with pytest.raises(ValueError, match="VOID or SUPERSEDED"):
            close_epoch(tmp_path, e.epoch_id, reason="r", status=bad)
    close_epoch(tmp_path, e.epoch_id, reason="r", status=STATUS_VOID)
    assert load_all(tmp_path)[0].status == STATUS_VOID


def test_two_epochs_cannot_be_open_at_once(tmp_path):
    """Two open epochs leave every ledger row ambiguous about which experiment
    it belongs to, which is the condition the epoch exists to remove."""
    open_epoch(tmp_path, _Cfg(tmp_path), label="first", identity=_ident())
    with pytest.raises(ValueError, match="still OPEN"):
        open_epoch(tmp_path, _Cfg(tmp_path), label="second",
                   identity=_ident(code_sha="def456"))

    # The override exists, is explicit, and is not the default.
    second = open_epoch(tmp_path, _Cfg(tmp_path), label="second",
                        identity=_ident(code_sha="def456"),
                        allow_while_open=True)
    assert second.is_open


def test_closing_an_unknown_epoch_raises_rather_than_appending(tmp_path):
    open_epoch(tmp_path, _Cfg(tmp_path), label="x", identity=_ident())
    with pytest.raises(KeyError):
        close_epoch(tmp_path, "2020-01-01-deadbeef", reason="r")
    assert len((tmp_path / "epochs.jsonl").read_text().strip().splitlines()) == 1


def test_a_corrupt_line_loses_one_epoch_and_not_the_ledger(tmp_path):
    """A crash mid-write must not take the whole provenance record with it."""
    open_epoch(tmp_path, _Cfg(tmp_path), label="good", identity=_ident())
    path = tmp_path / "epochs.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"epoch_id": "truncat\n')
    assert [e.label for e in load_all(tmp_path)] == ["good"]


def test_active_returns_the_latest_open_one(tmp_path):
    cfg = _Cfg(tmp_path)
    a = open_epoch(tmp_path, cfg, label="a", identity=_ident())
    close_epoch(tmp_path, a.epoch_id, reason="superseded", status=STATUS_SUPERSEDED,
                superseded_by="b")
    b = open_epoch(tmp_path, cfg, label="b", identity=_ident(code_sha="bbb"))
    assert active(tmp_path).epoch_id == b.epoch_id
    assert load_all(tmp_path)[0].superseded_by == "b"


# =============================================================================
# Drift is reported, never acted on
# =============================================================================


def test_drift_is_reported_and_the_ledger_is_untouched(tmp_path):
    """A gate that silently opens a new epoch when it notices a change is a
    gate nobody reads -- and it would decide, on its own, that a change is
    material. That judgement belongs to a person."""
    cfg = _Cfg(tmp_path)
    ident = _ident()
    open_epoch(tmp_path, cfg, label="v1", identity=ident)
    before = (tmp_path / "epochs.jsonl").read_text()

    moved = _Cfg(tmp_path, version="baseline@2")
    ep, diffs = drifted_from(tmp_path, moved)

    assert ep is not None
    # `current_identity` reads the real tree, so at minimum the config moved.
    assert any(d.startswith("config_version") for d in diffs), diffs
    assert (tmp_path / "epochs.jsonl").read_text() == before, (
        "noticing drift changed the ledger"
    )


def test_no_epoch_open_is_itself_the_finding(tmp_path):
    ep, diffs = drifted_from(tmp_path, _Cfg(tmp_path))
    assert ep is None
    assert diffs and "no epoch" in diffs[0]


# =============================================================================
# The shipped ledger
# =============================================================================


def test_v1_is_archived_and_says_why():
    """The engine this audit started on is retired ON THE RECORD.

    Not deleted, and not left OPEN either -- an epoch left open would mean the
    corrected engine's results were being attributed to the uncorrected one.
    """
    from prosignal.config.loader import load_config

    eps = load_all(Path(load_config().paths.ledger))
    if not eps:
        pytest.skip("no epoch ledger in this checkout")
    v1 = eps[0]
    assert v1.status in (STATUS_VOID, STATUS_SUPERSEDED)
    assert v1.close_reason, "an epoch closed with no reason"
    for fid in ("R1", "R3", "R9", "R13"):
        assert fid in v1.close_reason, (
            f"{fid} moves a traded number or voids the window and is not named "
            f"in the reason v1 was retired"
        )


def test_the_shipped_policies_name_the_findings_that_changed_them():
    """`universe_policy` and `execution_model` are strings in a fingerprint.
    They are only useful if they change when the policy does."""
    assert "r9" in UNIVERSE_POLICY.lower()
    assert "r13" in EXECUTION_MODEL.lower()
