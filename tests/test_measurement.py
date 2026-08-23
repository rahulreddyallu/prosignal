"""Measurement periods.

The guarantee under test is narrow and is the only reason the feature
exists: evidence gathered before a change must never be pooled with
evidence gathered after it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from prosignal import measurement as M


D1 = dt.date(2026, 1, 1)
D2 = dt.date(2026, 4, 1)


def test_nothing_is_running_on_a_fresh_install(tmp_path: Path):
    assert M.active(tmp_path) is None
    assert M.periods(tmp_path) == []


def test_a_period_pins_what_it_was_measuring(tmp_path: Path):
    p = M.start(tmp_path, config_version="v1@aaa", git_commit="c1", today=D1)
    assert p.status == "RUNNING" and p.open
    assert p.config_version == "v1@aaa" and p.git_commit == "c1"
    assert len(p.fingerprint) == 16


def test_starting_again_closes_the_previous_one(tmp_path: Path):
    """Two open periods make "which one does this run belong to" ambiguous,
    and that ambiguity is the thing a period exists to remove."""
    a = M.start(tmp_path, config_version="v1@aaa", today=D1)
    b = M.start(tmp_path, config_version="v2@bbb", today=D2)
    rows = {x.id: x for x in M.periods(tmp_path)}
    assert rows[a.id].ended == D2.isoformat()
    assert rows[b.id].open
    assert M.active(tmp_path).id == b.id


def test_two_periods_under_different_configs_get_different_fingerprints(tmp_path: Path):
    a = M.start(tmp_path, config_version="v1@aaa", git_commit="c1", today=D1)
    b = M.start(tmp_path, config_version="v2@bbb", git_commit="c2", today=D2)
    assert a.fingerprint != b.fingerprint


def test_a_period_covers_only_its_own_dates(tmp_path: Path):
    """This is what keeps the two samples apart."""
    a = M.start(tmp_path, config_version="v1", today=D1)
    M.start(tmp_path, config_version="v2", today=D2)
    a = next(x for x in M.periods(tmp_path) if x.id == a.id)
    assert a.covers("2026-02-15")
    assert not a.covers("2025-12-31")
    assert not a.covers("2026-06-01"), "a closed period must not claim later runs"


def test_an_open_period_claims_everything_after_it_starts(tmp_path: Path):
    p = M.start(tmp_path, config_version="v1", today=D1)
    assert p.covers("2027-09-09")


def test_a_config_change_under_an_open_period_marks_it_drifted(tmp_path: Path):
    """It is now measuring two different models, and an average over both
    describes neither."""
    M.start(tmp_path, config_version="v1@aaa", today=D1)
    got = M.active(tmp_path, config_version="v2@bbb")
    assert got.status == "DRIFTED"
    assert got.drifted_to == "v2@bbb"
    # Sticky: reading again does not clear it.
    assert M.active(tmp_path, config_version="v2@bbb").status == "DRIFTED"


def test_the_same_config_does_not_trip_drift(tmp_path: Path):
    M.start(tmp_path, config_version="v1@aaa", today=D1)
    assert M.active(tmp_path, config_version="v1@aaa").status == "RUNNING"


def test_stopping_keeps_the_period(tmp_path: Path):
    M.start(tmp_path, config_version="v1", today=D1)
    closed = M.stop(tmp_path, today=D2)
    assert closed.ended == D2.isoformat()
    assert M.active(tmp_path) is None
    assert len(M.periods(tmp_path)) == 1, "history must survive stopping"


def test_stopping_when_nothing_runs_is_not_an_error(tmp_path: Path):
    assert M.stop(tmp_path) is None


def test_period_of_finds_the_owning_window(tmp_path: Path):
    M.start(tmp_path, config_version="v1", label="Pilot", today=D1)
    M.start(tmp_path, config_version="v2", label="After tuning", today=D2)
    assert M.period_of(tmp_path, "2026-02-01").label == "Pilot"
    assert M.period_of(tmp_path, "2026-05-01").label == "After tuning"
    assert M.period_of(tmp_path, "2025-01-01") is None


def test_a_torn_line_does_not_lose_the_rest(tmp_path: Path):
    M.start(tmp_path, config_version="v1", today=D1)
    with (tmp_path / M.PERIODS_FILE).open("a", encoding="utf-8") as fh:
        fh.write('{"id": "torn')
    assert len(M.periods(tmp_path)) == 1
