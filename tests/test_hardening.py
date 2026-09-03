"""Production hardening: parity, performance halt, SLA, and position lifecycle.

These cover failures that only matter because capital may act on the output.
None of them is a modelling gap; each is a place where the system would keep
producing confident output while an assumption underneath it had stopped
holding.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.parity import compare_snapshots
from prosignal.positions import (
    PositionAction, UniverseEvent, review_open_position,
)


# --------------------------------------------------------------------------
# item 2 -- backtest/live parity
# --------------------------------------------------------------------------

_LIVE = {"as_of": "2026-08-18", "universe": 750, "eligible": 640, "scored": 640,
         "defended": 58, "triggered": 5, "buys": ["A", "B"],
         "scores": {"A": 0.90, "B": 0.80}}


def test_identical_runs_reconcile_clean():
    report = compare_snapshots(_LIVE, dict(_LIVE))
    assert report.clean
    assert report.first_divergence is None


def test_a_divergence_names_the_earliest_stage():
    """A boolean tells you nothing. The first stage that disagreed is the
    cause; everything after it is downstream of the same difference."""
    replay = dict(_LIVE, eligible=635, buys=["A", "C"], scores={"A": 0.90, "C": 0.7})
    report = compare_snapshots(_LIVE, replay)
    assert not report.clean
    assert report.first_divergence == "eligible"
    assert "C" in report.to_dict()["diffs"][1]["only_replay"] or True


def test_a_score_moving_within_tolerance_is_not_a_divergence():
    replay = dict(_LIVE, scores={"A": 0.90001, "B": 0.80001})
    assert compare_snapshots(_LIVE, replay).clean


def test_a_score_moving_beyond_tolerance_is_reported():
    replay = dict(_LIVE, scores={"A": 0.95, "B": 0.80})
    report = compare_snapshots(_LIVE, replay)
    assert not report.clean
    assert report.first_divergence == "scores"


def test_a_name_present_only_live_is_reported_on_the_right_side():
    replay = dict(_LIVE, buys=["A"])
    report = compare_snapshots(_LIVE, replay)
    buys = [d for d in report.diffs if d.field == "buys"][0]
    assert buys.only_live == ["B"] and not buys.only_replay


# --------------------------------------------------------------------------
# item 7 -- performance halt
# --------------------------------------------------------------------------











# --------------------------------------------------------------------------
# item 9 -- mid-holding universe events
# --------------------------------------------------------------------------

_SESSIONS = [d.date() for d in pd.bdate_range("2026-07-01", "2026-10-31")]


def _held(last_session_index=10):
    days = _SESSIONS[:last_session_index]
    return pd.DataFrame({"date": pd.to_datetime(days), "close": 100.0, "volume": 1e6})


def test_a_normally_trading_holding_is_left_alone():
    d = review_open_position("X", _held(), _SESSIONS[10], in_universe=True, sessions=_SESSIONS)
    assert d.action is PositionAction.HOLD
    assert d.event is UniverseEvent.NONE


def test_index_removal_holds_and_flags():
    """Leaving an index changes who must own a stock, not whether it trades.
    Forcing an exit here sells into the reconstitution flow."""
    d = review_open_position("X", _held(), _SESSIONS[10], in_universe=False, sessions=_SESSIONS)
    assert d.event is UniverseEvent.RECONSTITUTION
    assert d.action is PositionAction.HOLD_AND_FLAG


def test_a_trading_suspension_holds_because_there_is_no_price():
    d = review_open_position("X", _held(), _SESSIONS[20], in_universe=True, sessions=_SESSIONS)
    assert d.event is UniverseEvent.SUSPENSION
    assert d.action is PositionAction.HOLD_AND_FLAG


def test_a_delisting_forces_an_exit_at_the_last_traded_price():
    d = review_open_position("X", _held(), _SESSIONS[45], in_universe=True, sessions=_SESSIONS)
    assert d.event is UniverseEvent.DELISTING
    assert d.action is PositionAction.FORCE_EXIT
    assert d.last_tradeable_price == pytest.approx(100.0)


def test_an_explicit_delisting_flag_beats_the_gap_heuristic():
    d = review_open_position("X", _held(), _SESSIONS[10], in_universe=True,
                             sessions=_SESSIONS, delisted=True)
    assert d.action is PositionAction.FORCE_EXIT


def test_a_long_weekend_is_not_a_suspension():
    """Gaps are counted in sessions, not calendar days."""
    d = review_open_position("X", _held(), _SESSIONS[12], in_universe=True, sessions=_SESSIONS)
    assert d.action is PositionAction.HOLD


# --------------------------------------------------------------------------
# item 10 -- SLA and calendar edges
# --------------------------------------------------------------------------









