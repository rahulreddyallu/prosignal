"""Backtest driver -- the execution-realism properties.

The assertions here are the ones that decide whether a backtest is honest:
entry happens at the NEXT session's open, and a bar touching both stop and
target resolves as the stop. Both are the pessimistic choice, and both are the
difference between a truthful result and a flattering one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.backtest import BacktestResult, Trade, _simulate
from prosignal.costs import CostModel
from prosignal.data.types import DATE


class _Rec:
    """Minimal stand-in for a Recommendation, to isolate the fill logic."""

    def __init__(self, ticker="X", stop=90.0, t1=115.0, t2=130.0):
        self.ticker = ticker
        self.initial_stop = stop
        self.target_1 = t1
        self.target_2 = t2


def _bars(rows):
    return pd.DataFrame(
        [{DATE: pd.Timestamp(d), "open": o, "high": h, "low": l, "close": c}
         for d, o, h, l, c in rows]
    )


def test_entry_is_the_next_session_open_not_the_signal_close(cfg):
    """The single most important execution assumption.

    Filling at the signal session's close would grant a full session of
    foresight, which is enough on its own to make a losing strategy look good.
    """
    bars = _bars([
        ("2026-01-01", 100, 101, 99, 100),   # signal session
        ("2026-01-02", 105, 106, 104, 105),  # entry session -- open is 105
        ("2026-01-05", 106, 120, 105, 118),
    ])
    t = _simulate(_Rec(), dt.date(2026, 1, 1), None, {"X": bars}, CostModel(cfg), cfg)
    assert t.entry_price == 105.0, "entry must be the NEXT session's open"
    assert t.entry_date == dt.date(2026, 1, 2)


def test_bar_touching_both_stop_and_target_resolves_as_the_stop(cfg):
    """Daily bars cannot reveal intraday order, so take the adverse outcome."""
    bars = _bars([
        ("2026-01-01", 100, 101, 99, 100),
        ("2026-01-02", 100, 101, 99, 100),
        ("2026-01-05", 100, 120, 85, 100),  # hits target 115 AND stop 90
    ])
    t = _simulate(_Rec(stop=90.0, t1=115.0), dt.date(2026, 1, 1), None,
                  {"X": bars}, CostModel(cfg), cfg)
    assert t.exit_reason == "stop"
    assert t.exit_price == 90.0
    assert t.net_return < 0


def test_target_exit_is_taken_when_no_stop_touch(cfg):
    bars = _bars([
        ("2026-01-01", 100, 101, 99, 100),
        ("2026-01-02", 100, 101, 99, 100),
        ("2026-01-05", 101, 118, 100, 117),
    ])
    t = _simulate(_Rec(stop=90.0, t1=115.0), dt.date(2026, 1, 1), None,
                  {"X": bars}, CostModel(cfg), cfg)
    assert t.exit_reason == "target_1"
    assert t.gross_return > 0


def test_costs_always_reduce_the_return(cfg):
    bars = _bars([
        ("2026-01-01", 100, 101, 99, 100),
        ("2026-01-02", 100, 101, 99, 100),
        ("2026-01-05", 101, 118, 100, 117),
    ])
    t = _simulate(_Rec(), dt.date(2026, 1, 1), None, {"X": bars}, CostModel(cfg), cfg)
    assert t.net_return < t.gross_return
    assert t.cost_bps > 0


def test_mae_and_mfe_are_recorded(cfg):
    bars = _bars([
        ("2026-01-01", 100, 101, 99, 100),
        ("2026-01-02", 100, 101, 99, 100),
        ("2026-01-05", 100, 108, 94, 100),
        ("2026-01-06", 100, 118, 99, 117),
    ])
    t = _simulate(_Rec(stop=90.0, t1=115.0), dt.date(2026, 1, 1), None,
                  {"X": bars}, CostModel(cfg), cfg)
    assert t.mae < 0, "adverse excursion must be recorded"
    assert t.mfe > 0


def test_no_future_bars_means_no_trade(cfg):
    """A signal on the last available session cannot be filled."""
    bars = _bars([("2026-01-01", 100, 101, 99, 100)])
    assert _simulate(_Rec(), dt.date(2026, 1, 1), None, {"X": bars},
                     CostModel(cfg), cfg) is None


def test_empty_result_reports_no_statistics_rather_than_zeros(cfg):
    """Zero trades must not be presented as a 0% win rate."""
    stats = BacktestResult().stats()
    assert stats["n_trades"] == 0
    assert "no closed trades" in stats["note"]
    assert "win_rate" not in stats


def test_stats_are_computed_from_the_trades(cfg):
    res = BacktestResult()
    for i, (g, n) in enumerate([(0.10, 0.09), (-0.05, -0.06), (0.20, 0.19)]):
        res.trades.append(Trade(
            ticker=f"S{i}", signal_date=dt.date(2026, 1, 1), entry_date=dt.date(2026, 1, 2),
            entry_price=100.0, stop=90.0, target_1=115.0, target_2=130.0,
            exit_date=dt.date(2026, 2, 1), exit_price=100 * (1 + g), exit_reason="target_1",
            holding_sessions=10, gross_return=g, net_return=n,
        ))
    s = res.stats()
    assert s["n_trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["mean_return_net"] == pytest.approx((0.09 - 0.06 + 0.19) / 3)
    assert s["cost_drag_per_trade"] > 0
