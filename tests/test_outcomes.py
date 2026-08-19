"""Outcome resolution must match the backtester and must not score early."""
import datetime as dt
import json

import pandas as pd
import pytest

from prosignal import outcomes as O


def _bars(sym, start, n, open_=100.0, high=None, low=None, close=None):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates, "symbol": sym, "series": "EQ",
        "open": open_, "high": high if high is not None else open_ * 1.02,
        "low": low if low is not None else open_ * 0.98,
        "close": close if close is not None else open_,
        "volume": 1e6, "turnover": 1e8, "deliv_pct": 50.0,
    })


class _Store:
    def __init__(self, frame, sessions):
        self._f, self._s = frame, sessions
    def price_sessions(self): return self._s
    def read_prices(self, symbols=None, start=None, end=None, columns=None):
        return self._f.copy()


def _cfg(live_cfg):
    return live_cfg


def test_stop_wins_when_a_bar_touches_both(live_cfg, tmp_path):
    """A bar hitting stop and target must resolve as the stop.

    Daily bars cannot order intraday events, and assuming the favourable
    sequence is how a record inflates its own win rate.
    """
    n = int(live_cfg.params.stage7_risk.holding_period.max_holding_sessions.value) + 5
    f = _bars("AAA", "2024-01-01", n, open_=100.0, high=120.0, low=80.0, close=100.0)
    store = _Store(f, [d.date() for d in f["date"]])
    led, out = tmp_path / "ledger", tmp_path / "outcomes.jsonl"
    led.mkdir()
    (led / "runs-2024.jsonl").write_text(json.dumps({
        "run_id": "r1", "date": "2024-01-01", "config_version": "c", "engine_version": "e",
        "signals_generated": ["AAA"],
        "stocks_scored": [{"ticker": "AAA", "stop": 90.0, "target_1": 110.0,
                           "target_2": 115.0, "composite_score": 0.9}],
    }) + "\n")
    O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    rows = O.load_outcomes(out)
    assert len(rows) == 1
    assert rows[0]["exit_reason"] == "stop"
    assert rows[0]["gross_return"] < 0


def test_does_not_score_before_the_horizon_elapses(live_cfg, tmp_path):
    """A signal whose holding window has not finished must stay unresolved."""
    f = _bars("BBB", "2024-01-01", 4)
    store = _Store(f, [d.date() for d in f["date"]])
    led, out = tmp_path / "ledger", tmp_path / "outcomes.jsonl"
    led.mkdir()
    (led / "runs-2024.jsonl").write_text(json.dumps({
        "run_id": "r2", "date": "2024-01-01", "signals_generated": ["BBB"],
        "stocks_scored": [{"ticker": "BBB", "stop": 1.0, "target_1": 9999.0,
                           "composite_score": 0.8}],
    }) + "\n")
    res = O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    assert res["resolved"] == 0
    assert res["still_open"] == 1


def test_resolution_is_idempotent(live_cfg, tmp_path):
    """Running twice must not double-count a trade into the statistics."""
    n = int(live_cfg.params.stage7_risk.holding_period.max_holding_sessions.value) + 5
    f = _bars("CCC", "2024-01-01", n, open_=100.0, high=101.0, low=99.0, close=100.0)
    store = _Store(f, [d.date() for d in f["date"]])
    led, out = tmp_path / "ledger", tmp_path / "outcomes.jsonl"
    led.mkdir()
    (led / "runs-2024.jsonl").write_text(json.dumps({
        "run_id": "r3", "date": "2024-01-01", "signals_generated": ["CCC"],
        "stocks_scored": [{"ticker": "CCC", "stop": 50.0, "target_1": 200.0,
                           "composite_score": 0.7}],
    }) + "\n")
    a = O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    b = O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    assert a["resolved"] == 1
    assert b["resolved"] == 0
    assert len(O.load_outcomes(out)) == 1


def test_summarise_reports_sample_size_with_every_ratio():
    rows = [{"net_return": 0.1, "sessions_held": 5, "exit_reason": "target_1"},
            {"net_return": -0.05, "sessions_held": 9, "exit_reason": "stop"}]
    s = O.summarise(rows)
    assert s["n"] == 2
    assert 0.0 <= s["win_rate"] <= 1.0
    assert "expectancy_t" in s and "profit_factor" in s


def test_summarise_on_no_trades_returns_zero_not_a_ratio():
    assert O.summarise([]) == {"n": 0}


def test_calibration_refuses_to_report_on_too_few_trades():
    rows = [{"composite_score": 0.5, "net_return": 0.01}] * 4
    assert O.calibration(rows, buckets=4) == []
