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
        "stocks_scored": [{"ticker": "AAA", "last_close": 100.0, "stop": 90.0, "target_1": 110.0,
                           "target_2": 115.0, "composite_score": 0.9}],
    }) + "\n")
    O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    # epoch="*": what this test is about is the EXIT RULE, and the fixture's
    # config_version ("c") is deliberately not the open epoch's, so the default
    # per-epoch filter would hide the row and the assertion would pass
    # vacuously on an empty list. Reading every epoch keeps the subject the
    # subject.
    rows = O.load_outcomes(out, epoch="*")
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
        "stocks_scored": [{"ticker": "BBB", "last_close": 100.0, "stop": 1.0, "target_1": 9999.0,
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
        "stocks_scored": [{"ticker": "CCC", "last_close": 100.0, "stop": 50.0, "target_1": 200.0,
                           "composite_score": 0.7}],
    }) + "\n")
    a = O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    b = O.resolve_pending(store, led, out, live_cfg, as_of=f["date"].iloc[-1].date())
    assert a["resolved"] == 1
    assert b["resolved"] == 0
    # epoch="*" for the same reason as above: idempotence is about writing one
    # row, not about which experiment that row belongs to.
    assert len(O.load_outcomes(out, epoch="*")) == 1


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


def test_a_triggered_exit_resolves_without_waiting_out_the_window():
    """A stop hit on day two is final on day two. The gate that withheld it
    for a full 63 sessions hid 580 closed trades on the real ledger, and the
    average hold is 18 sessions against that window -- so most trades finish
    long before it does."""
    import inspect
    from prosignal import outcomes as O
    src = inspect.getsource(O._resolve_one)
    walk_at = src.index("walk = future.iloc[1:]")
    loop_at = src.index("for held, (_, bar) in enumerate")
    # No early return may sit between building the walk and inspecting it.
    assert "return None" not in src[walk_at:loop_at], \
        "a closed trade must not be withheld until the window elapses"
    # A partially elapsed window still must not become a time exit.
    assert "len(walk) < max_hold" in src[loop_at:]


def test_one_call_is_resolved_once_however_many_runs_issued_it():
    """A day can produce several ledger runs, each naming the same tickers,
    so the same call arrived at the resolver once per run -- 6,192 items over
    19 tickers on a real ledger. Each was resolved separately, wrote its own
    outcome row, and was collapsed again downstream. It cost 5.3s a request
    and it is where the duplicate counts on History came from."""
    from prosignal import outcomes as O
    rows = [
        {"run_id": "a", "date": "2024-01-09", "signals_generated": ["X"],
         "stocks_scored": [{"ticker": "X", "last_close": 100.0, "stop": 1.0, "target_1": 2.0}]},
        {"run_id": "b", "date": "2024-01-09", "signals_generated": ["X"],
         "stocks_scored": [{"ticker": "X", "last_close": 100.0, "stop": 1.0, "target_1": 2.0}]},
        {"run_id": "c", "date": "2024-01-10", "signals_generated": ["X"],
         "stocks_scored": [{"ticker": "X", "last_close": 100.0, "stop": 1.0, "target_1": 2.0}]},
    ]
    got = O._pending(rows, set())
    assert len(got) == 2, "two distinct calls, not three runs"
    assert {p["date"] for p in got} == {"2024-01-09", "2024-01-10"}


def test_a_call_already_on_file_is_not_retried_under_another_run_id():
    """Otherwise the other runs of a resolved call stay pending forever and
    are re-scanned on every single request."""
    from prosignal import outcomes as O
    rows = [{"run_id": "b", "date": "2024-01-09", "signals_generated": ["X"],
             "stocks_scored": [{"ticker": "X", "last_close": 100.0, "stop": 1.0, "target_1": 2.0}]}]
    assert O._pending(rows, set()) != []
    assert O._pending(rows, set(), {("X", "2024-01-09")}) == []
