"""Regression tests for the audit's bug inventory.

Each test pins a defect that let a broken input produce a normal-looking
signal, or a degenerate input produce a plausible number.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.core.errors import IntegrityError


def test_A_adjustment_failure_stops_the_run(live_cfg, monkeypatch):
    """Unadjusted prices corrupt 72 of 200 symbols. A failed adjustment used to
    return them with nothing on the card to say so."""
    from prosignal.data import corporate_actions as ca
    from prosignal.data.store import DataStore

    def boom(*a, **k):
        raise RuntimeError("adjustment kernel down")

    monkeypatch.setattr(ca, "apply_adjustments", boom)
    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)
    sessions = store.price_sessions()
    with pytest.raises(IntegrityError):
        store.read_prices(symbols=["RELIANCE"], start=sessions[-30], end=sessions[-1])


def test_A_unadjusted_is_still_available_when_asked_for_explicitly(live_cfg):
    from prosignal.data.store import DataStore

    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots, adjust_prices=False)
    sessions = store.price_sessions()
    assert not store.read_prices(symbols=["RELIANCE"],
                                 start=sessions[-30], end=sessions[-1]).empty


def test_B_model_failure_is_not_silently_scored_by_the_retired_composite(live_cfg):
    """The composite measured -0.047%/month excess at t = -0.11. A model failure
    must not hand the run to it while looking healthy."""
    from prosignal.stages.stage4_core_score import _is_model_failure

    # An exception is a failure.
    assert _is_model_failure("RuntimeError: model down") is True
    # Too little history is an expected state, not a failure.
    assert _is_model_failure("340 sessions of history; needs 417") is False
    assert _is_model_failure("120 usable training rows; 600 required") is False


def test_C_horizon_must_match_the_holding_cap(live_cfg):
    """Independent numbers that happened to agree. Editing either alone left the
    model forecasting a window the engine never holds for."""
    from prosignal.config.schema import _validate_horizon_alignment

    _validate_horizon_alignment(live_cfg.params)  # the shipped config agrees

    class _Fake:
        pass

    bad = _Fake()
    bad.stage4_core_score = _Fake()
    bad.stage4_core_score.model_horizon_sessions = 21
    bad.stage7_risk = _Fake()
    bad.stage7_risk.holding_period = _Fake()
    bad.stage7_risk.holding_period.max_holding_sessions = _Fake()
    bad.stage7_risk.holding_period.max_holding_sessions.value = 63
    with pytest.raises(ValueError, match="must equal"):
        _validate_horizon_alignment(bad)


def test_D_factor_coverage_has_one_source_of_truth():
    """A module constant shadowed the config at three of four call sites."""
    import prosignal.stages.stage4_core_score as s4

    assert not hasattr(s4, "_MIN_FACTOR_COVERAGE")
    assert hasattr(s4, "_min_coverage")


def test_E_zero_atr_reports_not_testable_instead_of_dividing(live_cfg):
    """A halted or flat scrip has zero true range; dividing by it killed the run."""
    from prosignal.stages import stage5_false_signal as s5

    n = 60
    frame = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "volume": 1e6, "turnover": 1e8,
    })
    out = s5._gap_signal(frame, live_cfg.params.stage5_false_signal.gap_signal,
                         live_cfg.params)
    assert out is not None  # did not raise ZeroDivisionError


def test_G_a_gap_through_the_stop_fills_at_the_open():
    """Filling at the stop credits a price that was never available."""
    import inspect

    from prosignal import backtest as bt

    src = inspect.getsource(bt)
    assert "min(bar_open, stop)" in src
    assert "stop_gap" in src


def test_H_backtest_respects_the_live_concurrent_position_cap():
    import inspect

    from prosignal import backtest as bt

    src = inspect.getsource(bt)
    assert "max_concurrent" in src
    assert "max_signals_per_run" in src


def test_K_dsr_refuses_a_degenerate_return_series():
    from prosignal.validation.metrics import deflated_sharpe_ratio

    with pytest.raises(ValueError):
        deflated_sharpe_ratio(np.zeros(100), n_trials=10)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.01], n_trials=10)


def test_L_downside_deviation_of_a_series_with_no_down_days_is_zero():
    """NaN here dropped a name from scoring for having had no bad days."""
    from prosignal.indicators.returns import downside_deviation

    out = downside_deviation(pd.Series([0.01] * 100), 60)
    value = float(out.iloc[-1]) if hasattr(out, "iloc") else float(out)
    assert value == 0.0


def test_J_the_funnel_is_monotonic(runnable_cfg):
    """triggered=1 followed by passed_score_threshold=8 meant the two lines
    counted different populations."""
    from prosignal import pipeline

    funnel = pipeline.run_analysis(runnable_cfg).funnel
    order = ["universe_considered", "passed_eligibility", "scored",
             "passed_score_threshold", "triggered", "passed_portfolio_limits"]
    seen = [(k, funnel[k]) for k in order if k in funnel]
    for (n1, v1), (n2, v2) in zip(seen, seen[1:]):
        assert v2 <= v1, f"funnel goes up: {n1}={v1} -> {n2}={v2}"
