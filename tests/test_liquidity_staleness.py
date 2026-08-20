"""Short-window liquidity check against the window the position was sized on.

Sizing uses 21-session ADTV. That is the slowest-moving estimate available
exactly when it matters most: liquidity leaves a name in days, and the trailing
window keeps quoting depth that is no longer there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.config.loader import load_config
from prosignal.costs import CostModel
from prosignal.stages.stage7_risk import _recent_liquidity, build_plan


def _frame(turnovers, n=60):
    idx = pd.bdate_range("2026-01-01", periods=n)
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 0.6, n))
    tno = np.full(n, float(turnovers[0]))
    tail = turnovers[1:]
    if tail:
        tno[-len(tail):] = tail
    return pd.DataFrame({
        "date": idx, "open": close, "high": close + 1.5,
        "low": close - 1.5, "close": close, "prev_close": close,
        "volume": 1e6, "turnover": tno,
    })


@pytest.fixture
def cfg():
    return load_config()


def test_a_liquidity_collapse_raises_a_warning(cfg):
    frame = _frame([1e8] + [5e6] * 5)          # 100% -> 5% of normal
    ratio, warning = _recent_liquidity(frame, 1e8, cfg.params.stage7_risk)
    assert ratio == pytest.approx(0.05)
    assert warning is not None
    assert "LIQUIDITY" in warning


def test_steady_liquidity_raises_nothing(cfg):
    frame = _frame([1e8] + [1e8] * 5)
    ratio, warning = _recent_liquidity(frame, 1e8, cfg.params.stage7_risk)
    assert ratio == pytest.approx(1.0)
    assert warning is None


def test_a_missing_short_window_is_silent_rather_than_alarming(cfg):
    """Absence of data is not evidence of a collapse."""
    frame = _frame([1e8]).head(3)
    ratio, warning = _recent_liquidity(frame, 1e8, cfg.params.stage7_risk)
    assert ratio is None and warning is None


def test_no_adtv_means_no_comparison(cfg):
    frame = _frame([1e8] + [1e6] * 5)
    assert _recent_liquidity(frame, None, cfg.params.stage7_risk) == (None, None)
    assert _recent_liquidity(frame, 0.0, cfg.params.stage7_risk) == (None, None)


def test_the_warning_reaches_the_risk_plan(cfg):
    """It has to be visible in the run output, not just computed."""
    frame = _frame([1e8] + [4e6] * 5)
    plan = build_plan("X", frame, float(frame["close"].iloc[-1]), 0.9, 1e8,
                      cfg, CostModel(cfg))
    assert plan.liquidity_ratio_recent is not None
    assert plan.liquidity_warning is not None
    assert any("LIQUIDITY" in n for n in plan.notes)


def test_the_check_never_changes_the_position_size(cfg):
    """It informs the operator; it does not act."""
    healthy = _frame([1e8] + [1e8] * 5)
    collapsed = _frame([1e8] + [4e6] * 5)
    price = float(healthy["close"].iloc[-1])
    a = build_plan("X", healthy, price, 0.9, 1e8, cfg, CostModel(cfg))
    b = build_plan("X", collapsed, price, 0.9, 1e8, cfg, CostModel(cfg))
    assert a.risk_category_inputs.get("qty") == b.risk_category_inputs.get("qty")
    assert a.stop_price == b.stop_price
