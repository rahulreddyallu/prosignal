"""The live monitors must flag and must never disable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prosignal import v2_monitor as M
from prosignal.features.v2 import V2_FACTORS


def _ic_panel(n_periods=60, n=250, inverted=(), seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for d in pd.bdate_range("2025-01-03", periods=n_periods, freq="W-FRI"):
        y = rng.normal(size=n)
        cols = {"date": [d] * n, "y42": y}
        for f in V2_FACTORS:
            x = rng.normal(size=n)
            if f.name in inverted:
                # oriented the wrong way: sign * rank correlates NEGATIVELY
                x = -f.sign * y * 0.4 + rng.normal(size=n) * 0.9
            cols[f.name + "_r"] = x
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def test_a_factor_that_has_inverted_is_flagged():
    p = _ic_panel(inverted=("mom_3_1",))
    health = {h.name: h for h in M.review_factors(M.rolling_factor_ic(p, "y42"))}
    assert health["mom_3_1"].inverted
    assert health["mom_3_1"].ic_t < -M.IC_ALERT_T


def test_a_short_history_returns_no_verdict_rather_than_a_confident_one():
    p = _ic_panel(n_periods=10)
    health = M.review_factors(M.rolling_factor_ic(p, "y42"))
    assert all(not h.inverted for h in health)
    assert all("floor" in h.note for h in health)


def test_the_monitor_returns_a_verdict_and_changes_no_state():
    """The point of the whole module: it reports. Nothing here mutates the
    factor set, the weights or the config."""
    before = [(f.name, f.sign, f.weight) for f in V2_FACTORS]
    p = _ic_panel(inverted=tuple(f.name for f in V2_FACTORS))
    M.review_factors(M.rolling_factor_ic(p, "y42"))
    after = [(f.name, f.sign, f.weight) for f in V2_FACTORS]
    assert before == after


def test_the_drawdown_breaker_flags_and_says_what_it_did_not_do():
    eq = np.array([1.0, 1.2, 1.3, 1.1, 1.0, 0.95])
    f = M.review_drawdown(eq)
    assert f.flagged
    assert f.drawdown < M.DRAWDOWN_FLAG
    assert "Nothing has been disabled" in f.note


def test_a_shallow_drawdown_does_not_flag():
    eq = np.array([1.0, 1.1, 1.05, 1.08])
    f = M.review_drawdown(eq)
    assert not f.flagged
    assert -0.06 < f.drawdown < 0.0


def test_an_empty_curve_is_a_stated_absence_not_a_zero():
    f = M.review_drawdown([])
    assert not f.flagged and np.isnan(f.drawdown)
    assert "no equity curve" in f.note
