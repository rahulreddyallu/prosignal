"""Price-band awareness at entry and at the stop.

Entry assumes a fill at the next session's open and a stopped exit assumes
min(open, stop). Neither holds when the stock is locked at its band: one price
was available all session, and it was not ours. Recording a fill anyway invents
a counterparty.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.indicators.circuit import (
    BandState, annotate_band_state, band_state, is_untradeable,
)


def test_a_frozen_bar_at_a_standard_band_is_a_circuit():
    assert band_state(105, 105, 105, 100, 1e6) is BandState.UPPER_CIRCUIT
    assert band_state(95, 95, 95, 100, 1e6) is BandState.LOWER_CIRCUIT
    assert band_state(110, 110, 110, 100, 1e6) is BandState.UPPER_CIRCUIT   # 10%
    assert band_state(120, 120, 120, 100, 1e6) is BandState.UPPER_CIRCUIT   # 20%


def test_a_frozen_bar_away_from_any_band_is_frozen_not_a_circuit():
    """One print in a thin name breaks the same execution assumption, but
    claiming it was a circuit would assert something we cannot see."""
    assert band_state(100.5, 100.5, 100.5, 100, 1e6) is BandState.FROZEN


def test_a_normal_bar_is_open():
    assert band_state(110, 90, 100, 100, 1e6) is BandState.OPEN


def test_zero_volume_is_no_trade():
    assert band_state(100, 100, 100, 100, 0) is BandState.NO_TRADE


def test_a_missing_column_is_unknown_and_never_untradeable():
    """The distinction that matters. NO_TRADE is a fact about the market;
    UNKNOWN is a fact about our inputs. Treating a missing column as a market
    event let a feed gap reject the entire universe while reading like a halt."""
    assert band_state(np.nan, 100, 100, 100, 1e6) is BandState.UNKNOWN
    assert band_state(100, 100, 100, 100, np.nan) is BandState.UNKNOWN
    assert not is_untradeable(BandState.UNKNOWN)
    for state in (BandState.UPPER_CIRCUIT, BandState.LOWER_CIRCUIT,
                  BandState.FROZEN, BandState.NO_TRADE):
        assert is_untradeable(state)


def test_a_missing_previous_close_costs_the_label_not_the_fact():
    """high == low establishes frozen on its own; prev_close only names it."""
    assert band_state(100, 100, 100, np.nan, 1e6) is BandState.FROZEN


def test_annotate_matches_a_real_locked_session():
    """SIEL and UEL both locked on 2026-08-18, opposite directions."""
    frame = pd.DataFrame([
        {"high": 38.57, "low": 38.57, "close": 38.57, "prev_close": 35.07, "volume": 5e5},
        {"high": 211.36, "low": 211.36, "close": 211.36, "prev_close": 222.48, "volume": 3e5},
        {"high": 250.0, "low": 240.0, "close": 245.0, "prev_close": 244.0, "volume": 9e5},
    ])
    states = annotate_band_state(frame)
    assert states.tolist() == ["upper_circuit", "lower_circuit", "open"]


def test_annotate_and_scalar_agree():
    rng = np.random.default_rng(4)
    n = 400
    close = 100 * np.exp(rng.normal(0, 0.03, n))
    frame = pd.DataFrame({
        "high": close * (1 + rng.uniform(0, 0.02, n)),
        "low": close * (1 - rng.uniform(0, 0.02, n)),
        "close": close, "prev_close": close / (1 + rng.normal(0, 0.02, n)),
        "volume": rng.integers(0, 1e6, n).astype(float),
    })
    lock = rng.integers(0, n, 40)
    frame.loc[lock, "high"] = frame.loc[lock, "low"] = frame.loc[lock, "close"]
    vec = annotate_band_state(frame)
    for i in range(n):
        r = frame.iloc[i]
        scalar = band_state(r["high"], r["low"], r["close"], r["prev_close"], r["volume"])
        assert vec.iloc[i] == scalar.value, f"row {i} disagrees"


# --------------------------------------------------------------------------
# backtest behaviour
# --------------------------------------------------------------------------

def _bars(rows):
    return pd.DataFrame([
        {"date": pd.Timestamp(d), "open": o, "high": h, "low": l,
         "close": c, "prev_close": p, "volume": v}
        for d, o, h, l, c, p, v in rows
    ])


def test_a_stop_breached_on_a_locked_bar_is_not_filled():
    """The bar the brief is about: the stop is breached, but the only price
    all session was the lower band. There was no bid to hit."""
    from prosignal.backtest import _simulate

    class _Rec:
        ticker = "X"
        initial_stop = 90.0
        target_1 = 130.0
        target_2 = 140.0

    rows = [("2026-01-01", 100, 101, 99, 100, 100, 1e6),
            ("2026-01-02", 100, 101, 99, 100, 100, 1e6)]   # entry bar
    # locked limit-down through the stop, three sessions, then recovers
    for day, px in [("2026-01-05", 80.0), ("2026-01-06", 76.0), ("2026-01-07", 72.2)]:
        rows.append((day, px, px, px, px, px / 0.95, 1e5))
    rows.append(("2026-01-08", 75.0, 78.0, 74.0, 77.0, 72.2, 9e5))
    bars = {"X": _bars(rows)}

    from prosignal.config.loader import load_config
    from prosignal.costs import CostModel
    cfg = load_config()
    trade = _simulate(_Rec(), dt.date(2026, 1, 1), None, bars, CostModel(cfg), cfg)

    assert trade is not None
    # It must not have filled at 80, 76 or 72.2 -- none of those were reachable.
    assert "circuit" in trade.exit_reason, trade.exit_reason
    assert trade.exit_price not in (80.0, 76.0, 72.2)


def test_an_ordinary_gap_through_the_stop_still_fills_at_the_open():
    """The circuit path must not swallow the existing gap-fill behaviour."""
    from prosignal.backtest import _simulate
    from prosignal.config.loader import load_config
    from prosignal.costs import CostModel

    class _Rec:
        ticker = "X"
        initial_stop = 90.0
        target_1 = 130.0
        target_2 = 140.0

    rows = [("2026-01-01", 100, 101, 99, 100, 100, 1e6),
            ("2026-01-02", 100, 101, 99, 100, 100, 1e6),
            ("2026-01-05", 85.0, 88.0, 84.0, 86.0, 100.0, 9e5)]   # real gap, traded
    cfg = load_config()
    trade = _simulate(_Rec(), dt.date(2026, 1, 1), None, {"X": _bars(rows)},
                      CostModel(cfg), cfg)
    assert trade is not None
    assert trade.exit_reason == "stop_gap"
    assert trade.exit_price == pytest.approx(85.0)
