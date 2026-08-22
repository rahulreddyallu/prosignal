"""A demerger is a corporate action that matches no clean split fraction.

VEDL separated on 2026-04-30: close 773.60 -> open 289.50, a ratio of 0.374.
It passed Stage 1 with no flags and stayed in the traded universe with every
lookback feature corrupted by a phantom -62.6% move.

Two independent defects let that through, and both are covered here.

  window   the unexplained-jump scan reused continuity_window_sessions (60)
           while prox_52w and resid_mom look back 253. The demerger sat 78
           sessions back: outside validation, inside every feature.

  rule     the detector required the ratio to sit within 3% of a plausible
           split fraction. The nearest to 0.374 is 0.40, 6.4% away -- because
           a demerger's ratio is the market value of what was spun out, not a
           clean fraction. Splits have clean ratios; demergers never do.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prosignal.config.loader import load_config
from prosignal.data.corporate_actions import detect_unexplained_jumps
from prosignal.data.types import DATE, SYMBOL


def _series(moves, symbol="X", start=100.0):
    px = [start]
    for m in moves:
        px.append(px[-1] * (1.0 + m))
    return pd.DataFrame({
        SYMBOL: symbol,
        DATE: pd.bdate_range("2026-01-01", periods=len(px)),
        "close": px,
    })


# =============================================================================
# the rule
# =============================================================================


def test_a_demerger_ratio_is_caught_even_though_it_matches_no_clean_factor():
    """The regression. 0.374 is 6.4% from the nearest clean factor of 0.40."""
    out = detect_unexplained_jumps(_series([0.01, 0.005, -0.626, 0.004]), None)
    assert len(out) == 1, (
        "a -62.6% overnight move is not something the market can produce; "
        "requiring it to match a clean split fraction reinstates the blind spot"
    )


def test_a_clean_split_is_still_caught():
    out = detect_unexplained_jumps(_series([0.01, 0.005, -0.800, 0.004]), None)
    assert len(out) == 1


@pytest.mark.parametrize("move", [-0.200, 0.200])
def test_a_move_exactly_on_the_circuit_limit_is_not_flagged(move):
    """NSE caps a scrip at 20%. Hitting the cap is legitimate and common; a
    rule that flags it would reject every limit-down day in the market."""
    out = detect_unexplained_jumps(_series([0.01, 0.005, move, 0.004]), None)
    assert out.empty


def test_a_move_beyond_the_circuit_limit_is_flagged():
    out = detect_unexplained_jumps(_series([0.01, 0.005, -0.205, 0.004]), None)
    assert len(out) == 1


def test_an_ordinary_move_is_not_flagged():
    out = detect_unexplained_jumps(_series([0.01, 0.005, -0.08, 0.004]), None)
    assert out.empty


def test_a_recorded_action_explains_the_jump():
    """The point of the detector is UNexplained jumps."""
    prices = _series([0.01, 0.005, -0.800, 0.004])
    ex = prices[DATE].iloc[3]
    actions = pd.DataFrame({SYMBOL: ["X"], "ex_date": [ex],
                            "action_type": ["split"], "ratio": [0.2]})
    assert detect_unexplained_jumps(prices, actions).empty


# =============================================================================
# the window
# =============================================================================


def test_the_scan_window_covers_the_longest_feature_lookback():
    """A corporate action inside the lookback corrupts the feature, so the scan
    must reach at least as far back as the features do."""
    from prosignal.features.crosssec import MIN_LOOKBACK

    cfg = load_config()
    scan = int(cfg.params.stage1_data_quality.unexplained_jump_lookback_sessions.value)
    assert scan >= MIN_LOOKBACK, (
        f"scan window {scan} is shorter than the {MIN_LOOKBACK}-session feature "
        f"lookback; an action in the gap is invisible to validation and fully "
        f"consumed by the model"
    )


def test_the_scan_window_is_not_the_continuity_window():
    """They are different concerns and sharing one parameter is what hid this.
    Continuity asks whether sessions are missing; the jump scan asks whether a
    price is discontinuous. The second must reach much further back."""
    cfg = load_config().params.stage1_data_quality
    scan = int(cfg.unexplained_jump_lookback_sessions.value)
    continuity = int(cfg.continuity_window_sessions.value)
    assert scan > continuity, (
        "the jump scan reusing continuity_window_sessions at 60 is exactly how "
        "the VEDL demerger passed validation"
    )


def test_a_jump_outside_the_window_is_not_scanned():
    """Confirms the window is actually applied rather than ignored."""
    prices = _series([0.01] * 40 + [-0.626] + [0.01] * 40)
    assert len(detect_unexplained_jumps(prices, None, lookback_sessions=200)) == 1
    assert detect_unexplained_jumps(prices, None, lookback_sessions=10).empty
