"""Corporate-action parsing, adjustment, and unexplained-jump detection.

Why this file matters more than its size suggests: an unadjusted 5:1 split
appears in the price series as a single-session return of -80%. That one number
poisons a 12-1 momentum score for a full year and would rank the stock at the
very bottom of the universe for reasons that have nothing to do with its
performance. Getting this right is load-bearing for every factor downstream.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.data.corporate_actions import (
    apply_adjustments,
    build_adjustment_factors,
    detect_unexplained_jumps,
    merge_action_sources,
    parse_action_subject,
    plausible_price_factors,
)
from prosignal.data.types import DATE, SYMBOL, coerce_ohlcv

from .conftest import make_sessions, synthetic_prices


# =============================================================================
# parsing
# =============================================================================


@pytest.mark.parametrize(
    "subject,expected_type,expected_factor",
    [
        ("Face Value Split From Rs.10/- To Rs.2/-", "split", 0.2),
        ("Face Value Split From Rs 10 To Rs 1", "split", 0.1),
        ("Bonus 1:1", "bonus", 0.5),
        ("Bonus 2:1", "bonus", 1 / 3),
        ("Bonus 1:2", "bonus", 2 / 3),
        ("Interim Dividend - Rs 5 Per Share", "dividend", 1.0),
    ],
)
def test_parse_action_subject(subject, expected_type, expected_factor):
    action_type, factor, _ = parse_action_subject(subject)
    assert action_type == expected_type
    assert factor == pytest.approx(expected_factor, rel=1e-6)


def test_rights_is_flagged_but_not_adjusted():
    """A rights issue's price effect needs the subscription price, which the
    subject line does not carry. Guessing would be worse than declining."""
    action_type, factor, note = parse_action_subject("Rights 1:5 @ Premium Rs 100")
    assert action_type == "rights"
    assert factor == 1.0
    assert "not derivable" in note


def test_unparsed_subject_defaults_to_no_rescaling():
    action_type, factor, note = parse_action_subject("Scheme of Arrangement")
    assert factor == 1.0
    assert action_type == "other"
    assert "unparsed" in note


def test_empty_subject_is_safe():
    assert parse_action_subject("")[1] == 1.0
    assert parse_action_subject(None)[1] == 1.0


# =============================================================================
# adjustment factors
# =============================================================================


def test_factors_apply_only_before_ex_date():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    actions = pd.DataFrame(
        [{"ex_date": pd.Timestamp("2026-01-03"), "ratio": 0.5, SYMBOL: "AAA"}]
    )
    factors = build_adjustment_factors(dates, actions)
    assert list(factors.to_numpy()) == [0.5, 0.5, 1.0, 1.0]


def test_multiple_actions_compound():
    dates = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])
    actions = pd.DataFrame(
        [
            {"ex_date": pd.Timestamp("2026-02-01"), "ratio": 0.5, SYMBOL: "AAA"},
            {"ex_date": pd.Timestamp("2026-03-01"), "ratio": 0.2, SYMBOL: "AAA"},
        ]
    )
    factors = build_adjustment_factors(dates, actions)
    assert factors.iloc[0] == pytest.approx(0.1)  # both actions still ahead
    assert factors.iloc[1] == pytest.approx(0.2)  # only the March split ahead
    assert factors.iloc[2] == pytest.approx(1.0)


def test_apply_adjustments_preserves_rupee_turnover():
    """Price down 2x, volume up 2x -- traded value must be unchanged."""
    sessions = make_sessions(10, end=dt.date(2026, 3, 10))
    prices = synthetic_prices(["AAA"], sessions)
    ex_date = pd.Timestamp(sessions[5])
    actions = pd.DataFrame(
        [
            {
                SYMBOL: "AAA",
                "ex_date": ex_date,
                "action_type": "bonus",
                "ratio": 0.5,
                "raw_details": "1:1",
                "source": "test",
            }
        ]
    )
    adjusted = apply_adjustments(prices, actions)

    before = adjusted[adjusted[DATE] < ex_date]
    raw_before = prices[prices[DATE] < ex_date]
    assert np.allclose(before["close"].to_numpy(), raw_before["close"].to_numpy() * 0.5)
    assert np.allclose(before["volume"].to_numpy(), raw_before["volume"].to_numpy() / 0.5)
    # price * volume invariant
    assert np.allclose(
        before["close"].to_numpy() * before["volume"].to_numpy(),
        raw_before["close"].to_numpy() * raw_before["volume"].to_numpy(),
    )
    # On and after the ex-date nothing is touched.
    after = adjusted[adjusted[DATE] >= ex_date]
    raw_after = prices[prices[DATE] >= ex_date]
    assert np.allclose(after["close"].to_numpy(), raw_after["close"].to_numpy())


def test_apply_adjustments_is_a_noop_without_actions(prices):
    out = apply_adjustments(prices, pd.DataFrame())
    assert np.allclose(out["close"].to_numpy(), prices["close"].to_numpy())


def test_adjustment_leaves_other_symbols_alone():
    sessions = make_sessions(8, end=dt.date(2026, 3, 10))
    prices = synthetic_prices(["AAA", "BBB"], sessions)
    actions = pd.DataFrame(
        [
            {
                SYMBOL: "AAA",
                "ex_date": pd.Timestamp(sessions[4]),
                "action_type": "split",
                "ratio": 0.2,
                "raw_details": "",
                "source": "test",
            }
        ]
    )
    adjusted = apply_adjustments(prices, actions)
    bbb_before = prices[prices[SYMBOL] == "BBB"].sort_values(DATE)["close"].to_numpy()
    bbb_after = adjusted[adjusted[SYMBOL] == "BBB"].sort_values(DATE)["close"].to_numpy()
    assert np.allclose(bbb_before, bbb_after)


# =============================================================================
# unexplained jumps
# =============================================================================


def test_plausible_factors_include_the_common_india_ratios():
    factors = plausible_price_factors()
    for expected in (0.5, 0.2, 0.1, 1 / 3, 2 / 3):
        assert any(abs(f - expected) < 1e-6 for f in factors)


def _inject_unadjusted_split(prices: pd.DataFrame, symbol: str, at: dt.date, factor: float):
    out = prices.copy()
    mask = (out[SYMBOL] == symbol) & (out[DATE] >= pd.Timestamp(at))
    for col in ("open", "high", "low", "close", "prev_close", "last"):
        out.loc[mask, col] = out.loc[mask, col] * factor
    return out


def test_detects_an_unadjusted_split():
    sessions = make_sessions(30, end=dt.date(2026, 8, 14))
    prices = synthetic_prices(["AAA", "BBB"], sessions)
    broken = _inject_unadjusted_split(prices, "AAA", sessions[20], 0.2)

    found = detect_unexplained_jumps(broken, None, min_ratio_gap=0.3, tolerance=0.05)
    assert not found.empty
    row = found.iloc[0]
    assert row[SYMBOL] == "AAA"
    assert row[DATE].date() == sessions[20]
    assert row["nearest_clean_factor"] == pytest.approx(0.2, abs=0.02)


def test_a_recorded_corporate_action_explains_the_jump():
    sessions = make_sessions(30, end=dt.date(2026, 8, 14))
    prices = synthetic_prices(["AAA"], sessions)
    broken = _inject_unadjusted_split(prices, "AAA", sessions[20], 0.2)
    actions = pd.DataFrame(
        [
            {
                SYMBOL: "AAA",
                "ex_date": pd.Timestamp(sessions[20]),
                "action_type": "split",
                "ratio": 0.2,
                "raw_details": "10 to 2",
                "source": "test",
            }
        ]
    )
    assert detect_unexplained_jumps(broken, actions, min_ratio_gap=0.3).empty


def test_clean_series_produces_no_false_positives(prices):
    """A normal random walk must not trip the detector."""
    assert detect_unexplained_jumps(prices, None, min_ratio_gap=0.3, tolerance=0.03).empty


def test_large_but_not_clean_move_is_ignored():
    """A genuine -45% crash is not a split and must not be flagged as one.

    0.55 sits between the nearest real corporate-action factors (0.5 from a
    1:1 bonus and 0.6 from a 2:3 bonus), so a well-calibrated detector leaves
    it alone.
    """
    sessions = make_sessions(30, end=dt.date(2026, 8, 14))
    prices = synthetic_prices(["AAA"], sessions)
    broken = _inject_unadjusted_split(prices, "AAA", sessions[20], 0.55)
    found = detect_unexplained_jumps(broken, None, min_ratio_gap=0.3, tolerance=0.03)
    assert found.empty


def test_shipped_tolerance_cannot_bridge_two_candidate_ratios(cfg):
    """Guards the calibration itself, against the config the engine ships with.

    The detector asks "is this move within `tolerance` of a clean corporate-
    action ratio?". If `tolerance` exceeds half the gap between two adjacent
    candidates, a single observed ratio can match both, and the check stops
    carrying information -- it just flags every large move. Since Stage 1 turns
    a hit into a hard rejection, that would silently shrink the universe.
    """
    factors = sorted(plausible_price_factors())
    gaps = [(b - a) / a for a, b in zip(factors, factors[1:]) if a > 0.05]
    min_gap = min(gaps)

    tolerance = float(
        cfg.params.stage1_data_quality.unexplained_split_ratio_tolerance.value
    )
    assert tolerance < min_gap / 2, (
        f"stage1.unexplained_split_ratio_tolerance={tolerance} is too loose: "
        f"adjacent candidate ratios are only {min_gap:.1%} apart, so a move can "
        f"be ambiguous between two of them"
    )

    search_hi = cfg.params.stage1_data_quality.unexplained_split_ratio_tolerance.search_range[1]
    assert float(search_hi) < min_gap / 2, (
        "the declared search_range permits a tolerance that would make the "
        "detector meaningless; tighten the upper bound"
    )


def test_reverse_split_is_detected():
    sessions = make_sessions(30, end=dt.date(2026, 8, 14))
    prices = synthetic_prices(["AAA"], sessions)
    broken = _inject_unadjusted_split(prices, "AAA", sessions[20], 5.0)
    found = detect_unexplained_jumps(broken, None, min_ratio_gap=0.3, tolerance=0.05)
    assert not found.empty


# =============================================================================
# merging
# =============================================================================


def test_merge_prefers_the_last_source():
    early = pd.DataFrame(
        [
            {
                SYMBOL: "AAA",
                "ex_date": pd.Timestamp("2026-01-05"),
                "action_type": "split",
                "ratio": 0.5,
                "raw_details": "scraped",
                "source": "yfinance",
            }
        ]
    )
    curated = early.copy()
    curated["ratio"] = 0.2
    curated["source"] = "csv_import"

    merged = merge_action_sources(early, curated)
    assert len(merged) == 1
    assert merged.iloc[0]["ratio"] == 0.2
    assert merged.iloc[0]["source"] == "csv_import"


def test_merge_handles_empty_inputs():
    assert merge_action_sources(None, pd.DataFrame()).empty
