"""The fundamental factor layer.

The failure mode that matters here is not arithmetic. A fundamental factor is
wrong when it was computed from a figure the market did not have yet, and that
mistake is invisible in the output: the numbers look reasonable and the
backtest improves.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.features.fundamental_factors import (
    ANNUAL_LAG_DAYS, FAMILIES, FUNDAMENTAL_FACTORS, QUARTERLY_LAG_DAYS,
    available_as_of, build_fundamental_panel, sector_neutralise, winsorise,
)


def _statements(symbols=("A", "B", "C"), period="2025-03-31"):
    rows = []
    for i, s in enumerate(symbols):
        rows.append({
            "symbol": s, "period_end": pd.Timestamp(period), "kind": "annual",
            "Total Revenue": 1000.0 * (i + 1), "Net Income": 100.0 * (i + 1),
            "EBITDA": 200.0 * (i + 1), "EBIT": 150.0 * (i + 1),
            "Gross Profit": 400.0 * (i + 1), "Interest Expense": 10.0,
            "Common Stock Equity": 500.0 * (i + 1), "Total Debt": 100.0 * (i + 1),
            "Total Assets": 2000.0 * (i + 1), "Operating Cash Flow": 120.0 * (i + 1),
            "Free Cash Flow": 80.0 * (i + 1), "Ordinary Shares Number": 100.0,
        })
    return pd.DataFrame(rows)


def test_availability_uses_the_filing_deadline_not_the_period_end():
    ends = pd.Series([pd.Timestamp("2025-03-31")])
    annual = available_as_of(ends, pd.Series(["annual"]))
    quarterly = available_as_of(ends, pd.Series(["quarterly"]))
    assert (annual.iloc[0] - ends.iloc[0]).days == ANNUAL_LAG_DAYS
    assert (quarterly.iloc[0] - ends.iloc[0]).days == QUARTERLY_LAG_DAYS


def test_a_real_filing_date_beats_the_deadline():
    """The deadline only approximates the fact. Where the fact exists, use it."""
    ends = pd.Series([pd.Timestamp("2025-03-31"), pd.Timestamp("2025-03-31")])
    filed = pd.Series([pd.Timestamp("2025-04-10"), pd.NaT])
    out = available_as_of(ends, pd.Series(["annual"] * 2), filed)
    assert out.iloc[0] == pd.Timestamp("2025-04-10")
    assert (out.iloc[1] - ends.iloc[1]).days == ANNUAL_LAG_DAYS


def test_kind_may_be_a_scalar():
    out = available_as_of(pd.Series([pd.Timestamp("2025-03-31")]), "annual")
    assert len(out) == 1


def test_nothing_is_visible_before_the_filing_deadline():
    """The whole point of the layer."""
    st = _statements()
    st["available_on"] = available_as_of(st["period_end"], st["kind"])
    mcap = pd.Series({"A": 1e5, "B": 2e5, "C": 3e5})
    just_before = (st["available_on"].iloc[0] - pd.Timedelta(days=1)).date()
    assert build_fundamental_panel(st, mcap, just_before).empty
    just_after = (st["available_on"].iloc[0] + pd.Timedelta(days=1)).date()
    assert not build_fundamental_panel(st, mcap, just_after).empty


def test_value_factors_are_computed_against_market_cap():
    st = _statements()
    st["available_on"] = available_as_of(st["period_end"], st["kind"])
    mcap = pd.Series({"A": 1000.0, "B": 1000.0, "C": 1000.0})
    p = build_fundamental_panel(st, mcap, dt.date(2025, 8, 1))
    # Net income rises A < B < C against an equal market cap, so does the yield.
    assert p.loc["A", "earnings_yield"] < p.loc["C", "earnings_yield"]
    assert p.loc["A", "earnings_yield"] == pytest.approx(0.1)


def test_accruals_are_earnings_the_cash_flow_does_not_support():
    st = _statements()
    st["available_on"] = available_as_of(st["period_end"], st["kind"])
    st.loc[st["symbol"] == "A", "Operating Cash Flow"] = 0.0   # all accrual
    mcap = pd.Series({"A": 1000.0, "B": 1000.0, "C": 1000.0})
    p = build_fundamental_panel(st, mcap, dt.date(2025, 8, 1))
    assert p.loc["A", "accruals"] > p.loc["B", "accruals"]


def test_a_factor_set_can_be_switched_off():
    st = _statements()
    st["available_on"] = available_as_of(st["period_end"], st["kind"])
    mcap = pd.Series({"A": 1000.0, "B": 1000.0, "C": 1000.0})
    p = build_fundamental_panel(st, mcap, dt.date(2025, 8, 1), enabled=["roe"])
    assert list(p.columns) == ["roe"]


def test_winsorise_clips_the_tail_that_would_swamp_a_z_score():
    """One company with equity near zero prints an ROE of several thousand
    percent; unclipped, its z-score collapses every other name toward zero."""
    s = pd.Series([1.0] * 495 + [10_000.0] * 5)
    w = winsorise(s)
    assert w.max() <= s.max() / 50.0        # the tail is pulled in hard
    assert w.std() <= s.std() / 50.0        # and stops dominating the spread
    assert w.iloc[0] == pytest.approx(1.0)  # the body is untouched


def test_sector_neutralise_removes_the_sector_median():
    s = pd.Series({"A": 10.0, "B": 20.0, "C": 1.0, "D": 3.0})
    sectors = pd.Series({"A": "Bank", "B": "Bank", "C": "IT", "D": "IT"})
    out = sector_neutralise(s, sectors)
    assert out["A"] == pytest.approx(-5.0)
    assert out["C"] == pytest.approx(-1.0)


def test_an_unknown_sector_is_not_pooled_into_a_fake_peer_group():
    s = pd.Series({"A": 10.0, "B": 20.0, "X": 99.0})
    sectors = pd.Series({"A": "Bank", "B": "Bank", "X": "Unknown"})
    out = sector_neutralise(s, sectors)
    assert out["X"] == 99.0


def test_every_factor_declares_a_family_and_a_direction():
    for spec in FUNDAMENTAL_FACTORS:
        assert spec.family in FAMILIES
        assert isinstance(spec.higher_is_better, bool)
        assert spec.rationale, f"{spec.name} has no stated rationale"


def test_division_by_a_near_zero_denominator_does_not_explode():
    st = _statements()
    st["available_on"] = available_as_of(st["period_end"], st["kind"])
    st.loc[st["symbol"] == "A", "Common Stock Equity"] = 0.0
    mcap = pd.Series({"A": 1000.0, "B": 1000.0, "C": 1000.0})
    p = build_fundamental_panel(st, mcap, dt.date(2025, 8, 1))
    assert not np.isinf(p.to_numpy(dtype="float64")).any()
