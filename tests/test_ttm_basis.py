"""Trailing-twelve-month construction, and the seasonal blackout it removes.

The defect these cover: `_ttm` returned the newest single filing rather than a
trailing year, so `build_fundamental_panel` had to restrict itself to annual
reports to keep the scale consistent. Between an annual report ageing out and
the next one landing -- about five months of every year on a March year end --
that left the value block computed for the handful of companies on an off-cycle
year end, roughly 2% of the universe, while reporting itself as available.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.features.fundamental_factors import (
    FLOW_FIELDS,
    _prior,
    _ttm,
    available_as_of,
    build_fundamental_panel,
)


def _rows(symbol, periods, kind, **fields):
    frame = pd.DataFrame({"symbol": symbol, "period_end": pd.to_datetime(periods), "kind": kind})
    for name, values in fields.items():
        frame[name.replace("_", " ")] = values
    frame["available_on"] = available_as_of(frame["period_end"], frame["kind"])
    return frame


AS_OF = pd.Timestamp("2026-02-15")


def test_flow_field_sums_four_quarters_rather_than_taking_the_latest():
    """A quarter's revenue over a full market cap is not a sales yield."""
    q = _rows("ACME", ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
              "quarterly", Total_Revenue=[100.0, 110.0, 120.0, 130.0])
    got = _ttm(q, "Total Revenue", AS_OF)
    assert got["ACME"] == pytest.approx(460.0)


def test_level_field_takes_the_newest_observation_and_is_never_summed():
    q = _rows("ACME", ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
              "quarterly", Total_Debt=[100.0, 110.0, 120.0, 130.0])
    assert "Total Debt" not in FLOW_FIELDS
    assert _ttm(q, "Total Debt", AS_OF)["ACME"] == pytest.approx(130.0)


def test_quarterly_filings_carry_the_block_once_the_annual_has_aged_out():
    """The regression. On 15 Feb the FY25 annual is 320 days old; four quarters
    filed since then are current, and the block must use them."""
    annual = _rows("ACME", ["2025-03-31"], "annual", Net_Income=[400.0])
    quarters = _rows("ACME", ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"],
                     "quarterly", Net_Income=[110.0, 120.0, 130.0, 140.0])
    # 2026-03-31 is not yet public on 2026-02-15, so three fresher quarters exist.
    both = pd.concat([annual, quarters], ignore_index=True)
    got = _ttm(both, "Net Income", AS_OF)
    # Three fresher quarters is not a full year, so the annual figure stands
    # rather than a nine-month sum being passed off as a year.
    assert got["ACME"] == pytest.approx(400.0)

    later = pd.Timestamp("2026-06-30")   # now all four post-annual quarters are public
    assert _ttm(both, "Net Income", later)["ACME"] == pytest.approx(500.0)


def test_nothing_is_read_before_it_was_public():
    """The newest quarter enters the sum only on its filing deadline, not before."""
    q = _rows("ACME", ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
              "quarterly", Net_Income=[100.0, 110.0, 120.0, 130.0])
    # quarterly deadline is 45 days, so 2025-12-31 becomes public on 2026-02-14.
    before = _ttm(q, "Net Income", pd.Timestamp("2026-02-13"))
    after = _ttm(q, "Net Income", pd.Timestamp("2026-02-14"))
    assert before.empty          # only three quarters public: no full year yet
    assert after["ACME"] == pytest.approx(460.0)


def test_a_single_quarter_does_not_become_a_year():
    """Three quarters is not a trailing year, and must not be reported as one."""
    q = _rows("ACME", ["2025-06-30", "2025-09-30", "2025-12-31"],
              "quarterly", Net_Income=[110.0, 120.0, 130.0])
    assert _ttm(q, "Net Income", AS_OF).empty


def test_a_symbol_whose_newest_filing_is_stale_is_dropped_not_carried():
    old = _rows("ACME", ["2021-03-31"], "annual", Net_Income=[400.0])
    assert _ttm(old, "Net Income", AS_OF, max_age_days=240).empty
    assert _ttm(old, "Net Income", AS_OF, max_age_days=None)["ACME"] == pytest.approx(400.0)


def test_growth_compares_a_year_against_the_year_before_it():
    """_prior on a different basis than _ttm turns arithmetic into growth."""
    quarters = _rows("ACME",
                     ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                      "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
                     "quarterly", Total_Revenue=[100.0] * 4 + [110.0] * 4)
    now = _ttm(quarters, "Total Revenue", AS_OF)["ACME"]
    before = _prior(quarters, "Total Revenue", AS_OF)["ACME"]
    assert now == pytest.approx(440.0)
    assert before == pytest.approx(400.0)
    assert now / before - 1.0 == pytest.approx(0.10)     # 10%, not 340%


def test_the_panel_is_computed_off_quarterly_filings_in_the_blackout_window():
    """End to end: the value block must not go dark between annual reports."""
    syms = [f"S{i:02d}" for i in range(30)]
    frames = []
    for i, s in enumerate(syms):
        frames.append(_rows(s, ["2025-03-31"], "annual",
                            Net_Income=[40.0 + i], Common_Stock_Equity=[200.0 + i],
                            Total_Revenue=[400.0 + i], Total_Debt=[10.0]))
        frames.append(_rows(s, ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"],
                            "quarterly", Net_Income=[11.0 + i] * 4,
                            Common_Stock_Equity=[210.0 + i] * 4,
                            Total_Revenue=[105.0 + i] * 4, Total_Debt=[10.0] * 4))
    statements = pd.concat(frames, ignore_index=True)
    mcap = pd.Series({s: 1000.0 for s in syms})

    # 2026-06-30: the FY25 annual is 456 days old and would be cut by any sane
    # staleness rule; the four quarters through 2026-03-31 are public.
    panel = build_fundamental_panel(statements, mcap, dt.date(2026, 6, 30),
                                    enabled=["earnings_yield", "book_to_price"],
                                    max_age_days=240)
    assert not panel.empty
    assert panel["earnings_yield"].notna().mean() == pytest.approx(1.0)
    # earnings_yield must reflect the four-quarter sum, not one quarter.
    assert panel.loc["S00", "earnings_yield"] == pytest.approx(44.0 / 1000.0)
