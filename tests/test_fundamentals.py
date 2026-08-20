"""XBRL parsing and point-in-time fundamental features.

`nse_fundamentals.parse_indas_xbrl` had ZERO test coverage until this file,
despite being regex-based parsing of third-party documents that feeds two
factors. These cases were found by adversarially probing the live parser; every
one of them passed, and they are pinned here so a future "tidy up the regex"
cannot silently break them.

The substring-collision case is the important one: `Income` is a substring of
`OtherIncome`, and a naive pattern would read one as the other.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.data.providers.nse_fundamentals import parse_indas_xbrl
from prosignal.data.types import SYMBOL
from prosignal.features import compute_features, point_in_time_snapshot


# =============================================================================
# XBRL parsing
# =============================================================================


def test_parses_a_namespaced_element():
    r = parse_indas_xbrl(
        b'<in-bse-fin:RevenueFromOperations contextRef="c1">1000</in-bse-fin:RevenueFromOperations>'
    )
    assert r["revenue"] == 1000.0


@pytest.mark.parametrize(
    "doc",
    [
        b"<RevenueFromOperations>1000</RevenueFromOperations>",
        b"<xyz:RevenueFromOperations>1000</xyz:RevenueFromOperations>",
        b'<in-bse-fin:RevenueFromOperations unitRef="INR">1000</in-bse-fin:RevenueFromOperations>',
    ],
)
def test_namespace_prefix_does_not_matter(doc):
    """Filers use inconsistent prefixes across years; matching binds to the
    local element name rather than a namespace that varies."""
    assert parse_indas_xbrl(doc)["revenue"] == 1000.0


def test_indian_comma_grouping_is_parsed():
    """1,00,000 is one lakh in Indian digit grouping, not one hundred."""
    r = parse_indas_xbrl(b"<in-bse-fin:RevenueFromOperations>1,00,000</in-bse-fin:RevenueFromOperations>")
    assert r["revenue"] == 100000.0


def test_negative_values_survive():
    """A loss must stay a loss; dropping the sign would flip a bad quarter good."""
    r = parse_indas_xbrl(b"<in-bse-fin:ProfitLossForPeriod>-500</in-bse-fin:ProfitLossForPeriod>")
    assert r["net_profit"] == -500.0


def test_income_does_not_collide_with_other_income():
    """`Income` is a substring of `OtherIncome`. A loose pattern reads one as
    the other, which would silently corrupt every margin computed from it."""
    r = parse_indas_xbrl(
        b"<in-bse-fin:OtherIncome>77</in-bse-fin:OtherIncome>"
        b"<in-bse-fin:Income>999</in-bse-fin:Income>"
    )
    assert r["other_income"] == 77.0
    assert r["total_income"] == 999.0


@pytest.mark.parametrize(
    "doc",
    [
        b"<in-bse-fin:RevenueFromOperations></in-bse-fin:RevenueFromOperations>",
        b'<in-bse-fin:RevenueFromOperations xsi:nil="true"/>',
        b"<in-bse-fin:Revenue",       # truncated download
        b"",                          # empty response
        b"<html>404 not found</html>",
    ],
)
def test_unusable_documents_yield_none_never_garbage(doc):
    """A malformed filing must produce absence, not a plausible wrong number."""
    r = parse_indas_xbrl(doc)
    assert r["revenue"] is None
    assert all(v is None for v in r.values())


# =============================================================================
# point-in-time gate -- the whole reason this feed is trustworthy
# =============================================================================


def _filings(rows):
    return pd.DataFrame(
        [
            {
                SYMBOL: "X",
                "filing_date": f,
                "period_end": p,
                "revenue": rev,
                "net_profit": np_,
                "profit_before_tax": np_ * 1.3,
                "finance_costs": 10.0,
                "paid_up_capital": 1000.0,
                "face_value": 10.0,
                "shares_outstanding": 100.0,
            }
            for f, p, rev, np_ in rows
        ]
    )


def test_filings_not_yet_public_are_excluded():
    """The single rule that prevents lookahead.

    A quarter ending 31-Mar is not usable on 01-Apr; it became public on the
    filing date, which the measured NSE lag puts 9-45 days later.
    """
    frame = _filings([
        (dt.date(2026, 5, 15), dt.date(2026, 3, 31), 100.0, 10.0),
        (dt.date(2026, 2, 10), dt.date(2025, 12, 31), 90.0, 9.0),
    ])
    before = point_in_time_snapshot(frame, dt.date(2026, 4, 1))
    assert len(before) == 1
    assert before.iloc[0]["period_end"] == dt.date(2025, 12, 31)

    after = point_in_time_snapshot(frame, dt.date(2026, 5, 20))
    assert len(after) == 2


def test_rows_without_a_filing_date_are_dropped():
    frame = _filings([(dt.date(2026, 5, 15), dt.date(2026, 3, 31), 100.0, 10.0)])
    frame.loc[0, "filing_date"] = None
    assert point_in_time_snapshot(frame, dt.date(2026, 6, 1)).empty


# =============================================================================
# derived features
# =============================================================================


def _four_quarters(profit=10.0, revenue=100.0):
    return _filings([
        (dt.date(2026, 2, 1), dt.date(2025, 12, 31), revenue, profit),
        (dt.date(2025, 11, 1), dt.date(2025, 9, 30), revenue, profit),
        (dt.date(2025, 8, 1), dt.date(2025, 6, 30), revenue, profit),
        (dt.date(2025, 5, 1), dt.date(2025, 3, 31), revenue, profit),
    ])


def test_market_cap_uses_derived_share_count():
    feats = compute_features(_four_quarters(), {"X": 50.0}, dt.date(2026, 3, 1))
    assert feats.iloc[0]["market_cap"] == pytest.approx(100.0 * 50.0)


def test_earnings_yield_is_ttm_over_market_cap():
    feats = compute_features(_four_quarters(profit=10.0), {"X": 50.0}, dt.date(2026, 3, 1))
    # TTM profit 40 / market cap 5000
    assert feats.iloc[0]["earnings_yield"] == pytest.approx(40.0 / 5000.0)


def test_loss_making_company_gets_a_negative_yield_not_a_drop():
    """Negative yield ranks them last, which is correct. Excluding them would
    silently treat the worst names as neutral."""
    feats = compute_features(_four_quarters(profit=-5.0), {"X": 50.0}, dt.date(2026, 3, 1))
    assert feats.iloc[0]["earnings_yield"] < 0


def test_partial_ttm_is_not_a_ttm():
    """Three quarters must not be summed and called trailing twelve months."""
    frame = _filings([
        (dt.date(2026, 2, 1), dt.date(2025, 12, 31), 100.0, 10.0),
        (dt.date(2025, 11, 1), dt.date(2025, 9, 30), 100.0, 10.0),
        (dt.date(2025, 8, 1), dt.date(2025, 6, 30), 100.0, 10.0),
    ])
    feats = compute_features(frame, {"X": 50.0}, dt.date(2026, 3, 1))
    assert pd.isna(feats.iloc[0]["earnings_yield"])


def test_net_margin_is_ttm_profit_over_ttm_revenue():
    feats = compute_features(_four_quarters(profit=10.0, revenue=100.0), {"X": 50.0},
                             dt.date(2026, 3, 1))
    assert feats.iloc[0]["net_margin"] == pytest.approx(0.10)


def test_interest_coverage_formula():
    """(PBT + finance costs) / finance costs. PBT is 1.3x profit in the fixture,
    so TTM PBT = 52, finance = 40 -> (52+40)/40 = 2.3."""
    feats = compute_features(_four_quarters(profit=10.0), {"X": 50.0}, dt.date(2026, 3, 1))
    assert feats.iloc[0]["interest_coverage"] == pytest.approx((52.0 + 40.0) / 40.0)


def _eight_quarters(profits):
    """Eight filings, newest first, with the given quarterly profits."""
    ends = [
        dt.date(2025, 12, 31), dt.date(2025, 9, 30), dt.date(2025, 6, 30),
        dt.date(2025, 3, 31), dt.date(2024, 12, 31), dt.date(2024, 9, 30),
        dt.date(2024, 6, 30), dt.date(2024, 3, 31),
    ]
    return _filings([
        (end + dt.timedelta(days=32), end, 100.0, p) for end, p in zip(ends, profits)
    ])


def test_earnings_stability_is_negated_so_higher_is_better():
    """Every quality component must point the same way, or the weighting step
    needs a per-component sign and will eventually get one wrong."""
    steady = compute_features(
        _eight_quarters([10.0] * 8), {"X": 50.0}, dt.date(2026, 3, 1))
    erratic = compute_features(
        _eight_quarters([30.0, 2.0, 25.0, 1.0, 40.0, 3.0, 5.0, 35.0]),
        {"X": 50.0}, dt.date(2026, 3, 1))
    assert steady.iloc[0]["earnings_stability"] > erratic.iloc[0]["earnings_stability"]


def test_a_seasonal_but_stable_company_is_not_called_unstable():
    """The reason this is measured on trailing-twelve-month windows.

    Indian earnings are genuinely seasonal -- festive-quarter retail, agri and
    monsoon cycles. A company earning 40/10/10/20 every single year is entirely
    predictable, but its raw quarters have a large spread. Measured on raw
    quarters it scores as unstable, which penalises seasonality rather than
    risk. On TTM windows every season appears once per window, so the pattern
    cancels and the company reads as what it is.
    """
    seasonal = compute_features(
        _eight_quarters([40.0, 10.0, 10.0, 20.0, 40.0, 10.0, 10.0, 20.0]),
        {"X": 50.0}, dt.date(2026, 3, 1))
    volatile = compute_features(
        _eight_quarters([40.0, 10.0, 10.0, 20.0, 5.0, 60.0, 2.0, 30.0]),
        {"X": 50.0}, dt.date(2026, 3, 1))

    # Perfectly repeating seasonality means an identical TTM every window.
    assert seasonal.iloc[0]["earnings_stability"] == pytest.approx(0.0, abs=1e-9)
    assert volatile.iloc[0]["earnings_stability"] < seasonal.iloc[0]["earnings_stability"]

    # And the raw-quarter measure, which is what this replaced, gets it wrong:
    raw = np.array([40.0, 10.0, 10.0, 20.0])
    assert float(np.std(raw, ddof=1)) / abs(float(np.mean(raw))) > 0.5


def test_stability_needs_enough_quarters_to_see_variation():
    """One TTM window has no variation to measure, so the factor is absent
    rather than reported as perfectly stable."""
    feats = compute_features(_four_quarters(profit=10.0), {"X": 50.0}, dt.date(2026, 3, 1))
    assert pd.isna(feats.iloc[0]["earnings_stability"])


def test_no_price_means_no_market_cap_and_no_yield():
    feats = compute_features(_four_quarters(), {}, dt.date(2026, 3, 1))
    assert pd.isna(feats.iloc[0]["market_cap"])
    assert pd.isna(feats.iloc[0]["earnings_yield"])
    # margin does not need a price, so it must still be computed
    assert not pd.isna(feats.iloc[0]["net_margin"])
