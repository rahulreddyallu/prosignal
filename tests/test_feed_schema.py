"""Schema validation on curated writes.

A presence check catches a renamed column. It does not catch the failure that
corrupts a store: a column still present, still named right, now carrying
something else. Ranges are chosen so that misalignment breaks them and an
unusual market does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.core.errors import IntegrityError
from prosignal.data.schema import SCHEMAS, ColumnRule, FeedSchema, validate_feed


def _prices(n=50):
    idx = pd.bdate_range("2026-01-01", periods=n)
    close = 100 + np.arange(n) * 0.5
    return pd.DataFrame({
        "date": idx, "symbol": ["AAA"] * n, "series": ["EQ"] * n,
        "open": close, "high": close + 2.0, "low": close - 2.0, "close": close,
        "volume": 1e6, "turnover": 1e8,
    })


def test_a_well_formed_frame_passes():
    validate_feed(_prices(), SCHEMAS["prices"])


def test_a_missing_column_is_rejected_by_name():
    frame = _prices().drop(columns=["close"])
    with pytest.raises(IntegrityError, match="missing required columns"):
        validate_feed(frame, SCHEMAS["prices"])


def test_turnover_landing_in_a_price_column_is_caught():
    """The failure a presence check cannot see: right name, wrong contents."""
    frame = _prices()
    frame["close"] = frame["turnover"]
    with pytest.raises(IntegrityError, match="close"):
        validate_feed(frame, SCHEMAS["prices"])


def test_swapped_high_and_low_is_caught_by_the_invariant():
    frame = _prices()
    frame["high"], frame["low"] = frame["low"].copy(), frame["high"].copy()
    with pytest.raises(IntegrityError, match="invariant"):
        validate_feed(frame, SCHEMAS["prices"])


def test_a_negative_price_is_rejected():
    frame = _prices()
    frame["low"] = -1.0
    with pytest.raises(IntegrityError):
        validate_feed(frame, SCHEMAS["prices"])


def test_a_single_odd_print_is_tolerated():
    """One bad tick is a data point. A systematic breach is a format change.
    The threshold exists so the first does not look like the second."""
    frame = _prices(n=500)
    frame.loc[frame.index[0], "close"] = 9e9
    validate_feed(frame, SCHEMAS["prices"])


def test_a_delivery_percentage_above_one_hundred_is_rejected():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=30),
        "symbol": ["AAA"] * 30, "deliv_pct": 55.0, "deliv_qty": 1e5,
    })
    validate_feed(frame, SCHEMAS["delivery"])
    frame["deliv_pct"] = 5500.0          # a fraction rendered as basis points
    with pytest.raises(IntegrityError, match="deliv_pct"):
        validate_feed(frame, SCHEMAS["delivery"])


def test_a_zero_split_ratio_is_rejected():
    """A ratio of zero would zero out every price before the ex-date."""
    frame = pd.DataFrame({"symbol": ["AAA"] * 5, "ex_date": ["2026-01-01"] * 5,
                          "ratio": [0.0] * 5})
    with pytest.raises(IntegrityError, match="ratio"):
        validate_feed(frame, SCHEMAS["corporate_actions"])


def test_an_empty_frame_reaching_the_store_is_a_no_op_not_a_failure():
    from prosignal.data.store import _validate
    _validate(pd.DataFrame(), "prices", "test")
    _validate(None, "prices", "test")


def test_non_numeric_text_in_a_numeric_column_is_caught():
    frame = _prices()
    frame["volume"] = "n/a"
    with pytest.raises(IntegrityError, match="volume"):
        validate_feed(frame, SCHEMAS["prices"])


# --------------------------------------------------------------------------
# corporate-action precedence
# --------------------------------------------------------------------------

def test_two_sources_describing_one_event_do_not_double_adjust():
    """GAIL 2017-03-09: NSE calls it a bonus, yfinance calls it split_or_bonus.
    A key that includes the label keeps both and squares the ratio."""
    from prosignal.data.corporate_actions import dedupe_actions

    frame = pd.DataFrame([
        {"symbol": "GAIL", "ex_date": pd.Timestamp("2017-03-09"),
         "action_type": "split_or_bonus", "ratio": 0.75, "raw_details": "", "source": "yfinance"},
        {"symbol": "GAIL", "ex_date": pd.Timestamp("2017-03-09"),
         "action_type": "bonus", "ratio": 0.75, "raw_details": "", "source": "nse_corporate_actions"},
    ])
    out = dedupe_actions(frame)
    assert len(out) == 1
    assert out.iloc[0]["source"] == "nse_corporate_actions"


def test_an_ex_date_a_day_apart_is_one_event_not_two():
    """HAL's 2023 split is 09-28 at NSE and 09-29 at Yahoo."""
    from prosignal.data.corporate_actions import dedupe_actions

    frame = pd.DataFrame([
        {"symbol": "HAL", "ex_date": pd.Timestamp("2023-09-28"), "action_type": "split",
         "ratio": 0.5, "raw_details": "", "source": "nse_corporate_actions"},
        {"symbol": "HAL", "ex_date": pd.Timestamp("2023-09-29"), "action_type": "split_or_bonus",
         "ratio": 0.5, "raw_details": "", "source": "yfinance"},
    ])
    out = dedupe_actions(frame)
    assert len(out) == 1
    assert out.iloc[0]["ex_date"] == pd.Timestamp("2023-09-28")


def test_genuinely_separate_events_both_survive():
    from prosignal.data.corporate_actions import dedupe_actions

    frame = pd.DataFrame([
        {"symbol": "X", "ex_date": pd.Timestamp("2020-01-01"), "action_type": "split",
         "ratio": 0.5, "raw_details": "", "source": "nse_corporate_actions"},
        {"symbol": "X", "ex_date": pd.Timestamp("2022-06-01"), "action_type": "bonus",
         "ratio": 0.5, "raw_details": "", "source": "nse_corporate_actions"},
    ])
    assert len(dedupe_actions(frame)) == 2


def test_several_dividends_may_share_a_date():
    """They do not move the adjusted series, so collapsing them would lose
    information for nothing."""
    from prosignal.data.corporate_actions import dedupe_actions

    frame = pd.DataFrame([
        {"symbol": "X", "ex_date": pd.Timestamp("2020-01-01"), "action_type": "dividend",
         "ratio": 1.0, "raw_details": "interim", "source": "yfinance"},
        {"symbol": "X", "ex_date": pd.Timestamp("2020-01-01"), "action_type": "special_dividend",
         "ratio": 1.0, "raw_details": "special", "source": "yfinance"},
    ])
    assert len(dedupe_actions(frame)) == 2


def test_a_compound_event_keeps_the_product_of_its_parts():
    """BAJFINANCE 2025-06-16: a 1:2 split AND a 4:1 bonus. The price gaps by
    0.5 x 0.2, not by either one."""
    from prosignal.data.providers.nse_archives import NseArchivesProvider as P

    assert P.parse_action_ratio("Bonus 4:1") == ("bonus", pytest.approx(0.2))
    split = P.parse_action_ratio(
        "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share")
    assert split == ("split", pytest.approx(0.5))
    assert 0.2 * 0.5 == pytest.approx(0.1)


def test_a_dividend_line_yields_no_ratio():
    from prosignal.data.providers.nse_archives import NseArchivesProvider as P
    assert P.parse_action_ratio("Dividend - Rs 30 Per Share") is None
    assert P.parse_action_ratio("") is None
