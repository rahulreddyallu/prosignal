"""Nothing may be visible before it was public — and "public" has a clock.

THE LEAK THIS CLOSES, measured 2026-09-04. NSE stamps filings with a time.
RELIANCE's Q3 FY25 carries `filingDate: "16-Jan-2025 20:20"` -- nearly five
hours after the 15:30 close. `nse_fundamentals._parse_dt` did
`str(value).split(" ")[0]`, kept the date, discarded the time, and said so in
its own docstring: *"Date is what we gate on."*

Stored as a midnight date, that filing became visible to the as-of join on
16 January -- the session whose decision is taken at 15:30, before the filing
existed.

Across 1,204 filings on ten symbols, **59.1% are stamped after the close**, the
modal filing hour being 17:00-19:00. So this was the majority of the feed, it
ran in the flattering direction, and no backtest could see it.

These tests hold the rule and the two properties that make it safe: an unknown
hour is treated as after the close, and no feature may read a record whose
availability date is later than the decision date.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.features.pit_fundamentals import (
    MARKET_CLOSE, asof_panel, availability_date,
)


# --------------------------------------------------------------- the rule
@pytest.mark.parametrize("stamp,expected,why", [
    ("2025-01-16 20:20", "2025-01-17", "after the close is not actionable today"),
    ("2025-01-16 14:07", "2025-01-16", "before the close is actionable today"),
    ("2025-01-16 15:30", "2025-01-16", "exactly at the close still counts"),
    ("2025-01-16 15:31", "2025-01-17", "one minute after does not"),
    ("2025-01-16 09:15", "2025-01-16", "at the open"),
    ("2025-01-16 23:59", "2025-01-17", "late night"),
    # 00:30 on the 16th is BEFORE the 16th's 15:30 close, so the 16th's
    # decision can act on it. The first draft of this case asserted the 17th,
    # reasoning that a small-hours filing follows the PREVIOUS session -- which
    # is true and irrelevant: the question is only whether the filing exists by
    # the close of the session being decided, and it does.
    ("2025-01-16 00:30", "2025-01-16", "small hours, still before today's close"),
])
def test_the_availability_rule(stamp, expected, why):
    got = availability_date([stamp]).iloc[0]
    assert str(got.date()) == expected, why


def test_the_close_is_the_nse_session_close():
    assert MARKET_CLOSE == dt.time(15, 30)


def test_an_unknown_hour_is_treated_as_after_the_close():
    """The conservative direction, and the engine already reasons this way.

    Rows already in the store carry a date with no time and there is no way to
    recover which side of 15:30 they fell on. Delaying costs a day of
    staleness; admitting costs a lookahead on three fifths of them. Unknown
    liquidity is already priced at the worst case the model allows; an unknown
    hour gets the same treatment.
    """
    assert str(availability_date(["2025-01-16"]).iloc[0].date()) == "2025-01-17"
    assert str(availability_date([pd.Timestamp("2025-01-16")]).iloc[0].date()) \
        == "2025-01-17"


def test_a_missing_timestamp_stays_missing():
    """NaT in, NaT out. A row with no filing date cannot be used PIT at all."""
    out = availability_date([None, pd.NaT, "not a date"])
    assert out.isna().all()


def test_availability_never_moves_earlier_than_the_filing_date():
    """The direction is the whole point: this may only ever DELAY visibility."""
    stamps = pd.to_datetime([
        "2025-01-16 20:20", "2025-01-16 14:07", "2025-03-31 15:30",
        "2024-12-31 23:59", "2024-06-30 09:00",
    ])
    av = availability_date(stamps)
    assert (av >= stamps.normalize()).all(), (
        "availability moved EARLIER than the filing date for some row. This "
        "function may only ever delay visibility; moving it earlier would "
        "manufacture the lookahead it exists to remove.")


# ------------------------------------------------------------- the leakage
def test_no_field_is_visible_before_its_availability_date():
    """G1's leakage test, on the real as-of join.

    Builds a record whose availability lands on a known session and asserts the
    panel is empty strictly before it and populated from it onward.
    """
    sessions = pd.DatetimeIndex(pd.bdate_range("2025-01-06", periods=15))
    recs = pd.DataFrame({
        "symbol": ["AAA"],
        # 20:20 on the 10th -> not actionable until the 13th (the 11th and 12th
        # are a weekend), which is what makes this a real test of the rule
        # rather than of the calendar.
        "effective_date": availability_date(["2025-01-10 20:20"]).to_numpy(),
        "period_end": pd.to_datetime(["2024-12-31"]),
        "src": ["filed"],
        "ttm_revenue": [100.0],
    })
    panel = asof_panel(recs, sessions, ["AAA"])["ttm_revenue"]["AAA"]
    visible = panel.notna()
    first = sessions[visible.to_numpy().argmax()]
    assert str(first.date()) == "2025-01-13", (
        f"the value first became visible on {first.date()}; a filing stamped "
        f"20:20 on Friday the 10th cannot be acted on before Monday the 13th.")
    assert not visible.iloc[:sessions.get_loc(first)].any(), (
        "the value was visible on a session before its availability date")


def test_a_before_close_filing_is_visible_the_same_session():
    """The rule must not over-correct. A 14:07 filing IS actionable that day."""
    sessions = pd.DatetimeIndex(pd.bdate_range("2025-01-06", periods=10))
    recs = pd.DataFrame({
        "symbol": ["AAA"],
        "effective_date": availability_date(["2025-01-09 14:07"]).to_numpy(),
        "period_end": pd.to_datetime(["2024-12-31"]),
        "src": ["filed"], "ttm_revenue": [100.0],
    })
    panel = asof_panel(recs, sessions, ["AAA"])["ttm_revenue"]["AAA"]
    first = sessions[panel.notna().to_numpy().argmax()]
    assert str(first.date()) == "2025-01-09"


def test_the_shipped_store_has_no_record_visible_before_it_was_filed():
    """The same assertion, against the real curated store.

    This is the one that would catch a regression in the provider: if
    `filing_ts` stops being carried, or `build_records` goes back to using
    `filing_date` directly, availability collapses onto the filing date and
    three fifths of the feed leaks a session again.
    """
    from prosignal.config.loader import load_config
    from prosignal.data.store import DataStore
    from prosignal.features.pit_fundamentals import build_records

    cfg = load_config()
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    recs = build_records(store=store)
    if recs is None or recs.empty:
        pytest.skip("no fundamentals in the store")

    filed = recs[recs["src"] == "filed"]
    if filed.empty:
        pytest.skip("no filed-source records in the store")

    raw = pd.read_parquet(cfg.paths.curated / "fundamentals.parquet")
    ts = raw["filing_ts"] if "filing_ts" in raw.columns else raw["filing_date"]
    want = availability_date(ts).dropna()
    eff = pd.to_datetime(filed["effective_date"]).dropna()

    assert eff.min() >= want.min(), (
        f"the earliest effective_date in the built records ({eff.min()}) is "
        f"before the earliest availability date the raw feed allows "
        f"({want.min()}). Something is gating on the filing date rather than "
        f"on when the filing could be acted upon.")

    # And the specific regression, checked PER ROW. The first draft compared
    # the two as SETS, which fails for a reason that has nothing to do with
    # leakage: row A filed on the 23rd becomes available on the 24th, and some
    # unrelated row B was filed on the 24th, so the sets intersect while every
    # row is individually correct. Only the row-wise comparison means anything.
    if "filing_ts" not in raw.columns:
        pairs = pd.DataFrame({
            "filed": pd.to_datetime(raw["filing_date"]),
            "available": availability_date(raw["filing_date"]),
        }).dropna()
        same = pairs["available"] == pairs["filed"]
        assert not same.any(), (
            f"{int(same.sum())} of {len(pairs)} rows are available on the very "
            f"date they were filed, with no time recorded. An unknown hour "
            f"must be treated as after the close, so availability belongs on "
            f"the FOLLOWING session.")
