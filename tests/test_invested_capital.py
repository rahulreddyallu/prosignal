"""How much of the book is actually in equities when every slot is full.

THE ASSERTION WHOSE ABSENCE COST THE STRATEGY ITS RETURN. 1,835 tests pass on
this tree. Not one of them multiplies the two shipped numbers together:

    position value = capital x risk_per_trade_pct x category_fraction
                     / stop_distance                       (stage7_risk:290)

`risk_per_trade_pct` is 1.0 and the stop is `8 x ATR` capped at 35%. An 8 ATR
stop on an NSE mid-cap lands at 20-35% of entry, so the risk budget always binds
ahead of the 16.7% capital slot and each position is 2.9-5.0% of capital. Six
slots is 17-30% invested.

`config/parameters.yaml` states the design intent in its own note: "With a 1%
risk budget and a 5% stop, the position is 20% of capital." The shipped stop is
not 5%. Nobody multiplied.

The cost of that is not subtle. `research/v3/experiments/book_sim.json` measures
the live book at beta 0.15 over the full window against an equal-weight eligible
universe that compounded at roughly 21% a year, with alpha of +0.09% per period
-- indistinguishable from zero. The book does not lose to the benchmark because
it picks badly. It loses because it is three-quarters cash.

Every check here is per-name arithmetic over the REAL config, so it needs no
store, no panel and no network. Liquidity is deliberately made abundant: this
file is about the risk-budget and capital constraints, and a binding liquidity
cap would mask them.

See docs/REBUILD_2026_09.md section 3.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.config.loader import load_config
from prosignal.costs import CostModel
from prosignal.stages.stage7_risk import build_plan

#: Daily volatilities spanning what the eligible universe actually holds: a
#: large-cap at 1.2% a day through a smallcap at 3.5%. One number would test one
#: stop distance and the whole point is that the stop distance drives the size.
DAILY_VOLS = (0.012, 0.018, 0.022, 0.026, 0.030, 0.035)

#: Rs 50 crore a day. At the shipped 1% participation cap that permits a Rs 50
#: lakh position -- five times the entire book -- so liquidity cannot bind and
#: the measurement is of the risk budget alone.
ABUNDANT_ADTV = 5e8

#: What a long-only equity book is for. A book that is meant to be invested and
#: is not has made a market-timing bet nobody authorised, and at these levels the
#: bet dominates every stock-selection decision above it.
INVESTED_BAND = (0.85, 1.05)


def _frame(vol: float, seed: int, sessions: int = 300) -> pd.DataFrame:
    """A price history with a known daily volatility.

    High and low are set from the same vol rather than a flat 1% band, because
    ATR reads the range and a fixed band would give every name the same stop
    however volatile its closes were -- which is precisely the confound.
    """
    idx = pd.bdate_range("2024-01-01", periods=sessions)
    rng = np.random.default_rng(seed)
    close = pd.Series(
        1000.0 * np.exp(np.cumsum(rng.normal(0.0004, vol, sessions))), index=idx
    )
    return pd.DataFrame({
        "close": close,
        "high": close * (1.0 + vol),
        "low": close * (1.0 - vol),
        "open": close,
        "turnover": pd.Series(ABUNDANT_ADTV, index=idx),
    })


def _book(cfg, vols=DAILY_VOLS):
    """One risk plan per slot, at the configured book size."""
    costs = CostModel(cfg)
    slots = int(cfg.params.capital.max_open_positions.value)
    plans = []
    for i in range(slots):
        frame = _frame(vols[i % len(vols)], seed=100 + i)
        price = float(frame["close"].iloc[-1])
        plans.append(build_plan(f"NAME{i}", frame, price, 0.85,
                                ABUNDANT_ADTV, cfg, costs))
    return plans


def _invested_fraction(cfg, plans) -> float:
    capital = float(cfg.params.capital.total_capital_inr.value)
    return sum(p.position_value_inr or 0.0 for p in plans) / capital


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# =============================================================================
# What the shipped configuration does today
# =============================================================================

def test_the_historical_defect_is_recorded_as_arithmetic(cfg):
    """WHAT THE SHIPPED SIZER DID, kept as a computation rather than a live check.

    Three tests here used to drive `stage7_risk.build_plan` and assert that six
    slots invested under 40% of capital. They passed on the tree that shipped
    them and they are gone, because both of their inputs moved on 2026-09-07:
    sizing left stage 7 for `pipeline._size_the_book`, and the book went from 6
    names to 50. A test that re-derives a historical defect from CURRENT config
    stops measuring the defect the moment the config is fixed -- and then either
    fails for the right reason (noise) or, worse, passes for a new one.

    So the arithmetic is frozen here with the constants it actually ran under.
    It is a record, and it is checkable.

    A NOTE WORTH KEEPING. At 50 slots the capital slot is 2% and the risk-based
    size is 2.9-5.0%, so the SLOT binds and stage 7's own sizer would now leave
    only ~3% in cash. The breadth change alone recovers most of the exposure;
    `size_book` makes it equal-weighted, explicit, and independent of which
    constraint happens to be tighter this month. Both were needed, and neither
    is redundant.
    """
    capital = 1_000_000.0
    risk_pct = 1.0            # capital.risk_per_trade_pct, as shipped
    slots = 6                 # capital.max_open_positions, as shipped
    slot_value = capital / slots

    # An 8xATR stop capped at 35% lands between 20% and 35% on this universe;
    # 21.6%-35.0% was the measured span across 1.2%-3.5% daily volatility.
    for stop_pct, expected_share in ((0.35, 0.0286), (0.216, 0.0463)):
        by_risk = capital * (risk_pct / 100.0) / stop_pct
        assert by_risk < slot_value, (
            "the risk budget must bind ahead of the capital slot; that is the "
            "whole mechanism"
        )
        assert abs(by_risk / capital - expected_share) < 0.001

    # Six slots at the cap: the measured figure was 18.9% deployed.
    invested_at_cap = slots * (capital * (risk_pct / 100.0) / 0.35) / capital
    assert 0.16 < invested_at_cap < 0.20

    # And what that cost, against the regenerated benchmark of +22.9% a year.
    forgone = (1.0 - 0.189) * 0.229
    assert 0.18 < forgone < 0.19
    assert forgone / 0.197 > 0.9, (
        "the cash drag should account for ~94% of the -19.7% measured net "
        "excess; if this ratio moves, the diagnosis in REBUILD_2026_09.md 3.3 "
        "needs revisiting rather than the test"
    )


def test_a_full_book_is_actually_invested(cfg):
    """THE ACCEPTANCE GATE, and it now measures the book rather than the sizer.

    It was a strict xfail against `stage7_risk` while sizing lived there. Sizing
    moved to `pipeline._size_the_book` on 2026-09-07, because 1/N is a property
    of the SET and `build_plan` is called once per symbol -- so a per-name test
    can no longer answer this question at all, whatever it asserts.

    The band is deliberately not 100%: whole-share rounding, a liquidity cap
    binding on a genuinely thin name, and a slot left empty on a day the
    ranking is short of eligible names all pull below 1.0, and none of those is
    a market-timing decision. Anything under 0.85 is.
    """
    from prosignal.book import BookSpec, size_book

    slots = int(cfg.params.capital.max_open_positions.value)
    names = [f"N{i:03d}" for i in range(slots)]
    spec = BookSpec(
        capital=float(cfg.params.capital.total_capital_inr.value),
        max_participation_of_adtv=float(
            cfg.params.capital.max_participation_of_adtv.value),
        min_names=1, max_names=slots,
    )
    b = size_book(names, {t: 250.0 for t in names},
                  {t: ABUNDANT_ADTV for t in names}, spec)

    lo, hi = INVESTED_BAND
    assert lo <= b.invested_fraction <= hi, (
        f"a full book invests {b.invested_fraction:.1%} of capital, outside the "
        f"{lo:.0%}-{hi:.0%} band. Below the band the engine is making an "
        f"unauthorised market-timing bet that dominates every stock-selection "
        f"decision above it; above it, it is levered."
    )
    assert all(p.binding == "equal weight" for p in b.positions)


def test_the_book_is_sized_as_a_set_not_per_name(cfg):
    """The structural claim, pinned so it cannot regress quietly.

    `stage7_risk.build_plan` takes one ticker. `pipeline._size_the_book` takes
    the selected list. If sizing ever moves back behind a per-name signature the
    total exposure becomes an accident of which names qualified that day, which
    is exactly how the book came to hold 18.9% of capital.
    """
    import inspect

    from prosignal.book import size_book
    from prosignal.pipeline import _size_the_book

    assert "chosen" in inspect.signature(size_book).parameters
    assert "buys" in inspect.signature(_size_the_book).parameters
