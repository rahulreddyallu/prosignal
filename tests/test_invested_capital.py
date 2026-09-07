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

def test_every_slot_is_sized_by_the_risk_budget_not_the_capital_slot(cfg):
    """The binding constraint, named on every card, on every name.

    If this ever reports "capital slot" the arithmetic below stops applying and
    the aggregate assertions need re-deriving rather than re-tuning.
    """
    for plan in _book(cfg):
        binding = plan.risk_category_inputs
        assert binding["qty_by_risk"] <= binding["qty_by_slot"], (
            f"{plan.ticker}: the capital slot bound before the risk budget "
            f"({binding['qty_by_risk']:.0f} vs {binding['qty_by_slot']:.0f}). "
            f"The cash-drag arithmetic in this file assumes the opposite."
        )
        assert binding["qty_by_risk"] <= binding["qty_by_liquidity"], (
            f"{plan.ticker}: liquidity bound, so this measures the wrong thing"
        )


def test_an_eight_atr_stop_is_a_fifth_to_a_third_of_the_entry_price(cfg):
    """The input to the sizing arithmetic, measured rather than assumed."""
    distances = [p.stop_distance_pct for p in _book(cfg)]
    assert min(distances) > 12.0, (
        f"the tightest stop was {min(distances):.1f}%; at that distance the "
        f"capital slot binds instead and this file's premise is wrong"
    )
    assert max(distances) <= 35.0 + 1e-9, (
        f"the widest stop was {max(distances):.1f}%, above the configured "
        f"max_stop_distance_pct cap"
    )


def test_no_single_position_exceeds_a_twentieth_of_capital(cfg):
    """2.9-5.0% per name, against a 16.7% slot that never binds."""
    capital = float(cfg.params.capital.total_capital_inr.value)
    for plan in _book(cfg):
        share = (plan.position_value_inr or 0.0) / capital
        assert share < 0.06, (
            f"{plan.ticker} took {share:.1%} of capital; the risk budget "
            f"should hold it near 3-5%"
        )


def test_the_shipped_book_is_mostly_cash(cfg):
    """THE DEFECT, pinned so it cannot be fixed by accident and unnoticed.

    This test PASSES on the tree that shipped it. It is not an aspiration; it is
    a measurement, kept because a number this consequential should not live only
    in a document. When the book layer is replaced under Phase 5 of
    docs/REBUILD_2026_09.md this test SHOULD start failing, and the correct
    response is to delete it and unmark the acceptance gate below -- not to
    widen the bound.
    """
    invested = _invested_fraction(cfg, _book(cfg))
    # Measured 2026-09-07 on the shipped config: 18.9% invested, 81.1% idle.
    # Forgone benchmark return on that idle cash at the universe's 21% a year is
    # -17.0%, against a measured net excess of -17.6% in book_sim.json. The cash
    # drag is 97% of the underperformance.
    assert invested < 0.40, (
        f"the shipped book now invests {invested:.1%} of capital. If this is a "
        f"deliberate fix, delete this test and remove the xfail marker from "
        f"test_a_full_book_is_actually_invested."
    )
    assert 0.10 < invested, (
        f"the book invests {invested:.1%}, below anything the arithmetic in "
        f"this file's docstring predicts; something else is binding"
    )


def test_the_cash_drag_costs_more_than_every_cost_model_assumption(cfg):
    """Perspective, and the reason this is the first thing to fix.

    Measured round-trip cost is 80 bps at the shipped impact coefficient and the
    live book's realised drag is 0.35% a year. The uninvested fraction gives up
    the benchmark's return on three-quarters of the book. The two are not the
    same order of magnitude and the repository spent a generation tuning the
    smaller one.
    """
    idle = 1.0 - _invested_fraction(cfg, _book(cfg))
    benchmark_annual = 0.21          # equal-weight eligible universe, full panel
    forgone = idle * benchmark_annual
    measured_cost_drag = 0.0035      # book_sim.json, live_6@0.1, full window
    assert forgone > 20 * measured_cost_drag, (
        f"forgone benchmark return on the idle {idle:.0%} is {forgone:.1%} a "
        f"year against {measured_cost_drag:.2%} of measured cost drag"
    )


# =============================================================================
# The acceptance gate for the rebuilt book
# =============================================================================

@pytest.mark.xfail(
    strict=True,
    reason="Phase 5 of docs/REBUILD_2026_09.md has not landed. The book is "
           "sized by risk budget against an 8xATR stop and holds 17-30% of "
           "capital. Remove this marker when equal-weight full-investment "
           "ships; a strict xfail fails the suite the moment it starts "
           "passing, which is exactly when this marker should be deleted.",
)
def test_a_full_book_is_actually_invested(cfg):
    """A long-only book with every slot filled holds equities, not cash.

    The band is deliberately not 100%: rounding to whole shares, a liquidity cap
    binding on a genuinely thin name, and an unfilled slot on a day the ranking
    is short of eligible names all pull below 1.0, and none of those is a
    market-timing decision. Anything under 0.85 is.
    """
    invested = _invested_fraction(cfg, _book(cfg))
    lo, hi = INVESTED_BAND
    assert lo <= invested <= hi, (
        f"a full book invests {invested:.1%} of capital, outside the "
        f"{lo:.0%}-{hi:.0%} band. Below the band the engine is making an "
        f"unauthorised market-timing bet that dominates every stock-selection "
        f"decision above it; above it, it is levered."
    )
