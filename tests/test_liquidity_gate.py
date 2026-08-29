"""R13 -- liquidity as a gate, and the execution model as a monotone function.

THE DEFECT. A name whose ADTV could not be measured received `qty_liq = slot /
entry` from the sizer -- the largest position the capital slot allows -- and the
half-spread alone from `costs.impact_bps` -- the cheapest fill in the model. An
absence of information bought the biggest size and the best execution
assumption in the engine.

The individual branches were each defensible in isolation, which is how they
survived: the sizer's fallback read as "no liquidity constraint applies" and the
cost model's read as "we cannot compute participation". Together they are an
engine manufacturing liquidity it does not know exists.

So most of this file is PROPERTY tests rather than example tests. The question
is not "does this case return the right number" but "is the execution model
monotone in the things it must be monotone in" -- because a single missing
branch can invert a relationship that every individual example still satisfies.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prosignal.config.loader import load_config
from prosignal.costs import CostModel
from prosignal.liquidity import (STALE_DISCOUNT, LiquidityState, assess)


@pytest.fixture(scope="module")
def cost_model():
    return CostModel(load_config())


# =============================================================================
# The four states
# =============================================================================


def test_missing_is_not_zero_and_neither_is_tradable():
    """They mean different things and both refuse, for different reasons.

    MISSING is "nobody measured it". INVALID/zero is "it was measured and the
    name did not trade". Collapsing them is what let `if adtv` -- which is
    false for None AND for 0.0 -- stand in for a liquidity policy.
    """
    missing = assess(None)
    zero = assess(0.0)
    assert missing.state is LiquidityState.MISSING
    assert zero.state is LiquidityState.INVALID
    assert missing.state is not zero.state, (
        "an unmeasured name and a name that did not trade are different "
        "findings and license different follow-up"
    )
    for v in (missing, zero):
        assert not v.tradable and not v.confident
        assert v.adtv_inr is None, (
            "an untradable reading must not carry a usable number; a caller "
            "that ignores `tradable` should hit a TypeError, not size a "
            "position against a plausible-looking float"
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, "x"])
def test_every_impossible_reading_is_invalid(bad):
    v = assess(bad)
    assert v.state is LiquidityState.INVALID
    assert not v.tradable and v.adtv_inr is None


def test_a_good_reading_passes_through_undiscounted():
    v = assess(5e7)
    assert v.state is LiquidityState.KNOWN_VALID
    assert v.tradable and v.confident
    assert v.adtv_inr == pytest.approx(5e7)


def test_a_stale_reading_is_evidence_at_a_discount():
    fresh = assess(5e7, age_sessions=1)
    stale = assess(5e7, age_sessions=40)
    assert fresh.state is LiquidityState.KNOWN_VALID
    assert stale.state is LiquidityState.KNOWN_STALE
    assert stale.tradable and not stale.confident
    assert stale.adtv_inr == pytest.approx(5e7 * STALE_DISCOUNT)
    assert stale.adtv_inr < fresh.adtv_inr, (
        "age can only reduce the capacity a reading licenses"
    )
    assert "40" in stale.reason


def test_staleness_can_be_refused_outright_as_a_policy():
    v = assess(5e7, age_sessions=40, allow_stale=False)
    assert not v.tradable
    assert v.raw_adtv_inr == pytest.approx(5e7), (
        "the reading is still reported even when it is refused; a refusal that "
        "discards the evidence cannot be reviewed"
    )


# =============================================================================
# Monotonicity of the execution model
# =============================================================================


LIQUIDITIES = [1e6, 5e6, 2e7, 5e7, 1e8, 5e8, 2e9]
SIZES = [1e4, 5e4, 1.2e5, 5e5, 2e6]


def test_more_liquidity_never_costs_more(cost_model):
    """Impact must be non-increasing in ADTV, at every order size."""
    for size in SIZES:
        costs = [cost_model.impact_bps(size, adv) for adv in LIQUIDITIES]
        for a, b, adv_a, adv_b in zip(costs, costs[1:], LIQUIDITIES, LIQUIDITIES[1:]):
            assert b <= a + 1e-9, (
                f"a Rs {size:,.0f} order priced {b:.1f} bps against Rs "
                f"{adv_b:,.0f} of ADTV and {a:.1f} bps against Rs {adv_a:,.0f}; "
                f"deeper liquidity got more expensive"
            )


def test_a_bigger_order_never_gets_cheaper(cost_model):
    """Impact must be non-decreasing in order size, at every liquidity."""
    for adv in LIQUIDITIES:
        costs = [cost_model.impact_bps(size, adv) for size in SIZES]
        for a, b in zip(costs, costs[1:]):
            assert b >= a - 1e-9, (
                f"against Rs {adv:,.0f} of ADTV a larger order priced "
                f"{b:.1f} bps against {a:.1f} bps for a smaller one"
            )


def test_unknown_liquidity_is_never_the_cheapest_answer(cost_model):
    """THE DEFECT, as a property.

    `impact_bps` returned the half-spread alone when ADTV was missing, which is
    the floor of the whole model -- cheaper than any measurable name at any
    size. Unknown liquidity must price at or above a name traded at the
    participation cap, because that is the most the engine would knowingly do.
    """
    half_spread = float(load_config().params.costs.impact_model
                        .assumed_half_spread_bps.value)
    for size in SIZES:
        unknown = cost_model.impact_bps(size, None)
        assert unknown > half_spread, (
            f"unknown liquidity priced at {unknown:.1f} bps, the model's own "
            f"floor of {half_spread:.1f}; an absence of information cannot buy "
            f"the best fill in the model"
        )
        at_cap = min(cost_model.impact_bps(size, adv) for adv in LIQUIDITIES
                     if size / adv <= 0.01)
        assert unknown >= at_cap - 1e-9, (
            f"unknown liquidity ({unknown:.1f} bps) priced below a MEASURED "
            f"name traded at the participation cap ({at_cap:.1f} bps)"
        )


def test_a_wider_spread_never_improves_execution(cost_model):
    """Spread enters additively; the test exists because it did not have to."""
    import copy

    base = cost_model.impact_bps(1.2e5, 5e7)
    wide = copy.deepcopy(cost_model)
    wide.c.impact_model.assumed_half_spread_bps.value *= 4.0
    assert wide.impact_bps(1.2e5, 5e7) > base


def test_a_round_trip_is_never_free(cost_model):
    """Statutory charges alone put a floor under every executable trade."""
    for adv in LIQUIDITIES + [None]:
        cb = cost_model.round_trip(300.0, 400, adtv_inr=adv)
        assert cb.total_bps_of_buy > 0
        assert cb.stressed_bps_of_buy >= cb.total_bps_of_buy


# =============================================================================
# The gate reaches the sizers
# =============================================================================


def _plan(adtv):
    from prosignal.stages.stage7_risk import build_plan

    cfg = load_config()
    n = 300
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(2)
    close = pd.Series(1000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n))),
                      index=idx)
    frame = pd.DataFrame({"close": close, "high": close * 1.01,
                          "low": close * 0.99, "open": close,
                          "turnover": pd.Series(5e8, index=idx)})
    return build_plan("TEST", frame, float(close.iloc[-1]), 0.8, adtv,
                      cfg, CostModel(cfg))


def test_stage_seven_refuses_to_size_an_unmeasured_name():
    """It used to substitute the capital slot for the liquidity cap."""
    known = _plan(5e8)
    unknown = _plan(None)
    assert known.risk_category_inputs["qty_by_liquidity"] > 0
    assert known.liquidity_state == LiquidityState.KNOWN_VALID.value

    assert unknown.liquidity_state == LiquidityState.MISSING.value
    assert unknown.risk_category_inputs["qty_by_liquidity"] == 0.0, (
        "an unmeasured name was sized against the capital slot again"
    )
    assert any("NOT SIZED" in note for note in unknown.notes), (
        "the refusal has to be stated on the card; a size of zero with no "
        "reason reads as a bug rather than as a policy"
    )


def test_a_name_that_did_not_trade_is_also_refused():
    zero = _plan(0.0)
    assert zero.liquidity_state == LiquidityState.INVALID.value
    assert zero.risk_category_inputs["qty_by_liquidity"] == 0.0


def test_the_simulator_refuses_the_same_names_the_live_sizer_does():
    """Two sizers, one policy. They disagreed, and both were optimistic."""
    from prosignal.validation.portfolio_sim import PortfolioParams, _position

    n, syms = 300, ["A", "B"]
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame(1000.0, index=idx, columns=syms)
    atr = pd.DataFrame(20.0, index=idx, columns=syms)
    adtv = pd.DataFrame({"A": 5e8, "B": np.nan}, index=idx)
    p = PortfolioParams(
        capital=1e6, max_positions=8, risk_per_trade_pct=1.0,
        max_participation_of_adtv=0.01, stop_atr_multiple=2.5,
        min_stop_distance_pct=2.0, max_stop_distance_pct=15.0,
        invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
        horizon_sessions=63, entry_rank=8, exit_rank=16)

    assert _position("A", 200, close, atr, adtv, p) is not None
    assert _position("B", 200, close, atr, adtv, p) is None, (
        "the simulator sized a name whose ADTV is NaN; the live sizer refuses "
        "it, so the backtest is measuring a book the engine would not hold"
    )

    # And with the gate off -- which exists only so the old behaviour stays
    # measurable -- it returns to the full slot, as it used to.
    from dataclasses import replace

    old = _position("B", 200, close, atr, adtv,
                    replace(p, refuse_unknown_liquidity=False))
    assert old is not None and old[0] == pytest.approx(p.slot), (
        "the switch that reproduces the old behaviour no longer reproduces it, "
        "so the price of the correction can no longer be recomputed"
    )
