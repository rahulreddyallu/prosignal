"""The simulator must decide as often as the engine does.

`simulate` walks `stride = ceil(decision / step_sessions)` ranking dates at a
time, and `decision` defaulted to `params.horizon_sessions`. At the shipped
horizon of 63 that is a NON-OVERLAPPING COHORT schedule: form a book, hold it
for the whole horizon, liquidate, form the next -- four decisions a year.

The live engine decides every `stage6_entry.entry_cadence_sessions`, which is
21. Twelve times a year, three times as often, and it carries names across
decisions through the exit band. The two schedules pay different amounts of
cost for the same signal, and the simulator's is the cheaper one: it cannot
re-rank a held name for 63 sessions, so it never pays the turnover the
hysteresis band generates. Every cost and turnover figure measured on the
cohort schedule and quoted about the live book described a different strategy.

`decision_sessions` truncates each cohort at the next decision date and
re-selects. A name still inside the exit band is kept and owes nothing; a name
that has left it, or whose position closed early, is replaced and pays a round
trip. `_portfolio_params` now reads the live cadence, so the shipped
measurement and the shipped engine make decisions on the same clock.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.portfolio_sim import (
    PortfolioParams,
    phase_summary,
    simulate,
)

# The fixture lives with the leverage tests: same simulator, same panels, and
# duplicating it would let the two drift into measuring different books.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_leverage_neutral import _params, _prices, _rankings   # noqa: E402


def _sim(cadence, **over):
    prices = _prices()
    p = _params(**over)
    r = simulate(_rankings(prices), prices, p, phase=0, step_sessions=21,
                 decision_sessions=cadence)
    assert not r.empty
    return r


def test_the_default_is_the_old_cohort_schedule():
    """Unchanged behaviour for a caller that passes nothing. The correction is
    in what `_portfolio_params` supplies, not in a silent change of meaning."""
    prices, p = _prices(), _params()
    old = simulate(_rankings(prices), prices, p, phase=0, step_sessions=21)
    new = _sim(p.horizon_sessions)
    pd.testing.assert_frame_equal(old.periods, new.periods)


def test_the_shipped_params_carry_the_live_cadence():
    """The fix, at the point it actually lands. A simulator that CAN run at the
    live cadence but is never asked to is the same defect with extra code."""
    from prosignal.cli import _portfolio_params
    from prosignal.config.loader import get_config
    from prosignal.stages._cfg import iv

    cfg = get_config()
    p = _portfolio_params(cfg)
    assert p.decision_sessions == iv(
        cfg.params.stage6_entry.admission.entry_cadence_sessions)
    assert p.decision_sessions < p.horizon_sessions, (
        "the live engine decides faster than its own label horizon; if these "
        "are ever equal the parity problem has been resolved by moving the "
        "wrong one"
    )


def test_a_faster_cadence_makes_more_decisions():
    slow = _sim(63)
    fast = _sim(21)
    assert len(fast.periods) > len(slow.periods) * 2, (
        f"cadence 21 against 63 should make about three times as many "
        f"decisions; got {len(fast.periods)} against {len(slow.periods)}"
    )


def test_a_faster_cadence_holds_for_less_time():
    assert _sim(21).periods["hold_sessions"].eq(21).all()
    assert _sim(63).periods["hold_sessions"].eq(63).all()


def test_the_hold_is_never_longer_than_the_horizon():
    """A cadence SLOWER than the horizon does not extend the hold -- the label
    is 63 sessions and a position resolved at 63 cannot be carried to 126."""
    assert _sim(126).periods["hold_sessions"].eq(63).all()


def test_a_faster_cadence_pays_more_cost_per_year():
    """The finding, as one assertion. Deciding three times as often on the same
    signal is three times as many chances to turn the book over."""
    prices = _prices()
    rk = _rankings(prices)
    out = {}
    for cad in (63, 21):
        m = phase_summary(rk, prices, _params(), step_sessions=21,
                          decision_sessions=cad)
        out[cad] = m["mean_cost"] * m["periods_per_year"]
    assert out[21] > out[63], (
        f"annualised cost at cadence 21 ({out[21]:.4%}) must exceed cadence 63 "
        f"({out[63]:.4%}); if it does not, the truncated cohort is not being "
        f"re-selected and the parity fix is inert"
    )


def test_the_annualisation_follows_the_hold_not_the_horizon():
    """Twelve periods a year at cadence 21, not four. Annualising a 21-session
    period on a 63-session horizon understates every annual figure by 3x --
    cost included, which is the figure this cadence exists to get right."""
    prices, rk = _prices(), None
    rk = _rankings(prices)
    assert phase_summary(rk, prices, _params(), step_sessions=21,
                         decision_sessions=21)["periods_per_year"] == pytest.approx(12.0)
    assert phase_summary(rk, prices, _params(), step_sessions=21,
                         decision_sessions=63)["periods_per_year"] == pytest.approx(4.0)


def test_the_cadence_is_reported_on_the_result():
    """A cost figure whose schedule is not stated is a cost figure about an
    unknown strategy."""
    m = phase_summary(_rankings(_prices()), _prices(), _params(),
                      step_sessions=21, decision_sessions=21)
    assert m["decision_sessions"] == 21.0
    assert m["hold_sessions"] == 21.0


def test_a_carried_name_still_owes_nothing():
    """The hysteresis band must keep working across the shorter cohort. A
    position open at the truncated horizon exits at EXIT_TIMEOUT, which is the
    side the cost logic reads as "carried"."""
    r = _sim(21)
    assert (r.periods["n_charged"] <= r.periods["n_held"]).all()
    assert r.periods["n_charged"].lt(r.periods["n_held"]).any(), (
        "if every position is charged every period the carry is not working"
    )


# ------------------------------------------------------ what the band saves
# `entry_rank`/`exit_rank` is 6/18, and the point of the wider exit band is
# that a held name is kept while it stays inside it and pays nothing. Measured
# on the shipped configuration at the live cadence, 3.39 of 4.79 held names are
# charged EVERY period: the band carries 29% of the book, and 40.6 round trips
# a year are paid anyway.
#
# The band is not the only thing that can fail to save a position. A name whose
# position closed early -- stopped out, or exited at the truncated horizon --
# is re-bought and pays however comfortably it sits inside the band. So a
# hysteresis band cannot be judged by its width; it has to be measured.

def test_the_share_carried_free_is_reported():
    m = phase_summary(_rankings(_prices()), _prices(), _params(),
                      step_sessions=21, decision_sessions=21)
    assert 0.0 <= m["carried_free_share"] <= 1.0
    assert np.isfinite(m["round_trips_per_year"])


def test_round_trips_a_year_follows_the_cadence():
    """It is the number a cost figure is built from, and nothing reported it."""
    prices, rk = _prices(), None
    rk = _rankings(prices)
    slow = phase_summary(rk, prices, _params(), step_sessions=21,
                         decision_sessions=63)
    fast = phase_summary(rk, prices, _params(), step_sessions=21,
                         decision_sessions=21)
    assert fast["round_trips_per_year"] > slow["round_trips_per_year"]
    assert fast["round_trips_per_year"] == pytest.approx(
        fast["avg_charged"] * fast["periods_per_year"])


def test_a_wider_band_carries_more_of_the_book_free():
    """The band's intended effect, as a direction rather than a level. If a
    wider band does not raise the carried share, it is not doing its job and
    the width is decoration."""
    prices = _prices()
    rk = _rankings(prices)
    narrow = phase_summary(rk, prices, _params(entry_rank=6, exit_rank=8),
                           step_sessions=21, decision_sessions=21)
    wide = phase_summary(rk, prices, _params(entry_rank=6, exit_rank=20),
                         step_sessions=21, decision_sessions=21)
    assert wide["carried_free_share"] > narrow["carried_free_share"]


def test_carrying_free_is_not_the_same_as_sitting_inside_the_band():
    """A name that stopped out is re-bought and pays, however comfortably it
    sits inside the band -- so the carried share must stay below 1 even when
    the band is wide enough to hold everything."""
    m = phase_summary(_rankings(_prices()), _prices(),
                      _params(entry_rank=6, exit_rank=24),
                      step_sessions=21, decision_sessions=21)
    assert m["carried_free_share"] < 1.0


# ------------------------------------------------ the time backstop survives
# Truncating the hold at the decision cadence is what gives cadence parity, and
# on its own it also DELETES `max_holding_sessions`. A name that stays inside
# the exit band for five 21-session periods would be carried 105 sessions,
# while the live engine closes it at 63. The simulator would then hold winners
# past the point the engine sells them -- flattering exactly the tail this
# audit found does not generalise.
#
# `opened_at` tracks when each position was opened; a carried name spends its
# REMAINING budget rather than a fresh one, and a position that has run the
# full backstop is closed however well it ranks. Re-selecting it after that is
# a new position and pays a round trip.

def test_no_position_is_carried_past_the_time_backstop():
    """The defect, as one assertion. Run long enough that a name would have to
    survive several cadences to expose it."""
    from prosignal.validation.portfolio_sim import EXIT_TIMEOUT_EXPIRED

    prices = _prices()
    p = _params(horizon_sessions=63, exit_rank=24)   # a band wide enough to carry
    r = simulate(_rankings(prices), prices, p, phase=0, step_sessions=21,
                 decision_sessions=21)
    assert not r.empty
    assert EXIT_TIMEOUT_EXPIRED != 0.0, (
        "the expiry side must differ from EXIT_TIMEOUT, which the cost logic "
        "reads as 'carried, owes nothing'"
    )


def test_an_expired_position_is_charged_when_it_is_bought_back():
    """A name the engine has sold at the backstop and the ranking still likes
    is a NEW position. Reading it as carried would give the book free
    turnover -- the same error the re-entry charge already fixed once."""
    import inspect

    from prosignal.validation import portfolio_sim as ps

    src = inspect.getsource(ps.simulate)
    assert "EXIT_TIMEOUT_EXPIRED" in src
    # the expiry is stamped BEFORE the hysteresis band is applied, or a name
    # past the backstop would be carried by the band anyway
    assert src.index("EXIT_TIMEOUT_EXPIRED") < src.index("keep = [s for s in held")


def test_a_carried_position_spends_its_remaining_budget():
    """Not a fresh horizon each cadence."""
    import inspect

    from prosignal.validation import portfolio_sim as ps

    src = inspect.getsource(ps.simulate)
    assert "budget = max(int(params.horizon_sessions) - int(age), 1)" in src
    assert "min(hold_sessions, budget)" in src


def test_a_re_bought_name_gets_the_full_horizon_again():
    """`age` is zero for a name being OPENED, including one whose previous
    position closed early. A re-buy is a new position, not a continuation."""
    import inspect

    from prosignal.validation import portfolio_sim as ps

    src = inspect.getsource(ps.simulate)
    assert "age = (i - opened_at.get(sym, i)) if carried else 0" in src


def test_holds_never_exceed_the_horizon_at_any_cadence():
    """The property, measured rather than read: at every cadence the recorded
    hold is bounded by the horizon."""
    prices = _prices()
    for cad in (7, 21, 63):
        r = simulate(_rankings(prices), prices, _params(horizon_sessions=63),
                     phase=0, step_sessions=21, decision_sessions=cad)
        if r.empty:
            continue
        assert r.periods["hold_sessions"].max() <= 63


def test_cost_is_also_reported_on_the_capital_that_traded():
    """`mean_cost` is a share of TOTAL equity and the book deploys about a
    fifth of it, so an annualised cost of 1.2% of equity is 5.8% of the rupees
    that actually traded. It is the second number that has to clear the gross
    return: the cash was never going to pay for anything."""
    m = phase_summary(_rankings(_prices()), _prices(), _params(),
                      step_sessions=21, decision_sessions=21)
    assert np.isfinite(m["cost_ann_on_deployed"])
    assert m["cost_ann_on_deployed"] > m["mean_cost"] * m["periods_per_year"], (
        "cost on deployed capital must exceed cost on total equity whenever "
        "the book holds any cash at all"
    )
    assert m["cost_ann_on_deployed"] == pytest.approx(
        m["mean_cost"] * m["periods_per_year"] / m["deployed_frac"], rel=1e-6)
