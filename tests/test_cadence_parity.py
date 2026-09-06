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
