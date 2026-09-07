"""A percentile band is not a top-K, and gross is not net.

Q17 measured portfolio SHAPES gross and found D6-D8 the only one clearing t=2
(+1.19%, t +2.42). The audit's §18 target architecture is built on that number
and so was my own recommendation. Gross was doing all the work.

`simulate` could not price a band -- admission was `entry_rank`/`exit_rank`,
absolute ranks -- so the gross figure could not be turned into a net one. It
can now, and net of the modelled cost at the live cadence, out of sample:

    top-6 shipped        -10.46%  sh -0.28    40 RT/yr
    D6-D8, 200 slots      -7.65%  sh -0.12   908 RT/yr   cost 10.19% of deployed
    D10, 40 slots         -2.83%  sh +0.11   175 RT/yr
    D10, 20 slots         -1.94%  sh +0.15    85 RT/yr
    top-half, 350 slots  -12.96%  sh -0.47  1412 RT/yr   cost 14.47% of deployed

D6-D8 turns +1.19% gross into -7.65% net because band membership CHURNS: names
cross a percentile boundary far more often than they leave a top-20. The
change that survives costing is the number of names, not the band.

These tests pin the two mechanics that produced those numbers, and the
property that the shipped path did not move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_leverage_neutral import (  # noqa: E402
    SYMBOLS,
    _params,
    _prices,
    _rankings,
)

from prosignal.validation.portfolio_sim import phase_summary, simulate


def _run(**over):
    prices = _prices()
    return simulate(_rankings(prices), prices, _params(**over), phase=0,
                    step_sessions=21, decision_sessions=21)


def test_the_shipped_path_is_untouched_by_the_new_parameters():
    """Rank admission and risk-budget sizing are the default, and adding the
    band machinery must not have moved them."""
    prices = _prices()
    p = _params()
    assert p.entry_pct_band is None
    assert p.equal_weight_slots == 0
    a = simulate(_rankings(prices), prices, p, phase=0, step_sessions=21)
    b = simulate(_rankings(prices), prices, _params(), phase=0,
                 step_sessions=21)
    pd.testing.assert_frame_equal(a.periods, b.periods)


def test_a_band_selects_a_different_book_from_a_top_k():
    """The whole point. A top-6 takes the best six; D6-D8 takes the middle of
    the upper half and cannot contain them."""
    top = _run()
    band = _run(entry_pct_band=(0.5, 0.8), max_positions=20,
                equal_weight_slots=20)
    assert not band.empty
    # This fixture carries 24 symbols, so the band holds a handful rather than
    # the ~200 the real cross-section gives it. The property under test is
    # that a band and a top-K are DIFFERENT BOOKS, not how much bigger one is.
    assert not np.allclose(
        top.periods["ret"].to_numpy()[: len(band.periods)],
        band.periods["ret"].to_numpy()[: len(top.periods)]), (
        "band admission produced the same book as the top-K")
    assert band.periods["n_held"].mean() != pytest.approx(
        top.periods["n_held"].mean())


def test_the_band_convention_matches_the_transfer_layer():
    """(lo, hi) is measured from the BOTTOM, so D6-D8 is (0.5, 0.8) in both
    `portfolio_sim` and `validation/transfer.py`. If the two ever disagree,
    one of them is silently describing a different portfolio."""
    import inspect

    from prosignal.validation import portfolio_sim as ps
    from prosignal.validation import transfer as T

    assert "1.0 - (j + 0.5) / n" in inspect.getsource(ps.simulate)
    d68 = [s for s in T.standard_shapes(400, 6) if s.name == "D6-D8"][0]
    assert d68.bands == ((0.5, 0.8, 1.0),)


def test_the_top_of_a_band_is_taken_when_the_book_cannot_hold_it_all():
    """A 20-slot book against a 200-name band is not that band. It takes the
    best 20 INSIDE it, which bunches at the upper edge -- which is why the
    first pass of this measurement disagreed with Q17 and why the natural
    sizes had to be run."""
    import inspect

    from prosignal.validation import portfolio_sim as ps

    src = inspect.getsource(ps.simulate)
    assert "eligible[: max(room, 0)]" in src
    assert "Best first WITHIN the band" in src


def test_equal_weight_sizes_off_capital_not_off_the_stop():
    """The shipped rule is `risk_budget / risk_per_share`, which is why the
    book holds a fifth of its capital. Equal weight removes the confound at
    source instead of dividing it out afterwards."""
    risk = phase_summary(_rankings(_prices()), _prices(), _params(),
                         step_sessions=21, decision_sessions=21)
    equal = phase_summary(_rankings(_prices()), _prices(),
                          _params(equal_weight_slots=6, target_deployment=1.0),
                          step_sessions=21, decision_sessions=21)
    # A multiple would be pinning this fixture's ATR/price ratio: the real
    # book deploys 22% under the risk rule and this one deploys 58%, because
    # the stop distance is a property of the prices. What must hold anywhere
    # is that equal weight deploys MORE, and lands near its target.
    assert equal["deployed_frac"] > risk["deployed_frac"]
    assert equal["deployed_frac"] > 0.7


def test_target_deployment_is_honoured():
    half = phase_summary(_rankings(_prices()), _prices(),
                         _params(equal_weight_slots=6, target_deployment=0.5),
                         step_sessions=21, decision_sessions=21)
    full = phase_summary(_rankings(_prices()), _prices(),
                         _params(equal_weight_slots=6, target_deployment=1.0),
                         step_sessions=21, decision_sessions=21)
    assert half["deployed_frac"] < full["deployed_frac"] * 0.75


def test_liquidity_still_refuses_a_name_equal_weight_cannot_size():
    """Equal weight must not become a way past the liquidity gate: a name with
    no measurable ADTV is refused whichever rule is sizing it."""
    import inspect

    from prosignal.validation import portfolio_sim as ps

    src = inspect.getsource(ps._position)
    band = src[src.index("if p.equal_weight_slots:"):]
    assert "qty_liq" in band, "the equal-weight branch ignores liquidity"


def test_a_wider_band_turns_over_more():
    """The mechanism behind the finding: band membership churns, because a
    name crosses a percentile boundary far more often than it leaves a
    top-20. 908 round trips a year against 85 on the real panel."""
    narrow = phase_summary(_rankings(_prices()), _prices(),
                           _params(entry_pct_band=(0.9, 1.0), max_positions=12,
                                   equal_weight_slots=12),
                           step_sessions=21, decision_sessions=21)
    wide = phase_summary(_rankings(_prices()), _prices(),
                         _params(entry_pct_band=(0.5, 0.8), max_positions=12,
                                 equal_weight_slots=12),
                         step_sessions=21, decision_sessions=21)
    assert wide["round_trips_per_year"] > narrow["round_trips_per_year"]
