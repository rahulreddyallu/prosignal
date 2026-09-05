"""The monitors that watch the scorer that actually orders the book.

Both of these exist because the thing they check was silently not happening.

`_redundancy` was fed `model_features`, which has been `None` on every run since
the fitted ranker was removed in the 2026-09-03 cleanup, so it fell through to a
legacy frame and NO SHIPPED v3 FACTOR PAIR HAD EVER BEEN CHECKED. A monitor
reading an empty frame reports no breaches, which is indistinguishable from a
clean bill of health.

The card printed each theme's DECLARED weight while serialising a contribution
computed from the EFFECTIVE one, so the two printed numbers did not multiply out
for any name missing a theme -- which, since `quality` covers about a fifth of
the universe, is most of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import engine
from prosignal.stages.stage4_core_score import (
    _residual_share,
    _v3_blocks,
    _v3_redundancy,
    theme_effective_weights,
)


class _Redundancy:
    max_abs_spearman = 0.6
    on_breach = "log"


class _Cfg:
    redundancy = _Redundancy()


def _scored(n=200, seed=7, drop_quality=False, clone=None):
    """A v3-shaped scored frame: `<factor>_r` ranks and `<theme>_sub` sub-scores.

    `clone` copies one factor's rank into another with the OPPOSITE shipped sign,
    which is how a genuine cross-theme duplicate looks once oriented.

    The original clone used `prox_52w` -> `ulcer_120`, the real leak this check
    was built for. `ulcer_120` was REMOVED from the engine on 2026-09-05 for
    exactly that reason, so the pair no longer exists to test with and the
    synthetic clone moved to another cross-theme pair. The check is the same;
    what it caught in production is now gone by construction.
    """
    rng = np.random.default_rng(seed)
    idx = [f"S{i:03d}" for i in range(n)]
    out = pd.DataFrame(index=idx)
    for f in engine.ALL_FACTORS:
        out[f + "_r"] = rng.uniform(-1, 1, n)
    if clone:
        src, dst = clone
        s_sign = engine.THEMES[engine.FACTOR_THEME[src]].signs[src]
        d_sign = engine.THEMES[engine.FACTOR_THEME[dst]].signs[dst]
        out[dst + "_r"] = out[src + "_r"] * (s_sign * d_sign)
    for t, th in engine.THEMES.items():
        cols = [f + "_r" for f in th.names]
        signs = np.array([th.signs[f] for f in th.names])
        out[t + "_sub"] = (out[cols].to_numpy() * signs).mean(axis=1)
    if drop_quality:
        out["quality_sub"] = np.nan
        for f in engine.THEMES["quality"].names:
            out[f + "_r"] = np.nan
    return out


# --------------------------------------------------------------- redundancy
def test_the_check_sees_the_twenty_two_factors_that_order_the_book():
    """The regression: it used to see none of them."""
    themes, factors = _v3_blocks(_scored())
    assert list(factors.columns) == list(engine.ALL_FACTORS)
    assert set(themes.columns) == set(engine.THEMES)


def test_factor_ranks_are_oriented_before_they_are_correlated():
    """Unoriented, a -1 factor duplicating a +1 factor reports a NEGATIVE rho and
    reads as diversification. That was exactly the `ulcer_120` / `prox_52w` case,
    and it is the reason orientation is not cosmetic."""
    scored = _scored(clone=("prox_52w", "downside_vol_60"))
    _, factors = _v3_blocks(scored)
    rho = factors["prox_52w"].corr(factors["downside_vol_60"], method="spearman")
    assert rho == pytest.approx(1.0, abs=1e-9)
    raw = scored["prox_52w_r"].corr(scored["downside_vol_60_r"], method="spearman")
    assert raw == pytest.approx(-1.0, abs=1e-9)


def test_a_cross_theme_duplicate_is_a_breach_because_the_cap_cannot_see_it():
    """A momentum factor living in `risk` moves with momentum. The 40% cap is
    applied per theme, so the exposure is carried twice and no cap notices --
    which is what `ulcer_120` did before it was removed."""
    report = _v3_redundancy(_scored(clone=("prox_52w", "downside_vol_60")), _Cfg())
    pairs = {tuple(sorted((a, b))) for a, b, _ in report.breaches}
    assert ("downside_vol_60", "prox_52w") in pairs
    assert any("CROSS-THEME" in n for n in report.notes)


def test_a_within_theme_duplicate_is_reported_but_is_not_a_breach():
    """Two factors inside one theme are SUPPOSED to overlap -- the theme averages
    them and the average is what carries the weight."""
    report = _v3_redundancy(_scored(clone=("prox_52w", "prox_52w_now")), _Cfg())
    pairs = {tuple(sorted((a, b))) for a, b, _ in report.breaches}
    assert ("prox_52w", "prox_52w_now") not in pairs
    assert any("Within-theme" in n for n in report.notes)


def test_independent_factors_produce_no_breaches():
    assert _v3_redundancy(_scored(), _Cfg()).breaches == []


# ----------------------------------------------------------------- weights
def test_effective_weight_equals_declared_only_when_every_theme_is_present():
    eff = theme_effective_weights(_scored())
    for t, th in engine.THEMES.items():
        assert eff[t][0] == pytest.approx(th.weight, abs=1e-9)
        assert eff[t][1] == pytest.approx(1.0)


def test_a_missing_theme_hands_its_weight_to_the_others():
    """The number the card used to hide. With `quality` absent, momentum's
    declared 40% is applied at 40/(1-0.18991) = 49.4%."""
    eff = theme_effective_weights(_scored(drop_quality=True))
    assert eff["quality"] == (pytest.approx(0.0), pytest.approx(0.0))
    expected = engine.THEMES["momentum"].weight / (1.0 - engine.THEMES["quality"].weight)
    assert eff["momentum"][0] == pytest.approx(expected, abs=1e-9)
    assert eff["momentum"][0] > engine.THEMES["momentum"].weight
    assert sum(e for e, _ in eff.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------- sector residual
def test_the_residual_bucket_counts_unmapped_and_too_small_sectors_alike():
    """Both land in the same `__RESID__` group and are ranked against each other,
    which is not neutralisation -- so both have to be counted."""
    index = [f"S{i:03d}" for i in range(100)]
    big = {s: "Financial Services" for s in index[:60]}
    tiny = {s: "Forest Materials" for s in index[60:65]}   # under MIN_SECTOR_NAMES
    sectors = {**big, **tiny}                              # index[65:] unmapped
    share = _residual_share(pd.Index(index), sectors)
    assert share == pytest.approx(0.40, abs=1e-9)
    assert _residual_share(pd.Index(index), {}) == 1.0


# ------------------------------------------------- the coverage floor
def test_the_coverage_floor_can_never_demand_more_names_than_exist():
    """It was `max(int(0.6 * n), 20)`, which asks for twenty names however small
    the universe is -- so a three-name cross-section failed at "covers 3 of 3,
    under the 20 floor".

    The two conditions answer different questions and only one was intended: the
    error says a ranking built on a MINORITY of the universe is a ranking of that
    minority, which is a share. The absolute term is a guard against a near-empty
    cross-section and must not exceed what exists.

    It bit exactly where refusing is worst. `absolute_floor`'s own note records
    11 names clearing at the COVID trough and 8 in the 2022 drawdown; on such a
    day the engine would have refused to rank at all rather than ranking the
    names that survived.
    """
    for universe in (1, 3, 11, 19, 20, 50, 500):
        floor = min(max(int(0.6 * universe), 20), universe)
        assert floor <= universe, f"floor {floor} exceeds a universe of {universe}"
        if universe >= 34:                      # 0.6 * 34 > 20
            assert floor == int(0.6 * universe)
    assert min(max(int(0.6 * 3), 20), 3) == 3
    assert min(max(int(0.6 * 500), 20), 500) == 300
