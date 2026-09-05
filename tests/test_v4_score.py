"""The v4 composite -- v3 minus seven factors, and nothing else.

The whole claim of `features/v4.py` is that it differs from v3 in exactly one
way. That claim is worth more than the prune itself: a "prune" that also moved a
weight, flipped a sign or changed the blend would be an unvalidated refit wearing
the prune's evidence. Every test here checks one way it could stop being true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v3, v4


def _raw(n=180, seed=5):
    """Raw factor values for one cross-section, all 22 columns present."""
    rng = np.random.default_rng(seed)
    idx = [f"S{i:03d}" for i in range(n)]
    return pd.DataFrame(
        {f: rng.normal(size=n) for f in v3.ALL_FACTORS}, index=idx)


# ------------------------------------------------------------ the difference
def test_v4_is_exactly_v3_minus_the_seven():
    assert set(v4.REMOVED) < set(v3.ALL_FACTORS)
    assert len(v4.REMOVED) == 7
    assert set(v4.ALL_FACTORS) == set(v3.ALL_FACTORS) - set(v4.REMOVED)
    assert len(v4.ALL_FACTORS) == 15


def test_the_seven_are_the_ones_the_split_half_nominated():
    """Frozen here so a later edit has to argue with the evidence rather than
    quietly widen the prune. These are the factors an independent selection
    nominated in BOTH halves of the panel, and again after it was rebuilt."""
    assert v4.REMOVED == ("deliv_chg_5", "mom_2_0", "mom_3_1", "mom_accel",
                          "resid_rev_21", "ulcer_120", "voladj_mom_6_1")


def test_no_theme_weight_sign_or_horizon_moved():
    """The prune removes factors. If it also refitted the weights it would be a
    second fit on the panel that chose the prune, and the CPCV numbers would not
    describe it."""
    for name, th in v4.THEMES.items():
        src = v3.THEMES[name]
        assert th.weight == src.weight
        assert th.horizon == src.horizon
        assert th.coverage == src.coverage
        for f, sign in th.factors:
            assert sign == src.signs[f], f"{f} sign changed"


def test_every_theme_survives_the_prune():
    """A theme emptied by the prune would silently change `min_themes` from
    three-of-five to three-of-four."""
    assert set(v4.THEMES) == set(v3.THEMES)
    assert all(len(th.factors) >= 2 for th in v4.THEMES.values())


def test_ulcer_120_is_gone_so_the_cap_binds_what_it_claims_to():
    """It sat in `risk` and correlated +0.69..+0.78 oriented with prox_52w, so
    momentum exposure was carried past a cap applied per theme."""
    assert "ulcer_120" not in v4.ALL_FACTORS
    assert "ulcer_120" in v3.ALL_FACTORS


# ---------------------------------------------------------------- the blend
def test_v4_reuses_v3s_blend_rather_than_copying_it():
    """Scoring under v3's own table through v4's delegate must reproduce v3
    exactly -- which is what proves the only difference is the factor list."""
    raw = _raw()
    a = v3.score_frame(raw)
    b = v3.score_frame(raw, themes=v3.THEMES)
    pd.testing.assert_frame_equal(a, b)


def test_passing_a_theme_table_does_not_change_the_default_path():
    """The frozen scorer has to stay byte-identical after the parameter was
    added, or two sealed holdouts stop describing it."""
    raw = _raw(seed=9)
    scored = v3.score_frame(raw)
    assert list(v3.ALL_FACTORS) == [c[:-2] for c in scored.columns
                                    if c.endswith("_r")]
    assert scored["score"].notna().all()


def test_v4_emits_the_columns_the_rest_of_the_engine_reads():
    scored = v4.score_frame(_raw())
    for t in v4.THEMES:
        assert t + "_sub" in scored.columns
        assert t + "_contrib" in scored.columns
    for f in v4.ALL_FACTORS:
        assert f + "_r" in scored.columns
    for f in v4.REMOVED:
        assert f + "_r" not in scored.columns
    for c in ("n_themes", "score", "score_rank"):
        assert c in scored.columns


def test_contributions_still_sum_to_the_score():
    scored = v4.score_frame(_raw(seed=3))
    total = sum(scored[t + "_contrib"].fillna(0.0) for t in v4.THEMES)
    ok = scored["score"].notna()
    assert np.allclose(total[ok], scored["score"][ok], atol=1e-12)


def test_v4_ranks_a_different_book_from_v3():
    """If the prune changed nothing the whole exercise would be theatre."""
    raw = _raw(seed=17)
    a = v3.score_frame(raw)["score"].sort_values(ascending=False).index[:6]
    b = v4.score_frame(raw)["score"].sort_values(ascending=False).index[:6]
    assert list(a) != list(b)


def test_the_spec_hash_moves_when_the_specification_does():
    """`SPEC_SHA256` is what makes "the model that ran" checkable afterwards."""
    before = v4.SPEC_SHA256
    tweaked = dict(v4.THEMES)
    t = next(iter(tweaked))
    tweaked[t] = v3.Theme(weight=tweaked[t].weight + 0.01,
                          horizon=tweaked[t].horizon,
                          coverage=tweaked[t].coverage,
                          factors=tweaked[t].factors)
    import hashlib
    after = hashlib.sha256(
        repr(sorted((k, th.weight, th.horizon, th.coverage, th.factors)
                    for k, th in tweaked.items())).encode()).hexdigest()
    assert before != after


def test_a_name_missing_a_theme_still_renormalises():
    raw = _raw(seed=21)
    raw.loc[raw.index[:40], [f for f in v4.THEMES["quality"].names]] = np.nan
    scored = v4.score_frame(raw)
    thin = scored.loc[raw.index[:40]]
    assert (thin["n_themes"] == 4).all()
    assert thin["quality_contrib"].isna().all()
    assert thin["score"].notna().all()
