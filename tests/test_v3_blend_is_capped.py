"""The 40% cap has to survive contact with a name that is missing a theme.

`THEMES[t].weight` is post-cap, post-floor and post-coverage-cap -- correct for
a name carrying all five themes and for no other name. Fundamentals reach 8.8%
of the live universe, so 91% of it carries four.

The blend used to divide by the sum of the weights a name actually had. That is
the right shape and the wrong arithmetic: dividing by 0.81009 does not re-impose
the constraint the weights were chosen under, it removes it. Momentum's 0.40
became 0.40/0.81009 = 49.38% on nine names in ten -- measured live on
2026-09-03 at 48.55% mean effective weight and 63.59% of the realised
cross-sectional spread, against a cap of 40%.

`cap_weights` existed, was correct, and was never called at scoring time. These
tests pin that it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v3


def _ranks(n: int = 240, seed: int = 11) -> pd.DataFrame:
    """A factor-rank frame in [-1, 1], the shape `score_frame` consumes."""
    rng = np.random.default_rng(seed)
    idx = [f"SYM{i:04d}" for i in range(n)]
    return pd.DataFrame(
        {f: rng.uniform(-1.0, 1.0, n) for f in v3.ALL_FACTORS}, index=idx)


def _drop_theme(raw: pd.DataFrame, theme: str, rows=None) -> pd.DataFrame:
    out = raw.copy()
    cols = v3.THEMES[theme].names
    if rows is None:
        out[cols] = np.nan
    else:
        out.loc[rows, cols] = np.nan
    return out


# =============================================================================
# the cap
# =============================================================================

def test_no_theme_exceeds_the_cap_when_a_theme_is_missing():
    """The live case: quality is absent for 91% of names, and momentum used to
    take its share."""
    raw = _drop_theme(_ranks(), "quality")
    scored = v3.score_frame(raw)
    for t in v3.THEMES:
        w = scored[t + "_w"].dropna()
        if not len(w):
            continue
        assert w.max() <= 0.40 + 1e-9, (
            f"{t} reached {w.max():.4f} of the blend against a 40% cap"
        )


@pytest.mark.parametrize("missing", ["quality", "ownership", "risk", "reversal"])
def test_the_cap_holds_whichever_theme_drops_out(missing):
    """Delivery and fundamentals are both behind a `log.warning` and continue,
    so any of these can vanish on a live run."""
    scored = v3.score_frame(_drop_theme(_ranks(), missing))
    assert scored["momentum_w"].dropna().max() <= 0.40 + 1e-9


def test_momentum_does_not_absorb_the_missing_theme():
    """The specific arithmetic: 0.40/0.81009 = 0.4938. That number must not
    appear."""
    scored = v3.score_frame(_drop_theme(_ranks(), "quality"))
    w = scored["momentum_w"].dropna()
    assert len(w)
    assert not np.isclose(w.max(), 0.40 / 0.81009, atol=1e-3), (
        "momentum is being renormalised over the surviving themes again"
    )


def test_weights_sum_to_one_over_the_themes_a_name_has():
    raw = _drop_theme(_ranks(), "quality")
    scored = v3.score_frame(raw)
    cols = [t + "_w" for t in v3.THEMES]
    total = scored[cols].sum(axis=1, min_count=1)
    scored_rows = scored["score"].notna()
    assert np.allclose(total[scored_rows], 1.0), (
        "a blend whose weights do not sum to 1 is not a weighted average"
    )


def test_two_names_with_different_coverage_get_different_weights():
    """The two-population problem, made visible.

    A name WITH fundamentals and a name without are scored by different weight
    VECTORS, and the frame now says which instead of leaving the reader to
    infer it. Note what does NOT differ: momentum sits at the cap either way --
    its fitted weight is exactly 0.40, and dropping a theme can only push it
    up, where the cap stops it. The weight the missing theme frees goes to the
    themes that were not at their cap, which is the point of capping.
    """
    raw = _ranks()
    half = list(raw.index[:120])
    raw = _drop_theme(raw, "quality", rows=half)
    scored = v3.score_frame(raw)
    four = scored.loc[half[0]]
    five = scored.loc[raw.index[200]]

    assert pd.isna(four["quality_w"]), "a name without quality carries no quality weight"
    assert five["quality_w"] == pytest.approx(v3.THEMES["quality"].weight)
    assert four["ownership_w"] > five["ownership_w"], (
        "the weight freed by the missing theme has to land somewhere, and it "
        "lands on the themes that were not at their cap"
    )
    assert four["momentum_w"] == pytest.approx(0.40)
    assert five["momentum_w"] == pytest.approx(0.40)


def test_the_floor_is_not_re_applied_to_already_floored_weights():
    """`cap_weights` blends `floor + (1 - floor*n) * w`. Run over the frozen
    vector it returns momentum at 0.06 + 0.70*0.40 = 0.34 -- a different model
    from the one that was fitted, applied to the names the fit was correct for.
    This is the regression that test above caught during the repair itself."""
    w = v3._weights_for_pattern(tuple([True] * len(v3.THEMES)))
    momentum = w[list(v3.THEMES).index("momentum")]
    assert momentum == pytest.approx(0.40), (
        f"momentum came back at {momentum:.4f}; the floor is being applied twice"
    )


# =============================================================================
# the property the card depends on
# =============================================================================

def test_weight_times_subscore_equals_contribution():
    """THE CARD'S ARITHMETIC. The panel prints a weight, a percentile and a
    contribution under a heading that says "sums to the score", so a reader can
    check it. Before the blend emitted `_w`, the presentation layer had only
    `Theme.weight` to show and every row was out by 1/den -- exactly 1.2344x on
    a four-theme name."""
    scored = v3.score_frame(_drop_theme(_ranks(), "quality"))
    for t in v3.THEMES:
        lhs = scored[t + "_w"] * scored[t + "_sub"]
        rhs = scored[t + "_contrib"]
        m = rhs.notna()
        if not m.any():
            continue
        assert np.allclose(lhs[m], rhs[m], rtol=0, atol=1e-12), (
            f"{t}: the weight shown does not produce the contribution shown"
        )


def test_contributions_still_sum_to_the_score():
    """Unchanged by the fix, and worth pinning beside it: the previous
    implementation satisfied this too, which is why the defect passed 1,638
    tests."""
    scored = v3.score_frame(_drop_theme(_ranks(), "quality"))
    total = scored[[t + "_contrib" for t in v3.THEMES]].sum(axis=1, min_count=1)
    m = scored["score"].notna()
    np.testing.assert_allclose(total[m].to_numpy(), scored.loc[m, "score"].to_numpy(),
                               rtol=1e-9, atol=1e-12)


def test_a_full_coverage_name_keeps_the_fitted_weights():
    """The fix must be a no-op where the frozen weights were already correct --
    a name carrying all five themes."""
    scored = v3.score_frame(_ranks())
    for t in v3.THEMES:
        w = scored[t + "_w"].dropna()
        assert np.allclose(w, v3.THEMES[t].weight, atol=1e-6), (
            f"{t} moved for a name that has every theme"
        )


# =============================================================================
# honesty of the labels
# =============================================================================

def test_every_theme_carries_a_label_that_is_not_its_key():
    """`quality` buys low and unstable margins -- that is what the search
    measured (IC -0.0351, t -5.52) and it is not a typo. The key is kept
    because it names columns everywhere; the label is what a human reads, and
    the run notes now use it instead of the key."""
    for name, theme in v3.THEMES.items():
        assert theme.label, f"{name} has no reader-facing label"
    assert v3.THEMES["quality"].label != "quality"
    assert "quality" not in v3.THEMES["quality"].label.lower(), (
        "the label must not reintroduce the word the signs contradict"
    )
