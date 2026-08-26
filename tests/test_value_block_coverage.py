"""Five factors that are a constant for three quarters of the universe.

`_attach_fundamentals` neutral-filled a missing filing, so every name without a
statement landed on exactly the same rank for all five value factors. On a card
they read as five independent z-scores all sitting at -0.01 — identical to two
decimals across five ratios built from five different line items, which cannot
happen by chance.

Measured on the live universe:

    statements feed covers          192 of 750 names
    value factors at exactly 0.0    74-78% of names
    coverage on the training panel  10-12%

And the gap is not random. Names WITH statements have **7.5x** the median
turnover of names without (Rs 176 cr against Rs 23 cr), so the value block was
substantially a disguised size bet: the model differentiated on which names the
feed happened to cover.

Stage 4 already states the rule — "a factor scored on a minority of names ranks
the rest by median fill, which is not a ranking" — and enforced it on the
hand-weighted composite. The fitted model, which is the one that actually ranks,
imputed instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import crossmodel as cm


def test_a_missing_filing_leaves_the_rank_absent_not_neutral():
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"] * 3),
                          "symbol": ["A", "B", "C"]})
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = pd.DataFrame({c: 100.0 for c in ["A", "B", "C"]}, index=idx)
    out = cm._attach_fundamentals(panel, None, close, None)
    for f in cm.FUNDAMENTAL_FEATURES:
        assert out[f + "_r"].isna().all(), (
            "a neutral fill here is what made the gap invisible downstream"
        )


def test_the_coverage_floor_is_the_rule_stage_4_already_states():
    assert cm.MIN_FACTOR_COVERAGE == pytest.approx(0.60)


def test_a_model_reports_the_factors_it_actually_used():
    m = cm.CrossSectionalModel(
        coef={"mom_6_1_r": 0.01, "prox_52w_r": 0.02},
        n_train=1000, train_end=pd.Timestamp("2026-01-01").date(),
    )
    assert m.features == ["mom_6_1_r", "prox_52w_r"]
    assert "ridge on 2 cross-sectional features" in m.summary()


def test_scoring_follows_the_models_feature_list_not_the_module_constant():
    """A model fitted on 12 factors must not be applied as though it had 17.
    `mu` and `sd` are positional, so a mismatch mis-standardises silently."""
    m = cm.CrossSectionalModel(
        coef={"mom_6_1_r": 1.0, "prox_52w_r": -1.0},
        n_train=1000, train_end=pd.Timestamp("2026-01-01").date(),
        features=["mom_6_1_r", "prox_52w_r"],
    )
    m.mu = np.array([0.0, 0.0])
    m.sd = np.array([1.0, 1.0])
    m.intercept = 0.0
    feats = pd.DataFrame({
        "symbol": ["A", "B"],
        "mom_6_1_r": [0.5, -0.5],
        "prox_52w_r": [-0.5, 0.5],
        # a column the model was NOT fitted on; it must be ignored, not
        # positionally consumed
        "earnings_yield_r": [9.9, -9.9],
    })
    out = cm.score_with(m, feats)
    assert out.loc["A"] > out.loc["B"]
    contrib = cm.contributions(m, feats)
    assert list(contrib.columns) == ["mom_6_1", "prox_52w"]
    assert "earnings_yield" not in contrib.columns


def test_a_cached_model_with_fewer_factors_is_accepted(tmp_path):
    """A stored model may legitimately carry fewer factors than the code
    declares -- one the feed could not serve was dropped, not renamed."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["mom_6_1_r", "prox_52w_r"],
        "coef": {"mom_6_1_r": 0.01, "prox_52w_r": 0.02},
        "mu": [0.0, 0.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    m = cm.load_cached(path, dt.date(2026, 1, 6), 21)
    assert m is not None
    assert m.features == ["mom_6_1_r", "prox_52w_r"]


def test_a_cached_model_naming_an_unknown_factor_forces_a_refit(tmp_path):
    """Fewer is fine. A factor the code no longer knows means the definitions
    moved, and the stored coefficients describe a different model."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["mom_6_1_r", "a_factor_we_deleted_r"],
        "coef": {"mom_6_1_r": 0.01, "a_factor_we_deleted_r": 0.02},
        "mu": [0.0, 0.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    assert cm.load_cached(path, dt.date(2026, 1, 6), 21) is None


def test_the_stored_feature_order_is_preserved(tmp_path):
    """`mu` and `sd` are positional. A dict whose iteration order changed would
    standardise every factor against the wrong column."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["prox_52w_r", "mom_6_1_r"],       # deliberately not sorted
        "coef": {"mom_6_1_r": 0.01, "prox_52w_r": 0.02},
        "mu": [1.0, 2.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    m = cm.load_cached(path, dt.date(2026, 1, 6), 21)
    assert m.features == ["prox_52w_r", "mom_6_1_r"]


# --------------------------------------------------------- redundancy
def test_the_redundancy_check_measures_the_model_not_the_legacy_composite():
    """It ran on `frame` -- the hand-weighted composite's factor block -- so the
    seventeen columns that actually rank the universe were never checked against
    each other. Measured on the live universe they are not independent:

        amihud / turnover_ratio   -0.869   one factor from two sides
        resid_mom / mom_6_1       +0.770
        resid_mom / prox_52w      +0.601   the momentum block is ~one bet
    """
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4.run)
    assert "model_block" in src
    assert "_redundancy(model_block if model_block is not None else frame, cfg)" in src


def test_a_breach_names_the_pair_rather_than_only_counting_them():
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4.run)
    block = src[src.index("if redundancy.breaches:"):]
    assert "pairs" in block and "{a}/{b}" in block, (
        "a count alone does not say which factors are the same bet"
    )
