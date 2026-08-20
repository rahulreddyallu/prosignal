"""The card must explain the number it prints.

Stage 4 hands the ranking to the fitted model when coverage allows, but the
evidence used to be taken from the hand-weighted composite regardless. That
described a calculation that did not happen: a score produced by 24 fitted
coefficients, explained by two factors weighted 50/50 by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.crossmodel import (
    FEATURE_COLUMNS, CrossSectionalModel, contributions, score_with,
    standardised_features,
)


@pytest.fixture
def model_and_features():
    rng = np.random.default_rng(7)
    n = 40
    coef = {c: float(rng.normal(0, 0.01)) for c in FEATURE_COLUMNS}
    m = CrossSectionalModel(coef=coef, n_train=1000, train_end=pd.Timestamp("2026-01-01").date())
    m.mu = rng.normal(0, 0.1, size=len(FEATURE_COLUMNS))
    m.sd = np.abs(rng.normal(1.0, 0.1, size=len(FEATURE_COLUMNS))) + 0.5
    m.intercept = 0.3
    feats = pd.DataFrame(
        rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    feats["symbol"] = [f"S{i}" for i in range(n)]
    return m, feats


def test_contributions_sum_back_to_the_score(model_and_features):
    """If the parts do not add up to the whole, the card is not an explanation."""
    m, feats = model_and_features
    contrib = contributions(m, feats)
    rebuilt = contrib.sum(axis=1) + m.intercept
    x = feats[FEATURE_COLUMNS].to_numpy("float64")
    coef = np.array([m.coef[c] for c in FEATURE_COLUMNS])
    direct = ((x - m.mu) / m.sd) @ coef + m.intercept
    np.testing.assert_allclose(rebuilt.to_numpy(), direct, rtol=1e-9)


def test_contribution_ordering_matches_the_ranking(model_and_features):
    """The name the model ranks first must have the larger total contribution."""
    m, feats = model_and_features
    ranked = score_with(m, feats)
    contrib = contributions(m, feats).sum(axis=1)
    assert contrib[ranked.index[0]] > contrib[ranked.index[-1]]


def test_every_fitted_factor_is_attributable(model_and_features):
    m, feats = model_and_features
    contrib = contributions(m, feats)
    expected = {c[:-2] if c.endswith("_r") else c for c in FEATURE_COLUMNS}
    assert set(contrib.columns) == expected
    assert len(contrib.columns) == 24


def test_standardised_features_are_the_z_scores_the_coefficients_multiply(model_and_features):
    m, feats = model_and_features
    z = standardised_features(m, feats)
    contrib = contributions(m, feats)
    for col in z.columns:
        coef = m.coef[col + "_r"]
        np.testing.assert_allclose(z[col].to_numpy() * coef, contrib[col].to_numpy(), rtol=1e-9)


def test_a_zero_scale_column_does_not_divide_by_zero(model_and_features):
    m, feats = model_and_features
    m.sd = m.sd.copy()
    m.sd[0] = 0.0
    contrib = contributions(m, feats)
    assert np.isfinite(contrib.to_numpy()).all()
