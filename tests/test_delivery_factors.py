"""Delivery-based factors and their missing-data convention."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.crosssec import (
    FEATURES, NEUTRAL_WHEN_MISSING, _features_at, build_panel,
)


@pytest.fixture
def frames():
    n, m = 340, 10
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(3)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, size=(n, m)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(m)],
    )
    turnover = pd.DataFrame(rng.uniform(1e7, 6e7, size=(n, m)), index=idx, columns=close.columns)
    delivery = pd.DataFrame(rng.uniform(20.0, 80.0, size=(n, m)), index=idx, columns=close.columns)
    bench = close.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")
    return close, turnover, delivery, bench


def test_both_delivery_factors_are_registered():
    assert "deliv_pct" in FEATURES
    assert "deliv_trend" in FEATURES
    assert NEUTRAL_WHEN_MISSING == {"deliv_pct", "deliv_trend"}


def test_deliv_pct_tracks_the_delivered_share(frames):
    close, turnover, delivery, bench = frames
    delivery = delivery.copy()
    delivery["S0"] = 90.0
    delivery["S1"] = 10.0
    f = _features_at(close, turnover, len(close) - 1, bench, delivery=delivery)
    assert f.loc["S0", "deliv_pct"] == pytest.approx(90.0)
    assert f.loc["S1", "deliv_pct"] == pytest.approx(10.0)


def test_deliv_trend_is_positive_when_delivery_is_rising(frames):
    close, turnover, delivery, bench = frames
    delivery = delivery.copy()
    delivery["S2"] = 30.0
    delivery.iloc[-21:, delivery.columns.get_loc("S2")] = 70.0
    f = _features_at(close, turnover, len(close) - 1, bench, delivery=delivery)
    assert f.loc["S2", "deliv_trend"] > 20.0


def test_absent_delivery_leaves_the_columns_nan_not_missing(frames):
    close, turnover, _, bench = frames
    f = _features_at(close, turnover, len(close) - 1, bench, delivery=None)
    for name in ("deliv_pct", "deliv_trend"):
        assert name in f.columns
        assert f[name].isna().all()


def test_a_name_with_no_delivery_ranks_neutral_rather_than_dropping(frames):
    """82% panel coverage: requiring delivery would discard a fifth of the
    universe over a feed gap, so a missing print scores 0.0 and keeps the row."""
    close, turnover, delivery, _ = frames
    delivery = delivery.copy()
    delivery["S3"] = np.nan
    panel = build_panel(close, turnover, horizon=21, step=40, min_names=3, delivery=delivery)
    assert not panel.empty
    kept = panel[panel["symbol"] == "S3"]
    assert not kept.empty, "the name was dropped instead of ranking neutral"
    assert (kept["deliv_pct_r"] == 0.0).all()
    assert (kept["deliv_trend_r"] == 0.0).all()


def test_a_panel_built_without_delivery_still_scores_every_name(frames):
    close, turnover, delivery, _ = frames
    with_d = build_panel(close, turnover, horizon=21, step=40, min_names=3, delivery=delivery)
    without = build_panel(close, turnover, horizon=21, step=40, min_names=3, delivery=None)
    assert len(with_d) == len(without)
    assert (without["deliv_pct_r"] == 0.0).all()


def test_delivery_factors_never_read_past_the_decision_bar(frames):
    close, turnover, delivery, bench = frames
    i = len(close) - 30
    before = _features_at(close, turnover, i, bench[: i + 1], delivery=delivery)
    tampered = delivery.copy()
    tampered.iloc[i + 1 :] = 99.0
    after = _features_at(close, turnover, i, bench[: i + 1], delivery=tampered)
    for name in ("deliv_pct", "deliv_trend"):
        pd.testing.assert_series_equal(before[name], after[name], check_names=False)
