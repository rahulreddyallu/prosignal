"""The panel must be point-in-time and the folds must actually purge."""
import numpy as np
import pandas as pd
import pytest

from prosignal.features.crosssec import MIN_LOOKBACK, build_panel, cross_sectional_rank
from prosignal.features.linear import (
    diebold_mariano, elastic_net_fit, predict, purged_walk_forward, ridge_fit,
)


def _prices(n=900, k=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(n, k)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(k)],
    )
    turnover = pd.DataFrame(1e7, index=idx, columns=close.columns)
    return close, turnover


def test_a_future_price_change_cannot_move_any_feature():
    """The leakage test that matters: perturb prices strictly after the feature
    date and every feature must be byte-identical."""
    close, tno = _prices()
    h = 21
    base = build_panel(close, tno, horizon=h, step=60)
    assert not base.empty
    cut = base["date"].iloc[0]
    tampered = close.copy()
    # 15%, not 300%: a larger shock pushes the label past the |label| < 1
    # sanity filter and empties the frame, which would prove nothing.
    tampered.loc[tampered.index > cut] *= 1.15
    after = build_panel(tampered, tno, horizon=h, step=60)

    feat_cols = [c for c in base.columns if c not in ("label", "label_rank", "date", "symbol")]
    a = base[base["date"] == cut].set_index("symbol")[feat_cols].sort_index()
    b = after[after["date"] == cut].set_index("symbol")[feat_cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_the_label_does_move_when_the_future_changes():
    """Control for the test above: if nothing moved, the test proves nothing."""
    close, tno = _prices()
    base = build_panel(close, tno, horizon=21, step=60)
    cut = base["date"].iloc[0]
    tampered = close.copy()
    tampered.loc[tampered.index > cut] *= 1.15
    after = build_panel(tampered, tno, horizon=21, step=60)
    a = base[base["date"] == cut]["label"].to_numpy()
    b = after[after["date"] == cut]["label"].to_numpy()
    assert not np.allclose(a, b)


def test_training_folds_never_touch_test_dates():
    close, tno = _prices()
    panel = build_panel(close, tno, horizon=21, step=21)
    folds = purged_walk_forward(panel, [], horizon=21, n_splits=3, embargo=1)
    assert folds
    for f in folds:
        train_dates = set(f["train"]["date"])
        test_dates = set(f["test"]["date"])
        assert not (train_dates & test_dates)
        # every training date precedes the test block, with purge+embargo removed
        assert max(train_dates) < min(test_dates)


def test_cross_sectional_rank_is_bounded_and_date_local():
    s = pd.Series([1.0, 5.0, 100.0, np.nan])
    r = cross_sectional_rank(s)
    assert r.min() >= -1.0 and r.max() <= 1.0
    assert pd.isna(r.iloc[3])


def test_elastic_net_shrinks_a_null_feature_to_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.5, size=400)
    m = elastic_net_fit(x, y, alpha=0.05, l1_ratio=0.9)
    assert abs(m["coef"][3]) < 0.05
    assert m["coef"][0] > 1.0


def test_diebold_mariano_reports_no_difference_for_identical_forecasts():
    e = np.random.default_rng(1).normal(size=60)
    stat, p = diebold_mariano(e, e.copy())
    assert not np.isfinite(stat) or p > 0.99


def test_cached_coefficients_reproduce_the_fit(tmp_path):
    """The cheap path must score identically to the model it cached.

    If it does not, the warm run silently diverges from the validated fit.
    """
    import datetime as dt

    from prosignal.features import crossmodel as cm

    close, tno = _prices(n=900, k=60)
    as_of = close.index[-1].date()
    scores, model, reason = cm.fit_predict(close, tno, as_of)
    if reason is not None:
        pytest.skip(f"fixture too small to fit: {reason}")

    path = tmp_path / "m.json"
    cm.save_cache(path, model, as_of)
    loaded = cm.load_cached(path, as_of)
    assert loaded is not None

    feats = cm.today_features(close, tno, as_of)
    assert feats is not None
    replayed = cm.score_with(loaded, feats)
    common = scores.index.intersection(replayed.index)
    assert len(common) > 20
    np.testing.assert_allclose(
        scores.loc[common].to_numpy(), replayed.loc[common].to_numpy(), atol=1e-9
    )


def test_stale_cache_is_rejected_rather_than_used(tmp_path):
    import datetime as dt

    from prosignal.features import crossmodel as cm

    m = cm.CrossSectionalModel(coef={c: 0.0 for c in cm.FEATURE_COLUMNS},
                               n_train=1000, train_end=dt.date(2020, 1, 1))
    m.mu = np.zeros(len(cm.FEATURE_COLUMNS))
    m.sd = np.ones(len(cm.FEATURE_COLUMNS))
    m.intercept = 0.0
    path = tmp_path / "m.json"
    cm.save_cache(path, m, dt.date(2020, 1, 1))
    assert cm.load_cached(path, dt.date(2020, 1, 1)) is not None
    # far beyond the refit cadence
    assert cm.load_cached(path, dt.date(2021, 1, 1)) is None


def test_model_abstains_rather_than_guessing_on_short_history():
    from prosignal.features import crossmodel as cm

    close, tno = _prices(n=120, k=60)
    scores, model, reason = cm.fit_predict(close, tno, close.index[-1].date())
    assert scores is None and model is None
    assert "history" in reason.lower()
