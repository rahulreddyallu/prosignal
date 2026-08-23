"""Alpha, or factor exposure wearing a different name.

Every factor this engine fits is a published premium. That gives it the
economic rationale a reviewer asks for, and it carries a corollary: a
portfolio built from published premia should be expected to LOAD on them. The
regression here asks whether anything survives once those loadings are priced.

The factors are built from the engine's own definitions on the same universe
and the same dates, which makes this a HOSTILE test -- the regressors are as
close to the strategy's own construction as they could be, so they explain
away as much as possible. An intercept that survived this would have survived
something worth surviving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.attribution import (
    FACTOR_SPEC, attribute, build_factor_returns,
)

H, STEP = 63, 21


def panel(n_dates=20, n_names=200, seed=0, alpha_per_period=0.0, mom_beta=1.0):
    """A panel where the truth is known: forward return is a momentum premium
    plus optional genuine alpha in the names the model would pick."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        mom = rng.normal(0, 1, n_names)
        noise = rng.normal(0, 0.08, n_names)
        fwd = mom_beta * 0.01 * mom + noise
        if alpha_per_period:
            # Alpha concentrated in the top decile of an independent score.
            extra = rng.normal(0, 1, n_names)
            fwd = fwd + alpha_per_period * (extra > np.quantile(extra, 0.9))
        rows.append(pd.DataFrame({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=21 * d),
            "symbol": [f"S{i}" for i in range(n_names)],
            "label": fwd,
            "mom_6_1_r": mom,
            "book_to_price_r": rng.normal(0, 1, n_names),
            "earnings_yield_r": rng.normal(0, 1, n_names),
            "downside_vol_r": rng.normal(0, 1, n_names),
            "amihud_r": rng.normal(0, 1, n_names),
            "turnover_ratio_r": rng.normal(0, 1, n_names),
        }))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------ construction
def test_factor_returns_are_built_for_every_usable_date():
    f = build_factor_returns(panel(n_dates=12), forward_col="label")
    assert len(f) == 12
    for name in FACTOR_SPEC:
        assert name in f.columns


def test_a_date_with_too_few_names_is_skipped_not_estimated():
    p = panel(n_dates=6)
    thin = p[p["date"] == p["date"].max()].head(10)
    p = pd.concat([p[p["date"] != p["date"].max()], thin])
    assert len(build_factor_returns(p, forward_col="label")) == 5


def test_the_momentum_factor_recovers_a_momentum_premium():
    """If the construction is right, a panel built with a momentum premium
    must produce a positive momentum factor."""
    f = build_factor_returns(panel(n_dates=25, mom_beta=1.0), forward_col="label")
    assert f["MOM"].mean() > 0


def test_low_volatility_is_long_the_low_leg():
    """Every risk factor is favourable when LOW. Orienting one naively would
    flip the sign of its entire loading."""
    assert FACTOR_SPEC["LOWVOL"][1] is False
    assert FACTOR_SPEC["SIZE"][1] is False


# -------------------------------------------------------------- regression
def test_a_pure_factor_portfolio_shows_no_alpha():
    """The case that matters. A strategy that is nothing but momentum exposure
    must not be credited with alpha."""
    p = panel(n_dates=25, mom_beta=1.0, seed=1)
    f = build_factor_returns(p, forward_col="label")
    a = attribute(f["MOM"].tolist(), f, horizon_sessions=H, step_sessions=STEP,
                  factor_names=["MOM"])
    assert abs(a.alpha_t_adjusted) < 2.0
    assert a.r_squared > 0.9


def test_genuine_alpha_is_detected_when_it_is_there():
    p = panel(n_dates=30, mom_beta=1.0, alpha_per_period=0.04, seed=2)
    f = build_factor_returns(p, forward_col="label")
    # The strategy: hold the names carrying the injected alpha.
    per_date = p[p["label"] > p.groupby("date")["label"].transform(
        lambda s: s.quantile(0.9))].groupby("date")["label"].mean()
    mkt = p.groupby("date")["label"].mean()
    excess = (per_date - mkt).reindex(f["date"]).tolist()
    a = attribute(excess, f, horizon_sessions=H, step_sessions=STEP,
                  factor_names=["MOM"])
    assert a.alpha_per_period > 0
    assert a.alpha_survives


def test_the_overlap_correction_is_applied_to_alpha_too():
    """The same inflation applies here as anywhere else. An alpha t that
    ignored it would be the audit's own error repeated inside its fix."""
    p = panel(n_dates=30, seed=3)
    f = build_factor_returns(p, forward_col="label")
    a = attribute(f["MOM"].tolist(), f, horizon_sessions=H, step_sessions=STEP,
                  factor_names=["MOM", "VALUE"])
    assert abs(a.alpha_t_adjusted) < abs(a.alpha_t)


def test_a_regression_with_no_degrees_of_freedom_refuses():
    p = panel(n_dates=7)
    f = build_factor_returns(p, forward_col="label")
    with pytest.raises(ValueError, match="degrees of freedom"):
        attribute(f["MOM"].tolist(), f, horizon_sessions=H, step_sessions=STEP,
                  factor_names=list(FACTOR_SPEC))


def test_a_thin_regression_says_so_rather_than_reporting_silently():
    p = panel(n_dates=15)
    f = build_factor_returns(p, forward_col="label")
    a = attribute(f["MOM"].tolist(), f, horizon_sessions=H, step_sessions=STEP,
                  factor_names=list(FACTOR_SPEC))
    assert any("poorly determined" in n for n in a.notes)


def test_loadings_come_back_ordered_by_what_they_explain():
    p = panel(n_dates=30, seed=4)
    f = build_factor_returns(p, forward_col="label")
    a = attribute(f["MOM"].tolist(), f, horizon_sessions=H, step_sessions=STEP)
    contribs = [abs(l.contribution) for l in a.loads]
    assert contribs == sorted(contribs, reverse=True)
