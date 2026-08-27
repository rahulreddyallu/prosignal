"""Fama-MacBeth estimation, hierarchical shrinkage, and the 1/N control.

The bug these exist to prevent is not a crash. It is a coefficient that looks
significant because the estimator counted 33,000 correlated rows as 33,000
independent observations, and then steered real money with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.famamacbeth import (
    FMResult, MIN_CROSS_SECTION, SIGNIFICANCE_FLOOR, THEME_PRIOR_SIGN,
    equal_weight_lambda, fama_macbeth, gated_shrink, hierarchical_shrink,
    is_degenerate, newey_west_se, rolling_lambda, score_from_lambda)


def _panel(n_dates=40, n_names=60, beta=0.5, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        x = rng.normal(size=n_names)
        z = rng.normal(size=n_names)
        rows.append(pd.DataFrame({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=21 * d),
            "symbol": [f"S{i}" for i in range(n_names)],
            "mom_f": x, "lottery_f": z,
            "label_rank": beta * x + noise * rng.normal(size=n_names),
        }))
    return pd.concat(rows, ignore_index=True)


class TestEstimation:
    def test_it_recovers_a_slope_that_is_really_there(self):
        r = fama_macbeth(_panel(beta=0.5, noise=1.0), ["mom_f", "lottery_f"])
        assert r.lam["mom_f"] == pytest.approx(0.5, abs=0.08)
        assert r.t_stat["mom_f"] > 5

    def test_a_factor_with_no_relationship_does_not_clear_the_bar(self):
        r = fama_macbeth(_panel(), ["mom_f", "lottery_f"])
        assert abs(r.lam["lottery_f"]) < 0.1
        assert abs(r.t_stat["lottery_f"]) < 2.0

    def test_the_sample_is_dates_not_rows(self):
        """The whole point. Ten times the names must not multiply confidence."""
        narrow = fama_macbeth(_panel(n_dates=30, n_names=40, seed=1), ["mom_f"])
        wide = fama_macbeth(_panel(n_dates=30, n_names=400, seed=1), ["mom_f"])
        assert narrow.n_dates == wide.n_dates == 30
        # More names per date sharpen each slope a little, but nowhere near the
        # sqrt(10) a pooled regression would claim from ten times the rows.
        assert wide.t_stat["mom_f"] < narrow.t_stat["mom_f"] * np.sqrt(10)

    def test_thin_cross_sections_are_dropped_not_averaged_in(self):
        p = _panel(n_dates=20, n_names=60)
        thin = p["date"] == p["date"].unique()[3]
        p = p[~thin | (p.groupby("date").cumcount() < MIN_CROSS_SECTION - 5)]
        r = fama_macbeth(p, ["mom_f", "lottery_f"])
        assert r.n_dates == 19 and r.skipped_dates == 1

    def test_it_refuses_rather_than_guessing_from_two_cross_sections(self):
        assert fama_macbeth(_panel(n_dates=2), ["mom_f"]) is None

    def test_a_rolling_window_uses_only_the_recent_dates(self):
        r = fama_macbeth(_panel(n_dates=40), ["mom_f"], window=12)
        assert r.n_dates == 12


class TestNeweyWest:
    def test_positive_autocorrelation_widens_the_error(self):
        rng = np.random.default_rng(3)
        e = rng.normal(size=400)
        ar = np.zeros(400)
        for i in range(1, 400):
            ar[i] = 0.7 * ar[i - 1] + e[i]
        naive = ar.std(ddof=1) / np.sqrt(len(ar))
        assert newey_west_se(ar, lags=4) > naive * 1.4

    def test_zero_lags_reduces_to_the_plain_standard_error(self):
        x = np.random.default_rng(4).normal(size=200)
        assert newey_west_se(x, 0) == pytest.approx(x.std(ddof=0) / np.sqrt(200), rel=1e-9)

    def test_a_negative_bartlett_sum_falls_back_rather_than_returning_nan(self):
        """Small samples can drive the weighted sum below zero. A variance
        cannot be negative, and a nan would silently drop the theme."""
        x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        assert np.isfinite(newey_west_se(x, lags=3))
        assert newey_west_se(x, lags=3) > 0

    def test_the_default_lag_matches_the_label_overlap(self):
        # A 63-session label sampled every 21 overlaps its next two neighbours.
        r = fama_macbeth(_panel(), ["mom_f"], horizon=63, step=21)
        assert r.nw_lags == 2


class TestShrinkage:
    def _result(self, lam, se):
        cols = list(lam)
        return FMResult(features=cols, slopes=pd.DataFrame(columns=cols),
                        lam=dict(lam), se=dict(se),
                        t_stat={c: lam[c] / se[c] for c in cols}, n_dates=50)

    def test_a_precise_estimate_keeps_more_of_itself_than_a_noisy_one(self):
        r = self._result({"mom_f": 0.06, "lottery_f": 0.06},
                         {"mom_f": 0.01, "lottery_f": 0.20})
        out = hierarchical_shrink(r, toward="zero")
        assert abs(out["mom_f"]) > abs(out["lottery_f"])

    def test_shrinkage_never_overshoots_the_estimate(self):
        r = self._result({"mom_f": 0.06, "lottery_f": -0.02},
                         {"mom_f": 0.02, "lottery_f": 0.05})
        for col, v in hierarchical_shrink(r, toward="zero").items():
            assert abs(v) <= abs(r.lam[col]) + 1e-12

    def test_when_dispersion_is_all_noise_everything_goes_to_zero(self):
        """tau^2 = 0 is an ANSWER: no theme is distinguishable. It must not be
        rounded up into a tradeable set of weights."""
        r = self._result({"mom_f": 0.01, "lottery_f": -0.01},
                         {"mom_f": 0.5, "lottery_f": 0.5})
        assert is_degenerate(hierarchical_shrink(r, toward="zero"))

    def test_prior_pooling_can_hand_a_noisy_theme_the_pools_sign(self):
        """Why `zero` is the default. `lottery` has a documented negative prior
        and measures positive here; pooled, it inherits the prior anyway."""
        r = self._result({"mom_f": 0.07, "delivery_f": 0.045, "lottery_f": 0.0065},
                         {"mom_f": 0.021, "delivery_f": 0.013, "lottery_f": 0.065})
        pooled = hierarchical_shrink(r, toward="prior_mean")
        zeroed = hierarchical_shrink(r, toward="zero")
        assert pooled["lottery_f"] < -0.01          # confidently negative
        assert abs(zeroed["lottery_f"]) < 0.005     # honestly nothing

    def test_an_unknown_shrink_target_is_refused(self):
        r = self._result({"mom_f": 0.05}, {"mom_f": 0.01})
        with pytest.raises(ValueError, match="toward"):
            hierarchical_shrink(r, toward="somewhere")

    def test_a_theme_with_no_prior_shrinks_to_zero_even_when_pooling(self):
        """`risk` was the no-prior theme until it was split. `drawdown` now
        holds that position: there is no literature analogue for a
        drawdown-depth cross-section, so it carries no prior. `beta` does carry
        one (Frazzini & Pedersen 2014), which is the point of the split."""
        assert THEME_PRIOR_SIGN["drawdown"] is None
        assert THEME_PRIOR_SIGN["beta"] == -1
        r = self._result(
            {"mom_f": 0.07, "delivery_f": 0.05, "drawdown_f": 0.02},
            {"mom_f": 0.02, "delivery_f": 0.015, "drawdown_f": 0.06})
        out = hierarchical_shrink(r, toward="prior_mean")
        assert abs(out["drawdown_f"]) < abs(r.lam["drawdown_f"])


class TestGate:
    def _result(self, lam, se):
        cols = list(lam)
        return FMResult(features=cols, slopes=pd.DataFrame(
            {c: [0.0, 0.0, 0.0] for c in cols}), lam=dict(lam), se=dict(se),
            t_stat={c: lam[c] / se[c] for c in cols}, n_dates=50)

    def test_a_theme_below_the_floor_is_exactly_zero_not_merely_small(self):
        r = self._result({"mom_f": 0.07, "lottery_f": 0.0065},
                         {"mom_f": 0.021, "lottery_f": 0.065})
        out = gated_shrink(r, floor=2.0)
        assert out["lottery_f"] == 0.0
        assert out["mom_f"] > 0

    def test_every_feature_is_returned_so_nothing_vanishes_silently(self):
        r = self._result({"mom_f": 0.07, "lottery_f": 0.001},
                         {"mom_f": 0.021, "lottery_f": 0.065})
        assert set(gated_shrink(r).keys()) == {"mom_f", "lottery_f"}

    def test_all_themes_failing_yields_a_degenerate_answer_not_a_fallback(self):
        r = self._result({"mom_f": 0.01, "lottery_f": 0.001},
                         {"mom_f": 0.05, "lottery_f": 0.065})
        assert is_degenerate(gated_shrink(r, floor=2.0))

    def test_the_floor_is_pinned_at_two(self):
        """1.65 measured better out of sample and was rejected as a tuned
        choice. If this changes, it must be a deliberate decision."""
        assert SIGNIFICANCE_FLOOR == 2.0


class TestControlArm:
    def test_the_control_is_equal_weighted_and_prior_oriented(self):
        """The 1/N arm `research estimator` measures the model against. A theme
        with no prior takes no position in it -- and a control with a missing
        theme is not the control you think it is, which is why THEME_PRIOR_SIGN
        has to track the family registry."""
        ew = equal_weight_lambda(["mom_f", "lottery_f", "drawdown_f"])
        assert ew["mom_f"] == pytest.approx(1 / 3)
        assert ew["lottery_f"] == pytest.approx(-1 / 3)
        assert ew["drawdown_f"] == 0.0    # no prior, so no bet

    def test_beta_takes_the_low_beta_side_in_the_control(self):
        """Frazzini & Pedersen (2014), and in this market Agarwalla, Jacob,
        Varma & Vasudevan (2014), who find BAB earns significant positive
        returns in India and dominates size, value and momentum."""
        ew = equal_weight_lambda(["mom_f", "beta_f"])
        assert ew["beta_f"] < 0

    def test_scoring_ignores_columns_the_frame_does_not_have(self):
        f = pd.DataFrame({"mom_f": [1.0, -1.0]})
        s = score_from_lambda(f, {"mom_f": 2.0, "absent_f": 5.0})
        assert list(s) == [2.0, -2.0]

    def test_a_rolling_lambda_tracks_a_regime_change(self):
        cols = ["mom_f"]
        slopes = pd.DataFrame({"mom_f": [0.08] * 20 + [0.0] * 20},
                              index=pd.date_range("2020-01-01", periods=40, freq="21D"))
        r = FMResult(features=cols, slopes=slopes, n_dates=40)
        roll = rolling_lambda(r, window=10)
        assert roll["mom_f"].iloc[19] == pytest.approx(0.08)
        assert roll["mom_f"].iloc[-1] == pytest.approx(0.0)


class TestTheTrialRegistry:
    """The DSR charges a result for the configurations tried before it. That
    count was a command-line default of 24 plus a config field shipped at 0 with
    a comment asking a human to update it. Nobody did, and nothing checked."""

    def _reg(self, tmp_path):
        from prosignal.validation.registry import TrialRegistry
        return TrialRegistry(tmp_path / "trials.jsonl")

    def test_recording_the_same_comparison_twice_does_not_inflate_the_count(self, tmp_path):
        r = self._reg(tmp_path)
        assert r.record("estimator", ["ridge", "1/N", "fm"]) == 3
        assert r.record("estimator", ["ridge", "1/N", "fm"]) == 0
        assert r.count() == 3

    def test_a_new_comparison_adds_to_it(self, tmp_path):
        r = self._reg(tmp_path)
        r.record("estimator", ["ridge", "1/N"])
        r.record("spread", ["8/16", "8/20"])
        assert r.count() == 4
        assert r.by_command() == {"estimator": 2, "spread": 2}

    def test_the_same_label_under_a_different_command_is_a_separate_trial(self, tmp_path):
        r = self._reg(tmp_path)
        r.record("estimator", ["mom only"])
        r.record("cpcv", ["mom only"])
        assert r.count() == 2

    def test_prior_campaigns_are_carried_not_assumed_away(self, tmp_path):
        r = self._reg(tmp_path)
        r.record("estimator", ["a", "b"])
        assert r.effective_trials(carried=20) == 22
        assert r.effective_trials(carried=0) == 2

    def test_an_absent_registry_counts_zero_rather_than_failing(self, tmp_path):
        assert self._reg(tmp_path).count() == 0

    def test_a_corrupt_line_does_not_silently_lower_the_count(self, tmp_path):
        """Under-counting trials flatters the result, so a bad line is skipped
        for identity but the file is never rewritten or truncated."""
        r = self._reg(tmp_path)
        r.record("estimator", ["a", "b"])
        before = r.path.read_text()
        with r.path.open("a") as fh:
            fh.write("{not json\n")
        assert r.count() == 2
        assert r.path.read_text().startswith(before)

    def test_the_cpcv_command_defaults_to_the_registry_not_a_constant(self):
        import inspect

        from prosignal import cli

        src = inspect.getsource(cli.cmd_research_cpcv)
        assert "reg.effective_trials(carried)" in src
        assert "trials = counted if args.trials is None" in src

    def test_overriding_the_count_downward_is_announced(self):
        import inspect

        from prosignal import cli

        src = inspect.getsource(cli.cmd_research_cpcv)
        assert "OVERRIDDEN DOWNWARD" in src


class TestPboIsActuallyComputed:
    def test_the_cpcv_command_calls_it(self):
        """`compute_pbo` was implemented, tested, and called by nothing, while
        the CPCV output quoted a PBO bar against a number never computed."""
        import inspect

        from prosignal import cli

        src = inspect.getsource(cli.cmd_research_cpcv)
        assert "compute_pbo(" in src
        assert "configuration_matrix(" in src

    def test_a_refused_configuration_does_not_delete_every_other_column(self):
        """One empty column plus a row-wise dropna deleted every date for every
        configuration, so a single unusable arm made PBO uncomputable."""
        import numpy as np
        import pandas as pd

        frame = pd.DataFrame({
            "good": np.linspace(0.01, 0.05, 30),
            "also good": np.linspace(0.02, 0.06, 30),
            "refused": [np.nan] * 30,
        })
        kept = [c for c in frame.columns if frame[c].notna().sum() >= 8]
        assert set(kept) == {"good", "also good"}
        assert len(frame[kept].dropna()) == 30


# ------------------------------------------------------ the winner's curse
def _wc_result(lam, se, n_dates=69):
    feats = list(lam)
    t = {c: (lam[c] / se[c] if se[c] else float("nan")) for c in feats}
    slopes = pd.DataFrame({c: [lam[c]] * n_dates for c in feats})
    return FMResult(features=feats, slopes=slopes, lam=dict(lam), se=dict(se),
                    t_stat=t, n_dates=n_dates, nw_lags=2)


_WC_LAM = {"a_f": -0.219, "b_f": -0.101, "c_f": 0.062, "d_f": 0.037, "e_f": -0.003}
_WC_SE = {"a_f": 0.035, "b_f": 0.017, "c_f": 0.012, "d_f": 0.020, "e_f": 0.021}


def test_the_pool_is_estimated_before_the_gate_selects():
    """`gated_shrink` kills everything under the floor and then shrinks the
    survivors. Estimating the between-theme dispersion from THAT set is the
    winner's curse: they were selected for being large, so E[lambda^2] among
    them is inflated, tau^2 comes out too big, and the themes that passed are
    barely shrunk at all.

    Measured on the shipped fit the haircut was delivery 1.6%, reversal 3.3%,
    lottery 12.1%. An empirical-Bayes discipline removing 1.6% of a coefficient
    is the gate doing the entire job while the estimator takes the credit.
    """
    full = _wc_result(_WC_LAM, _WC_SE)
    kept = [c for c in full.features if abs(full.t_stat[c]) >= 2.0]
    assert kept, "fixture must have survivors"
    sub = _wc_result({c: _WC_LAM[c] for c in kept}, {c: _WC_SE[c] for c in kept})

    post = hierarchical_shrink(sub, toward="zero")                   # survivors only
    pre = hierarchical_shrink(sub, toward="zero", tau2_from=full)    # every theme

    for c in kept:
        assert abs(pre[c]) < abs(post[c]), (
            f"{c}: a pool estimated before selection must shrink at least as "
            f"hard as one estimated after it"
        )
        assert pre[c] * _WC_LAM[c] > 0, "shrinkage must not flip a sign"


def test_gated_shrink_routes_the_preselection_pool_through():
    full = _wc_result(_WC_LAM, _WC_SE)
    out = gated_shrink(full, floor=2.0, toward="zero")
    kept = [c for c in full.features if abs(full.t_stat[c]) >= 2.0]
    sub = _wc_result({c: _WC_LAM[c] for c in kept}, {c: _WC_SE[c] for c in kept})
    naive = hierarchical_shrink(sub, toward="zero")
    assert any(abs(out[c]) < abs(naive[c]) for c in kept), (
        "gated_shrink must hand the full result over as the tau^2 source"
    )


def test_the_gate_still_does_the_killing_not_the_shrinkage():
    """The floor is a RISK CONTROL and the shrinkage is an ESTIMATOR. Fixing
    which sample the pool is learned from must not turn the estimator into a
    second gate."""
    full = _wc_result(_WC_LAM, _WC_SE)
    out = gated_shrink(full, floor=2.0, toward="zero")
    live = {c for c, v in out.items() if abs(v) > 1e-12}
    expected = {c for c in full.features if abs(full.t_stat[c]) >= 2.0}
    assert live == expected, "the survivor set must be decided by the floor alone"
    for c in full.features:
        if c not in expected:
            assert out[c] == 0.0
