"""Guards so the defects the repair fixed cannot come back silently.

Each of these would have caught a real defect on the day it shipped, and each
one failed to exist:

  A  the model cache did not record WHAT THE MODEL WAS FITTED AGAINST, so a
     label change kept scoring on stale coefficients for up to 42 sessions
     while every run looked normal;
  C  nothing checked that a family's members point the same way, so `risk`
     averaged beta (higher = riskier) with max_dd (higher = SHALLOWER, so
     safer) under a common sign and cancelled the axis it was built to measure.

Guard B -- softening the |t| >= 2 cliff to a continuous weight -- is an
ESTIMATOR change rather than a correctness fix, so it is implemented as an
explicitly selectable option and measured through `research estimator` as a
recorded trial, not asserted here. See `famamacbeth.SIGNIFICANCE_FLOOR`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.features import crossmodel as cm
from prosignal.features.crosssec import _features_at
from prosignal.features.famamacbeth import FMResult, gated_shrink
from prosignal.features.exits import ExitRules
from prosignal.features.labels import BarrierSpec


# ------------------------------------------------------------------ Guard A
class TestTheCacheKnowsWhatItWasFittedAgainst:
    """`load_cached` checked the fit date, the feature-column set and the
    estimator. None of those move when the LABEL changes, and the label was not
    stored in the blob at all.

    Splitting a family invalidates the cache on its own, through the
    feature-column check. Changing the label does not, and changing the horizon
    does not -- so this is the gap the label repair had to be walked around by
    hand, by archiving the live model.
    """

    @staticmethod
    def _model():
        m = cm.CrossSectionalModel(
            coef={c: 0.01 for c in cm.FAMILY_COLUMNS},
            n_train=10_000, train_end=dt.date(2026, 2, 11),
            features=list(cm.FAMILY_COLUMNS),
        )
        m.mu = np.zeros(len(cm.FAMILY_COLUMNS))
        m.sd = np.ones(len(cm.FAMILY_COLUMNS))
        m.intercept = 0.0
        m.estimator = "fama_macbeth"
        return m

    def _write(self, path, label):
        model = self._model()
        model.label = label
        cm.save_cache(path, model, dt.date(2026, 8, 25))

    def test_the_same_label_loads(self, tmp_path):
        fp = cm.label_fingerprint(63)
        path = tmp_path / "crosssec_model.json"
        self._write(path, fp)
        got = cm.load_cached(path, dt.date(2026, 8, 26), 21,
                             estimator="fama_macbeth", label=fp)
        assert got is not None

    def test_turning_the_barrier_label_off_refuses_the_blob(self, tmp_path):
        """The exact change the repair made."""
        barrier = cm.label_fingerprint(63, exit_rules=ExitRules())
        forward = cm.label_fingerprint(63)
        assert barrier != forward
        path = tmp_path / "crosssec_model.json"
        self._write(path, barrier)
        assert cm.load_cached(path, dt.date(2026, 8, 26), 21,
                              estimator="fama_macbeth", label=forward) is None

    def test_a_changed_horizon_refuses_the_blob(self, tmp_path):
        """The other change that moves every coefficient and no other check
        would have caught."""
        path = tmp_path / "crosssec_model.json"
        self._write(path, cm.label_fingerprint(63))
        assert cm.load_cached(path, dt.date(2026, 8, 26), 21,
                              estimator="fama_macbeth",
                              label=cm.label_fingerprint(21)) is None

    def test_a_changed_stop_multiple_refuses_the_blob(self, tmp_path):
        """The label's geometry IS the traded geometry, so moving the stop
        moves what the ranker was fitted on."""
        path = tmp_path / "crosssec_model.json"
        self._write(path, cm.label_fingerprint(63, exit_rules=ExitRules()))
        wider = ExitRules(stop_atr_multiple=3.5)
        assert cm.load_cached(path, dt.date(2026, 8, 26), 21,
                              estimator="fama_macbeth",
                              label=cm.label_fingerprint(63, exit_rules=wider)) is None

    def test_a_blob_written_before_the_field_existed_is_refused(self, tmp_path):
        """Treated as a mismatch rather than a pass. The whole failure mode is
        a stale blob that looks valid, so refitting once on upgrade is the
        cheap side of the trade."""
        path = tmp_path / "crosssec_model.json"
        self._write(path, None)              # pre-fingerprint blob
        assert cm.load_cached(path, dt.date(2026, 8, 26), 21,
                              estimator="fama_macbeth",
                              label=cm.label_fingerprint(63)) is None

    def test_the_sigma_and_engine_geometries_are_distinguishable(self, tmp_path):
        spec = BarrierSpec(upper=1.0, lower=0.75, horizon=63, vol_window=60)
        assert (cm.label_fingerprint(63, barriers=spec)
                != cm.label_fingerprint(63, exit_rules=ExitRules()))


# ------------------------------------------------------------------ Guard C
def _heterogeneous_panel(n=700, k=60, seed=17):
    """A cross-section that varies in BOTH beta and idiosyncratic volatility.

    Both have to vary, and roughly together, or the fixture tests nothing. An
    earlier version held idiosyncratic volatility constant across names, which
    left `idio_vol` near-constant and its correlation with `max5_21` pure
    sampling noise -- the guard then fired on the fixture rather than on the
    engine. Varying them together is also the realistic shape: a high-beta name
    in this market is usually the more volatile one outright.

    That gives the two properties the guard needs. The lottery moments all rise
    with total volatility, so they cohere. Drawdown depth deepens with it while
    beta rises with it, so `beta_120` and `max_dd_120` anticorrelate.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    market = rng.normal(0.0004, 0.010, n)
    cols, series = [], []
    for i in range(k):
        t = i / (k - 1)
        beta = 0.3 + 1.7 * t                       # 0.3 .. 2.0
        idio_sd = 0.004 + 0.016 * t                # 0.4% .. 2.0% daily
        r = beta * market + rng.normal(0.0, idio_sd, n)
        cols.append(f"S{i:02d}")
        series.append(100 * np.cumprod(1 + r))
    close = pd.DataFrame(dict(zip(cols, series)), index=idx)
    turnover = pd.DataFrame(rng.uniform(1e7, 5e7, (n, k)), index=idx, columns=cols)
    bench = close.mean(axis=1).pct_change(fill_method=None).to_numpy("float64")
    return close, turnover, bench


def test_no_family_averages_two_members_that_point_opposite_ways():
    """THE TEST THAT WOULD HAVE CAUGHT THE `risk` DEFECT ON DAY ONE.

    A family is averaged into one column and given one coefficient, which only
    means something if its members agree about which direction is which. `risk`
    averaged `beta_120` (higher = riskier) with `max_dd_120`, which is a
    NEGATIVE number whose higher values are SHALLOWER drawdowns and therefore
    safer. Both entered with a + sign, they correlate -0.42 within date on the
    real panel, and the average cancelled the common axis: beta alone t -3.67
    and max_dd alone t +4.69 became a composite at t -0.93, which the
    significance gate then discarded for being insignificant.

    Measured here on a synthetic cross-section spanning beta 0.3 to 2.0, where
    the anticorrelation is structural rather than incidental -- a high-beta
    name draws down harder, so its max_dd is more negative.

    A member whose natural sign is opposite is legitimate, but it must be
    declared in NEGATED_IN_FAMILY and flipped before averaging, which is how
    the quality family carries accruals, asset growth and net issuance.
    """
    close, turnover, bench = _heterogeneous_panel()
    feats = _features_at(close, turnover, len(close) - 1, bench)
    ranks = feats.rank(pct=True)

    offenders = []
    for family, members in cm.FAMILIES.items():
        present = [m for m in members if m.removesuffix("_r") in ranks.columns]
        if len(present) < 2:
            continue
        block = ranks[[m.removesuffix("_r") for m in present]].copy()
        # Flip exactly as build_families does, so a declared negation passes.
        for m in present:
            if m in cm.NEGATED_IN_FAMILY:
                block[m.removesuffix("_r")] *= -1.0
        corr = block.corr()
        for a in corr.columns:
            for b in corr.columns:
                if a >= b:
                    continue
                rho = corr.loc[a, b]
                if np.isfinite(rho) and rho < -0.10:
                    offenders.append(f"{family}: {a} vs {b} rho {rho:+.3f}")

    assert not offenders, (
        "these family members anticorrelate, so averaging them cancels the "
        "axis the family is supposed to measure. Split them into separate "
        "themes, or declare the flipped one in NEGATED_IN_FAMILY:\n  "
        + "\n  ".join(offenders))


def test_beta_and_drawdown_would_have_failed_that_guard_together():
    """The guard is only worth having if it actually fires on the old shape.

    Rebuilds the retired `risk` family and asserts the audit's finding
    reproduces: the two members anticorrelate, so the guard above is testing
    something real rather than passing vacuously.
    """
    close, turnover, bench = _heterogeneous_panel()
    ranks = _features_at(close, turnover, len(close) - 1, bench).rank(pct=True)
    rho = ranks["beta_120"].corr(ranks["max_dd_120"])
    assert rho < -0.10, (
        f"expected beta and drawdown depth to anticorrelate, got {rho:+.3f}; "
        "if this no longer holds the guard above is vacuous")


# ------------------------------------------------------------------ Guard B
class TestTheSignificanceCliffCanBeSmoothedButIsNot:
    """The cliff is a real weakness and the smoothing is deliberately unused.

    `gated_shrink` zeroes any theme below |t| 2.0 and keeps the rest near full
    strength, which makes a traded coefficient a STEP FUNCTION of a noisy
    statistic. `risk` measured t +1.86 on the live fit and +2.45 on the 69-date
    rebuild -- one theme worth either exactly nothing or nearly its whole
    coefficient, decided by 0.6 of a t-statistic.

    The continuous form is available and OFF. Switching it on moves live
    coefficients, so it is an estimator change and therefore a trial, and the
    DSR is already charging 81 of those. These tests pin the mechanism and pin
    that it is not switched on quietly.
    """

    @staticmethod
    def _result(lam, se):
        cols = list(lam)
        return FMResult(
            features=cols,
            slopes=pd.DataFrame({c: [0.0, 0.0, 0.0] for c in cols}),
            lam=dict(lam), se=dict(se),
            t_stat={c: lam[c] / se[c] for c in cols}, n_dates=50)

    def test_it_ships_off(self):
        from prosignal.config.loader import load_config
        est = load_config().params.stage4_core_score.estimator
        assert est.significance_taper is False, (
            "the taper moves live coefficients; turning it on is a trial and "
            "must go through `research estimator`, not this file")
        assert est.taper_c == pytest.approx(4.0)
        assert est.taper_hard_floor == pytest.approx(1.0)

    def test_the_half_weight_point_is_exactly_the_old_cliff(self):
        """c = 4 makes |t| = 2 keep half. That is what makes the taper a
        smoothing of the shipped rule rather than a different rule."""
        r = self._result({"mom_f": 0.08}, {"mom_f": 0.04})     # t = +2.0
        plain = gated_shrink(r, floor=0.0, toward="zero")
        tapered = gated_shrink(r, floor=0.0, toward="zero", taper=True)
        assert tapered["mom_f"] == pytest.approx(0.5 * plain["mom_f"], rel=1e-9)

    def test_a_t_of_one_keeps_about_a_fifth(self):
        r = self._result({"mom_f": 0.04}, {"mom_f": 0.04})     # t = +1.0
        plain = gated_shrink(r, floor=0.0, toward="zero")
        tapered = gated_shrink(r, floor=0.0, toward="zero", taper=True)
        assert tapered["mom_f"] == pytest.approx(0.2 * plain["mom_f"], rel=1e-9)

    def test_below_the_hard_floor_is_still_exactly_zero(self):
        """A theme the window cannot measure must not steer the book at a fifth
        weight any more than at full weight."""
        r = self._result({"mom_f": 0.07, "noise_f": 0.002},
                         {"mom_f": 0.021, "noise_f": 0.05})    # t = +0.04
        out = gated_shrink(r, toward="zero", taper=True)
        assert out["noise_f"] == 0.0

    def test_the_theme_the_cliff_would_have_killed_survives_tapered(self):
        """`risk` at t +1.86 was worth exactly zero under the cliff. Under the
        taper it is worth something small, which is the substantive difference
        the trial would be measuring."""
        r = self._result({"risk_f": 0.0372}, {"risk_f": 0.02})  # t = +1.86
        assert gated_shrink(r, floor=2.0, toward="zero")["risk_f"] == 0.0
        assert abs(gated_shrink(r, toward="zero", taper=True)["risk_f"]) > 0.0
