"""Statistics that hold up at the sample size this project actually has.

Every performance figure here is a mean over OVERLAPPING observations: a
63-session label sampled every 21 sessions, so each shares two thirds of its
window with the next. Measured against simulated noise, the naive t rejects a
true null 32% of the time at a nominal 5%. That is the single largest
statistical error available to this project and it is invisible in the output.

The tests below are calibration tests, not unit tests. They ask what a
statistic DOES on data where the answer is known, because a correction that is
merely present is worth nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from prosignal.validation.significance import (
    BOOTSTRAP_MIN_N, analytic_vif, newey_west_t, overlap_lag,
    stationary_bootstrap, summarise,
)

H, STEP = 63, 21


def overlapping(daily: np.ndarray, horizon: int = H, step: int = STEP) -> np.ndarray:
    """Form the same overlapping windows the panel does."""
    return np.array([daily[i:i + horizon].sum()
                     for i in range(0, len(daily) - horizon, step)])


# ------------------------------------------------------------------ overlap
def test_overlap_lag_is_read_off_the_sampling_scheme():
    assert overlap_lag(63, 21) == 2
    assert overlap_lag(63, 63) == 0      # sampled at the horizon: no overlap
    assert overlap_lag(63, 5) == 12


def test_analytic_vif_matches_the_closed_form_at_large_n():
    """rho_k = (m-k)/m gives VIF = 1 + 2(2/3 + 1/3) = 3 for m = 3."""
    assert analytic_vif(63, 21, n=10_000) == pytest.approx(3.0, abs=0.01)


def test_no_overlap_means_no_inflation():
    assert analytic_vif(63, 63, n=1000) == pytest.approx(1.0)


# -------------------------------------------------------------- calibration
def test_the_naive_t_is_badly_wrong_on_overlapping_noise():
    """The reason this module exists. Not a hypothetical."""
    rng = np.random.default_rng(1)
    rejects = sum(
        abs(newey_west_t(overlapping(rng.normal(0, 0.01, 15 * STEP + H)),
                         horizon_sessions=H, step_sessions=STEP).naive_t) > 2.145
        for _ in range(200)
    )
    assert rejects / 200 > 0.20, (
        "the naive t should over-reject badly here; if it no longer does, the "
        "fixture stopped producing overlapping data"
    )


def test_the_corrected_t_gets_much_closer_to_nominal():
    rng = np.random.default_rng(2)
    rejects = sum(
        abs(newey_west_t(overlapping(rng.normal(0, 0.01, 15 * STEP + H)),
                         horizon_sessions=H, step_sessions=STEP).adjusted_t) > 2.145
        for _ in range(200)
    )
    assert rejects / 200 < 0.16, f"corrected t still rejects {rejects/200:.0%}"


def test_the_correction_does_not_destroy_power():
    """A conservative statistic that never detects anything is not useful."""
    rng = np.random.default_rng(3)
    found = sum(
        newey_west_t(overlapping(rng.normal(0.0012, 0.01, 15 * STEP + H)),
                     horizon_sessions=H, step_sessions=STEP).adjusted_t > 2.145
        for _ in range(150)
    )
    assert found / 150 > 0.45


def test_the_analytic_inflation_beats_estimating_it_at_this_n():
    """Newey-West estimates the inflation from the data and, at fifteen
    observations, recovers about 1.74 where the truth is 3.00. The sampling
    scheme is known, so estimating it spends degrees of freedom to get a worse
    answer."""
    rng = np.random.default_rng(4)
    est = [newey_west_t(overlapping(rng.normal(0, 0.01, 15 * STEP + H)),
                        horizon_sessions=H, step_sessions=STEP,
                        use_analytic_vif=False).vif for _ in range(150)]
    assert np.mean(est) < 2.4, "the estimator is supposed to be biased low here"
    assert analytic_vif(H, STEP, 15) > np.mean(est)


# --------------------------------------------------------------- bootstrap
def test_the_bootstrap_refuses_to_claim_significance_below_its_calibration():
    """Measured on overlapping noise, its 95% interval excludes zero 25-30% of
    the time at n=15 for every block length from 2 to 8. It is reported, and
    flagged, rather than quietly used as a test."""
    rng = np.random.default_rng(5)
    b = stationary_bootstrap(overlapping(rng.normal(0.02, 0.01, 15 * STEP + H)),
                             horizon_sessions=H, step_sessions=STEP, draws=800)
    assert b.uncalibrated is True
    assert b.is_evidence is False, "an uncalibrated interval is not evidence"


def test_the_bootstrap_is_usable_once_there_are_enough_observations():
    rng = np.random.default_rng(6)
    b = stationary_bootstrap(overlapping(rng.normal(0.0, 0.01, 60 * STEP + H)),
                             horizon_sessions=H, step_sessions=STEP, draws=800)
    assert b.uncalibrated is False
    assert BOOTSTRAP_MIN_N == 30


def test_the_bootstrap_is_deterministic_for_a_given_seed():
    """A significance number that moves between runs is not a number."""
    x = list(np.random.default_rng(7).normal(0.03, 0.04, 40))
    a = stationary_bootstrap(x, mean_block=3, draws=500)
    b = stationary_bootstrap(x, mean_block=3, draws=500)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


# ----------------------------------------------------------------- guards
def test_a_series_too_short_to_say_anything_raises():
    with pytest.raises(ValueError, match="at least 3"):
        newey_west_t([0.1, 0.2], horizon_sessions=H, step_sessions=STEP)


def test_a_flat_series_raises_rather_than_returning_infinity():
    with pytest.raises(ValueError, match="zero variance"):
        newey_west_t([0.05] * 20, horizon_sessions=H, step_sessions=STEP)


def test_the_lag_must_come_from_somewhere():
    with pytest.raises(ValueError, match="pass lag"):
        newey_west_t([0.1, 0.2, 0.3, 0.4])


def test_summarise_names_the_bar_it_is_judged_against():
    rng = np.random.default_rng(8)
    _, _, verdict = summarise(list(rng.normal(0.03, 0.04, 15)),
                              horizon_sessions=H, step_sessions=STEP, draws=500)
    assert "MISSES" in verdict or "CLEARS" in verdict
    assert "NOT CALIBRATED" in verdict, "n=15 must carry the bootstrap warning"
