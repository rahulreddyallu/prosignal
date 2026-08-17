"""CPCV, PBO and DSR.

These tests check the properties that make the numbers *mean* something:
no leakage across the purge boundary, PBO near 1 when configurations are pure
noise, and DSR collapsing as the trial count rises. A CPCV implementation that
quietly leaks produces optimistic results for a mechanical reason, and you
would believe them.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from prosignal.core.errors import ConfigError
from prosignal.validation import (
    CombinatorialPurgedCV,
    compute_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


# =============================================================================
# normal distribution helpers
# =============================================================================


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-4)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-4)


def test_norm_ppf_is_inverse_of_cdf():
    for p in (0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_known_values():
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# CPCV
# =============================================================================


def test_split_count_is_n_choose_k():
    cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2)
    assert cv.n_splits == comb(10, 2) == 45
    assert len(list(cv.split(1000))) == 45


def test_backtest_path_count():
    cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2)
    assert cv.paths_per_observation() == comb(9, 1) == 9


def test_train_and_test_never_overlap():
    cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2, label_horizon=5, embargo=5)
    for sp in cv.split(400):
        assert np.intersect1d(sp.train_idx, sp.test_idx).size == 0


def test_purge_removes_label_overlap():
    """A training row whose forward label reaches into the test block must go."""
    horizon = 10
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=1, label_horizon=horizon, embargo=0)
    n = 100
    for sp in cv.split(n):
        test_start = int(sp.test_idx.min())
        before = sp.train_idx[sp.train_idx < test_start]
        if before.size:
            # The closest surviving training row must end its label window
            # strictly before the test block begins.
            assert before.max() + horizon < test_start


def test_embargo_removes_rows_after_the_block():
    embargo = 7
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=1, label_horizon=0, embargo=embargo)
    for sp in cv.split(100):
        test_end = int(sp.test_idx.max())
        after = sp.train_idx[sp.train_idx > test_end]
        if after.size:
            assert after.min() > test_end + embargo


def test_zero_purge_and_embargo_keeps_everything_else():
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=1, label_horizon=0, embargo=0)
    for sp in cv.split(100):
        assert sp.n_train + sp.n_test == 100
        assert sp.purged_count == 0
        assert sp.embargoed_count == 0


def test_purging_shrinks_the_training_set():
    strict = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_horizon=21, embargo=21)
    loose = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_horizon=0, embargo=0)
    strict_sizes = [s.n_train for s in strict.split(600)]
    loose_sizes = [s.n_train for s in loose.split(600)]
    assert sum(strict_sizes) < sum(loose_sizes)


def test_adjacent_test_groups_merge_into_one_block():
    """Groups 0 and 1 chosen together form a single contiguous test block, so
    only one embargo region should follow, not two."""
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2, label_horizon=0, embargo=5)
    splits = {sp.test_groups: sp for sp in cv.split(200)}
    adjacent = splits[(0, 1)]
    assert np.all(np.diff(adjacent.test_idx) == 1)


def test_every_observation_is_tested_somewhere():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    covered = set()
    for sp in cv.split(300):
        covered.update(sp.test_idx.tolist())
    assert covered == set(range(300))


def test_invalid_configurations_are_rejected():
    with pytest.raises(ConfigError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=5)
    with pytest.raises(ConfigError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=0)
    with pytest.raises(ConfigError):
        list(CombinatorialPurgedCV(n_groups=10, n_test_groups=2).split(5))


def test_describe_reports_geometry():
    cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, label_horizon=21, embargo=21)
    info = cv.describe(2000)
    assert info["n_splits"] == 45
    assert info["backtest_paths"] == 9
    assert info["total_purged"] > 0
    assert info["total_embargoed"] > 0
    assert info["min_train_size"] > 0


def test_split_dates_aligns_with_indices():
    import datetime as dt

    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(100)]
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=1, label_horizon=0, embargo=0)
    for sp, train_dates, test_dates in cv.split_dates(dates):
        assert len(train_dates) == sp.n_train
        assert len(test_dates) == sp.n_test
        assert test_dates[0] == dates[int(sp.test_idx.min())]


# =============================================================================
# Sharpe / PSR / DSR
# =============================================================================


def test_sharpe_ratio_basics():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 2000)
    sr = sharpe_ratio(r)
    assert sr == pytest.approx(0.1, abs=0.05)
    # Annualisation is opt-in and scales by sqrt(periods).
    assert sharpe_ratio(r, periods_per_year=252) == pytest.approx(sr * np.sqrt(252), rel=1e-9)


def test_sharpe_of_constant_series_is_zero():
    assert sharpe_ratio([0.01] * 50) == 0.0
    assert sharpe_ratio([]) == 0.0


def test_psr_rises_with_track_record_length():
    rng = np.random.default_rng(1)
    short = rng.normal(0.001, 0.01, 60)
    long = np.concatenate([short, rng.normal(0.001, 0.01, 2000)])
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(10, 0.25)
    b = expected_max_sharpe(1000, 0.25)
    assert 0 < a < b
    assert expected_max_sharpe(1, 0.25) == 0.0


def test_dsr_collapses_as_trial_count_rises():
    """The same track record must look worse once you admit how much you searched."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0015, 0.01, 1000)
    few = deflated_sharpe_ratio(r, n_trials=1, trial_sharpes=[0.15])
    many = deflated_sharpe_ratio(r, n_trials=5000, trial_sharpes=list(rng.normal(0, 0.5, 200)))
    assert few.deflated_sr > many.deflated_sr
    assert many.benchmark_sr > few.benchmark_sr


def test_dsr_on_a_worthless_strategy_fails():
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.01, 800)
    res = deflated_sharpe_ratio(noise, n_trials=432, trial_sharpes=list(rng.normal(0, 0.4, 100)))
    assert not res.passes
    assert "do not search further" in res.interpretation


def test_dsr_handles_short_series_without_pretending():
    res = deflated_sharpe_ratio([0.01, 0.02], n_trials=10)
    assert res.deflated_sr == 0.0
    assert "insufficient" in res.interpretation


# =============================================================================
# PBO
# =============================================================================


def test_pbo_is_high_when_every_configuration_is_noise():
    """With no real edge, the in-sample winner should land below the OOS median
    about half the time or worse."""
    rng = np.random.default_rng(4)
    M = rng.normal(0.0, 0.01, size=(600, 20))
    res = compute_pbo(M, n_splits=10)
    assert res.pbo > 0.35
    assert res.n_configurations == 20
    assert res.n_combinations == comb(10, 5)


def test_pbo_is_low_when_one_configuration_genuinely_dominates():
    rng = np.random.default_rng(5)
    M = rng.normal(0.0, 0.01, size=(600, 10))
    M[:, 3] += 0.004  # a persistent, real edge in one configuration
    res = compute_pbo(M, n_splits=10)
    assert res.pbo < 0.1
    assert res.median_oos_rank_of_selected > 0.8


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compute_pbo(np.zeros((100, 1)), n_splits=10)
    with pytest.raises(ValueError):
        compute_pbo(np.zeros((100, 5)), n_splits=7)  # odd split count
    with pytest.raises(ValueError):
        compute_pbo(np.zeros((5, 5)), n_splits=10)  # too few observations


def test_pbo_interpretation_matches_the_number():
    rng = np.random.default_rng(6)
    M = rng.normal(0.0, 0.01, size=(400, 12))
    M[:, 0] += 0.005
    res = compute_pbo(M, n_splits=8)
    if res.pbo <= 0.2:
        assert "Consistent with a real effect" in res.interpretation
    elif res.pbo > 0.5:
        assert "Simplify the model" in res.interpretation


# =============================================================================
# integration with the shipped config
# =============================================================================


def test_shipped_cpcv_config_is_internally_consistent(cfg):
    v = cfg.params.validation
    cv = CombinatorialPurgedCV(
        n_groups=int(v.cpcv.n_groups.value),
        n_test_groups=int(v.cpcv.n_test_groups.value),
        label_horizon=int(v.label.forward_return_sessions.value),
        embargo=int(v.cpcv.embargo_sessions.value),
    )
    assert cv.n_splits == cfg.params.search_space_report()["cpcv_paths"]
    # Ten years of sessions must still leave a usable training set per split.
    info = cv.describe(2500)
    assert info["min_train_size"] > 1000


def test_search_budget_keeps_the_dsr_penalty_payable(cfg):
    """The whole point of the tier cap: the trial count must stay clearable."""
    report = cfg.params.search_space_report()
    trials = report["effective_trials_if_swept"]
    rng = np.random.default_rng(7)
    # A genuinely good strategy: ~1.5 annualised Sharpe over 4 years.
    good = rng.normal(0.0009, 0.0095, 1000)
    res = deflated_sharpe_ratio(good, n_trials=trials, trial_sharpes=list(rng.normal(0, 0.3, 50)))
    assert res.n_trials == trials
    assert res.benchmark_sr > 0
    # And the same strategy under the unconstrained sweep would be hopeless.
    naive = deflated_sharpe_ratio(
        good,
        n_trials=10**9,
        trial_sharpes=list(rng.normal(0, 0.3, 50)),
    )
    assert naive.benchmark_sr > res.benchmark_sr
