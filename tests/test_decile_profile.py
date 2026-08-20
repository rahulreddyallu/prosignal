"""Decile profile: the check the information coefficient cannot make.

IC is a rank correlation. A rank correlation can be healthy while the top of
the ranking is not the best part of it, and the engine trades the top decile,
so a chart peaking at decile 6 says the traded set is not the set the IC
describes.

On the shipped configuration over the 36-period holdout the deciles run +1.97%
to +3.98% -- a real spread -- with only 6 of 9 steps increasing and decile 6
above decile 9. The spread is real, the monotonicity is not.
"""
from __future__ import annotations

import numpy as np
import pytest

from prosignal.validation.metrics import decile_profile


def test_a_clean_signal_climbs_and_peaks_at_the_top():
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, 4000)
    f = 0.3 * s + rng.normal(0, 1, 4000)
    r = decile_profile(s, f)
    assert r["top_is_peak"]
    assert r["monotone_steps"] >= 7
    assert r["spread"] > 0


def test_a_signal_that_breaks_at_the_top_is_reported_as_such():
    """The failure IC hides: everything below the top behaves, the top does not."""
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, 4000)
    f = 0.3 * s + rng.normal(0, 1, 4000)
    f[s > 1.2] -= 2.0
    r = decile_profile(s, f)
    assert not r["top_is_peak"]
    assert r["peak_bucket"] < 9


def test_a_pure_noise_signal_has_no_spread_worth_the_name():
    rng = np.random.default_rng(9)
    s = rng.normal(0, 1, 4000)
    f = rng.normal(0, 1, 4000)
    r = decile_profile(s, f)
    assert abs(r["spread"]) < 0.25
    assert r["monotone_steps"] < 8


def test_an_inverted_signal_reports_a_negative_spread():
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, 4000)
    r = decile_profile(s, -0.3 * s + rng.normal(0, 1, 4000))
    assert r["spread"] < 0
    assert not r["top_is_peak"]


def test_too_few_observations_reports_why_rather_than_guessing():
    r = decile_profile(np.arange(5.0), np.arange(5.0))
    assert r["buckets"] == []
    assert "needed" in r["reason"]


def test_nans_are_dropped_pairwise():
    rng = np.random.default_rng(7)
    s = rng.normal(0, 1, 3000)
    f = 0.3 * s + rng.normal(0, 1, 3000)
    s[::10] = np.nan
    r = decile_profile(s, f)
    assert r["spread"] is not None
    assert len(r["buckets"]) == 10
