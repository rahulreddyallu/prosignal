"""BASELINE_V1 must be reproducible, and must carry the sample-size ceiling.

A baseline captured from uncommitted code cannot be reconstructed, and a
comparison against something unreproducible is not a comparison. The field
that matters most is not a Sharpe: it is how many independent forward-return
windows each factor family actually has, because that bounds every DSR, PBO
and calibration claim the engine makes -- and the families differ by a factor
of three.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosignal.validation.baseline import BASELINE_PATH, Baseline, capture, load

ROOT = Path(__file__).resolve().parents[1]


def test_the_baseline_exists_and_parses():
    b = load(ROOT / BASELINE_PATH)
    assert b is not None, "BASELINE_V1 is missing; nothing can be compared against it"
    assert b.git_commit, "a baseline without a commit cannot be reconstructed"
    assert b.config_hash


def test_a_dirty_tree_is_refused_by_default(monkeypatch):
    import prosignal.validation.baseline as mod

    monkeypatch.setattr(mod, "_git", lambda *a: "M src/foo.py" if a[0] == "status" else "abc")
    with pytest.raises(ValueError, match="uncommitted"):
        capture(_Cfg(), allow_dirty=False)


def test_a_dirty_tree_is_recorded_when_explicitly_allowed(monkeypatch):
    import prosignal.validation.baseline as mod

    monkeypatch.setattr(mod, "_git", lambda *a: "M src/foo.py" if a[0] == "status" else "abc")
    b = capture(_Cfg(), allow_dirty=True)
    assert b.git_dirty is True, "an approximate record must say that it is one"


def test_it_carries_independent_observations_per_family():
    b = load(ROOT / BASELINE_PATH)
    obs = b.independent_observations
    for family in ("momentum", "delivery", "value"):
        assert family in obs, f"{family} has no recorded sample size"
        assert obs[family]["independent_obs_at_horizon"] > 0


def test_the_families_are_not_equally_evidenced_and_the_record_says_so():
    """The finding this baseline exists to pin: value has ~a third the history."""
    obs = load(ROOT / BASELINE_PATH).independent_observations
    price = obs["momentum"]["independent_obs_at_horizon"]
    value = obs["value"]["independent_obs_at_horizon"]
    assert value < price, (
        "the value block's vendor coverage begins later than the price block's; "
        "if this ever inverts, the baseline is stale"
    )
    assert value <= 15, (
        f"value has {value} independent observations. Any ML, ensemble or "
        f"calibration work on the fundamental block is bounded by this number, "
        f"and the baseline must keep saying so."
    )


def test_known_limits_are_recorded_rather_than_left_to_memory():
    b = load(ROOT / BASELINE_PATH)
    assert len(b.known_limits) >= 5
    joined = " ".join(b.known_limits).lower()
    for topic in ("independent observations", "pbo", "calibration"):
        assert topic in joined, f"the baseline does not mention {topic}"


class _Cfg:
    version = "test-v1"
