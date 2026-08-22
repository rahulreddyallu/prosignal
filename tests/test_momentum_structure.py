"""The momentum block is one dimension, and the model must keep saying so.

Phase 14 measured that a single sign-pinned principal component of the three
shipped momentum factors reproduces their equal-weighted IC to three decimals:
+0.0755 against +0.0752. The 41% share of the model's IC that momentum carries
is one latent factor, not three sources of information.

Two candidate variants (ts_mom, mom_consist) cleared a marginal-IC screen on
the selection period at t +2.85 and +2.67 out of nine tested, and both failed
the holdout. Nothing was shipped. These tests pin the reasoning so a later
addition has to argue against it rather than around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "research" / "PHASE14_MOMENTUM.md"


def _pc1_signed(X: np.ndarray) -> np.ndarray:
    """First PC with the loading oriented toward the mean of its inputs.

    SVD sign is arbitrary. Left unpinned it flips between dates and any pooled
    statistic collapses toward zero for a reason that has nothing to do with
    the data -- which is exactly what happened in the first draft of the Phase
    14 measurement, and it inverted the conclusion.
    """
    Xc = X - X.mean(axis=0)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    load = vt[0]
    if load.sum() < 0:
        load = -load
    return Xc @ load


def test_pinning_the_pc_sign_changes_the_answer():
    """Guards the bug that inverted the Phase 14 conclusion."""
    rng = np.random.default_rng(0)
    agree, disagree = [], []
    for _ in range(40):
        common = rng.normal(size=200)
        X = np.column_stack([common + rng.normal(scale=0.4, size=200) for _ in range(3)])
        target = common + rng.normal(scale=1.0, size=200)
        agree.append(np.corrcoef(_pc1_signed(X), target)[0, 1])
        Xc = X - X.mean(axis=0)
        raw = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]
        disagree.append(np.corrcoef(raw, target)[0, 1])
    assert np.mean(agree) > 0.3, "the pinned component must track the common factor"
    assert abs(np.mean(disagree)) < abs(np.mean(agree)), (
        "the unpinned component must be the one whose pooled correlation "
        "collapses; if it is not, this test no longer guards the bug"
    )


def test_one_component_reproduces_a_correlated_block():
    """The structural claim: three highly correlated factors are one dimension."""
    rng = np.random.default_rng(3)
    common = rng.normal(size=500)
    X = np.column_stack([common + rng.normal(scale=0.35, size=500) for _ in range(3)])
    target = common + rng.normal(scale=1.2, size=500)
    pc = abs(np.corrcoef(_pc1_signed(X), target)[0, 1])
    eq = abs(np.corrcoef(X.mean(axis=1), target)[0, 1])
    assert abs(pc - eq) < 0.02, (
        "when a block is one dimension, its first component and its equal "
        "weighting carry the same information"
    )


def test_the_finding_is_recorded_with_its_rejections():
    """A negative result that is not written down gets rediscovered and shipped."""
    assert NOTE.is_file(), "Phase 14 note is missing"
    text = NOTE.read_text(encoding="utf-8").lower()
    for claim in ("ts_mom", "mom_consist", "rejected", "bonferroni", "0.96"):
        assert claim in text, f"the note no longer records {claim!r}"
    assert "one" in text and "latent factor" in text
