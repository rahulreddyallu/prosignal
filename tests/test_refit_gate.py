"""Promotion gate and rollback for model coefficients.

The 21-session refit is the one path where a corrupted upstream date reaches
every future decision at once, silently: the fit succeeds, the numbers look
like numbers, and the ranking changes. So a refit is proposed rather than
installed, and the file it would replace is kept.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from prosignal.features import crossmodel as cm
from prosignal.features.refit_gate import (
    MAX_MAGNITUDE_RATIO, MAX_SIGN_FLIPS, SIGN_FLIP_FLOOR, review_refit,
)

#: A plausible live fit over the model's actual columns, so load_cached (which
#: checks the feature set) accepts a restored version in the rollback drill.
# Families, because that is what a fitted model now carries: one coefficient per
# family rather than seventeen over a collinear set. A cache naming individual
# factors is a pre-family fit and `load_cached` refuses it on purpose.
LIVE = {name: value for name, value in zip(
    cm.FAMILY_COLUMNS,
    [0.0255, 0.0047, -0.0190, 0.0039, 0.0156, 0.0060],
)}
_FIRST = cm.FAMILY_COLUMNS[0]
_SECOND = cm.FAMILY_COLUMNS[2]


def test_an_ordinary_refit_is_accepted():
    nudged = {k: v * 1.15 for k, v in LIVE.items()}
    assert review_refit(nudged, LIVE).accepted


def test_the_first_fit_has_nothing_to_compare_against():
    v = review_refit(LIVE, None)
    assert v.accepted
    assert "no previous fit" in v.summary()


def test_a_wholesale_sign_reversal_is_rejected():
    flipped = {k: -v for k, v in LIVE.items()}
    v = review_refit(flipped, LIVE)
    assert not v.accepted
    # Every coefficient above the noise floor is reported; the ones below it
    # are not, because their sign was never meaningful.
    above_floor = sum(1 for x in LIVE.values() if abs(x) >= SIGN_FLIP_FLOOR)
    assert len(v.sign_flips) == above_floor
    assert "reversed sign" in v.summary()


def test_one_marginal_flip_is_tolerated():
    """A single small factor reversing is ordinary refit noise."""
    nearly = dict(LIVE)
    nearly[_SECOND] = -LIVE[_SECOND] - 1e-5
    assert review_refit(nearly, LIVE).accepted


def test_a_coefficient_below_the_noise_floor_may_flip_freely():
    live = dict(LIVE, tiny_r=SIGN_FLIP_FLOOR / 10)
    proposed = dict(LIVE, tiny_r=-SIGN_FLIP_FLOOR / 10)
    v = review_refit(proposed, live)
    assert v.accepted
    assert not v.sign_flips


def test_a_magnitude_explosion_is_rejected():
    blown = dict(LIVE)
    blown[_FIRST] = LIVE[_FIRST] * (MAX_MAGNITUDE_RATIO + 2)
    v = review_refit(blown, LIVE)
    assert not v.accepted
    assert v.magnitude_jumps


def test_a_non_finite_coefficient_is_rejected():
    bad = dict(LIVE); bad[_FIRST] = float("nan")
    assert not review_refit(bad, LIVE).accepted


def test_a_disjoint_feature_set_is_not_a_refit():
    """Changing the model is a decision for a person, not an auto-promotion."""
    v = review_refit({"something_else_r": 0.01}, LIVE)
    assert not v.accepted
    assert "shares no factors" in v.summary()


def test_an_empty_proposal_is_rejected():
    assert not review_refit({}, LIVE).accepted


# --------------------------------------------------------------------------
# versioning and the rollback drill
# --------------------------------------------------------------------------

def _model(coef):
    m = cm.CrossSectionalModel(coef=dict(coef), n_train=5000,
                               train_end=dt.date(2026, 1, 1))
    import numpy as np
    m.mu = np.zeros(len(coef)); m.sd = np.ones(len(coef)); m.intercept = 0.0
    return m


def test_saving_over_a_live_file_keeps_the_previous_version(tmp_path):
    path = tmp_path / "crosssec_model.json"
    cm.save_cache(path, _model(LIVE), dt.date(2026, 6, 1))
    archived = cm.archive_cache(path)
    assert archived is not None
    cm.save_cache(path, _model({k: v * 2 for k, v in LIVE.items()}), dt.date(2026, 7, 1))

    live = json.loads(path.read_text())
    kept = json.loads(open(archived).read())
    assert live["coef"][_FIRST] == pytest.approx(LIVE[_FIRST] * 2)
    assert kept["coef"][_FIRST] == pytest.approx(LIVE[_FIRST])


def test_rollback_drill(tmp_path):
    """Corrupt the live coefficients, restore the prior version, confirm the
    model loads again. Without this the only way back from a bad refit is a
    retrain, which reproduces whatever caused it."""
    path = tmp_path / "crosssec_model.json"
    cm.save_cache(path, _model(LIVE), dt.date(2026, 6, 1))
    good = cm.archive_cache(path)

    path.write_text("{ this is not json", encoding="utf-8")
    assert cm.load_cached(path, dt.date(2026, 6, 2)) is None      # fails loudly

    path.write_text(open(good).read(), encoding="utf-8")
    restored = cm.load_cached(path, dt.date(2026, 6, 2))
    assert restored is not None
    assert restored.coef[_FIRST] == pytest.approx(LIVE[_FIRST])


def test_only_a_bounded_number_of_versions_is_kept(tmp_path):
    path = tmp_path / "crosssec_model.json"
    for day in range(1, 16):
        cm.save_cache(path, _model(LIVE), dt.date(2026, 6, day))
        cm.archive_cache(path, keep=5)
    kept = list((tmp_path / "crosssec_model_versions").glob("*.json"))
    assert len(kept) == 5


def test_reading_live_coefficients_ignores_staleness(tmp_path):
    """The gate asks what it would be replacing, not whether that is scoreable."""
    path = tmp_path / "crosssec_model.json"
    cm.save_cache(path, _model(LIVE), dt.date(2020, 1, 1))
    coef, train_end, estimator = cm.read_cached_coefficients(path)
    assert coef[_FIRST] == pytest.approx(LIVE[_FIRST])
    assert train_end == "2026-01-01"
    assert estimator == "ridge"          # what _model() carries by default
    assert cm.load_cached(path, dt.date(2026, 6, 1)) is None       # too stale to score


def test_a_blob_written_before_the_estimator_field_reads_as_ridge(tmp_path):
    """Those blobs were ridge fits. Reporting the estimator as unknown would
    make the gate skip its comparison on every legacy file."""
    path = tmp_path / "crosssec_model.json"
    cm.save_cache(path, _model(LIVE), dt.date(2026, 6, 1))
    blob = json.loads(path.read_text())
    del blob["estimator"]
    path.write_text(json.dumps(blob), encoding="utf-8")
    assert cm.read_cached_coefficients(path)[2] == "ridge"


# ------------------------------------------------- changing the estimator
def test_an_estimator_change_is_not_reviewed_as_a_refit():
    """Ridge and Fama-MacBeth produce different numbers from the same data by
    construction, so comparing their magnitudes is meaningless. On the recorded
    switch it read as a sign flip and a 5.4x jump -- indistinguishable, by
    magnitude alone, from the corrupted upstream date this gate is for."""
    # Exactly the shape of the real ridge -> Fama-MacBeth change.
    previous = dict(LIVE)
    proposed = dict(LIVE)
    proposed[_FIRST] = 0.0                       # momentum gated out
    proposed[_SECOND] = -LIVE[_SECOND] * 6.0     # sign flip and a 6x jump

    # Reviewed as a refit, this is rejected -- correctly, for a refit.
    same = review_refit(proposed, previous,
                        proposed_estimator="ridge", previous_estimator="ridge")
    assert not same.accepted

    # Reviewed across estimators, the comparison must not be applied at all.
    switched = review_refit(proposed, previous,
                            proposed_estimator="fama_macbeth",
                            previous_estimator="ridge")
    assert switched.accepted
    assert not switched.sign_flips and not switched.magnitude_jumps
    assert "estimator changed" in switched.summary()


def test_an_unknown_estimator_on_either_side_keeps_the_comparison():
    """Absence of the field must not become a way past the gate."""
    proposed = {k: v * 20 for k, v in LIVE.items()}
    assert not review_refit(proposed, LIVE).accepted
    assert not review_refit(proposed, LIVE, proposed_estimator="ridge").accepted
    assert not review_refit(proposed, LIVE, previous_estimator="ridge").accepted


# ------------------------------------------------------ archive collisions
def test_two_refits_fitted_for_one_date_do_not_overwrite_each_other(tmp_path):
    """A config or estimator change on a single day produces exactly this. The
    archive was named by `fitted_for` alone, so the second copy destroyed the
    first -- removing the recovery path at the moment it was most needed."""
    path = tmp_path / "crosssec_model.json"
    cm.save_cache(path, _model(LIVE), dt.date(2026, 6, 1))
    first = cm.archive_cache(path)

    cm.save_cache(path, _model({k: v * 3 for k, v in LIVE.items()}),
                  dt.date(2026, 6, 1))          # SAME fitted_for
    second = cm.archive_cache(path)

    assert first != second, "the second archive overwrote the first"
    assert json.loads(open(first).read())["coef"][_FIRST] == pytest.approx(LIVE[_FIRST])
    assert json.loads(open(second).read())["coef"][_FIRST] == pytest.approx(LIVE[_FIRST] * 3)


def test_archiving_identical_content_twice_is_idempotent(tmp_path):
    path = tmp_path / "crosssec_model.json"
    cm.save_cache(path, _model(LIVE), dt.date(2026, 6, 1))
    assert cm.archive_cache(path) == cm.archive_cache(path)
    kept = list((tmp_path / "crosssec_model_versions").glob("*.json"))
    assert len(kept) == 1
