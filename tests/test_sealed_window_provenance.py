"""A sealed window that lies inside the fit window is not a seal.

The shipped signs and weights were fitted 2018-11-27..2024-10-25
(`v3.FIT_WINDOW`). Two windows were sealed and reported side by side:

    A  2025-03-06 .. 2026-08-17   begins after the fit closed  -> OUT-OF-SAMPLE
    B  2021-07-01 .. 2022-12-27   entirely inside the fit      -> IN-SAMPLE

and B is the better-looking of the two -- it carries the positive book
(+2.0%/yr against A's -2.8%) and the significant top-ten excess (t 2.50 against
A's 0.81). Presented as a matched pair, an in-sample result was corroborating
an out-of-sample one.

B remains real evidence about the METHOD: the whole pipeline was re-run on data
ending 2021-02-17 and evaluated once on what followed. It is not evidence about
the shipped configuration, because the model it graded was a different fit.

These tests pin the classification so it cannot drift back into prose.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.features import v3
from prosignal.validation import v3_panel as vp


def test_the_fit_window_is_data_not_prose():
    lo, hi = v3.FIT_WINDOW
    assert lo == dt.date(2018, 11, 27)
    assert hi == dt.date(2024, 10, 25)


def test_window_a_is_out_of_sample():
    p = vp.SEALED_WINDOW_PROVENANCE["A"]
    assert p["verdict"] == v3.OUT_OF_SAMPLE
    assert p["share_in_sample"] == 0.0


def test_window_b_is_in_sample():
    """The finding, as one assertion."""
    p = vp.SEALED_WINDOW_PROVENANCE["B"]
    assert p["verdict"] == v3.IN_SAMPLE, (
        "window B is 2021-07..2022-12 and the fit window is "
        "2018-11..2024-10. If this ever reports OUT_OF_SAMPLE, either the fit "
        "window moved without the seal being re-earned, or the classification "
        "broke."
    )
    assert p["share_in_sample"] == pytest.approx(1.0)


def test_only_window_a_may_be_cited_for_the_shipped_model():
    assert vp.CITABLE_SEALED_WINDOWS == ("A",), (
        "a claim about the shipped configuration may cite window A only; "
        f"got {vp.CITABLE_SEALED_WINDOWS}"
    )


def test_the_deploy_reference_flags_its_own_in_sample_entries():
    assert vp.DEPLOY_REFERENCE["window_b_is_in_sample"] is True
    assert vp.DEPLOY_REFERENCE["citable_windows"] == ["A"]
    assert vp.DEPLOY_REFERENCE["book_excess_ann_is_leverage_confounded"] is True


def test_a_window_straddling_the_boundary_is_named_as_such():
    """Pooling in-sample and out-of-sample dates into one statistic is its own
    failure, and it is the one a naive 'last N months' re-check walks into."""
    p = v3.window_provenance(dt.date(2024, 1, 1), dt.date(2025, 6, 30))
    assert p["verdict"] == v3.STRADDLES_FIT_BOUNDARY
    assert 0.0 < p["share_in_sample"] < 1.0


def test_a_window_before_the_fit_is_out_of_sample_but_qualified():
    p = v3.window_provenance(dt.date(2015, 1, 1), dt.date(2016, 1, 1))
    assert p["verdict"] == v3.OUT_OF_SAMPLE
    assert "factor SET" in p["note"], (
        "a pre-fit window is out-of-sample for the WEIGHTS and not for the "
        "choice of factors; the note has to say so or it overstates the seal"
    )


def test_the_docstring_no_longer_presents_the_windows_as_equivalent():
    doc = v3.__doc__ or ""
    assert "IN-SAMPLE" in doc, (
        "the module docstring reports both windows in one table; it must mark "
        "which of them is in-sample or the table reads as two seals"
    )
    assert "provenance vs FIT_WINDOW" in doc
