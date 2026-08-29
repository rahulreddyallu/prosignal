"""W2 -- the correction, its six acceptance criteria, and the one it fails.

The criteria were fixed before the correction was written. Five are asserted
here as properties. The sixth is asserted to FAIL, on purpose: a criterion that
is quietly dropped once it turns out to be inconvenient was never a criterion,
and the whole disposition of W2 -- reported, not traded -- rests on it having
been stated in advance and missed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prosignal.validation.selection import (ACCEPTANCE, assert_not_traded,
                                            correct_all, correct_t,
                                            simulate_recovery)

GATE = 2.0


# =============================================================================
# It reproduces the figures the readiness dossier published
# =============================================================================


@pytest.mark.parametrize("observed,expected,theme", [
    (2.87, 2.20, "mom_f"),
    (2.63, 1.44, "delivery_f"),
])
def test_it_reproduces_the_dossiers_published_correction(observed, expected, theme):
    """An independent implementation landing on the same number is the only
    check available on either -- the dossier ships no code for this."""
    assert correct_t(observed, GATE) == pytest.approx(expected, abs=0.03), theme


# =============================================================================
# Criteria 1-5
# =============================================================================


def test_1_monotone():
    ts = np.arange(GATE, 10.0, 0.05)
    out = [correct_t(float(t), GATE) for t in ts]
    for a, b in zip(out, out[1:]):
        assert b >= a - 1e-9, ACCEPTANCE["monotone"]


def test_2_never_inflates():
    for t in np.arange(GATE, 12.0, 0.05):
        c = correct_t(float(t), GATE)
        assert abs(c) <= abs(t) + 1e-9, ACCEPTANCE["never_inflates"]


def test_3_decays_far_above_the_gate():
    """The gate stops binding, so the correction must stop correcting.

    Also the numerically dangerous end: the inverse Mills ratio is computed
    from `1 - Phi(x)`, which underflows above about x = 8. Without the
    asymptotic branch a theme at t = 15 corrects to nonsense -- and t = 15 is
    exactly where the answer should be 15.
    """
    for t in (8.0, 10.0, 15.0, 25.0):
        assert correct_t(t, GATE) == pytest.approx(t, abs=1e-6), ACCEPTANCE["decays"]
    assert abs(correct_t(6.0, GATE) - 6.0) < 0.01


def test_4_sign_preserving():
    """A positive survivor never comes back negative.

    The unclamped MLE does go negative just above the gate -- at t = 2.0 it
    solves to about -0.4 -- which would report a theme the engine is LONG as
    having a negative true effect. Clamping at zero is what makes the output a
    statement about magnitude rather than an artefact of the estimator.
    """
    for t in np.arange(GATE, 8.0, 0.05):
        assert correct_t(float(t), GATE) >= 0.0, ACCEPTANCE["sign_preserving"]
        assert correct_t(-float(t), GATE) <= 0.0, ACCEPTANCE["sign_preserving"]
    assert correct_t(2.0, GATE) == 0.0, (
        "a theme that landed exactly on the gate carries no evidence of a real "
        "effect beyond having been selected"
    )


def test_5_no_new_parameter():
    """The gate is the only input. Nothing here is tuned.

    Asserted against the signature rather than by inspection, so adding a
    knob -- a shrinkage factor, a prior width, a blend weight -- breaks this
    test. Every one of those would be a parameter chosen after seeing the
    result it changes.
    """
    import inspect

    sig = inspect.signature(correct_t)
    positional = [p for p in sig.parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert [p.name for p in positional] == ["observed_t", "gate"], (
        ACCEPTANCE["no_new_parameter"] + f" -- found {[p.name for p in positional]}"
    )
    for p in sig.parameters.values():
        if p.kind is p.KEYWORD_ONLY:
            assert p.name in ("tol", "max_iter"), (
                f"{p.name} is a keyword argument that is not a solver setting; "
                f"if it changes the answer it is a tuned parameter"
            )


def test_the_conditioning_is_one_sided():
    """The error the first attempt made, pinned so it cannot come back.

    MEASURED ON THE POSITIVE TAIL, AND THAT IS THE WHOLE POINT. The first
    version of this test averaged the signed corrections over `|t| >= c`, where
    the two tails cancel at a true effect of zero and BOTH estimators return
    zero. Mutation testing caught it: the two-sided estimator was put back in
    the source and this test stayed green. A guard that cannot fail is not a
    guard, and this one could not.

    A surviving positive coefficient is what an engine reads off a card, so the
    conditioning event is `t >= c`. There the two estimators are far apart: at
    a true effect of zero the one-sided correction reports about +0.59 and the
    two-sided one about +1.00 -- a pure-noise theme returned as a genuine
    effect, because the correction could not tell which tail it came from.
    """
    rec = simulate_recovery(GATE, [0.0, 0.5, 1.0], draws=60_000,
                            positive_tail=True)
    assert rec[0.0] < 0.80, (
        f"a true effect of zero recovered as {rec[0.0]:+.3f} on the positive "
        f"tail. Under two-sided conditioning this lands near +1.00, because "
        f"the correction cannot tell which tail the survivor came from."
    )
    for m in (0.5, 1.0):
        assert rec[m] < m + 0.55, (
            f"true effect {m} recovered as {rec[m]:+.3f}; two-sided "
            f"conditioning inflates small effects by roughly +0.4 and that is "
            f"manufactured evidence, not conservatism"
        )


def test_even_the_correct_estimator_over_reports_a_true_zero():
    """The residual the clamp cannot remove, stated rather than hidden.

    On the positive tail a true effect of zero comes back at about +0.59, not
    at zero. The survivors that clamp to zero are the ones with the least
    evidence, and a clamp cannot subtract more than all of the effect, so what
    is left over is positive by construction.

    This is a property of conditioning on survival, not a bug in the solve, and
    it is the sharpest single reason the number is REPORTED and not TRADED --
    sharper than criterion 6, because it holds at the one value where the
    answer is unambiguous.
    """
    rec = simulate_recovery(GATE, [0.0], draws=60_000, positive_tail=True)
    assert 0.3 < rec[0.0] < 0.8, (
        f"a true zero recovered as {rec[0.0]:+.3f} on the positive tail. If "
        f"this is now near zero the estimator changed and W2's disposition "
        f"must be re-decided in the open."
    )
    signed = simulate_recovery(GATE, [0.0], draws=60_000)
    assert abs(signed[0.0]) < 0.10, (
        "the two-tailed signed mean must still cancel -- if it does not, the "
        "estimator is not sign-symmetric and something else is wrong"
    )


# =============================================================================
# Criterion 6 -- stated in advance, and NOT met
# =============================================================================


def test_6_recovers_truth_FAILS_and_that_is_why_it_is_not_traded():
    """The criterion the correction misses. Asserted as a failure.

    The correction is convex in `t` near the zero clamp, so by Jensen the mean
    of the corrections sits below the correction of the mean. The failure is
    worst in the MIDDLE of the range, not the tail: at m = 0 almost nothing
    survives except noise and the answer is zero; far above the gate the
    truncation barely operates and the correction is nearly the identity. In
    between, the gate binds hard and the clamp bites.

    That middle is exactly where this engine's traded coefficients sit, which
    is the sharpest argument against trading on it -- sharper than the
    readiness dossier's, which attributed the failure to m >= 3.0.

    This test asserts the criterion FAILS. If it starts passing, someone has
    improved the correction, and the disposition of W2 must be re-decided in
    the open rather than drifting.
    """
    rec = simulate_recovery(GATE, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                            draws=20_000)
    errors = {m: v - m for m, v in rec.items() if math.isfinite(v)}
    worst_m, worst = max(errors.items(), key=lambda kv: abs(kv[1]))

    assert abs(worst) > 0.25, (
        f"criterion 6 now PASSES (worst error {worst:+.3f} at m={worst_m}). "
        f"That is good news and it is a decision, not a test fix: W2's whole "
        f"disposition is that the correction is reported and not traded "
        f"BECAUSE it missed a criterion set in advance."
    )
    assert worst < 0, "the failure is under-recovery; over-recovery would be a "\
                      "different and much more dangerous defect"
    assert 1.0 <= worst_m <= 3.5, (
        f"the worst error moved to m={worst_m}; the failure region is what "
        f"decides whether it matters for THIS engine's coefficients"
    )
    assert abs(errors[0.0]) < 0.10 and abs(errors[4.0]) < 0.25, (
        "the ends must still recover -- if they do not, the problem is not "
        "Jensen near the clamp and the diagnosis in the module is wrong"
    )


def test_the_acceptance_criteria_are_all_still_declared():
    """Including the one that fails. A list of only the passing criteria is
    not a list of acceptance criteria."""
    assert set(ACCEPTANCE) == {
        "monotone", "never_inflates", "decays", "sign_preserving",
        "no_new_parameter", "recovers_truth"}


# =============================================================================
# Reported, not traded
# =============================================================================


def test_the_scoring_path_cannot_reach_the_correction():
    """Wiring it into a score has to fail a test, not pass a review."""
    with pytest.raises(RuntimeError, match="REPORTED, not TRADED"):
        assert_not_traded("prosignal.stages.stage4_core_score")
    with pytest.raises(RuntimeError):
        assert_not_traded("prosignal.features.crossmodel")
    # Reporting surfaces may.
    for ok in ("prosignal.cli", "prosignal.presentation.evidence",
               "prosignal.validation.harness"):
        assert_not_traded(ok)


def test_nothing_on_the_scoring_path_imports_it():
    """The guard above only fires if it is called. This checks the imports."""
    from pathlib import Path

    from prosignal.modelprint import MODEL_SOURCES

    src = Path(__file__).resolve().parents[1] / "src" / "prosignal"
    for rel in MODEL_SOURCES:
        text = (src / rel).read_text(encoding="utf-8")
        assert "validation.selection" not in text and \
               "from ..validation import selection" not in text, (
            f"{rel} decides a ranking and is importing the selection "
            f"correction. It failed its own acceptance criterion and is "
            f"reported, not traded."
        )


# =============================================================================
# What it says about the shipped fit
# =============================================================================


def test_a_theme_that_only_just_cleared_is_corrected_hard():
    """Where the correction is largest is where the decision is closest.

    The engine's own floor is 2.0. A theme at t = 2.3 corrects to zero and a
    theme at t = 4.1 barely moves, so the correction does the most work exactly
    where a coefficient's inclusion was most marginal -- which is the argument
    for computing it at all.
    """
    marginal = correct_t(2.3, GATE)
    comfortable = correct_t(4.1, GATE)
    assert marginal == 0.0
    assert comfortable == pytest.approx(4.1, abs=0.2)
    assert (4.1 - comfortable) < (2.3 - marginal)


def test_correct_all_ignores_themes_the_gate_zeroed():
    """A theme that did not survive is not a survivor and has no curse."""
    out = correct_all({"a_f": 3.4, "b_f": 1.1, "c_f": -2.9, "d_f": float("nan")},
                      GATE)
    assert [c.name for c in out] == ["a_f", "c_f"]
    assert out[1].corrected_t < 0, "sign is preserved through the batch path"
