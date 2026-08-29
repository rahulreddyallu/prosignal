"""The register has to be true about itself.

The readiness dossier listed fourteen findings as fixed and four of them were
absent from the source. Nothing detected that, because the claim and the code
lived in different files and only one of them ran. A register in code is only
an improvement on a document if something checks it -- otherwise it is a
document with a `.py` extension.

So: every finding claimed FIXED names a regression test, and every named test
EXISTS AND COLLECTS. The first draft of the register named five tests that did
not exist -- `test_dsr.py`, `test_holdout.py`, `test_trial_registry.py` and two
others -- and it took this check to notice, which is the whole argument for
having it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from prosignal.validation.findings import (REGISTER, Category, Status, by_id,
                                           categorised, open_findings,
                                           restart_blockers,
                                           unresolved_restart_blockers)

ROOT = Path(__file__).resolve().parents[1]


def _claimed_tests():
    return [(f.fid, f.regression_test) for f in REGISTER if f.regression_test]


# =============================================================================
# The register describes something real
# =============================================================================


@pytest.mark.parametrize("fid,ref", _claimed_tests(),
                         ids=[fid for fid, _ in _claimed_tests()])
def test_every_named_regression_test_exists(fid, ref):
    path = ROOT / ref.split("::")[0]
    assert path.is_file(), (
        f"{fid} names {ref} as the test that fails when its fix is reverted, "
        f"and that file does not exist. A fix nothing would catch the "
        f"reversion of is a claim, not a fix."
    )


def test_every_named_regression_test_actually_collects():
    """One pytest run for all of them: a file can exist while the node id
    inside it has been renamed, which reads as a passing reference and is not
    one."""
    ids = [ref for _, ref in _claimed_tests() if "::" in ref]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *ids],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, (
        "a finding names a test node that pytest cannot collect:\n"
        + proc.stdout[-2000:]
    )


def test_a_fixed_finding_cannot_be_claimed_without_a_test():
    """`_validate` enforces this at import. Asserted here too, because an
    import-time check that nothing imports is not a check."""
    for f in REGISTER:
        if f.status in (Status.FIXED, Status.BUILT_OFF):
            assert f.regression_test, f"{f.fid} is {f.status.value} with no test"


def test_moving_coefficients_implies_moving_history():
    """A refit supersedes every result computed under the old fit. A register
    that let one be true and the other false would let a coefficient change
    ship as a no-op."""
    for f in REGISTER:
        if f.moves_coefficients:
            assert f.moves_history, (
                f"{f.fid} moves coefficients and claims published results are "
                f"unchanged"
            )


def test_every_finding_carries_the_nine_fields_the_review_asked_for():
    """finding, root cause, location, fix, test, before/after, coefficients,
    history, restart. `reviewed` is not a disposition."""
    for f in REGISTER:
        for name in ("title", "root_cause", "location", "fix", "before_after"):
            assert getattr(f, name).strip(), f"{f.fid} has no {name}"
        assert isinstance(f.moves_coefficients, bool)
        assert isinstance(f.moves_history, bool)
        assert isinstance(f.forces_restart, bool)
        assert isinstance(f.category, Category)
        assert isinstance(f.status, Status)


def test_ids_are_unique_and_findable():
    ids = [f.fid for f in REGISTER]
    assert len(ids) == len(set(ids))
    for fid in ids:
        assert by_id(fid) is not None
        assert by_id(fid.lower()) is not None, "lookup is case sensitive"


# =============================================================================
# The classification is load-bearing
# =============================================================================


def test_every_finding_is_classified_into_the_review_s_categories():
    groups = categorised()
    assert sum(len(v) for v in groups.values()) == len(REGISTER)
    assert set(groups) <= {c.value for c in Category}


def test_the_dossier_s_own_open_items_are_all_present():
    """W2, C3 and C4 were listed as untouched. They are the reason this pass
    exists, so their absence from the register would be the register lying by
    omission."""
    for fid in ("W2", "C3", "C4"):
        f = by_id(fid)
        assert f is not None, f"{fid} is not in the register"
        assert f.status is not Status.OPEN, f"{fid} is still open"


def test_r1_is_the_only_thing_left_open():
    """The honest state of this engine. If another finding opens, this fails
    and somebody has to say which."""
    assert [f.fid for f in open_findings()] == ["R1"]


def test_every_restart_blocker_changes_the_engine_its_identity_or_the_question():
    """Three legitimate grounds, and nothing else.

    A window can be void because the ENGINE changed (coefficients or published
    results moved), because its IDENTITY is unrecordable (the data or the code
    cannot be named), or because the QUESTION changed -- which is R2: the
    pre-registration itself gained a hypothesis, so the old window was asking
    something weaker than the new one will.

    The first draft of this test allowed only the first two and failed on R2,
    which was the right failure: "changes what is being asked" is a real reason
    to restart and it was missing from the rule rather than from the finding.
    A blocker justified by none of the three is a gate that will be bypassed
    the first time it is inconvenient.
    """
    for f in restart_blockers():
        if f.fid == "R1":
            continue
        engine = f.moves_coefficients or f.moves_history
        identity = f.category is Category.REPRODUCIBILITY
        question = f.location.startswith("prosignal.validation.forward")
        assert engine or identity or question, (
            f"{f.fid} blocks a restart without changing the engine "
            f"(coefficients or published history), its identity "
            f"(reproducibility), or the question (the pre-registration). Say "
            f"which of the three, or stop blocking on it."
        )


def test_the_question_ground_is_used_once_and_deliberately():
    """The escape hatch above must not become the usual answer.

    Exactly one finding restarts the window by changing what is asked rather
    than what is measured. If a second appears, somebody is routing around the
    other two grounds.
    """
    by_question = [f.fid for f in restart_blockers()
                   if f.fid != "R1"
                   and not (f.moves_coefficients or f.moves_history)
                   and f.category is not Category.REPRODUCIBILITY]
    assert by_question == ["R2"], by_question


def test_r1_cannot_block_itself():
    r1 = by_id("R1")
    assert r1.forces_restart and r1.status is Status.OPEN
    assert "R1" not in [f.fid for f in unresolved_restart_blockers()], (
        "R1 IS the restart; a gate that required it closed could never open, "
        "and would read as a bug rather than as a policy"
    )
