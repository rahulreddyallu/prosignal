"""R1 -- the forward test may not restart onto an engine that is still changing.

The current window is void because the configuration moved under it. The
temptation is to restart it, and restarting it now would void the new window
the same way -- except eighteen months later, after the waiting had been done.

So `register` gates on `validation.readiness`. These tests pin three things:
that the gate cannot be bypassed by accident, that it refuses for reasons a
person can act on, and that it is reached from both places that can open a
window. The third matters most: the finding these tests belong to (R12) is
about a config flag that was read by nothing, and a gate only the CLI calls is
the same defect.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from prosignal.validation import forward
from prosignal.validation.findings import (REGISTER, Status,
                                           unresolved_restart_blockers)
from prosignal.validation.readiness import (DIMENSIONS, RESTART_GATES, Gate,
                                            Readiness, RestartRefused, assess,
                                            check_may_restart,
                                            restart_refusals)


# =============================================================================
# The gate cannot be skipped by accident
# =============================================================================


def test_register_refuses_to_run_ungated_without_being_told_to(tmp_path):
    """An optional gate is a gate nobody calls.

    This is R12's lesson applied to R1: `holdout.sacred` was a config flag no
    code read, and it looked exactly like a safety feature for months.
    """
    with pytest.raises(ValueError, match="unchecked_reason"):
        forward.register(tmp_path, config_version="v1", engine_version="0.1",
                         git_commit="abc")


def test_the_bypass_has_to_say_why_and_appears_nowhere_in_src():
    """Fixtures may bypass. Production may not, and this is what says so."""
    src = Path(forward.__file__).resolve().parents[1]
    offenders = []
    for py in src.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "unchecked_reason=" in text and py.name != "forward.py":
            offenders.append(py.relative_to(src).as_posix())
    assert not offenders, (
        f"{offenders} open a forward window without the readiness gate. The "
        f"bypass exists for test fixtures; a production caller that needs it "
        f"is a production caller that should not be opening a window."
    )


def test_a_fixture_may_bypass_explicitly(tmp_path):
    reg = forward.register(tmp_path, config_version="v1", engine_version="0.1",
                           git_commit="abc", unchecked_reason="unit fixture")
    assert reg.config_version == "v1"


def test_both_ways_of_opening_a_window_pass_the_configuration():
    """The CLI and the admin endpoint. Either one bypassing makes the other
    decorative."""
    from prosignal import api, cli

    cli_src = inspect.getsource(cli.cmd_research_forward)
    assert "cfg=cfg" in cli_src and "RestartRefused" in cli_src

    api_src = inspect.getsource(api.create_app)
    start = api_src.index("def admin_forward_register")
    body = api_src[start:start + 2500]
    assert "cfg=cfg" in body, "the admin endpoint opens a window ungated"
    assert "RestartRefused" in body


# =============================================================================
# What it refuses, and why
# =============================================================================


def test_it_refuses_while_a_restart_blocking_finding_is_open(monkeypatch, live_cfg):
    """The findings register is the human half of the gate."""
    from prosignal.validation import readiness as R

    fake = next(f for f in REGISTER if f.fid == "R9")
    monkeypatch.setattr(R._find, "unresolved_restart_blockers",
                        lambda: [fake])
    reasons = R.restart_refusals(live_cfg)
    assert any(r.startswith("R9 is open") for r in reasons), reasons


def test_r1_is_excluded_from_its_own_gate():
    """R1 IS the restart. A gate that required R1 closed before restarting
    could never open, and would read as a bug rather than as a policy."""
    assert any(f.fid == "R1" and f.forces_restart and f.status is Status.OPEN
               for f in REGISTER)
    assert "R1" not in [f.fid for f in unresolved_restart_blockers()]


def test_the_refusal_names_every_reason_rather_than_the_first(live_cfg):
    """A gate that reports one blocker at a time turns a day's work into a
    week of rediscovering the next one."""
    exc = RestartRefused(["a is open", "b gate not met", "c gate not met"])
    body = str(exc)
    for r in ("a is open", "b gate not met", "c gate not met"):
        assert r in body
    assert exc.reasons == ["a is open", "b gate not met", "c gate not met"]


def test_both_halves_of_the_gate_are_reported_together(monkeypatch, live_cfg):
    """An open finding AND a failing gate must both appear in one refusal.

    The two sources are independent by design -- a finding can be marked
    resolved while the switch that resolves it is off, and a switch can be on
    while the finding was never really settled -- so returning as soon as one
    of them has something to say hides the other. Mutation testing found this:
    an early `return out` after the findings loop left every existing test
    green, because none of them had both kinds of blocker at once.
    """
    from prosignal.validation import readiness as R

    r9 = next(f for f in REGISTER if f.fid == "R9")
    monkeypatch.setattr(R._find, "unresolved_restart_blockers", lambda: [r9])

    reasons = R.restart_refusals(live_cfg)      # live_cfg has no epoch open
    assert any(x.startswith("R9 is open") for x in reasons), reasons
    assert any("REPRODUCIBILITY" in x for x in reasons), (
        f"the gate stopped after the findings register and never assessed the "
        f"gates: {reasons}"
    )


def test_check_raises_only_when_there_is_something_to_raise_about(monkeypatch,
                                                                 live_cfg):
    from prosignal.validation import readiness as R

    monkeypatch.setattr(R, "restart_refusals", lambda cfg: [])
    check_may_restart(live_cfg)               # must not raise

    monkeypatch.setattr(R, "restart_refusals", lambda cfg: ["nope"])
    with pytest.raises(RestartRefused):
        check_may_restart(live_cfg)


def test_register_is_blocked_by_the_gate_it_calls(monkeypatch, tmp_path, live_cfg):
    """End to end: the refusal reaches `register` and nothing is written."""
    from prosignal.validation import readiness as R

    monkeypatch.setattr(R, "restart_refusals",
                        lambda cfg: ["DATA gate not met: no manifest"])
    with pytest.raises(RestartRefused, match="no manifest"):
        forward.register(tmp_path, config_version="v1", engine_version="0.1",
                         git_commit="abc", cfg=live_cfg)
    assert not (tmp_path / forward.REGISTRATION_NAME).exists(), (
        "a refused registration still wrote the file, so the clock started"
    )


def test_the_gate_runs_before_the_overwrite_check(monkeypatch, tmp_path, live_cfg):
    """Order matters. If `overwrite=False` raised first, a restart onto an
    unready engine would look like 'a test is already registered' -- a
    completely different problem, with a completely different fix."""
    from prosignal.validation import readiness as R

    forward.register(tmp_path, config_version="v1", engine_version="0.1",
                     git_commit="abc", unchecked_reason="fixture")
    monkeypatch.setattr(R, "restart_refusals", lambda cfg: ["UNIVERSE gate"])
    with pytest.raises(RestartRefused):
        forward.register(tmp_path, config_version="v2", engine_version="0.1",
                         git_commit="def", cfg=live_cfg)


# =============================================================================
# The eight gates
# =============================================================================


def test_every_dimension_is_assessed(live_cfg):
    r = assess(live_cfg)
    assert [g.name for g in r.gates] == list(DIMENSIONS)


def test_the_dimensions_are_in_dependency_order():
    """Data decides the universe, the universe decides the fit, the fit
    decides what execution applies to, and the forward test consumes all of
    them. Printed in any other order, the last one gets fixed first."""
    order = list(DIMENSIONS)
    assert order.index("DATA") < order.index("UNIVERSE") < order.index("MODEL")
    assert order.index("MODEL") < order.index("EXECUTION")
    assert order[-1] == "FORWARD"


def test_forward_is_not_a_precondition_for_restarting_the_forward_test():
    assert "FORWARD" not in RESTART_GATES
    assert set(RESTART_GATES) <= set(DIMENSIONS)


def test_an_undecidable_gate_is_not_a_pass():
    """"No data" must never read as "no problem"."""
    g = Gate("DATA", False, "cannot tell", unknown=True)
    assert not g.passed and "UNKNOWN" in g.line()
    assert not Readiness([g]).ready


def test_ready_means_all_eight(live_cfg):
    passing = [Gate(n, True, "ok") for n in DIMENSIONS]
    assert Readiness(passing).ready
    for i in range(len(passing)):
        one_bad = list(passing)
        one_bad[i] = Gate(DIMENSIONS[i], False, "no")
        r = Readiness(one_bad)
        assert not r.ready
        assert DIMENSIONS[i] in r.verdict()


def test_a_failed_gate_says_what_would_fix_it(live_cfg):
    """A gate that reports a state and not a remedy is a complaint."""
    for g in assess(live_cfg).failed:
        assert g.remedy or g.unknown, (
            f"{g.name} fails with no remedy: {g.detail}"
        )


# =============================================================================
# The two gates that check behaviour rather than configuration
# =============================================================================


def test_the_execution_gate_runs_the_cost_model_rather_than_reading_a_flag(live_cfg):
    """R13 is a property of the arithmetic, not of a switch. A config flag
    saying liquidity is gated proves nothing about whether it is."""
    src = inspect.getsource(__import__(
        "prosignal.validation.readiness", fromlist=["_execution_gate"]
    )._execution_gate)
    assert "impact_bps" in src
    g = assess(live_cfg).gate("EXECUTION")
    assert g.passed, g.detail
    assert "bps" in g.detail


def test_the_validation_gate_reads_the_register(live_cfg):
    g = assess(live_cfg).gate("VALIDATION")
    assert str(len(REGISTER)) in g.detail


def test_a_tree_with_no_epoch_is_refused_a_restart(live_cfg):
    """`live_cfg` redirects the ledger to a temp directory, so no epoch is open
    under it and REPRODUCIBILITY must refuse.

    Named for what it actually checks. Its first version was called
    "the shipped engine is refused a restart today" and read `live_cfg`, which
    is not the shipped ledger -- so it passed for a reason unrelated to its
    name, and went on passing after the shipped engine became restartable. A
    test whose name and subject disagree is worse than no test: it reports on
    something nobody is watching.
    """
    reasons = restart_refusals(live_cfg)
    assert any("REPRODUCIBILITY" in r for r in reasons), reasons


def test_the_shipped_engine_is_ready_to_be_restarted_and_has_not_been():
    """The honest state of THIS repository, read from the real config.

    Every precondition now holds: the data is manifested and verifies, the
    panel is the population the book can buy, the fit is attributable, the
    execution model is monotone, no finding but R1 is open, and an epoch
    describes the engine. So the gate permits a restart.

    It has not been taken. Restarting starts an eighteen-month clock and
    discards the observations collected so far, and that is an operator's
    decision -- which is exactly why the gate REPORTS and the person ACTS. R1
    stays OPEN in the register until somebody does it.

    If this fails on the first assertion a gate has started refusing and the
    reason is worth reading. If it fails on the second, the window was
    restarted, and the register should say so rather than this test being
    updated to match.
    """
    from prosignal.config.loader import load_config
    from prosignal.validation.findings import Status, by_id

    reasons = restart_refusals(load_config())
    assert not reasons, (
        "the shipped engine can no longer be restarted:\n  "
        + "\n  ".join(reasons)
    )
    assert by_id("R1").status is Status.OPEN, (
        "R1 is marked resolved. Closing it means the forward window was "
        "actually restarted, which is a decision that belongs in the epoch "
        "ledger and the register, not in a status field alone."
    )
