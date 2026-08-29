"""The forward test is only a test if its criteria were fixed beforehand.

Nine years of history gives roughly thirty independent 63-session windows for
the price factors and eleven for the value factors, and the holdout
attribution's alpha swung from +3.67% to -1.01% depending on which factors
entered. The only remaining source of independent observations is time that has
not happened yet.

What turns that time into evidence rather than a log is discipline that has to
be enforced mechanically, because it is exactly the discipline a disappointing
result tempts you to abandon: the criteria are hashed before the first
observation, the model is frozen for the duration, and no interim performance
is reported.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from prosignal.validation.forward import (
    REGISTRATION_NAME, TARGET_MONTHS, TARGET_SESSIONS,
    load_registration, progress, register, verify,
)


def open_test(tmp_path, started=dt.date(2026, 8, 23), version="cfg@aaa"):
    return register(tmp_path, config_version=version, engine_version="0.1.0",
                    git_commit="deadbeef" * 5, started_on=started,
                    unchecked_reason="fixture: this file tests the registration itself, not the readiness gate that guards it")


def row(date, version="cfg@aaa", error=None):
    return {"date": date.isoformat(), "config_version": version, "error": error}


# ------------------------------------------------------- pre-registration
def test_the_criteria_are_hashed_at_registration():
    """A forward test whose success condition is decided after the data
    arrives measures nothing: any outcome can be made a pass by choosing what
    to report."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        reg = open_test(Path(d))
        assert verify(Path(d)) is True
        assert len(reg.fingerprint()) == 16


def test_editing_the_criteria_afterwards_is_detected(tmp_path):
    open_test(tmp_path)
    path = tmp_path / REGISTRATION_NAME
    data = json.loads(path.read_text())
    data["primary"] = "the engine passes if it makes money"
    path.write_text(json.dumps(data))
    assert verify(tmp_path) is False


def test_a_second_registration_is_refused(tmp_path):
    """Overwriting silently resets the clock, which lets a disappointing
    window be replaced by a fresh one."""
    open_test(tmp_path)
    with pytest.raises(FileExistsError, match="already registered"):
        open_test(tmp_path)


def test_restarting_is_possible_but_must_be_explicit(tmp_path):
    open_test(tmp_path)
    reg = register(tmp_path, config_version="cfg@bbb", engine_version="0.1.0",
                   git_commit="x" * 40, overwrite=True,
                   unchecked_reason="fixture: the gate has its own file")
    assert reg.config_version == "cfg@bbb"


def test_both_hypotheses_are_stated_as_pass_conditions(tmp_path):
    reg = open_test(tmp_path)
    for text in (reg.primary, reg.secondary):
        assert "passes if" in text
    assert "fails if" in reg.primary


def test_the_primary_test_is_the_one_the_holdout_could_not_settle(tmp_path):
    """The forward window is sized for the attribution, not for the IC."""
    reg = open_test(tmp_path)
    assert "intercept" in reg.primary
    assert "degrees of freedom" in reg.primary
    assert "NOT the question at issue" in reg.secondary


def test_acting_on_the_signal_invalidates_the_test(tmp_path):
    reg = open_test(tmp_path)
    assert any("real capital" in i for i in reg.invalidation)
    assert any("config_version changes" in i for i in reg.invalidation)


# -------------------------------------------------------------- progress
def test_progress_counts_distinct_sessions_not_runs(tmp_path):
    """A date re-run five times is one observation."""
    open_test(tmp_path)
    d = dt.date(2026, 8, 24)
    p = progress(tmp_path, [row(d)] * 5, today=dt.date(2026, 8, 24))
    assert p.sessions_elapsed == 1
    assert p.runs_recorded == 5


def test_runs_before_the_start_are_not_counted(tmp_path):
    open_test(tmp_path, started=dt.date(2026, 8, 23))
    p = progress(tmp_path, [row(dt.date(2026, 1, 1)), row(dt.date(2026, 9, 1))],
                 today=dt.date(2026, 9, 1))
    assert p.sessions_elapsed == 1


def test_a_failed_run_is_not_an_observation(tmp_path):
    open_test(tmp_path)
    p = progress(tmp_path, [row(dt.date(2026, 8, 25), error="blocked")],
                 today=dt.date(2026, 8, 25))
    assert p.sessions_elapsed == 0


def test_a_configuration_change_breaks_the_window(tmp_path):
    """Observations after the change came from a different model. Even an
    improvement ends the experiment."""
    open_test(tmp_path, version="cfg@aaa")
    p = progress(tmp_path, [row(dt.date(2026, 8, 24), "cfg@aaa"),
                            row(dt.date(2026, 9, 24), "cfg@bbb")],
                 today=dt.date(2026, 9, 24))
    assert p.broken
    assert p.complete is False
    assert "configuration versions" in p.broken[0]


def test_a_window_that_never_reaches_its_target_is_not_complete(tmp_path):
    open_test(tmp_path)
    rows = [row(dt.date(2026, 8, 24) + dt.timedelta(days=i)) for i in range(100)]
    p = progress(tmp_path, rows, today=dt.date(2027, 1, 1))
    assert p.complete is False
    assert "Nothing may be concluded" in p.summary()


def test_both_targets_must_be_met_not_either(tmp_path):
    """375 sessions arriving in nine months would mean the runs were not
    daily, and eighteen months with forty runs is not a period."""
    open_test(tmp_path, started=dt.date(2026, 8, 23))
    rows = [row(dt.date(2026, 8, 24) + dt.timedelta(days=i))
            for i in range(TARGET_SESSIONS)]
    early = progress(tmp_path, rows, today=dt.date(2027, 2, 1))   # ~6 months
    assert early.sessions_elapsed >= TARGET_SESSIONS
    assert early.complete is False, "sessions alone must not complete the test"


def test_progress_reports_no_performance_at_all(tmp_path):
    """Interim results invite optional stopping. A test stopped when it looks
    good has no p-value worth quoting."""
    open_test(tmp_path)
    p = progress(tmp_path, [row(dt.date(2026, 8, 24))], today=dt.date(2026, 8, 24))
    blob = json.dumps(p.__dict__, default=str).lower() + p.summary().lower()
    for banned in ("sharpe", "return", "alpha", "excess", "profit", "ic "):
        assert banned not in blob


def test_a_tampered_registration_breaks_progress_too(tmp_path):
    open_test(tmp_path)
    path = tmp_path / REGISTRATION_NAME
    data = json.loads(path.read_text())
    data["target_months"] = 1
    path.write_text(json.dumps(data))
    p = progress(tmp_path, [row(dt.date(2026, 8, 24))], today=dt.date(2026, 8, 24))
    assert any("no longer matches its hash" in b for b in p.broken)


def test_no_registration_means_no_progress(tmp_path):
    assert progress(tmp_path, []) is None
    assert load_registration(tmp_path) is None


def test_the_window_is_sized_for_the_attribution(tmp_path):
    assert TARGET_MONTHS == 18
    assert TARGET_SESSIONS == 375
    reg = open_test(tmp_path)
    assert any("six non-overlapping" in n for n in reg.notes)


def test_a_run_against_an_earlier_close_is_not_a_forward_observation(tmp_path):
    """The ledger stamps the MARKET date. A run made after registration but
    scored on the previous session's close is a re-score of data the model has
    already seen, and counting it would put in-sample observations inside the
    forward window on day one."""
    open_test(tmp_path, started=dt.date(2026, 8, 23))
    p = progress(tmp_path, [row(dt.date(2026, 8, 21))], today=dt.date(2026, 8, 23))
    assert p.sessions_elapsed == 0
    assert "market date" in p.summary().lower()


# =========================================================================
# Drift that happens BEFORE the first observation
# =========================================================================
# The window is void from the moment the config moves. Reading drift off the
# ledger rows inside the window cannot see that, because before the first
# observation there are no rows -- so `broken` came back empty and the summary
# read "Registered today; no forward session yet", which is the wording for a
# healthy start. Observed on a live registration: registered against a30a8d48,
# engine running 9776e5d6, nothing flagged anywhere.

def test_a_config_change_before_the_first_observation_breaks_the_window(tmp_path):
    open_test(tmp_path, version="cfg@aaa")
    prog = progress(tmp_path, [], live_config_version="cfg@bbb")
    assert prog.broken, "a window registered against a config the engine no longer runs is void"
    assert "cfg@bbb" in prog.broken[0] and "cfg@aaa" in prog.broken[0]


def test_the_summary_never_reads_healthy_on_a_broken_window(tmp_path):
    """The nightly job pipes these four lines into the log. If they read
    reassuringly on a void window, the log is worse than no log."""
    open_test(tmp_path, version="cfg@aaa")
    prog = progress(tmp_path, [], live_config_version="cfg@bbb")
    assert prog.summary().startswith("The forward test is INVALID")
    assert "no forward session yet" not in prog.summary()


def test_the_unchanged_config_is_not_reported_as_drift(tmp_path):
    open_test(tmp_path, version="cfg@aaa")
    assert not progress(tmp_path, [], live_config_version="cfg@aaa").broken


def test_omitting_the_live_config_keeps_the_old_behaviour(tmp_path):
    """Callers that cannot supply it must not start seeing false drift."""
    open_test(tmp_path, version="cfg@aaa")
    assert not progress(tmp_path, []).broken


# =========================================================================
# The coverage criterion the registration names and nothing evaluated
# =========================================================================

def _sessions(start, n):
    """n consecutive weekday sessions from `start`."""
    out, cur = [], start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += dt.timedelta(days=1)
    return out


def test_the_sixty_percent_coverage_criterion_is_actually_evaluated(tmp_path):
    """'Fewer than 60% of expected sessions produce a recorded run, which would
    make the sample a selection rather than a period.' -- the registration has
    said this since the first window and MIN_SESSION_COVERAGE was referenced by
    nothing in src/ or tests/."""
    start = dt.date(2026, 8, 24)
    open_test(tmp_path, started=start)
    printed = _sessions(start, 50)
    rows = [row(d) for d in printed[:20]]          # 20 of 50 = 40%
    prog = progress(tmp_path, rows, today=printed[-1], sessions_printed=50)
    assert prog.broken
    assert "selection, not a period" in prog.broken[0]
    assert prog.coverage == pytest.approx(0.4)


def test_adequate_coverage_is_not_flagged(tmp_path):
    start = dt.date(2026, 8, 24)
    open_test(tmp_path, started=start)
    printed = _sessions(start, 50)
    rows = [row(d) for d in printed[:40]]          # 80%
    prog = progress(tmp_path, rows, today=printed[-1], sessions_printed=50)
    assert not prog.broken
    assert prog.coverage == pytest.approx(0.8)


def test_coverage_is_not_judged_before_it_means_anything(tmp_path):
    """One missed night out of three is 67% and says nothing. The ratio is
    only judged once the window is long enough to carry it."""
    from prosignal.validation.forward import COVERAGE_GRACE_SESSIONS
    start = dt.date(2026, 8, 24)
    open_test(tmp_path, started=start)
    printed = _sessions(start, 5)
    prog = progress(tmp_path, [row(printed[0])], today=printed[-1],
                    sessions_printed=COVERAGE_GRACE_SESSIONS - 1)
    assert not prog.broken


def test_duplicate_runs_on_one_date_do_not_manufacture_coverage(tmp_path):
    """A one-day NSE holiday re-runs the previous session. Coverage counts
    distinct market dates, so the duplicate must not buy back a missing day."""
    start = dt.date(2026, 8, 24)
    open_test(tmp_path, started=start)
    printed = _sessions(start, 40)
    rows = [row(d) for d in printed[:10]] + [row(printed[0])] * 30
    prog = progress(tmp_path, rows, today=printed[-1], sessions_printed=40)
    assert prog.sessions_elapsed == 10
    assert prog.runs_recorded == 40
    assert prog.broken, "40 rows over 10 dates is still 25% coverage"


# ------------------------------------------------------ the window helper
def test_sessions_in_window_counts_only_what_falls_inside(tmp_path):
    from prosignal.validation.forward import sessions_in_window
    days = _sessions(dt.date(2026, 8, 3), 40)
    n = sessions_in_window(days, "2026-08-17", dt.date(2026, 8, 28))
    assert n == sum(1 for d in days
                    if dt.date(2026, 8, 17) <= d <= dt.date(2026, 8, 28))
    assert sessions_in_window([], "2026-08-17") == 0
    assert sessions_in_window(days, "not-a-date") == 0
