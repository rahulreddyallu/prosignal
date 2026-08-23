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
                    git_commit="deadbeef" * 5, started_on=started)


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
                   git_commit="x" * 40, overwrite=True)
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
