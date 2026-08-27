"""The unattended path, checked as text because that is what cron executes.

Three defects lived here and none of them were visible from Python:

  * cron runs Mon-Fri with no holiday calendar. On a one-day NSE holiday the
    ingest fetches nothing and exits 0, the staleness gate counts a single
    weekday and passes, and the analysis re-ranks the previous session --
    writing a second ledger row for a date that already has one.
  * the only notification the system has ever had is a line in a log file, so
    silence and success are the same signal.
  * that log file is appended forever and never rotated.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

SCRIPT = Path("scripts/forward_run.sh")


@pytest.fixture(scope="module")
def script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_the_nightly_analysis_declines_a_session_already_recorded(script):
    body = script[script.index("analyse run"):]
    assert "--skip-if-recorded" in script.split("analyse run")[1][:120], (
        "without it, every NSE holiday adds a duplicate ledger row")


def test_a_person_pressing_scan_still_gets_the_rerun():
    """The flag is opt-in. The API starts the pipeline directly and must not
    inherit a nightly job's reluctance to re-rank."""
    api = Path("src/prosignal/api.py").read_text(encoding="utf-8")
    assert "skip_if_recorded" not in api


def test_failures_reach_something_other_than_the_log(script):
    assert "PROSIGNAL_ALERT_CMD" in script
    for failure in ("ingest FAILED", "analysis FAILED"):
        line = [l for l in script.splitlines() if failure in l]
        assert line and line[0].strip().startswith("alert "), (
            f"{failure!r} still only reaches the log file")


def test_an_unset_notifier_changes_nothing(script):
    """A deployment that has not configured one must behave exactly as before,
    and a broken notifier must not become a missing observation."""
    hook = script[script.index("alert() {"):script.index("export ARROW")]
    assert 'if [ -n "${PROSIGNAL_ALERT_CMD:-}" ]' in hook
    assert "|| true" in hook


def test_a_void_forward_test_is_reported_rather_than_logged(script):
    """`research forward` exits non-zero on a broken window. Every night it
    runs unnoticed on one is a night of observations that are not evidence."""
    tail = script[script.index("research forward"):]
    assert "FORWARD TEST INVALID" in tail
    assert "alert " in tail.split("\n")[1] or "alert " in tail[:400]


def test_the_log_is_rotated(script):
    head = script[:script.index("say()")]
    assert "wc -c" in head and "mv " in head, "forward.log grew without bound"


def test_a_pause_is_still_honoured_and_still_written_down(script):
    body = script[script.index("cron.paused"):]
    assert "PAUSED by operator" in body
    assert "exit 0" in body


def test_the_script_still_stops_dead_on_a_failed_ingest(script):
    """A run scored on a half-updated store is worse than a missing
    observation: the missing one is visible in the session count."""
    assert "set -euo pipefail" in script
    ingest = script[script.index("data ingest"):]
    assert "exit 1" in ingest[:300]


# --------------------------------------------------- the CLI side of the flag
def test_skip_if_recorded_resolves_the_same_session_the_pipeline_would(tmp_path):
    """It must ask about the session the run would actually rank, not about
    today -- the pipeline resolves back to the last stored session."""
    import inspect

    from prosignal import cli
    src = inspect.getsource(cli._already_recorded)
    assert "last_session_on_or_before" in src and "calendar.last" in src
    assert 'row.get("error")' in src, "a failed run is not a recorded observation"


def test_an_unreadable_ledger_means_run_not_skip():
    """Skipping on an error would turn a broken ledger into a silent gap in
    the observation record."""
    import inspect

    from prosignal import cli
    src = inspect.getsource(cli._already_recorded)
    tail = src[src.index("except Exception"):]
    assert "return None" in tail
