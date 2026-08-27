"""The single job slot must not hand a caller someone else's job.

Single flight is global on purpose -- both job kinds touch the same store, and
analysing one that is being rewritten underneath produces a result from
half-written data. But `start()` returned whichever job held the slot, whatever
its kind, and every caller treated it as its own.

Executed, before the fix:

    started bootstrap : 0214e0a47812 bootstrap QUEUED
    pressed SCAN      -> 0214e0a47812 kind=bootstrap state=RUNNING
    build_view(bootstrap payload) SUCCEEDED
      -> as_of null, buy 0, watch 0, "No names qualified"
    UI: "Scan complete. 0 qualifying, 0 monitored."

No scan ran and no ledger row was written. The interface invited it: the
refresh handler said the scan button would work once the refresh finished, and
left the button enabled.
"""
from __future__ import annotations

import threading
import time

import pytest

from prosignal.jobs import Job, JobBusy, JobManager, JobState


def _mgr(tmp_path, runner=None, timeout=900.0):
    return JobManager(db_path=tmp_path / "jobs.sqlite3",
                      runner=runner or (lambda p: {"ok": True}),
                      timeout_seconds=timeout)


def _blocking(gate):
    def runner(progress):
        progress(0, "working")
        gate.wait(timeout=10)
        return {"sessions_in_store": 2212, "complete": True}
    return runner


# ------------------------------------------------------------------ refusal
def test_a_scan_pressed_during_a_refresh_is_refused_not_substituted(tmp_path):
    gate = threading.Event()
    mgr = _mgr(tmp_path)
    boot = mgr.start(kind="ingest", runner=_blocking(gate))
    try:
        with pytest.raises(JobBusy) as caught:
            mgr.start(kind="analysis")
        assert caught.value.running.id == boot.id
        assert caught.value.wanted == "analysis"
        assert "refresh" in str(caught.value).lower()
    finally:
        gate.set()


def test_the_same_kind_is_still_idempotent(tmp_path):
    """A double click on one button must not launch a second full run."""
    gate = threading.Event()
    mgr = _mgr(tmp_path, runner=_blocking(gate))
    try:
        a = mgr.start()
        b = mgr.start()
        c = mgr.start()
        assert a.id == b.id == c.id
    finally:
        gate.set()


def test_the_message_names_both_jobs_in_words(tmp_path):
    """It reaches the reader verbatim, so it has to read like a sentence."""
    gate = threading.Event()
    mgr = _mgr(tmp_path)
    mgr.start(kind="bootstrap", runner=_blocking(gate))
    try:
        with pytest.raises(JobBusy) as caught:
            mgr.start(kind="analysis")
        msg = str(caught.value)
        assert "data build" in msg and "market scan" in msg
        assert "Wait for it" in msg
    finally:
        gate.set()


# ------------------------------------------------------- cancel that sticks
def test_a_cancelled_job_does_not_come_back_completed(tmp_path):
    """`cancel` cannot stop a Python thread. The worker arrives at the end and
    used to write COMPLETED over CANCELLED -- so the job the operator cancelled
    returned, finished, with a result, minutes later."""
    gate = threading.Event()
    mgr = _mgr(tmp_path, runner=_blocking(gate))
    job = mgr.start()
    for _ in range(100):
        if mgr.get(job.id).state is JobState.RUNNING:
            break
        time.sleep(0.01)
    assert mgr.cancel(job.id) is True
    gate.set()
    mgr._thread.join(timeout=10)
    assert mgr.get(job.id).state is JobState.CANCELLED
    assert mgr.get(job.id).result is None


def test_a_new_job_waits_for_a_cancelled_worker_to_actually_stop(tmp_path):
    """Two full-universe runs overlapping is exactly what single flight exists
    to prevent, and a cancel opened that door: the row went terminal, so
    `active_job()` returned None while the thread kept reading the store and
    still had its ledger row to append."""
    gate = threading.Event()
    mgr = _mgr(tmp_path, runner=_blocking(gate))
    job = mgr.start()
    for _ in range(100):
        if mgr.get(job.id).state is JobState.RUNNING:
            break
        time.sleep(0.01)
    mgr.cancel(job.id)
    with pytest.raises(JobBusy) as caught:
        mgr.start()
    assert caught.value.orphaned
    assert "still finishing" in str(caught.value)
    gate.set()
    mgr._thread.join(timeout=10)
    assert mgr.start().id != job.id          # once it stops, the slot is free


# ------------------------------------------------------------- the API side
def test_the_endpoint_answers_409_rather_than_a_foreign_job(tmp_path):
    from fastapi.testclient import TestClient
    from prosignal.api import create_app
    from prosignal.config.loader import load_config

    # An isolated job database. `JobManager` lives at `paths.data/jobs.sqlite3`,
    # so sharing the project's own file means this test both sees other tests'
    # running jobs and leaves rows in the real one.
    cfg = load_config(use_cache=False)
    cfg.paths.data = tmp_path
    app = create_app(cfg)
    gate = threading.Event()
    app.state.jobs.start(kind="ingest", runner=_blocking(gate))
    # No context manager: the startup hook spawns a daemon thread that warms
    # the performance cache, and it outlives pytest's stream teardown.
    client = TestClient(app)
    try:
        r = client.post("/analysis/run")
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert detail["wanted"] == "analysis"
        assert detail["running"]["kind"] == "ingest"
        assert "Wait for it" in detail["message"]
    finally:
        gate.set()
        if app.state.jobs._thread:
            app.state.jobs._thread.join(timeout=10)
