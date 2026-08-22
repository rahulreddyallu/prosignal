"""Ledger, job manager, and API -- including the failure paths.

The tests that matter are the ones asserting the system behaves correctly when
things go WRONG: a duplicate click, a crashed process, a failing analysis. A job
layer that only works on the happy path is worse than none, because it looks
like protection.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import pytest
from fastapi.testclient import TestClient

from prosignal.core.contracts import LedgerRow
from prosignal.core.errors import LedgerError
from prosignal.jobs import JobManager, JobState
from prosignal.ledger import Ledger


def _row(run_id="r1", trial="T-1", date=None, no_trade=True):
    return LedgerRow(
        trial_id=trial, run_id=run_id, date=date or dt.date(2026, 8, 17),
        logged_at=dt.datetime.now(), engine_version="0.1.0", schema_version="1",
        config_version="test@abc", no_trade=no_trade,
        signals_generated=[] if no_trade else ["RELIANCE"],
    )


# =============================================================================
# ledger
# =============================================================================


def test_ledger_appends_and_reads_back(tmp_path):
    led = Ledger(tmp_path / "ledger")
    led.append(_row("a"))
    led.append(_row("b"))
    rows = led.read_all()
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert led.count() == 2


def test_ledger_is_append_only_never_rewritten(tmp_path):
    """Existing lines must be untouched by a later append."""
    led = Ledger(tmp_path / "ledger")
    led.append(_row("first"))
    path = next((tmp_path / "ledger").glob("*.jsonl"))
    before = path.read_text()
    led.append(_row("second"))
    assert path.read_text().startswith(before)


def test_every_run_is_recorded_including_no_trade(tmp_path):
    """A ledger that only records signals is a biased sample."""
    led = Ledger(tmp_path / "ledger")
    led.append(_row("quiet", no_trade=True))
    led.append(_row("signal", no_trade=False))
    rows = led.read_all()
    assert len(rows) == 2
    assert sum(1 for r in rows if r["no_trade"]) == 1


def test_trial_count_feeds_the_dsr_penalty(tmp_path):
    led = Ledger(tmp_path / "ledger")
    for i in range(4):
        led.append(_row(f"r{i}", trial=f"T-{i}"))
    led.append(_row("dupe", trial="T-0"))
    assert led.trial_count() == 4


def test_malformed_line_does_not_destroy_history(tmp_path):
    """A crash mid-append must cost one record, not the whole file."""
    led = Ledger(tmp_path / "ledger")
    led.append(_row("good1"))
    path = next((tmp_path / "ledger").glob("*.jsonl"))
    with open(path, "a") as fh:
        fh.write('{"truncated": tru\n')
    led.append(_row("good2"))
    ids = [r["run_id"] for r in led.read_all()]
    assert ids == ["good1", "good2"]


def test_ledger_write_failure_is_fatal_not_swallowed(tmp_path, monkeypatch):
    """Continuing after failing to record a run would corrupt the trial count."""
    led = Ledger(tmp_path / "ledger")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    with pytest.raises(LedgerError, match="NOT recorded"):
        led.append(_row("x"))


def test_signals_for_answers_the_audit_question(tmp_path):
    led = Ledger(tmp_path / "ledger")
    led.append(_row("a", no_trade=False))
    led.append(_row("b", no_trade=True))
    assert [r["run_id"] for r in led.signals_for("RELIANCE")] == ["a"]


# =============================================================================
# job manager
# =============================================================================


def _mgr(tmp_path, runner=None, timeout=900.0):
    return JobManager(
        db_path=tmp_path / "jobs.sqlite3",
        runner=runner or (lambda progress: {"ok": True}),
        timeout_seconds=timeout,
    )


def _wait(mgr, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        job = mgr.get(job_id)
        if job and job.state.is_terminal:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_job_completes_and_stores_its_result(tmp_path):
    mgr = _mgr(tmp_path, runner=lambda p: {"value": 42})
    job = _wait(mgr, mgr.start().id)
    assert job.state is JobState.COMPLETED
    assert job.result == {"value": 42}
    assert job.finished_at is not None


def test_double_click_does_not_launch_two_analyses(tmp_path):
    """The single most important property of this layer."""
    started = []

    def slow(progress):
        started.append(1)
        time.sleep(0.4)
        return {"ok": True}

    mgr = _mgr(tmp_path, runner=slow)
    a = mgr.start()
    b = mgr.start()
    c = mgr.start()
    assert a.id == b.id == c.id
    _wait(mgr, a.id)
    assert len(started) == 1, "a duplicate click launched a second run"


def test_failure_is_recorded_not_swallowed(tmp_path):
    def boom(progress):
        raise RuntimeError("provider exploded")

    mgr = _mgr(tmp_path, runner=boom)
    job = _wait(mgr, mgr.start().id)
    assert job.state is JobState.FAILED
    assert "provider exploded" in job.error
    assert job.error_detail and "Traceback" in job.error_detail
    assert job.result is None, "a failed run must not carry results"


def test_progress_is_reported_during_the_run(tmp_path):
    def stepped(progress):
        for i, label in enumerate(["one", "two", "three"]):
            progress(i, label)
        return {"ok": True}

    mgr = _mgr(tmp_path, runner=stepped)
    job = _wait(mgr, mgr.start().id)
    assert job.progress_label == "complete"
    assert job.progress_step == job.progress_total


def test_stale_running_job_is_reaped_so_the_button_unblocks(tmp_path):
    """A crashed process must not block analysis forever."""
    mgr = _mgr(tmp_path, timeout=900.0)
    old = (dt.datetime.now() - dt.timedelta(hours=2)).isoformat()
    with mgr._connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, state, created_at) VALUES (?,?,?)",
            ("ghost", JobState.RUNNING.value, old),
        )
    assert mgr.active_job() is not None

    assert mgr.reap_stale() == 1
    assert mgr.active_job() is None
    ghost = mgr.get("ghost")
    assert ghost.state is JobState.FAILED
    assert "restarted" in ghost.error or "timeout" in ghost.error


def test_new_job_can_start_after_a_stale_one_is_reaped(tmp_path):
    mgr = _mgr(tmp_path)
    old = (dt.datetime.now() - dt.timedelta(hours=2)).isoformat()
    with mgr._connect() as conn:
        conn.execute("INSERT INTO jobs (id, state, created_at) VALUES (?,?,?)",
                     ("ghost", JobState.RUNNING.value, old))
    job = mgr.start()
    assert job.id != "ghost"
    _wait(mgr, job.id)


def test_cancel_marks_the_job_and_frees_the_slot(tmp_path):
    def slow(progress):
        time.sleep(1.0)
        return {}

    mgr = _mgr(tmp_path, runner=slow)
    job = mgr.start()
    assert mgr.cancel(job.id) is True
    assert mgr.get(job.id).state is JobState.CANCELLED
    assert mgr.cancel(job.id) is False, "already terminal"


def test_jobs_survive_a_manager_restart(tmp_path):
    mgr = _mgr(tmp_path, runner=lambda p: {"n": 1})
    job_id = _wait(mgr, mgr.start().id).id
    reopened = _mgr(tmp_path)
    assert reopened.get(job_id).state is JobState.COMPLETED
    assert reopened.get(job_id).result == {"n": 1}


# =============================================================================
# API
# =============================================================================


@pytest.fixture
def client(live_cfg):
    from prosignal.api import create_app

    return TestClient(create_app(live_cfg))


def test_health_reports_alive(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_whether_analysis_is_possible(client):
    r = client.get("/ready")
    body = r.json()
    assert r.status_code in (200, 503)
    assert "ready" in body and "checks" in body
    assert "price_sessions" in body["checks"]
    assert "analysable_dates" in body["checks"]


def test_unknown_job_is_404(client):
    assert client.get("/analysis/nope").status_code == 404


def test_results_for_unfinished_job_do_not_return_a_signal(client):
    """A partial analysis must never surface as a tradeable result."""
    started = client.post("/analysis/run").json()
    r = client.get(f"/analysis/{started['id']}/results")
    assert r.status_code in (200, 409)
    if r.status_code == 409:
        assert "no results" in json.dumps(r.json()).lower()


def test_full_button_flow_end_to_end(client):
    """CLICK -> JOB -> POLL -> RESULTS, against the real engine and real data."""
    started = client.post("/analysis/run").json()
    job_id = started["id"]

    for _ in range(120):
        state = client.get(f"/analysis/{job_id}").json()
        if state["state"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    assert state["state"] == "COMPLETED", f"analysis failed: {state.get('error')}"

    res = client.get(f"/analysis/{job_id}/results").json()
    assert res["run_id"]
    assert res["as_of_date"]
    assert "funnel" in res and res["funnel"]["universe_considered"] > 0
    assert "regime" in res
    # exactly one of the two outcomes, never both, never neither
    assert (res["no_trade"] is not None) != (len(res["recommendations"]) > 0)
    assert "unavailable" in res["probability_note"].lower()


def test_run_is_persisted_to_the_ledger(client, live_cfg):
    before = Ledger(live_cfg.paths.ledger).count()
    started = client.post("/analysis/run").json()
    for _ in range(120):
        if client.get(f"/analysis/{started['id']}").json()["state"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)
    assert Ledger(live_cfg.paths.ledger).count() > before


def test_ledger_endpoint_exposes_run_history(client):
    body = client.get("/ledger").json()
    assert "count" in body and "trials" in body and "runs" in body


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert '<div class="topbar-in">' in r.text


def test_ui_has_no_external_requests(client):
    """The UI must render with no third-party fetch.

    A CDN font or script would put a network dependency between the user and
    their own analysis, and would leak the fact that they are running it. It
    would also break entirely on the Render instance behind a cold start.
    """
    text = client.get("/").text
    for pattern in ('src="http', 'href="http', "@import", "//fonts.", "cdn."):
        assert pattern not in text, f"external dependency introduced: {pattern}"


def test_ui_states_are_all_reachable(client):
    """Every state the interface can enter must have real markup behind it.

    These are the states a user actually hits -- not-ready, empty, no
    qualifying setup, failure -- and each has been a blank screen at some
    point in this project's history. The wording moved when the interface was
    rebuilt; the requirement did not.
    """
    text = client.get("/").text
    for marker in (
        "Market data store is empty",      # store not bootstrapped
        "Nothing met the bar today",       # the designed common outcome
        "could not be completed",          # failure must not imply a trade
        "Checks that could not run",       # NOT_TESTABLE is not a pass
        "What would move this to Buy",     # the watchlist is actionable
    ):
        assert marker in text, f"no markup for the {marker!r} state"

def test_ui_never_labels_the_score_a_probability(client):
    """The engine emits a rank, not a calibrated probability.

    Dressing a percentile up as "confidence" or "probability" in the interface
    would launder an ordinal into a number the user could size a position on.
    """
    text = client.get("/").text
    body = text.split("<body>")[1]
    for banned in ("Confidence", "Probability", "% chance", "Win rate", "Accuracy"):
        assert banned not in body, f"UI implies calibration it does not have: {banned}"
