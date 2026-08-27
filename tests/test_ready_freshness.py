"""`/ready` and the pipeline must agree about whether an analysis can run.

They did not. Observed on 2026-08-27 against a store ending 2026-08-25:

    GET /ready -> 200 {"ready": true, "latest_session": "2026-08-25",
                       "price_data": "ok"}

while `_sessions_behind` gave 2 against a tolerance of 1, so every analysis
halted market-wide at Stage 1. The readiness probe reported ready for something
the engine refused to do -- and the nightly script warms `/ready` after each
run, so the healthy answer was being produced on purpose.

The second defect is in the same response. `latest_session` was set only on the
branch where the store had reached full validated depth, and the interface
reads `isCurrent()` off it:

    if (!state.latestSession) return true;   // nothing to compare against

With nothing to compare against, a run from any past date is "current". So on a
still-building store the freshness indicator switched itself off, in exactly
the state where the store is most likely to be behind.
"""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from prosignal.api import create_app
from prosignal.config.loader import load_config
from prosignal.pipeline import _sessions_behind


@pytest.fixture(scope="module")
def checks():
    client = TestClient(create_app(load_config()))
    r = client.get("/ready")
    return r.status_code, r.json()


def test_latest_session_is_reported_on_every_branch(checks):
    _, body = checks
    assert "latest_session" in body["checks"]
    assert body["checks"]["latest_session"], (
        "the interface's staleness check compares against this and treats a "
        "missing value as 'nothing to compare against', which reads as current"
    )


def test_freshness_is_reported_at_all(checks):
    _, body = checks
    for key in ("sessions_behind", "staleness_limit", "data_stale"):
        assert key in body["checks"], f"/ready says nothing about {key}"


def test_ready_agrees_with_the_gate_the_pipeline_halts_on(checks):
    """One arithmetic, one tolerance, one answer."""
    status, body = checks
    c = body["checks"]
    if c.get("latest_session") is None:
        pytest.skip("empty store")
    cfg = load_config()
    from prosignal.core.clock import market_today
    behind = _sessions_behind(dt.date.fromisoformat(c["latest_session"]),
                              market_today(cfg))
    limit = int(cfg.params.feeds["equity_ohlcv"].max_age_sessions)
    assert c["sessions_behind"] == behind
    assert c["staleness_limit"] == limit
    assert c["data_stale"] is (behind > limit)
    if behind > limit:
        assert body["ready"] is False and status == 503, (
            "an analysis started now would halt at Stage 1, so this endpoint "
            "must not call the engine ready"
        )


def test_a_stale_store_is_told_to_refresh_not_to_rebuild(checks):
    """A store too SHORT and a store too OLD are different problems with
    different remedies. Offering a two-hour rebuild for a two-minute download
    is the wrong instruction, and `ready:false` alone cannot tell them apart."""
    _, body = checks
    c = body["checks"]
    if not c.get("data_stale"):
        pytest.skip("store is fresh")
    assert c["model_will_fit"] is True, "this store is deep enough; only old"
    assert "refresh" in c["remedy"].lower()
    assert "bootstrap" not in c["remedy"].lower()


def test_the_interface_splits_the_two_reasons(checks):
    """`notReady` drives the BUILD screen. It must key on depth, not on
    `ready`, or a stale store sends the reader to rebuild the store."""
    from pathlib import Path
    src = Path("src/prosignal/static/index.html").read_text(encoding="utf-8")
    assert "state.notReady = !(c && c.model_will_fit);" in src
    assert "state.notReady = !(j && j.ready);" not in src, (
        "checkReady and pollReady must compute this the same way"
    )
    assert 'onclick="doIngest()"' in src, "the stale path must offer a refresh"
    assert "!state.stale && !isCurrent(v)" in src, (
        "the header scan button must not offer a scan that halts at Stage 1"
    )
