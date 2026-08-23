"""Days NSE serves no index for must be probed once, not once per press.

A day that came back empty was left unrecorded, so the next chunk selected
it again. Deep in a backfill the whole 90-session chunk could be dead dates,
which is why the counter crawled by one or two sessions a press near the end
while the early presses moved by ninety.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from prosignal.data.ingest import DataIngestor


def _ing(cfg):
    return DataIngestor(cfg)


def test_a_confirmed_non_session_is_remembered(live_cfg, tmp_path):
    ing = _ing(live_cfg)
    ing._dead_path().unlink(missing_ok=True)
    try:
        assert ing._read_dead_days() == set()
        ing._record_dead_days([dt.date(2024, 1, 26), dt.date(2024, 3, 8)])
        assert ing._read_dead_days() == {dt.date(2024, 1, 26), dt.date(2024, 3, 8)}
        # and it accumulates rather than replacing
        ing._record_dead_days([dt.date(2024, 8, 15)])
        assert len(ing._read_dead_days()) == 3
    finally:
        ing._dead_path().unlink(missing_ok=True)


def test_recording_nothing_writes_nothing(live_cfg):
    ing = _ing(live_cfg)
    ing._dead_path().unlink(missing_ok=True)
    ing._record_dead_days([])
    assert not ing._dead_path().exists()


def test_a_torn_file_does_not_block_the_backfill(live_cfg):
    ing = _ing(live_cfg)
    ing._dead_path().write_text("{ not json", encoding="utf-8")
    try:
        assert ing._read_dead_days() == set()
    finally:
        ing._dead_path().unlink(missing_ok=True)


def test_a_transport_failure_is_never_recorded_as_a_dead_day():
    """Marking one would put a real trading day permanently out of reach.
    Only an archive that ANSWERED and had nothing counts."""
    src = Path("src/prosignal/data/ingest.py").read_text(encoding="utf-8")
    body = src[src.index("index_frame = self.nse.fetch_index_close_all"):]
    body = body[:body.index("buffers[\"indices\"]")]
    assert "failed = True" in body
    assert "if not failed:" in body, "a ProviderError must not mark the day dead"


def test_dead_days_are_skipped_when_choosing_what_to_fetch():
    src = Path("src/prosignal/data/ingest.py").read_text(encoding="utf-8")
    body = src[src.index("def _sessions_to_fetch"):src.index("def _backfill_sessions")]
    assert "dead" in body
    assert "elif day not in dead:" in body
    # A non-session must not count toward the requested depth, or the walk
    # stops short of the depth actually asked for.
    assert body.count("collected += 1") == 2


# ===================================================================
# The calendar constant that capped the store at 2,171
# ===================================================================

def test_the_session_density_allows_for_a_year_nse_actually_trades():
    """It was 1.45, assuming 250 sessions a year. NSE does not deliver 250.

    Measured across 2017-09-08 to 2026-08-21 the real density is 1.480, and
    the gap is not academic: at 1.45 a request for 2,200 sessions was allowed
    3,210 calendar days, which reaches 2017-11-06, and exactly 2,171 sessions
    exist in that window. The build stopped dead at 2,171 and every further
    press re-walked the same wall.
    """
    from prosignal.data.ingest import _DAYS_PER_SESSION
    assert _DAYS_PER_SESSION >= 1.50, (
        "below the real 1.480 density plus slack, a deep backfill silently "
        "stops short of the depth it was asked for"
    )
    # And not so generous that the span outruns the real bound.
    assert int(2200 * _DAYS_PER_SESSION) + 20 < 4200


def test_a_walk_that_runs_out_of_calendar_says_so():
    """It returned fewer days each press and nothing explained why."""
    src = Path("src/prosignal/data/ingest.py").read_text(encoding="utf-8")
    body = src[src.index("def _sessions_to_fetch"):src.index("def _backfill_sessions")]
    assert "if collected < history_sessions:" in body
    assert "span exhausted" in body


def test_the_requested_depth_is_reachable_for_the_validated_target(live_cfg):
    """2,200 is the depth the shipped coefficients were validated on, so the
    backfill has to be able to physically get there."""
    import datetime as dt
    from prosignal.data.ingest import _DAYS_PER_SESSION
    from prosignal.data.store import DataStore

    sessions = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots).price_sessions()
    if len(sessions) < 2200:
        pytest.skip("local store is not deep enough to check reachability")
    span = min(int(2200 * _DAYS_PER_SESSION) + 20,
               int(live_cfg.params.storage.max_backfill_calendar_days))
    cutoff = sessions[-1] - dt.timedelta(days=span)
    assert sum(1 for d in sessions if d >= cutoff) >= 2200
