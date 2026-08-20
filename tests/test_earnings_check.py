"""Stage 5's earnings check, now that confirmed results dates exist.

It reported NOT_TESTABLE for as long as the only dates on file were yfinance
estimates projected from past quarters. NSE's event-calendar carries dates the
company itself filed, which is what a hard rejection needs to stand on.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from prosignal.core.enums import CheckOutcome
from prosignal.stages.stage5_false_signal import _earnings_distortion


class _Cfg:
    def __init__(self, ahead=5, behind=3, penalty=0.15):
        self.upcoming_earnings_sessions = ahead
        self.recent_earnings_sessions = behind
        self.recent_earnings_penalty = penalty


class _Cal:
    """Calendar sessions approximated as weekdays, which is enough here."""
    @staticmethod
    def sessions_until(start, end):
        import numpy as np
        return max(0, int(np.busday_count(start, end)))


def _cal(sym, dates, confirmed=True):
    return pd.DataFrame({
        "symbol": [sym] * len(dates),
        "earnings_date": pd.to_datetime(dates),
        "confirmed": [confirmed] * len(dates),
        "source": ["nse_board_meetings" if confirmed else "yfinance"] * len(dates),
    })


AS_OF = dt.date(2026, 8, 18)


def test_results_due_shortly_hard_rejects():
    frame = _cal("ACME", ["2026-08-20"])
    r = _earnings_distortion(frame, "ACME", AS_OF, _Cal(), _Cfg())
    assert r.outcome is CheckOutcome.HARD_REJECT


def test_results_far_ahead_pass():
    frame = _cal("ACME", ["2026-11-05"])
    r = _earnings_distortion(frame, "ACME", AS_OF, _Cal(), _Cfg())
    assert r.outcome is CheckOutcome.PASS


def test_results_just_behind_penalise_for_drift():
    frame = _cal("ACME", ["2026-08-17"])
    r = _earnings_distortion(frame, "ACME", AS_OF, _Cal(), _Cfg())
    assert r.outcome is CheckOutcome.SCORE_PENALTY
    assert r.penalty == pytest.approx(0.15)


def test_an_estimate_alone_cannot_support_a_rejection():
    """The whole reason this check was NOT_TESTABLE. An unconfirmed date must
    not hard-reject a name, and must not silently pass either."""
    frame = _cal("ACME", ["2026-08-20"], confirmed=False)
    r = _earnings_distortion(frame, "ACME", AS_OF, _Cal(), _Cfg())
    assert r.outcome is CheckOutcome.NOT_TESTABLE
    assert "estimated" in r.reason


def test_a_name_with_no_date_on_file_is_not_testable():
    frame = _cal("OTHER", ["2026-08-20"])
    r = _earnings_distortion(frame, "ACME", AS_OF, _Cal(), _Cfg())
    assert r.outcome is CheckOutcome.NOT_TESTABLE


def test_an_empty_calendar_is_not_testable_rather_than_a_pass():
    r = _earnings_distortion(pd.DataFrame(), "ACME", AS_OF, _Cal(), _Cfg())
    assert r.outcome is CheckOutcome.NOT_TESTABLE


def test_the_removed_checks_are_gone_from_the_codebase():
    """No dead code: neither had a reachable source, so neither survives."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for dead in ("insider_activity", "regulatory_shock"):
        for path in list((root / "src").rglob("*.py")) + [root / "config" / "parameters.yaml"]:
            assert dead not in path.read_text(encoding="utf-8"), f"{dead} still in {path}"
