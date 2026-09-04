"""The horizon sits on a plateau, and the config must stay inside it.

Phase 13 measured annualised net return across horizons at portfolio level,
with real turnover and a participation-scaled cost:

    H=21 +11.65%   H=42 +11.45%   H=63 +11.61%
    H=84 +11.20%   H=126 +9.93%   H=189 +7.29%

Flat from 21 to 84 within 0.45 percentage points. That flatness is the evidence
that 63 is robust rather than tuned -- and it is also why the horizon must not
be nudged for a fraction of a Sharpe, since the CPCV spread has sd 0.83.

The gross IC table, which rises monotonically to H=189, would have chosen the
worst horizon net of costs. This file exists so that table cannot be read alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosignal.config.loader import load_config

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "findings" / "PHASE13_HORIZON.md"

#: Horizons whose annualised net return is within half a point of the best.
PLATEAU = range(21, 85)


def test_the_configured_horizon_is_inside_the_measured_plateau():
    cfg = load_config()
    h = int(cfg.params.stage4_core_score.model_horizon_sessions)
    assert h in PLATEAU, (
        f"model_horizon_sessions={h} sits outside the 21-84 session plateau "
        f"where annualised net return is flat. Outside it, net return falls "
        f"away: +9.93% at 126 and +7.29% at 189."
    )


def test_the_holding_period_still_matches_the_forecast():
    """A model forecasting 63 sessions while the engine sells at 21 is two
    different systems. The loader enforces this; the test states why."""
    cfg = load_config()
    h = int(cfg.params.stage4_core_score.model_horizon_sessions)
    m = int(cfg.params.stage7_risk.holding_period.max_holding_sessions.value)
    assert h == m


def test_the_note_records_that_gross_ic_would_pick_the_worst_horizon():
    assert NOTE.is_file()
    text = NOTE.read_text(encoding="utf-8").lower()
    assert "189" in text and "worst" in text, (
        "the note must keep recording that the gross IC table nominates H=189 "
        "and that H=189 is the worst horizon net of costs"
    )
    for claim in ("plateau", "per-session", "cost amortisation"):
        assert claim in text, f"the note no longer records {claim!r}"


def test_the_plateau_range_is_not_silently_widened():
    """If someone widens PLATEAU to admit a horizon, the measurement should be
    redone rather than the constant edited."""
    assert PLATEAU.start == 21 and PLATEAU.stop == 85, (
        "the plateau bounds come from a measurement in PHASE13_HORIZON.md. "
        "Changing them means re-running the sweep, not editing this line."
    )
