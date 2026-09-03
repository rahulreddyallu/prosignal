"""The engine has an unmitigated momentum-crash exposure, and must keep saying so.

Phase 6 measured the one real crash in this sample. The top decile
underperformed in 13 of 15 periods through the COVID rebound, worst -13.14%.
The risk family AMPLIFIED it -- -9.82% with, -5.43% without -- because low-beta,
low-drawdown tilts point away from exactly the beaten-down names that lead a
rebound. The regime gate recovered +0.62 points of -4.39%.

Nothing was retuned, on fifteen observations of one event. These tests pin the
exposure so it cannot be quietly forgotten, and pin the reason the standard
validation cannot see it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosignal.config.loader import load_config

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "findings" / "PHASE6_MOMENTUM_CRASH.md"
BASELINE = ROOT / "docs" / "findings" / "BASELINE_V1.json"


def test_the_daniel_moskowitz_state_is_still_on_the_no_entry_list():
    """It recovers only +0.62 points, but removing it makes things worse."""
    cfg = load_config()
    buckets = set(cfg.params.stage2_regime.no_new_entry_buckets.value)
    assert "uptrend_highvol_rebound" in buckets, (
        "the Daniel & Moskowitz momentum-crash state was removed from "
        "no_new_entry_buckets. It blocked the four worst acute-phase periods "
        "in 2020; partial protection is not no protection."
    )


def test_the_crash_exposure_is_recorded_in_the_baseline():
    limits = _plain(" ".join(json.loads(BASELINE.read_text())["known_limits"]).lower())
    assert "crash" in limits, "the baseline no longer records the crash exposure"
    assert "13.1" in limits or "-13" in limits, (
        "the baseline must keep the magnitude, not just the word"
    )


def test_the_note_records_that_validation_excludes_the_crash():
    """The most important sentence in the note: every other figure in this
    repository was computed on a window with no crash in it."""
    text = NOTE.read_text(encoding="utf-8").lower()
    for claim in ("2022-01", "excludes", "training window"):
        assert claim in text, f"the note no longer records {claim!r}"


def _plain(text: str) -> str:
    """Normalise the dashes. The note is written with en-dashes; a test that
    greps for ASCII hyphens fails on formatting rather than on content."""
    return text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


def test_the_note_records_that_the_risk_family_amplified_the_loss():
    text = _plain(NOTE.read_text(encoding="utf-8").lower())
    assert "amplified" in text
    assert "-9.82%" in text and "-5.43%" in text, (
        "the note must keep both numbers; the finding is the difference"
    )


def test_nothing_was_refitted_to_the_crash():
    """Guards against a later fix that tunes regime buckets on one event."""
    text = NOTE.read_text(encoding="utf-8").lower()
    assert "none of that was done" in text or "was not done" in text, (
        "the note must keep recording that no defence was fitted to n=15"
    )
