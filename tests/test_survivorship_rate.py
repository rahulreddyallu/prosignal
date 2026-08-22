"""The delisting rate is the reason the price history is not extended.

Several vendors serve daily candles back to 2000 -- 26 years against the 8.9
held -- which would take the price block from ~30 independent observations to
~100. Their instrument masters carry only currently-listed securities and
publish no archived versions, so that reconstruction would contain only
companies still listed today.

Measured on the local store, which WAS ingested progressively and therefore
holds names that have since stopped trading, the disappearance rate is 4.0-4.6%
per year across four independent windows. Over 26 years that is 68% of the
then-universe missing. These tests pin the measurement so the extension cannot
be revisited on the argument that the bias is small.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "research" / "HISTORY_EXTENSION.md"

#: Measured on this store, four windows 2017-2023.
ANNUAL_DISAPPEARANCE = 0.043


def test_the_measured_rate_implies_a_disqualifying_bias_at_26_years():
    missing = 1.0 - (1.0 - ANNUAL_DISAPPEARANCE) ** 26
    assert missing > 0.60, (
        f"at {ANNUAL_DISAPPEARANCE:.1%}/yr, a 26-year reconstruction from "
        f"today's instrument master is missing {missing:.0%} of the companies "
        f"that actually traded. That is not a sample, it is a survivor list."
    )


def test_a_short_extension_is_the_only_tolerable_one():
    """NSE serves natively from 2016-01, about 1.7 years beyond the store."""
    missing = 1.0 - (1.0 - ANNUAL_DISAPPEARANCE) ** 1.7
    assert missing < 0.10, (
        "a 1.7-year extension carries under 10% missing names, which is "
        "tolerable if stated; anything longer is not"
    )


def test_the_note_records_why_a_long_extension_was_refused():
    text = NOTE.read_text(encoding="utf-8").lower()
    for claim in ("68%", "delisted", "instrument master", "artefact"):
        assert claim in text, f"the note no longer records {claim!r}"


def test_the_note_records_that_broker_statements_lack_filing_dates():
    """The finding that decides whether n=11 moves. It does not."""
    text = NOTE.read_text(encoding="utf-8").lower()
    assert "filing date" in text
    assert "period-end" in text or "period end" in text


def test_the_note_states_the_rule_rather_than_only_the_measurement():
    text = NOTE.read_text(encoding="utf-8").lower()
    assert "point-in-time universe" in text, (
        "the note must keep the rule -- extend only as far as a point-in-time "
        "universe can be reconstructed -- not just the numbers behind it"
    )
