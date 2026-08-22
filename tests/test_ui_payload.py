"""Every field the table renders must be in the payload the API sends.

The signals table was numbered by `rank`, the display position after Stage 5
penalties re-sort the survivors, while admission runs on `model_rank`. The two
disagree whenever a penalty lands, so the model's #1 pick appeared eighth and
the buys were scattered through rows ordered by a different quantity.

Numbering by model_rank fixed the ordering and broke the display: the API
serialised only `rank`, so the column rendered "undefined" on every row. A
column name and a payload key drifting apart is invisible to Python tests and
to the type checker, which is why the contract is asserted here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "prosignal" / "static" / "index.html"
API = ROOT / "src" / "prosignal" / "api.py"


def _column_keys() -> set:
    """The `k` of every entry in the UI's COLS table."""
    html = UI.read_text(encoding="utf-8")
    block = html[html.index("const COLS=["):]
    block = block[:block.index("];")]
    return set(re.findall(r'\{k:"([a-z_0-9]+)"', block))


def _payload_keys() -> set:
    return set(re.findall(r'"([a-z_0-9]+)":\s*rec\.', API.read_text(encoding="utf-8")))


def test_every_rendered_column_is_serialised():
    missing = _column_keys() - _payload_keys() - {"ticker", "decision"}
    assert not missing, (
        f"the table renders {sorted(missing)} but the API does not send "
        f"{'it' if len(missing) == 1 else 'them'}; the column will read "
        f"'undefined' on every row"
    )


def test_model_rank_is_sent_because_the_table_numbers_by_it():
    assert "model_rank" in _payload_keys()
    assert "model_rank" in _column_keys()


def test_rank_is_still_sent_since_the_card_reports_the_demotion():
    """The expanded card says 'the model placed it #N ... it now sits #M'.
    Dropping either number makes that sentence unwriteable."""
    assert "rank" in _payload_keys()


@pytest.mark.parametrize("dead", ["momentum_12_1", "value", "rr"])
def test_dead_columns_are_not_rendered(dead):
    """R:R was t1_r_multiple restated -- 1.50 on all 52 cards. Mom and Value
    looked up hand-composite factor names the fitted model replaced, present in
    0 of 52 cards. All three rendered a dash or a constant forever."""
    assert dead not in _column_keys()
