"""An inferred label must not read as a recorded one.

`Ledger.repair_lineage` stamps each row's `mode` from what its own timestamps
prove -- `logged_at` against `date` -- and writes `mode_source` saying so. That
is the right repair for a field that was never recorded properly, and it does
not turn the label into a recording. Measured on the shipped ledger, **250 of
253 rows (99%)** carry it.

Nothing downstream could see that. `outcomes.load_outcomes` partitions on
`exit_model` and on the research epoch, both recorded at write time, and never
on `mode`; `forward.progress` counts observations without asking how their
liveness was established. So a forward test reporting "N sessions elapsed" was
resting on an inference and had no way to say so.

The fix reports rather than refuses. The inference rule is sound, the window's
own rows are what matter, and a caveat that travels with the count is the
honest treatment -- a hard failure here would invalidate a window for a defect
in how its rows were labelled rather than in what they contain.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from prosignal.ledger import Ledger


def _ledger(tmp_path: Path, rows) -> Ledger:
    d = tmp_path / "ledger"
    d.mkdir()
    (d / "runs-2026.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return Ledger(d)


def _row(day: str, *, reconstructed: bool, mode: str = "replay") -> dict:
    r = {"date": day, "logged_at": f"{day}T18:00:00", "mode": mode,
         "config_version": "cfg@1", "model_fingerprint": "v3/1"}
    if reconstructed:
        r["mode_source"] = "repair_lineage: logged_at vs date"
    return r


def test_a_fully_recorded_ledger_raises_nothing(tmp_path):
    """The detector must be able to say 'fine'."""
    led = _ledger(tmp_path, [_row("2026-01-05", reconstructed=False),
                             _row("2026-01-06", reconstructed=False)])
    p = led.mode_provenance()
    assert p["mode_reconstructed"] == 0
    assert p["reconstructed_share"] == 0.0
    assert led.live_evidence_warning() is None


def test_a_reconstructed_ledger_says_so_and_says_by_what_rule(tmp_path):
    led = _ledger(tmp_path, [_row("2026-01-05", reconstructed=True),
                             _row("2026-01-06", reconstructed=True),
                             _row("2026-01-07", reconstructed=False)])
    p = led.mode_provenance()
    assert p["rows"] == 3
    assert p["mode_reconstructed"] == 2
    assert p["mode_recorded_at_write_time"] == 1
    assert p["reconstructed_share"] == pytest.approx(2 / 3)

    warn = led.live_evidence_warning()
    assert "RECONSTRUCTED" in warn
    assert "logged_at" in warn, "the warning must name the rule it was inferred by"
    assert "still an inference" in warn


def test_the_shipped_ledger_is_almost_entirely_reconstructed():
    """The one that matters. If this ever reads mostly-recorded, the engine has
    started writing its own lineage and a forward test can stop caveating."""
    root = Path(__file__).resolve().parents[1] / "data/ledger"
    if not root.is_dir():
        pytest.skip("no ledger in this checkout")
    p = Ledger(root).mode_provenance()
    assert p["rows"] > 0
    assert p["reconstructed_share"] > 0.9, (
        f"expected the shipped ledger to be overwhelmingly reconstructed; "
        f"got {p['reconstructed_share']:.1%}"
    )


def test_the_forward_test_carries_the_caveat_and_is_not_broken_by_it(tmp_path):
    """Reported, not refused. A hard failure here would void a window for a
    defect in how its rows were LABELLED rather than in what they contain."""
    from prosignal.validation import forward

    prog = forward.Progress(
        started_on="2026-01-01", latest_session="2026-03-01",
        sessions_elapsed=40, sessions_target=250,
        months_elapsed=2, months_target=18, runs_recorded=40,
        reconstructed_mode_rows=39)
    caveats = prog.caveats()
    assert len(caveats) == 1
    assert "39 of 40" in caveats[0]
    assert "sessions_elapsed" in caveats[0]
    assert prog.broken == [], "a labelling defect must not void the window"


def test_no_caveat_when_every_row_recorded_its_own_mode():
    from prosignal.validation import forward

    prog = forward.Progress(
        started_on="2026-01-01", latest_session="2026-03-01",
        sessions_elapsed=40, sessions_target=250,
        months_elapsed=2, months_target=18, runs_recorded=40)
    assert prog.reconstructed_mode_rows == 0
    assert prog.caveats() == []


def test_progress_counts_reconstructed_rows_inside_the_window(tmp_path):
    """It must count the WINDOW's rows, not the ledger's -- a window opened
    after the repair inherits none of it."""
    import inspect

    from prosignal.validation import forward

    src = inspect.getsource(forward.progress)
    assert 'mode_source' in src
    assert "reconstructed_mode_rows=reconstructed" in src
    # counted after the `when < start` guard, so pre-window rows do not count
    assert src.index("if when < start") < src.index("mode_source")
