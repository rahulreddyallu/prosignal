"""A declared parameter must be read by something, or say why not.

Three parameters shipped stating behaviour the engine did not have:
ma_reclaim.reference promised an anchored VWAP nothing computed,
require_above_average_volume promised a volume bar the function was never
passed the data for, and reject_if_overextended promised a distance check no
stage performed. All three validated on every startup. A fourth,
model_refit_every_sessions, was read from a module constant instead of the
config, so editing it changed nothing -- invisible only because the two values
happened to agree.

The guard is deliberately two-directional. Adding a parameter nothing reads
fails; wiring one up and leaving it on the reserved list also fails. Neither
can be satisfied by editing parameters.yaml alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosignal.config.liveness import (
    RESERVED,
    consumed_names,
    declared_parameters,
    inert_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "parameters.yaml"
PACKAGE = ROOT / "src" / "prosignal"


def test_every_parameter_is_read_or_explicitly_reserved():
    inert = set(inert_parameters(CONFIG, PACKAGE))
    reserved = set(RESERVED)

    unlisted = sorted(inert - reserved)
    assert not unlisted, (
        "these parameters are declared in parameters.yaml and read by nothing:\n  "
        + "\n  ".join(unlisted)
        + "\n\nWire each one up, delete it, or add it to "
          "prosignal.config.liveness.RESERVED with the reason it may stay. A "
          "parameter that validates on startup and changes nothing is a claim "
          "the engine does not honour."
    )


def test_reserved_parameters_are_still_actually_inert():
    inert = set(inert_parameters(CONFIG, PACKAGE))
    now_live = sorted(set(RESERVED) - inert)
    assert not now_live, (
        "these are on the reserved list but something now reads them:\n  "
        + "\n  ".join(now_live)
        + "\n\nRemove them from RESERVED -- the list is a record of what is NOT "
          "implemented, and leaving a wired parameter on it hides the fact that "
          "it works."
    )


def test_every_reserved_entry_gives_a_reason():
    for name, reason in RESERVED.items():
        assert reason and len(reason) > 20, (
            f"RESERVED[{name!r}] must say why the parameter is allowed to be "
            f"inert; a bare entry is how the next one gets waved through"
        )


def test_the_detector_sees_a_parameter_the_engine_actually_reads():
    """Guards the guard: a detector that finds nothing would pass everything."""
    declared = declared_parameters(CONFIG)
    consumed = consumed_names(PACKAGE)
    assert "atr_multiple" in declared
    assert "atr_multiple" in consumed, "Stage 7 reads this; the detector missed it"
    assert "entry_rank" in consumed, "Stage 6 reads this; the detector missed it"


def test_section_headers_are_not_counted_as_parameters():
    """`costs` and `validation` group settings; they are not settings."""
    declared = declared_parameters(CONFIG)
    for section in ("costs", "validation", "capital", "providers"):
        assert section not in declared, (
            f"{section!r} is a section, not a parameter; counting it would "
            f"report a permanent false positive"
        )
