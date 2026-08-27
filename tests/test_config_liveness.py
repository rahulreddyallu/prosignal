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


# ------------------------------------------------ name-level ambiguity
def test_no_leaf_name_is_declared_twice_without_being_declared_shared():
    """The blind spot in the check above, closed.

    Liveness matches by NAME, so a parameter read in one section and ignored in
    another reads as live. `stage4_core_score.metalabel.min_win_probability`
    was exactly that: read by nothing, passing on every startup because
    `stage8_final_signal.scarcity.min_win_probability` is read. Setting it
    looked like arming the NO TRADE veto and did nothing at all.
    """
    from prosignal.config.liveness import ambiguous_parameters

    ambiguous = ambiguous_parameters(CONFIG)
    assert not ambiguous, (
        "these leaf names are declared at more than one path, so the liveness "
        "check cannot tell whether every copy is read:\n  "
        + "\n  ".join(f"{name}: {', '.join(paths)}"
                      for name, paths in sorted(ambiguous.items()))
        + "\n\nIf the two mean different things (every check has an `enabled`), "
          "add the name to liveness.SHARED_LEAF_NAMES. If they are one setting "
          "written twice, delete the copy that nothing reads."
    )


def test_the_shared_list_does_not_outlive_the_duplicates():
    """The other direction, so the allowlist cannot quietly accumulate."""
    from prosignal.config.liveness import SHARED_LEAF_NAMES, declared_paths

    paths = declared_paths(CONFIG)
    stale = sorted(n for n in SHARED_LEAF_NAMES if len(paths.get(n, [])) < 2)
    assert not stale, (
        "these are on SHARED_LEAF_NAMES but are no longer declared twice:\n  "
        + "\n  ".join(stale)
        + "\n\nRemove them; the list is a record of real ambiguity, and an "
          "entry that no longer applies is one more thing that reads as "
          "checked when it is not."
    )


def test_the_ambiguity_detector_actually_detects():
    """Guards the guard."""
    from prosignal.config.liveness import declared_paths

    paths = declared_paths(CONFIG)
    assert len(paths.get("enabled", [])) > 5, (
        "the detector found no duplicates at all, so it would pass anything"
    )


# ------------------------------------------ inert WITHIN its owning section
def test_no_shared_name_is_unread_by_the_stage_that_declares_it():
    """The blind spot the allowlist opened.

    `SHARED_LEAF_NAMES` says a repeated name is legitimate because the copies
    mean different things. It does not check that each copy is READ. Stage 5
    declared twelve `enabled` flags and read none of them -- every check ran
    unconditionally, so setting one to false changed nothing -- and `enabled`
    passed the name-level check because Stage 4, Stage 6 and the providers read
    the same word.
    """
    from prosignal.config.liveness import (RESERVED_IN_SECTION,
                                           unread_in_owning_section)

    found = unread_in_owning_section(CONFIG, PACKAGE)
    flat = {p.split(" (owner")[0] for paths in found.values() for p in paths}
    undeclared = sorted(flat - set(RESERVED_IN_SECTION))
    assert not undeclared, (
        "these parameters are declared under a stage that never reads them:\n  "
        + "\n  ".join(undeclared)
        + "\n\nWire each one into its own stage, delete it, or add it to "
          "liveness.RESERVED_IN_SECTION with the reason. A flag its own stage "
          "ignores states a behaviour the engine does not have, and the "
          "name-level check cannot see it."
    )


def test_the_section_reserved_list_does_not_outlive_its_entries():
    from prosignal.config.liveness import (RESERVED_IN_SECTION,
                                           unread_in_owning_section)

    found = unread_in_owning_section(CONFIG, PACKAGE)
    flat = {p.split(" (owner")[0] for paths in found.values() for p in paths}
    stale = sorted(set(RESERVED_IN_SECTION) - flat)
    assert not stale, (
        "these are on RESERVED_IN_SECTION but their stage now reads them:\n  "
        + "\n  ".join(stale) + "\n\nRemove them; leaving a wired parameter on "
        "the list hides the fact that it works."
    )


def test_stage_5_actually_honours_its_enabled_flags():
    """Twelve declared, none read. Every check ran regardless."""
    import inspect

    from prosignal.stages import stage5_false_signal as s5

    src = inspect.getsource(s5.run)
    assert "bv(flag.enabled)" in src, (
        "Stage 5's per-stock checks must be gated on their own enabled flag"
    )
    for market in ("regime_transition", "volatility_shock", "momentum_crash"):
        assert f"bv(cfg.{market}.enabled)" in src, (
            f"the market-wide {market} check ignores its enabled flag"
        )
