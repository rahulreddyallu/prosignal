"""Single source of truth for the engine version.

The engine version is written into every research-ledger row alongside the
config hash, so that a live signal can later be attributed to an exact
(code version, parameter set) pair -- see research program section 3.5 and
section 17 (research ledger).
"""

from __future__ import annotations

__all__ = ["ENGINE_VERSION", "ENGINE_NAME", "SCHEMA_VERSION"]

ENGINE_NAME = "prosignal"

#: Bumped whenever pipeline *logic* changes in a way that could alter output.
ENGINE_VERSION = "0.1.0"

#: Bumped whenever a stage input/output contract changes shape.
#: Ledger rows carry this so historical rows stay interpretable.
SCHEMA_VERSION = "1"
