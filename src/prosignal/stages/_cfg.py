"""Config access helper shared by the stages.

`parameters.yaml` mixes two shapes: research parameters are `Tunable` objects
carrying provenance (`.value`, `.status`, ...), while pure switches are bare
scalars. A boolean `enabled: true` has no search range or evidence tier, so
wrapping it would add noise.

`v()` accepts either shape and returns the underlying value, so a stage does not
have to track which form a key uses. Real parameters are still read as `.value`
at the call site; `v()` covers the places where the shape varies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["v", "fv", "iv", "bv"]


def v(node: Any) -> Any:
    """Unwrap a Tunable, or pass a bare scalar through unchanged."""
    return node.value if hasattr(node, "value") else node


def fv(node: Any) -> float:
    return float(v(node))


def iv(node: Any) -> int:
    return int(v(node))


def bv(node: Any) -> bool:
    return bool(v(node))
