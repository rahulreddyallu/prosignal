"""Config access helper shared by the stages.

`parameters.yaml` mixes two shapes: research parameters are `Tunable` objects
carrying provenance metadata (`.value`, `.status`, ...), while pure switches are
bare scalars. Both are legitimate -- a boolean `enabled: true` has no search
range and no evidence tier, so wrapping it in a Tunable would be noise.

`v()` accepts either and returns the underlying value, so a stage never has to
remember which shape a given key uses. Reading `.value` directly at the call
site was a deliberate design choice in chunk 1 (a constant reminder that the
number is a hypothesis); that still holds for real parameters, and `v()` is for
the places where the shape genuinely varies.
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
