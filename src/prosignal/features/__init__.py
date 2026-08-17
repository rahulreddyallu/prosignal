"""Derived features computed from stored data.

Separate from `indicators/` on purpose: indicators are pure functions over
price series with no notion of when a fact became public. Fundamentals carry a
filing date, and the point-in-time gate that enforces it is the whole reason
this package exists.
"""

from __future__ import annotations

from .fundamentals import FEATURE_NAMES, compute_features, point_in_time_snapshot

__all__ = ["FEATURE_NAMES", "compute_features", "point_in_time_snapshot"]
