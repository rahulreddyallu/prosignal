"""Stage 0 data layer: providers, point-in-time store, universe, adjustments."""

from __future__ import annotations

from .corporate_actions import (
    apply_adjustments,
    build_adjustment_factors,
    detect_unexplained_jumps,
    merge_action_sources,
    parse_action_subject,
)
from .ingest import DataIngestor, IngestOptions, IngestResult
from .store import DataStore
from .types import (
    DATE,
    OHLCV_COLUMNS,
    SYMBOL,
    coerce_ohlcv,
    from_wide,
    normalise_symbol,
    to_wide,
    validate_ohlcv,
)
from .universe import UniverseResolver, UniverseSnapshot

__all__ = [
    "apply_adjustments",
    "build_adjustment_factors",
    "detect_unexplained_jumps",
    "merge_action_sources",
    "parse_action_subject",
    "DataIngestor",
    "IngestOptions",
    "IngestResult",
    "DataStore",
    "DATE",
    "SYMBOL",
    "OHLCV_COLUMNS",
    "coerce_ohlcv",
    "validate_ohlcv",
    "to_wide",
    "from_wide",
    "normalise_symbol",
    "UniverseResolver",
    "UniverseSnapshot",
]
