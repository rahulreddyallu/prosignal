"""Cross-cutting primitives: errors, paths, logging, enums, calendar, contracts."""

from __future__ import annotations

from .enums import (
    BreadthState,
    CheckOutcome,
    Decision,
    EntryStatus,
    ExitReason,
    FeedStatus,
    GateResult,
    RegimeCompatibility,
    RejectionReason,
    RiskCategory,
    SourceName,
    StrengthBand,
    TrendRegime,
    TriggerType,
    VolContext,
    VolTercile,
)
from .errors import (
    ConfigError,
    DataError,
    IntegrityError,
    LedgerError,
    MarketWideHalt,
    PipelineError,
    ProSignalError,
    ProviderError,
    StaleDataError,
)
from .logging import bind_run_id, current_run_id, get_logger, setup_logging
from .paths import ProjectPaths, find_project_root

__all__ = [
    "BreadthState",
    "CheckOutcome",
    "Decision",
    "EntryStatus",
    "ExitReason",
    "FeedStatus",
    "GateResult",
    "RegimeCompatibility",
    "RejectionReason",
    "RiskCategory",
    "SourceName",
    "StrengthBand",
    "TrendRegime",
    "TriggerType",
    "VolContext",
    "VolTercile",
    "ConfigError",
    "DataError",
    "IntegrityError",
    "LedgerError",
    "MarketWideHalt",
    "PipelineError",
    "ProSignalError",
    "ProviderError",
    "StaleDataError",
    "bind_run_id",
    "current_run_id",
    "get_logger",
    "setup_logging",
    "ProjectPaths",
    "find_project_root",
]
