"""Typed exception hierarchy.

Design rule (master prompt, non-negotiable rule #1 and #3): a hard reject is
never an exception that bubbles up as a generic crash, and `NO TRADE` is never
an error. Exceptions in this codebase mean "the engine could not run", never
"the engine ran and found nothing".

    ProSignalError
    ├── ConfigError            -- parameters.yaml is wrong / unloadable
    ├── DataError              -- something is wrong with data we hold
    │   ├── ProviderError      -- a feed could not be fetched
    │   ├── StaleDataError     -- a feed is older than its tolerance
    │   └── IntegrityError     -- data violates a point-in-time invariant
    ├── PipelineError          -- a stage could not execute
    │   └── MarketWideHalt     -- Stage 1 market-wide FAIL: no signals at all
    └── LedgerError            -- the append-only ledger could not be written
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "ProSignalError",
    "ConfigError",
    "DataError",
    "ProviderError",
    "StaleDataError",
    "IntegrityError",
    "PipelineError",
    "MarketWideHalt",
    "LedgerError",
]


class ProSignalError(Exception):
    """Base class for every error this engine raises deliberately."""

    #: Short machine-readable code, surfaced to the API/webapp so the UI can
    #: show a specific message instead of a generic spinner-of-death (FR-9).
    code: str = "PROSIGNAL_ERROR"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = dict(context)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        ctx = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({ctx})"


class ConfigError(ProSignalError):
    """config/parameters.yaml is missing, malformed, or internally inconsistent."""

    code = "CONFIG_ERROR"


class DataError(ProSignalError):
    """Base class for data-layer problems."""

    code = "DATA_ERROR"


class ProviderError(DataError):
    """A named data provider failed to return usable data."""

    code = "PROVIDER_ERROR"

    def __init__(self, provider: str, message: str, **context: Any) -> None:
        super().__init__(message, provider=provider, **context)
        self.provider = provider


class StaleDataError(DataError):
    """A feed's last-updated timestamp is beyond its configured tolerance."""

    code = "STALE_DATA"


class IntegrityError(DataError):
    """A point-in-time / survivorship / adjustment invariant was violated.

    This is the class of error the research program's section 7 checklist exists
    to prevent. It is raised (not warned) because silently proceeding is the
    single most expensive failure mode in a retail-built backtest.
    """

    code = "INTEGRITY_ERROR"


class PipelineError(ProSignalError):
    """A pipeline stage could not execute (bug, contract violation, bad input)."""

    code = "PIPELINE_ERROR"

    def __init__(self, stage: str, message: str, **context: Any) -> None:
        super().__init__(message, stage=stage, **context)
        self.stage = stage


class MarketWideHalt(PipelineError):
    """Stage 1 hard-failed at the market level: no signals generated this run.

    This is NOT the same as NO TRADE. NO TRADE means "the engine ran cleanly and
    nothing qualified". MarketWideHalt means "the engine refuses to form an
    opinion because its inputs cannot be trusted".
    """

    code = "MARKET_WIDE_HALT"

    def __init__(self, reasons: List[str], stage: str = "stage1_data_quality") -> None:
        msg = "Data quality gate failed market-wide: " + "; ".join(reasons)
        super().__init__(stage=stage, message=msg, reasons=reasons)
        self.reasons = reasons


class LedgerError(ProSignalError):
    """The append-only research ledger could not be written.

    Treated as fatal by design: an unlogged run breaks the Deflated Sharpe Ratio
    trial count (research program section 8) and therefore breaks the honesty of
    every subsequent statistical claim.
    """

    code = "LEDGER_ERROR"


def describe_exception(exc: BaseException) -> Dict[str, Any]:
    """Normalise any exception into the dict shape the API returns to the webapp."""
    if isinstance(exc, ProSignalError):
        return exc.to_dict()
    return {
        "code": "UNEXPECTED_ERROR",
        "message": f"{type(exc).__name__}: {exc}",
        "context": {},
    }


def first_or_none(items: Optional[List[Any]]) -> Optional[Any]:  # pragma: no cover
    return items[0] if items else None
