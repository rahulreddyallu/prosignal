"""Vocabulary shared by every stage.

Stages communicate through these enums rather than free strings, so a typo in
one stage cannot create an unhandled state downstream.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "CheckOutcome",
    "GateResult",
    "TrendRegime",
    "VolTercile",
    "VolContext",
    "BreadthState",
    "Decision",
    "StrengthBand",
    "RegimeCompatibility",
    "RiskCategory",
    "TriggerType",
    "EntryStatus",
    "ExitReason",
    "RejectionReason",
    "FeedStatus",
    "SourceName",
]


class CheckOutcome(str, Enum):
    """Result of a single false-signal / data-quality check.

    NOT_TESTABLE is a first-class outcome, not a synonym for PASS. It means the
    data required to run the check does not exist, and it is reported verbatim
    in the recommendation card's "Not testable with current data" line.
    """

    PASS = "PASS"
    SCORE_PENALTY = "SCORE_PENALTY"
    HARD_REJECT = "HARD_REJECT"
    NOT_TESTABLE = "NOT_TESTABLE"


class GateResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class TrendRegime(str, Enum):
    UPTREND = "Uptrend"
    RANGE_BOUND = "Range-bound"
    DOWNTREND = "Downtrend"

    @property
    def bucket_key(self) -> str:
        return {
            TrendRegime.UPTREND: "uptrend",
            TrendRegime.RANGE_BOUND: "range",
            TrendRegime.DOWNTREND: "downtrend",
        }[self]


class VolTercile(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def bucket_key(self) -> str:
        return {
            VolTercile.LOW: "lowvol",
            VolTercile.MEDIUM: "midvol",
            VolTercile.HIGH: "highvol",
        }[self]


class VolContext(str, Enum):
    """Thenmozhi & Chandra (2013): India VIX is not a symmetric fear gauge.

    "VIX rising during a decline" and "VIX rising during a rally" are different
    states and must not collapse into one "high VIX = bad" rule.
    """

    RISING_IN_DECLINE = "rising-in-decline"
    RISING_IN_RALLY = "rising-in-rally"
    FALLING = "falling"
    STABLE = "stable"


class BreadthState(str, Enum):
    STRONG = "Strong"
    NEUTRAL = "Neutral"
    WEAK = "Weak"


class Decision(str, Enum):
    BUY_CANDIDATE = "BUY CANDIDATE"
    WATCHLIST = "WATCHLIST"
    NO_TRADE = "NO TRADE"


class StrengthBand(str, Enum):
    """Bands, never a 0-100 score: no CPCV-derived scoring exists yet."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class RegimeCompatibility(str, Enum):
    FAVORABLE = "Favorable"
    NEUTRAL = "Neutral"
    UNFAVORABLE = "Unfavorable"


class RiskCategory(str, Enum):
    """Fractional-Kelly *logic* without a Kelly percentage the evidence does
    not support. Never a literal allocation figure."""

    STANDARD = "Standard"
    REDUCED = "Reduced"
    MINIMUM = "Minimum"


class TriggerType(str, Enum):
    PULLBACK = "pullback"
    MA_RECLAIM = "ma_reclaim"
    BREAKOUT = "breakout"
    NONE = "none"


class EntryStatus(str, Enum):
    TRIGGERED = "TRIGGERED"
    WATCHLIST = "WATCHLIST"
    NOT_TRIGGERED = "NOT_TRIGGERED"


class ExitReason(str, Enum):
    """Ordered exactly as the research program's section 6.2 hierarchy.

    `priority` is the rung number; lower fires first.
    """

    THESIS_INVALIDATION = "thesis_invalidation"
    STOP_LOSS_BREACH = "stop_loss_breach"
    NEW_HARD_REJECTION = "new_hard_rejection"
    SEVERE_REGIME_CHANGE = "severe_regime_change"
    SIGNAL_REVERSAL = "signal_reversal"
    TRAILING_STOP = "trailing_stop"
    TARGET_ACHIEVED = "target_achieved"
    TIME_EXPIRATION = "time_expiration"

    @property
    def priority(self) -> int:
        return _EXIT_PRIORITY[self]


_EXIT_PRIORITY = {
    ExitReason.THESIS_INVALIDATION: 1,
    ExitReason.STOP_LOSS_BREACH: 2,
    ExitReason.NEW_HARD_REJECTION: 3,
    ExitReason.SEVERE_REGIME_CHANGE: 4,
    ExitReason.SIGNAL_REVERSAL: 5,
    ExitReason.TRAILING_STOP: 6,
    ExitReason.TARGET_ACHIEVED: 7,
    ExitReason.TIME_EXPIRATION: 8,
}


class RejectionReason(str, Enum):
    """Why a stock left the pipeline. Every drop is attributable."""

    NOT_IN_UNIVERSE = "not_in_universe"
    INSUFFICIENT_HISTORY = "insufficient_history"
    ILLIQUID = "illiquid"
    PRICE_FLOOR = "price_floor"
    PLEDGING_BREACH = "pledging_breach"
    EARNINGS_CONFLICT = "earnings_conflict"
    DATA_QUALITY = "data_quality"
    REGULATORY_COOLDOWN = "regulatory_cooldown"
    MANUAL_EXCLUSION = "manual_exclusion"
    SERIES_NOT_ALLOWED = "series_not_allowed"
    FALSE_SIGNAL_HARD_REJECT = "false_signal_hard_reject"
    FALSE_SIGNAL_PENALTY_CAP = "false_signal_penalty_cap"
    SCORE_THRESHOLD = "score_threshold"
    NOT_TRIGGERED = "not_triggered"
    PORTFOLIO_CORRELATION = "portfolio_correlation"
    SECTOR_CAP = "sector_cap"
    REGIME_NO_NEW_ENTRY = "regime_no_new_entry"
    SIGNAL_CAP = "signal_cap"


class FeedStatus(str, Enum):
    OK = "OK"
    STALE = "STALE"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    DEGRADED = "DEGRADED"  # served by a fallback provider, not the primary


class SourceName(str, Enum):
    NSE_ARCHIVES = "nse_archives"
    NSE_JSON_API = "nse_json_api"
    YFINANCE = "yfinance"
    CSV_IMPORT = "csv_import"
    DERIVED = "derived"
