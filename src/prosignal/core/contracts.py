"""Stage input/output contracts.

The master prompt fixes the pipeline order and requires each stage to be a pure
function with a declared schema, so that each can be tested and re-validated
independently. These pydantic models *are* those schemas. They are also the
wire format for the API, so the webapp and the engine can never drift apart:
if a stage adds a field, the card renders it or fails loudly.

    Stage 0  RawDataManifest
    Stage 1  DataQualityReport
    Stage 2  RegimeState
    Stage 3  EligibilityReport
    Stage 4  CoreScoreReport
    Stage 5  FalseSignalReport
    Stage 6  EntryReport
    Stage 7  RiskPlan
    Stage 8  FinalSignalOutput  (Recommendation | NoTradeReport)
             LedgerRow

Conventions enforced here rather than by convention:

* prices are ``Optional[float]`` and are allowed to be ``None`` -- a missing
  level is reported as missing, never as ``0.0``;
* every band/category is an enum, never a free string;
* every check carries an ``evidence`` citation so the "RESEARCH BASIS" section
  of the output writes itself from data rather than from prose in a template.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

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

__all__ = [
    "RunContext",
    "FeedRecord",
    "RawDataManifest",
    "StockDataFlags",
    "DataQualityReport",
    "RegimeState",
    "EligibilityReport",
    "FactorScore",
    "StockScore",
    "RedundancyReport",
    "CoreScoreReport",
    "CheckResult",
    "StockDefenseResult",
    "FalseSignalReport",
    "EntryDecision",
    "EntryReport",
    "ExitCondition",
    "RiskPlan",
    "Recommendation",
    "ClosestCandidate",
    "NoTradeReport",
    "FinalSignalOutput",
    "LedgerRow",
]

_MODEL = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=False)


class _Contract(BaseModel):
    model_config = _MODEL


# =============================================================================
# Run context -- threaded through every stage
# =============================================================================


class RunContext(_Contract):
    """Immutable identity of a single pipeline execution."""

    run_id: str
    trial_id: str = Field(..., description="Research-ledger trial id, e.g. 'T-014'")
    as_of_date: dt.date = Field(..., description="Resolved decision date (a real session)")
    requested_date: Optional[dt.date] = Field(
        None, description="What the caller asked for, before resolving to a session"
    )
    started_at: dt.datetime
    engine_version: str
    schema_version: str
    config_version: str = Field(..., description="'<label>@<hash>' of parameters.yaml")
    mode: str = Field("live", description="live | backtest | dry_run")


# =============================================================================
# Stage 0 -- RAW DATA
# =============================================================================


class FeedRecord(_Contract):
    """One row of the Stage 0 manifest: what we pulled, from where, how fresh."""

    feed: str
    status: FeedStatus
    source: Optional[SourceName] = None
    fallback_used: bool = False
    primary_source_error: Optional[str] = None
    last_timestamp: Optional[dt.date] = None
    age_sessions: Optional[int] = None
    max_age_sessions: Optional[int] = None
    required: bool = False
    row_count: int = 0
    symbols_covered: int = 0
    notes: List[str] = Field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        if self.age_sessions is None or self.max_age_sessions is None:
            return False
        return self.age_sessions > self.max_age_sessions


class RawDataManifest(_Contract):
    """Versioned, timestamped snapshot description for one run."""

    run_id: str
    as_of_date: dt.date
    generated_at: dt.datetime
    snapshot_id: str
    feeds: Dict[str, FeedRecord] = Field(default_factory=dict)
    universe_size_raw: int = 0
    calendar_sessions_available: int = 0
    calendar_last_session: Optional[dt.date] = None
    calendar_is_approximate: bool = False
    survivorship_risk: bool = Field(
        False,
        description=(
            "True when index membership for as_of_date had to be inferred from a "
            "snapshot taken later than as_of_date."
        ),
    )
    survivorship_note: Optional[str] = None

    def missing_required(self) -> List[str]:
        return [
            name
            for name, rec in self.feeds.items()
            if rec.required and rec.status in (FeedStatus.MISSING,)
        ]

    def stale_required(self) -> List[str]:
        return [
            name
            for name, rec in self.feeds.items()
            if rec.required and (rec.status is FeedStatus.STALE or rec.is_stale)
        ]


# =============================================================================
# Stage 1 -- DATA QUALITY / LEAKAGE GATE
# =============================================================================


class StockDataFlags(_Contract):
    status: GateResult = GateResult.PASS
    failed_checks: List[str] = Field(default_factory=list)
    soft_flags: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(_Contract):
    run_status: GateResult
    market_wide_failures: List[str] = Field(default_factory=list)
    market_wide_soft_flags: List[str] = Field(default_factory=list)
    per_stock_flags: Dict[str, StockDataFlags] = Field(default_factory=dict)
    pit_audit: Dict[str, bool] = Field(default_factory=dict)
    pit_audit_failures: List[str] = Field(default_factory=list)
    checked_symbols: int = 0
    failed_symbols: int = 0

    def failed_tickers(self) -> List[str]:
        return [
            t for t, f in self.per_stock_flags.items() if f.status is GateResult.FAIL
        ]

    def is_clean(self, ticker: str) -> bool:
        flags = self.per_stock_flags.get(ticker)
        return flags is None or flags.status is GateResult.PASS


# =============================================================================
# Stage 2 -- MARKET REGIME
# =============================================================================


class RegimeState(_Contract):
    as_of_date: dt.date

    trend_regime: TrendRegime
    trend_slope_annualised: Optional[float] = None
    index_vs_fast_ma_pct: Optional[float] = None
    index_vs_slow_ma_pct: Optional[float] = None

    vol_tercile: VolTercile
    vol_context: VolContext
    vix_level: Optional[float] = None
    vix_percentile: Optional[float] = None
    vix_change_pct: Optional[float] = None
    #: G.C. & Kothari (2016): a rising-VIX read is more reliable than a
    #: falling-VIX all-clear. This weight makes that asymmetry explicit.
    vol_signal_confidence: float = 1.0

    breadth_pct_above_ma: Optional[float] = None
    breadth_state: BreadthState = BreadthState.NEUTRAL
    breadth_divergence_flag: bool = False
    breadth_sample_size: int = 0

    regime_bucket: str
    transition_flag: bool = False
    transition_components: List[str] = Field(default_factory=list)

    momentum_multiplier: float
    quality_multiplier: float
    sector_rs_multiplier: float
    dampener_applied: float = 1.0

    allow_new_entries: bool = True
    block_reason: Optional[str] = None

    notes: List[str] = Field(default_factory=list)

    def compatibility(self) -> RegimeCompatibility:
        """Human-facing 'Regime Compatibility' line on the recommendation card."""
        if not self.allow_new_entries:
            return RegimeCompatibility.UNFAVORABLE
        if self.momentum_multiplier >= 0.9 and not self.transition_flag:
            return RegimeCompatibility.FAVORABLE
        if self.momentum_multiplier <= 0.5 or self.transition_flag:
            return RegimeCompatibility.UNFAVORABLE
        return RegimeCompatibility.NEUTRAL


# =============================================================================
# Stage 3 -- ELIGIBILITY
# =============================================================================


class EligibilityReport(_Contract):
    as_of_date: dt.date
    universe_considered: int
    eligible_universe: List[str] = Field(default_factory=list)
    rejected: Dict[str, RejectionReason] = Field(default_factory=dict)
    rejection_details: Dict[str, str] = Field(default_factory=dict)
    #: Checks that could not run at all because the data does not exist.
    #: Reported verbatim on every card -- never silently treated as PASS.
    not_testable: Dict[str, List[str]] = Field(default_factory=dict)
    sector_map: Dict[str, str] = Field(default_factory=dict)
    adtv_inr: Dict[str, float] = Field(default_factory=dict)
    position_value_inr: float = 0.0

    def rejection_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for reason in self.rejected.values():
            key = reason.value
            out[key] = out.get(key, 0) + 1
        return out


# =============================================================================
# Stage 4 -- CORE SCORE
# =============================================================================


class FactorScore(_Contract):
    name: str
    raw_value: Optional[float] = None
    standardised: Optional[float] = None
    weight: float = 0.0
    available: bool = True
    horizon_note: Optional[str] = None
    evidence_tier: Optional[str] = None
    citation: Optional[str] = None


class StockScore(_Contract):
    ticker: str
    sector: Optional[str] = None
    factors: Dict[str, FactorScore] = Field(default_factory=dict)
    composite_raw: float = 0.0
    #: Composite mapped onto 0..1 across the eligible universe. Every threshold
    #: in Stage 5/8 operates on this scale.
    composite_score: float = 0.0
    percentile: float = 0.0
    rank: int = 0

    def factor_value(self, name: str) -> Optional[float]:
        f = self.factors.get(name)
        return f.standardised if f else None


class RedundancyReport(_Contract):
    """Measured, not assumed (master prompt Stage 4).

    Also carries the technical-collapse diagnostic: RSI / MACD / MA-crossover
    are expected to collapse into momentum. Expecting it is not the same as
    verifying it, so the engine verifies it and logs the number.
    """

    pairwise_spearman: Dict[str, float] = Field(default_factory=dict)
    breaches: List[Tuple[str, str, float]] = Field(default_factory=list)
    cutoff: float = 0.6
    action_taken: str = "log"
    technical_collapse: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class CoreScoreReport(_Contract):
    as_of_date: dt.date
    weighting_mode: str
    standardisation: str
    effective_weights: Dict[str, float] = Field(default_factory=dict)
    dropped_factors: Dict[str, str] = Field(default_factory=dict)
    ranked_scores: List[StockScore] = Field(default_factory=list)
    redundancy: RedundancyReport = Field(default_factory=RedundancyReport)
    universe_size: int = 0
    notes: List[str] = Field(default_factory=list)

    def top(self, n: int) -> List[StockScore]:
        return self.ranked_scores[:n]

    def by_ticker(self, ticker: str) -> Optional[StockScore]:
        for s in self.ranked_scores:
            if s.ticker == ticker:
                return s
        return None


# =============================================================================
# Stage 5 -- FALSE-SIGNAL DEFENSE
# =============================================================================


class CheckResult(_Contract):
    """One row of the false-signal defense matrix.

    ``NOT_TESTABLE`` is a real outcome. A check whose input data is absent is
    reported as untestable and printed under "Not testable with current data";
    it is never quietly upgraded to PASS.
    """

    check: str
    outcome: CheckOutcome
    penalty: float = 0.0
    reason: Optional[str] = None
    observed: Dict[str, Any] = Field(default_factory=dict)
    threshold: Optional[Any] = None
    evidence_tier: Optional[str] = None
    citation: Optional[str] = None


class StockDefenseResult(_Contract):
    ticker: str
    checks: List[CheckResult] = Field(default_factory=list)
    total_penalty: float = 0.0
    score_before: float = 0.0
    score_after: float = 0.0
    final_status: str = "CLEARED"  # CLEARED | PENALIZED | REJECTED

    def passed(self) -> List[CheckResult]:
        return [c for c in self.checks if c.outcome is CheckOutcome.PASS]

    def penalised(self) -> List[CheckResult]:
        return [c for c in self.checks if c.outcome is CheckOutcome.SCORE_PENALTY]

    def rejected(self) -> List[CheckResult]:
        return [c for c in self.checks if c.outcome is CheckOutcome.HARD_REJECT]

    def not_testable(self) -> List[CheckResult]:
        return [c for c in self.checks if c.outcome is CheckOutcome.NOT_TESTABLE]


class FalseSignalReport(_Contract):
    as_of_date: dt.date
    market_wide_checks: List[CheckResult] = Field(default_factory=list)
    market_wide_penalty: float = 0.0
    market_halt: bool = False
    market_halt_reason: Optional[str] = None
    per_stock: Dict[str, StockDefenseResult] = Field(default_factory=dict)

    def survivors(self) -> List[str]:
        return [t for t, r in self.per_stock.items() if r.final_status != "REJECTED"]


# =============================================================================
# Stage 6 -- ENTRY CONFIRMATION
# =============================================================================


class EntryDecision(_Contract):
    ticker: str
    status: EntryStatus
    trigger_type: TriggerType = TriggerType.NONE
    entry_zone: Optional[Tuple[float, float]] = None
    reference_price: Optional[float] = None
    confirmations_passed: List[str] = Field(default_factory=list)
    confirmations_failed: List[str] = Field(default_factory=list)
    confirmations_not_testable: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class EntryReport(_Contract):
    as_of_date: dt.date
    decisions: Dict[str, EntryDecision] = Field(default_factory=dict)

    def triggered(self) -> List[str]:
        return [t for t, d in self.decisions.items() if d.status is EntryStatus.TRIGGERED]

    def watchlist(self) -> List[str]:
        return [t for t, d in self.decisions.items() if d.status is EntryStatus.WATCHLIST]


# =============================================================================
# Stage 7 -- RISK / POSITION
# =============================================================================


class ExitCondition(_Contract):
    reason: ExitReason
    priority: int
    description: str
    level: Optional[float] = None
    active: bool = True


class RiskPlan(_Contract):
    ticker: str
    reference_price: Optional[float] = None
    atr: Optional[float] = None
    atr_pct_of_price: Optional[float] = None

    stop_price: Optional[float] = None
    stop_distance_pct: Optional[float] = None
    stop_basis: Optional[str] = None

    #: Distinct from the stop: the level at which the ORIGINAL THESIS is dead.
    invalidation_level: Optional[float] = None
    invalidation_basis: Optional[str] = None

    trailing_stop_rule: Optional[str] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_basis: Optional[str] = None
    reward_to_risk_t1: Optional[float] = None
    reward_to_risk_t2: Optional[float] = None

    risk_category: RiskCategory = RiskCategory.MINIMUM
    risk_category_inputs: Dict[str, float] = Field(default_factory=dict)

    expected_holding_sessions: Tuple[int, int] = (0, 0)
    expected_holding_weeks: Tuple[int, int] = (0, 0)

    exit_conditions: List[ExitCondition] = Field(default_factory=list)
    estimated_round_trip_cost_bps: Optional[float] = None
    estimated_impact_bps: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


# =============================================================================
# Stage 8 -- FINAL SIGNAL
# =============================================================================


class Recommendation(_Contract):
    """The full recommendation card. Mirrors the master prompt output schema.

    ``false_signal_*`` fields are mandatory content, not optional decoration --
    the webapp renders them expanded by default (FR-4).
    """

    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None

    decision: Decision
    signal_strength_band: StrengthBand
    regime_compatibility: RegimeCompatibility
    expected_holding_period: str

    entry_zone: Optional[Tuple[float, float]] = None
    invalidation_level: Optional[float] = None
    initial_stop: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    position_risk_category: RiskCategory

    last_close: Optional[float] = None
    composite_score: float = 0.0
    universe_percentile: float = 0.0
    rank: int = 0

    why_this_signal_exists: List[str] = Field(default_factory=list)
    market_regime: List[str] = Field(default_factory=list)
    sector_state: List[str] = Field(default_factory=list)
    confirmation: List[str] = Field(default_factory=list)

    false_signal_cleared: List[str] = Field(default_factory=list)
    false_signal_flagged: List[str] = Field(default_factory=list)
    false_signal_not_testable: List[str] = Field(default_factory=list)

    sell_conditions: List[str] = Field(default_factory=list)
    research_basis: List[str] = Field(default_factory=list)
    data_quality_note: List[str] = Field(default_factory=list)

    cost_note: Optional[str] = None
    unvalidated_parameter_warning: str = (
        "Every threshold behind this output is an UNVALIDATED hypothesis until a "
        "CPCV run on point-in-time India data promotes it."
    )


class ClosestCandidate(_Contract):
    ticker: str
    composite_score: float
    rank: int
    gate_failed: str
    detail: Optional[str] = None


class NoTradeReport(_Contract):
    """A first-class output state, never an empty array or an error."""

    reason: str
    closest_candidates: List[ClosestCandidate] = Field(default_factory=list)
    eligible_universe_size: int = 0
    scored_count: int = 0
    survived_defense_count: int = 0
    triggered_count: int = 0
    gate_summary: Dict[str, int] = Field(default_factory=dict)


class FinalSignalOutput(_Contract):
    run_id: str
    trial_id: str
    as_of_date: dt.date
    generated_at: dt.datetime
    engine_version: str
    config_version: str

    regime_state: RegimeState
    recommendations: List[Recommendation] = Field(default_factory=list)
    watchlist: List[Recommendation] = Field(default_factory=list)
    no_trade: Optional[NoTradeReport] = None

    data_quality_flags: List[str] = Field(default_factory=list)
    manifest: Optional[RawDataManifest] = None

    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = (
        "Decision-support tool. Not financial advice. No trades are placed "
        "automatically."
    )

    @property
    def is_no_trade(self) -> bool:
        return not self.recommendations


# =============================================================================
# Research ledger
# =============================================================================


class LedgerRow(_Contract):
    """One append-only row per run (research program section 17).

    The honest trial count this produces is a direct input to the Deflated
    Sharpe Ratio. A run that does not appear here corrupts every subsequent
    statistical claim, which is why the ledger writer is fatal-on-failure.
    """

    trial_id: str
    run_id: str
    date: dt.date
    logged_at: dt.datetime
    engine_version: str
    schema_version: str
    config_version: str
    mode: str = "live"

    regime_state: Dict[str, Any] = Field(default_factory=dict)
    eligible_universe_size: int = 0
    universe_considered: int = 0
    stocks_scored: List[Dict[str, Any]] = Field(default_factory=list)
    signals_generated: List[str] = Field(default_factory=list)
    watchlist_generated: List[str] = Field(default_factory=list)
    no_trade: bool = False
    no_trade_reason: Optional[str] = None

    gate_counts: Dict[str, int] = Field(default_factory=dict)
    data_quality_flags: List[str] = Field(default_factory=list)
    survivorship_risk: bool = False
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
