"""Stage input/output contracts.

Each stage is a pure function with a declared schema so it can be tested and
re-validated independently. These pydantic models are those schemas, and also
the API wire format, so a stage that adds a field either renders on the card or
fails loudly.

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

Conventions enforced by the models rather than by discipline:

* prices are ``Optional[float]`` and may be ``None`` -- a missing level is
  reported as missing, never as ``0.0``;
* every band or category is an enum, never a free string;
* every check carries an ``evidence`` citation, so the research-basis section
  of the output is generated from data rather than template prose.
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
    "TradePlan",
    "Recommendation",
    "ClosestCandidate",
    "NoTradeReport",
    "SlateEntry",
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


class FactorMember(_Contract):
    """One measured factor inside a fitted theme.

    The model fits ONE coefficient per theme, over the average of its members'
    cross-sectional ranks -- seventeen coefficients over a set this collinear
    is not estimable. So the theme is what carries weight, and these are what
    the theme is made of. They are carried onto the card because "lottery
    -1.81 sd" says nothing about WHICH lottery moment moved, and the reader
    cannot check the theme against the measurements without them.

    `rank` is the cross-sectional rank in [-1, +1], taken within sector where
    the sector is large enough -- the same number the family averages.
    """

    name: str
    rank: Optional[float] = None
    available: bool = True
    description: Optional[str] = None


class FactorScore(_Contract):
    name: str
    raw_value: Optional[float] = None
    standardised: Optional[float] = None
    #: THE WEIGHT THIS NAME WAS BLENDED AT, not the frozen one from the fit.
    #: Under the v3 composite the two differ whenever a name is missing a
    #: theme, which is 91% of the live universe: the blend re-caps over the
    #: themes a name actually has, so momentum's fitted 0.40 is the weight for a
    #: five-theme name and the ceiling for a four-theme one. Serving the frozen
    #: number here is what made the card print a weight that did not multiply
    #: its own z into its own contribution -- out by exactly 1/den, 1.2344x on a
    #: four-theme name, under a heading reading "sums to the score".
    weight: float = 0.0
    #: standardised x weight -- the signed amount this factor moved THIS name's
    #: composite. With `weight` now per-name, the identity `standardised *
    #: weight == contribution` HOLDS, and `test_v3_blend_is_capped` pins it.
    #: It is still carried explicitly rather than recomputed downstream, so
    #: there is one arithmetic and not two.
    contribution: Optional[float] = None
    available: bool = True
    horizon_note: Optional[str] = None
    evidence_tier: Optional[str] = None
    citation: Optional[str] = None
    #: The measured factors this theme averages. Empty on the hand-weighted
    #: composite path, which has no members.
    members: List["FactorMember"] = Field(default_factory=list)


class StockScore(_Contract):
    ticker: str
    sector: Optional[str] = None
    factors: Dict[str, FactorScore] = Field(default_factory=dict)
    #: Whether the name clears the ABSOLUTE floor and may be BOUGHT. A name
    #: below it is still ranked and still shown -- it is a holdable position and
    #: a legitimate watchlist entry -- it just cannot be opened. When too few
    #: names clear the floor the book holds cash, which is how NO TRADE happens.
    entry_admissible: bool = True
    #: Why not, when not.
    entry_block_reason: Optional[str] = None
    #: MEASURED ALWAYS, ENFORCED NEVER. Whether the name closes above its long
    #: moving average -- the one ABSOLUTE test in the engine, in the sense that
    #: it can fail for every name at once. `entry_admissible` cannot carry this:
    #: it is the per-name entry floor, Stage 8 gates on it, and that floor is
    #: shipped DISABLED on a measured treatment effect of -2.2% (95% CI
    #: [-3.2%, -1.4%]) -- ejecting a name the day it slips below its average
    #: cost more in forced turnover than it saved.
    #:
    #: The book-LEVEL rule is a different question and was never measured
    #: against that: not "may this name be bought" but "can the market supply a
    #: book at all today". Recording the bar separately from the gate is what
    #: lets Stage 8 ask the second without re-enabling the first.
    absolute_bar_cleared: Optional[bool] = None
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
    #: Gap between the top decile's PREDICTED rank and the median's, before the
    #: rank transform flattens it. The score itself is uniform every day, so
    #: nothing else in the pipeline can tell a day the model had a view from a
    #: day it did not.
    prediction_dispersion: Optional[float] = None
    #: What this model's spread normally is, measured on its own training
    #: panel. The gate is a ratio to this, because the level is a function of
    #: the ridge penalty rather than of the market.
    typical_dispersion: Optional[float] = None
    #: P(target barrier before stop barrier) per ticker, from the meta-label
    #: veto. None when the veto is disabled or could not be fitted -- which is
    #: the shipped default. The meta-label veto was removed in the 2026-09-03
    #: cleanup on its own measurement -- per-date AUC 0.4996, a coin flip.
    win_probability: Optional[Dict[str, float]] = None
    #: Why the veto is inert, when it is. An absent probability with no reason
    #: is indistinguishable from a veto that ran and approved everything.
    win_probability_unavailable: Optional[str] = None
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
    """One name's Stage 5 verdict.

    THE PENALTY IS A RANK DEMOTION, NOT A SCORE ADJUSTMENT. `score_before` is
    `composite_unit`, a cross-sectional rank uniform on [0, 1] by construction,
    so `total_penalty` of 0.10 means "drop this name ten percentile points". It
    does not mean 0.10 of anything returnlike, and it carries no magnitude: the
    same 0.10 is applied on a day when the model's predictions are tightly
    bunched and on a day when they are widely dispersed.

    The field names are kept rather than renamed because they are serialised
    into the ledger and the API, and renaming them would break stored runs to
    fix a naming problem. The description is the fix.
    """

    ticker: str
    checks: List[CheckResult] = Field(default_factory=list)
    #: Percentile points to demote by, summed across triggered checks.
    total_penalty: float = Field(
        0.0, description="rank demotion in percentile points, not a return")
    #: The cross-sectional rank in [0, 1] before demotion.
    score_before: float = Field(0.0, description="cross-sectional rank in [0,1]")
    #: The same rank after demotion, floored at zero.
    score_after: float = Field(0.0, description="demoted rank in [0,1]")
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
    #: Was the book allowed to OPEN a position on this session? Carried on the
    #: report rather than re-derived downstream, because Stage 8 has to tell
    #: "nothing qualified" from "the book was not buying today" and the only
    #: other way to know is to parse a reason string.
    entries_open: bool = True
    entries_closed_reason: Optional[str] = None

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

    #: The size, structurally. It was computed on every plan and reached the
    #: card only inside a formatted note string -- so nothing downstream could
    #: read it: not the trade plan that has to state what is at risk, not the
    #: ledger, not a reconciliation against a broker statement. A number that
    #: exists only inside prose is a number the engine cannot check itself
    #: against.
    position_size_shares: Optional[int] = None
    position_value_inr: Optional[float] = None
    #: Entry price minus stop price, times the size. What the position loses if
    #: the disaster floor fills exactly at the stop -- and "exactly" is doing
    #: work, because a floor eight ATRs out is reached by a name in collapse and
    #: a name in collapse gaps through it. Measured on the nine study trades
    #: that hit the floor, the realised loss averaged -31.0% against a stop
    #: placed at most 35% away. This is the optimistic end of what a stop costs.
    risk_at_stop_inr: Optional[float] = None

    expected_holding_sessions: Tuple[int, int] = (0, 0)
    expected_holding_weeks: Tuple[int, int] = (0, 0)

    exit_conditions: List[ExitCondition] = Field(default_factory=list)
    estimated_round_trip_cost_bps: Optional[float] = None
    estimated_impact_bps: Optional[float] = None

    #: Recent turnover against the trailing average the position was sized on.
    #: Sizing uses 21 sessions, which is exactly the window that lags when
    #: liquidity dries up. None when either window is unavailable.
    liquidity_ratio_recent: Optional[float] = None
    liquidity_warning: Optional[str] = None
    #: KNOWN_VALID | KNOWN_STALE | MISSING | INVALID -- `liquidity.assess`.
    #:
    #: Carried on the plan because it decides whether there is a position at
    #: all. An unmeasurable ADTV used to fall through to the full capital slot,
    #: so the state was invisible AND consequential; a card that shows a size
    #: without showing what liquidity it was sized against cannot be checked.
    liquidity_state: Optional[str] = None

    notes: List[str] = Field(default_factory=list)


# =============================================================================
# Stage 8 -- FINAL SIGNAL
# =============================================================================


class TradePlan(_Contract):
    """What this trade IS, recorded at the moment it is issued.

    The engine could always say what it liked and, months later, what happened.
    It could not say what it EXPECTED, so nothing could ever be scored against
    its own forecast: a run that produced eight names and a run that produced
    eight names with a stated 58% win probability and a 42-session hold left
    identical records. Calibration needs the second one.

    Every field here is either a rule the engine is about to follow (cadence,
    planned hold, stop, target) or a measured frequency from the study named in
    `basis` -- never a forecast for THIS name. There is no per-name edge
    estimate in this engine and inventing one on the card would be the most
    dangerous number it prints. `expected_return_pct` is the mean outcome of the
    258 trades this configuration took in the study, and `probability_of_profit`
    is how many of them ended positive. They describe the population the trade
    is drawn from. That is the honest claim and it is the one worth recording,
    because it is falsifiable: after fifty live trades the realised rates can be
    compared against these and the difference is evidence.
    """

    #: Sessions between entry opportunities -- the cron's own clock.
    cadence_sessions: int
    #: Sessions this position is planned to be held at most. The time backstop.
    planned_hold_sessions: int
    #: MEDIAN sessions actually held in the study. Differs from the plan because
    #: most positions leave on the rank band well before the backstop.
    expected_hold_sessions: Optional[float] = None
    #: Mean net return per trade in the study, after `assumed_cost_bps`.
    expected_return_pct: Optional[float] = None
    #: Median, stated beside the mean because the distribution is right-skewed
    #: and quoting only the mean would describe a typical trade that does not
    #: exist. In the study the mean was +7.1% and the median +3.7%.
    median_return_pct: Optional[float] = None
    #: Mean return in EXCESS of the equal-weight eligible universe over the same
    #: holding window. The mean return above is not an edge; this is.
    expected_excess_pct: Optional[float] = None
    #: And its median, for the same reason the median return is carried: the
    #: mean excess is +3.74% and the median is +0.69%, so the typical trade
    #: barely beats the market and a minority of them carry the result. Quoting
    #: the mean alone would describe an edge that most trades do not have.
    median_excess_pct: Optional[float] = None
    #: Share of study trades that ended net positive.
    probability_of_profit: Optional[float] = None
    #: Share that beat the benchmark over their own holding window. Lower than
    #: `probability_of_profit`, and the gap is the point: most of these trades
    #: make money because the market does, and only about half add anything.
    probability_of_beating_benchmark: Optional[float] = None
    #: Round-trip cost the two figures above were computed net of.
    assumed_cost_bps: Optional[float] = None
    #: The same frequencies at every cost scenario, so the card can state what
    #: happens if execution is worse than assumed rather than leaving the reader
    #: to wonder. Cost is the one input certain to be worse live, and on this
    #: configuration the edge survives it: at 120 bps the win rate is still
    #: 55.0% and the mean +6.29%. Keyed by scenario name, each entry carrying
    #: cost_bps, p_win, p_beat, mean_net_pct and median_net_pct.
    cost_sensitivity: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    #: What the position risks, in rupees and as a share of the book, if the
    #: disaster floor fills exactly at the stop.
    risk_at_stop_inr: Optional[float] = None
    risk_at_stop_pct_of_book: Optional[float] = None
    #: The study these frequencies come from, so a reader can go and check it,
    #: and so a stale plan is visible as stale.
    basis: Optional[str] = None
    basis_trades: Optional[int] = None
    basis_period: Optional[str] = None
    #: Never absent. The frequencies above are a population, not a promise.
    caveat: str = (
        "These are frequencies from a historical study of this configuration, "
        "not a forecast for this name. The engine estimates no per-name edge."
    )


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
    #: What this trade is and what the configuration it belongs to has done
    #: historically. Optional because a watchlist name on a run with no risk
    #: plan has no trade to describe -- not because it is decoration.
    trade_plan: Optional[TradePlan] = None

    entry_zone: Optional[Tuple[float, float]] = None
    invalidation_level: Optional[float] = None
    initial_stop: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    #: Optional because a candidate can reach the card with no risk plan: Stage
    #: 7 builds one only when Stage 6 produced a reference price, and Stage 6
    #: declines to for a name with under 70 sessions of history or an ATR it
    #: cannot compute. Every other plan-derived field on this contract is
    #: already guarded that way; this one was declared required, so `_card`'s
    #: own `if plan else None` raised a ValidationError and took down the whole
    #: run over one name. `api._card` has always read it as optional.
    position_risk_category: Optional[RiskCategory] = None

    last_close: Optional[float] = None
    #: Score after Stage 5. This is what the gates compare against, so it is
    #: what the card prints and what the list is ordered by.
    composite_score: float = 0.0
    universe_percentile: float = 0.0
    #: Position among the defended candidates, ordered by composite_score.
    rank: int = 0
    #: Position the model put it in before Stage 5 argued against it. Kept so a
    #: penalty is visible as a demotion rather than silently rewriting history.
    model_rank: int = 0

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

    #: Structured factor values, mirroring `why_this_signal_exists` in numeric
    #: form so the UI can sort and align on them without parsing prose.
    factor_detail: Dict[str, FactorScore] = Field(default_factory=dict)

    #: Earnings proximity, when there is something to say. Carried as its own
    #: field rather than left inside `false_signal_flagged`, because the screen
    #: has to show it WITHOUT string-matching prose -- and a name reporting
    #: tomorrow is the one risk on this card a stop cannot cover, so it must
    #: not end up in a collapsed technical drawer.
    earnings_note: Optional[str] = None
    #: Whether this name clears the ABSOLUTE floor and may be OPENED. Carried
    #: onto the card from `StockScore` because the screen needs it: a name that
    #: may be held but not bought is a different instruction from one that may
    #: be bought, and a watchlist that does not distinguish them invites the
    #: reader to open a position the engine would refuse.
    entry_admissible: bool = True
    #: Why not, when not.
    entry_block_reason: Optional[str] = None

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


class SlateEntry(_Contract):
    """One row of the screen, as the run itself decided it.

    The slate used to be recomputed by the presentation layer on every request,
    from a payload that carried no memory of the previous screen. That made it
    impossible to hold a name across sessions, and it meant the live screen, the
    history page and the outcome record each reconstructed a different list from
    the same run. The run decides once, here, and every reader renders this.
    """

    ticker: str
    position: int
    status: str
    model_rank: Optional[int] = None
    #: True when this name kept its slot under the Stage 6 exit band rather
    #: than being chosen fresh today.
    carried: bool = False
    #: The session this name entered the screen and has held it since. With
    #: `as_of_date` this gives the dwell directly, which is the figure the
    #: turnover defect was invisible without.
    shown_since: Optional[dt.date] = None
    reason: str = ""


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
    #: What the screen leads with, decided by the run and not by the reader.
    slate: List[SlateEntry] = Field(default_factory=list)
    #: Names that were on the previous screen and are not on this one, each
    #: with the reason. A name vanishing silently is how an eviction hides.
    slate_departures: List[Dict[str, str]] = Field(default_factory=list)
    #: Set when the regime or a defence halt refused NEW entries. The book is
    #: unaffected -- what closes a position is the Stage 6 exit band.
    new_entries_blocked: Optional[str] = None
    #: THE ENTRY CLOCK, on every run rather than only on the runs it blocks.
    #:
    #: `new_entries_blocked` carries the cadence's refusal, which means the
    #: schedule is recorded only on the twenty sessions out of twenty-one when
    #: it says no. On the session it says yes it recorded nothing at all, so
    #: no reader of a payload could tell "today is a buying session" from "this
    #: engine has no schedule" -- and the screen, which has only the payload,
    #: could never show the clock. Keys: cadence_sessions, is_entry_date,
    #: sessions_since_anchor, next_entry_date, sessions_until_next.
    entry_clock: Dict[str, Any] = Field(default_factory=dict)
    #: What happens to held names the run produced no card for -- suspended,
    #: dropped from the universe, or delisted. Without this a position left the
    #: book by omission and no exit was ever recorded.
    position_directives: List[Dict[str, Any]] = Field(default_factory=list)

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
    #: What actually decided the ranking: a hash of the model's source next
    #: to the training depth it was fitted on. config_version covers only
    #: parameters.yaml, so a code change or a store that grew leaves it
    #: identical -- which is the blind spot this closes.
    model_fingerprint: str = "unknown/?"
    mode: str = "live"

    regime_state: Dict[str, Any] = Field(default_factory=dict)
    eligible_universe_size: int = 0
    universe_considered: int = 0
    stocks_scored: List[Dict[str, Any]] = Field(default_factory=list)
    signals_generated: List[str] = Field(default_factory=list)
    watchlist_generated: List[str] = Field(default_factory=list)
    #: What the screen actually showed, in order. Without this the record could
    #: not answer "what did the user see on that day?" -- the history page was
    #: re-deriving it from `signals_generated` in emission order while the live
    #: screen sorted by model rank, so the two disagreed about a past run.
    slate_shown: List[Dict[str, Any]] = Field(default_factory=list)
    #: Held names the run never evaluated, and what was decided about each.
    #: `Ledger.open_book` unions the ones still held into the next run's book,
    #: so a temporary eligibility gap no longer closes a position silently.
    position_directives: List[Dict[str, Any]] = Field(default_factory=list)
    no_trade: bool = False
    #: Set when new entries were refused while the book stood. Distinct from
    #: `no_trade`, which used to absorb this case and report an empty book.
    new_entries_blocked: Optional[str] = None
    no_trade_reason: Optional[str] = None

    gate_counts: Dict[str, int] = Field(default_factory=dict)
    data_quality_flags: List[str] = Field(default_factory=list)
    survivorship_risk: bool = False
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
