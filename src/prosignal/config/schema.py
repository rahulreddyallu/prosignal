"""Typed schema for config/parameters.yaml.

Two design rules drive this module:

1. **Every research parameter carries its own honesty metadata.** A bare number
   in a quant system is a lie by omission -- it hides whether it came from a
   journal, a blog, or a fat finger. `Tunable` forces each one to declare a
   `status` (UNVALIDATED / VALIDATED / STATUTORY / STRUCTURAL / OPERATIONAL),
   an optional `search_range`, and a free-text `note`. The `/config`
   transparency endpoint (webapp FR-8) renders exactly this metadata.

2. **`extra="forbid"` everywhere.** A typo in parameters.yaml must crash on
   load, not silently fall back to a default the user never saw. This is the
   single most valuable property of the whole config layer: the user edits one
   file, and that file cannot fail quietly.

`Tunable` accepts either form in YAML::

    atr_multiple: 2.5                      # short form -> status UNVALIDATED
    atr_multiple:                          # long form
      value: 2.5
      status: UNVALIDATED
      search_range: [1.5, 3.5]
      note: "..."

Access in engine code is always explicit: ``cfg.stage7_risk.stop_loss.atr_multiple.value``
(or the shorthand ``.v``). There is deliberately no magic unwrapping -- reading
``.value`` in the call site is a constant reminder that the number is a
hypothesis, not a fact.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ParamStatus",
    "Tunable",
    "RootConfig",
    "KNOWN_FEEDS",
]

T = TypeVar("T")

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)


# =============================================================================
# Tunable
# =============================================================================


class ParamStatus(str, Enum):
    """Provenance of a parameter value. See parameters.yaml header."""

    UNVALIDATED = "UNVALIDATED"
    VALIDATED = "VALIDATED"
    STATUTORY = "STATUTORY"
    STRUCTURAL = "STRUCTURAL"
    OPERATIONAL = "OPERATIONAL"

    @property
    def is_research_hypothesis(self) -> bool:
        """True when this value still needs CPCV before it can be trusted."""
        return self is ParamStatus.UNVALIDATED


class OptimizationTier(str, Enum):
    """How a parameter may be treated during validation.

    This is the engine's structural defence against parameter overfitting.
    "UNVALIDATED" says a value has not been tested; it does NOT say the value
    should be searched. Those are different questions, and conflating them is
    how a research programme ends up with a 132-dimensional search space and a
    Probability of Backtest Overfitting near 1.

    Harvey, Liu & Zhu (2016) is the relevant discipline: the significance bar
    has to rise with the number of things you tried. The cheapest way to keep
    the bar clearable is to try fewer things on purpose.

    A_SEARCH
        Genuinely changes the edge. Gets a real grid in CPCV. Every value tried
        counts toward the Deflated Sharpe trial budget. Must be opted in
        explicitly, and the loader caps how many may exist.
    B_SENSITIVITY
        Perturbed to confirm the result is not knife-edge, but the winning
        configuration is NEVER selected on it. Robustness evidence, not a
        degree of freedom. This is the safe default for anything UNVALIDATED.
    C_FIXED
        Set once from evidence or convention and never searched. Academic
        constructions (12-1 momentum), statutory rates, definitional constants.
    D_OPERATIONAL
        Your business constraint -- capital, broker fees, appetite. Not a
        research parameter at all; changing it changes the problem, not the
        answer.
    """

    A_SEARCH = "A_SEARCH"
    B_SENSITIVITY = "B_SENSITIVITY"
    C_FIXED = "C_FIXED"
    D_OPERATIONAL = "D_OPERATIONAL"


#: Conservative default tier implied by a parameter's provenance. Note that
#: UNVALIDATED maps to B_SENSITIVITY, not A_SEARCH: a parameter has to be
#: deliberately promoted into the search, never swept in by default.
_DEFAULT_TIER_BY_STATUS = {
    ParamStatus.UNVALIDATED: OptimizationTier.B_SENSITIVITY,
    ParamStatus.VALIDATED: OptimizationTier.B_SENSITIVITY,
    ParamStatus.STATUTORY: OptimizationTier.C_FIXED,
    ParamStatus.STRUCTURAL: OptimizationTier.C_FIXED,
    ParamStatus.OPERATIONAL: OptimizationTier.D_OPERATIONAL,
}


class Tunable(BaseModel, Generic[T]):
    """A single parameter plus the evidence metadata that keeps it honest."""

    model_config = ConfigDict(extra="forbid")

    value: T
    status: ParamStatus = ParamStatus.UNVALIDATED
    search_range: Optional[List[Any]] = None
    note: Optional[str] = None
    validated_by: Optional[str] = None  # research-ledger trial id, e.g. "T-014"
    validated_on: Optional[dt.date] = None

    #: Explicit optimisation tier. When omitted, :attr:`tier` derives a
    #: conservative one from ``status``.
    optimization_tier: Optional[OptimizationTier] = None
    #: Number of grid points this parameter contributes when tier is A_SEARCH.
    #: Feeds the search-space and effective-trial-count accounting.
    search_points: Optional[int] = None

    # -- ergonomics ---------------------------------------------------------
    @property
    def v(self) -> T:
        """Shorthand for ``.value``."""
        return self.value

    @property
    def tier(self) -> OptimizationTier:
        if self.optimization_tier is not None:
            return self.optimization_tier
        return _DEFAULT_TIER_BY_STATUS[self.status]

    @property
    def grid_points(self) -> int:
        """How many values CPCV will try for this parameter."""
        if self.tier is not OptimizationTier.A_SEARCH:
            return 1
        return int(self.search_points or 3)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Tunable({self.value!r}, {self.status.value}, {self.tier.value})"

    # -- coercion -----------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _accept_bare_scalar(cls, data: Any) -> Any:
        """Allow ``key: 2.5`` as shorthand for ``key: {value: 2.5}``."""
        if isinstance(data, dict):
            # A dict that looks like the long form passes through untouched.
            if "value" in data or "status" in data:
                return data
            # A dict that does NOT look like the long form is a real dict value
            # (we do not currently use Tunable[Dict], so this is defensive).
            return {"value": data}
        return {"value": data}

    # -- invariants ---------------------------------------------------------
    @model_validator(mode="after")
    def _check_invariants(self) -> "Tunable[T]":
        # A VALIDATED parameter must name the trial that promoted it. Without
        # this the Deflated Sharpe trial count (research program section 8) is
        # unauditable, which defeats the purpose of claiming validation.
        if self.status is ParamStatus.VALIDATED:
            if not self.validated_by or not self.validated_on:
                raise ValueError(
                    "status=VALIDATED requires both `validated_by` (ledger trial "
                    "id, e.g. 'T-014') and `validated_on` (YYYY-MM-DD). A value "
                    "cannot claim validation without naming the run that "
                    "validated it."
                )
        # You cannot search a parameter you have not bounded. Requiring the
        # range at the point of opting in stops an unbounded sweep from being
        # one keystroke away.
        if self.tier is OptimizationTier.A_SEARCH and self.search_range is None:
            raise ValueError(
                "optimization_tier=A_SEARCH requires a `search_range`: a "
                "parameter cannot enter the CPCV grid without declared bounds."
            )
        if self.tier is not OptimizationTier.A_SEARCH and self.search_points:
            raise ValueError(
                f"`search_points` is only meaningful for A_SEARCH parameters; "
                f"this one is {self.tier.value}. Remove it, or promote the "
                f"parameter deliberately."
            )
        if self.search_points is not None and not (2 <= self.search_points <= 9):
            raise ValueError(
                "`search_points` must be between 2 and 9. A finer grid buys "
                "precision the data cannot support and inflates the multiple-"
                "testing penalty for nothing."
            )

        if self.search_range is not None:
            if len(self.search_range) != 2:
                raise ValueError("`search_range` must be exactly [low, high]")
            lo, hi = self.search_range
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                if lo > hi:
                    raise ValueError(f"`search_range` low ({lo}) exceeds high ({hi})")
                val = self.value
                if isinstance(val, bool):
                    pass
                elif isinstance(val, (int, float)):
                    if not (lo <= val <= hi):
                        raise ValueError(
                            f"value {val} lies outside its declared search_range "
                            f"[{lo}, {hi}]. Either the value is a typo or the "
                            f"range needs widening -- the engine will not guess."
                        )
        return self


# Convenient aliases used throughout the schema below.
TF = Tunable[float]
TI = Tunable[int]
TS = Tunable[str]
TB = Tunable[bool]
TLS = Tunable[List[str]]
TLF = Tunable[List[float]]
TOF = Tunable[Optional[float]]
TOI = Tunable[Optional[int]]


class _Base(BaseModel):
    """All config sections are strict: unknown keys are an error, not a shrug."""

    model_config = _STRICT


# =============================================================================
# 0-1. meta / runtime
# =============================================================================


class MetaConfig(_Base):
    config_label: str = "unlabelled"
    owner: str = "unknown"
    description: str = ""


class PathsConfig(_Base):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    curated_dir: str = "data/curated"
    snapshot_dir: str = "data/snapshots"
    cache_dir: str = "data/cache"
    ledger_dir: str = "data/ledger"
    reference_dir: str = "config/reference"
    log_dir: str = "logs"


class LoggingConfig(_Base):
    level: str = "INFO"
    to_file: bool = True
    to_console: bool = True
    backup_count: int = Field(14, ge=0)

    @field_validator("level")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unknown log level {v!r}")
        return v


class DateResolutionConfig(_Base):
    max_lookback_calendar_days: int = Field(10, ge=1, le=90)
    allow_future_dates: bool = False


class RuntimeConfig(_Base):
    timezone: str = "Asia/Kolkata"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    date_resolution: DateResolutionConfig = Field(default_factory=DateResolutionConfig)


# =============================================================================
# 1b. storage budget
# =============================================================================


class RawCacheConfig(_Base):
    enabled: bool = True
    max_mb: float = Field(384.0, ge=0)
    max_payload_mb_to_cache: float = Field(0.6, ge=0)
    never_cache_feeds: List[str] = Field(default_factory=list)


class AuditRawConfig(_Base):
    enabled: bool = False
    retain_sessions: int = Field(10, ge=0)


class StorageConfig(_Base):
    max_total_mb: float = Field(3072.0, gt=0)
    raw_cache: RawCacheConfig = Field(default_factory=RawCacheConfig)
    audit_raw: AuditRawConfig = Field(default_factory=AuditRawConfig)
    warn_free_disk_mb: float = Field(4096.0, ge=0)
    halt_free_disk_mb: float = Field(768.0, ge=0)
    write_batch_sessions: int = Field(25, ge=1, le=500)

    @model_validator(mode="after")
    def _check(self) -> "StorageConfig":
        if self.halt_free_disk_mb >= self.warn_free_disk_mb:
            raise ValueError("storage.halt_free_disk_mb must be < warn_free_disk_mb")
        if self.raw_cache.max_mb > self.max_total_mb:
            raise ValueError("storage.raw_cache.max_mb exceeds storage.max_total_mb")
        return self


# =============================================================================
# 2. capital
# =============================================================================


class CapitalConfig(_Base):
    total_capital_inr: TF
    max_open_positions: TI
    per_position_inr: TOF
    max_participation_of_adtv: TF

    def position_value_inr(self) -> float:
        """Rupee value of one new position -- explicit override or an even split."""
        explicit = self.per_position_inr.value
        if explicit is not None and explicit > 0:
            return float(explicit)
        n = max(int(self.max_open_positions.value), 1)
        return float(self.total_capital_inr.value) / n


# =============================================================================
# 3. universe
# =============================================================================


class UniverseConfig(_Base):
    index_name: TS
    allowed_series: TLS
    min_history_sessions: TI
    min_price_inr: TF
    manual_exclusions: TLS
    pre_snapshot_policy: TS

    @model_validator(mode="after")
    def _check_policy(self) -> "UniverseConfig":
        allowed = {"halt", "flag"}
        if self.pre_snapshot_policy.value not in allowed:
            raise ValueError(
                f"universe.pre_snapshot_policy must be one of {sorted(allowed)}"
            )
        return self


# =============================================================================
# 4. providers
# =============================================================================


class HttpConfig(_Base):
    timeout_seconds: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)
    backoff_base_seconds: float = Field(1.5, gt=1.0)
    min_interval_seconds: float = Field(0.35, ge=0)
    user_agent: str
    cache_enabled: bool = True
    cache_ttl_days_historical: int = Field(3650, ge=0)
    cache_ttl_hours_current: int = Field(6, ge=0)


class NseArchivesConfig(_Base):
    enabled: bool = True
    base_archives: str
    base_legacy: str
    bhavcopy_udiff_path: str
    bhavcopy_udiff_from: dt.date
    bhavcopy_legacy_path: str
    sec_bhavdata_full_path: str
    index_close_all_path: str
    fo_bhavcopy_path: str
    equity_master_path: str
    index_constituent_files: Dict[str, str]


class YFinanceConfig(_Base):
    enabled: bool = True
    equity_suffix: str = ".NS"
    symbol_overrides: Dict[str, str] = Field(default_factory=dict)
    batch_size: int = Field(40, ge=1, le=200)
    pause_between_batches_seconds: float = Field(1.0, ge=0)
    index_symbols: Dict[str, str] = Field(default_factory=dict)


class NseJsonApiConfig(_Base):
    enabled: bool = True
    treat_failure_as: str = "soft"
    base: str
    warmup_path: str = "/"
    corporate_actions_path: str
    event_calendar_path: str
    shareholding_path: str

    @field_validator("treat_failure_as")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in {"soft", "hard"}:
            raise ValueError("providers.nse_json_api.treat_failure_as must be soft|hard")
        return v


class CsvImportConfig(_Base):
    enabled: bool = True
    pledging_file: str
    fundamentals_file: str
    earnings_calendar_file: str
    corporate_actions_file: str
    regulatory_events_file: str
    index_membership_file: str


class ProvidersConfig(_Base):
    http: HttpConfig
    nse_archives: NseArchivesConfig
    yfinance: YFinanceConfig
    nse_json_api: NseJsonApiConfig
    csv_import: CsvImportConfig


# =============================================================================
# 5. feeds
# =============================================================================

#: Feed names the engine actually knows how to reason about. Anything else in
#: the `feeds:` block is a typo and will be rejected on load.
KNOWN_FEEDS = (
    "equity_ohlcv",
    "index_ohlcv",
    "india_vix",
    "index_membership",
    "equity_master",
    "delivery_data",
    "corporate_actions",
    "earnings_calendar",
    "fundamentals",
    "pledging",
    "fo_open_interest",
    "regulatory_events",
)


class FeedPolicy(_Base):
    max_age_sessions: int = Field(..., ge=0)
    required: bool = False


# =============================================================================
# 6. stage 1 -- data quality
# =============================================================================


class PitAuditConfig(_Base):
    enforce_historical_membership: bool = True
    enforce_delisted_inclusion: bool = True
    enforce_fundamentals_filing_date: bool = True
    enforce_pledging_disclosure_date: bool = True
    enforce_historical_sector: bool = True
    forbid_forward_fill_across_sessions: bool = True
    max_forward_fill_sessions: int = Field(0, ge=0)


class Stage1Config(_Base):
    require_two_price_sources: TB
    source_agreement_tolerance_bps: TF
    source_disagreement_action: TS
    outlier_return_sigma: TF
    outlier_sigma_lookback_sessions: TI
    outlier_absolute_return_pct: TF
    outlier_corroborating_volume_multiple: TF
    unexplained_split_ratio_tolerance: TF
    unexplained_split_min_ratio_gap: TF
    max_consecutive_missing_sessions: TI
    continuity_window_sessions: TI
    max_universe_failure_fraction: TF
    pit_audit: PitAuditConfig = Field(default_factory=PitAuditConfig)

    @model_validator(mode="after")
    def _check(self) -> "Stage1Config":
        if self.source_disagreement_action.value not in {"flag", "reject"}:
            raise ValueError("stage1.source_disagreement_action must be flag|reject")
        return self


# =============================================================================
# 7. stage 2 -- regime
# =============================================================================


class TrendConfig(_Base):
    fast_ma_sessions: TI
    slow_ma_sessions: TI
    slope_lookback_sessions: TI
    slope_flat_band_annualised: TF

    @model_validator(mode="after")
    def _check(self) -> "TrendConfig":
        if self.fast_ma_sessions.value >= self.slow_ma_sessions.value:
            raise ValueError(
                "stage2_regime.trend.fast_ma_sessions must be < slow_ma_sessions"
            )
        return self


class AsymmetricConfidence(_Base):
    rising_vix_weight: float = Field(1.0, ge=0, le=1)
    falling_vix_weight: float = Field(0.5, ge=0, le=1)


class VolatilityRegimeConfig(_Base):
    vix_percentile_lookback_sessions: TI
    low_tercile_pct: TF
    high_tercile_pct: TF
    vix_roc_lookback_sessions: TI
    vix_rising_threshold_pct: TF
    market_move_threshold_pct: TF
    asymmetric_confidence: AsymmetricConfidence = Field(
        default_factory=AsymmetricConfidence
    )

    @model_validator(mode="after")
    def _check(self) -> "VolatilityRegimeConfig":
        if self.low_tercile_pct.value >= self.high_tercile_pct.value:
            raise ValueError("low_tercile_pct must be < high_tercile_pct")
        return self


class BreadthConfig(_Base):
    ma_sessions: TI
    weak_threshold_pct: TF
    strong_threshold_pct: TF
    divergence_lookback_sessions: TI
    divergence_min_breadth_drop_pct: TF

    @model_validator(mode="after")
    def _check(self) -> "BreadthConfig":
        if self.weak_threshold_pct.value >= self.strong_threshold_pct.value:
            raise ValueError("breadth weak_threshold_pct must be < strong_threshold_pct")
        return self


class TransitionConfig(_Base):
    lookback_sessions: TI
    min_components_disagreeing: TI
    dampener: TF


class RegimeMultipliersConfig(_Base):
    status: ParamStatus = ParamStatus.UNVALIDATED
    note: Optional[str] = None
    table: Dict[str, List[float]]
    weak_breadth_momentum_penalty: TF

    @field_validator("table")
    @classmethod
    def _triples(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
        for bucket, row in v.items():
            if len(row) != 3:
                raise ValueError(
                    f"regime multiplier row {bucket!r} must be exactly "
                    f"[momentum, quality, sector_rs]; got {row!r}"
                )
            for x in row:
                if x < 0:
                    raise ValueError(f"negative multiplier in bucket {bucket!r}")
        return v


class Stage2Config(_Base):
    benchmark_index: TS
    secondary_index: TS
    trend: TrendConfig
    volatility: VolatilityRegimeConfig
    breadth: BreadthConfig
    transition: TransitionConfig
    multipliers: RegimeMultipliersConfig
    no_new_entry_buckets: TLS

    @model_validator(mode="after")
    def _buckets_known(self) -> "Stage2Config":
        unknown = [
            b for b in self.no_new_entry_buckets.value if b not in self.multipliers.table
        ]
        if unknown:
            raise ValueError(
                f"stage2_regime.no_new_entry_buckets references buckets absent "
                f"from multipliers.table: {unknown}"
            )
        return self


# =============================================================================
# 8. stage 3 -- eligibility
# =============================================================================


class LiquidityConfig(_Base):
    adtv_lookback_sessions: TI
    min_adtv_inr: TF
    use_participation_gate: bool = True
    reject_on_zero_volume_sessions: TI


class PledgingConfig(_Base):
    max_pledged_pct_of_promoter_holding: TF
    on_missing_data: TS

    @model_validator(mode="after")
    def _check(self) -> "PledgingConfig":
        if self.on_missing_data.value not in {"not_testable", "reject"}:
            raise ValueError("pledging.on_missing_data must be not_testable|reject")
        return self


class EarningsProximityConfig(_Base):
    holding_window_sessions: TI
    pead_conditional_signal_enabled: TB
    on_missing_data: TS

    @model_validator(mode="after")
    def _check(self) -> "EarningsProximityConfig":
        if self.on_missing_data.value not in {"not_testable", "reject"}:
            raise ValueError(
                "earnings_proximity.on_missing_data must be not_testable|reject"
            )
        return self


class RegulatoryCooldownConfig(_Base):
    default_cooldown_sessions: TI


class SectorConcentrationConfig(_Base):
    max_candidates_per_sector_soft: TI


class Stage3Config(_Base):
    liquidity: LiquidityConfig
    pledging: PledgingConfig
    earnings_proximity: EarningsProximityConfig
    regulatory_cooldown: RegulatoryCooldownConfig
    sector_concentration: SectorConcentrationConfig


# =============================================================================
# 9. stage 4 -- core score
# =============================================================================


class MomentumFactorConfig(_Base):
    enabled: bool = True
    lookback_sessions: TI
    skip_sessions: TI
    weight_band: TLF
    explicit_weight: TOF

    @model_validator(mode="after")
    def _check(self) -> "MomentumFactorConfig":
        if self.skip_sessions.value >= self.lookback_sessions.value:
            raise ValueError("momentum skip_sessions must be < lookback_sessions")
        return self


class QualityComponentConfig(_Base):
    enabled: bool = True
    weight: float = Field(..., ge=0)
    higher_is_better: bool


class QualityFactorConfig(_Base):
    enabled: bool = True
    weight_band: TLF
    explicit_weight: TOF
    components: Dict[str, QualityComponentConfig]
    min_components_required: TI
    on_universe_missing: TS

    @model_validator(mode="after")
    def _check(self) -> "QualityFactorConfig":
        if self.on_universe_missing.value not in {"drop_factor", "halt"}:
            raise ValueError("quality.on_universe_missing must be drop_factor|halt")
        live = [c for c in self.components.values() if c.enabled]
        if self.enabled and not live:
            raise ValueError("quality factor enabled but every component is disabled")
        if self.enabled and sum(c.weight for c in live) <= 0:
            raise ValueError("quality component weights sum to zero")
        return self


class SectorRsFactorConfig(_Base):
    enabled: bool = True
    horizons_sessions: Tunable[List[int]]
    market_relative_weight: TF
    weight_band: TLF
    explicit_weight: TOF

    @model_validator(mode="after")
    def _check(self) -> "SectorRsFactorConfig":
        if not self.horizons_sessions.value:
            raise ValueError("sector_relative_strength.horizons_sessions is empty")
        if any(h <= 0 for h in self.horizons_sessions.value):
            raise ValueError("sector_relative_strength horizons must be positive")
        if not 0.0 <= self.market_relative_weight.value <= 1.0:
            raise ValueError("market_relative_weight must be within [0, 1]")
        return self


class EstimateRevisionFactorConfig(_Base):
    enabled: bool = False
    weight_band: TLF

    @model_validator(mode="after")
    def _locked(self) -> "EstimateRevisionFactorConfig":
        if self.enabled and max(self.weight_band.value or [0.0]) > 0:
            raise ValueError(
                "estimate_revision_momentum cannot carry weight: it requires "
                "timestamped point-in-time India analyst data. Approximating it "
                "with an untimestamped source is exactly the leakage the "
                "research program forbids."
            )
        return self


class FactorsConfig(_Base):
    momentum_12_1: MomentumFactorConfig
    quality: QualityFactorConfig
    sector_relative_strength: SectorRsFactorConfig
    estimate_revision_momentum: EstimateRevisionFactorConfig


class RedundancyConfig(_Base):
    enabled: bool = True
    max_abs_spearman: TF
    on_breach: TS
    run_technical_collapse_diagnostic: bool = True

    @model_validator(mode="after")
    def _check(self) -> "RedundancyConfig":
        if self.on_breach.value not in {"log", "shrink"}:
            raise ValueError("redundancy.on_breach must be log|shrink")
        return self


class Stage4Config(_Base):
    weighting_mode: TS
    standardisation: TS
    winsorize_pct: TF
    sector_neutral: TB
    factors: FactorsConfig
    redundancy: RedundancyConfig
    data_quality_gate_penalty: TF

    @model_validator(mode="after")
    def _check(self) -> "Stage4Config":
        modes = {"equal_weight", "band_midpoint", "explicit", "rank_ic"}
        if self.weighting_mode.value not in modes:
            raise ValueError(f"stage4.weighting_mode must be one of {sorted(modes)}")
        if self.standardisation.value not in {"rank", "zscore"}:
            raise ValueError("stage4.standardisation must be rank|zscore")
        if self.weighting_mode.value == "explicit":
            missing = [
                name
                for name, f in (
                    ("momentum_12_1", self.factors.momentum_12_1),
                    ("quality", self.factors.quality),
                    ("sector_relative_strength", self.factors.sector_relative_strength),
                )
                if f.enabled and f.explicit_weight.value is None
            ]
            if missing:
                raise ValueError(
                    "weighting_mode=explicit but these enabled factors have no "
                    f"explicit_weight: {missing}"
                )
        return self


# =============================================================================
# 10. stage 5 -- false-signal defense
# =============================================================================


class LowVolumeBreakoutConfig(_Base):
    enabled: bool = True
    lookback_sessions: TI
    min_volume_multiple: TF
    penalty: TF


class LiquidityDistortionConfig(_Base):
    enabled: bool = True
    min_session_turnover_vs_adtv: TF
    action: str = "hard_reject"


class GapSignalConfig(_Base):
    enabled: bool = True
    max_gap_atr_multiple: TF
    require_next_session_confirmation: bool = True
    penalty: TF


class NewsSpikeConfig(_Base):
    enabled: bool = True
    move_sigma: TF
    volume_multiple: TF
    persistence_sessions: TI
    penalty: TF


class ShortCoveringConfig(_Base):
    enabled: bool = True
    lookback_sessions: TI
    min_price_change_pct: TF
    min_oi_decline_pct: TF
    penalty: TF


class BetaExplainedConfig(_Base):
    enabled: bool = True
    lookback_sessions: TI
    beta_estimation_sessions: TI
    explained_fraction_threshold: TF
    max_penalty: TF


class OverextensionConfig(_Base):
    enabled: bool = True
    short_horizon_sessions: TI
    extended_atr_multiple: TF
    action: TS
    penalty: TF

    @model_validator(mode="after")
    def _check(self) -> "OverextensionConfig":
        if self.action.value not in {"watchlist", "penalty"}:
            raise ValueError("overextension.action must be watchlist|penalty")
        return self


class EarningsDistortionConfig(_Base):
    enabled: bool = True
    action: str = "hard_reject"
    recent_earnings_sessions: TI
    recent_earnings_penalty: TF


class CorporateActionDistortionConfig(_Base):
    enabled: bool = True
    action: str = "hard_reject"
    lookback_sessions: TI


class InsiderActivityConfig(_Base):
    enabled: bool = True
    lookback_sessions: TI
    penalty: TF


class RegulatoryShockConfig(_Base):
    enabled: bool = True
    action: str = "hard_reject"


class RegimeTransitionCheckConfig(_Base):
    enabled: bool = True
    penalty: TF


class VolatilityShockConfig(_Base):
    enabled: bool = True
    vix_spike_pct: TF
    vix_spike_lookback_sessions: TI
    penalty: TF


class MomentumCrashConfig(_Base):
    enabled: bool = True
    prior_decline_lookback_sessions: TI
    prior_decline_threshold_pct: TF
    rebound_lookback_sessions: TI
    rebound_threshold_pct: TF
    action: str = "hard_reject"


class DataIntegrityCheckConfig(_Base):
    enabled: bool = True
    stale_data_action: str = "hard_reject"
    outlier_action: str = "hard_reject"
    source_disagreement_action: str = "flag"
    execution_realism_hard_reject_participation: TF
    execution_realism_penalty_per_pct_over: TF


class Stage5Config(_Base):
    top_n_to_defend: TI
    max_cumulative_penalty: TF
    low_volume_breakout: LowVolumeBreakoutConfig
    liquidity_distortion: LiquidityDistortionConfig
    gap_signal: GapSignalConfig
    news_spike: NewsSpikeConfig
    short_covering: ShortCoveringConfig
    beta_explained_move: BetaExplainedConfig
    overextension: OverextensionConfig
    earnings_distortion: EarningsDistortionConfig
    corporate_action_distortion: CorporateActionDistortionConfig
    insider_activity: InsiderActivityConfig
    regulatory_shock: RegulatoryShockConfig
    regime_transition: RegimeTransitionCheckConfig
    volatility_shock: VolatilityShockConfig
    momentum_crash: MomentumCrashConfig
    data_integrity: DataIntegrityCheckConfig


# =============================================================================
# 11. stage 6 -- entry
# =============================================================================


class ConfirmationConfig(_Base):
    require_volume_confirmation: bool = True
    volume_multiple: TF
    volume_lookback_sessions: TI
    require_delivery_confirmation: TB
    min_delivery_pct: TF
    reject_if_overextended: bool = True


class PullbackTriggerConfig(_Base):
    enabled: bool = True
    support_ma_sessions: TI
    max_distance_atr: TF
    require_reversal_candle: bool = True
    min_close_position_in_range: TF


class MaReclaimTriggerConfig(_Base):
    enabled: bool = True
    reference: TS
    ma_sessions: TI
    lookback_sessions: TI
    require_above_average_volume: bool = True

    @model_validator(mode="after")
    def _check(self) -> "MaReclaimTriggerConfig":
        if self.reference.value not in {"vwap_anchored", "ma"}:
            raise ValueError("ma_reclaim.reference must be vwap_anchored|ma")
        return self


class BreakoutTriggerConfig(_Base):
    enabled: bool = True
    lookback_sessions: TI
    min_volume_multiple: TF
    min_breakout_margin_pct: TF


class TriggersConfig(_Base):
    order: TLS
    pullback: PullbackTriggerConfig
    ma_reclaim: MaReclaimTriggerConfig
    breakout: BreakoutTriggerConfig

    @model_validator(mode="after")
    def _check(self) -> "TriggersConfig":
        known = {"pullback", "ma_reclaim", "breakout"}
        bad = [t for t in self.order.value if t not in known]
        if bad:
            raise ValueError(f"unknown trigger(s) in triggers.order: {bad}")
        if len(set(self.order.value)) != len(self.order.value):
            raise ValueError("triggers.order contains duplicates")
        return self


class EntryZoneConfig(_Base):
    half_width_atr: TF
    max_width_pct: TF
    round_to_paise: int = Field(5, ge=1, le=100)


class Stage6Config(_Base):
    confirmation: ConfirmationConfig
    triggers: TriggersConfig
    entry_zone: EntryZoneConfig


# =============================================================================
# 12. stage 7 -- risk
# =============================================================================


class AtrConfig(_Base):
    period_sessions: TI
    method: TS

    @model_validator(mode="after")
    def _check(self) -> "AtrConfig":
        if self.method.value not in {"wilder", "sma"}:
            raise ValueError("atr.method must be wilder|sma")
        return self


class StopLossConfig(_Base):
    atr_multiple: TF
    max_stop_distance_pct: TF
    min_stop_distance_pct: TF

    @model_validator(mode="after")
    def _check(self) -> "StopLossConfig":
        if self.min_stop_distance_pct.value >= self.max_stop_distance_pct.value:
            raise ValueError("stop_loss min_stop_distance_pct must be < max")
        return self


class TrailingStopConfig(_Base):
    enabled: bool = True
    style: TS
    atr_multiple: TF
    activate_after_r: TF


class TargetsConfig(_Base):
    t1_r_multiple: TF
    t2_r_multiple: TF
    snap_to_resistance_within_pct: TF
    resistance_lookback_sessions: TI

    @model_validator(mode="after")
    def _check(self) -> "TargetsConfig":
        if self.t1_r_multiple.value >= self.t2_r_multiple.value:
            raise ValueError("targets.t1_r_multiple must be < t2_r_multiple")
        return self


class ThesisInvalidationConfig(_Base):
    momentum_rank_exit_percentile: TF
    structure_ma_sessions: TI
    structure_buffer_atr: TF


class RiskCategoryConfig(_Base):
    standard_min_score: TF
    reduced_min_score: TF
    relative_vol_lookback_sessions: TI

    @model_validator(mode="after")
    def _check(self) -> "RiskCategoryConfig":
        if self.reduced_min_score.value >= self.standard_min_score.value:
            raise ValueError("risk_category.reduced_min_score must be < standard_min_score")
        return self


class HoldingPeriodConfig(_Base):
    min_holding_sessions: TI
    max_holding_sessions: TI

    @model_validator(mode="after")
    def _check(self) -> "HoldingPeriodConfig":
        if self.min_holding_sessions.value >= self.max_holding_sessions.value:
            raise ValueError("holding_period.min must be < max")
        return self


class ExitHierarchyConfig(_Base):
    thesis_invalidation: bool = True
    stop_loss_breach: bool = True
    new_hard_rejection: bool = True
    severe_regime_change: bool = True
    signal_reversal: bool = True
    trailing_stop: bool = True
    target_achieved: bool = True
    time_expiration: bool = True


class Stage7Config(_Base):
    atr: AtrConfig
    stop_loss: StopLossConfig
    trailing_stop: TrailingStopConfig
    targets: TargetsConfig
    thesis_invalidation: ThesisInvalidationConfig
    risk_category: RiskCategoryConfig
    holding_period: HoldingPeriodConfig
    exit_hierarchy: ExitHierarchyConfig


# =============================================================================
# 13. stage 8 -- final signal
# =============================================================================


class PortfolioCheckConfig(_Base):
    max_pairwise_correlation: TF
    correlation_lookback_sessions: TI
    max_signals_per_sector: TI
    max_signals_per_run: TI


class ScarcityConfig(_Base):
    min_composite_score: TF
    min_universe_percentile: TF
    expect_frequent_no_trade: bool = True


class StrengthBandsConfig(_Base):
    high_min: TF
    medium_min: TF

    @model_validator(mode="after")
    def _check(self) -> "StrengthBandsConfig":
        if self.medium_min.value >= self.high_min.value:
            raise ValueError("strength_bands.medium_min must be < high_min")
        return self


class NoTradeConfig(_Base):
    show_closest_n: TI


class Stage8Config(_Base):
    portfolio: PortfolioCheckConfig
    scarcity: ScarcityConfig
    strength_bands: StrengthBandsConfig
    no_trade: NoTradeConfig


# =============================================================================
# 14. costs
# =============================================================================


class ImpactModelConfig(_Base):
    type: TS
    coefficient: TF
    exponent: TF
    assumed_half_spread_bps: TF

    @model_validator(mode="after")
    def _check(self) -> "ImpactModelConfig":
        if self.type.value not in {"square_root", "linear", "fixed_bps"}:
            raise ValueError("impact_model.type must be square_root|linear|fixed_bps")
        return self


class StressTestConfig(_Base):
    cost_multiplier: float = Field(2.0, ge=1.0)
    impact_multiplier: float = Field(3.0, ge=1.0)
    require_edge_survives_stress: bool = True


class CostsConfig(_Base):
    segment: TS
    stt_delivery_buy_pct: TF
    stt_delivery_sell_pct: TF
    stt_intraday_sell_pct: TF
    stamp_duty_buy_pct: TF
    exchange_transaction_charge_pct: TF
    sebi_turnover_fee_pct: TF
    gst_pct_on_charges: TF
    brokerage_flat_per_order_inr: TF
    brokerage_pct_of_turnover: TF
    brokerage_cap_per_order_inr: TF
    dp_charge_per_scrip_sell_inr: TF
    impact_model: ImpactModelConfig
    stress_tests: StressTestConfig

    @model_validator(mode="after")
    def _check(self) -> "CostsConfig":
        if self.segment.value not in {"delivery", "intraday"}:
            raise ValueError("costs.segment must be delivery|intraday")
        return self


# =============================================================================
# 15-17. ledger / validation / api
# =============================================================================


class LedgerConfig(_Base):
    enabled: bool = True
    fail_run_if_unwritable: bool = True
    format: TS
    filename: str = "research_ledger.jsonl"
    write_run_detail: bool = True
    run_detail_subdir: str = "runs"
    retain_runs: int = Field(2000, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "LedgerConfig":
        if self.format.value not in {"jsonl", "sqlite", "both"}:
            raise ValueError("ledger.format must be jsonl|sqlite|both")
        return self


class CpcvConfig(_Base):
    n_groups: TI
    n_test_groups: TI
    purge_sessions: TI
    embargo_sessions: TI

    @model_validator(mode="after")
    def _check(self) -> "CpcvConfig":
        if self.n_test_groups.value >= self.n_groups.value:
            raise ValueError("cpcv.n_test_groups must be < n_groups")
        if self.n_groups.value < 3:
            raise ValueError("cpcv.n_groups must be at least 3")
        return self


class HoldoutConfig(_Base):
    reserve_most_recent_sessions: TI
    sacred: bool = True


class SignificanceConfig(_Base):
    t_stat_bar: TF


class LabelConfig(_Base):
    forward_return_sessions: TI


class SearchBudgetConfig(_Base):
    """Hard caps on how much searching is allowed to happen.

    The Deflated Sharpe Ratio penalises a result by the number of
    configurations tried. Rather than discovering after the fact that the
    penalty is unpayable, the budget is declared up front and the loader
    enforces it. A configuration that cannot fit inside the budget is a
    research-design problem, not a number to raise.
    """

    max_tier_a_parameters: int = Field(8, ge=1, le=25)
    max_grid_configurations: int = Field(512, ge=1)
    #: Trials already spent, carried across sessions. Feeds the DSR trial count.
    cumulative_trials_logged: int = Field(0, ge=0)
    #: Refuse to promote a parameter to VALIDATED when PBO exceeds this.
    max_acceptable_pbo: float = Field(0.5, ge=0.0, le=1.0)


class ValidationConfig(_Base):
    cpcv: CpcvConfig
    holdout: HoldoutConfig
    significance: SignificanceConfig
    label: LabelConfig
    search_budget: SearchBudgetConfig = Field(default_factory=SearchBudgetConfig)

    @model_validator(mode="after")
    def _check(self) -> "ValidationConfig":
        if self.cpcv.purge_sessions.value < self.label.forward_return_sessions.value:
            raise ValueError(
                "cpcv.purge_sessions must be >= label.forward_return_sessions, "
                "otherwise training labels overlap the test window and the "
                "backtest leaks (research program section 8, step 2)."
            )
        return self


class ApiConfig(_Base):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: List[str] = Field(default_factory=list)
    auth_token: Optional[str] = None
    allow_order_placement: bool = False
    disclaimer: str

    @model_validator(mode="after")
    def _no_trading(self) -> "ApiConfig":
        if self.allow_order_placement:
            raise ValueError(
                "api.allow_order_placement is a safety interlock and must stay "
                "false: this project contains no order-routing code and is a "
                "decision-support tool, not an auto-trader."
            )
        return self


# =============================================================================
# root
# =============================================================================


class RootConfig(_Base):
    """The fully validated contents of config/parameters.yaml."""

    meta: MetaConfig = Field(default_factory=MetaConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    capital: CapitalConfig
    universe: UniverseConfig
    providers: ProvidersConfig
    feeds: Dict[str, FeedPolicy]
    stage1_data_quality: Stage1Config
    stage2_regime: Stage2Config
    stage3_eligibility: Stage3Config
    stage4_core_score: Stage4Config
    stage5_false_signal: Stage5Config
    stage6_entry: Stage6Config
    stage7_risk: Stage7Config
    stage8_final_signal: Stage8Config
    costs: CostsConfig
    ledger: LedgerConfig
    validation: ValidationConfig
    api: ApiConfig

    # -- cross-section invariants ------------------------------------------
    @field_validator("feeds")
    @classmethod
    def _feeds_known(cls, v: Dict[str, FeedPolicy]) -> Dict[str, FeedPolicy]:
        unknown = sorted(set(v) - set(KNOWN_FEEDS))
        if unknown:
            raise ValueError(
                f"feeds contains unknown feed name(s): {unknown}. "
                f"Known feeds: {list(KNOWN_FEEDS)}"
            )
        missing = sorted(set(KNOWN_FEEDS) - set(v))
        if missing:
            raise ValueError(f"feeds is missing policy for: {missing}")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "RootConfig":
        errs: List[str] = []

        # The universe index must have a constituent file we know how to fetch.
        idx = self.universe.index_name.value
        if idx not in self.providers.nse_archives.index_constituent_files:
            errs.append(
                f"universe.index_name={idx!r} has no entry in "
                f"providers.nse_archives.index_constituent_files"
            )

        # Regime benchmark must be resolvable from the daily all-index file.
        for key, name in (
            ("stage2_regime.benchmark_index", self.stage2_regime.benchmark_index.value),
            ("stage2_regime.secondary_index", self.stage2_regime.secondary_index.value),
        ):
            if not name.strip():
                errs.append(f"{key} is empty")

        # History must cover the longest lookback any stage asks for.
        longest = max(
            self.stage4_core_score.factors.momentum_12_1.lookback_sessions.value
            + self.stage4_core_score.factors.momentum_12_1.skip_sessions.value,
            self.stage2_regime.trend.slow_ma_sessions.value,
            self.stage2_regime.breadth.ma_sessions.value,
            self.stage2_regime.volatility.vix_percentile_lookback_sessions.value,
            self.stage5_false_signal.beta_explained_move.beta_estimation_sessions.value,
            max(self.stage4_core_score.factors.sector_relative_strength.horizons_sessions.value),
            self.stage7_risk.targets.resistance_lookback_sessions.value,
        )
        if self.universe.min_history_sessions.value < longest:
            errs.append(
                f"universe.min_history_sessions="
                f"{self.universe.min_history_sessions.value} is shorter than the "
                f"longest lookback any stage requires ({longest}). Stocks would be "
                f"scored on incomplete windows."
            )

        # Earnings holding window should track the risk engine's max hold.
        hw = self.stage3_eligibility.earnings_proximity.holding_window_sessions.value
        mh = self.stage7_risk.holding_period.max_holding_sessions.value
        if hw > mh:
            errs.append(
                f"stage3_eligibility.earnings_proximity.holding_window_sessions"
                f"={hw} exceeds stage7_risk.holding_period.max_holding_sessions"
                f"={mh}; the earnings gate would reject names the engine would "
                f"never still be holding."
            )

        # PEAD interlock: the Stage 5 earnings hard-reject can only be relaxed
        # when the Stage 3 PEAD-conditional signal is explicitly enabled.
        if (
            self.stage5_false_signal.earnings_distortion.enabled
            and self.stage5_false_signal.earnings_distortion.action != "hard_reject"
            and not self.stage3_eligibility.earnings_proximity.pead_conditional_signal_enabled.value
        ):
            errs.append(
                "stage5.earnings_distortion.action may only be relaxed from "
                "hard_reject when stage3.earnings_proximity."
                "pead_conditional_signal_enabled is true (India PEAD evidence is "
                "contradictory -- see research program section 1.1)."
            )

        # Cost segment must match a delivery-oriented holding period.
        if self.costs.segment.value == "intraday" and mh > 1:
            errs.append(
                "costs.segment='intraday' contradicts "
                f"stage7_risk.holding_period.max_holding_sessions={mh}"
            )

        # Scarcity threshold must be reachable given the strength bands.
        if (
            self.stage8_final_signal.scarcity.min_composite_score.value
            > self.stage8_final_signal.strength_bands.high_min.value
        ):
            errs.append(
                "stage8.scarcity.min_composite_score exceeds "
                "strength_bands.high_min: every surviving signal would be 'High' "
                "by construction, which makes the band meaningless."
            )

        # --- anti-overfitting budget, enforced at load time ------------------
        # The Deflated Sharpe Ratio charges you for every configuration you
        # tried. Discovering after a three-week campaign that the penalty is
        # unpayable is too late, so the budget binds before the campaign runs.
        tunables = self.iter_tunables()
        tier_a = [t for t in tunables if t["tier"] == OptimizationTier.A_SEARCH.value]
        budget = self.validation.search_budget

        if len(tier_a) > budget.max_tier_a_parameters:
            errs.append(
                f"{len(tier_a)} parameters are marked optimization_tier="
                f"A_SEARCH but validation.search_budget.max_tier_a_parameters "
                f"is {budget.max_tier_a_parameters}. Demote the ones that do "
                f"not genuinely change the edge to B_SENSITIVITY. Raising the "
                f"cap instead raises the significance bar you then have to "
                f"clear (Harvey, Liu & Zhu 2016)."
            )

        configurations = 1
        for entry in tier_a:
            configurations *= max(int(entry["grid_points"]), 1)
        if configurations > budget.max_grid_configurations:
            errs.append(
                f"the declared A_SEARCH grid spans {configurations:,} "
                f"configurations, over the budget of "
                f"{budget.max_grid_configurations:,}. Grid size is "
                f"multiplicative -- drop a parameter or coarsen a grid rather "
                f"than raising the cap."
            )

        # A parameter cannot be both settled and up for search.
        contradictory = [
            t["path"]
            for t in tier_a
            if t["status"] in (ParamStatus.STATUTORY.value, ParamStatus.STRUCTURAL.value)
        ]
        if contradictory:
            errs.append(
                f"these parameters are STATUTORY or STRUCTURAL yet marked "
                f"A_SEARCH: {contradictory}. A statutory rate is set by law and "
                f"a structural constant defines what the factor is -- neither "
                f"is a free parameter."
            )

        if errs:
            raise ValueError(
                "parameters.yaml failed cross-section validation:\n  - "
                + "\n  - ".join(errs)
            )
        return self

    # -- introspection ------------------------------------------------------
    def iter_tunables(self) -> List[Dict[str, Any]]:
        """Flatten every Tunable in the tree, for the /config panel (FR-8)."""
        out: List[Dict[str, Any]] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, Tunable):
                out.append(
                    {
                        "path": path,
                        "value": node.value,
                        "status": node.status.value,
                        "tier": node.tier.value,
                        "tier_explicit": node.optimization_tier is not None,
                        "grid_points": node.grid_points,
                        "search_range": node.search_range,
                        "note": node.note,
                        "validated_by": node.validated_by,
                        "validated_on": (
                            node.validated_on.isoformat() if node.validated_on else None
                        ),
                    }
                )
                return
            if isinstance(node, BaseModel):
                for name in type(node).model_fields:
                    walk(getattr(node, name), f"{path}.{name}" if path else name)
                return
            if isinstance(node, dict):
                for k, val in node.items():
                    walk(val, f"{path}.{k}" if path else str(k))
                return

        walk(self, "")
        return sorted(out, key=lambda d: d["path"])

    def unvalidated_count(self) -> int:
        return sum(1 for t in self.iter_tunables() if t["status"] == "UNVALIDATED")

    def tier_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {t.value: 0 for t in OptimizationTier}
        for entry in self.iter_tunables():
            out[entry["tier"]] += 1
        return out

    def search_space_report(self) -> Dict[str, Any]:
        """What a CPCV campaign over this config would actually cost.

        ``grid_configurations`` is the product of the grid sizes of every
        A_SEARCH parameter -- the number of distinct configurations a full
        sweep would evaluate. It is the number that belongs in the Deflated
        Sharpe Ratio's trial count, and it grows multiplicatively, which is
        exactly why the tier system exists.
        """
        tunables = self.iter_tunables()
        tier_a = [t for t in tunables if t["tier"] == OptimizationTier.A_SEARCH.value]

        configurations = 1
        for entry in tier_a:
            configurations *= max(int(entry["grid_points"]), 1)

        budget = self.validation.search_budget
        naive = 1
        for entry in tunables:
            if entry["status"] == ParamStatus.UNVALIDATED.value:
                naive *= 3  # what a 3-point sweep of *everything* would cost

        return {
            "total_parameters": len(tunables),
            "tier_counts": self.tier_counts(),
            "tier_a_parameters": [
                {
                    "path": e["path"],
                    "value": e["value"],
                    "grid_points": e["grid_points"],
                    "search_range": e["search_range"],
                }
                for e in sorted(tier_a, key=lambda x: x["path"])
            ],
            "grid_configurations": configurations,
            "max_grid_configurations": budget.max_grid_configurations,
            "within_budget": configurations <= budget.max_grid_configurations,
            "cumulative_trials_logged": budget.cumulative_trials_logged,
            "effective_trials_if_swept": configurations
            + budget.cumulative_trials_logged,
            "naive_all_unvalidated_3pt_sweep": naive,
            "cpcv_paths": _cpcv_path_count(
                int(self.validation.cpcv.n_groups.value),
                int(self.validation.cpcv.n_test_groups.value),
            ),
        }


def _cpcv_path_count(n_groups: int, n_test_groups: int) -> int:
    """Number of distinct train/test splits CPCV generates: C(N, k)."""
    from math import comb

    if n_test_groups >= n_groups or n_test_groups < 1:
        return 0
    return comb(n_groups, n_test_groups)
