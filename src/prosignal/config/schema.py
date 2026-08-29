"""Typed schema for config/parameters.yaml.

Each research parameter carries metadata rather than being a bare number:
`Tunable` requires a `status` (UNVALIDATED / VALIDATED / STATUTORY /
STRUCTURAL / OPERATIONAL) and accepts an optional `search_range` and `note`.
The `/config` endpoint renders this metadata.

`extra="forbid"` applies throughout, so a typo in parameters.yaml fails at load
rather than falling back to an unseen default.

Both YAML forms are accepted::

    atr_multiple: 2.5                      # short form -> status UNVALIDATED
    atr_multiple:                          # long form
      value: 2.5
      status: UNVALIDATED
      search_range: [1.5, 3.5]
      note: "..."

Engine code reads ``cfg.stage7_risk.stop_loss.atr_multiple.value`` (or ``.v``).
Values are not unwrapped implicitly.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

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
    #: A THIRD THING, between hypothesis and promotion, and it exists because
    #: the two-state ladder forced a lie in both directions.
    #:
    #: A value chosen from a 4,877-configuration trade-level measurement with a
    #: six-year walk-forward, a stationary block bootstrap and a PBO estimate is
    #: not UNVALIDATED -- calling it that puts it beside numbers nobody has ever
    #: looked at, and the honest warning on every output stops meaning anything.
    #: It is also emphatically not VALIDATED: this file reserves that for a CPCV
    #: run that clears the Deflated Sharpe, and the shipped configuration does
    #: not clear it (DSR 0.028 against the conservative trial count).
    #:
    #: MEASURED means: there is a recorded experiment behind this number, the
    #: experiment is named in the note beside it, and it has NOT cleared the
    #: promotion bar. It stays a research hypothesis for every purpose the
    #: engine cares about -- the unvalidated-parameter warning still fires, the
    #: search budget still counts it -- and it tells a reader that changing it
    #: means arguing with evidence rather than with a placeholder.
    MEASURED = "MEASURED"
    VALIDATED = "VALIDATED"
    STATUTORY = "STATUTORY"
    STRUCTURAL = "STRUCTURAL"
    OPERATIONAL = "OPERATIONAL"

    @property
    def is_research_hypothesis(self) -> bool:
        """True when this value still needs CPCV before it can be trusted.

        MEASURED is included deliberately. Evidence short of the promotion bar
        is still evidence short of the promotion bar, and the whole point of
        the tier is that it does not buy silence.
        """
        return self in (ParamStatus.UNVALIDATED, ParamStatus.MEASURED)


class OptimizationTier(str, Enum):
    """How a parameter may be treated during validation.

    UNVALIDATED means a value has not been tested; it does not mean the value
    should be searched. Under Harvey, Liu & Zhu (2016) the significance bar
    rises with the number of configurations tried, so the search space is
    restricted deliberately.

    A_SEARCH
        Changes the edge. Gets a grid in CPCV, and every value tried counts
        toward the Deflated Sharpe trial budget. Opt-in, and the loader caps
        how many may exist.
    B_SENSITIVITY
        Perturbed to confirm the result is not knife-edge; the winning
        configuration is never selected on it. Default for UNVALIDATED values.
    C_FIXED
        Set once from evidence or convention and never searched: academic
        constructions such as 12-1 momentum, statutory rates, constants.
    D_OPERATIONAL
        A business constraint -- capital, fees, appetite. Changing it changes
        the problem rather than the answer.
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
    ParamStatus.MEASURED: OptimizationTier.B_SENSITIVITY,
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
    #: How far back a backfill may probe. Bounds an unattended run rather than
    #: expressing a data limit: NSE serves bhavcopy to at least 2017, so the
    #: default covers roughly eleven years.
    max_backfill_calendar_days: int = Field(4200, ge=100, le=20000)
    #: The training span the shipped coefficients were validated against. The
    #: model refits from stored history on EVERY run, so the store IS the
    #: training set and this is the depth at which the engine is the one that
    #: was measured. It decides what `/ready` calls "full validated depth" and,
    #: through that, whether the interface can detect a stale ranking at all --
    #: which is far too much to leave in a `getattr(..., 2200)`.
    validated_training_sessions: int = Field(2200, ge=376, le=10000)

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
    risk_per_trade_pct: TF
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
    source: TS
    pit_min_adtv_inr: TF
    pit_adtv_lookback_sessions: TI
    pit_max_names: TI
    #: Fit and rank ONLY on names the engine could actually open on the date --
    #: `exits.tradeable_at_entry`, the predicate stage 3 and stage 6 apply live.
    #:
    #: It was applied live and, in the label path, inside `resolve_exits`, which
    #: `build_panel` reaches only when exit rules exist -- which under
    #: `labels.triple_barrier: false` they do not. So the model was fitted and
    #: ranked on a population about a fifth larger than the book can buy, and
    #: the simulator discovered the difference at fill time by leaving slots
    #: empty: 7.29 of 8 filled.
    #:
    #: Turning this on changes the population the model is fitted on and
    #: therefore the traded coefficients. That is not a reason to leave it off;
    #: it is the expected consequence of correcting the training set. It is a
    #: config value rather than a code constant so the change is dated, hashed
    #: and carried on every ledger row from the moment it is made.
    train_on_admissible_only: TB

    @model_validator(mode="after")
    def _check_policy(self) -> "UniverseConfig":
        allowed = {"halt", "flag"}
        if self.pre_snapshot_policy.value not in allowed:
            raise ValueError(
                f"universe.pre_snapshot_policy must be one of {sorted(allowed)}"
            )
        sources = {"index_snapshot", "liquidity_pit"}
        if self.source.value not in sources:
            raise ValueError(f"universe.source must be one of {sorted(sources)}")
        if self.pit_adtv_lookback_sessions.value < 5:
            raise ValueError("universe.pit_adtv_lookback_sessions must be at least 5")
        if self.pit_max_names.value < 20:
            raise ValueError("universe.pit_max_names must be at least 20")
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
    unexplained_jump_lookback_sessions: TI
    max_universe_failure_fraction: TF
    min_universe_for_failure_fraction: TI
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


class ValueFactorConfig(_Base):
    """Earnings yield -- the factor with the strongest India-specific evidence.

    Computable only since 2026-08-18, when shares outstanding became derivable
    from the quarterly filing (paid-up capital / face value), which is what
    makes market capitalisation and therefore a yield possible.

    Earnings yield rather than P/E because it is defined for loss-making
    companies, where a negative yield ranks them last. A negative P/E is
    meaningless and usually dropped, which treats the worst names as neutral.
    """

    enabled: bool = True
    weight_band: TLF
    explicit_weight: TOF = None
    metric: TS = None

    @model_validator(mode="after")
    def _check(self) -> "ValueFactorConfig":
        allowed = {"earnings_yield"}
        if self.metric is not None and self.metric.value not in allowed:
            raise ValueError(
                f"stage4.factors.value.metric must be one of {sorted(allowed)}; "
                f"book-to-price and EV/EBITDA need a balance sheet, which Indian "
                f"quarterly filings do not carry."
            )
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


class FactorsConfig(_Base):
    """The factors that can actually be computed from obtainable data.

    ``estimate_revision_momentum`` was removed on 2026-08-17. It needed
    timestamped point-in-time analyst consensus estimates; a source audit found
    none available on any free or scrapeable India feed, and unlike the other
    gaps it cannot be derived from prices or filings -- a changed analyst
    opinion leaves no trace in market data. See DATA_SOURCES.md.
    """

    momentum_12_1: MomentumFactorConfig
    quality: QualityFactorConfig
    value: ValueFactorConfig
    sector_relative_strength: SectorRsFactorConfig


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


class LabelConfig(_Base):
    """How the training label is built.

    The horizon return is blind to the path: a name that fell 20% and recovered
    by day 63 scores the same as one that drifted up quietly, and the engine was
    stopped out of the first in week two. Triple-barrier labelling records
    whichever of the profit, stop or time barrier is touched FIRST, so the model
    is fitted against the trade it would actually have taken.
    """

    triple_barrier: bool = True
    #: "engine" derives the barriers from the stop and target Stage 7 ACTUALLY
    #: places -- 2.5 x ATR and 3.0R -- so the model is fitted against the trade
    #: the engine takes. "sigma" uses `upper_sigma`/`lower_sigma` below and is
    #: kept for research only.
    #:
    #: The sigma geometry shipped first and was wrong in a measurable way:
    #: across 156,446 observations its stop was 1.48x looser than the engine's
    #: for 88% of names and its target 1.57x tighter, giving the label a 1.33:1
    #: reward-to-risk profile against the engine's 3.0:1. 14% of everything it
    #: called a winner would have been stopped out.
    barrier_source: str = Field("engine", pattern="^(engine|sigma)$")
    #: Research only. See `barrier_source`.
    upper_sigma: float = Field(1.0, gt=0, le=5.0)
    lower_sigma: float = Field(0.75, gt=0, le=5.0)
    vol_window_sessions: int = Field(60, ge=20, le=252)
    #: Weight each row by how much of its outcome window it holds alone.
    #: Without it a 63-session label sampled every 21 counts one market shock
    #: once per overlapping row.
    uniqueness_weighting: bool = True


class EstimatorConfig(_Base):
    """How the theme coefficients are estimated.

    The pooled ridge stacked every (symbol, date) row into one design matrix and
    solved once, which treats ~33,000 rows as 33,000 independent observations.
    They are 70 cross-sections. Measured on a purged walk-forward over 50
    out-of-sample dates:

        arm                        IC    t(NW)   top-decile      t
        ridge                   +0.0021  +0.06     -0.11%    -0.17
        equal weight 1/N        -0.0022  -0.07     -0.38%    -0.79
        Fama-MacBeth, gated     +0.0510  +3.21     +1.25%    +2.55

    The ridge could not beat the 1/N control it is supposed to justify.
    """

    #: "fama_macbeth" or "ridge". Ridge is kept because it is the arm the
    #: Fama-MacBeth result is measured AGAINST, and an unreproducible baseline
    #: is not a baseline.
    method: str = Field("fama_macbeth", pattern="^(fama_macbeth|ridge)$")
    #: A theme below this on its own training window is set to exactly zero.
    #: Pre-committed at the conventional two-sigma bar. A floor of 1.65 measured
    #: BETTER out of sample here and was rejected for that reason -- choosing
    #: the threshold that scored best on the dates used to score it is how a
    #: backtest gets manufactured.
    significance_floor: float = Field(2.0, ge=0.0, le=6.0)
    #: Trailing cross-sections to estimate over, or null for every date in the
    #: training panel. This is what makes it ROLLING.
    window_dates: Optional[int] = Field(None, ge=12, le=600)
    #: Shrink surviving themes toward zero, or toward the prior-oriented pool
    #: mean. Zero by default: `lottery` carries a documented negative prior and
    #: measures IC +0.0485 here, and pooling hands an imprecise theme nearly the
    #: full prior mean -- a confident coefficient built out of an assumption.
    shrink_toward: str = Field("zero", pattern="^(zero|prior_mean)$")
    #: Replace the hard |t| >= floor CLIFF with a continuous taper,
    #: t^2 / (t^2 + c), so |t| = 2 keeps half its weight and |t| = 1 a fifth,
    #: with a hard zero below `taper_hard_floor` so a theme the window truly
    #: cannot measure still cannot steer the book.
    #:
    #: OFF by default, deliberately. The cliff is a real defect -- it makes a
    #: coefficient a step function of a noisy statistic, and `risk` sat at
    #: t +1.86 live and +2.45 on the rebuild, so the same theme is worth either
    #: nothing or nearly everything depending on which window is asked. But
    #: turning the taper ON changes live coefficients, which makes it an
    #: ESTIMATOR CHANGE rather than a correctness fix, and estimator changes are
    #: trials: PBO is already being charged against 81 of them and the Deflated
    #: Sharpe already reads 0.000. So the mechanism ships available and unused,
    #: to be decided by `research estimator` as a recorded comparison rather
    #: than by whoever edits this file.
    significance_taper: bool = False
    #: Curvature of the taper. 4.0 puts the half-weight point exactly at the
    #: |t| = 2 the cliff used, so the taper is a smoothing of the shipped rule
    #: rather than a different rule wearing its name.
    taper_c: float = Field(4.0, gt=0.0, le=100.0)
    #: Below this the coefficient is zero outright, tapered or not.
    taper_hard_floor: float = Field(1.0, ge=0.0, le=6.0)


class MetaLabelConfig(_Base):
    """The NO TRADE veto: a second model that decides whether to act.

    DISABLED, and the reason is measured. Meta-labelling (Lopez de Prado ch. 3)
    fits a binary classifier on the trades the primary model would actually have
    taken, predicting whether one reaches its profit barrier before its stop.
    Evaluated here on 1,432 out-of-sample shortlist rows over 40 dates:

        pooled AUC                     0.5698
        mean per-date AUC              0.4996   t vs 0.5   -0.02
        dates above 0.5                50%
        top-half minus bottom-half    -0.16% per period   t -0.15

    The pooled figure is the pooled-N illusion in a new place. Pooling across
    dates lets "this was a good period" masquerade as "this was a good name":
    within a date, which is the only question a per-name veto can answer, the
    classifier is a coin. Its calibration is also wrong in the direction that
    matters -- the top bucket predicts 0.817 and realises 0.547.

    Read as a DATE-level gate the pooled signal does reappear (trading only the
    higher-probability half of dates returns +8.62% against +0.86%), but that is
    market timing rather than trade selection; it rests on ~13 independent
    windows once the 63-session overlap is counted, not 40; and it was found by
    looking a second time after the first look failed. It is not enabled on that
    basis.

    The machinery is here, tested and wired, because the constraint is DATA:
    eight positions over seventy rebalances is roughly 370 decided trades in the
    whole history. Re-run `research metalabel` when the panel is longer.
    """

    enabled: bool = False
    #: How far down the primary ranking counts as a trade the engine would
    #: consider. Eight rows a date cannot support a classifier.
    shortlist_top_k: int = Field(50, ge=8, le=200)
    #: NO THRESHOLD HERE. The veto's floor is
    #: `stage8_final_signal.scarcity.min_win_probability`, which is where the
    #: gate actually runs. A second field of the same name lived here, was read
    #: by nothing, and survived the liveness check precisely because the leaf
    #: name is consumed elsewhere -- so the check that exists to catch a
    #: parameter stating behaviour the engine does not have was blind to it.
    #: See `liveness.SHARED_LEAF_NAMES` for the guard that now covers this.
    l2: float = Field(1.0, gt=0.0, le=1000.0)


class VolatilityScalingConfig(_Base):
    """Scaling total book exposure by how turbulent the market has been.

    Moreira & Muir (2017) show that scaling a portfolio by the inverse of its
    recent realised variance raises the Sharpe ratio. Measured here over 50
    out-of-sample rebalances it does not:

        target vol    mean ret      sd    Sharpe   avg scale
        off             +3.12%   7.87%    +0.79      1.00
        10%             +2.69%   7.40%    +0.73      0.74
        20%             +3.83%  10.15%    +0.76      1.21
        25%             +4.22%  11.04%    +0.77      1.35

    A 25% target returns +1.11% more per period at t +2.24, and none of that is
    alpha: average exposure is 1.35x, volatility rises with it, and the Sharpe
    falls. On mean return the overlay looks like it works; on the only measure
    invariant to leverage, switching it off wins.

    Note that position sizing is ALREADY inverse-volatility, through the ATR
    stop -- `risk_budget / (entry * atr_distance)` gives a high-ATR name a
    smaller position by construction. This block is the separate, aggregate
    question of how much book to have on at all, and that is the part the data
    does not support.
    """

    enabled: bool = False
    target_vol_annual: float = Field(0.20, gt=0.0, le=1.0)
    window_sessions: int = Field(21, ge=5, le=252)
    #: An uncapped inverse-variance rule takes enormous positions in the calmest
    #: stretch of the sample -- where a volatility estimate is least reliable and
    #: where a variance-scaled backtest earns most of its result.
    max_scale: float = Field(1.5, ge=1.0, le=4.0)
    min_scale: float = Field(0.5, ge=0.05, le=1.0)


class DecayMonitorConfig(_Base):
    """The pre-committed kill criterion for a factor theme.

    Declared HERE, before the numbers are looked at. A rule chosen after seeing
    which themes it would remove is not a rule; it is the selection it exists to
    prevent, wearing a lab coat.

    The criterion is stated as a principle rather than a tuned pair of numbers:

      A theme is killed when its trailing-window Newey-West t has been
      NON-POSITIVE on every check across a COMPLETE REFRESH of that window.

    Both halves are chosen for a reason and not for a score. Non-positive,
    rather than some threshold, because a t at or below zero says there is no
    positive relationship left at all -- it is a sign test, not a level somebody
    picked. A complete refresh, rather than "a few checks", because the rolling
    windows overlap almost entirely: requiring the breach to persist until every
    observation in the window arrived AFTER the breach began means no single bad
    quarter can end a theme.

    `required_breaches` is therefore tied to `window_dates` and not tuned
    separately.
    """

    #: Trailing cross-sections per check. 24 dates at a 21-session step is about
    #: two years -- long enough for a Newey-West t to mean something, short
    #: enough that a theme which died three years ago is not still being carried
    #: by its own history.
    window_dates: int = Field(24, ge=12, le=120)
    #: Non-positive. See above: a sign test, not a tuned level.
    kill_t_stat: float = Field(0.0, ge=-2.0, le=2.0)
    #: A complete refresh of the window. Kept as its own field so it is visible,
    #: but it is meant to equal `window_dates`.
    required_breaches: int = Field(24, ge=1, le=120)
    #: McLean & Pontiff (2016): published anomalies lose about 58% of their
    #: return out of sample. Every theme here comes from a published paper, so
    #: the honest expectation is the haircut coefficient -- and a theme merely
    #: MEETING it is behaving as the literature predicts, not underperforming.
    post_publication_haircut: float = Field(0.58, ge=0.0, lt=1.0)


class RankingConfig(_Base):
    """What orders the book.

    Separated from the estimator on purpose. The estimator answers "what are
    these themes worth?", which the engine still asks and still records; this
    answers "what do we buy?", which turned out to be a different question with
    a different answer. Keeping them as one setting is what let a scorer that
    fails at the second keep doing it on the strength of doing the first.

    Measured over 4,877 trade-level configurations on a rebuilt point-in-time
    panel, scored against an investable equal-weight benchmark of the same
    eligible universe:

        signal                         configs   median alpha   share positive
        mom_6_1_r (sector-neutral)         960        +7.0%          98.1%
        1/N over the seven families        960        +3.0%          84.9%
        composite refitted on RETURN       960        -4.7%           6.2%
        fitted composite as shipped        144           --           0.0%

    The mechanism is in `parameters.yaml` beside the setting: the composite is
    fitted against a cross-sectional RANK, the cross-section is strongly
    right-skewed, and a book of eight names out of seven hundred lives entirely
    in the tail a rank target is indifferent to. Its rank IC is positive
    (+0.0338 at H=63) while its top-decile excess is negative (-0.35%, t -0.28).
    """

    #: v2_composite | measured_factor | fitted_composite | family_average.
    source: str = Field("measured_factor",
                        pattern="^(v2_composite|measured_factor|fitted_composite"
                                "|family_average)$")
    #: v2_composite only: how many of the ten v2 factors a name must have before
    #: it is scored at all. A name ranked on four of ten is not comparable with
    #: one ranked on ten, and median-filling the gap ranks it by a number nobody
    #: computed for it.
    v2_min_factors: TI = Field(default_factory=lambda: Tunable[int](
        value=7, status="MEASURED",
        note="Seven of ten. Below it a name is ranked on a minority of the "
             "model and is not comparable with one ranked on all of it."))
    #: The column to rank on when `source` is not `fitted_composite`. Must be a
    #: ranked feature column (`_r`) or a family column (`_f`) the model builds;
    #: Stage 4 refuses a name it cannot find rather than falling back silently,
    #: because a silent fallback here restores the scorer this setting exists
    #: to retire.
    column: str = "mom_6_1_r"
    note: Optional[str] = None


class Stage4Config(_Base):
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    weighting_mode: TS
    standardisation: TS
    winsorize_pct: TF
    sector_neutral: TB
    factors: FactorsConfig
    redundancy: RedundancyConfig
    min_name_factor_coverage: TF
    data_quality_gate_penalty: TF
    #: A filing older than this is not evidence about current profitability.
    #: Beyond it the fundamental factors report unavailable, so Stage 4 drops
    #: them and renormalises rather than scoring on a stale figure. 240 days
    #: allows a missed quarter plus the 45-day disclosure lag.
    max_fundamental_age_days: int = Field(240, ge=60, le=1095)
    #: Quarters pulled per symbol when refreshing. Eight covers two years of
    #: TTM plus the prior-year comparison that earnings growth needs.
    fundamental_quarters: int = Field(12, ge=4, le=40)
    #: Permit scoring from the hand-weighted composite when the fitted model
    #: cannot run. Default false: that composite measured -0.047%/month excess
    #: at t = -0.11, so a silent fallback issues signals from a scorer known
    #: not to work.
    allow_composite_fallback: bool = False
    #: Cross-sectional model. These lived as module constants, exempt from the
    #: one-config-file rule and from the search-budget accounting.
    model_horizon_sessions: int = Field(63, ge=5, le=252)
    model_ridge_alpha: float = Field(20_000.0, gt=0)
    model_max_train_sessions: int = Field(3000, ge=300, le=20000)
    model_refit_every_sessions: int = Field(21, ge=1, le=252)
    model_min_train_rows: int = Field(600, ge=100)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    metalabel: MetaLabelConfig = Field(default_factory=MetaLabelConfig)
    decay_monitor: DecayMonitorConfig = Field(default_factory=DecayMonitorConfig)

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
    penalty: TF


class NewsSpikeConfig(_Base):
    enabled: bool = True
    move_sigma: TF
    volume_multiple: TF
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
    upcoming_earnings_sessions: TI
    recent_earnings_sessions: TI
    recent_earnings_penalty: TF


class CorporateActionDistortionConfig(_Base):
    enabled: bool = True
    action: str = "hard_reject"
    lookback_sessions: TI


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
    beta_explained_move: BetaExplainedConfig
    overextension: OverextensionConfig
    earnings_distortion: EarningsDistortionConfig
    corporate_action_distortion: CorporateActionDistortionConfig
    regime_transition: RegimeTransitionCheckConfig
    volatility_shock: VolatilityShockConfig
    momentum_crash: MomentumCrashConfig
    data_integrity: DataIntegrityCheckConfig


# =============================================================================
# 11. stage 6 -- entry
# =============================================================================


class AdmissionConfig(_Base):
    """Rank bands with hysteresis. See parameters.yaml for the measurement."""

    entry_rank: TI
    exit_rank: TI
    #: Sessions between entry opportunities. The engine runs EVERY session --
    #: the disaster floor, the open book and outcome resolution all need it --
    #: but new positions open only on a cadence date. A daily entry clock and a
    #: 21-session entry clock are different strategies and only one of them was
    #: measured; 1 restores the daily behaviour.
    entry_cadence_sessions: TI = Field(default_factory=lambda: Tunable[int](
        value=1, status=ParamStatus.OPERATIONAL))
    #: The session on or after this date is cadence date zero. Counted in
    #: SESSIONS against the exchange calendar rather than in calendar days, so a
    #: holiday shifts nothing and the schedule is reproducible on any machine.
    entry_cadence_anchor: TS = Field(default_factory=lambda: Tunable[str](
        value="2026-09-01", status=ParamStatus.OPERATIONAL))
    #: Refuse an entry on a name already below its thesis-invalidation level.
    #: This is the population the model is fitted on -- `resolve_exits` gives
    #: such a name a NaN label and `build_panel` drops the row -- so with this
    #: off the engine trades a population no validation has measured.
    require_above_invalidation: TB

    @model_validator(mode="after")
    def _check(self) -> "AdmissionConfig":
        if int(self.exit_rank.value) <= int(self.entry_rank.value):
            raise ValueError(
                "stage6_entry.admission.exit_rank must be wider than entry_rank; "
                "equal bands are no hysteresis at all and a name on the boundary "
                "pays a round trip at every rebalance"
            )
        if int(self.entry_cadence_sessions.value) < 1:
            raise ValueError(
                "stage6_entry.admission.entry_cadence_sessions must be at least "
                "1; a cadence of zero would mean the book can never open"
            )
        try:
            dt.date.fromisoformat(str(self.entry_cadence_anchor.value))
        except ValueError as exc:
            raise ValueError(
                f"stage6_entry.admission.entry_cadence_anchor must be an "
                f"ISO date (YYYY-MM-DD); got "
                f"{self.entry_cadence_anchor.value!r}. The anchor decides which "
                f"sessions are entry dates, so an unparseable one would silently "
                f"re-phase every entry in the recorded history."
            ) from exc
        return self


class ConfirmationConfig(_Base):
    require_volume_confirmation: bool = True
    volume_multiple: TF
    volume_lookback_sessions: TI
    require_delivery_confirmation: TB
    min_delivery_pct: TF


class PullbackTriggerConfig(_Base):
    enabled: bool = True
    support_ma_sessions: TI
    max_distance_atr: TF
    require_reversal_candle: bool = True
    min_close_position_in_range: TF


class MaReclaimTriggerConfig(_Base):
    """The reclaim trigger.

    ``reference`` accepted "vwap_anchored" and the config shipped set to it,
    with a note saying the output would say so. Nothing read the field:
    stage6._ma_reclaim has only ever used a simple moving average, and vwap is
    not among the columns read_prices returns. ``require_above_average_volume``
    was likewise never read -- _ma_reclaim is not passed volumes at all, and the
    volume bar that does apply is confirmation.volume_multiple, shared by every
    trigger. Both are removed rather than left declaring behaviour that does not
    exist. Restoring either means implementing it first.
    """

    enabled: bool = True
    ma_sessions: TI
    lookback_sessions: TI


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
    admission: AdmissionConfig
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
    #: Whether the level CLOSES a position. False ships: measured alone on the
    #: production configuration it cost 15.6 points of per-trade win probability
    #: and 14.3 points of annual alpha. The level is still computed, still shown
    #: as exit condition #1 on the card, and still the ADMISSION predicate -- a
    #: name already below it is not bought, which is a different and cheaper use
    #: of the same number and the one the training panel depends on.
    #:
    #: Duplicated by `exit_hierarchy.thesis_invalidation` on purpose, and they
    #: must agree: the hierarchy switch is where an operator turns a rung off,
    #: and this one is where the RULE says whether it is an exit at all. The
    #: validator below refuses a configuration where one says exit and the other
    #: says not, because a disagreement there is silent and decides trades.
    enabled_as_exit: bool = False
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
        # `<` rather than `<=`, because 0 is now a legitimate minimum: the
        # 15-session floor was inert on a 21-session entry clock and saying so
        # in the file is better than implying a rule the engine does not run.
        if self.min_holding_sessions.value >= self.max_holding_sessions.value:
            raise ValueError("holding_period.min must be < max")
        if self.min_holding_sessions.value < 0:
            raise ValueError("holding_period.min_holding_sessions cannot be negative")
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
    volatility_scaling: VolatilityScalingConfig = Field(
        default_factory=VolatilityScalingConfig)

    @model_validator(mode="after")
    def _exits_agree(self) -> "Stage7Config":
        """Two switches control one rule; they must not disagree.

        `thesis_invalidation.enabled_as_exit` says whether the RULE is an exit.
        `exit_hierarchy.thesis_invalidation` is the rung an operator turns off.
        Both were true when only one was ever read, so an operator could switch
        off a rung and have it keep firing -- or the reverse. A disagreement
        here is silent and decides trades, which is the definition of a setting
        that has to be checked at load rather than at the point of use.

        The same pairing applies to the trailing stop, which has an `enabled`
        flag of its own next to a hierarchy rung.
        """
        if bool(self.thesis_invalidation.enabled_as_exit) != bool(
                self.exit_hierarchy.thesis_invalidation):
            raise ValueError(
                f"stage7_risk.thesis_invalidation.enabled_as_exit "
                f"({self.thesis_invalidation.enabled_as_exit}) and "
                f"stage7_risk.exit_hierarchy.thesis_invalidation "
                f"({self.exit_hierarchy.thesis_invalidation}) disagree. They "
                f"control the same exit and the engine reads both, so one of "
                f"them would be silently ignored. Set them the same."
            )
        if bool(self.trailing_stop.enabled) != bool(
                self.exit_hierarchy.trailing_stop):
            raise ValueError(
                f"stage7_risk.trailing_stop.enabled "
                f"({self.trailing_stop.enabled}) and "
                f"stage7_risk.exit_hierarchy.trailing_stop "
                f"({self.exit_hierarchy.trailing_stop}) disagree. They control "
                f"the same exit. Set them the same."
            )
        return self


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
    #: Today's prediction spread as a FRACTION of what this model's spread
    #: normally is, measured on its own training panel. A percentile gate cannot
    #: express a flat day -- `min_universe_percentile = 90` admits the top 10% by
    #: construction whether or not the top 10% is any better than the middle.
    #:
    #: A ratio rather than a level, because the level is a function of the ridge
    #: penalty: measured across 88 panel dates the entire range was 0.0355 to
    #: 0.0607, so any absolute floor near the label's own scale blocks every day.
    min_dispersion_ratio: TF
    #: A NEW entry whose modelled probability of reaching target before stop
    #: falls below this is refused. 0.0 vetoes nothing, which is the shipped
    #: default -- see MetaLabelConfig for the measurement behind that. Held
    #: positions are exempt: they are governed by the Stage 6 exit band, and a
    #: classifier refitted every 21 sessions must not be able to close a trade
    #: it was not consulted about opening.
    min_win_probability: TF
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
    #: Participation assumed when ADTV is unknown. The engine refuses to TRADE
    #: such a name; this is what the cost model answers if it is asked anyway,
    #: and it is set at the model's own participation cap so that unknown
    #: liquidity can never price cheaper than known-thin liquidity.
    unknown_liquidity_participation: TF

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


class ValidationLabelConfig(_Base):
    """The label horizon the VALIDATION harness measures over.

    Renamed from `LabelConfig`, which collided with the stage-4 block of the
    same name defined earlier in this module. The second definition silently
    replaced the first in the module namespace, so `from ...schema import
    LabelConfig` returned this one -- a class with a single required field --
    while `Stage4Config` kept a reference to the real one because pydantic
    resolved the annotation before the shadowing happened. Production was
    unaffected and any importer got the wrong class.
    """

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
    label: ValidationLabelConfig
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
    #: When a RUNNING job is declared dead so the button unblocks. Read through
    #: `getattr(cfg.params.api, "job_timeout_seconds", 900)` and present in
    #: neither this schema nor parameters.yaml, so it read like configuration,
    #: could not be inspected by `config show`, and could not be changed. It is
    #: also the moment a still-running analysis is marked FAILED and its slot
    #: freed for a second one, which on a 1 GB instance under swap is not a
    #: remote possibility.
    job_timeout_seconds: float = Field(1800.0, ge=60.0, le=21600.0)
    #: Sessions fetched per press of BUILD DATA STORE, and per nightly backfill
    #: step. Same problem: a hardcoded 90 behind a getattr.
    bootstrap_chunk_sessions: int = Field(90, ge=10, le=1000)

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



def _validate_horizon_alignment(params) -> None:
    """The model's label horizon and the engine's holding cap must agree.

    They were independent numbers that happened to match. Editing either alone
    would leave the model forecasting a 63-session return while the engine sold
    after 21, with nothing to report the mismatch.
    """
    h = int(params.stage4_core_score.model_horizon_sessions)
    m = int(params.stage7_risk.holding_period.max_holding_sessions.value)
    if h != m:
        raise ValueError(
            f"stage4_core_score.model_horizon_sessions ({h}) must equal "
            f"stage7_risk.holding_period.max_holding_sessions ({m}); the model "
            f"would otherwise forecast a window the engine never holds for"
        )


class CostScenarioExpectancy(_Base):
    """One cost scenario's realised frequencies. A ROW, not a setting."""

    name: str
    cost_bps: float
    p_win: float = Field(ge=0.0, le=1.0)
    p_beat: float = Field(ge=0.0, le=1.0)
    mean_net_pct: float
    median_net_pct: float


class ExpectancyConfig(_Base):
    """What a trade from the shipped configuration has historically done.

    Recorded in the config rather than computed at run time on purpose. These
    are the results of a study that took twenty minutes of compute over eight
    years of panel data; recomputing them on every run would be absurd, and
    recomputing them on LIVE data as it arrives would be worse -- the figures
    would drift with the very record they are supposed to be scored against,
    and calibration would become impossible by construction.

    So they are frozen, dated, and named. When the study is re-run, this block
    is rewritten and `measured_on` moves, which is a visible, reviewable event.

    NOT A FORECAST. Every field is a frequency over the study's 258 trades. The
    engine estimates no per-name edge and this block must never be read as one;
    `TradePlan.caveat` carries that sentence onto every card that quotes it.
    """

    enabled: bool = True
    #: The study these came from. Free text, but it has to identify a run.
    study: str
    measured_on: dt.date
    sample_trades: int = Field(ge=1)
    sample_period: str
    assumed_cost_bps: float = Field(ge=0.0)
    probability_of_profit: float = Field(ge=0.0, le=1.0)
    probability_of_beating_benchmark: float = Field(ge=0.0, le=1.0)
    expected_return_pct: float
    median_return_pct: float
    expected_excess_pct: float
    median_excess_pct: float
    expected_hold_sessions: float = Field(gt=0.0)
    by_cost: List[CostScenarioExpectancy] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ExpectancyConfig":
        if self.by_cost:
            names = [v.name for v in self.by_cost]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"expectancy.by_cost has duplicate scenario names {names}; "
                    f"a reader looking one up would get whichever came last."
                )
            match = [v for v in self.by_cost
                     if abs(float(v.cost_bps) - float(self.assumed_cost_bps)) < 1e-9]
            if not match:
                raise ValueError(
                    f"expectancy.assumed_cost_bps ({self.assumed_cost_bps}) "
                    f"matches none of the by_cost scenarios "
                    f"({[(v.name, v.cost_bps) for v in self.by_cost]}). The "
                    f"headline figures would then be net of a cost the scenario "
                    f"table does not contain, and nobody could tell which."
                )
            row = match[0]
            if abs(float(row.p_win) - float(self.probability_of_profit)) > 5e-4:
                raise ValueError(
                    f"expectancy.probability_of_profit "
                    f"({self.probability_of_profit}) disagrees with the "
                    f"{row.name!r} row of by_cost ({row.p_win}) at the same "
                    f"cost. The headline and the table came from one study and "
                    f"cannot disagree; one of them was edited by hand."
                )
        # The mean of a right-skewed distribution sits above its median, and
        # these two came from the same 258 trades. If they cross, the block was
        # edited by hand and one of them is stale -- which is precisely the
        # failure this whole section exists to make impossible.
        if self.median_return_pct > self.expected_return_pct:
            raise ValueError(
                f"expectancy.median_return_pct ({self.median_return_pct}) is "
                f"above expected_return_pct ({self.expected_return_pct}). The "
                f"trade distribution is right-skewed, so the mean is above the "
                f"median; these two cannot have come from the same study."
            )
        return self


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
    expectancy: ExpectancyConfig
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

        # The unexplained-jump scan must cover the longest feature lookback.
        # It used to share continuity_window_sessions at 60 while prox_52w and
        # resid_mom look back 253, so an unadjusted corporate action between 61
        # and 253 sessions back was invisible to Stage 1 and fully consumed by
        # the model. VEDL's 2026-04-30 demerger passed with no flags.
        from ..features.crosssec import MIN_LOOKBACK

        scan = int(self.stage1_data_quality.unexplained_jump_lookback_sessions.value)
        if scan < MIN_LOOKBACK:
            errs.append(
                f"stage1_data_quality.unexplained_jump_lookback_sessions ({scan}) "
                f"is shorter than the model's longest feature lookback "
                f"({MIN_LOOKBACK}); a corporate action in the gap would corrupt "
                f"features that validation never inspects"
            )

        # Purging must cover the whole label window. At purge 21 against a
        # 63-session label, 42 sessions of every training row's label reached
        # into the test block -- the exact leak purging exists to remove, and it
        # flatters every number measured through it.
        horizon = int(self.stage4_core_score.model_horizon_sessions)
        purge = int(self.validation.cpcv.purge_sessions.value)
        label = int(self.validation.label.forward_return_sessions.value)
        if purge < horizon:
            errs.append(
                f"validation.cpcv.purge_sessions ({purge}) is shorter than "
                f"stage4_core_score.model_horizon_sessions ({horizon}); training "
                f"rows would keep {horizon - purge} sessions of label overlap "
                f"with the test block"
            )
        if label != horizon:
            errs.append(
                f"validation.label.forward_return_sessions ({label}) must equal "
                f"stage4_core_score.model_horizon_sessions ({horizon}); the "
                f"harness would otherwise purge for a different label than the "
                f"model it is validating forecasts"
            )

        # The admission band, the book size and the per-run cap describe the
        # same book from three angles and were independent numbers. With
        # entry_rank 8 and max_signals_per_run 5, Stage 6 admits eight names and
        # Stage 8 silently drops three of them -- a smaller book than either
        # setting asks for, and nothing reports the disagreement.
        entry = int(self.stage6_entry.admission.entry_rank.value)
        book = int(self.capital.max_open_positions.value)
        per_run = int(self.stage8_final_signal.portfolio.max_signals_per_run.value)
        if entry != book:
            errs.append(
                f"stage6_entry.admission.entry_rank ({entry}) must equal "
                f"capital.max_open_positions ({book}); the entry band IS the "
                f"book size, and two different numbers mean one of them is "
                f"never reached"
            )
        if per_run < entry:
            errs.append(
                f"stage8_final_signal.portfolio.max_signals_per_run ({per_run}) "
                f"is below stage6_entry.admission.entry_rank ({entry}), so "
                f"Stage 8 would discard names Stage 6 admitted and the book "
                f"could never reach its own target size"
            )

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

    @model_validator(mode="after")
    def _check_horizon_alignment(self) -> "RootConfig":
        _validate_horizon_alignment(self)
        return self


def _cpcv_path_count(n_groups: int, n_test_groups: int) -> int:
    """Number of distinct train/test splits CPCV generates: C(N, k)."""
    from math import comb

    if n_test_groups >= n_groups or n_test_groups < 1:
        return 0
    return comb(n_groups, n_test_groups)
