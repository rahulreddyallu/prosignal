"""Pipeline orchestrator -- what RUN MARKET ANALYSIS executes.

Composes the eight stages in order and returns one `FinalSignalOutput`. Stages
never call each other; only this module knows the sequence, which keeps each
independently testable.

Every run stamps the engine version, config hash, resolved decision date and
data timestamps into the output, so a signal can be reconstructed later.

A `MarketWideHalt` from Stage 1 propagates as a blocked run with reasons
attached rather than degrading into NO TRADE. Refusing to form a view and
finding nothing that qualifies are different results, and conflating them would
hide a broken feed behind a normal-looking one.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from .config.loader import AppConfig
from .core.calendar import TradingCalendar
from .core.contracts import (
    FinalSignalOutput,
    RawDataManifest,
    RiskPlan,
    RunContext,
)
from .core.errors import IntegrityError, MarketWideHalt
from .core.logging import get_logger
from .core.memory import release_memory
from .costs import CostModel
from .data.store import DataStore
from .data.storelock import store_lock
from .data.types import DATE, SYMBOL
from .data.universe import UniverseSnapshot
from .ledger import Ledger, row_from_output
from .stages._cfg import v
from .stages import (
    stage1_data_quality,
    stage2_regime,
    stage3_eligibility,
    stage4_core_score,
    stage5_false_signal,
    stage6_entry,
    stage7_risk,
    stage8_final_signal,
)
from .version import ENGINE_VERSION, SCHEMA_VERSION

__all__ = ["run_analysis", "AnalysisRun", "PipelineBlocked"]

log = get_logger(__name__)

#: Ordered stage labels the UI shows as progress.
STAGE_LABELS = [
    "Loading market data",
    "Validating data quality",
    "Assessing market regime",
    "Screening eligibility",
    "Scoring the universe",
    "Running false-signal defense",
    "Checking entry triggers",
    "Building risk plans",
    "Applying decision gates",
]


class PipelineBlocked(Exception):
    """The run refused to produce an opinion. NOT the same as NO TRADE."""

    def __init__(self, reasons: List[str], stage: str) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.stage = stage


@dataclass
class AnalysisRun:
    output: FinalSignalOutput
    context: RunContext
    timings_ms: Dict[str, float] = field(default_factory=dict)
    funnel: Dict[str, int] = field(default_factory=dict)


def run_analysis(
    config: AppConfig,
    as_of: Optional[dt.date] = None,
    progress: Optional[Callable[[int, str], None]] = None,
    manifest: Optional[RawDataManifest] = None,
) -> AnalysisRun:
    """Execute the full decision pipeline for one date."""
    started = dt.datetime.now()
    run_id = uuid.uuid4().hex[:12]
    timings: Dict[str, float] = {}
    step = _stepper(progress)

    # A shared lock for the whole run. Every file the stages read has to come
    # from one moment: prices at Friday and delivery at Tuesday are each valid
    # and describe no day that existed. Readers do not block each other; only a
    # writing ingest excludes them, and then the analysis says so rather than
    # waiting out a multi-minute rewrite.
    lock = store_lock(config.paths.curated, exclusive=False, what="analysis")
    with lock:
        return _run_analysis_locked(config, as_of, progress, manifest, started,
                                    run_id, timings, step)


def _run_analysis_locked(config, as_of, progress, manifest, started, run_id,
                         timings, step) -> AnalysisRun:
    store = DataStore(config.paths.curated, config.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise PipelineBlocked(
            ["the local store has no price sessions; run `prosignal data ingest --full`"],
            stage="stage0_data",
        )
    calendar = TradingCalendar(sessions)

    step(0)
    t = _clock()
    resolved = calendar.last_session_on_or_before(as_of) if as_of else calendar.last
    if resolved is None:
        raise PipelineBlocked(
            [f"no trading session on or before {as_of} in the local store"],
            stage="stage0_data",
        )
    universe = _universe(store, config, resolved)
    manifest = manifest or _manifest_from_store(store, config, run_id, resolved, universe)
    timings["stage0_data"] = t()

    context = RunContext(
        run_id=run_id,
        trial_id=f"T-{run_id[:6]}",
        as_of_date=resolved,
        requested_date=as_of,
        started_at=started,
        engine_version=ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        config_version=config.version,
        mode="live",
    )

    # ---- Stage 1 ----------------------------------------------------------
    # One read for the stages that share a window. Measured on a warm run,
    # stages 1 to 5 made seven calls that decoded 255,270 rows to produce
    # 15.4 MB of frames, and the widest window contained almost all of the
    # others. The cache serves a later read only when its symbols, dates and
    # columns are all inside what was fetched, so anything unusual still goes
    # to the store.
    _prefetch_prices(store, config, universe, resolved, calendar)

    step(1)
    t = _clock()
    try:
        quality = stage1_data_quality.run(manifest, store, calendar, universe, config)
    except MarketWideHalt as halt:
        raise PipelineBlocked(halt.reasons, stage=stage1_data_quality.STAGE_NAME) from halt
    timings[stage1_data_quality.STAGE_NAME] = t()
    release_memory()

    # ---- Stage 2 ----------------------------------------------------------
    step(2)
    t = _clock()
    regime = stage2_regime.run(store, calendar, universe.symbols, config, as_of=resolved)
    timings[stage2_regime.STAGE_NAME] = t()
    release_memory()

    # ---- Stage 3 ----------------------------------------------------------
    step(3)
    t = _clock()
    eligibility = stage3_eligibility.run(
        universe, store, calendar, quality, config, as_of=resolved
    )
    timings[stage3_eligibility.STAGE_NAME] = t()
    release_memory()

    # ---- Stage 4 ----------------------------------------------------------
    step(4)
    t = _clock()
    scores = stage4_core_score.run(
        eligibility, store, calendar, regime, config, as_of=resolved
    )
    timings[stage4_core_score.STAGE_NAME] = t()
    release_memory()

    # ---- Stage 5 ----------------------------------------------------------
    step(5)
    t = _clock()
    defense = stage5_false_signal.run(
        scores, store, calendar, regime, config, as_of=resolved
    )
    timings[stage5_false_signal.STAGE_NAME] = t()
    release_memory()

    # ---- price frames shared by stages 6-8 --------------------------------
    defended = list(defense.per_stock)
    frames = _frames(store, calendar, defended, config, resolved)
    closes = _closes(frames)

    # ---- Stage 6 ----------------------------------------------------------
    step(6)
    t = _clock()
    # Stage 6 admits on rank with hysteresis, so it needs two things this run
    # does not otherwise carry: where each name sits in the model's ranking, and
    # what the previous run committed to. The engine holds no live position
    # state -- the ledger is the record of the open book.
    ranks = {s.ticker: s.rank for s in scores.ranked_scores}
    open_book = Ledger(config.paths.ledger).open_book(before=resolved)
    entries = stage6_entry.run(defended, frames, config, resolved,
                               ranks=ranks, held=open_book)
    timings[stage6_entry.STAGE_NAME] = t()

    # ---- Stage 7 ----------------------------------------------------------
    step(7)
    t = _clock()
    costs = CostModel(config)
    plans: Dict[str, RiskPlan] = {}
    for sym in defended:
        frame = frames.get(sym)
        decision = entries.decisions.get(sym)
        if frame is None or decision is None or decision.reference_price is None:
            continue
        score = next((s for s in scores.ranked_scores if s.ticker == sym), None)
        if score is None:
            continue
        plans[sym] = stage7_risk.build_plan(
            ticker=sym,
            frame=frame,
            reference_price=float(decision.reference_price),
            composite_score=defense.per_stock[sym].score_after,
            adtv_inr=eligibility.adtv_inr.get(sym),
            config=config,
            costs=costs,
        )
    timings[stage7_risk.STAGE_NAME] = t()
    release_memory()

    # ---- Stage 8 ----------------------------------------------------------
    step(8)
    t = _clock()
    buys, watch, no_trade = stage8_final_signal.run(
        regime=regime,
        eligibility=eligibility,
        scores=scores,
        defense=defense,
        entries=entries,
        plans=plans,
        closes=closes,
        config=config,
        company_names=dict(universe.company_names),
    )
    timings[stage8_final_signal.STAGE_NAME] = t()

    flags = list(quality.market_wide_soft_flags)
    if quality.failed_symbols:
        flags.append(
            f"{quality.failed_symbols} of {quality.checked_symbols} names failed "
            f"Stage 1 data-quality checks and were excluded."
        )

    output = FinalSignalOutput(
        run_id=run_id,
        trial_id=context.trial_id,
        as_of_date=resolved,
        generated_at=dt.datetime.now(),
        engine_version=ENGINE_VERSION,
        config_version=config.version,
        regime_state=regime,
        recommendations=buys,
        watchlist=watch,
        no_trade=no_trade,
        data_quality_flags=flags,
        manifest=manifest,
        stage_timings_ms={k: round(v, 1) for k, v in timings.items()},
    )

    funnel = no_trade.gate_summary if no_trade else {
        "universe_considered": eligibility.universe_considered,
        "passed_eligibility": len(eligibility.eligible_universe),
        "scored": len(scores.ranked_scores),
        "survived_defense": len(defense.survivors()),
        "triggered": len(entries.triggered()),
        "buys": len(buys),
    }

    # -- persist BEFORE returning. A run that is not recorded must not be
    # -- reported as evidence, so a ledger failure fails the run.
    duration_ms = (dt.datetime.now() - started).total_seconds() * 1000.0
    # The store IS the training set -- the model refits from it every run --
    # so the depth is part of what produced this ranking.
    try:
        train_sessions = len(store.price_sessions())
    except Exception:
        train_sessions = None
    Ledger(config.paths.ledger).append(
        row_from_output(output, context, funnel, duration_ms,
                        train_sessions=train_sessions)
    )

    log.info(
        "analysis complete",
        extra={"run_id": run_id, "as_of": resolved.isoformat(),
               "buys": len(buys), "watch": len(watch), "no_trade": no_trade is not None},
    )
    return AnalysisRun(output=output, context=context, timings_ms=timings, funnel=funnel)


# =============================================================================
def _stepper(progress):
    def step(i: int) -> None:
        if progress:
            progress(i, STAGE_LABELS[i])
    return step


def _clock():
    start = time.perf_counter()
    return lambda: (time.perf_counter() - start) * 1000.0


def _universe(store, config, as_of) -> UniverseSnapshot:
    u = config.params.universe
    if str(v(u.source)).lower() == "liquidity_pit":
        return _universe_liquidity_pit(store, config, as_of)
    return _universe_index_snapshot(store, config, as_of)


def _universe_liquidity_pit(store, config, as_of) -> UniverseSnapshot:
    """Trailing-turnover screen. No membership list, no survivorship risk."""
    from .data.universe import UniverseResolver

    u = config.params.universe
    sectors = store.read_sector_map()
    sector_map = (
        dict(zip(sectors["symbol"], sectors["sector"]))
        if sectors is not None and not sectors.empty and "sector" in sectors.columns
        else {}
    )
    try:
        return UniverseResolver(store, config).resolve_liquidity_pit(
            as_of=as_of,
            min_adtv_inr=float(v(u.pit_min_adtv_inr)),
            lookback_sessions=int(v(u.pit_adtv_lookback_sessions)),
            max_names=int(v(u.pit_max_names)),
            min_history_sessions=int(v(u.min_history_sessions)),
            min_price_inr=float(v(u.min_price_inr)),
            manual_exclusions=list(v(u.manual_exclusions) or []),
            sector_map=sector_map,
        )
    except IntegrityError as exc:
        raise PipelineBlocked([str(exc)], stage="stage0_data") from exc


def _universe_index_snapshot(store, config, as_of) -> UniverseSnapshot:
    index = str(config.params.universe.index_name.value)
    dates = store.universe_snapshot_dates(index)
    if not dates:
        raise PipelineBlocked(
            [f"no universe snapshot for {index}; run `prosignal data ingest`"],
            stage="stage0_data",
        )
    # Latest snapshot at or before the decision date; else the earliest available.
    usable = [d for d in dates if d <= as_of]
    chosen = usable[-1] if usable else dates[0]
    survivorship = not usable
    # The only membership list available for this date was recorded later, so it
    # holds today's constituents: names promoted for performing well are present
    # and names dropped for performing badly are absent. universe.pre_snapshot_policy
    # decides whether that is acceptable. It is for a live run, where today's list
    # IS the point-in-time list, and it is not for anything historical.
    if survivorship:
        policy = str(v(config.params.universe.pre_snapshot_policy)).lower()
        if policy == "halt":
            raise PipelineBlocked(
                [
                    f"no {index} membership snapshot on or before {as_of}; the "
                    f"earliest available is {dates[0]}. Running would use today's "
                    f"constituents for a past date, which is survivorship bias. "
                    f"Set universe.pre_snapshot_policy to 'flag' only for live runs."
                ],
                stage="stage0_data",
            )
    snap = store.read_universe_snapshot(index, chosen)
    sectors = dict(zip(snap["symbol"], snap["sector"])) if "sector" in snap.columns else {}
    names = dict(zip(snap["symbol"], snap["company_name"])) if "company_name" in snap.columns else {}
    return UniverseSnapshot(
        index_name=index, as_of=as_of, symbols=snap["symbol"].tolist(),
        sector_map=sectors, company_names=names,
        source=f"snapshot {chosen}", survivorship_risk=survivorship,
        note=(f"snapshot dated {chosen} is LATER than the decision date {as_of}; "
              f"membership is survivorship-biased" if survivorship else None),
    )


def _sessions_behind(last: Optional[dt.date], today: Optional[dt.date] = None) -> int:
    """Trading sessions between a feed's last row and TODAY.

    The staleness gate used to measure the store against a calendar built from
    the store, which is circular: age came back 0 for every feed on every run,
    stale_required() was permanently empty, and a store frozen for a month
    reported itself fresh. `analyse run` performs no ingest, so that is the
    normal way for the store to fall behind rather than an exotic one.

    Counted as weekdays, which overstates by roughly one session per NSE
    holiday in the window. Overstating is the safe direction: it flags early
    rather than late, and the caller allows a tolerance rather than this
    function guessing at a holiday calendar it does not have.
    """
    if last is None:
        return 0
    today = today or dt.date.today()
    if last >= today:
        return 0
    days = 0
    cursor = last
    while cursor < today:
        cursor += dt.timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


def _manifest_from_store(store, config, run_id, as_of, universe) -> RawDataManifest:
    """Describe what the store actually holds, for runs that skip a fresh ingest."""
    from .core.contracts import FeedRecord
    from .core.enums import FeedStatus, SourceName

    calendar = TradingCalendar(store.price_sessions())
    # Wall-clock staleness applies to a LIVE run only. A deliberate historical
    # analysis is legitimately behind today and must not be failed for it.
    live = as_of >= calendar.last if calendar.last else False
    feeds: Dict[str, FeedRecord] = {}
    # delivery_data is REQUIRED. deliv_pct carries the largest coefficient in the
    # fitted model (+0.0233 of 17 factors) and crosssec treats it as neutral-when-
    # missing, so an outage does not fail anything: every name scores as if its
    # delivered share were exactly average. Measured, that silently replaces 33%
    # of the top decile and costs 18% of IC while the run reports no flag at all.
    # A feed the model leans on that hardest cannot be optional.
    checks = [
        ("equity_ohlcv", store.prices.max_date(), True, 1),
        ("index_ohlcv", store.indices.max_date(), True, 1),
        ("india_vix", store.indices.max_date(), True, 1),
        ("delivery_data", store.delivery.max_date(), True, 2),
    ]
    for name, last, required, max_age in checks:
        if last is None:
            status, age = FeedStatus.MISSING, None
        else:
            age = (_sessions_behind(last) if live
                   else calendar.age_in_sessions(last, as_of))
            # No holiday allowance. Weekday counting overstates by about one
            # session per NSE holiday in the window, and that is the direction
            # to err: a false positive costs an ingest, a false negative issues
            # signals from stale prices. FeedRecord.is_stale compares this same
            # raw age, so an allowance applied here and not there would have the
            # status and the property disagree.
            status = FeedStatus.STALE if age > max_age else FeedStatus.OK
        feeds[name] = FeedRecord(
            feed=name,
            status=status,
            source=SourceName.NSE_ARCHIVES if last else None,
            last_timestamp=last, age_sessions=age, max_age_sessions=max_age,
            required=required,
            notes=([f"age is weekdays since {last} counted against today "
                    f"({dt.date.today()}), not sessions in the local store. An "
                    f"NSE holiday in the window reads as one extra session; "
                    f"running `prosignal data ingest` settles it either way."]
                   if live else []),
        )

    # These two were stamped OK with age 0 unconditionally, which meant
    # missing_required() and stale_required() could never see them however empty
    # the store was -- a required feed that reports itself healthy by
    # construction is not a check. Their real state is read from the store.
    master = store.read_table("equity_master")
    feeds["equity_master"] = FeedRecord(
        feed="equity_master",
        status=FeedStatus.OK if master is not None and not master.empty else FeedStatus.MISSING,
        source=SourceName.NSE_ARCHIVES if master is not None and not master.empty else None,
        last_timestamp=as_of, age_sessions=0, max_age_sessions=25, required=True,
    )
    # The fundamental block. Two different tables, and the distinction is the
    # whole point: "fundamentals" is the NSE Ind-AS filings feed, which carries
    # true filing dates; "statements" carries period end only, and the factor
    # layer derives availability from the SEBI LODR deadline instead. Stage 1
    # reads these to decide what it can honestly claim about filing-date
    # alignment, and previously found neither -- so it reported the block
    # absent on every run while the model scored five fundamental factors.
    for name, frame in (("fundamentals", store.read_fundamentals()),
                        ("statements", store.read_statements())):
        rows = 0 if frame is None or frame.empty else len(frame)
        feeds[name] = FeedRecord(
            feed=name,
            status=FeedStatus.OK if rows else FeedStatus.MISSING,
            source=SourceName.NSE_ARCHIVES if rows else None,
            last_timestamp=as_of, age_sessions=0, max_age_sessions=None,
            required=False, row_count=rows,
            symbols_covered=0 if not rows or "symbol" not in frame.columns
                            else int(frame["symbol"].nunique()),
        )

    # Membership is only required when the universe is built from a membership
    # list. Under universe.source = liquidity_pit nothing reads it, so demanding
    # it would halt runs over a feed the decision never touches.
    uses_membership = str(v(config.params.universe.source)).lower() != "liquidity_pit"
    snapshots = store.universe_snapshot_dates(str(v(config.params.universe.index_name)))
    feeds["index_membership"] = FeedRecord(
        feed="index_membership",
        status=FeedStatus.OK if snapshots else FeedStatus.MISSING,
        source=SourceName.NSE_ARCHIVES if snapshots else None,
        last_timestamp=snapshots[-1] if snapshots else None,
        age_sessions=calendar.age_in_sessions(snapshots[-1], as_of) if snapshots else None,
        max_age_sessions=25, required=uses_membership,
    )
    return RawDataManifest(
        run_id=run_id, as_of_date=as_of, generated_at=dt.datetime.now(),
        snapshot_id=f"store-{as_of.isoformat()}", feeds=feeds,
        universe_size_raw=len(universe.symbols),
        calendar_sessions_available=len(calendar.sessions),
        calendar_last_session=calendar.last,
        survivorship_risk=universe.survivorship_risk,
        survivorship_note=universe.note,
    )


def _frames(store, calendar, symbols, config, as_of) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    need = int(config.params.stage7_risk.targets.resistance_lookback_sessions.value) + 20
    window = calendar.trailing_window(as_of, need)
    start = window[0] if window else calendar.first
    prices = store.read_prices(symbols=symbols, start=start, end=as_of)
    if prices.empty:
        return {}
    prices = prices.copy()
    prices[DATE] = pd.to_datetime(prices[DATE]).dt.normalize()
    return {s: f.sort_values(DATE).reset_index(drop=True)
            for s, f in prices.groupby(SYMBOL, sort=False, observed=True)}


def _closes(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    cols = {
        s: pd.Series(f["close"].to_numpy(dtype="float64"),
                     index=pd.DatetimeIndex(f[DATE]))
        for s, f in frames.items()
    }
    return pd.DataFrame(cols).sort_index()


def _prefetch_prices(store, config, universe, as_of, calendar) -> None:
    """Warm the store's slice cache with the widest window the stages need."""
    p = config.params
    try:
        need = max(
            int(v(p.universe.min_history_sessions)),
            int(v(p.stage3_eligibility.liquidity.adtv_lookback_sessions)),
            int(v(p.stage7_risk.targets.resistance_lookback_sessions)),
        ) + 30
        window = calendar.trailing_window(as_of, need)
        start = window[0] if window else calendar.first
        rows = store.prefetch_prices(universe.symbols, start, as_of)
        log.info("price window prefetched", extra={"rows": rows, "from": str(start)})
    except Exception as exc:
        # A prefetch failure must never fail the run; the stages read directly.
        log.warning("price prefetch skipped", extra={"error": str(exc)})
        store.clear_price_cache()
