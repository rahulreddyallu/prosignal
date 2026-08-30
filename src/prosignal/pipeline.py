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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .config.loader import AppConfig
from .core.calendar import TradingCalendar
from .cadence import clock_from_config
from .core.clock import market_today
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

#: How far back the open-position review needs to see to tell a quiet name
#: from a delisted one. positions.DELISTING_SESSIONS is 30; the margin
#: covers the gap itself plus enough prior history to find the last print.
DELISTING_LOOKBACK_SESSIONS = 90

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
    #: What stage 4 said about HOW it ranked -- the scorer, the sealed-holdout
    #: numbers behind it, and any theme running more of today's spread than the
    #: weight it was given. Stage 4 has written these on every run since the v2
    #: deploy and `CoreScores.notes` was read by nothing: not rendered, not
    #: persisted, not in the ledger. A monitor that flags into a field nobody
    #: reads is not a monitor, so the notes are carried out of the run here.
    scoring_notes: List[str] = field(default_factory=list)


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

    # ---- the open book, read once, BEFORE anything can exclude a name ------
    # Stage 3 needs it as much as Stage 6 does. Its model-domain filter removes
    # names below their thesis-invalidation level from the universe -- correct
    # for an ENTRY, and catastrophic for a position already open: the name
    # reaches neither Stage 6's exit band nor Stage 7's exit hierarchy, falls
    # through to the orphan review, and is reported "hold, trading normally"
    # at the exact moment it has met its own first exit condition.
    #
    # Entry constraints do not govern open positions. That is Stage 8's
    # contract and Stage 6 already honours it; Stage 3 must too.
    ledger = Ledger(config.paths.ledger)
    previous = ledger.previous_run(before=resolved)
    open_book = list(previous.get("signals_generated") or []) if previous else []
    previous_slate = list(previous.get("slate_shown") or []) if previous else []

    # ---- Stage 3 ----------------------------------------------------------
    step(3)
    t = _clock()
    eligibility = stage3_eligibility.run(
        universe, store, calendar, quality, config, as_of=resolved,
        held=open_book,
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
    # THE ENTRY CLOCK. The engine runs every session; it BUYS on a cadence. See
    # `prosignal.cadence` for why the two are different and why the schedule is
    # counted in sessions from a fixed anchor rather than in calendar days.
    # Resolved here, once, and carried into the run's record so a reader can
    # tell "the book was not buying today" from "the market offered nothing".
    clock = clock_from_config(config, sessions, resolved)
    # `open_book` and `previous_slate` were read once, above Stage 3 -- this
    # file grows without bound and scanning it twice per run for two fields of
    # one record was waste.
    entries = stage6_entry.run(defended, frames, config, resolved,
                               ranks=ranks, held=open_book,
                               entries_open=clock.is_entry_date,
                               entries_closed_reason=clock.blocked_reason())
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
    # EARNINGS PROXIMITY for the names that could be carded. Computed here
    # because stage 8 has no store, and computed for the SCORED set rather than
    # the whole universe so it costs a lookup rather than a scan. It is a risk
    # disclosure and never a gate: it changes no score, no rank and no
    # admission.
    earnings_notes: Dict[str, str] = {}
    try:
        from .features import earnings as _earn
        _syms = [s.ticker for s in scores.ranked_scores]
        _cal = _earn.earnings_dates(store)
        _until = _earn.sessions_until_next(_cal, _syms, resolved, sessions)
        _since = _earn.days_since_last(_cal, _syms, resolved)
        for _s in _syms:
            _n = _earn.risk_note(_s, _until.get(_s), _since.get(_s))
            if _n:
                earnings_notes[_s] = _n
    except Exception as exc:
        log.warning("earnings proximity unavailable", extra={"error": str(exc)})

    buys, watch, no_trade, gate_counts = stage8_final_signal.run(
        regime=regime,
        eligibility=eligibility,
        scores=scores,
        defense=defense,
        entries=entries,
        plans=plans,
        closes=closes,
        config=config,
        company_names=dict(universe.company_names),
        earnings_notes=earnings_notes,
        # Stage 8 needs the same book Stage 6 got. Its sector, correlation and
        # book-size limits are ENTRY limits; without knowing what is already
        # held it applied them to open positions and evicted them, which is how
        # Stage 6's hysteresis was being undone one session after it worked.
        held=open_book,
    )
    timings[stage8_final_signal.STAGE_NAME] = t()

    # ---- open positions the run never reached -----------------------------
    # A held name that fails eligibility, fails a data-quality check or leaves
    # the universe never reaches Stage 8 at all. It simply stopped appearing,
    # the next run rebuilt the book without it, and the position left with no
    # recorded exit -- measured on the recorded ledger, that is how 23 of 54
    # held-name transitions on adjacent sessions ended.
    #
    # `positions.review_open_position` was written for exactly this, with the
    # rules and the tests to go with it, and nothing in the engine had ever
    # called it. It is called here.
    directives = _review_open_positions(
        open_book, buys, watch, store, universe, calendar, resolved,
        eligibility=eligibility,
    )

    # ---- the slate --------------------------------------------------------
    # The screen is decided HERE, by the run, and recorded with it. It used to
    # be recomputed by the presentation layer on every request from a payload
    # with no memory of the previous screen, which made a stable list
    # impossible and let the live view, the history page and the outcome record
    # each derive a different list from the same run.
    admission = config.params.stage6_entry.admission
    # The screen is the BOOK, so its size comes from the same parameter the
    # book does. It used to fall back to a hardcoded 5 in the presentation
    # layer while `entry_rank` was 6, which put five of six positions on the
    # screen and, because the slate is recorded, into the ledger with them.
    slate_entries, slate_departures = _build_slate(
        buys, watch, previous_slate,
        slots=int(v(admission.entry_rank)),
        exit_rank=int(v(admission.exit_rank)),
        as_of=resolved,
    )

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
        slate=slate_entries,
        slate_departures=slate_departures,
        position_directives=directives,
        # THREE REASONS THE BOOK MIGHT NOT BUY, and they are not the same. A
        # regime halt and a market halt are conditions; the entry cadence is a
        # schedule. Reporting the schedule through the same field is right --
        # the field means "new entries were refused and here is why" -- but the
        # cadence must be named as itself, because an operator seeing "no new
        # entries" three sessions running should be able to tell a halted
        # market from a book that simply is not due to buy until the 21st.
        new_entries_blocked=(
            clock.blocked_reason()
            if not clock.is_entry_date
            else (None if regime.allow_new_entries and not defense.market_halt
                  else (no_trade.reason if no_trade else None))
        ),
        entry_clock=_clock_record(clock, sessions, resolved),
        data_quality_flags=flags,
        manifest=manifest,
        stage_timings_ms={k: round(v, 1) for k, v in timings.items()},
    )

    # Stage 8's own counts, on every path. Rebuilding them here read
    # `entries.triggered()` for the trigger line -- the population BEFORE the
    # score gate -- so the displayed funnel could run backwards (triggered=1
    # above passed_score=8) on exactly the path that produces a trade. Stage 8
    # documents having fixed that; the fix never reached the screen because the
    # screen was reading a different dict.
    funnel = no_trade.gate_summary if no_trade else gate_counts

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

    # THE DRAWDOWN FLAG, on the same channel as the theme flags. It reads
    # CLOSED trades, so it lags an open book and is a floor on the drawdown
    # rather than an estimate -- `review_realised_drawdown` says so on the line
    # itself. It stays silent below twenty closed trades: a six-name book's
    # realised curve after four exits is noise, and "0%, inside the flag" would
    # reassure about a book that has not been tested yet.
    scoring_notes = list(getattr(scores, "notes", []) or [])
    try:
        from . import v3_monitor as _v3mon
        from . import outcomes as _out
        _p = Path(config.paths.ledger) / "outcomes.jsonl"
        if _p.exists():
            scoring_notes += _v3mon.review_realised_drawdown(
                _out.load_outcomes(_p))
    except Exception as exc:                        # never fail a run to report
        log.warning("drawdown check did not run", extra={"error": str(exc)})

    result = AnalysisRun(output=output, context=context, timings_ms=timings,
                         funnel=funnel, scoring_notes=scoring_notes)

    # THE SCREEN READS THIS, not the API's job queue.
    #
    # The interface builds Today from `GET /analysis`, which lists jobs. The
    # nightly cron runs this same function from the CLI, in a process the API
    # knows nothing about, so no job row is ever created and the screen asks
    # for a scan of a market that was scanned hours ago. `/analysis/{id}/view`
    # cannot help either -- it needs the result held in the job row, and the
    # ledger keeps a summary with no `factor_detail`, so Today would render
    # cards with an empty evidence panel.
    #
    # Written HERE so every path persists identically: cron, CLI and API job
    # all come through `run_analysis`. It also survives an API restart, which
    # the in-memory job result does not.
    #
    # After the ledger, and never in place of it. The ledger is the permanent
    # record and a failure to write it fails the run; this is a display cache
    # and `rundetail.save` swallows its own errors for that reason.
    from .rundetail import save as _save_detail
    _save_detail(result, config)

    log.info(
        "analysis complete",
        extra={"run_id": run_id, "as_of": resolved.isoformat(),
               "buys": len(buys), "watch": len(watch), "no_trade": no_trade is not None},
    )
    return result


# =============================================================================
def _clock_record(clock, sessions, as_of: dt.date) -> Dict[str, Any]:
    """The entry schedule as data, for the record and for the screen.

    `sessions_until_next` is counted on the exchange calendar rather than in
    calendar days, for the same reason the clock itself is: a count in days
    changes meaning across a holiday and cannot be reproduced.
    """
    until = None
    nxt = clock.next_entry_date
    if nxt is not None:
        try:
            here = [d for d in sessions if d > as_of and d <= nxt]
            until = len(here) or None
        except TypeError:
            until = None
    return {
        "cadence_sessions": int(clock.cadence_sessions),
        "is_entry_date": bool(clock.is_entry_date),
        "sessions_since_anchor": clock.sessions_since_anchor,
        "next_entry_date": nxt.isoformat() if nxt else None,
        "sessions_until_next": until,
        "anchor": clock.anchor.isoformat() if clock.anchor else None,
        # The clock FAILS OPEN when it cannot place the run date -- an anchor
        # still in the future is the ordinary case before an epoch starts. That
        # is the right behaviour and the wrong thing to hide: "entries open
        # because the schedule has not begun" and "entries open because today
        # is the twenty-first session" are different states, and a screen that
        # renders both as "buying session" says the schedule is running when it
        # is not.
        "resolved": clock.sessions_since_anchor is not None,
        "reason": clock.reason,
    }


def _build_slate(buys, watch, previous_slate, *, slots: int, exit_rank: int,
                 as_of: dt.date):
    """Decide the screen for this run, carrying the previous one where the band allows.

    Returns the ordered slate and the departures. The selection rule itself
    lives in `presentation.selection` so that there is exactly one of it; what
    happens here is the part only the run can do -- supplying the previous
    screen and stamping how long each name has held its slot.
    """
    from .core.contracts import SlateEntry
    from .presentation.selection import select_slate

    def _card(rec) -> Dict[str, object]:
        return {
            "ticker": rec.ticker,
            "model_rank": rec.model_rank,
            "percentile": rec.universe_percentile,
            "score": rec.composite_score,
        }

    held_tickers = [str(e.get("ticker")) for e in previous_slate if e.get("ticker")]
    # When a name is carried, the session it entered on carries with it. That is
    # what makes dwell readable straight off the record instead of needing a
    # walk back through the ledger to reconstruct it.
    since = {
        str(e.get("ticker")): e.get("shown_since")
        for e in previous_slate if e.get("ticker")
    }

    slate = select_slate(
        [_card(r) for r in buys],
        [_card(r) for r in watch],
        slots=slots,
        held_slate=held_tickers,
        exit_rank=exit_rank,
    )

    entries: List = []
    for pick in slate.picks:
        ticker = str(pick["ticker"])
        first = since.get(ticker) if pick.get("carried") else None
        if isinstance(first, str):
            try:
                first = dt.date.fromisoformat(first[:10])
            except ValueError:
                first = None
        entries.append(SlateEntry(
            ticker=ticker,
            position=int(pick["slate_position"]),
            status=str(pick["status"]),
            model_rank=pick.get("model_rank"),
            carried=bool(pick.get("carried")),
            shown_since=first or as_of,
            reason=str(pick.get("slate_reason") or ""),
        ))
    return entries, [d.to_dict() for d in slate.departures]


def _review_open_positions(open_book, buys, watch, store, universe, calendar,
                           as_of: dt.date, eligibility=None) -> List[Dict[str, object]]:
    """Decide what happens to held names the run never produced a card for.

    A name in `buys` is still held and a name in `watch` was seen and set aside
    by a rule that recorded its reason. Neither needs a directive. What needs
    one is a held name that appears in neither: the run did not evaluate it, so
    nothing decided anything about it, and the position would otherwise leave
    the book by omission.

    Entry-time gates do not govern open positions. Leaving the tradeable
    universe changes who must own a stock, not whether it can be sold, and
    exiting into a reconstitution pays the worst price available for a reason
    the thesis never priced. So the default here is to hold and flag; only a
    name that has stopped trading long enough to read as delisted is exited,
    at the last price that actually existed.
    """
    from .positions import PositionAction, review_open_position

    accounted = {r.ticker for r in buys} | {r.ticker for r in watch}
    orphans = [t for t in dict.fromkeys(open_book) if t not in accounted]
    if not orphans:
        return []

    in_universe = set(universe.symbols)
    frames: Dict[str, pd.DataFrame] = {}
    try:
        window = calendar.trailing_window(as_of, DELISTING_LOOKBACK_SESSIONS)
        start = window[0] if window else calendar.first
        prices = store.read_prices(symbols=orphans, start=start, end=as_of)
        if not prices.empty:
            prices = prices.copy()
            prices[DATE] = pd.to_datetime(prices[DATE]).dt.normalize()
            frames = {s: f.sort_values(DATE).reset_index(drop=True)
                      for s, f in prices.groupby(SYMBOL, sort=False, observed=True)}
    except Exception as exc:
        # No price history is itself informative -- review_open_position reads
        # an absent frame as a name that has not printed -- so this must not
        # fail the run. It must also not silently look like "trading normally",
        # which is why the failure is logged rather than passed over.
        log.warning("could not read prices for open-position review",
                    extra={"error": str(exc), "tickers": len(orphans)})

    out: List[Dict[str, object]] = []
    for ticker in orphans:
        # WHY the run produced no card, when the run knows. `in_universe` is
        # the RAW universe, so an eligibility rejection still reads as "in
        # universe" and the directive said "trading normally".
        why = None
        if eligibility is not None and ticker in eligibility.rejected:
            why = (eligibility.rejection_details.get(ticker)
                   or eligibility.rejected[ticker].value)
        directive = review_open_position(
            ticker, frames.get(ticker), as_of,
            in_universe=ticker in in_universe,
            sessions=calendar.sessions,
            excluded_because=why,
        )
        out.append(directive.to_dict())
        log.info("open position reviewed",
                 extra={"ticker": ticker, "event": directive.event.value,
                        "action": directive.action.value})
    exits = sum(1 for d in out if d["action"] == PositionAction.FORCE_EXIT.value)
    log.info("open-position review complete",
             extra={"reviewed": len(out), "forced_exits": exits})
    return out


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

    ``today`` is the MARKET's date, supplied by the caller from
    `runtime.timezone`. Defaulting to the host clock made the tolerance of one
    session depend on which timezone the box happened to be in: a UTC host at
    20:30 IST is already on the next calendar day, so every Monday run would
    read one session staler than it is.
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
    now = market_today(config)
    feeds: Dict[str, FeedRecord] = {}
    # delivery_data is REQUIRED. deliv_pct carries the largest coefficient in the
    # fitted model (+0.0233 of 17 factors) and crosssec treats it as neutral-when-
    # missing, so an outage does not fail anything: every name scores as if its
    # delivered share were exactly average. Measured, that silently replaces 33%
    # of the top decile and costs 18% of IC while the run reports no flag at all.
    # A feed the model leans on that hardest cannot be optional.
    # THE LIMITS COME FROM THE CONFIG, and they did not. `feeds:` declares a
    # `max_age_sessions` for every feed and this list hardcoded four of them, so
    # editing the config's staleness policy changed the `data status` report and
    # nothing about whether a run was allowed to proceed. The two agreed on the
    # shipped values -- 1/1/1/2 both places -- which is exactly why it went
    # unnoticed: the defect was invisible until someone tried to change it.
    #
    # `required` stays hardcoded. It is a claim about what the MODEL leans on
    # rather than an operational preference: delivery is required because
    # deliv_pct carries the largest coefficient in the fit and crosssec treats
    # it as neutral-when-missing, so an outage silently replaces 33% of the top
    # decile and costs 18% of IC while the run reports nothing. Letting that be
    # switched off in config would make a measurement into a preference.
    def _max_age(feed: str, fallback: int) -> int:
        policy = (config.params.feeds or {}).get(feed)
        value = getattr(policy, "max_age_sessions", None)
        return int(value) if value is not None else fallback

    checks = [
        ("equity_ohlcv", store.prices.max_date(), True, _max_age("equity_ohlcv", 1)),
        ("index_ohlcv", store.indices.max_date(), True, _max_age("index_ohlcv", 1)),
        ("india_vix", store.indices.max_date(), True, _max_age("india_vix", 1)),
        ("delivery_data", store.delivery.max_date(), True, _max_age("delivery_data", 2)),
    ]
    for name, last, required, max_age in checks:
        if last is None:
            status, age = FeedStatus.MISSING, None
        else:
            age = (_sessions_behind(last, now) if live
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
                    f"({now}), not sessions in the local store. An "
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
