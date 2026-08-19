"""Command-line interface.

    prosignal version
    prosignal config show [--unvalidated-only] [--grep TEXT]
    prosignal config validate
    prosignal config templates [--overwrite]
    prosignal data ingest [--sessions N] [--date YYYY-MM-DD] [--offline]
                          [--refetch] [--no-secondary] [--full]
    prosignal data status
    prosignal data check [--date YYYY-MM-DD]
    prosignal data purge-cache

Every command exits non-zero on failure with a specific reason.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import List, Optional

from .config.loader import AppConfig, load_config
from .core.errors import DataError, ProSignalError
from .core.logging import get_logger, setup_logging
from .version import ENGINE_NAME, ENGINE_VERSION, SCHEMA_VERSION

log = get_logger(__name__)

try:
    from rich.console import Console
    from rich.table import Table

    _console: Optional["Console"] = Console()
except ImportError:  # pragma: no cover - rich is a listed dependency
    _console = None


# =============================================================================
# output helpers
# =============================================================================


def _print(msg: str = "") -> None:
    if _console is not None:
        _console.print(msg)
    else:  # pragma: no cover
        print(msg)


def _rule(title: str) -> None:
    if _console is not None:
        _console.rule(f"[bold]{title}")
    else:  # pragma: no cover
        print(f"\n== {title} ==")


def _table(title: str, columns: List[str], rows: List[List[str]]) -> None:
    if _console is None:  # pragma: no cover
        print(title)
        print(" | ".join(columns))
        for r in rows:
            print(" | ".join(str(c) for c in r))
        return
    table = Table(title=title, header_style="bold", show_lines=False)
    for col in columns:
        table.add_column(col, overflow="fold")
    for row in rows:
        table.add_row(*[str(c) for c in row])
    _console.print(table)


_STATUS_STYLE = {
    "UNVALIDATED": "yellow",
    "VALIDATED": "green",
    "STATUTORY": "cyan",
    "STRUCTURAL": "blue",
    "OPERATIONAL": "magenta",
    "OK": "green",
    "STALE": "yellow",
    "MISSING": "red",
    "PARTIAL": "yellow",
    "DEGRADED": "yellow",
}


def _tag(text: str) -> str:
    style = _STATUS_STYLE.get(text)
    return f"[{style}]{text}[/{style}]" if style and _console else text


# =============================================================================
# config commands
# =============================================================================


def cmd_config_show(cfg: AppConfig, args: argparse.Namespace) -> int:
    report = cfg.transparency_report()
    _rule("Configuration")
    _print(f"file          : {report['source_file']}")
    _print(f"config version: [bold]{report['config_version']}[/bold]" if _console else report["config_version"])
    _print(f"parameters    : {report['total_parameters']}")
    counts = ", ".join(
        f"{_tag(k)}={v}" for k, v in sorted(report["counts_by_status"].items())
    )
    _print(f"by status     : {counts}")
    _print()

    params = report["parameters"]
    if args.unvalidated_only:
        params = [p for p in params if p["status"] == "UNVALIDATED"]
    if args.grep:
        needle = args.grep.lower()
        params = [
            p
            for p in params
            if needle in p["path"].lower() or needle in str(p.get("note", "")).lower()
        ]

    rows = []
    for p in params:
        rng = (
            f"[{p['search_range'][0]}, {p['search_range'][1]}]"
            if p.get("search_range")
            else "-"
        )
        rows.append([p["path"], repr(p["value"]), _tag(p["status"]), rng])
    _table(f"Parameters ({len(rows)})", ["path", "value", "status", "search range"], rows)

    _print()
    _print(f"[dim]{report['honesty_note']}[/dim]" if _console else report["honesty_note"])
    return 0


def cmd_config_validate(cfg: AppConfig, args: argparse.Namespace) -> int:
    # Reaching this point means load_config() already validated everything.
    _rule("Configuration valid")
    _print(f"config version : {cfg.version}")
    _print(f"project root   : {cfg.paths.root}")
    _print(f"parameters     : {len(cfg.params.iter_tunables())}")
    _print(f"unvalidated    : {cfg.params.unvalidated_count()}")
    _print(f"universe       : {cfg.params.universe.index_name.value}")
    _print(f"capital        : Rs {cfg.params.capital.total_capital_inr.value:,.0f}")
    _print(f"position value : Rs {cfg.params.capital.position_value_inr():,.0f}")
    _print()
    _print(
        "[yellow]Reminder:[/yellow] an UNVALIDATED parameter is a hypothesis. "
        "Nothing here has been through CPCV on point-in-time India data yet."
        if _console
        else "Reminder: UNVALIDATED parameters are hypotheses, not results."
    )
    return 0


_TIER_STYLE = {
    "A_SEARCH": "red",
    "B_SENSITIVITY": "yellow",
    "C_FIXED": "blue",
    "D_OPERATIONAL": "magenta",
}

_TIER_MEANING = {
    "A_SEARCH": "searched in CPCV; every value tried is charged to the DSR trial count",
    "B_SENSITIVITY": "perturbed to prove robustness; NEVER selected on",
    "C_FIXED": "set from evidence or convention; never searched",
    "D_OPERATIONAL": "your business constraint, not a research parameter",
}


def cmd_config_tiers(cfg: AppConfig, args: argparse.Namespace) -> int:
    report = cfg.params.search_space_report()
    _rule("Optimisation tiers and search budget")

    counts = report["tier_counts"]
    rows = []
    for tier in ("A_SEARCH", "B_SENSITIVITY", "C_FIXED", "D_OPERATIONAL"):
        label = (
            f"[{_TIER_STYLE[tier]}]{tier}[/{_TIER_STYLE[tier]}]" if _console else tier
        )
        rows.append([label, str(counts.get(tier, 0)), _TIER_MEANING[tier]])
    _table("Classification", ["tier", "count", "meaning"], rows)

    _print()
    _table(
        "Tier A -- the only parameters allowed into the search grid",
        ["path", "value", "grid points", "search range"],
        [
            [
                e["path"],
                repr(e["value"]),
                str(e["grid_points"]),
                str(e["search_range"]),
            ]
            for e in report["tier_a_parameters"]
        ],
    )

    grid = report["grid_configurations"]
    paths = report["cpcv_paths"]
    _print()
    _print(f"grid configurations   : {grid:,} / budget {report['max_grid_configurations']:,}")
    _print(f"CPCV paths per config : {paths:,}")
    _print(f"total model fits      : {grid * paths:,}")
    _print(f"trials already logged : {report['cumulative_trials_logged']:,}")
    _print(
        f"DSR trial count if swept: [bold]{report['effective_trials_if_swept']:,}[/bold]"
        if _console
        else f"DSR trial count if swept: {report['effective_trials_if_swept']:,}"
    )

    naive = report["naive_all_unvalidated_3pt_sweep"]
    _print()
    _print(
        f"For contrast, a 3-point sweep of every UNVALIDATED parameter would be "
        f"[red]{naive:.2e}[/red] configurations -- not expensive, arithmetically "
        f"impossible, with a Probability of Backtest Overfitting of essentially 1. "
        f"Keeping the real number near {grid:,} is the entire point of the tier "
        f"system."
        if _console
        else f"A 3-point sweep of every UNVALIDATED parameter would be {naive:.2e} configurations."
    )

    if not report["within_budget"]:
        _print()
        _print("[red]OVER BUDGET[/red]" if _console else "OVER BUDGET")
        return 2
    return 0


def cmd_config_templates(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .data.providers.csv_import import CsvImportProvider

    provider = CsvImportProvider(
        cfg=cfg.params.providers.csv_import, project_root=cfg.paths.root
    )
    written = provider.write_templates(overwrite=args.overwrite)
    _rule("Reference CSV templates")
    if not written:
        _print("All template files already exist. Use --overwrite to reset them.")
    for path in written:
        _print(f"  created {path}")
    _print()
    _print(
        "These are the feeds no free India source supplies reliably. Until a "
        "file has rows, the dependent check reports NOT_TESTABLE -- it does not "
        "pass."
    )
    return 0


# =============================================================================
# data commands
# =============================================================================


def cmd_data_ingest(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .data.ingest import DataIngestor, IngestOptions

    requested: Optional[dt.date] = (
        dt.date.fromisoformat(args.date) if args.date else None
    )
    sessions = args.sessions
    if args.full:
        sessions = int(cfg.params.universe.min_history_sessions.value) + 30

    opts = IngestOptions(
        history_sessions=sessions,
        offline=args.offline,
        include_secondary_prices=not args.no_secondary,
        refetch_stored_sessions=args.refetch,
        force_reference_refresh=args.refetch,
    )

    _rule("Stage 0 -- data ingestion")
    if sessions and sessions > 60 and not args.offline:
        _print(
            f"[dim]Backfilling ~{sessions} sessions. First run pulls each session "
            f"from NSE (a few minutes); later runs come from the local HTTP "
            f"cache in seconds.[/dim]"
            if _console
            else f"Backfilling ~{sessions} sessions; first run is slow, later runs are cached."
        )

    with DataIngestor(cfg) as ingestor:
        result = ingestor.run(requested_date=requested, options=opts)

    m = result.manifest
    _print()
    _print(f"decision date : [bold]{m.as_of_date}[/bold]" if _console else str(m.as_of_date))
    _print(f"universe      : {m.universe_size_raw} names ({cfg.params.universe.index_name.value})")
    _print(f"calendar      : {m.calendar_sessions_available} sessions, last {m.calendar_last_session}")
    _print(f"sessions new  : {result.sessions_fetched}")
    _print(f"http          : {result.http_stats}")

    rows = []
    for name, rec in sorted(m.feeds.items()):
        rows.append(
            [
                name,
                _tag(rec.status.value),
                rec.source.value if rec.source else "-",
                str(rec.last_timestamp or "-"),
                "-" if rec.age_sessions is None else str(rec.age_sessions),
                "yes" if rec.required else "",
                f"{rec.row_count:,}",
            ]
        )
    _table(
        "Feed manifest",
        ["feed", "status", "source", "last", "age", "required", "rows"],
        rows,
    )

    notes = [(n, note) for n, rec in sorted(m.feeds.items()) for note in rec.notes]
    if notes:
        _rule("Notes")
        for feed, note in notes:
            _print(f"  [dim]{feed}[/dim]: {note}" if _console else f"  {feed}: {note}")

    if m.survivorship_risk:
        _print()
        _print(
            f"[red]SURVIVORSHIP RISK[/red]: {m.survivorship_note}"
            if _console
            else f"SURVIVORSHIP RISK: {m.survivorship_note}"
        )

    missing_required = m.missing_required()
    stale_required = m.stale_required()
    if missing_required or stale_required:
        _print()
        _print(
            "[yellow]Stage 1 will halt this run:[/yellow] "
            f"missing={missing_required} stale={stale_required}"
            if _console
            else f"Stage 1 will halt: missing={missing_required} stale={stale_required}"
        )
        return 2
    return 0


def cmd_data_status(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .data.store import DataStore

    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    summary = store.summary()
    _rule("Data store")
    _print(f"location: {cfg.paths.curated}")
    _print()

    rows = []
    for name in ("prices", "indices", "delivery"):
        blob = summary[name]
        rows.append(
            [
                name,
                str(blob.get("sessions", 0)),
                str(blob.get("first") or "-"),
                str(blob.get("last") or "-"),
                f"{blob.get('rows', 0):,}" if "rows" in blob else "-",
                str(blob.get("symbols", blob.get("names", "-"))),
            ]
        )
    _table("Time series", ["table", "sessions", "first", "last", "rows", "symbols"], rows)

    ref_rows = [
        ["equity_master", f"{summary['equity_master_rows']:,}"],
        ["corporate_actions", f"{summary['corporate_actions_rows']:,}"],
        ["earnings_calendar", f"{summary['earnings_rows']:,}"],
        ["pledging", f"{summary['pledging_rows']:,}"],
        ["fundamentals", f"{summary['fundamentals_rows']:,}"],
    ]
    _table("Reference tables", ["table", "rows"], ref_rows)

    index_name = cfg.params.universe.index_name.value
    snaps = store.universe_snapshot_dates(index_name)
    _print(f"universe snapshots for {index_name}: {len(snaps)}")
    if snaps:
        _print(f"  earliest {snaps[0]}   latest {snaps[-1]}")
        _print(
            "[dim]Point-in-time membership is only trustworthy from the earliest "
            "snapshot onwards. Backtests before that date are "
            "survivorship-biased.[/dim]"
            if _console
            else "  Point-in-time membership is trustworthy only from the earliest snapshot onwards."
        )
    return 0


def cmd_data_check(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .data.corporate_actions import detect_unexplained_jumps
    from .data.store import DataStore

    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    s1 = cfg.params.stage1_data_quality

    _rule("Data integrity checks")
    problems = 0

    try:
        store.validate_no_duplicates()
        _print("duplicate (symbol, date) rows : [green]none[/green]" if _console else "duplicates: none")
    except ProSignalError as exc:
        problems += 1
        _print(f"[red]{exc.message}[/red]" if _console else exc.message)

    prices = store.read_prices()
    if prices.empty:
        _print("price store is empty -- run `prosignal data ingest` first")
        return 1

    actions = store.read_corporate_actions()
    jumps = detect_unexplained_jumps(
        prices,
        actions,
        min_ratio_gap=float(s1.unexplained_split_min_ratio_gap.value),
        tolerance=float(s1.unexplained_split_ratio_tolerance.value),
    )
    if jumps.empty:
        _print(
            "unexplained split-like jumps    : [green]none[/green]"
            if _console
            else "unexplained jumps: none"
        )
    else:
        problems += 1
        rows = [
            [
                r["symbol"],
                str(r["date"].date()),
                f"{r['ratio']:.4f}",
                f"{r['nearest_clean_factor']:.4f}",
                f"{r['prev_close']:.2f}",
                f"{r['close']:.2f}",
            ]
            for _, r in jumps.head(40).iterrows()
        ]
        _table(
            f"Unexplained split-like jumps ({len(jumps)})",
            ["symbol", "date", "ratio", "nearest clean", "prev close", "close"],
            rows,
        )
        _print(
            "These look like unadjusted corporate actions. Stage 1 hard-rejects "
            "the affected names -- an unadjusted 5:1 split reads as a -80% "
            "single-session return and would poison a 12-1 momentum score for a "
            "year."
        )

    sessions = store.known_sessions()
    _print(f"trading sessions known          : {len(sessions)}")
    if sessions:
        _print(f"  {sessions[0]} .. {sessions[-1]}")
    return 1 if problems else 0


def _dir_mb(path) -> float:
    from pathlib import Path

    p = Path(path)
    if not p.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _storage_rows(cfg: AppConfig):
    return [
        ("raw cache (data/cache)", _dir_mb(cfg.paths.cache), cfg.params.storage.raw_cache.max_mb),
        ("curated (data/curated)", _dir_mb(cfg.paths.curated), None),
        ("snapshots", _dir_mb(cfg.paths.snapshots), None),
        ("ledger", _dir_mb(cfg.paths.ledger), None),
        ("logs", _dir_mb(cfg.paths.logs), None),
    ]



def cmd_data_fundamentals(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Ingest point-in-time fundamentals from NSE quarterly filings."""
    from .data.providers.http import HttpClient, NseJsonSession
    from .data.providers.nse_fundamentals import NseFundamentalsProvider
    from .data.store import DataStore

    p = cfg.params.providers
    client = HttpClient(
        cache_dir=cfg.paths.cache, user_agent=p.http.user_agent,
        min_interval_seconds=p.http.min_interval_seconds,
    )
    session = NseJsonSession(
        client=client, base=p.nse_json_api.base, warmup_path=p.nse_json_api.warmup_path
    )
    provider = NseFundamentalsProvider(
        session=session, client=client, max_quarters=int(getattr(args, "quarters", 8))
    )
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)

    index = str(cfg.params.universe.index_name.value)
    dates = store.universe_snapshot_dates(index)
    if not dates:
        raise DataError(f"no universe snapshot for {index}; run `prosignal data ingest` first")
    symbols = store.read_universe_snapshot(index, dates[-1])["symbol"].tolist()

    _rule("Point-in-time fundamentals")
    _print(f"  source : NSE quarterly results (Ind-AS XBRL), gated on filing_date")
    _print(f"  symbols: {len(symbols)}")
    _print()

    frame = provider.fetch_universe(
        symbols,
        progress=lambda i, n, sym: _print(f"  [{i}/{n}] {sym}") if i % 25 == 0 else None,
    )
    if frame.empty:
        _print("[red]No fundamentals retrieved.[/red]" if _console else "No fundamentals retrieved.")
        _print(f"  last error: {provider.last_error}")
        return 3

    store.write_fundamentals(frame)
    covered = frame["symbol"].nunique()
    _print()
    _table("Result", ["metric", "value"], [
        ["filings stored", f"{len(frame):,}"],
        ["symbols covered", f"{covered} of {len(symbols)} ({covered/len(symbols):.0%})"],
        ["quarters per symbol", f"{len(frame)/max(covered,1):.1f}"],
    ])
    _print()
    _print(
        "[dim]Banks and financials file a different Ind-AS schema, so their line "
        "items are absent. Those names score neutrally on value/quality rather "
        "than being excluded.[/dim]" if _console
        else "Banks file a different schema; their line items are absent."
    )
    return 0


def cmd_data_budget(cfg: AppConfig, args: argparse.Namespace) -> int:
    import shutil as _shutil

    storage = cfg.params.storage
    _rule("Storage budget")

    rows = []
    for name, used, cap in _storage_rows(cfg):
        rows.append([name, f"{used:,.1f}", f"{cap:,.0f}" if cap else "-"])
    total = _dir_mb(cfg.paths.data)
    rows.append(["TOTAL data/", f"{total:,.1f}", f"{storage.max_total_mb:,.0f}"])
    _table("Usage (MB)", ["area", "used", "cap"], rows)

    free_mb = _shutil.disk_usage(str(cfg.paths.data)).free / 1e6
    _print(f"free on volume : {free_mb:,.0f} MB")
    _print(f"warn below     : {storage.warn_free_disk_mb:,.0f} MB")
    _print(f"halt below     : {storage.halt_free_disk_mb:,.0f} MB")

    over = total > storage.max_total_mb
    low = free_mb < storage.halt_free_disk_mb
    if over or low:
        _print()
        _print(
            "[red]over budget[/red]" if over else "[red]free disk below halt floor[/red]"
            if _console
            else "OVER BUDGET"
        )
        _print("Run `prosignal data gc` to reclaim the raw cache.")
        return 2

    # Projection, using the measured per-session curated cost.
    from .data.store import DataStore

    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    known = len(store.known_sessions())
    curated_mb = _dir_mb(cfg.paths.curated)
    if known:
        per_session = curated_mb / known
        _print()
        _print(f"measured curated cost: {per_session:.3f} MB/session over {known} sessions")
        proj = [
            ["live signals", int(cfg.params.universe.min_history_sessions.value) + 30],
            ["+ 18mo sacred holdout", int(cfg.params.universe.min_history_sessions.value) + 30 + int(cfg.params.validation.holdout.reserve_most_recent_sessions.value)],
            ["CPCV 5 years", 1250],
            ["CPCV 10 years", 2500],
        ]
        _table(
            "Projected curated size",
            ["purpose", "sessions", "curated MB"],
            [[label, str(n), f"{n * per_session:,.0f}"] for label, n in proj],
        )
    return 0


def cmd_data_gc(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .data.providers.http import HttpClient
    from .data.providers.nse_archives import NseArchivesProvider

    storage = cfg.params.storage
    before = _dir_mb(cfg.paths.data)

    client = HttpClient(
        cache_dir=cfg.paths.cache,
        user_agent=cfg.params.providers.http.user_agent,
        max_payload_bytes_to_cache=int(storage.raw_cache.max_payload_mb_to_cache * 1e6),
        max_cache_bytes=int(storage.raw_cache.max_mb * 1e6),
    )
    provider = NseArchivesProvider(
        client=client,
        cfg=cfg.params.providers.nse_archives,
        ttl_historical_s=0,
        ttl_current_s=0,
        never_cache_feeds=storage.raw_cache.never_cache_feeds,
    )

    _rule("Storage garbage collection")

    policy = client.purge_violating_policy(provider.never_cache_url_markers())
    _print(
        f"policy sweep : removed {policy['removed']:,} entries, "
        f"freed {policy['freed_bytes'] / 1e6:,.1f} MB"
    )
    _print(
        "[dim]  (entries the current policy would never have written: "
        "oversized payloads and never-cache feeds)[/dim]"
        if _console
        else "  (oversized payloads and never-cache feeds)"
    )

    lru = client.evict_lru()
    _print(
        f"LRU eviction : removed {lru['evicted']:,} entries, "
        f"freed {lru['freed_bytes'] / 1e6:,.1f} MB "
        f"(cap {storage.raw_cache.max_mb:,.0f} MB)"
    )

    after = _dir_mb(cfg.paths.data)
    _print()
    _print(f"data/ {before:,.1f} MB -> {after:,.1f} MB  (reclaimed {before - after:,.1f} MB)")
    _print(
        "[dim]Nothing durable was removed: the curated parquet store is the "
        "record, the cache only ever saves a re-download.[/dim]"
        if _console
        else "Nothing durable removed; cache only saves a re-download."
    )
    return 0


def _load_market(cfg: AppConfig):
    """Store, calendar and current universe -- the inputs every analysis needs."""
    from .core.calendar import TradingCalendar
    from .data.store import DataStore

    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError(
            "the local store has no price sessions. Run `prosignal data ingest --full` first."
        )
    calendar = TradingCalendar(sessions)

    index_name = str(cfg.params.stage2_regime.benchmark_index.value)
    snapshot_dates = store.universe_snapshot_dates(index_name)
    symbols: List[str] = []
    if snapshot_dates:
        snapshot = store.read_universe_snapshot(index_name, snapshot_dates[-1])
        if snapshot is not None and not snapshot.empty:
            symbols = snapshot["symbol"].tolist()
    return store, calendar, symbols


def _resolve_as_of(calendar, requested: Optional[str]):
    """Resolve a requested date back to a real session."""
    import datetime as _dt

    if not requested:
        return calendar.last
    try:
        wanted = _dt.date.fromisoformat(requested)
    except ValueError as exc:
        raise DataError(f"--date must be YYYY-MM-DD; got {requested!r}") from exc

    resolved = calendar.last_session_on_or_before(wanted)
    if resolved is None:
        raise DataError(
            f"no trading session on or before {wanted} in the local store "
            f"(earliest is {calendar.first})."
        )
    return resolved



def cmd_analyse_run(cfg: AppConfig, args: argparse.Namespace) -> int:
    """RUN MARKET ANALYSIS -- the full eight-stage decision pipeline."""
    from .pipeline import PipelineBlocked, run_analysis
    from .stages.stage8_final_signal import PROBABILITY_UNAVAILABLE

    _rule("RUN MARKET ANALYSIS")
    try:
        run = run_analysis(
            cfg,
            as_of=_resolve_arg_date(getattr(args, "date", None)),
            progress=lambda i, label: _print(f"  [{i+1}/9] {label}"),
        )
    except PipelineBlocked as blocked:
        _print()
        _print("[red]ANALYSIS BLOCKED -- INSUFFICIENT DATA[/red]" if _console
               else "ANALYSIS BLOCKED -- INSUFFICIENT DATA")
        _print(f"stage: {blocked.stage}")
        for r in blocked.reasons:
            _print(f"  - {r}")
        _print()
        _print("No signal is produced. This is deliberately NOT reported as NO TRADE: "
               "the engine refuses to form a view, which is different from having "
               "looked and found nothing.")
        return 3

    o = run.output
    r = o.regime_state

    _print()
    _rule(f"Market regime -- {o.as_of_date}")
    _print(f"  {r.regime_bucket}  |  trend {r.trend_regime.value}  |  "
           f"volatility {r.vol_tercile.value}/{r.vol_context.value}  |  "
           f"breadth {r.breadth_state.value}"
           + (f" ({r.breadth_pct_above_ma:.0f}%)" if r.breadth_pct_above_ma is not None else ""))
    _print(f"  new entries allowed: {'YES' if r.allow_new_entries else 'NO'}   "
           f"compatibility: {r.compatibility().value}")

    _rule("Funnel")
    _table("Candidates surviving each gate", ["gate", "count"],
           [[k.replace('_', ' '), f"{v:,}"] for k, v in run.funnel.items()])

    decision = f"{len(o.recommendations)} BUY / {len(o.watchlist)} WATCH"
    _print()
    _rule("Decision")
    if o.no_trade:
        _print("[bold red]NO TRADE[/bold red]" if _console else "NO TRADE")
        _print(f"  {o.no_trade.reason}")
        if o.no_trade.closest_candidates:
            _table("Closest candidates and the gate each failed",
                   ["rank", "ticker", "score", "gate failed", "detail"],
                   [[str(c.rank), c.ticker, f"{c.composite_score:.3f}",
                     c.gate_failed, (c.detail or "")[:70]]
                    for c in o.no_trade.closest_candidates])
    else:
        _print(f"[bold green]{decision}[/bold green]" if _console else decision)

    for rec in o.recommendations + o.watchlist[: int(getattr(args, "watch", 3) or 3)]:
        _print()
        _rule(f"{rec.decision.value} -- {rec.ticker}"
              + (f" ({rec.company_name})" if rec.company_name else ""))
        rows = [
            ["last close", _money(rec.last_close)],
            ["entry zone", f"{_money(rec.entry_zone[0])} - {_money(rec.entry_zone[1])}"
             if rec.entry_zone else "no trigger active"],
            ["initial stop", _money(rec.initial_stop)],
            ["invalidation", _money(rec.invalidation_level)],
            ["target 1", _money(rec.target_1)],
            ["target 2", _money(rec.target_2)],
            ["signal strength", rec.signal_strength_band.value],
            ["composite (rank)", f"{rec.composite_score:.3f}  #{rec.rank}, "
                                 f"{rec.universe_percentile:.0f}th pct"],
            ["regime fit", rec.regime_compatibility.value],
            ["holding period", rec.expected_holding_period],
            ["risk category", rec.position_risk_category.value if rec.position_risk_category else "-"],
        ]
        _table("Trade", ["field", "value"], rows)

        if rec.cost_note:
            _print(f"  cost: {rec.cost_note}")

        _print()
        _print("[bold]WHY THIS SIGNAL EXISTS[/bold]" if _console else "WHY THIS SIGNAL EXISTS")
        for line in rec.why_this_signal_exists:
            _print(f"  + {line}")

        _print()
        _print("[bold yellow]WHY THIS TRADE MAY BE WRONG[/bold yellow]" if _console
               else "WHY THIS TRADE MAY BE WRONG")
        for line in rec.false_signal_flagged or []:
            _print(f"  - {line}")
        for line in rec.market_regime:
            _print(f"  - context: {line}")

        if rec.false_signal_not_testable:
            _print()
            _print("[dim]NOT TESTABLE WITH CURRENT DATA[/dim]" if _console
                   else "NOT TESTABLE WITH CURRENT DATA")
            for line in rec.false_signal_not_testable:
                _print(f"  ? {line}")

        if rec.sell_conditions:
            _print()
            _print("[bold]EXIT HIERARCHY[/bold]" if _console else "EXIT HIERARCHY")
            for line in rec.sell_conditions[:4]:
                _print(f"  {line}")

    _print()
    _rule("Honesty")
    _print(f"  {PROBABILITY_UNAVAILABLE}")
    _print(f"  {rec_warning()}")
    _print()
    _print(f"run_id {o.run_id} | config {o.config_version} | engine {o.engine_version}")
    _print(f"stage timings (ms): {o.stage_timings_ms}")
    for f in o.data_quality_flags:
        _print(f"  data-quality: {f}")
    return 0


def rec_warning() -> str:
    from .core.contracts import Recommendation
    return Recommendation.model_fields["unvalidated_parameter_warning"].default


def _money(v) -> str:
    return "-" if v is None else f"Rs {v:,.2f}"


def _resolve_arg_date(raw):
    import datetime as _dt
    if not raw:
        return None
    try:
        return _dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise DataError(f"--date must be YYYY-MM-DD; got {raw!r}") from exc


def cmd_analyse_regime(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Stage 2 in isolation. Also the query behind the webapp's regime strip."""
    from .stages import stage2_regime

    store, calendar, symbols = _load_market(cfg)
    as_of = _resolve_as_of(calendar, getattr(args, "date", None))

    history = int(getattr(args, "history", 1) or 1)
    if history > 1:
        return _regime_history(cfg, store, calendar, symbols, as_of, history)

    state = stage2_regime.run(store, calendar, symbols, cfg, as_of=as_of)

    _rule(f"Market regime -- {state.as_of_date}")
    if as_of != _requested_date(args, as_of):
        _print(f"[dim]requested {args.date}; resolved back to session {as_of}[/dim]"
               if _console else f"requested {args.date}; resolved to session {as_of}")

    rows = [
        ["Regime bucket", state.regime_bucket],
        ["Trend", f"{state.trend_regime.value}"],
        ["  slope (annualised)", _opt_pct(state.trend_slope_annualised)],
        ["  vs fast MA", _opt_num(state.index_vs_fast_ma_pct, "%")],
        ["  vs slow MA", _opt_num(state.index_vs_slow_ma_pct, "%")],
        ["Volatility", f"{state.vol_tercile.value} / {state.vol_context.value}"],
        ["  India VIX", _opt_num(state.vix_level)],
        ["  percentile of own year", _opt_num(state.vix_percentile, "%")],
        ["  change", _opt_num(state.vix_change_pct, "%")],
        ["  signal confidence", f"{state.vol_signal_confidence:.2f}"],
        ["Breadth", state.breadth_state.value],
        ["  above long-term MA", _opt_num(state.breadth_pct_above_ma, "%")],
        ["  sample", str(state.breadth_sample_size)],
        ["  divergence", "yes" if state.breadth_divergence_flag else "no"],
        ["Transition", "YES" if state.transition_flag else "no"],
    ]
    for component in state.transition_components:
        rows.append(["  ", component])
    _table("Regime", ["read", "value"], rows)

    _table(
        "Factor multipliers",
        ["factor", "multiplier"],
        [
            ["momentum", f"{state.momentum_multiplier:.3f}"],
            ["quality", f"{state.quality_multiplier:.3f}"],
            ["sector RS", f"{state.sector_rs_multiplier:.3f}"],
            ["dampener applied", f"{state.dampener_applied:.2f}"],
        ],
    )

    _print()
    _print(f"New entries allowed : {'YES' if state.allow_new_entries else 'NO'}")
    _print(f"Regime compatibility: {state.compatibility().value}")
    if state.block_reason:
        _print(f"[red]{state.block_reason}[/red]" if _console else state.block_reason)

    if state.notes:
        _rule("Notes")
        for note in state.notes:
            _print(f"  {note}")

    _print()
    _print(
        "[dim]Every multiplier above is an UNVALIDATED hypothesis until CPCV "
        "promotes it.[/dim]"
        if _console
        else "Multipliers are UNVALIDATED until CPCV promotes them."
    )
    return 0


def _regime_history(cfg, store, calendar, symbols, as_of, history: int) -> int:
    """Print the regime for the last N sessions.

    The sanity check that matters for a regime engine: buckets should persist
    for weeks at a time. A bucket that changes every session is not detecting
    regime, it is tracking noise -- and that is a finding to log, not a number
    to tune until the output looks tidy.
    """
    from .stages import stage2_regime

    window = calendar.trailing_window(as_of, history)
    rows = []
    buckets = []
    for day in window:
        state = stage2_regime.run(store, calendar, symbols, cfg, as_of=day)
        buckets.append(state.regime_bucket)
        rows.append(
            [
                day.isoformat(),
                state.regime_bucket,
                state.trend_regime.value,
                f"{state.vol_tercile.value}/{state.vol_context.value}",
                _opt_num(state.breadth_pct_above_ma, "%"),
                "T" if state.transition_flag else "",
                f"{state.momentum_multiplier:.2f}",
                "" if state.allow_new_entries else "BLOCKED",
            ]
        )

    _rule(f"Regime over the last {len(window)} sessions")
    _table(
        "History",
        ["date", "bucket", "trend", "volatility", "breadth", "trn", "mom", "entries"],
        rows,
    )

    changes = sum(1 for a, b in zip(buckets, buckets[1:]) if a != b)
    _print()
    _print(f"bucket changes over {len(buckets)} sessions: {changes}")
    if len(buckets) > 4 and changes > len(buckets) // 3:
        _print(
            "[red]The bucket is changing more than once every three sessions. "
            "That is flapping, not regime detection -- investigate the tercile "
            "lookback and the transition detector before trusting this.[/red]"
            if _console
            else "WARNING: bucket is flapping; investigate before trusting it."
        )
    else:
        _print(
            "[dim]Buckets persist across multiple sessions, which is what a "
            "regime read should do.[/dim]"
            if _console
            else "Buckets persist across sessions, as expected."
        )
    return 0


def _requested_date(args: argparse.Namespace, fallback):
    import datetime as _dt

    raw = getattr(args, "date", None)
    if not raw:
        return fallback
    try:
        return _dt.date.fromisoformat(raw)
    except ValueError:
        return fallback


def _opt_num(value: Optional[float], suffix: str = "") -> str:
    """Format an optional number. Missing prints as '-', never as 0."""
    if value is None:
        return "-"
    return f"{value:,.2f}{suffix}"


def _opt_pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.1%}"


def cmd_data_purge_cache(cfg: AppConfig, args: argparse.Namespace) -> int:
    from .data.providers.http import HttpClient

    client = HttpClient(
        cache_dir=cfg.paths.cache,
        user_agent=cfg.params.providers.http.user_agent,
    )
    removed = client.purge_cache()
    _print(f"removed {removed} cached HTTP payload file(s) from {cfg.paths.cache}")
    return 0


# =============================================================================
# parser
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prosignal",
        description=(
            "India solo-quant decision-support signal engine (NSE equities). "
            "Not financial advice. No trades are placed automatically."
        ),
    )
    parser.add_argument("--config", help="path to parameters.yaml", default=None)
    parser.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    parser.add_argument("--quiet", action="store_true", help="suppress console logging")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="print engine and schema versions")

    # -- config -----------------------------------------------------------
    cfg_p = sub.add_parser("config", help="inspect and validate parameters.yaml")
    cfg_sub = cfg_p.add_subparsers(dest="subcommand")

    show = cfg_sub.add_parser("show", help="print every tunable with its status")
    show.add_argument("--unvalidated-only", action="store_true")
    show.add_argument("--grep", default=None, help="filter by substring")
    show.set_defaults(func=cmd_config_show)

    validate = cfg_sub.add_parser("validate", help="load and validate the config")
    validate.set_defaults(func=cmd_config_validate)

    tiers = cfg_sub.add_parser(
        "tiers", help="optimisation-tier classification and search budget"
    )
    tiers.set_defaults(func=cmd_config_tiers)

    templates = cfg_sub.add_parser("templates", help="write blank reference CSVs")
    templates.add_argument("--overwrite", action="store_true")
    templates.set_defaults(func=cmd_config_templates)

    # -- data -------------------------------------------------------------
    data_p = sub.add_parser("data", help="Stage 0 data ingestion and inspection")
    data_sub = data_p.add_subparsers(dest="subcommand")

    ingest = data_sub.add_parser("ingest", help="pull every Stage 0 feed")
    ingest.add_argument("--sessions", type=int, default=None, help="sessions of history to guarantee")
    ingest.add_argument("--date", default=None, help="decision date (YYYY-MM-DD)")
    ingest.add_argument("--offline", action="store_true", help="use the store only")
    ingest.add_argument("--refetch", action="store_true", help="re-pull sessions already stored")
    ingest.add_argument("--no-secondary", action="store_true", help="skip the yfinance cross-check")
    ingest.add_argument("--full", action="store_true", help="backfill the full required history")
    ingest.set_defaults(func=cmd_data_ingest)

    status = data_sub.add_parser("status", help="summarise the local store")
    status.set_defaults(func=cmd_data_status)

    check = data_sub.add_parser("check", help="run data-integrity checks")
    check.add_argument("--date", default=None)
    check.set_defaults(func=cmd_data_check)

    fundamentals = data_sub.add_parser(
        "fundamentals", help="ingest point-in-time fundamentals from NSE filings"
    )
    fundamentals.add_argument("--quarters", type=int, default=8,
                              help="quarters of history per symbol")
    fundamentals.set_defaults(func=cmd_data_fundamentals)

    budget = data_sub.add_parser("budget", help="storage usage against the budget")
    budget.set_defaults(func=cmd_data_budget)

    gc = data_sub.add_parser("gc", help="reclaim raw cache to fit policy and budget")
    gc.set_defaults(func=cmd_data_gc)

    purge = data_sub.add_parser("purge-cache", help="delete cached HTTP payloads")
    purge.set_defaults(func=cmd_data_purge_cache)

    # -- analyse ------------------------------------------------------------
    analyse_p = sub.add_parser("analyse", help="run individual pipeline stages")
    analyse_sub = analyse_p.add_subparsers(dest="subcommand")

    regime = analyse_sub.add_parser(
        "regime", help="Stage 2 -- market regime for a date"
    )
    regime.add_argument(
        "--date",
        help="decision date (YYYY-MM-DD). Resolved back to the last real session.",
    )
    regime.add_argument(
        "--history",
        type=int,
        default=1,
        help=(
            "print the regime for the last N sessions instead of one, to check "
            "the buckets persist rather than flapping"
        ),
    )
    regime.set_defaults(func=cmd_analyse_regime)

    full = analyse_sub.add_parser(
        "run", help="RUN MARKET ANALYSIS -- the full eight-stage decision pipeline"
    )
    full.add_argument("--date", help="decision date (YYYY-MM-DD)")
    full.add_argument("--watch", type=int, default=3,
                      help="how many watchlist cards to print")
    full.set_defaults(func=cmd_analyse_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        _print(f"{ENGINE_NAME} {ENGINE_VERSION} (stage schema v{SCHEMA_VERSION})")
        return 0

    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    try:
        cfg = load_config(config_path=args.config)
    except ProSignalError as exc:
        _print(f"[red]{exc.message}[/red]" if _console else exc.message)
        return 1

    setup_logging(
        level=args.log_level or cfg.params.runtime.logging.level,
        log_dir=cfg.paths.logs,
        to_console=cfg.params.runtime.logging.to_console and not args.quiet,
        to_file=cfg.params.runtime.logging.to_file,
        backup_count=cfg.params.runtime.logging.backup_count,
        force=True,
    )

    try:
        return int(args.func(cfg, args))
    except ProSignalError as exc:
        _print()
        _print(f"[red]{exc.code}[/red]: {exc.message}" if _console else f"{exc.code}: {exc.message}")
        if exc.context:
            for k, v in sorted(exc.context.items()):
                _print(f"  {k}: {v}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        _print("\ninterrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
