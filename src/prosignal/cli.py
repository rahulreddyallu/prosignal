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
import os
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

    if sessions is None:
        # Advance the backfill as well as fetching the new day.
        #
        # The default was min_history_sessions + 30 = 330, which is enough to
        # score today and nothing more. A store below the validated depth
        # therefore stayed below it forever: the nightly job added the latest
        # session and never reached further back, so the only thing that grew
        # the history was a person pressing a button. One chunk a night gets
        # there on its own.
        from .data.coverage import assess
        from .data.store import DataStore

        try:
            have = len(DataStore(cfg.paths.curated, cfg.paths.snapshots).price_sessions())
        except Exception:
            have = 0
        cov = assess(cfg, have)
        if have and have < cov.validated_target:
            chunk = int(getattr(cfg.params.api, "bootstrap_chunk_sessions", 0) or 90)
            sessions = min(have + chunk, cov.validated_target)
            _print(f"Store is {have} of {cov.validated_target} sessions; "
                   f"reaching for {sessions} tonight.")

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



def cmd_research_cpcv(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Combinatorial purged cross-validation over the cross-sectional model.

    Reports the DISTRIBUTION of out-of-sample estimates rather than one path.
    Every other number this engine quotes comes from a single walk-forward
    sequence; the spread across paths is what that cannot show.
    """
    import numpy as np
    import pandas as pd

    from .data.store import DataStore
    from .data.types import DATE, SYMBOL
    from .data.universe import UniverseResolver
    from .features import crossmodel as cm
    from .features.crosssec import build_panel, liquidity_mask
    from .stages._cfg import fv, iv, v
    from .validation.harness import run_cpcv
    from .validation.metrics import deflated_sharpe_ratio

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")

    holdout_start = sessions[-iv(val.holdout.reserve_most_recent_sessions)]
    end = holdout_start if not args.include_holdout else sessions[-1]
    if args.include_holdout:
        _print(_tag("HOLDOUT INCLUDED -- this consumes the one honest test"))

    u = p.universe
    sectors = store.read_sector_map()
    sector_map = (dict(zip(sectors["symbol"], sectors["sector"]))
                  if sectors is not None and not sectors.empty else {})
    snap = UniverseResolver(store, cfg).resolve_liquidity_pit(
        as_of=sessions[-1], min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names), min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr),
        manual_exclusions=list(v(u.manual_exclusions) or []), sector_map=sector_map,
    )
    symbols = list(snap.symbols)

    _rule("Building the panel")
    # UNRESTRICTED. `symbols` is the screen resolved for the LATEST session, and
    # building the panel from it projected today's survivors backwards over
    # every training date. Measured against the screen resolved properly per
    # date: 27% of the names eligible on 2024-08-12 are absent from today's set,
    # excluded for what happened afterwards, while names eligible today
    # contributed rows on dates they could not have been traded on. The panel is
    # masked per date by `liquidity_mask` below instead.
    px = store.read_prices(start=sessions[0], end=end,
                           columns=[DATE, SYMBOL, "close", "turnover"])
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
    close = px.pivot_table(index=DATE, columns=SYMBOL, values="close",
                           aggfunc="last", observed=True).sort_index()
    turnover = px.pivot_table(index=DATE, columns=SYMBOL, values="turnover",
                              aggfunc="last", observed=True).sort_index()
    del px
    delivery = None
    dl = store.read_delivery(start=sessions[0], end=end)
    if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
        dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
        delivery = dl.pivot_table(index=DATE, columns=SYMBOL, values="deliv_pct",
                                  aggfunc="last", observed=True).sort_index()
    del dl
    horizon = iv(c4.model_horizon_sessions)
    eligible = liquidity_mask(
        close, turnover,
        min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names),
        min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr),
    )
    panel = build_panel(close, turnover, horizon=horizon, step=21,
                        delivery=delivery, eligible=eligible)
    panel = cm._attach_fundamentals(panel, store.read_statements(), close,
                                    iv(c4.max_fundamental_age_days))
    _print(f"  {len(panel):,} rows over {panel['date'].nunique()} dates")

    def progress(n, total):
        if n % 20 == 0 or n == total:
            _print(f"  split {n}/{total}")

    _rule(f"CPCV  N={iv(val.cpcv.n_groups)}  k={args.test_groups}")
    result = run_cpcv(
        panel, list(cm.FEATURE_COLUMNS), horizon_sessions=horizon, step_sessions=21,
        alpha=fv(c4.model_ridge_alpha), n_groups=iv(val.cpcv.n_groups),
        n_test_groups=args.test_groups,
        purge_sessions=iv(val.cpcv.purge_sessions),
        embargo_sessions=iv(val.cpcv.embargo_sessions),
        progress=progress,
    )

    spread = result.path_spread()
    _table("Out-of-sample path distribution", ["metric", "value"], [
        ["splits fitted", f"{result.n_splits}"],
        ["paths woven", f"{len(result.path_sharpes)} of {result.n_paths}"],
        ["observations purged", f"{result.purged_total:,}"],
        ["observations embargoed", f"{result.embargoed_total:,}"],
        ["pooled rank IC", f"{result.mean_ic:+.4f}"],
        ["path Sharpe -- min", f"{spread.get('min', float('nan')):+.2f}"],
        ["path Sharpe -- median", f"{spread.get('median', float('nan')):+.2f}"],
        ["path Sharpe -- max", f"{spread.get('max', float('nan')):+.2f}"],
        ["path Sharpe -- sd", f"{spread.get('sd', float('nan')):.2f}"],
        ["paths below zero", f"{spread.get('share_negative', float('nan')):.0%}"],
    ])
    trials = args.trials
    dsr = result.deflated(n_trials=trials)
    t_bar = fv(val.significance.t_stat_bar)
    pbo_bar = float(val.search_budget.max_acceptable_pbo)
    paths = np.asarray(result.path_sharpes, dtype="float64")
    _print()
    _print(f"  pooled top-decile excess {np.mean(result.excess):+.2%} per "
           f"{horizon}-session period")
    _print(f"  Deflated Sharpe {dsr.deflated_sr:.3f} charging {trials} trials -- "
           f"{'PASS' if dsr.passes else 'FAIL'}")
    worst = float(paths.min()) if paths.size else float("nan")
    _print(f"  worst of {paths.size} paths: Sharpe {worst:+.2f}; "
           f"{spread.get('share_negative', float('nan')):.0%} of paths below zero")
    _print(f"  significance bar in config: t >= {t_bar:.1f}; "
           f"PBO for promotion to VALIDATED: <= {pbo_bar:.0%}")
    _print()
    _print("  No t-statistic is quoted, for either the pooled observations or "
           "the paths. A test date appears in many splits, and the paths are "
           "fitted on heavily overlapping training sets over one shared "
           "calendar, so neither is a sample of independent experiments and a t "
           "from either is inflated. What CPCV honestly gives is the SPREAD: "
           "where the worst path landed, and how much of the distribution sits "
           "below zero.")
    if result.notes:
        _print()
        for note in result.notes[:5]:
            _print(f"  note: {note}")
    return 0



def _portfolio_inputs(cfg: AppConfig, store, sessions, symbols, end):
    """Aligned OHLCV panels plus ATR, the structure MA and ADTV.

    ``symbols`` of None reads every name in the store. The portfolio CPCV needs
    that: its panel must span every name the universe screen would have admitted
    on each past date, not the names it admits today.
    """
    import numpy as np
    import pandas as pd

    from .data.types import DATE, SYMBOL
    from .stages._cfg import fv, iv

    c7 = cfg.params.stage7_risk
    px = store.read_prices(symbols=symbols, start=sessions[0], end=end,
                           columns=[DATE, SYMBOL, "open", "high", "low", "close", "volume"])
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
    panels = {
        col: px.pivot_table(index=DATE, columns=SYMBOL, values=col,
                            aggfunc="last", observed=True).sort_index()
        for col in ("open", "high", "low", "close", "volume")
    }
    del px
    close, high, low = panels["close"], panels["high"], panels["low"]
    prev = close.shift(1)
    true_range = pd.concat(
        [(high - low).stack(), (high - prev).abs().stack(), (low - prev).abs().stack()],
        axis=1,
    ).max(axis=1).unstack()
    period = iv(c7.atr.period_sessions)
    panels["atr"] = true_range.ewm(alpha=1.0 / period, adjust=False,
                                   min_periods=period).mean()
    panels["ma"] = close.rolling(iv(c7.thesis_invalidation.structure_ma_sessions)).mean()
    panels["adtv"] = (close * panels["volume"]).rolling(21).mean()
    return panels


def _portfolio_params(cfg: AppConfig):
    """The shipped stage settings, with the real cost model attached.

    Cost is a function of position size and the name's liquidity, not a
    constant: impact is square-root in participation, so the same rupee
    position costs ~86 bps against a Rs 20 crore ADTV and ~135 bps against
    Rs 5 crore. Passing a flat number understates exactly the thin names a
    screen surfaces.
    """
    from .costs import CostModel
    from .stages._cfg import fv, iv
    from .validation.portfolio_sim import PortfolioParams

    cap, c6, c7 = cfg.params.capital, cfg.params.stage6_entry, cfg.params.stage7_risk
    model = CostModel(cfg)

    def cost_bps(price: float, quantity: float, adtv: float) -> float:
        if price <= 0 or quantity <= 0:
            return 0.0
        return float(
            model.round_trip(price, int(max(quantity, 1)),
                             adtv_inr=adtv if adtv > 0 else None).total_bps_of_buy
        )

    return PortfolioParams(
        cost_fn=cost_bps,
        capital=fv(cap.total_capital_inr),
        max_positions=iv(cap.max_open_positions),
        risk_per_trade_pct=fv(cap.risk_per_trade_pct),
        max_participation_of_adtv=fv(cap.max_participation_of_adtv),
        stop_atr_multiple=fv(c7.stop_loss.atr_multiple),
        min_stop_distance_pct=fv(c7.stop_loss.min_stop_distance_pct),
        max_stop_distance_pct=fv(c7.stop_loss.max_stop_distance_pct),
        invalidation_ma_sessions=iv(c7.thesis_invalidation.structure_ma_sessions),
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        horizon_sessions=iv(cfg.params.stage4_core_score.model_horizon_sessions),
        entry_rank=iv(c6.admission.entry_rank),
        exit_rank=iv(c6.admission.exit_rank),
    )


def cmd_research_factors(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Standalone IC, ICIR, decay and breakeven turnover -- per factor.

    Answers the three questions that have to be answered BEFORE any blending and
    were not being answered at all: does this factor carry information on its
    own, how reliably date to date, and for how long does the information last.
    """
    import numpy as np
    import pandas as pd

    from .data.store import DataStore
    from .data.types import DATE, SYMBOL
    from .features import crossmodel as cm
    from .features.crosssec import FEATURES, build_panel, liquidity_mask
    from .stages._cfg import fv, iv
    from .costs import CostModel
    from .validation.factor_ic import breakeven_turnover, factor_ic, net_of_cost

    def _short(name: str) -> str:
        """Strip only the TRAILING suffix. `str.replace` took the `_r` out of
        the middle of `resid_reversal_r` and printed `resideversal`."""
        for suffix in ("_r", "_f"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name


    p = cfg.params
    val, c4, u = p.validation, p.stage4_core_score, p.universe
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = (sessions[-1] if args.include_holdout
           else sessions[-iv(val.holdout.reserve_most_recent_sessions)])
    if args.include_holdout:
        _print(_tag("HOLDOUT INCLUDED -- this spends the one honest test"))

    _rule("Building the panel")
    px = store.read_prices(start=sessions[0], end=end,
                           columns=[DATE, SYMBOL, "close", "turnover"])
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
    close = px.pivot_table(index=DATE, columns=SYMBOL, values="close",
                           aggfunc="last", observed=True).sort_index()
    turnover = px.pivot_table(index=DATE, columns=SYMBOL, values="turnover",
                              aggfunc="last", observed=True).sort_index()
    del px
    delivery = None
    dl = store.read_delivery(start=sessions[0], end=end)
    if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
        dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
        delivery = dl.pivot_table(index=DATE, columns=SYMBOL, values="deliv_pct",
                                  aggfunc="last", observed=True).sort_index()
    del dl
    sectors = store.read_sector_map()
    sector_map = (dict(zip(sectors["symbol"], sectors["sector"]))
                  if sectors is not None and not sectors.empty else {})
    eligible = liquidity_mask(
        close, turnover, min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names),
        min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr))
    horizon = iv(c4.model_horizon_sessions)
    panel = build_panel(close, turnover, horizon=horizon, step=21,
                        delivery=delivery, eligible=eligible, sectors=sector_map)
    panel = cm._attach_fundamentals(panel, store.read_statements(), close,
                                    iv(c4.max_fundamental_age_days))
    _print(f"  {len(panel):,} rows over {panel['date'].nunique()} dates")

    _rule("Standalone rank IC, before any blending")
    _print(f"  {'factor':<20}{'dates':>7}{'IC':>9}{'ICIR':>8}{'t':>8}{'hit':>7}")
    rows = []
    columns = [f + "_r" for f in FEATURES] + [
        f + "_r" for f in cm.FUNDAMENTAL_FEATURES]
    for col in columns:
        result = factor_ic(panel, col, label="label")
        if result is None:
            _print(f"  {_short(col):<20}{'not computable':>39}")
            continue
        rows.append(result)
        _print(f"  {_short(result.factor):<20}{result.n_dates:>7}"
               f"{result.ic_mean:>+9.4f}{result.icir:>+8.3f}"
               f"{result.t_stat:>+8.2f}{result.hit_rate:>7.0%}")

    _rule("The families, and the same question of them")
    built = cm.build_families(panel, columns)
    for col in built:
        result = factor_ic(panel, col, label="label")
        if result is not None:
            _print(f"  {_short(result.factor):<20}{result.n_dates:>7}"
                   f"{result.ic_mean:>+9.4f}{result.icir:>+8.3f}"
                   f"{result.t_stat:>+8.2f}{result.hit_rate:>7.0%}")

    _rule("Correlation, and what is not independent evidence")
    cutoff = fv(c4.redundancy.max_abs_spearman)
    block = panel[[c for c in columns if c in panel.columns]].dropna(axis=1, how="all")
    corr = block.rank().corr()
    breaches = []
    names = list(corr.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= cutoff:
                breaches.append((a, b, float(r)))
    if breaches:
        for a, b, r in sorted(breaches, key=lambda t: -abs(t[2])):
            _print(f"  {_short(a):<20}{_short(b):<20}{r:>+8.3f}")
    else:
        _print(f"  no pair reaches |rho| = {cutoff}")

    _rule("Breakeven turnover")
    costs = CostModel(cfg)
    round_trip = float(costs.round_trip(300.0, 400).total_bps_of_buy)
    _print(f"  round trip modelled at {round_trip:.0f} bps")
    _print(f"  {'gross annual edge':<24}{'round trips it pays for':>26}")
    for gross in (0.02, 0.04, 0.08):
        _print(f"  {gross:<24.0%}{breakeven_turnover(gross, round_trip):>26.2f}")
    _print()
    _print("  A factor whose implied turnover exceeds T* loses money however")
    _print("  good the gross IC looks. STT alone is 20 bps round trip.")
    return 0


def cmd_research_forward(cfg: AppConfig, args: argparse.Namespace) -> int:
    """How far the pre-registered forward test has run.

    Reports elapsed time and integrity only. It deliberately shows NO
    performance: reading an interim result and stopping when it looks good is
    optional stopping, and a test stopped that way has no p-value worth
    quoting. The numbers arrive when both targets are met, or not at all.
    """
    from .ledger import Ledger
    from .validation.forward import load_registration, progress, verify

    if getattr(args, "start", False) or getattr(args, "restart", False):
        import subprocess

        from . import __version__
        from .validation.forward import register
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        try:
            opened = register(cfg.paths.ledger, config_version=cfg.version,
                              engine_version=__version__, git_commit=commit,
                              overwrite=bool(args.restart))
        except FileExistsError as exc:
            _print(str(exc))
            return 1
        _print(f"Forward test registered {opened.started_on}, fingerprint "
               f"{opened.fingerprint()}.")

    reg = load_registration(cfg.paths.ledger)
    if reg is None:
        _print("No forward test is registered.")
        _print("  `prosignal research forward --start` opens one against the "
               "current configuration.")
        return 1

    _rule("Pre-registered forward test")
    _table("registration", ["field", "value"], [
        ["started", reg.started_on],
        ["configuration", reg.config_version],
        ["commit", reg.git_commit[:12]],
        ["fingerprint", reg.fingerprint()],
        ["intact", "yes" if verify(cfg.paths.ledger) else "NO -- FILE EDITED"],
    ])

    try:
        rows = Ledger(cfg.paths.ledger).iter_rows()
    except Exception as exc:
        _print(f"  ledger unreadable: {exc}")
        return 1
    prog = progress(cfg.paths.ledger, rows)
    if prog is None:
        return 1

    _print()
    _table("progress", ["field", "value"], [
        ["sessions", f"{prog.sessions_elapsed} of {prog.sessions_target}"],
        ["months", f"{prog.months_elapsed} of {prog.months_target}"],
        ["runs recorded", f"{prog.runs_recorded}"],
        ["latest session", prog.latest_session or "none yet"],
        ["configurations seen", ", ".join(prog.config_versions) or "none yet"],
    ])
    _print()
    _print(f"  {prog.summary()}")
    if not prog.complete and not prog.broken:
        _print()
        _print("  No performance figure is shown by design. The pre-registered "
               "tests run once, at the end.")
    return 0 if not prog.broken else 1


def cmd_research_portfolio(cfg: AppConfig, args: argparse.Namespace) -> int:
    """CPCV over the BOOK, not the ranking.

    An IC says the ordering is right. It says nothing about what a book built
    on that ordering returns after sizing, a stop, an invalidation level and
    costs -- and sizing is risk_budget/risk_per_share, so those interact.
    """
    import numpy as np
    import pandas as pd

    from .data.store import DataStore
    from .data.types import DATE, SYMBOL
    from .data.universe import UniverseResolver
    from .features import crossmodel as cm
    from .features.crosssec import build_panel, liquidity_mask
    from .stages._cfg import fv, iv, v
    from .validation.harness import run_portfolio_cpcv

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = (sessions[-1] if args.include_holdout
           else sessions[-iv(val.holdout.reserve_most_recent_sessions)])
    if args.include_holdout:
        _print(_tag("HOLDOUT INCLUDED -- this spends the one honest test"))

    u = p.universe
    sectors = store.read_sector_map()
    sector_map = (dict(zip(sectors["symbol"], sectors["sector"]))
                  if sectors is not None and not sectors.empty else {})
    snap = UniverseResolver(store, cfg).resolve_liquidity_pit(
        as_of=sessions[-1], min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names), min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr),
        manual_exclusions=list(v(u.manual_exclusions) or []), sector_map=sector_map,
    )
    symbols = list(snap.symbols)

    _rule("Building panels")
    # UNRESTRICTED, then masked per date. `symbols` is the screen resolved for
    # the LATEST session; building the panel from it projects today's survivors
    # backwards over every training date.
    panels = _portfolio_inputs(cfg, store, sessions, None, end)
    turnover_panel = (panels["close"] * panels["volume"])
    delivery = None
    dl = store.read_delivery(start=sessions[0], end=end)
    if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
        dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
        delivery = dl.pivot_table(index=DATE, columns=SYMBOL, values="deliv_pct",
                                  aggfunc="last", observed=True).sort_index()
    del dl
    horizon = iv(c4.model_horizon_sessions)
    eligible = liquidity_mask(
        panels["close"], turnover_panel,
        min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names),
        min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr),
    )
    panel = build_panel(panels["close"], turnover_panel, horizon=horizon,
                        step=21, delivery=delivery, eligible=eligible)
    panel = cm._attach_fundamentals(panel, store.read_statements(), panels["close"],
                                    iv(c4.max_fundamental_age_days))
    params = _portfolio_params(cfg)
    _print(f"  {len(panel):,} rows over {panel['date'].nunique()} dates")
    sample = params.cost_bps(300.0, 400, 2e8)
    thin = params.cost_bps(300.0, 400, 5e7)
    _print(f"  book: {params.max_positions} slots, entry rank {params.entry_rank}, "
           f"exit rank {params.exit_rank}, stop {params.stop_atr_multiple:g}x ATR")
    _print(f"  cost: size- and liquidity-dependent -- {sample:.0f} bps on a "
           f"Rs 1.2L position at Rs 20cr ADTV, {thin:.0f} bps at Rs 5cr")

    def progress(n, total):
        if n % 20 == 0 or n == total:
            _print(f"  split {n}/{total}")

    _rule(f"Portfolio CPCV  N={iv(val.cpcv.n_groups)}  k={args.test_groups}")
    result = run_portfolio_cpcv(
        panel, list(cm.FEATURE_COLUMNS), panels, params,
        step_sessions=21, alpha=fv(c4.model_ridge_alpha),
        n_groups=iv(val.cpcv.n_groups), n_test_groups=args.test_groups,
        purge_sessions=iv(val.cpcv.purge_sessions),
        embargo_sessions=iv(val.cpcv.embargo_sessions),
        progress=progress,
    )
    sharpe = result.spread("sharpe")
    ret = result.spread("mean_return")
    dd = result.spread("max_drawdown")
    if not sharpe:
        _print("  no split produced a tradeable book; nothing to report")
        return 1
    _table("Book performance across CPCV splits",
           ["metric", "min", "p25", "median", "p75", "max"],
           [["Sharpe"] + [f"{sharpe[k]:+.2f}" for k in ("min", "p25", "median", "p75", "max")],
            ["return/period"] + [f"{ret[k]:+.2%}" for k in ("min", "p25", "median", "p75", "max")],
            ["max drawdown"] + [f"{dd[k]:+.1%}" for k in ("min", "p25", "median", "p75", "max")]])
    _print()
    _print(f"  splits scored          {sharpe['n']} of {result.n_splits}")
    _print(f"  splits with Sharpe < 0 {sharpe['share_negative']:.0%}")
    _print(f"  mean names held        {np.mean([m['avg_names'] for m in result.split_metrics]):.1f}")
    _print()
    _print("  Each split trades only the dates it holds out, with purging and "
           "embargo applied, so a cohort never runs through a training block. "
           "The spread is the report; no t-statistic is quoted, because splits "
           "share training data and calendar and are not independent trials.")
    return 0


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

    shadow = analyse_sub.add_parser(
        "shadow",
        help="run the full pipeline and record it WITHOUT issuing anything",
    )
    shadow.add_argument("--date", help="decision date (YYYY-MM-DD)")
    shadow.set_defaults(func=_cmd_analyse_shadow)

    parity = analyse_sub.add_parser(
        "parity",
        help="diff a recorded shadow run against a replay of the same date",
    )
    parity.add_argument("--date", required=True, help="date to reconcile (YYYY-MM-DD)")
    parity.set_defaults(func=_cmd_analyse_parity)

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

    research_p = sub.add_parser(
        "research", help="validation runs that do not issue signals"
    )
    research_sub = research_p.add_subparsers(dest="subcommand")
    cpcv_p = research_sub.add_parser(
        "cpcv",
        help="combinatorial purged cross-validation -- a DISTRIBUTION of "
             "out-of-sample estimates rather than one walk-forward path",
    )
    cpcv_p.add_argument("--test-groups", type=int, default=3,
                        help="k in C(N,k). Higher k means more paths and less "
                             "training data per split (default 3 -> 36 paths)")
    cpcv_p.add_argument("--trials", type=int, default=24,
                        help="configurations tried, charged against the Deflated "
                             "Sharpe. Understating it inflates the result")
    cpcv_p.add_argument("--include-holdout", action="store_true",
                        help="extend the panel through the reserved holdout. "
                             "This spends the one honest test -- do not use it "
                             "while still choosing between models")
    cpcv_p.set_defaults(func=cmd_research_cpcv)

    fac_p = research_sub.add_parser(
        "factors",
        help="standalone IC, ICIR and correlation per factor, BEFORE blending",
    )
    fac_p.add_argument("--include-holdout", action="store_true",
                       help="extend the panel through the reserved holdout. "
                            "This spends the one honest test")
    fac_p.set_defaults(func=cmd_research_factors)

    fwd_p = research_sub.add_parser(
        "forward", help="status of the pre-registered forward test")
    fwd_p.add_argument("--start", action="store_true",
                       help="open a forward test against the current configuration")
    fwd_p.add_argument("--restart", action="store_true",
                       help="discard the running test and open a new one")
    fwd_p.set_defaults(func=cmd_research_forward)

    port_p = research_sub.add_parser(
        "portfolio",
        help="CPCV over the BOOK -- sizing, stop, invalidation, buffer bands "
             "and costs, not just the ranking",
    )
    port_p.add_argument("--test-groups", type=int, default=2,
                        help="k in C(N,k). Lower than the ranking default: a "
                             "book needs contiguous test dates to trade")
    port_p.add_argument("--include-holdout", action="store_true",
                        help="extend through the reserved holdout. Spends the "
                             "one honest test")
    port_p.set_defaults(func=cmd_research_portfolio)

    return parser


SHADOW_DIR = "shadow"


def _shadow_path(config, as_of):
    root = config.paths.curated.parent / SHADOW_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{as_of.isoformat()}.json"


def _cmd_analyse_shadow(config, args) -> int:
    """Run the pipeline live and record it, wired to nothing.

    Shadow mode exists to answer a question the backtest cannot: did the live
    pipeline, on the day, see what the backtest assumes it saw. It writes a
    snapshot and returns. It does not issue, notify, or persist a
    recommendation anywhere a reader could mistake for a decision.
    """
    import json as _json
    from .parity import snapshot_run
    from .pipeline import run_analysis

    run = run_analysis(config, as_of=_resolve_arg_date(getattr(args, "date", None)))
    as_of = dt.date.fromisoformat(str(run.output.as_of_date)[:10])
    snapshot = snapshot_run(run)
    snapshot["shadow"] = True
    path = _shadow_path(config, as_of)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(snapshot, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    _print(f"shadow run recorded for {as_of} -> {path}")
    _print("  NOT a decision: this output is wired to nothing.")
    return 0


def _cmd_analyse_parity(config, args) -> int:
    """Replay a date from the settled store and diff it against the shadow run."""
    import json as _json
    from .parity import compare_snapshots, snapshot_run
    from .pipeline import run_analysis

    as_of = dt.date.fromisoformat(args.date)
    path = _shadow_path(config, as_of)
    if not path.is_file():
        _print(f"no shadow run recorded for {as_of}. Run `analyse shadow` on the day.")
        return 2
    live = _json.loads(path.read_text(encoding="utf-8"))
    replay = snapshot_run(run_analysis(config, as_of=as_of))
    report = compare_snapshots(live, replay)
    _print(report.render())
    return 0 if report.clean else 1


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
