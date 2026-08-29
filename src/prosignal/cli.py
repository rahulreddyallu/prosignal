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
from pathlib import Path
from typing import List, Optional

from .config.loader import AppConfig, load_config
from .core.errors import ConfigError, DataError, ProSignalError
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


def _selection_end(cfg: AppConfig, args: argparse.Namespace, sessions):
    """Last session a research command may read, and the holdout rule enforced.

    `validation.holdout.sacred` shipped as `true` and was read by NOTHING. Eight
    research commands each carried their own copy of

        end = sessions[-1] if args.include_holdout else sessions[-reserve]

    with a printed warning and no check, so the flag that names the holdout
    untouchable was decoration and a single keystroke spent it.

    Spending a holdout is a legitimate thing to do ONCE, which is what a holdout
    is for. It is not a legitimate thing to do by accident, and it must not be
    possible to do it invisibly. So it is refused here while `sacred` is true,
    and the way to spend it is to set that flag to false in
    `config/parameters.yaml` -- which moves `config_version`, which is stamped
    on every ledger row and every registration from that moment on. The decision
    then has a date, an author and a hash, instead of a flag on one command.
    """
    from .stages._cfg import iv

    reserve = iv(cfg.params.validation.holdout.reserve_most_recent_sessions)
    if not getattr(args, "include_holdout", False):
        return sessions[-reserve]
    if bool(getattr(cfg.params.validation.holdout, "sacred", False)):
        raise ConfigError(
            "--include-holdout was refused: validation.holdout.sacred is true.\n"
            "\n"
            "The holdout can be read exactly once and the reading is only worth "
            "anything if nothing has been tuned against it. To spend it, set\n"
            "\n"
            "    validation.holdout.sacred: false\n"
            "\n"
            "in config/parameters.yaml. That changes config_version, which is "
            "recorded on every ledger row and every forward-test registration "
            "from then on, so the decision is dated and attributable rather "
            "than a flag somebody passed on a Tuesday.",
            config_version=str(getattr(cfg, "version", "")),
        )
    _print(_tag("HOLDOUT INCLUDED -- this spends the one honest test"))
    return sessions[-1]


def _arm_sharpe(values) -> float:
    """Per-period Sharpe of one research arm's out-of-sample scores.

    What gets stored beside a trial in the registry. Any consistent statistic
    would do -- the DSR needs the DISPERSION across arms, not the level -- but
    it has to be the same one for every arm or the variance is measuring the
    mixture of statistics rather than the mixture of configurations.
    """
    import numpy as _np

    a = _np.asarray([v for v in values if v is not None], dtype="float64")
    a = a[_np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    sd = float(a.std(ddof=1))
    return float(a.mean() / sd) if sd > 0 else float("nan")


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
            chunk = int(cfg.params.api.bootstrap_chunk_sessions)
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

    if getattr(args, "skip_if_recorded", False):
        already = _already_recorded(cfg, _resolve_arg_date(getattr(args, "date", None)))
        if already:
            _print(f"Session {already} already has a recorded run; nothing to do.")
            _print("  The store gained no session since the last observation. On "
                   "an NSE holiday that is the expected outcome -- re-ranking it "
                   "would write a second ledger row for a date that already has "
                   "one.")
            return 0

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
    # THE ENTRY CLOCK, said plainly. Without this line a run on a non-cadence
    # session shows a full watchlist and no buys, which reads exactly like a
    # market that offered nothing -- and an operator seeing that three sessions
    # running would reasonably conclude the engine had stopped working.
    if o.new_entries_blocked:
        _print(f"  ENTRIES CLOSED: {o.new_entries_blocked}")

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

        # WHAT THIS TRADE IS, and what its KIND has done. Printed as its own
        # block so nobody mistakes a population frequency for a per-name
        # forecast: the caveat is on the last line and it is not optional.
        tp = rec.trade_plan
        if tp is not None:
            plan_rows = [
                ["entry cadence", f"every {tp.cadence_sessions} sessions"],
                ["planned hold", f"up to {tp.planned_hold_sessions} sessions"],
            ]
            if tp.expected_hold_sessions is not None:
                plan_rows.append(["typical hold (study)",
                                  f"{tp.expected_hold_sessions:.0f} sessions (median)"])
            if tp.probability_of_profit is not None:
                plan_rows += [
                    ["P(net profit), study", f"{tp.probability_of_profit:.1%}"],
                    ["P(beats the universe)",
                     f"{tp.probability_of_beating_benchmark:.1%}"],
                    ["mean return, study",
                     f"{tp.expected_return_pct:+.2f}%  (median "
                     f"{tp.median_return_pct:+.2f}%)"],
                    ["mean EXCESS, study", f"{tp.expected_excess_pct:+.2f}%"],
                    ["net of", f"{tp.assumed_cost_bps:.0f} bps round trip"],
                ]
            if tp.risk_at_stop_inr is not None:
                plan_rows.append([
                    "risk at the floor",
                    f"{_money(tp.risk_at_stop_inr)}"
                    + (f"  ({tp.risk_at_stop_pct_of_book:.2f}% of the book)"
                       if tp.risk_at_stop_pct_of_book is not None else "")])
            if tp.basis:
                plan_rows.append(["basis", f"{tp.basis}, {tp.basis_trades} trades, "
                                           f"{tp.basis_period}"])
            _table("Plan and what this configuration has done",
                   ["field", "value"], plan_rows)
            _print(f"  {tp.caveat}")

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
            # The last line is the "NOT an exit" disclosure and it must not be
            # truncated away: someone holding a position needs to know that
            # reaching the target does not sell it. Slicing to four was fine
            # while every rung was armed and the list was a simple ordering.
            head = [c for c in rec.sell_conditions if not c.startswith("NOT an exit")]
            tail = [c for c in rec.sell_conditions if c.startswith("NOT an exit")]
            for line in head[:5] + tail:
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


def _already_recorded(cfg: AppConfig, requested: "Optional[dt.date]") -> "Optional[dt.date]":
    """The session this run would rank, if the ledger already holds it.

    Resolved the same way the pipeline resolves it -- the last stored session on
    or before the requested date -- so the answer cannot drift from what the run
    would actually do. Returns None when the store is empty, the date does not
    resolve, or no completed run exists for it; in all three the caller should
    run.
    """
    from .core.calendar import TradingCalendar
    from .data.store import DataStore
    from .ledger import Ledger

    try:
        sessions = DataStore(cfg.paths.curated, cfg.paths.snapshots).price_sessions()
    except Exception:                       # noqa: BLE001 - let the run report it
        return None
    if not sessions:
        return None
    calendar = TradingCalendar(sessions)
    resolved = (calendar.last_session_on_or_before(requested) if requested
                else calendar.last)
    if resolved is None:
        return None
    try:
        for row in Ledger(cfg.paths.ledger).iter_rows():
            if row.get("error"):
                continue
            if str(row.get("date") or "")[:10] == resolved.isoformat():
                return resolved
    except Exception:                       # noqa: BLE001 - an unreadable ledger
        return None                         # is not a reason to skip the run
    return None


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




def _label_name(rp) -> str:
    """What the panel was actually labelled with, named honestly.

    "triple-barrier" alone hid the distinction that mattered: sigma barriers and
    the engine's own stop/target are both triple-barrier and they disagree on
    16% of outcomes.
    """
    if getattr(rp, "exit_rules", None) is not None:
        r = rp.exit_rules
        return (f"engine exits ({r.stop_atr_multiple:g}x ATR stop, "
                f"{r.target_r_multiple:g}R target, invalidation, "
                f"{r.horizon}-session cap)")
    if rp.barriers is not None:
        return (f"sigma barriers ({rp.barriers.upper:g}/{rp.barriers.lower:g} "
                f"horizon-sigma) -- RESEARCH GEOMETRY, not what ships")
    return "horizon return"


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
    from .stages._cfg import fv, iv, v
    from .validation.harness import configuration_matrix, run_cpcv
    from .validation.metrics import compute_pbo, deflated_sharpe_ratio
    from .validation.registry import TrialRegistry, registry_path
    from .validation.research_panel import build_research_panel

    def _short_f(name: str) -> str:
        for suffix in ("_r", "_f"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")

    holdout_start = sessions[-iv(val.holdout.reserve_most_recent_sessions)]
    end = _selection_end(cfg, args, sessions)

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
    # Built by the SHARED builder, so what CPCV validates is what stage 4
    # trades -- including the triple-barrier label. These commands used to
    # construct the panel by hand and did not read high/low, so after the label
    # changed they were still measuring against the horizon return.
    rp = build_research_panel(cfg, store, end)
    panel, features, dropped, horizon = rp.panel, rp.features, rp.dropped, rp.horizon
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")
    _print(f"  label: {_label_name(rp)}")
    _print(f"  fitting {len(features)} famil(y/ies): {', '.join(features)}")
    if dropped:
        _print(f"  {len(dropped)} factor(s) dropped on coverage")

    def progress(n, total):
        if n % 20 == 0 or n == total:
            _print(f"  split {n}/{total}")

    _rule(f"CPCV  N={iv(val.cpcv.n_groups)}  k={args.test_groups}")
    result = run_cpcv(
        panel, features, horizon_sessions=horizon, step_sessions=21,
        alpha=fv(c4.model_ridge_alpha), n_groups=iv(val.cpcv.n_groups),
        n_test_groups=args.test_groups,
        purge_sessions=iv(val.cpcv.purge_sessions),
        embargo_sessions=iv(val.cpcv.embargo_sessions),
        estimator=str(c4.estimator.method),
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
    # The trial count is COUNTED, not typed. It was a command-line default of
    # 24 plus a config field shipped at 0 with a comment asking a human to
    # update it after every campaign -- so the engine's central defence against
    # selection bias was a constant somebody entered once.
    reg = TrialRegistry(registry_path(cfg.paths.curated))
    carried = int(val.search_budget.cumulative_trials_logged)
    counted = reg.effective_trials(carried)
    trials = counted if args.trials is None else int(args.trials)

    _rule("What this result is being charged for")
    by_cmd = reg.by_command()
    if by_cmd:
        for cmd, n in sorted(by_cmd.items(), key=lambda kv: -kv[1]):
            _print(f"  {cmd:<28}{n:>5} configuration(s) compared")
    else:
        _print("  the registry is empty -- no research command has recorded a "
               "comparison yet")
    _print(f"  {'carried from earlier campaigns':<28}{carried:>5}")
    _print(f"  {'CHARGED':<28}{trials:>5}")
    if args.trials is not None and int(args.trials) < counted:
        _print()
        _print(_tag(f"OVERRIDDEN DOWNWARD: the registry counts {counted}"))
        _print("  Charging fewer trials than were actually looked at is exactly")
        _print("  the bias the Deflated Sharpe exists to remove.")

    # INDEPENDENT observations, and Var[SR] from the registry's recorded trial
    # scores when it has any. This used to be `result.deflated(n_trials=trials)`
    # on the pooled (split, date) vector with no variance estimate, which is
    # 612 entries describing 70 dates and a fallback variance of 1/(n-1) =
    # 0.0016. That combination returned 1.0000 PASS for any positive result at
    # any trial count: the defence against multiple testing could not fail.
    dsr = result.deflated(n_trials=trials, horizon_sessions=horizon,
                          step_sessions=21,
                          trial_sharpes=reg.recorded_scores())
    t_bar = fv(val.significance.t_stat_bar)
    pbo_bar = float(val.search_budget.max_acceptable_pbo)
    paths = np.asarray(result.path_sharpes, dtype="float64")
    _print()
    # PER DISTINCT TEST DATE. `result.excess` holds one entry per (split, date)
    # pair -- about nine copies of each date on the shipped geometry -- and
    # printing its count as "scored test dates" invited every reader to treat
    # the duplication as sample size. The pooled count is still shown, labelled
    # for what it is.
    ex = np.asarray(result.excess_per_date(), dtype="float64")
    ex = ex[np.isfinite(ex)]
    n_pooled = int(np.isfinite(np.asarray(result.excess, dtype="float64")).sum())
    if ex.size:
        _print(f"  top-decile excess {ex.mean():+.2%} per {horizon}-session "
               f"period, over {ex.size:,} DISTINCT test dates "
               f"({n_pooled:,} split-date pairs, {result.effective_observations():.1f} "
               f"independent after the {horizon}/{result.step_sessions} overlap)")
    else:
        # Printing "+nan%" here read as a number. It is the absence of one.
        _print("  top-decile excess: NOT MEASURABLE -- no test date "
               "produced a top decile to measure")
    _print(f"  Deflated Sharpe {dsr.deflated_sr:.3f} charging {trials} trials, "
           f"on {dsr.n_observations} independent {horizon}-session windows -- "
           f"{'PASS' if dsr.passes else 'FAIL'}")
    # The inputs the DSR was computed from, printed alongside it. A deflated
    # Sharpe cannot be checked by a reader who cannot see how many independent
    # observations it credited or where its Var[SR] came from, and the previous
    # version -- which returned 1.000 and passed at 100,000 trials -- survived
    # for exactly that reason.
    _print(f"    on {dsr.effective_n:.1f} effective observations "
           f"from {dsr.n_observations:,} values; "
           f"Var[SR] {dsr.sr_variance:.4g} ({dsr.sr_variance_source})")
    # WHAT THESE NUMBERS ARE ABOUT. Every IC and excess above is measured
    # against the LABEL the ranker is fitted on -- the h-session forward return
    # -- and not against what the book earns, which is whatever the stop, the
    # target and the invalidation level produce. The two are different
    # quantities and the gap is measured, so it is printed rather than left for
    # a reader to assume away.
    _print(f"  measured against {result.target}, the {horizon}-session forward "
           f"return -- NOT against the book's realised outcome")
    if result.label_book_rank_corr is not None:
        rc = float(result.label_book_rank_corr)
        _print(f"  label vs realised book outcome: within-date rank corr "
               f"{rc:+.3f} ({rc * rc:.0%} of variance). An excess figure here "
               f"is a statement about the ranking, not about the book.")
    else:
        # NOT a number. The panel handed to CPCV carries no realised-outcome
        # column, so this run did not measure the gap. Quoting the audit's
        # figure here would present a constant as a measurement -- which is
        # what this branch used to do, on every run, forever.
        _print("  the gap between this label and the book's realised outcome "
               "was NOT MEASURED on this run: the panel carries no realised "
               "outcome column. Measured separately during the audit at rank "
               "corr +0.53 on 85 dates, i.e. the label explains about 28% of "
               "what the book earns. That figure is not from this run.")
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
    # ---- PBO -------------------------------------------------------------
    # `compute_pbo` has existed, tested, and been called by NOTHING. The line
    # above quotes "PBO for promotion to VALIDATED" against a number the engine
    # never computed. The configurations compared are the ones a person would
    # actually choose between: the full theme set, and each single-theme and
    # drop-one variant of it.
    _rule("Probability of Backtest Overfitting")
    configurations = {"all themes": list(features)}
    for f in features:
        configurations[f"{_short_f(f)} alone"] = [f]
        if len(features) > 2:
            configurations[f"without {_short_f(f)}"] = [c for c in features if c != f]
    matrix = None
    try:
        matrix = configuration_matrix(
            panel, configurations, step_sessions=21,
            alpha=fv(c4.model_ridge_alpha),
            purge_sessions=iv(val.cpcv.purge_sessions),
            estimator=str(c4.estimator.method))
    except ValueError as exc:
        _print(f"  not computable: {exc}")
        _print()
        _print("  Under the gated estimator a single-theme arm that never")
        _print("  clears the significance floor produces no coefficients at")
        _print("  all -- so it is not a configuration anyone could have chosen,")
        _print("  and there is nothing for PBO to rank it against.")
    if matrix is not None:
        gone = matrix.attrs.get("dropped_configurations") or []
        if gone:
            _print(f"  {len(gone)} configuration(s) the estimator refused "
                   f"outright and could never have been chosen: "
                   f"{', '.join(sorted(gone))}")
        n_splits = min(16, (len(matrix) // 2) * 2)
        if n_splits < 4:
            _print(f"  {len(matrix)} common dates cannot form enough symmetric "
                   f"sub-matrices; PBO needs at least 4")
        else:
            pbo = compute_pbo(matrix.to_numpy("float64"), n_splits=n_splits)
            _print(f"  {len(matrix)} dates scored by all "
                   f"{len(configurations)} configurations, S={n_splits}")
            _print(f"  PBO {pbo.pbo:.0%} -- "
                   f"{'PASS' if pbo.pbo <= pbo_bar else 'FAIL'} against the "
                   f"{pbo_bar:.0%} bar for promotion to VALIDATED")
            if len(matrix) < 24:
                _print()
                _print(_tag(f"ON {len(matrix)} COMMON DATES, THIS PASS MEANS "
                            f"ALMOST NOTHING"))
                _print("  The configurations refuse on different splits, so the")
                _print("  dates ALL of them scored are few. A PBO computed on")
                _print("  this many observations has a sampling error wider")
                _print("  than the bar it is being compared to, and passing it")
                _print("  is not evidence of anything. It is reported because")
                _print("  the alternative -- quoting a bar against a number the")
                _print("  engine never computed -- is worse.")
            _print()
            _print("  PBO is the share of symmetric train/test splits where the")
            _print("  configuration that won in-sample landed BELOW the median")
            _print("  out-of-sample. A high figure is an instruction to")
            _print("  simplify, not to keep searching for a configuration that")
            _print("  happens to score better.")
            reg.record("research cpcv (pbo)", sorted(configurations))

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
    # `turnover` and `adj_factor` travel with the read. `close * volume` is NOT
    # turnover once prices are back-adjusted: close is divided by the cumulative
    # factor and volume is not multiplied by it, so every pre-split date reads a
    # turnover smaller than the exchange printed -- by 20x for a 1:20 split. The
    # ADTV screen and the participation cap both consume this, so the error made
    # names look thinner than they were exactly where a split later occurred.
    # The exchange prints turnover directly; use it.
    px = store.read_prices(symbols=symbols, start=sessions[0], end=end,
                           columns=[DATE, SYMBOL, "open", "high", "low", "close",
                                    "volume", "turnover", "adj_factor"])
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
    panels = {
        col: px.pivot_table(index=DATE, columns=SYMBOL, values=col,
                            aggfunc="last", observed=True).sort_index()
        for col in ("open", "high", "low", "close", "volume", "turnover",
                    "adj_factor")
        if col in px.columns
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
    panels["adtv"] = panels["turnover"].rolling(21).mean()
    # THE ALTERNATIVE, as a price index: equal-weight across the ELIGIBLE
    # universe on each date, compounded. Every simulator run gets it, so no book
    # can be reported without the number it has to beat.
    #
    # The screen is not optional here. Built over every symbol in the store this
    # is not "the universe the book selects from" -- it is a different, wider
    # and less liquid population, and calling it the eligible universe while
    # computing it over everything would put a mislabelled number under the one
    # conclusion that reverses the verdict. The screen is lagged one session,
    # so membership on date t is decided by data through t-1.
    from .features.crosssec import liquidity_mask
    from .stages._cfg import fv as _fv, iv as _iv
    u = cfg.params.stage1_universe
    elig = liquidity_mask(
        close, panels["turnover"],
        min_adtv_inr=_fv(u.pit_min_adtv_inr),
        lookback_sessions=_iv(u.pit_adtv_lookback_sessions),
        max_names=_iv(u.pit_max_names),
        min_history_sessions=_iv(u.min_history_sessions),
        min_price_inr=_fv(u.min_price_inr),
        adj_factor=panels.get("adj_factor"),
    ).shift(1).fillna(False)
    ret = close.pct_change(fill_method=None).where(elig)
    panels["benchmark"] = (1.0 + ret.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
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
        target_r_multiple=fv(c7.targets.t2_r_multiple),
        # THE ARMING SWITCHES, which this did not pass. `PortfolioParams`
        # defaulted `target_r_multiple` to 3.0 and had no notion of a disarmed
        # rung, so `research portfolio` measured a book that takes profit at 3R
        # and sells on invalidation -- neither of which the engine now does.
        # Harmless while every rung was armed; a measurement of a different
        # strategy the moment two were not.
        use_stop=bool(getattr(c7.exit_hierarchy.stop_loss_breach, "value",
                              c7.exit_hierarchy.stop_loss_breach)),
        use_target=bool(getattr(c7.exit_hierarchy.target_achieved, "value",
                                c7.exit_hierarchy.target_achieved)),
        use_invalidation=bool(getattr(c7.exit_hierarchy.thesis_invalidation,
                                      "value",
                                      c7.exit_hierarchy.thesis_invalidation)),
        # None when disabled, so the overlay is genuinely absent rather than
        # present at a neutral setting -- a scale of exactly 1.0 still clips,
        # still reads the window, and still has to be reasoned about.
        target_vol_annual=(float(c7.volatility_scaling.target_vol_annual)
                           if bool(c7.volatility_scaling.enabled) else None),
        vol_window_sessions=int(c7.volatility_scaling.window_sessions),
        max_vol_scale=float(c7.volatility_scaling.max_scale),
        min_vol_scale=float(c7.volatility_scaling.min_scale),
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
    from .validation.factor_ic import (
        breakeven_turnover, factor_ic, ic_decay, net_of_cost)
    from .validation.research_panel import build_research_panel

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
    end = _selection_end(cfg, args, sessions)

    _rule("Building the panel")
    # The SHARED builder -- same label, same universe screen, same coverage
    # tests as stage 4. Measuring factors against a label the engine no longer
    # fits on is how a factor table stops describing the model.
    rp = build_research_panel(cfg, store, end)
    panel, horizon, sector_map, close = rp.panel, rp.horizon, rp.sector_map, rp.close
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")
    _print(f"  label: {_label_name(rp)}")

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
            _print(f"  {_short(a):<24}{_short(b):<24}{r:>+8.3f}")
    else:
        _print(f"  no pair reaches |rho| = {cutoff}")

    _rule("Holding period, measured rather than chosen")
    costs = CostModel(cfg)
    round_trip = float(costs.round_trip(300.0, 400).total_bps_of_buy)
    _print(f"  round trip modelled at {round_trip:.0f} bps\n")
    # The composite the engine actually ranks on, at every horizon. An
    # EQUAL-WEIGHTED mean of the families is not that composite -- the engine
    # weights them by the fitted coefficients, and under the gated estimator
    # three of five typically carry exactly zero. Measuring decay on the
    # equal-weight blend measured a model nothing runs.
    from .features.famamacbeth import (fama_macbeth, gated_shrink,
                                       score_from_lambda, is_degenerate)
    live = panel[panel["date"].notna()].copy()
    est = c4.estimator
    weights = {c: 1.0 for c in built}
    if str(est.method) == "fama_macbeth":
        _fm = fama_macbeth(panel, built, horizon=horizon, step=21)
        if _fm is not None:
            weights = gated_shrink(_fm, floor=float(est.significance_floor),
                                   toward=str(est.shrink_toward))
            _print(f"  composite weighted by the fitted coefficients: "
                   + ", ".join(f"{_short(k)} {v:+.4f}" for k, v in weights.items()))
            if is_degenerate(weights):
                _print("  every theme was gated out; falling back to equal weight "
                       "for the decay shape only")
                weights = {c: 1.0 for c in built}
    live["composite"] = score_from_lambda(live, weights)
    wide = live.pivot_table(index="date", columns="symbol", values="composite",
                            aggfunc="last", observed=True)
    table = ic_decay(close, wide, horizons=(5, 10, 21, 42, 63, 126))
    if table.empty:
        _print("  not enough dates to measure decay")
    else:
        table = net_of_cost(table, round_trip)
        _print(f"  {'horizon':>8}{'dates':>7}{'IC':>9}{'ICIR':>8}"
               f"{'turns/yr':>10}{'gross':>9}{'cost':>8}{'NET':>9}")
        for _, r in table.sort_values("horizon").iterrows():
            _print(f"  {int(r['horizon']):>8}{int(r['n_dates']):>7}"
                   f"{r['ic_mean']:>+9.4f}{r['icir']:>+8.3f}"
                   f"{r['turnovers_per_year']:>10.1f}{r['gross_annual']:>+9.2%}"
                   f"{r['cost_annual']:>8.2%}{r['net_annual']:>+9.2%}")
        best = table.iloc[0]
        _print()
        _print(f"  Best NET of turnover cost: {int(best['horizon'])} sessions "
               f"at {best['net_annual']:+.2%} a year.")
        _print("  Not the highest raw IC, which is almost always the shortest")
        _print("  horizon and the one that pays the most in costs.")
        _print()
        _print(_tag("READ THE SHAPE, NOT THE LEVELS"))
        _print("  gross/yr converts a rank IC to an annual return at a fixed")
        _print("  0.10 -- a rule of thumb, not a measurement, and the argument")
        _print("  `alpha_per_ic` exists so it can be replaced with a fitted")
        _print("  number. What survives that assumption is the ORDERING: IC")
        _print("  rises monotonically with horizon here, so a longer hold keeps")
        _print("  more of whatever the edge turns out to be. What does not")
        _print("  survive it is any absolute claim about profitability.")

    _rule("Breakeven turnover")
    _print(f"  {'gross annual edge':<24}{'round trips it pays for':>26}")
    for gross in (0.02, 0.04, 0.08):
        _print(f"  {gross:<24.0%}{breakeven_turnover(gross, round_trip):>26.2f}")
    _print()
    _print(f"  {'family':<14}{'IC':>9}{'gross/yr':>11}{'T*':>9}   verdict")
    for col in built:
        result = factor_ic(panel, col, label="label")
        if result is None:
            continue
        gross = result.ic_mean * 0.10
        star = breakeven_turnover(gross, round_trip)
        # The engine holds for `horizon` sessions, so it turns a slot over
        # 252/horizon times a year. A family whose T* is below that cannot pay
        # for the way it is actually traded.
        implied = 252.0 / horizon
        verdict = "pays for itself" if star >= implied else "BELOW implied turnover"
        _print(f"  {_short(col):<14}{result.ic_mean:>+9.4f}{gross:>+11.2%}"
               f"{star:>9.2f}   {verdict}")
    _print()
    _print(f"  The engine turns a slot over {252.0/horizon:.1f} times a year at a")
    _print(f"  {horizon}-session hold. STT alone is 20 bps round trip.")
    _print()
    _print("  T* uses the same 0.10 IC-to-return assumption, so the verdicts")
    _print("  move with it. They are a ranking of families by cost tolerance,")
    _print("  not a profitability test.")
    return 0



def cmd_research_estimator(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Every estimator arm, out of sample, against the equal-weight control.

    An estimated model has to earn its estimation. DeMiguel, Garlappi & Uppal
    (2009) found 1/N beat fourteen optimising rules out of sample because
    estimation error cost more than optimisation gained, and any coefficient
    this engine fits is subject to the same test. Before this command existed
    the comparison had never been run, and the pooled ridge in production turned
    out not to beat the control at all.
    """
    import numpy as np
    import pandas as pd

    from .data.store import DataStore
    from .features.famamacbeth import (equal_weight_lambda, fama_macbeth,
                                       gated_shrink, hierarchical_shrink,
                                       is_degenerate, newey_west_se,
                                       score_from_lambda)
    from .features.linear import predict, purged_walk_forward, rank_ic, ridge_fit
    from .stages._cfg import fv, iv
    from .validation.registry import TrialRegistry, registry_path
    from .validation.research_panel import build_research_panel

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = _selection_end(cfg, args, sessions)

    _rule("Building the panel")
    rp = build_research_panel(cfg, store, end)
    panel, fams, horizon = rp.panel, rp.features, rp.horizon
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")
    _print(f"  label: {_label_name(rp)}")
    if len(fams) < 2:
        _print("  too few families to compare estimators")
        return 1

    _rule("In-sample Fama-MacBeth, for orientation only")
    full = fama_macbeth(panel, fams, horizon=horizon, step=21)
    if full is None:
        _print("  not enough cross-sections")
        return 1
    _print(f"  {'theme':<12}{'lambda':>10}{'se(NW)':>10}{'t':>8}"
           f"{'gated+shrunk':>14}{'1/N':>8}")
    shrunk = gated_shrink(full, floor=fv(c4.estimator.significance_floor),
                          toward=str(c4.estimator.shrink_toward))
    control = equal_weight_lambda(fams)
    for c in fams:
        _print(f"  {c.removesuffix('_f'):<12}{full.lam[c]:>+10.4f}"
               f"{full.se[c]:>10.4f}{full.t_stat[c]:>+8.2f}"
               f"{shrunk[c]:>+14.4f}{control[c]:>+8.2f}")
    _print(f"\n  Newey-West at {full.nw_lags} lags, for the overlap a "
           f"{horizon}-session label sampled every 21 induces.")

    _rule(f"Out of sample -- purged walk-forward, {args.splits} splits")
    folds = purged_walk_forward(panel, fams, horizon=horizon,
                                n_splits=args.splits, embargo=args.embargo)
    if not folds:
        _print("  the panel cannot support a purged walk-forward")
        return 1
    per: dict = {}
    refused: dict = {}
    for f in folds:
        tr, te = f["train"], f["test"]
        w = (tr["uniqueness"].to_numpy("float64")
             if "uniqueness" in tr.columns else None)
        rf = fama_macbeth(tr, fams, horizon=horizon, step=21)
        fit = ridge_fit(tr[fams].to_numpy("float64"),
                        tr["label_rank"].to_numpy("float64"),
                        alpha=fv(c4.model_ridge_alpha), weights=w)
        arms = {"ridge, pooled": None, "equal weight 1/N": control}
        if rf is not None:
            arms["fama-macbeth raw"] = rf.lam
            arms["shrunk, no gate"] = hierarchical_shrink(rf, toward="zero")
            arms["PRODUCTION: gate+shrink"] = gated_shrink(
                rf, floor=fv(c4.estimator.significance_floor),
                toward=str(c4.estimator.shrink_toward))
        for d, g in te.groupby("date", observed=True):
            if len(g) < 30:
                continue
            actual = g["label"].to_numpy("float64")
            for name, lam in arms.items():
                if lam is not None and is_degenerate(lam):
                    refused[name] = refused.get(name, 0) + 1
                    continue
                pred = (predict(fit, g[fams].to_numpy("float64")) if lam is None
                        else score_from_lambda(g, lam).to_numpy())
                if not np.isfinite(pred).any() or float(np.std(pred)) < 1e-12:
                    refused[name] = refused.get(name, 0) + 1
                    continue
                top = actual[pred >= np.quantile(pred, args.top_decile)]
                per.setdefault(name, []).append(
                    (d, rank_ic(pred, actual), float(top.mean() - actual.mean())))

    _print(f"  {'arm':<26}{'dates':>7}{'IC':>9}{'t(NW)':>8}{'hit':>7}"
           f"{'top-decile':>12}{'t':>7}{'no model':>10}")
    for name, rows in per.items():
        ic = np.array([b for _, b, _ in rows])
        ex = np.array([c for _, _, c in rows])
        ic = ic[np.isfinite(ic)]
        if len(ic) < 3:
            _print(f"  {name:<26}{len(ic):>7}   too few scored dates")
            continue
        _print(f"  {name:<26}{len(ic):>7}{ic.mean():>+9.4f}"
               f"{ic.mean() / newey_west_se(ic, 2):>+8.2f}{(ic > 0).mean():>7.0%}"
               f"{ex.mean():>+12.2%}{ex.mean() / newey_west_se(ex, 2):>+7.2f}"
               f"{refused.get(name, 0):>10}")

    # Every arm above was compared and could have been chosen. That is the
    # definition of a trial, and the DSR charges for it whether or not it won.
    # The SCORE goes in with the label. The DSR needs the trial count and the
    # trial DISPERSION; the registry supplied the first and nothing supplied the
    # second, so Var[SR] had to be assumed -- and the answer is more sensitive
    # to that assumption than to anything else it is given.
    _arms = sorted(per.keys())
    TrialRegistry(registry_path(cfg.paths.curated)).record(
        "research estimator", _arms,
        scores=[_arm_sharpe([c for _, _, c in per[a]]) for a in _arms])

    if "equal weight 1/N" in per:
        base = {d: b for d, b, _ in per["equal weight 1/N"]}
        _rule("Paired against the control, on the dates both scored")
        for name, rows in per.items():
            if name == "equal weight 1/N":
                continue
            d = np.array([b - base[dt] for dt, b, _ in rows if dt in base])
            d = d[np.isfinite(d)]
            if len(d) < 3:
                continue
            _print(f"  {name:<26}{d.mean():>+9.4f}   t "
                   f"{d.mean() / newey_west_se(d, 2):>+6.2f}   "
                   f"beats control on {(d > 0).mean():.0%} of dates")
        _print()
        _print(_tag("WHAT THIS CANNOT SETTLE"))
        _print("  These are a few dozen overlapping 63-session windows on one")
        _print("  market. A difference of 0.01 in rank IC is not resolvable")
        _print("  here, and the arms were compared on the same dates that")
        _print("  informed the estimator's design -- so read the ORDERING, and")
        _print("  count every arm above as a trial when deflating a Sharpe.")
    return 0



def _band_periods(rankings, panels, base, entry: int, exit_: int):
    """(phase, date) -> (net, gross, cost) for one band.

    Keyed by phase AND date so two bands are compared on identical rebalances.
    Pooling by date alone would silently pair a period from one phase against a
    different phase's period on the same day.
    """
    from dataclasses import replace

    from .validation.portfolio_sim import simulate

    import numpy as np

    out = {}
    params = replace(base, entry_rank=entry, exit_rank=exit_)
    stride = max(int(np.ceil(params.horizon_sessions / 21)), 1)
    for phase in range(stride):
        result = simulate(rankings, panels, params, phase=phase, step_sessions=21)
        if result.empty:
            continue
        for _, row in result.periods.iterrows():
            out[(phase, row["date"])] = (
                float(row["ret"]),
                float(row.get("gross_ret", float("nan"))),
                float(row.get("cost_ret", float("nan"))),
            )
    return out


def _nw_se(series, lags: int) -> float:
    from .features.famamacbeth import newey_west_se
    return newey_west_se(series, lags)


def cmd_research_spread(cfg: AppConfig, args: argparse.Namespace) -> int:
    """What the buy/hold spread costs, and what it buys.

    The engine enters at rank 8 and holds until rank 16. That gap is the whole
    of its turnover control, and it had never been priced: the simulator netted
    cost into the return and threw the parts away, so a wider band's saving --
    it does not pay entry cost on a name it already holds -- was invisible.

    A no-spread book (entry == exit) is included as the control. If hysteresis
    does not beat it net of cost, the band is decoration.
    """
    import numpy as np
    import pandas as pd

    from dataclasses import replace

    from .data.store import DataStore
    from .features.linear import predict, purged_walk_forward
    from .features.crossmodel import fit_coefficients
    from .stages._cfg import fv, iv
    from .validation.portfolio_sim import phase_summary
    from .validation.registry import TrialRegistry, registry_path
    from .validation.research_panel import build_research_panel

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = _selection_end(cfg, args, sessions)

    _rule("Building panels")
    panels = _portfolio_inputs(cfg, store, sessions, None, end)
    turnover_panel = panels["turnover"]
    rp = build_research_panel(cfg, store, end, prices=panels,
                              turnover=turnover_panel)
    panel, fams, horizon = rp.panel, rp.features, rp.horizon
    base = _portfolio_params(cfg)
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")
    _print(f"  shipped band: enter at rank {base.entry_rank}, "
           f"exit past rank {base.exit_rank}, {base.max_positions} slots")

    # Rankings come from a PURGED walk-forward fit, never from a model that saw
    # the dates it is ranking. The band comparison is still in-sample in the
    # sense that every band is scored on the same dates -- that is what the
    # nested search exists to price, and it is said plainly below.
    folds = purged_walk_forward(panel, fams, horizon=horizon,
                                n_splits=args.splits, embargo=args.embargo)
    if not folds:
        _print("  the panel cannot support a purged walk-forward")
        return 1
    rankings: list = []
    for f in folds:
        fit, _fm, why = fit_coefficients(
            f["train"], fams, estimator=str(c4.estimator.method),
            alpha=fv(c4.model_ridge_alpha), horizon=horizon, step=21,
            significance_floor=fv(c4.estimator.significance_floor),
            shrink_toward=str(c4.estimator.shrink_toward),
            weights=(f["train"]["uniqueness"].to_numpy("float64")
                     if "uniqueness" in f["train"].columns else None))
        if fit is None:
            continue
        for d, g in f["test"].groupby("date", observed=True):
            pred = predict(fit, g[fams].to_numpy("float64"))
            rankings.append((pd.Timestamp(d),
                             pd.Series(pred, index=g["symbol"].to_numpy())
                             .sort_values(ascending=False)))
    rankings.sort(key=lambda t: t[0])
    if len(rankings) < 6:
        _print(f"  only {len(rankings)} out-of-sample ranking dates; too few")
        return 1
    _print(f"  {len(rankings)} out-of-sample ranking dates from "
           f"{len(folds)} purged folds")

    _rule("Every spread, gross and net")
    _print(f"  {'entry':>6}{'exit':>6}{'spread':>8}{'new/reb':>9}"
           f"{'gross':>9}{'cost':>8}{'NET':>9}{'cost/gross':>12}{'Sharpe':>8}")
    rows = []
    for entry in args.entries:
        for exit_ in args.exits:
            if exit_ < entry:
                continue
            m = phase_summary(rankings, panels,
                              replace(base, entry_rank=entry, exit_rank=exit_),
                              step_sessions=21)
            if not m:
                continue
            share = m.get("cost_share_of_gross", float("nan"))
            rows.append(dict(entry=entry, exit=exit_, **m))
            _print(f"  {entry:>6}{exit_:>6}{exit_ - entry:>8}"
                   f"{m['avg_new']:>9.1f}{m.get('mean_gross', float('nan')):>+9.2%}"
                   f"{m.get('mean_cost', float('nan')):>8.2%}"
                   f"{m['mean_return']:>+9.2%}"
                   f"{share:>12.0%}{m['sharpe']:>+8.2f}")
    if not rows:
        _print("  no band produced a tradeable book")
        return 1

    TrialRegistry(registry_path(cfg.paths.curated)).record(
        "research spread",
        [f"entry={r['entry']},exit={r['exit']}" for r in rows],
        scores=[r.get("sharpe") for r in rows])

    frame = pd.DataFrame(rows)
    if any(e > base.max_positions for e in args.entries):
        _print()
        _print(f"  note: an entry rank above {base.max_positions} is INERT -- "
               f"the book has {base.max_positions} slots, so nothing beyond the "
               f"{base.max_positions}th candidate is ever added. Those rows "
               f"duplicate entry {base.max_positions}.")

    # Comparing column means across bands is weak: the bands are walked over the
    # same rebalances, so the difference should be measured PAIRED, period by
    # period, where the shared market move cancels.
    _rule("Paired against a no-spread book, period by period")
    ctrl = _band_periods(rankings, panels, base, base.entry_rank, base.entry_rank)
    if not ctrl:
        _print("  the no-spread control produced no periods")
    else:
        _print(f"  control: enter and exit at rank {base.entry_rank}, no "
               f"hysteresis at all")
        _print(f"  {'band':<10}{'n':>5}{'net diff':>11}{'t(NW)':>8}"
               f"{'gross given up':>16}{'cost saved':>12}{'wins':>7}")
        for exit_ in sorted(x for x in args.exits if x > base.entry_rank):
            test = _band_periods(rankings, panels, base, base.entry_rank, exit_)
            keys = sorted(set(ctrl) & set(test))
            if len(keys) < 6:
                continue
            d = np.array([test[k][0] - ctrl[k][0] for k in keys])
            dg = np.array([test[k][1] - ctrl[k][1] for k in keys])
            dc = np.array([ctrl[k][2] - test[k][2] for k in keys])
            t = d.mean() / _nw_se(d, 2)
            _print(f"  {base.entry_rank}/{exit_:<8}{len(d):>5}{d.mean():>+11.3%}"
                   f"{t:>+8.2f}{dg.mean():>+16.3%}{dc.mean():>+12.3%}"
                   f"{(d > 0).mean():>7.0%}")
        _print()
        _print("  'gross given up' is the alpha the band forfeits by holding a")
        _print("  name that has slipped instead of replacing it with the")
        _print("  current best. 'cost saved' is the entry cost it avoided. The")
        _print("  spread pays only when the second exceeds the first.")

    shipped = frame[(frame["entry"] == base.entry_rank)
                    & (frame["exit"] == base.exit_rank)]
    if not shipped.empty:
        r = shipped.iloc[0]
        rank = int((frame["mean_return"] > r["mean_return"]).sum()) + 1
        _print()
        _print(f"  the SHIPPED band ({base.entry_rank}/{base.exit_rank}) ranks "
               f"{rank} of {len(frame)} on net return.")

    _print()
    _print(_tag("THIS TABLE CANNOT CHOOSE THE BAND"))
    _print("  Every row is scored on the same dates, so the best of them is a")
    _print("  selected maximum and its margin is not evidence. `research")
    _print("  portfolio` runs the nested search that prices that selection:")
    _print("  the band is chosen inside each training block and reported")
    _print("  outside it. Read this table for the SHAPE -- how cost scales")
    _print("  with turnover -- not for a winner.")
    return 0


def cmd_research_metalabel(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Does a second model know when the first one is wrong?

    Meta-labelling (Lopez de Prado ch. 3) fits a binary classifier on the trades
    the PRIMARY model would actually have taken, predicting whether one reaches
    its profit barrier before its stop. It cannot pick a name the primary did
    not, so its only power is to veto -- which is the shape the NO TRADE gate
    needs.

    The number that matters is the PER-DATE one. A veto chooses between names on
    a single day; pooling across dates lets "this was a good period" masquerade
    as "this was a good name", and a pooled AUC can look convincing while the
    within-date figure is a coin.
    """
    import numpy as np
    import pandas as pd

    from .data.store import DataStore
    from .features.crossmodel import fit_coefficients
    from .features.famamacbeth import newey_west_se
    from .features.linear import predict, purged_walk_forward
    from .features.metalabel import (auc, fit_meta_out_of_sample, meta_label,
                                     reliability, shortlist)
    from .validation.registry import TrialRegistry, registry_path
    from .stages._cfg import fv, iv
    from .validation.research_panel import build_research_panel

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = _selection_end(cfg, args, sessions)

    _rule("Building the panel")
    rp = build_research_panel(cfg, store, end)
    panel, fams, horizon = rp.panel, rp.features, rp.horizon
    if rp.barriers is None and rp.exit_rules is None:
        _print("  the label is not triple-barrier, so there is no meta-label "
               "to fit. Nothing to measure.")
        return 1
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")
    decided = int((panel["barrier_side"] != 0).sum())
    _print(f"  {decided:,} rows reached a barrier; "
           f"{len(panel) - decided:,} timed out and are excluded")

    top_k = int(args.top_k)

    def primary(train, frame):
        fit, _fm, _why = fit_coefficients(
            train, fams, estimator=str(c4.estimator.method),
            alpha=fv(c4.model_ridge_alpha), horizon=horizon, step=21,
            significance_floor=fv(c4.estimator.significance_floor),
            shrink_toward=str(c4.estimator.shrink_toward),
            weights=(train["uniqueness"].to_numpy("float64")
                     if "uniqueness" in train.columns else None))
        if fit is None:
            return None
        return predict(fit, frame[fams].to_numpy("float64"))

    _rule(f"Purged walk-forward, shortlist = top {top_k}")
    feats = list(fams) + ["_meta_score"]
    collected = []
    for f in purged_walk_forward(panel, fams, horizon=horizon,
                                 n_splits=args.splits, embargo=args.embargo):
        model, why = fit_meta_out_of_sample(
            f["train"], feats, primary, top_k=top_k,
            l2=fv(c4.metalabel.l2), min_rows=args.min_rows)
        if model is None:
            _print(f"  fold {f['fold']}: no veto fitted -- {why}")
            continue
        test_scores = primary(f["train"], f["test"])
        if test_scores is None:
            continue
        rows = shortlist(f["test"], test_scores, top_k).copy()
        rows["_meta_y"] = meta_label(rows["barrier_side"])
        rows = rows.dropna(subset=feats + ["_meta_y"])
        if len(rows) < 40:
            continue
        rows["p"] = model.predict_proba(rows)
        collected.append(rows[["date", "p", "_meta_y", "label"]])
        _print(f"  fold {f['fold']}: trained on {model.n_train} shortlist rows "
               f"(base {model.base_rate:.1%}), tested on {len(rows)}, "
               f"AUC {auc(rows['_meta_y'].to_numpy(), rows['p'].to_numpy()):.3f}")

    if not collected:
        _print()
        _print("  NO FOLD COULD FIT A VETO. Eight positions over seventy")
        _print("  rebalances is roughly 370 decided trades in the entire")
        _print("  history; that is the binding constraint, not the code.")
        return 1

    TrialRegistry(registry_path(cfg.paths.curated)).record(
        "research metalabel", [f"top_k={top_k},l2={fv(c4.metalabel.l2):g}"])

    all_rows = pd.concat(collected, ignore_index=True)
    per_date = []
    for _, g in all_rows.groupby("date", observed=True):
        if len(g) < 20 or g["_meta_y"].nunique() < 2:
            continue
        cut = g["p"].median()
        per_date.append((auc(g["_meta_y"].to_numpy(), g["p"].to_numpy()),
                         float(g[g["p"] >= cut]["label"].mean()
                               - g[g["p"] < cut]["label"].mean())))
    _rule("Discrimination")
    pooled = auc(all_rows["_meta_y"].to_numpy(), all_rows["p"].to_numpy())
    _print(f"  {len(all_rows):,} out-of-sample shortlist rows over "
           f"{all_rows['date'].nunique()} dates")
    _print(f"  pooled AUC                 {pooled:.4f}")
    if len(per_date) >= 5:
        a = np.array([x for x, _ in per_date])
        sp = np.array([y for _, y in per_date])
        t_auc = (a.mean() - 0.5) / newey_west_se(a, 2)
        _print(f"  mean per-date AUC          {a.mean():.4f}   "
               f"t vs 0.5 {t_auc:+.2f}   above 0.5 on {(a > 0.5).mean():.0%} of dates")
        _print(f"  top-half minus bottom-half {sp.mean():+.2%} per period   "
               f"t {sp.mean() / newey_west_se(sp, 2):+.2f}")
        _print()
        if abs(t_auc) < 2.0:
            _print(_tag("THE VETO DOES NOT DISCRIMINATE WITHIN A DATE"))
            _print("  which is the only question it can answer. A pooled AUC")
            _print("  above 0.5 with a per-date AUC at 0.5 means the classifier")
            _print("  is separating good PERIODS from bad ones, not good names")
            _print("  from bad ones -- and a gate built on that is market")
            _print("  timing wearing a trade-selection costume.")

    _rule("Calibration -- a gate reads the LEVEL, not the ordering")
    table = reliability(all_rows["_meta_y"].to_numpy(), all_rows["p"].to_numpy(),
                        bins=args.bins)
    if table.empty:
        _print("  not enough rows to bucket")
    else:
        _print(f"  {'bucket':>8}{'n':>7}{'predicted':>12}{'realised':>11}{'error':>9}")
        for _, r in table.iterrows():
            _print(f"  {int(r['bucket']):>8}{int(r['n']):>7}{r['predicted']:>12.3f}"
                   f"{r['realised']:>11.3f}{r['realised'] - r['predicted']:>+9.3f}")

    enabled = bool(c4.metalabel.enabled)
    floor = fv(p.stage8_final_signal.scarcity.min_win_probability)
    _print()
    _print(f"  shipped: metalabel.enabled = {enabled}, "
           f"min_win_probability = {floor:g}")
    if not enabled or floor <= 0:
        _print("  The veto is inert. Nothing above is gating the book.")
    return 0


def cmd_research_trials(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Every configuration this engine has been compared at.

    The Deflated Sharpe charges a result for the number of configurations tried
    before it. That number used to be a command-line default of 24 and a config
    field shipped at 0 with a comment asking a human to update it. This is the
    audit trail behind it.
    """
    from .validation.registry import TrialRegistry, registry_path

    reg = TrialRegistry(registry_path(cfg.paths.curated))
    trials = reg.load()
    carried = int(cfg.params.validation.search_budget.cumulative_trials_logged)

    _rule("Trial registry")
    _print(f"  {registry_path(cfg.paths.curated)}")
    if not trials:
        _print("  empty. No research command has recorded a comparison yet, so "
               "the Deflated Sharpe would charge only the carried count.")
    else:
        by_cmd = reg.by_command()
        for cmd_name, n in sorted(by_cmd.items(), key=lambda kv: -kv[1]):
            _print(f"\n  {cmd_name}  ({n})")
            for t in trials:
                if t.command == cmd_name:
                    _print(f"      {t.recorded_at[:10]}  {t.label}")
    _print()
    _print(f"  recorded                        {len(trials):>5}")
    _print(f"  carried from earlier campaigns  {carried:>5}")
    _print(f"  CHARGED BY THE DSR              {reg.effective_trials(carried):>5}")
    if carried == 0 and trials:
        _print()
        _print("  `cumulative_trials_logged` is 0. Everything before this "
               "registry existed is therefore uncounted -- it cannot be "
               "reconstructed, and is not being silently assumed to be nothing.")
    return 0


def cmd_research_volscale(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Does scaling total exposure by recent market volatility help?

    Moreira & Muir (2017) say it should: volatility is far more forecastable at
    short horizons than return is, so scaling by the inverse of recent realised
    variance raises the Sharpe ratio.

    The number to read is the SHARPE, not the mean return. The overlay changes
    leverage, so a higher target mechanically raises both return and volatility;
    only a ratio invariant to that can say whether anything was gained.
    """
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from .data.store import DataStore
    from .features.crossmodel import fit_coefficients
    from .features.famamacbeth import newey_west_se
    from .features.linear import predict, purged_walk_forward
    from .stages._cfg import fv, iv
    from .validation.portfolio_sim import simulate
    from .validation.registry import TrialRegistry, registry_path
    from .validation.research_panel import build_research_panel

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = _selection_end(cfg, args, sessions)

    _rule("Building panels")
    panels = _portfolio_inputs(cfg, store, sessions, None, end)
    rp = build_research_panel(cfg, store, end, prices=panels,
                              turnover=panels["turnover"])
    panel, fams, horizon = rp.panel, rp.features, rp.horizon
    base = _portfolio_params(cfg)
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")

    rankings = []
    for f in purged_walk_forward(panel, fams, horizon=horizon,
                                 n_splits=args.splits, embargo=args.embargo):
        fit, _fm, _why = fit_coefficients(
            f["train"], fams, estimator=str(c4.estimator.method),
            alpha=fv(c4.model_ridge_alpha), horizon=horizon, step=21,
            significance_floor=fv(c4.estimator.significance_floor),
            shrink_toward=str(c4.estimator.shrink_toward),
            weights=(f["train"]["uniqueness"].to_numpy("float64")
                     if "uniqueness" in f["train"].columns else None))
        if fit is None:
            continue
        for d, g in f["test"].groupby("date", observed=True):
            rankings.append((pd.Timestamp(d),
                             pd.Series(predict(fit, g[fams].to_numpy("float64")),
                                       index=g["symbol"].to_numpy())
                             .sort_values(ascending=False)))
    rankings.sort(key=lambda t: t[0])
    if len(rankings) < 6:
        _print(f"  only {len(rankings)} out-of-sample ranking dates; too few")
        return 1
    _print(f"  {len(rankings)} out-of-sample ranking dates")

    def periods(params):
        out = {}
        stride = max(int(np.ceil(params.horizon_sessions / 21)), 1)
        for phase in range(stride):
            r = simulate(rankings, panels, params, phase=phase, step_sessions=21)
            if r.empty:
                continue
            for _, row in r.periods.iterrows():
                out[(phase, row["date"])] = (
                    float(row["ret"]), float(row.get("vol_scale", 1.0)),
                    float(row.get("realised_vol", float("nan"))))
        return out

    ppy = 252.0 / float(horizon)
    off = periods(replace(base, target_vol_annual=None))
    if not off:
        _print("  the book produced no periods")
        return 1

    _rule("Exposure scaled by recent market volatility")
    _print(f"  {'target':>8}{'n':>5}{'mean ret':>11}{'sd':>9}{'Sharpe':>9}"
           f"{'vs off':>10}{'t(NW)':>8}{'avg scale':>11}")
    r0 = np.array([v[0] for _, v in sorted(off.items())])
    _print(f"  {'OFF':>8}{len(r0):>5}{r0.mean():>+11.2%}{r0.std(ddof=1):>9.2%}"
           f"{r0.mean() / r0.std(ddof=1) * np.sqrt(ppy):>+9.2f}"
           f"{'--':>10}{'--':>8}{1.0:>11.2f}")
    best_sharpe, best_name = r0.mean() / r0.std(ddof=1) * np.sqrt(ppy), "OFF"
    # (label, score) per arm, so the registry records what each one scored
    # rather than only that it was looked at.
    _vol_arms = [("off", float(best_sharpe))]
    for target in args.targets:
        on = periods(replace(base, target_vol_annual=float(target)))
        keys = sorted(set(off) & set(on))
        if len(keys) < 6:
            continue
        a = np.array([on[k][0] for k in keys])
        b = np.array([off[k][0] for k in keys])
        d = a - b
        scale = np.array([on[k][1] for k in keys])
        sharpe = a.mean() / a.std(ddof=1) * np.sqrt(ppy)
        _vol_arms.append((f"target={target:.2f}", float(sharpe)))
        if sharpe > best_sharpe:
            best_sharpe, best_name = sharpe, f"{target:.0%}"
        _print(f"  {target:>8.0%}{len(a):>5}{a.mean():>+11.2%}{a.std(ddof=1):>9.2%}"
               f"{sharpe:>+9.2f}{d.mean():>+10.2%}"
               f"{d.mean() / newey_west_se(d, 2):>+8.2f}{scale.mean():>11.2f}")
    TrialRegistry(registry_path(cfg.paths.curated)).record(
        "research volscale", [lbl for lbl, _ in _vol_arms],
        scores=[sc for _, sc in _vol_arms])

    vols = np.array([v[2] for v in off.values() if np.isfinite(v[2])])
    _print()
    if vols.size:
        _print(f"  realised universe volatility over these periods: "
               f"median {np.median(vols):.1%}, range {vols.min():.1%} to "
               f"{vols.max():.1%}")
    _print(f"  best Sharpe: {best_name}")
    if best_name == "OFF":
        _print()
        _print(_tag("THE OVERLAY DOES NOT EARN ITS PLACE"))
        _print("  A higher target raises the mean return, and raises the")
        _print("  volatility with it -- that is leverage, not alpha, and a")
        _print("  t-statistic on the return difference will happily confirm it.")
        _print("  The Sharpe is the measure invariant to how much of the book")
        _print("  is on, and on that measure switching the overlay off wins.")
    enabled = bool(p.stage7_risk.volatility_scaling.enabled)
    _print()
    _print(f"  shipped: volatility_scaling.enabled = {enabled}")
    return 0


def cmd_research_decay(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Is a theme still working, or did it used to work?

    Evaluates the kill criterion declared in
    `stage4_core_score.decay_monitor`. This command does NOT choose it: a rule
    picked after seeing which themes it would remove is the selection it exists
    to prevent.
    """
    import numpy as np

    from .data.store import DataStore
    from .stages._cfg import fv, iv
    from .validation.decay import assess_decay
    from .validation.research_panel import build_research_panel

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    dm = c4.decay_monitor
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = _selection_end(cfg, args, sessions)

    _rule("Building the panel")
    rp = build_research_panel(cfg, store, end)
    _print(f"  {len(rp.panel):,} rows over {rp.n_dates} dates")

    _rule("The criterion, declared before the numbers")
    _print(f"  A theme is killed when its trailing {int(dm.window_dates)}-date "
           f"Newey-West t has been")
    _print(f"  at or below {float(dm.kill_t_stat):+.2f} on every check across "
           f"{int(dm.required_breaches)} consecutive checks --")
    _print(f"  a complete refresh of the window, so no single bad quarter can "
           f"end a theme.")

    verdict = assess_decay(
        rp.panel, rp.features,
        window=int(dm.window_dates), kill_t=float(dm.kill_t_stat),
        required_breaches=int(dm.required_breaches),
        haircut=float(dm.post_publication_haircut),
        horizon=rp.horizon, step=21)
    if verdict is None:
        _print("  not enough cross-sections to estimate any theme")
        return 1

    _rule("Theme health")
    _print(f"  {'theme':<12}{'full':>9}{'t':>7}{'recent':>9}{'t':>7}"
           f"{'expected':>10}{'of exp.':>9}{'breach':>8}{'verdict':>9}")
    for h in verdict.themes:
        share = h.share_of_expected
        _print(f"  {h.theme.removesuffix('_f'):<12}{h.full_lambda:>+9.4f}"
               f"{h.full_t:>+7.2f}{h.recent_lambda:>+9.4f}{h.recent_t:>+7.2f}"
               f"{h.expected_lambda:>+10.4f}"
               f"{(f'{share:.0%}' if np.isfinite(share) else 'n/a'):>9}"
               f"{h.breaches:>8}"
               f"{('KILL' if h.killed else 'keep'):>9}")
    _print()
    _print("  'of exp.' is blank where the FULL-SAMPLE coefficient is itself")
    _print("  indistinguishable from zero: there is no expectation to fall")
    _print("  short of, and dividing by noise printed reversal at 837% and")
    _print("  lottery at 4341% -- both reading as though the theme thrived.")
    _print()
    _print(f"  'expected' applies the McLean-Pontiff haircut of "
           f"{float(dm.post_publication_haircut):.0%} to the full-sample")
    _print("  coefficient. Every theme here comes from a published paper, so a")
    _print("  theme merely MEETING that expectation is behaving exactly as the")
    _print("  literature predicts -- not underperforming. Above 100% of")
    _print("  expected is the honest definition of still working.")

    if verdict.notes:
        _print()
        for note in verdict.notes:
            _print(f"  note: {note}")

    _rule("Verdict")
    if verdict.killed:
        _print(f"  KILL: {', '.join(t.removesuffix('_f') for t in verdict.killed)}")
        for h in verdict.themes:
            if h.killed:
                _print(f"    {h.theme.removesuffix('_f')}: {h.reason}")
        _print()
        _print("  This is a pre-committed removal, not a suggestion. The theme")
        _print("  is removed from FAMILIES; it is not left in at a small")
        _print("  coefficient to see whether it recovers.")
    else:
        _print("  No theme meets the kill criterion.")
        weak = [h for h in verdict.themes
                if np.isfinite(h.recent_t) and h.recent_t <= float(dm.kill_t_stat)]
        if weak:
            _print(f"  {len(weak)} theme(s) currently breaching but not for long "
                   f"enough: "
                   f"{', '.join(h.theme.removesuffix('_f') for h in weak)}")
            _print("  Breaching is not dying. That distinction is the whole")
            _print("  reason the criterion requires a full window refresh.")
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
        from .validation.readiness import RestartRefused
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        try:
            opened = register(cfg.paths.ledger, config_version=cfg.version,
                              engine_version=__version__, git_commit=commit,
                              overwrite=bool(args.restart), cfg=cfg)
        except FileExistsError as exc:
            _print(str(exc))
            return 1
        except RestartRefused as exc:
            _rule("Restart refused")
            for r in exc.reasons:
                _print(f"  - {r}")
            _print()
            _print("  The window this would replace is void because the engine "
                   "changed under it. Opening a new one before the engine "
                   "stops changing repeats that, and costs another eighteen "
                   "months to find out.")
            _print("  `prosignal research readiness` shows all eight gates.")
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
    # Same two inputs the API supplies. A CLI that could not see a config
    # change made between registration and the first observation reported
    # "Registered today; no forward session yet" on a window already void.
    from .data.store import DataStore
    from .validation.forward import sessions_in_window
    try:
        sessions = DataStore(cfg.paths.curated, cfg.paths.snapshots).price_sessions()
    except Exception:
        sessions = []
    prog = progress(cfg.paths.ledger, rows,
                    live_config_version=str(cfg.version),
                    sessions_printed=sessions_in_window(sessions, reg.started_on))
    if prog is None:
        return 1

    _print()
    _table("progress", ["field", "value"], [
        ["sessions", f"{prog.sessions_elapsed} of {prog.sessions_target}"],
        ["months", f"{prog.months_elapsed} of {prog.months_target}"],
        ["runs recorded", f"{prog.runs_recorded}"],
        ["latest session", prog.latest_session or "none yet"],
        ["configurations seen", ", ".join(prog.config_versions) or "none yet"],
        ["session coverage",
         "not yet meaningful" if prog.coverage is None
         else f"{prog.coverage:.0%} of {prog.sessions_expected} printed"],
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
    from .validation.research_panel import build_research_panel

    p = cfg.params
    val, c4 = p.validation, p.stage4_core_score
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    if not sessions:
        raise DataError("the local store has no price sessions.")
    end = _selection_end(cfg, args, sessions)

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
    # The SHARED builder for the feature panel -- same label, same universe
    # screen, same coverage tests as stage 4 -- with the price frames passed in
    # so the store is read once. The simulator needs open/high/low/atr, which
    # is why this path loads them itself.
    panels = _portfolio_inputs(cfg, store, sessions, None, end)
    turnover_panel = panels["turnover"]
    rp = build_research_panel(cfg, store, end, prices=panels,
                              turnover=turnover_panel)
    panel, features, dropped, horizon = rp.panel, rp.features, rp.dropped, rp.horizon
    _print(f"  label: {_label_name(rp)}")
    _print(f"  fitting {len(features)} famil(y/ies): {', '.join(features)}")
    _print(f"  {len(panel):,} rows over {rp.n_dates} dates")
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
        panel, features, panels, params,
        step_sessions=21, alpha=fv(c4.model_ridge_alpha),
        n_groups=iv(val.cpcv.n_groups), n_test_groups=args.test_groups,
        purge_sessions=iv(val.cpcv.purge_sessions),
        embargo_sessions=iv(val.cpcv.embargo_sessions),
        estimator=str(c4.estimator.method),
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

    man_p = data_sub.add_parser(
        "manifest",
        help="content hash of the curated store, so a result can name its "
             "inputs without the data going into Git")
    man_p.add_argument("--write", action="store_true",
                       help="rebuild the manifest from the store as it is now")
    man_p.add_argument("--verify", action="store_true",
                       help="check the store still matches the manifest")
    man_p.add_argument("--quick", action="store_true",
                       help="size and mtime only; skips the content hash")
    man_p.set_defaults(func=cmd_data_manifest)

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
    full.add_argument(
        "--skip-if-recorded", action="store_true",
        help=("exit 0 without running when the session this would rank already "
              "has a recorded run. For the unattended nightly job: on an NSE "
              "holiday the ingest fetches nothing and this would otherwise "
              "re-rank the previous session and write a second ledger row for "
              "a date that already has one. A person pressing SCAN wants the "
              "rerun, so this is opt-in."),
    )
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
    cpcv_p.add_argument("--trials", type=int, default=None,
                        help="configurations charged against the Deflated Sharpe. "
                             "Defaults to the TRIAL REGISTRY count plus "
                             "validation.search_budget.cumulative_trials_logged. "
                             "Overriding it downward is announced, because "
                             "charging fewer trials than were looked at is the "
                             "bias the DSR exists to remove")
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

    est_p = research_sub.add_parser(
        "estimator",
        help="compare estimator arms out of sample against an equal-weight control")
    est_p.add_argument("--splits", type=int, default=6,
                       help="purged walk-forward splits (default 6)")
    est_p.add_argument("--embargo", type=int, default=5,
                       help="embargo in sampled dates (default 5)")
    est_p.add_argument("--top-decile", type=float, default=0.90,
                       dest="top_decile", help="top-decile cutoff (default 0.90)")
    est_p.add_argument("--include-holdout", action="store_true",
                       help="spend the reserved holdout; normally left alone")
    est_p.set_defaults(func=cmd_research_estimator)

    spread_p = research_sub.add_parser(
        "spread", help="price the buy/hold spread: gross, cost and net per band")
    spread_p.add_argument("--entries", type=int, nargs="+", default=[5, 8, 10],
                          help="entry ranks to sweep (default 5 8 10)")
    spread_p.add_argument("--exits", type=int, nargs="+",
                          default=[5, 8, 10, 12, 16, 20, 25],
                          help="exit ranks to sweep")
    spread_p.add_argument("--splits", type=int, default=6,
                          help="purged walk-forward splits (default 6)")
    spread_p.add_argument("--embargo", type=int, default=5,
                          help="embargo in sampled dates (default 5)")
    spread_p.add_argument("--include-holdout", action="store_true",
                          help="spend the reserved holdout; normally left alone")
    spread_p.set_defaults(func=cmd_research_spread)

    meta_p = research_sub.add_parser(
        "metalabel", help="measure the NO TRADE veto: does a second model know "
                          "when the first is wrong?")
    meta_p.add_argument("--top-k", type=int, default=50, dest="top_k",
                        help="shortlist depth (default 50)")
    meta_p.add_argument("--splits", type=int, default=6,
                        help="purged walk-forward splits (default 6)")
    meta_p.add_argument("--embargo", type=int, default=5,
                        help="embargo in sampled dates (default 5)")
    meta_p.add_argument("--min-rows", type=int, default=100, dest="min_rows",
                        help="minimum shortlist rows to fit a fold (default 100)")
    meta_p.add_argument("--bins", type=int, default=5,
                        help="calibration buckets (default 5)")
    meta_p.add_argument("--include-holdout", action="store_true",
                        help="spend the reserved holdout; normally left alone")
    meta_p.set_defaults(func=cmd_research_metalabel)

    trials_p = research_sub.add_parser(
        "trials", help="every configuration compared, and what the DSR charges")
    trials_p.set_defaults(func=cmd_research_trials)

    vol_p = research_sub.add_parser(
        "volscale", help="price the portfolio-level volatility overlay")
    vol_p.add_argument("--targets", type=float, nargs="+",
                       default=[0.10, 0.15, 0.20, 0.25],
                       help="annualised volatility targets to sweep")
    vol_p.add_argument("--splits", type=int, default=6)
    vol_p.add_argument("--embargo", type=int, default=5)
    vol_p.add_argument("--include-holdout", action="store_true",
                       help="spend the reserved holdout; normally left alone")
    vol_p.set_defaults(func=cmd_research_volscale)

    decay_p = research_sub.add_parser(
        "decay", help="evaluate the pre-committed kill criterion per theme")
    decay_p.add_argument("--include-holdout", action="store_true",
                         help="spend the reserved holdout; normally left alone")
    decay_p.set_defaults(func=cmd_research_decay)

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

    ep_p = research_sub.add_parser(
        "epoch",
        help="which engine produced a result -- code, config, data, features, "
             "universe policy and execution model as one identity")
    ep_sub = ep_p.add_subparsers(dest="epoch_action")
    ep_p.set_defaults(func=cmd_research_epoch, epoch_action="status")
    ep_sub.add_parser("status", help="the open epoch and any drift from it")
    ep_sub.add_parser("list", help="every epoch ever opened")
    ep_open = ep_sub.add_parser("open", help="open a new epoch")
    ep_open.add_argument("--label", required=True,
                         help="what this epoch is, in a few words")
    ep_open.add_argument("--note", default="",
                         help="why it was opened; what changed")
    ep_open.add_argument("--force", action="store_true",
                         help="open while another is still OPEN. Two open "
                              "epochs means two answers to 'which engine "
                              "produced this'")
    ep_close = ep_sub.add_parser("close", help="retire an epoch, with a reason")
    ep_close.add_argument("epoch_id")
    ep_close.add_argument("--reason", required=True,
                          help="why. A close with no reason is a deletion")
    ep_close.add_argument("--status", default="SUPERSEDED",
                          choices=["SUPERSEDED", "VOID"])
    ep_close.add_argument("--superseded-by", default="",
                          dest="superseded_by")

    find_p = research_sub.add_parser(
        "findings",
        help="the defect register: category, disposition, regression test and "
             "restart consequence for each finding")
    find_p.add_argument("--id", dest="fid", help="one finding, in full")
    find_p.set_defaults(func=cmd_research_findings)

    ready_p = research_sub.add_parser(
        "readiness",
        help="the eight READY gates -- data, universe, features, model, "
             "execution, validation, reproducibility, forward test")
    ready_p.set_defaults(func=cmd_research_readiness)

    rec_p = research_sub.add_parser(
        "record",
        help="the operating record, partitioned by the engine that produced it")
    rec_p.set_defaults(func=cmd_research_record)

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



# =============================================================================
# Provenance: the data a result came from, and the engine that produced it
# =============================================================================


def cmd_data_manifest(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Describe the curated store precisely enough to reconstruct it.

    D1. Not a substitute for backing the data up -- a substitute for putting a
    quarter of a gigabyte of market data into Git in order to be able to say
    which quarter of a gigabyte a figure was computed from. The manifest
    commits; the data does not.

    The digest is over content only -- path, size, sha256, row count, date
    range, schema -- and never over mtime, so re-downloading an identical file
    does not invent a new dataset. It is recomputed on load and never trusted
    from the file it was read out of.
    """
    from .data.manifest import build, load, verify, write

    root = Path(cfg.paths.curated)
    if getattr(args, "write", False):
        man = build(root)
        path = write(man, root)
        _rule("Manifest written")
        _print(f"  {path}")
        _print(f"  {man.summary()}")
        _print(f"  digest {man.digest}")
        _print()
        _print("  Commit this file. A result that names this digest can be "
               "checked against the store it claims to describe; one that "
               "names nothing cannot be reproduced, only re-run.")
        return 0

    man = load(root)
    if man is None:
        _rule("No manifest")
        _print(f"  {root} is not described by a manifest.")
        _print("  `prosignal data manifest --write` writes one.")
        return 1

    _rule("Data manifest")
    _table("store", ["field", "value"], [
        ["root", str(root)],
        ["digest", man.digest],
        ["built", man.built_at],
        ["contents", man.summary()],
    ])

    if getattr(args, "verify", False):
        ok, drift = verify(root, quick=bool(getattr(args, "quick", False)))
        _print()
        if ok:
            _print("  VERIFIED -- every file matches the manifest.")
            return 0
        _print(f"  DRIFTED -- {len(drift)} discrepancies:")
        for d in drift[: int(getattr(args, "limit", 20) or 20)]:
            _print(f"    {d.path}: {d.what}")
        if len(drift) > 20:
            _print(f"    ... and {len(drift) - 20} more")
        _print()
        _print("  The store is not the one this manifest describes, so no "
               "figure computed against the manifest is a figure about this "
               "data. Re-ingest, or re-manifest and open a new epoch.")
        return 1
    return 0


def cmd_research_epoch(cfg: AppConfig, args: argparse.Namespace) -> int:
    """The research epoch: which engine produced a result.

    D2. `config_version` identified a run and does not cover the code, the
    data, the universe policy, the feature schema or the execution model, so
    two materially different engines could produce results labelled the same.
    An epoch fixes all eight at once. Any material change opens a new one --
    and therefore a new out-of-sample question, because a result carried across
    a change of engine is an in-sample result wearing a date.
    """
    from .validation.epoch import (STATUS_OPEN, active, close_epoch,
                                   current_identity, drifted_from, load_all,
                                   open_epoch)

    root = Path(cfg.paths.ledger)
    action = getattr(args, "epoch_action", "status")

    if action == "open":
        try:
            ep = open_epoch(root, cfg, label=args.label, note=args.note or "",
                            allow_while_open=bool(args.force))
        except Exception as exc:
            _print(str(exc))
            return 1
        _rule("Epoch opened")
        _print(f"  {ep.summary()}")
        _print(f"  fingerprint {ep.ident().fingerprint()}")
        _print()
        _print("  Every outcome resolved from here carries this id. Results "
               "from earlier epochs are kept and labelled, never pooled.")
        return 0

    if action == "close":
        try:
            ep = close_epoch(root, args.epoch_id, reason=args.reason,
                             status=args.status,
                             superseded_by=getattr(args, "superseded_by", "") or "")
        except Exception as exc:
            _print(str(exc))
            return 1
        _rule("Epoch closed")
        _print(f"  {ep.summary()}")
        return 0

    if action == "list":
        eps = load_all(root)
        if not eps:
            _print("No epoch has ever been opened.")
            return 1
        _rule("Research epochs")
        _table("ledger", ["id", "label", "opened", "status", "fingerprint"],
               [[e.epoch_id, e.label, e.opened_on, e.status,
                 e.ident().fingerprint()] for e in eps])
        return 0

    # status
    ep, diffs = drifted_from(root, cfg)
    now = current_identity(cfg)
    _rule("Research epoch")
    if ep is None:
        _print("  No epoch is open.")
        _print("  Nothing produced now can be attributed to a described "
               "engine, which is the condition D2 exists to end.")
    else:
        _print(f"  {ep.summary()}")
        _print(f"  fingerprint {ep.ident().fingerprint()}")
    _print()
    _table("this tree", ["field", "value"], [
        ["code", now.code_sha + (" (DIRTY)" if now.code_dirty else "")],
        ["scoring path", now.model_sources_sha],
        ["config", now.config_version],
        ["data", now.data_manifest_sha],
        ["features", now.feature_schema_sha],
        ["universe policy", now.universe_policy],
        ["execution model", now.execution_model],
        ["fingerprint", now.fingerprint()],
    ])
    if diffs:
        _print()
        _print("  DRIFTED from the open epoch:")
        for d in diffs:
            _print(f"    {d}")
        _print()
        _print("  This is reported, not acted on. Whether a change is material "
               "enough to open a new epoch is a judgement, and one made in the "
               "open beats one made by a rule that guessed.")
    elif ep is not None:
        _print()
        _print("  The tree matches the open epoch exactly.")
    return 0 if (ep is not None and not diffs) else 1


def cmd_research_findings(cfg: AppConfig, args: argparse.Namespace) -> int:
    """The defect register, classified. `reviewed` is not a status here.

    Each entry carries its root cause, location, fix, the regression test that
    goes red when the fix is reverted, the before/after, and three consequence
    flags: whether coefficients moved, whether published history moved, and
    whether the forward test must restart. A finding claimed FIXED with no
    regression test raises on import.
    """
    from .validation.findings import (REGISTER, Status, by_id, categorised,
                                      open_findings, restart_blockers)

    one = getattr(args, "fid", None)
    if one:
        f = by_id(one)
        if f is None:
            _print(f"no finding {one!r}")
            return 1
        _rule(f"{f.fid} -- {f.title}")
        _table("disposition", ["field", "value"], [
            ["category", f.category.value],
            ["severity", f.severity],
            ["status", f.status.value],
            ["location", f.location],
            ["regression test", f.regression_test or "(none)"],
            ["moves coefficients", "yes" if f.moves_coefficients else "no"],
            ["moves history", "yes" if f.moves_history else "no"],
            ["forces restart", "yes" if f.forces_restart else "no"],
        ])
        for head, body in (("root cause", f.root_cause), ("fix", f.fix),
                           ("before / after", f.before_after),
                           ("notes", f.notes)):
            if body:
                _print()
                _print(f"  {head.upper()}")
                _print(f"    {body}")
        return 0

    _rule("Defect register")
    for cat, group in categorised().items():
        _print(f"\n  {cat}")
        for f in group:
            _print(f"    {f.describe()}")

    still_open = open_findings()
    blockers = restart_blockers()
    _print()
    _table("summary", ["field", "value"], [
        ["registered", str(len(REGISTER))],
        ["open", ", ".join(f.fid for f in still_open) or "none"],
        ["restart-blocking", ", ".join(f.fid for f in blockers)],
        ["move coefficients",
         ", ".join(f.fid for f in REGISTER if f.moves_coefficients) or "none"],
    ])
    _print()
    _print("  `prosignal research findings --id R9` for one in full.")
    return 0 if not still_open else 1


def cmd_research_readiness(cfg: AppConfig, args: argparse.Namespace) -> int:
    """READY as eight gates, not as a green test suite.

    A green suite says the code does what its tests say. It said so while the
    model was fitted on a population the book cannot buy (R9), while an
    unmeasured name got the cheapest fill in the model (R13), while no result
    could name the data it came from (D1), and while the window measuring all
    of it was void (R1). The gates below are the things a green suite does not
    check.
    """
    from .validation.readiness import assess, restart_refusals

    r = assess(cfg)
    _rule("Readiness")
    for g in r.gates:
        _print("  " + g.line())
    _print()
    _print(f"  {r.verdict()}")

    if r.failed:
        _print()
        _print("  WHAT EACH ONE NEEDS")
        for g in r.failed:
            if g.remedy:
                _print(f"    {g.name:<16} {g.remedy}")

    refusals = restart_refusals(cfg)
    _print()
    if refusals:
        _print("  The forward test cannot restart yet:")
        for x in refusals:
            _print(f"    - {x}")
    else:
        _print("  The forward test MAY be restarted: every precondition holds.")
        _print("  `prosignal research forward --restart` opens the window.")
    return 0 if r.ready else 1


def cmd_research_record(cfg: AppConfig, args: argparse.Namespace) -> int:
    """The operating record, partitioned by the engine that produced it.

    C3/C4. Trades decided before the population and liquidity corrections were
    produced by a different engine on a different universe. They are the only
    operating record that exists, so they are shown -- labelled, beside the
    epoch that supersedes them, and never averaged into it. What pooling would
    have claimed is printed too, because a partition whose cost is invisible
    gets re-collapsed by the next person who finds the per-epoch samples small.
    """
    from . import outcomes as _out

    path = Path(cfg.paths.ledger) / "outcomes.jsonl"
    if not path.is_file():
        _print(f"No outcome record at {path}.")
        return 1
    rows = _out.load_outcomes(path, epoch="*")
    part = _out.summarise_by_epoch(rows, ledger_root=Path(cfg.paths.ledger))

    _rule("Operating record by research epoch")
    _table("epochs", ["epoch", "n", "win rate", "expectancy", "t", "status"],
           [[e["epoch_id"],
             str(e.get("n", 0)),
             f"{e['win_rate']:.0%}" if e.get("n") else "--",
             f"{e['expectancy']:+.2%}" if e.get("n") else "--",
             f"{e['expectancy_t']:+.2f}" if e.get("n") else "--",
             "CURRENT" if e["is_current"] else "retired"]
            for e in part["epochs"]])

    if part["spans_multiple_epochs"]:
        pooled, gap = part["pooled"], part["pooling_overstates_expectancy_by"]
        _print()
        _print(f"  Pooled across every epoch: n={pooled['n']}, "
               f"expectancy {pooled['expectancy']:+.2%}. That figure describes "
               f"no engine.")
        if gap == gap:                                    # not NaN
            _print(f"  Pooling would misstate the current epoch's expectancy "
                   f"by {gap:+.2%}.")
    if part["current_epoch_has_no_record"]:
        _print()
        _print("  The current epoch has no resolved trades. That is an EMPTY "
               "forward record, not a bad one -- and it is the honest state "
               "for an engine whose universe and execution model just changed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
