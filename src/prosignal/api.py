"""FastAPI layer -- the front door. Deliberately thin.

Route handlers do three things: call a service, shape the response, return it.
No analysis logic lives here, so the engine stays testable without HTTP and the
API stays replaceable without touching the engine.

Health vs readiness are genuinely different questions and are answered
separately: `/health` says the process is alive, `/ready` says it could actually
run an analysis right now (config loads, store has sessions, universe exists).
A load balancer needs the first; a human deciding whether to press the button
needs the second.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config.loader import AppConfig, load_config
from .core.logging import get_logger, setup_logging
from .core.memory import release_memory, trim_available
from .data.store import DataStore
from .jobs import JobManager
from .ledger import Ledger
from .pipeline import run_analysis
from .version import ENGINE_NAME, ENGINE_VERSION, SCHEMA_VERSION

__all__ = ["create_app"]

log = get_logger(__name__)

_STATIC = Path(__file__).parent / "static"


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    cfg = config or load_config()
    log_cfg = cfg.params.runtime.logging
    setup_logging(
        level=str(log_cfg.level),
        log_dir=cfg.paths.logs,
        to_console=bool(log_cfg.to_console),
        to_file=bool(log_cfg.to_file),
        backup_count=int(log_cfg.backup_count),
    )

    app = FastAPI(
        title="Pro Stock Signal",
        version=ENGINE_VERSION,
        description=(
            "Decision-support for NSE equities. Produces BUY / WATCH / NO TRADE "
            "with evidence. Places no orders, ever."
        ),
    )

    def _runner(progress) -> Dict[str, Any]:
        run = run_analysis(cfg, progress=progress)
        return _shape(run)

    jobs = JobManager(
        db_path=cfg.paths.data / "jobs.sqlite3",
        runner=_runner,
        timeout_seconds=float(cfg.params.api.job_timeout_seconds)
        if hasattr(cfg.params.api, "job_timeout_seconds")
        else 900.0,
    )
    app.state.cfg = cfg
    app.state.jobs = jobs

    # =====================================================================
    # health / readiness
    # =====================================================================
    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Liveness, plus resident memory.

        RSS is included because it is the number a container platform kills on,
        and it is not otherwise observable from inside a running deployment.
        On a 512 MB instance this is the field to watch.
        """
        payload: Dict[str, Any] = {
            "status": "ok",
            "engine": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "schema": SCHEMA_VERSION,
            "time": dt.datetime.now().isoformat(),
        }
        payload["memory"] = _memory_report()
        return payload

    @app.post("/admin/release-memory")
    def release() -> Dict[str, Any]:
        """Hand free allocator arenas back to the OS.

        Called automatically after every job; exposed so an operator can
        confirm it works on their host and see the effect.
        """
        before = _rss_mb()
        trimmed = release_memory()
        after = _rss_mb()
        return {
            "trimmed": trimmed,
            "rss_before_mb": before,
            "rss_after_mb": after,
            "freed_mb": (round(before - after, 1) if before and after else None),
            "note": (
                "trimmed=false means malloc_trim is unavailable (macOS, or musl "
                "libc such as Alpine). Python memory is still collected; it is "
                "simply not returned to the OS, so RSS will not fall."
            ),
        }

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Could we actually run an analysis right now?"""
        checks: Dict[str, Any] = {}
        ok = True

        try:
            store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
            sessions = store.price_sessions()
            need = int(cfg.params.universe.min_history_sessions.value)
            checks["price_sessions"] = len(sessions)
            checks["min_history_required"] = need
            checks["analysable_dates"] = max(len(sessions) - need, 0)
            if len(sessions) <= need:
                ok = False
                checks["price_data"] = (
                    f"only {len(sessions)} sessions; need more than {need} before "
                    f"any stock has enough history"
                )
                checks["remedy"] = (
                    "the market-data store is empty or short. POST /admin/bootstrap "
                    "(or press BUILD DATA STORE in the UI) to populate it from NSE. "
                    "This is expected on a fresh deployment: data/ is not in version "
                    "control."
                )
            else:
                checks["price_data"] = "ok"
                checks["latest_session"] = sessions[-1].isoformat()
        except Exception as exc:  # noqa: BLE001
            ok = False
            checks["price_data"] = f"{type(exc).__name__}: {exc}"

        try:
            index = str(cfg.params.universe.index_name.value)
            store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
            snaps = store.universe_snapshot_dates(index)
            checks["universe"] = f"{index}: {len(snaps)} snapshot(s)"
            if not snaps:
                ok = False
                checks["universe"] = f"no snapshot for {index}"
        except Exception as exc:  # noqa: BLE001
            ok = False
            checks["universe"] = f"{type(exc).__name__}: {exc}"

        checks["config_version"] = cfg.version
        active = jobs.active_job()
        checks["analysis_in_progress"] = active.id if active else None

        return JSONResponse(
            status_code=200 if ok else 503,
            content={"ready": ok, "checks": checks},
        )

    # =====================================================================
    # analysis
    # =====================================================================
    @app.post("/analysis/run")
    def start_analysis() -> Dict[str, Any]:
        """Start an analysis, or return the one already running.

        Idempotent: a double click gets the same job back rather than launching
        a second full-universe run.
        """
        job = jobs.start()
        return {**job.to_dict(), "already_running": job.state.value == "RUNNING"}

    @app.get("/analysis/{run_id}")
    def job_status(run_id: str) -> Dict[str, Any]:
        job = jobs.get(run_id)
        if job is None:
            raise HTTPException(404, f"no job with id {run_id}")
        payload = job.to_dict()
        payload.pop("result", None)  # status endpoint stays small for polling
        return payload

    @app.get("/analysis/{run_id}/results")
    def job_results(run_id: str) -> Dict[str, Any]:
        job = jobs.get(run_id)
        if job is None:
            raise HTTPException(404, f"no job with id {run_id}")
        if job.state.value == "FAILED":
            raise HTTPException(
                409,
                {
                    "message": "analysis did not complete; no results exist",
                    "error": job.error,
                    "run_id": run_id,
                },
            )
        if job.result is None:
            raise HTTPException(409, f"job {run_id} is {job.state.value}; no results yet")
        return job.result


    def _bootstrap_runner(progress) -> Dict[str, Any]:
        """Populate the data store from NSE, in a bounded chunk.

        A fresh deployment clones no market data -- `data/` is not in version
        control -- so the analysis button is unusable until the store is built
        ON the server. Measured cost is roughly 20s per session against NSE, so
        the full ~360-session requirement is a ~2 hour job. That is too long to
        treat as one atomic operation on a host that can restart.

        So it runs in CHUNKS. Ingest is incremental and cache-backed, meaning a
        repeated call resumes rather than restarting: each press adds another
        chunk and the store converges. A killed process costs one chunk, not
        the whole build.
        """
        from .data.ingest import DataIngestor, IngestOptions

        need = int(cfg.params.universe.min_history_sessions.value) + 30
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        have = len(store.price_sessions())

        chunk = int(getattr(cfg.params.api, "bootstrap_chunk_sessions", 0) or 90)
        target = min(have + chunk, need) if have else min(chunk, need)

        progress(0, f"Have {have} of {need} sessions. Fetching up to {target}...")
        result = DataIngestor(cfg).run(
            options=IngestOptions(
                history_sessions=target,
                include_secondary_prices=False,
                include_delivery=True,
            )
        )
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        sessions = store.price_sessions()
        done = len(sessions) >= need
        progress(8, "complete" if done else f"{len(sessions)}/{need} sessions - press again to continue")

        return {
            "sessions_in_store": len(sessions),
            "sessions_required": need,
            "complete": done,
            "sessions_fetched_this_run": result.sessions_fetched,
            "latest_session": sessions[-1].isoformat() if sessions else None,
            "universe_size": len(result.universe.symbols),
            "next_step": (
                "ready to analyse"
                if done
                else "press BUILD DATA STORE again to fetch the next chunk"
            ),
        }

    @app.post("/admin/bootstrap")
    def bootstrap() -> Dict[str, Any]:
        """Build the market-data store on this host.

        Single-flight shares the analysis slot deliberately: analysing a store
        that is being rewritten underneath would produce a result from
        half-written data.
        """
        job = jobs.start(kind="bootstrap", runner=_bootstrap_runner)
        return {**job.to_dict(), "already_running": job.state.value == "RUNNING"}

    @app.get("/analysis")
    def recent_jobs(limit: int = 20) -> Dict[str, Any]:
        return {
            "jobs": [
                {k: v for k, v in j.to_dict().items() if k != "result"}
                for j in jobs.recent(limit)
            ]
        }

    @app.post("/analysis/{run_id}/cancel")
    def cancel(run_id: str) -> Dict[str, Any]:
        if not jobs.cancel(run_id):
            raise HTTPException(409, f"job {run_id} is not cancellable")
        return {"cancelled": run_id}

    # =====================================================================
    # ledger / history
    # =====================================================================
    @app.get("/ledger")
    def ledger(limit: int = 50) -> Dict[str, Any]:
        led = Ledger(cfg.paths.ledger)
        rows = led.read_all()[-limit:]
        return {"count": led.count(), "trials": led.trial_count(), "runs": rows}

    @app.get("/config")
    def config_report() -> Dict[str, Any]:
        """Full parameter transparency, as the terminal already provides."""
        return cfg.transparency_report()

    # =====================================================================
    # UI
    # =====================================================================
    if _STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(_STATIC / "index.html"))

    return app


def _rss_mb() -> Optional[float]:
    """Resident set size in MB, or None when psutil is not installed."""
    try:
        import psutil  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:
        return None
    import os

    return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)


def _memory_report() -> Dict[str, Any]:
    rss = _rss_mb()
    return {
        "rss_mb": rss,
        "malloc_trim_available": trim_available(),
        "hint": (
            None if rss is None or rss < 400
            else "RSS is high; POST /admin/release-memory or check instance size"
        ),
    }


def _shape(run) -> Dict[str, Any]:
    """Flatten an AnalysisRun into the JSON the UI consumes."""
    o = run.output
    r = o.regime_state
    return {
        "run_id": o.run_id,
        "as_of_date": o.as_of_date.isoformat(),
        "generated_at": o.generated_at.isoformat(),
        "engine_version": o.engine_version,
        "config_version": o.config_version,
        "regime": {
            "bucket": r.regime_bucket,
            "trend": r.trend_regime.value,
            "volatility": f"{r.vol_tercile.value}/{r.vol_context.value}",
            "breadth_pct": r.breadth_pct_above_ma,
            "breadth_state": r.breadth_state.value,
            "transition": r.transition_flag,
            "allow_new_entries": r.allow_new_entries,
            "compatibility": r.compatibility().value,
            "notes": r.notes,
        },
        "funnel": run.funnel,
        "no_trade": (
            {
                "reason": o.no_trade.reason,
                "closest": [
                    {
                        "ticker": c.ticker,
                        "rank": c.rank,
                        "score": c.composite_score,
                        "gate_failed": c.gate_failed,
                        "detail": c.detail,
                    }
                    for c in o.no_trade.closest_candidates
                ],
            }
            if o.no_trade
            else None
        ),
        "recommendations": [_card(x) for x in o.recommendations],
        "watchlist": [_card(x) for x in o.watchlist],
        "data_quality_flags": o.data_quality_flags,
        "stage_timings_ms": o.stage_timings_ms,
        "disclaimer": o.disclaimer,
        "probability_note": (
            "Probability estimate unavailable: no out-of-sample calibration "
            "exists. The score is a RANK within today's eligible universe, not "
            "a likelihood of profit."
        ),
    }


def _card(rec) -> Dict[str, Any]:
    """Shape one recommendation for the UI.

    `factors` is exposed structurally in addition to the prose in `why`. The
    scanner table needs the raw numbers to sort and align on, and parsing them
    back out of formatted English in JavaScript would be fragile in exactly the
    way that breaks silently. No calculation happens here -- these values are
    already computed in stage 4.
    """
    return {
        "factors": {
            name: {
                "raw": f.raw_value,
                "standardised": f.standardised,
                "weight": f.weight,
                "available": f.available,
            }
            for name, f in (getattr(rec, "factor_detail", None) or {}).items()
        },
        "ticker": rec.ticker,
        "company_name": rec.company_name,
        "sector": rec.sector,
        "decision": rec.decision.value,
        "strength": rec.signal_strength_band.value,
        "regime_fit": rec.regime_compatibility.value,
        "last_close": rec.last_close,
        "entry_zone": list(rec.entry_zone) if rec.entry_zone else None,
        "stop": rec.initial_stop,
        "invalidation": rec.invalidation_level,
        "target_1": rec.target_1,
        "target_2": rec.target_2,
        "score": rec.composite_score,
        "percentile": rec.universe_percentile,
        "rank": rec.rank,
        "risk_category": rec.position_risk_category.value if rec.position_risk_category else None,
        "holding_period": rec.expected_holding_period,
        "why": rec.why_this_signal_exists,
        "against": rec.false_signal_flagged,
        "cleared": rec.false_signal_cleared,
        "not_testable": rec.false_signal_not_testable,
        "exits": rec.sell_conditions,
        "cost_note": rec.cost_note,
        "research_basis": rec.research_basis,
        "warning": rec.unvalidated_parameter_warning,
    }
