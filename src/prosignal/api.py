"""FastAPI layer.

Route handlers call a service, shape the response and return it. No analysis
logic lives here, so the engine is testable without HTTP.

`/health` reports that the process is alive; `/ready` reports whether an
analysis could run now -- config loads, the store has sessions, the universe
resolves. A load balancer needs the first, a user deciding whether to press the
button needs the second.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config.loader import AppConfig, load_config
from .core.logging import get_logger, setup_logging
from .core.memory import release_memory, trim_available
from .data.coverage import MINIMUM_NOTE, assess
from .data.store import DataStore
from .auth import (OPEN_PATHS, assert_safe_to_serve, resolve_token,
                    token_matches)
from .jobs import JobManager
from .ledger import Ledger
from .rundetail import card as _card, shape as _shape
from .pipeline import run_analysis
from .version import ENGINE_NAME, ENGINE_VERSION, SCHEMA_VERSION

__all__ = ["create_app"]

log = get_logger(__name__)

_STATIC = Path(__file__).parent / "static"


def _in_outcome_basis(record: Dict[str, Any], outcome) -> Dict[str, Any]:
    """One price basis per row.

    `outcomes_for` re-bases the recorded levels onto the prices the store serves
    now, because a corporate action since the call makes the two different
    currencies. The raw record still carries the originals, so merging them
    verbatim shipped both: a stop of 8195.05 beside a re-based stop of 819.51
    for the same call, and the interface free to render either.
    """
    merged = dict(record)
    merged["signal_price"] = outcome.signal_price
    merged["stop"] = outcome.stop
    merged["target_1"] = outcome.target_1
    return merged


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
    # ---- access control -------------------------------------------------
    # `api.auth_token` existed in the config from v1 and nothing read it. On a
    # laptop bound to 127.0.0.1 that cost nothing; on a public bind it is the
    # entire security model, and the endpoints behind it start jobs that run
    # for minutes (/analysis/run) or hours (/admin/bootstrap).
    # Fails closed: a hosted instance with no token does not start.
    assert_safe_to_serve(cfg, getattr(getattr(cfg.params, "api", None), "host", None))
    _token = resolve_token(cfg)

    @app.middleware("http")
    async def _require_token(request, call_next):
        if _token and request.url.path not in OPEN_PATHS:
            supplied = request.headers.get("authorization", "")
            if supplied.lower().startswith("bearer "):
                supplied = supplied[7:]
            else:
                supplied = request.headers.get("x-api-key", "")
            # The browser cannot set headers on a top-level navigation, so the
            # UI itself is also allowed through a cookie set once at sign-in.
            if not supplied:
                supplied = request.cookies.get("prosignal_token", "")
            if not token_matches(supplied, _token):
                return JSONResponse(
                    {"detail": "not authorised"}, status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    @app.post("/auth")
    def sign_in(request: Request, payload: Dict[str, Any]) -> JSONResponse:
        """Exchange the token for a cookie, so the interface works in a browser.

        Rate limiting is deliberately absent: this is a single-user instance
        and the token is 32 random bytes, which is not guessable at any rate a
        network permits.
        """
        if not _token:
            return JSONResponse({"detail": "no token configured"}, status_code=400)
        if not token_matches(str(payload.get("token", "")), _token):
            return JSONResponse({"detail": "not authorised"}, status_code=401)
        response = JSONResponse({"ok": True})
        # `secure` follows the scheme the request actually arrived on. Hardcoding
        # it True is right for the deployed instance and silently breaks a local
        # HTTP one -- the browser accepts the cookie and never sends it back, so
        # sign-in appears to succeed and every subsequent request 401s.
        forwarded = request.headers.get("x-forwarded-proto", "")
        over_tls = (request.url.scheme == "https"
                    or forwarded.split(",")[0].strip() == "https")
        response.set_cookie(
            "prosignal_token", _token, httponly=True, samesite="strict",
            secure=over_tls, max_age=60 * 60 * 24 * 90,
        )
        return response

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
            # One source of truth. This used to compare against
            # min_history_sessions (300), which is the bar for a STOCK to be
            # scoreable, not the bar for the MODEL to fit -- so a store of 330
            # reported ready and then produced no ranking at all.
            cov = assess(cfg, len(sessions))
            checks.update(cov.to_dict())
            checks["analysable_dates"] = max(len(sessions) - cov.eligibility_minimum, 0)
            if not cov.model_will_fit:
                ok = False
                checks["price_data"] = cov.status()
                checks["remedy"] = (
                    "the market-data store is too short for the ranking model. "
                    "POST /admin/bootstrap (or press BUILD DATA STORE in the UI) "
                    "until it reports the validated depth. This is expected on a "
                    "fresh deployment: data/ is not in version control."
                )
            elif not cov.matches_validation:
                # Usable, but not the model that was validated. Serving without
                # saying so would let a 16-month fit pass for a 9-year one.
                checks["price_data"] = cov.status()
                checks["warning"] = MINIMUM_NOTE
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

    def _reference_names() -> tuple:
        """Company names and sectors, read once per request.

        The engine leaves `company_name` unset on every recommendation -- 0 of
        52 on a live run -- while the equity master holds 2,565 of them. The
        join belongs here rather than in the interface, which should never have
        to know that two stores exist.
        """
        try:
            store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
            master = store.read_equity_master()
            names = (dict(zip(master["symbol"], master["company_name"]))
                     if not master.empty else {})
            smap = store.read_sector_map()
            sectors = (dict(zip(smap["symbol"], smap["sector"]))
                       if not smap.empty else {})
            return names, sectors
        except Exception:
            # A missing display name must not cost the whole screen. The view
            # falls back to the ticker.
            log.warning("reference names unavailable; falling back to tickers")
            return {}, {}

    @app.get("/analysis/{run_id}/view")
    def job_view(run_id: str) -> Dict[str, Any]:
        """The screen's data: curated, named, explained.

        `/results` stays exactly as it was and remains the research view's
        source. This endpoint is the only thing the interface binds to, so a
        change to the engine's payload shape cannot reach the interface
        without passing through the presentation layer first.
        """
        from .presentation import build_view

        job = jobs.get(run_id)
        if job is None:
            raise HTTPException(404, f"no job with id {run_id}")
        if job.state.value == "FAILED":
            raise HTTPException(409, {
                "message": "analysis did not complete; no results exist",
                "error": job.error, "run_id": run_id,
            })
        if job.result is None:
            raise HTTPException(409, f"job {run_id} is {job.state.value}; no results yet")

        names, sectors = _reference_names()
        admission = cfg.params.stage6_entry.admission
        return build_view(
            job.result,
            company_names=names,
            sectors=sectors,
            entry_rank=int(admission.entry_rank.value),
            exit_rank=int(admission.exit_rank.value),
        )

    def _ledger_rows():
        from .ledger import Ledger
        return Ledger(cfg.paths.ledger).iter_rows()

    @app.get("/history")
    def history(limit: int = 30) -> Dict[str, Any]:
        """Past runs and what moved between them.

        Read from the ledger, which has recorded every completed run all along
        -- date, names admitted, names monitored, regime and funnel. No new
        storage was added for this; the record already existed.
        """
        from .presentation import build_history
        from .presentation.clearmark import read_mark

        try:
            rows = _ledger_rows()
        except Exception as exc:
            log.warning("ledger unreadable", extra={"error": str(exc)})
            return {"days": [], "latest_changes": None,
                    "note": "The run history could not be read."}
        names, _ = _reference_names()
        return build_history(rows, limit=max(1, min(int(limit), 120)),
                             company_names=names,
                             since=read_mark(cfg.paths.ledger))

    @app.delete("/history")
    def clear_history() -> Dict[str, Any]:
        """Hide everything recorded so far, and start again from here.

        This sets a watermark rather than deleting rows. The ledger is the
        permanent record every run is written to -- `fail_run_if_unwritable`
        is true precisely because an unlogged run corrupts the
        deflated-Sharpe trial count -- so removing rows to clear a screen
        would silently invalidate the statistic that decides whether the
        strategy is distinguishable from luck. The screen clears; the record
        underneath stays whole, and the clear is reversible.
        """
        from .presentation.clearmark import set_mark

        stamp = set_mark(cfg.paths.ledger)
        log.info("history cleared", extra={"cleared_at": stamp})
        return {
            "cleared_at": stamp,
            "message": (
                "History cleared. Runs from here on will be recorded and will "
                "appear on this page."
            ),
            "note": (
                "The underlying research ledger is preserved -- the "
                "deflated-Sharpe trial count depends on it, so it is hidden "
                "from this screen rather than deleted."
            ),
        }

    @app.get("/history/names")
    def history_names(limit: int = 60) -> Dict[str, Any]:
        """Every name the engine has surfaced, collapsed to one row each.

        The date-by-date table repeated the same handful of tickers down the
        page and left the reader to deduplicate by eye. This is the same
        record keyed by name.
        """
        import datetime as _dt

        from .presentation.clearmark import read_mark
        from .presentation.history import distinct_names
        from .presentation.outcome import outcomes_for

        rows = distinct_names(_ledger_rows(), since=read_mark(cfg.paths.ledger))
        if not rows:
            return {"names": [], "note": "No runs have been recorded yet."}
        rows = rows[: max(1, min(int(limit), 200))]

        names, _ = _reference_names()
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        earliest = min(_dt.date.fromisoformat(r["first_seen"]) for r in rows)
        prices = store.read_prices(
            symbols=[r["ticker"] for r in rows], start=earliest,
            columns=["date", "symbol", "high", "low", "close"],
        )

        out = []
        for row in rows:
            company = names.get(row["ticker"]) or row["ticker"]
            for suffix in (" Limited", " Ltd.", " Ltd", " LIMITED"):
                if company.endswith(suffix):
                    company = company[: -len(suffix)].strip()
                    break
            # Measured from the FIRST time it was surfaced -- that is when a
            # reader could first have acted on it.
            outcome = outcomes_for(
                [{"ticker": row["ticker"], "signal_price": row["first_price"]}],
                _dt.date.fromisoformat(row["first_seen"]), prices,
            )[0]
            # Same basis rule as the other two views: the row's own price is
            # the recorded one, the outcome's is re-based onto today's prices,
            # and shipping both lets the interface render either.
            out.append({**row, "first_price": outcome.signal_price,
                        "company": company, "outcome": outcome.__dict__})
        return {"names": out, "note": ""}

    _resolved: Dict[str, Any] = {"key": None, "rows": None, "open": None}

    def _resolved_rows():
        """Resolution is the expensive part and it is the same work for the
        History page and for any one name, so both share it -- opening a
        stock after History should not pay for it a second time."""
        from . import outcomes as _out
        from . import performance as _perf
        from .stages._cfg import iv
        key = _ledger_fingerprint()
        if _resolved["key"] != key:
            store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
            led = Path(cfg.paths.ledger)
            path = led / "outcomes.jsonl"
            _out.resolve_pending(store, led, path, cfg)
            rows = _apply_clear_mark(_out.load_outcomes(path))
            horizon = int(iv(cfg.params.stage4_core_score.model_horizon_sessions))
            op = _perf.open_positions(_ledger_after_clear(), rows,
                                      store, max_hold=horizon)
            _resolved.update(key=key, rows=rows, open=op)
        return _resolved["rows"], _resolved["open"]

    @app.get("/stock/{ticker}/calls")
    def stock_calls(ticker: str) -> Dict[str, Any]:
        """Every call on one name, from the same source the list uses.

        The old panel read the ledger while the list read resolved outcomes,
        so a name showing five calls opened a panel saying it had never
        reached a shortlist. One source now.
        """
        from . import outcomes as _out
        from . import performance as _perf
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        rows, op = _resolved_rows()
        names, _ = _reference_names()
        sym = str(ticker).upper()
        company = names.get(sym) or sym
        for suffix in (" Limited", " Ltd.", " Ltd", " LIMITED"):
            if company.endswith(suffix):
                company = company[: -len(suffix)].strip()
                break
        out = _perf.calls_for(sym, rows, store, open_rows=op.get("positions") or [])
        return {**out, "company": company}

    @app.get("/stock/{ticker}/history")
    def stock_history(ticker: str) -> Dict[str, Any]:
        """Every time this name reached the shortlist, and what followed.

        Answers the question the day view cannot: is this a name the engine
        keeps returning to, and how have those calls turned out?
        """
        import datetime as _dt

        from .presentation.clearmark import read_mark
        from .presentation.history import runs_for_ticker
        from .presentation.outcome import outcomes_for, summarise

        symbol = str(ticker).upper()
        runs = runs_for_ticker(_ledger_rows(), symbol,
                               since=read_mark(cfg.paths.ledger))
        names, _ = _reference_names()
        company = names.get(symbol) or symbol
        for suffix in (" Limited", " Ltd.", " Ltd", " LIMITED"):
            if company.endswith(suffix):
                company = company[: -len(suffix)].strip()
                break

        if not runs:
            return {"ticker": symbol, "company": company, "runs": [],
                    "summary": {"tracked": 0,
                                "text": "This name has not reached a shortlist."}}

        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        earliest = min(_dt.date.fromisoformat(r["date"]) for r in runs)
        prices = store.read_prices(
            symbols=[symbol], start=earliest,
            columns=["date", "symbol", "high", "low", "close"],
        )

        merged, outcomes = [], []
        for run in runs:
            out = outcomes_for([{**run, "ticker": symbol}],
                               _dt.date.fromisoformat(run["date"]), prices)[0]
            outcomes.append(out)
            merged.append({**_in_outcome_basis(run, out), "outcome": out.__dict__})

        latest = None
        if prices is not None and not prices.empty:
            latest = float(prices.sort_values("date")["close"].iloc[-1])

        return {
            "ticker": symbol,
            "company": company,
            "last_price": latest,
            "times_flagged": len(runs),
            "runs": merged,
            "summary": summarise(outcomes),
            "disclaimer": (
                "Each call is followed from the session after it was made, at "
                "the levels recorded that day. A record, not a backtest."
            ),
        }

    @app.get("/history/{date}")
    def history_day(date: str) -> Dict[str, Any]:
        """One past run, and what the market did afterwards.

        The levels are the ones written down that day. Nothing is re-derived,
        so this is a record of what the engine said and what followed -- not a
        backtest, and the response says so.
        """
        import datetime as _dt

        from .presentation.clearmark import read_mark
        from .presentation.history import load_days, slate_picks
        from .presentation.outcome import outcomes_for, summarise

        try:
            as_of = _dt.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, f"{date!r} is not a date")

        days = [d for d in load_days(_ledger_rows(), limit=400,
                                     since=read_mark(cfg.paths.ledger))
                if d.date == date]
        if not days:
            raise HTTPException(404, f"no run recorded for {date}")
        day = days[0]
        picks = slate_picks(day)
        if not picks:
            return {"date": date, "regime": day.regime, "picks": [],
                    "summary": {"tracked": 0,
                                "text": "This run produced no shortlist."},
                    "is_backtest": False}

        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        prices = store.read_prices(
            symbols=[p["ticker"] for p in picks], start=as_of,
            columns=["date", "symbol", "high", "low", "close"],
        )
        outcomes = outcomes_for(picks, as_of, prices)
        names, _ = _reference_names()

        def company(ticker: str) -> str:
            full = names.get(ticker) or ticker
            for suffix in (" Limited", " Ltd.", " Ltd", " LIMITED"):
                if full.endswith(suffix):
                    return full[: -len(suffix)].strip()
            return full

        merged = []
        for pick, out in zip(picks, outcomes):
            merged.append({**_in_outcome_basis(pick, out),
                           "company": company(pick["ticker"]),
                           "outcome": out.__dict__})
        return {
            "date": date,
            "regime": day.regime,
            "allows_new_positions": day.allows_new_positions,
            "picks": merged,
            "summary": summarise(outcomes),
            "disclaimer": (
                "What the engine said on this date, followed forward at the "
                "levels it recorded. No position sizing, no costs and no "
                "portfolio -- this is a record, not a backtest."
            ),
        }

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

        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        have = len(store.price_sessions())
        cov = assess(cfg, have)
        # Aim at the depth the shipped coefficients were validated on, not at
        # the minimum that makes the fit consent to run. The model refits from
        # stored history every analysis, so the store IS the training set.
        need = cov.validated_target

        chunk = int(getattr(cfg.params.api, "bootstrap_chunk_sessions", 0) or 90)
        target = min(have + chunk, need) if have else min(chunk, need)

        progress(0, f"{have} of {need} sessions. Fetching up to {target}...")
        result = DataIngestor(cfg).run(
            options=IngestOptions(
                history_sessions=target,
                include_secondary_prices=False,
                include_delivery=True,
            )
        )
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        sessions = store.price_sessions()
        after = assess(cfg, len(sessions))
        done = after.matches_validation
        progress(8, "complete" if done else
                 f"{len(sessions)}/{need} sessions - press again to continue")

        return {
            "sessions_in_store": len(sessions),
            "sessions_required": need,
            "complete": done,
            "model_will_fit": after.model_will_fit,
            "matches_validation": after.matches_validation,
            "status": after.status(),
            "sessions_fetched_this_run": result.sessions_fetched,
            "latest_session": sessions[-1].isoformat() if sessions else None,
            "universe_size": len(result.universe.symbols),
            "next_step": (
                "ready to analyse at full validated depth"
                if done else
                "press BUILD DATA STORE again to fetch the next chunk"
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

    @app.get("/today")
    def today() -> Dict[str, Any]:
        """The newest completed run, whichever process produced it.

        The interface used to build Today from the JOB QUEUE, so a run started
        by the nightly cron -- a separate process that never touches the API --
        left the screen asking for a scan of a market already scanned. This
        reads the persisted run payload instead, so cron, the CLI and the
        button all land on the same screen, and a restarted API still has it.
        """
        from . import rundetail

        payload = rundetail.load_latest(cfg)
        if payload is None:
            return {"view": None,
                    "note": "No completed run has been recorded yet."}
        names, sectors = _reference_names()
        admission = cfg.params.stage6_entry.admission
        from .presentation import build_view
        return {
            "view": build_view(payload, company_names=names, sectors=sectors,
                               entry_rank=int(admission.entry_rank.value),
                               exit_rank=int(admission.exit_rank.value)),
            "run_id": payload.get("run_id"),
            "as_of_date": payload.get("as_of_date"),
            "generated_at": payload.get("generated_at"),
            "note": "",
        }

    # =====================================================================
    # Operator actions that used to require the CLI
    # =====================================================================
    @app.post("/admin/ingest")
    def admin_ingest() -> Dict[str, Any]:
        """Refresh the store. The analysis does not fetch data, so a store two
        sessions behind halts every run at Stage 1 -- and clearing that needed
        a terminal."""
        def _runner(progress) -> Dict[str, Any]:
            from .data.ingest import DataIngestor, IngestOptions

            progress(0, "Refreshing market data")
            # The same defaults the nightly CLI uses: fetch the new session and
            # advance the backfill one chunk, so a store below the validated
            # depth climbs on its own instead of waiting for a person.
            sessions = None
            try:
                from .data.coverage import assess
                have = len(DataStore(cfg.paths.curated,
                                     cfg.paths.snapshots).price_sessions())
                cov = assess(cfg, have)
                if have and have < cov.validated_target:
                    chunk = int(getattr(cfg.params.api,
                                        "bootstrap_chunk_sessions", 0) or 90)
                    sessions = min(have + chunk, cov.validated_target)
            except Exception:
                sessions = None
            with DataIngestor(cfg) as ingestor:
                res = ingestor.run(options=IngestOptions(history_sessions=sessions))
            m = res.manifest
            progress(1, "Market data refreshed")
            return {
                "kind": "ingest",
                "as_of_date": str(m.as_of_date),
                "sessions_fetched": res.sessions_fetched,
                "calendar_sessions": m.calendar_sessions_available,
                "last_session": str(m.calendar_last_session),
                "feeds": {k: v.status.value for k, v in sorted(m.feeds.items())},
            }

        job = jobs.start(kind="ingest", runner=_runner)
        return {**job.to_dict(), "already_running": job.state.value == "RUNNING"}

    @app.get("/admin/model")
    def admin_model() -> Dict[str, Any]:
        """What the model currently prices, and when it was fitted."""
        import json as _json

        path = Path(cfg.paths.curated) / "crosssec_model.json"
        if not path.is_file():
            return {"fitted": False,
                    "note": "No model has been fitted yet. The next run fits one."}
        try:
            blob = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"fitted": False, "note": "The model cache is unreadable."}
        coef = blob.get("coef") or {}
        t = blob.get("fm_t_stat") or {}
        themes = [{"theme": k.removesuffix("_f"), "coefficient": v,
                   "t_stat": t.get(k),
                   "priced": abs(float(v)) > 1e-12} for k, v in coef.items()]
        themes.sort(key=lambda r: -abs(float(r["coefficient"])))
        return {
            "fitted": True,
            "fitted_for": blob.get("fitted_for"),
            "train_end": blob.get("train_end"),
            "n_train": blob.get("n_train"),
            "estimator": blob.get("estimator"),
            "cross_sections": blob.get("fm_n_dates"),
            "themes": themes,
            "priced": sum(1 for r in themes if r["priced"]),
            "note": ("A theme at exactly zero was not consulted and found "
                     "neutral -- the significance floor removed it."),
        }

    @app.post("/admin/refit")
    def admin_refit() -> Dict[str, Any]:
        """Retire the cached coefficients so the next run fits fresh ones.

        The cache is archived, never deleted: a refit that turns out badly has
        to be recoverable, and a full retrain would reproduce whatever caused
        it. The refit itself happens on the next analysis, which is where the
        promotion gate can review it.
        """
        from .features import crossmodel as cm

        active = jobs.active_job()
        if active is not None:
            raise HTTPException(409, "A job is running. Wait for it to finish.")
        path = Path(cfg.paths.curated) / "crosssec_model.json"
        if not path.is_file():
            return {"retired": False,
                    "message": "There was no cached model; the next run fits one."}
        archived = cm.archive_cache(path)
        path.unlink()
        log.info("model cache retired", extra={"archived": archived})
        return {
            "retired": True,
            "archived_to": archived,
            "message": ("The cached model was archived and retired. The next "
                        "scan fits fresh coefficients, which takes several "
                        "minutes rather than seconds."),
        }

    @app.post("/admin/resolve-outcomes")
    def admin_resolve() -> Dict[str, Any]:
        """Score every signal whose holding window has fully elapsed."""
        from . import outcomes as _out

        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        led = Path(cfg.paths.ledger)
        counts = _out.resolve_pending(store, led, led / "outcomes.jsonl", cfg)
        _resolved["key"] = None          # force the shared cache to rebuild
        return {**counts, "exit_model": _out.EXIT_MODEL,
                "message": (f"{counts.get('resolved', 0)} resolved, "
                            f"{counts.get('still_open', 0)} still running.")}

    @app.get("/admin/forward")
    def admin_forward_state() -> Dict[str, Any]:
        """Whether the forward test is registered, current, and still valid."""
        import datetime as _dt

        from .validation import forward

        led = Path(cfg.paths.ledger)
        reg = forward.load_registration(led)
        live = str(getattr(cfg, "version", "") or "")
        if reg is None:
            return {"registered": False, "config_version": live,
                    "note": ("No forward test is registered. Registering opens "
                             "an 18-month window against the current "
                             "configuration.")}
        prog = forward.progress(led, list(_ledger_rows()), today=_dt.date.today())
        return {
            "registered": True,
            "started_on": reg.started_on,
            "registered_config": reg.config_version,
            "config_version": live,
            "config_matches": reg.config_version == live,
            "hash_intact": forward.verify(led),
            "sessions_elapsed": prog.sessions_elapsed if prog else 0,
            "sessions_target": reg.target_sessions,
            "broken": prog.broken if prog else [],
            "summary": prog.summary() if prog else "",
        }

    @app.post("/admin/forward/register")
    def admin_forward_register(body: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
        """Open a new forward-test window against the CURRENT configuration.

        Overwriting discards the observations collected so far, which is the
        one thing a pre-registration exists to prevent -- so the confirmation
        is required by the server, not only by the interface.
        """
        import subprocess

        from .validation import forward

        led = Path(cfg.paths.ledger)
        existing = forward.load_registration(led)
        if existing is not None and (body or {}).get("confirm") != "RESTART":
            raise HTTPException(400, {
                "message": ('A forward test is already registered. Send '
                            '{"confirm": "RESTART"} to discard it and start a '
                            'new window.'),
                "started_on": existing.started_on,
                "registered_config": existing.config_version,
            })
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip() or "unknown"
        except Exception:
            commit = "unknown"
        reg = forward.register(led, config_version=str(cfg.version),
                               engine_version=ENGINE_VERSION,
                               git_commit=commit, overwrite=True)
        log.info("forward test registered", extra={"config": reg.config_version})
        return {"registered": True, "started_on": reg.started_on,
                "config_version": reg.config_version,
                "fingerprint": reg.fingerprint(),
                "message": ("Forward test opened. Any change to the "
                            "configuration from here invalidates the window.")}

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
    @app.get("/outcomes")
    def outcomes_summary() -> Dict[str, Any]:
        """Resolved signals and what the market did with them.

        Resolution runs on read so the record catches up without a scheduler.
        Only signals whose full holding window has elapsed are scored.
        """
        from . import outcomes as _out
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        led = Path(cfg.paths.ledger)
        path = led / "outcomes.jsonl"
        counts = _out.resolve_pending(store, led, path, cfg)
        rows = _out.load_outcomes(path)
        return {
            "resolution": counts,
            "summary": _out.summarise(rows),
            "calibration": _out.calibration(rows),
            "note": (
                "composite_score is a cross-sectional rank, not a probability. "
                "The calibration table tests only whether a higher rank wins "
                "more often."
            ),
        }

    # Resolution walks the whole ledger and reads prices for every name it
    # finds, which on a deep ledger is seconds -- paid on every open of the
    # History page even when nothing had changed since the last one. The
    # inputs are all files, so their mtimes say when the answer is stale.
    _perf_cache: Dict[str, Any] = {"key": None, "value": None}

    def _ledger_fingerprint() -> str:
        led = Path(cfg.paths.ledger)
        parts = []
        for f in sorted(led.glob("*.jsonl")) + sorted(led.glob(".history-cleared")):
            try:
                st = f.stat()
                parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                continue
        return "|".join(parts)

    def _clear_cut():
        try:
            from .presentation.clearmark import read_mark
            mark = read_mark(cfg.paths.ledger)
        except Exception:
            return None
        return str(mark)[:10] if mark else None

    def _ledger_after_clear():
        """Ledger rows the clear did not hide.

        open_positions counts signals that have not closed, and it was handed
        every row ever written while the outcomes beside it were filtered.
        So a clear emptied the results and left the open count intact -- one
        run after a clear reported fourteen open calls it had not made.
        """
        rows = Ledger(cfg.paths.ledger).read_all()
        cut = _clear_cut()
        if not cut:
            return rows
        return [r for r in rows if str(r.get("date") or "")[:10] >= cut]

    def _apply_clear_mark(rows):
        """Drop resolved results issued before the last clear."""
        try:
            from .presentation.clearmark import read_mark
            mark = read_mark(cfg.paths.ledger)
        except Exception:
            return rows
        if not mark:
            return rows
        cut = str(mark)[:10]
        return [r for r in rows
                if str(r.get("signal_date") or "")[:10] >= cut]

    def _git_commit() -> str:
        """Best effort. A period without a commit is still a period; one that
        refused to open because git was unavailable would not be."""
        import subprocess
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cfg.paths.root),
                                 capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    def _measurement_state() -> Dict[str, Any]:
        from . import measurement as _m
        led = Path(cfg.paths.ledger)
        cv = str(getattr(cfg, "version", "") or "")
        cur = _m.active(led, config_version=cv)
        return {
            "active": cur.to_dict() if cur else None,
            "periods": [x.to_dict() for x in _m.periods(led)[:12]],
            "config_version": cv,
        }

    @app.get("/measurement")
    def measurement_state() -> Dict[str, Any]:
        return _measurement_state()

    @app.post("/admin/run-now")
    def run_now() -> Dict[str, Any]:
        """What the nightly job does, on demand: refresh the data, then rank.

        The only one of these controls anyone presses. Opening the app before
        the job has run and finding yesterday's answer with no way to move it
        forward is the case it exists for.
        """
        job = jobs.start(kind="bootstrap", runner=_bootstrap_runner)
        return {**job.to_dict(), "already_running": job.state.value == "RUNNING"}

    @app.get("/performance")
    def performance_report(period: str = "all") -> Dict[str, Any]:
        key = period + "@" + _ledger_fingerprint()
        if _perf_cache["key"] == key and _perf_cache["value"] is not None:
            return _perf_cache["value"]
        """Did following the shortlist beat not following it?

        Resolution runs on read, like /outcomes, so the scheduled job does not
        need to remember to do it and a missed night self-heals.
        """
        from . import outcomes as _out
        from . import performance as _perf
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        led = Path(cfg.paths.ledger)
        rows, _op = _resolved_rows()

        # Clearing history sets a watermark rather than deleting ledger rows,
        # because fail_run_if_unwritable exists precisely so the deflated-
        # Sharpe trial count cannot be corrupted by a missing run. The mark
        # therefore has to be applied HERE, where results are read -- the
        # outcomes file is derived and rebuilt on every request, so deleting
        # it would clear the screen only until the next one.

        # Scoping defaults to OFF. Keeping evidence from before a config
        # change out of evidence from after it matters for a t-statistic;
        # this endpoint feeds a record of what the calls did, which has no
        # such problem. Scoping it by default meant that turning the daily
        # run on opened a period and instantly emptied a history of 136
        # closed trades -- the isolation was real and the screen was wrong.
        from . import measurement as _m
        state = _measurement_state()
        window = None
        if period == "active" and state["active"]:
            window = _m.Period(**{k: v for k, v in state["active"].items()
                                  if k not in ("status", "open")})
        elif period not in ("active", "all"):
            window = next((x for x in _m.periods(led) if x.id == period), None)
        if window is not None:
            rows = [r for r in rows if window.covers(str(r.get("signal_date") or ""))]

        from .stages._cfg import iv
        horizon = int(iv(cfg.params.stage4_core_score.model_horizon_sessions))

        # The last session a signal could have been issued on and still have
        # had its whole window. Past it, only the fast movers have finished.
        # The store keeps a session calendar. Deriving one by reading every
        # price row instead cost 3.4s of a 3.8-million-row scan to learn 2,210
        # dates it already had.
        cutoff = None
        try:
            days = store.price_sessions()
            if len(days) > horizon:
                cutoff = str(days[-horizon])[:10]
        except Exception:
            cutoff = None
        rows, partial = _perf.split_cohorts(rows, cutoff)
        bench = str(cfg.params.stage2_regime.benchmark_index.value)
        payload = {
            "headline": _perf.performance(rows, store, horizon=horizon,
                                          benchmark=bench),
            "by_ticker": _perf.by_ticker(rows, store, benchmark=bench),
            "curve": _perf.equity_curve(rows, store, benchmark=bench),
            "calibration": _out.calibration(rows),
            "cohort_cutoff": cutoff,
            "recent": _perf.recent_activity(partial),
            # What holds actually last, from the record. The card's
            # configured range is identical on every name and off by an
            # order of magnitude.
            "holding": _perf.holding_profile(rows),
            "measurement": state,
            "scope": ("period" if window is not None else "all"),
            # Kept apart from every figure above: a mark is not an outcome.
            "open": _op,
        }
        _perf_cache["key"], _perf_cache["value"] = key, payload
        return payload

    # ------------------------------------------------------------------
    # Operator actions
    # ------------------------------------------------------------------

    @app.get("/operations")
    def operations_state() -> Dict[str, Any]:
        from . import operations as _ops
        led = Path(cfg.paths.ledger)
        return {
            "schedule": {
                **_ops.pause_state(led).to_dict(),
                "cron": "20:30 IST, weekdays",
                "note": ("Pausing does not stop cron -- editing /etc/cron.d "
                         "needs root. The job still wakes and declines, and "
                         "the decline is recorded."),
            },
            "log": _ops.operations_log(led, limit=25),
        }

    @app.post("/operations/pause")
    def operations_pause(body: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
        """Turning the daily run off also closes the measurement period.

        They were two switches for one idea. Nobody wants a measurement
        period that keeps running over days the engine did not record, and
        nobody wants the engine recording into no period at all -- so there
        is one switch and it moves both.
        """
        from . import operations as _ops
        from . import measurement as _m
        reason = (body or {}).get("reason", "")
        st = _ops.pause(Path(cfg.paths.ledger), reason)
        _m.stop(Path(cfg.paths.ledger))
        return {**st.to_dict(), "measurement": None}

    @app.post("/operations/resume")
    def operations_resume() -> Dict[str, Any]:
        """Turning it on opens a period, so what follows is measured.

        If the config has changed since the last period this is also the
        re-registration: a new period, a new fingerprint, and the previous
        evidence kept separate from what comes next.
        """
        from . import operations as _ops
        from . import measurement as _m
        from .version import ENGINE_VERSION as _ver
        led = Path(cfg.paths.ledger)
        st = _ops.resume(led)
        cur = _m.active(led, config_version=str(getattr(cfg, "version", "") or ""))
        if cur is None or cur.status == "DRIFTED":
            cur = _m.start(led,
                           config_version=str(getattr(cfg, "version", "") or ""),
                           engine_version=str(_ver), git_commit=_git_commit())
        return {**st.to_dict(), "measurement": cur.to_dict()}

    @app.post("/admin/reset/market-data")
    def reset_market_data() -> Dict[str, Any]:
        """Clear the price store so the build can run again. The record of
        what the engine said is kept -- that cannot be re-fetched."""
        from . import operations as _ops
        active = jobs.active_job()
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail="A job is running. Cancel it before resetting the store.")
        return _ops.reset_market_data(cfg.paths)

    @app.post("/admin/reset/everything")
    def reset_everything(body: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
        """Market data AND the entire record. The confirmation phrase is
        required by the server, not only by the interface -- a destructive
        endpoint that trusts the caller to have asked is not guarded."""
        from . import operations as _ops
        if (body or {}).get("confirm") != "ERASE":
            raise HTTPException(
                status_code=400,
                detail='Send {"confirm": "ERASE"} to erase the record.')
        active = jobs.active_job()
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail="A job is running. Cancel it before erasing.")
        return _ops.erase_everything(cfg.paths)

    @app.get("/ledger")
    def ledger(limit: int = 50) -> Dict[str, Any]:
        led = Ledger(cfg.paths.ledger)
        rows = led.read_all()[-limit:]
        # `trial_id` is one uuid per RUN, so this counts executions, not the
        # configurations the Deflated Sharpe charges for. It was labelled
        # "trials", which reads as the DSR input and is off by two orders of
        # magnitude -- 1,929 runs against a research registry of 40. The DSR
        # reads `TrialRegistry.effective_trials`; this number is operational.
        return {"count": led.count(), "runs_recorded": led.trial_count(),
                "runs": rows}

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

    @app.on_event("startup")
    def _warm_on_start() -> None:
        """Resolve outcomes once, off the request path, after a restart.

        The performance cache lives in this process, so a deploy or a systemd
        restart empties it and hands the next visitor the full resolution.
        The nightly job warms it after each run; this covers every other way
        the process comes back.

        On a daemon thread, so a slow or failing warm delays nothing and
        cannot keep the process alive. Failures are swallowed deliberately: a
        cold cache is slow, not wrong, and the next read rebuilds it.
        """
        import threading

        def _run() -> None:
            try:
                # The whole payload, not just the resolution behind it.
                # Warming only the resolution left the first open at 1.0s
                # instead of 3ms, because the endpoint still had to build
                # the response it caches separately.
                performance_report()
                log.info("performance cache warmed at startup")
            except Exception as exc:            # noqa: BLE001 - never fatal
                log.info("startup warm skipped", extra={"reason": str(exc)})

        threading.Thread(target=_run, name="prosignal-warm", daemon=True).start()

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

