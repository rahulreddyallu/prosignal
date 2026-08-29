"""The full run payload: shaped once, persisted, and readable by anything.

WHY THIS EXISTS. The interface builds Today from `GET /analysis`, which lists
the API's own JOB QUEUE. The nightly cron does not go through the API -- it runs
`prosignal.cli analyse run` in a separate process -- so no job row is ever
created, `/analysis` comes back with nothing completed, and the screen asks the
operator to scan a market that was scanned hours ago.

`/analysis/{id}/view` cannot rescue that either: it needs `jobs.get(run_id)` and
the result held in the job row. The ledger keeps a SUMMARY of every run, which
is what History renders, but it carries no `factor_detail` -- so the evidence
panel cannot be rebuilt from it and Today would render blank cards.

The engine already anticipated this. `ledger.write_run_detail` and
`ledger.run_detail_subdir` have been declared in parameters.yaml since v1,
carried on the RESERVED list with the note "per-run detail files are not
written". This implements them.

Every completed run writes its full payload here, whoever started it, so the
screen shows the newest run rather than the newest JOB -- and it survives an
API restart, which the in-memory job result does not.

The shaping functions moved here from `api.py` unchanged. They are pure -- they
touch no config, no job store and no request -- and `api` imports `pipeline`,
so leaving them there meant the engine could not persist its own output without
a circular import.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.logging import get_logger

__all__ = ["shape", "save", "load_latest", "load", "available_runs"]

log = get_logger(__name__)

#: Runs older than this are pruned on write. The ledger keeps the permanent
#: record; these are display payloads and the newest is the one that matters.
KEEP_RUNS = 60


def _dir(cfg) -> Path:
    sub = str(getattr(cfg.params.ledger, "run_detail_subdir", "runs"))
    sub = getattr(sub, "value", sub)
    return Path(cfg.paths.ledger) / str(sub)


def save(run, cfg) -> Optional[Path]:
    """Persist one completed run. Never raises -- a display cache that fails
    must not fail the run that produced it, which IS recorded, in the ledger."""
    enabled = getattr(cfg.params.ledger, "write_run_detail", True)
    if not bool(getattr(enabled, "value", enabled)):
        return None
    try:
        root = _dir(cfg)
        root.mkdir(parents=True, exist_ok=True)
        payload = shape(run)
        # `as_of` first so a listing sorts by MARKET DATE, then the moment the
        # run was generated so it sorts chronologically WITHIN a date, then the
        # run id for uniqueness.
        #
        # The run id alone is not an ordering. It is random hex, so for two
        # runs on one date -- which the ledger shows is the norm, not the
        # exception -- a lexical sort picks an arbitrary one. It served the
        # OLDEST run of three and the screen showed a stale payload while
        # looking entirely normal.
        stamp = str(payload.get("generated_at") or "")[:19].replace(":", "").replace("-", "").replace("T", "-")
        name = f"{payload['as_of_date']}_{stamp}_{payload['run_id']}.json"
        path = root / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(str(tmp), str(path))
        for old in sorted(root.glob("*.json"))[:-KEEP_RUNS]:
            try:
                old.unlink()
            except OSError:
                pass
        return path
    except Exception as exc:
        log.warning("run detail not written", extra={"error": str(exc)})
        return None


def available_runs(cfg) -> List[Path]:
    """Newest last. Sorted by filename, which starts with the market date."""
    try:
        return sorted(_dir(cfg).glob("*.json"))
    except OSError:
        return []


def load(cfg, run_id: str) -> Optional[Dict[str, Any]]:
    for path in reversed(available_runs(cfg)):
        if path.stem.endswith(run_id):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def load_latest(cfg) -> Optional[Dict[str, Any]]:
    """The newest run on disk, whichever process produced it.

    Ordered by MARKET DATE, then by when the run was generated -- not by file
    mtime, so a backfill written today for a past date cannot displace today's
    screen, and not by run id, which is random hex and orders nothing.
    """
    for path in reversed(available_runs(cfg)):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue          # a truncated file is skipped, not fatal
    return None


def shape(run) -> Dict[str, Any]:
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
            # The measurements the labels are read off. Stage 2 computes all
            # of these and nothing was serialising them, so "Uptrend" reached
            # the screen as an assertion with no way to check it.
            "trend_slope_annualised": r.trend_slope_annualised,
            "index_vs_fast_ma_pct": r.index_vs_fast_ma_pct,
            "index_vs_slow_ma_pct": r.index_vs_slow_ma_pct,
            "vix_level": r.vix_level,
            "vix_percentile": r.vix_percentile,
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
        "recommendations": [card(x) for x in o.recommendations],
        "watchlist": [card(x) for x in o.watchlist],
        # The screen the RUN decided, not one the reader re-derives. Serialising
        # only the two lists left every consumer to reconstruct the slate, and
        # they did not agree with each other.
        "slate": [e.model_dump(mode="json") for e in o.slate],
        "slate_departures": list(o.slate_departures),
        "new_entries_blocked": o.new_entries_blocked,
        "entry_clock": dict(o.entry_clock or {}),
        "position_directives": list(o.position_directives),
        "data_quality_flags": o.data_quality_flags,
        "stage_timings_ms": o.stage_timings_ms,
        "disclaimer": o.disclaimer,
        "probability_note": (
            "Probability estimate unavailable: no out-of-sample calibration "
            "exists. The score is a RANK within today's eligible universe, not "
            "a likelihood of profit."
        ),
    }


def card(rec) -> Dict[str, Any]:
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
                # THE FOURTH COLUMN OF THE PER-STOCK TABLE. `standardised x
                # weight` is not always the term that was used -- the v2
                # composite renormalises its weights over the factors a name
                # actually has -- so the contribution is serialised rather than
                # left for the client to multiply and get subtly wrong.
                "contribution": getattr(f, "contribution", None),
                "available": f.available,
                # What the theme is made of. One coefficient is fitted per
                # theme over the average of its members' ranks, so "lottery
                # -1.81 sd" is a summary of these -- and without them the
                # reader cannot see WHICH moment moved, or check the theme
                # against the measurements it came from.
                "members": [
                    {"name": m.name, "rank": m.rank,
                     "available": m.available, "description": m.description}
                    for m in (getattr(f, "members", None) or [])
                ],
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
        # Both, and they are different things. `rank` is the display position
        # after Stage 5 penalties re-sort the survivors; `model_rank` is where
        # the model put the name, and it is the only input to admission. The
        # table numbers by model_rank -- serialising only `rank` rendered the
        # column as "undefined".
        "rank": rec.rank,
        "model_rank": rec.model_rank,
        "risk_category": rec.position_risk_category.value if rec.position_risk_category else None,
        "holding_period": rec.expected_holding_period,
        # THE PLAN, serialised with the card. The History page has to be able to
        # show what a trade was issued AS -- its cadence, its planned hold and
        # the expectation stamped on it -- because that is the only thing a
        # resolved outcome can be scored against besides the market. Reading it
        # from today's config instead would describe a strategy the trade was
        # never part of.
        "trade_plan": (rec.trade_plan.model_dump(mode="json")
                       if rec.trade_plan else None),
        "why": rec.why_this_signal_exists,
        "against": rec.false_signal_flagged,
        "cleared": rec.false_signal_cleared,
        "not_testable": rec.false_signal_not_testable,
        "exits": rec.sell_conditions,
        "cost_note": rec.cost_note,
        "research_basis": rec.research_basis,
        "warning": rec.unvalidated_parameter_warning,
    }
