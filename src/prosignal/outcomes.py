"""Resolve past signals against what the market actually did.

The ledger records every decision but nothing scores it, so a signal issued
today produces no evidence tomorrow. Without resolved outcomes there is no
calibration, no live validation and no paper-trading record -- and every day the
engine runs without one is a day of evidence permanently lost.

Execution follows `backtest._simulate` exactly: entry at the next session's
open, and a bar touching both stop and target counts as the stop, since daily
bars cannot order intraday events. A second convention here would make live
results incomparable with the backtest.

Outcomes are appended to their own JSONL file rather than written back into the
ledger, which is append-only. Resolution is idempotent on ``(run_id, ticker)``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .core.logging import get_logger
from .costs import CostModel
from .data.store import DataStore
from .data.types import DATE

__all__ = ["Outcome", "resolve_pending", "load_outcomes", "summarise", "calibration"]

log = get_logger(__name__)


class Outcome(dict):
    """One resolved signal. A plain dict so it serialises without ceremony."""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read one JSONL file, or every ``runs-*.jsonl`` in a ledger directory.

    The ledger partitions by year, so the path it is configured with is a
    directory rather than a file.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("runs-*.jsonl"))
    elif path.is_file():
        files = [path]
    else:
        return []
    rows: List[Dict[str, Any]] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A crash mid-write can truncate the final line. Skip it
                # rather than lose every prior record.
                log.warning("skipping malformed line", extra={"file": f.name})
    return rows


def load_outcomes(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(Path(path))


def _pending(ledger_rows, resolved_keys, resolved_calls=frozenset()) -> List[Dict[str, Any]]:
    """Signals not yet scored, one entry per CALL rather than per run.

    A day can produce several ledger runs and each names the same tickers, so
    the same call arrived here once per run: 6,192 items over 19 tickers on
    a real ledger. Every one was resolved separately, wrote its own outcome
    row, and was then collapsed again downstream by performance.dedupe --
    which is also where the duplicate counts on the History page came from.

    One call is resolved once. `resolved_calls` carries (ticker, date) pairs
    already on file so the other runs of a resolved call are never retried,
    which is what stopped them being re-scanned on every single request.
    """
    out, seen = [], set()
    for row in ledger_rows:
        run_id = row.get("run_id")
        signals = set(row.get("signals_generated") or [])
        if not signals:
            continue
        for rec in row.get("stocks_scored") or []:
            ticker = rec.get("ticker")
            if ticker not in signals:
                continue
            if (run_id, ticker) in resolved_keys:
                continue
            call = (ticker, str(row.get("date") or "")[:10])
            if call in resolved_calls or call in seen:
                continue
            if rec.get("stop") is None or rec.get("target_1") is None:
                continue
            seen.add(call)
            out.append({"run_id": run_id, "date": row.get("date"), "rec": rec,
                        "config_version": row.get("config_version"),
                        "engine_version": row.get("engine_version")})
    return out


def resolve_pending(
    store: DataStore,
    ledger_path: Path,
    outcomes_path: Path,
    config,
    as_of: Optional[dt.date] = None,
) -> Dict[str, int]:
    """Score every signal whose horizon has fully elapsed.

    A signal is only resolved once the market has had the full holding window to
    act on it. Scoring early would bias results toward whatever the first few
    sessions happened to do.
    """
    ledger_path, outcomes_path = Path(ledger_path), Path(outcomes_path)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_rows = _read_jsonl(ledger_path)
    existing = _read_jsonl(outcomes_path)
    resolved_keys = {(r.get("run_id"), r.get("ticker")) for r in existing}
    resolved_calls = {(r.get("ticker"), str(r.get("signal_date") or "")[:10])
                      for r in existing}

    pending = _pending(ledger_rows, resolved_keys, resolved_calls)
    if not pending:
        return {"pending": 0, "resolved": 0, "still_open": 0}

    sessions = store.price_sessions()
    if not sessions:
        return {"pending": len(pending), "resolved": 0, "still_open": len(pending)}
    as_of = as_of or sessions[-1]
    max_hold = int(config.params.stage7_risk.holding_period.max_holding_sessions.value)
    costs = CostModel(config)

    tickers = sorted({p["rec"]["ticker"] for p in pending})
    earliest = min(pd.to_datetime(p["date"]).date() for p in pending)
    bars = store.read_prices(symbols=tickers, start=earliest, end=as_of)
    if bars.empty:
        return {"pending": len(pending), "resolved": 0, "still_open": len(pending)}
    bars[DATE] = pd.to_datetime(bars[DATE]).dt.normalize()
    by_symbol = {s: f.sort_values(DATE) for s, f in bars.groupby("symbol", observed=True)}

    written = 0
    still_open = 0
    with outcomes_path.open("a", encoding="utf-8") as fh:
        for item in pending:
            res = _resolve_one(item, by_symbol, max_hold, costs, config, as_of)
            if res is None:
                still_open += 1
                continue
            fh.write(json.dumps(res, default=str) + "\n")
            written += 1
        fh.flush()
        import os
        os.fsync(fh.fileno())

    log.info("outcomes resolved", extra={"resolved": written, "still_open": still_open})
    return {"pending": len(pending), "resolved": written, "still_open": still_open}


def _resolve_one(item, by_symbol, max_hold, costs, config, as_of) -> Optional[Dict[str, Any]]:
    rec = item["rec"]
    ticker = rec["ticker"]
    frame = by_symbol.get(ticker)
    if frame is None or frame.empty:
        return None

    signal_date = pd.to_datetime(item["date"]).normalize()
    future = frame[frame[DATE] > signal_date]
    if future.empty:
        return None

    entry_row = future.iloc[0]
    entry_price = float(entry_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    stop, t1 = float(rec["stop"]), float(rec["target_1"])
    t2 = float(rec["target_2"]) if rec.get("target_2") else t1
    walk = future.iloc[1:].head(max_hold)

    exit_price = exit_date = None
    reason = "open"
    held = 0
    mae = mfe = 0.0
    for held, (_, bar) in enumerate(walk.iterrows(), start=1):
        low, high = float(bar["low"]), float(bar["high"])
        mae = min(mae, (low - entry_price) / entry_price)
        mfe = max(mfe, (high - entry_price) / entry_price)
        if low <= stop:
            exit_price, reason = stop, "stop"
        elif high >= t2:
            exit_price, reason = t2, "target_2"
        elif high >= t1:
            exit_price, reason = t1, "target_1"
        if exit_price is not None:
            exit_date = bar[DATE].date()
            break
    if exit_price is None:
        # Nothing triggered. Only a FULLY elapsed window turns that into a
        # time exit -- a partially elapsed one is a position still running,
        # and marking it closed would report whatever the opening sessions
        # happened to do.
        #
        # A triggered stop or target is different and is handled above: that
        # outcome is final on the day it happened and waiting out the rest of
        # the window would hide a closed trade for months. It did. Average
        # hold is 18 sessions against a 63-session window, so most trades
        # finish long before the window does.
        if walk.empty or len(walk) < max_hold:
            return None
        last = walk.iloc[-1]
        exit_price, exit_date, reason = float(last["close"]), last[DATE].date(), "time_exit"
        held = len(walk)

    slot = float(config.params.capital.position_value_inr())
    qty = max(int(slot / entry_price), 1)
    gross = exit_price / entry_price - 1.0
    breakdown = costs.round_trip(entry_price, qty, exit_price=exit_price)
    cost_inr = float(getattr(breakdown, "total_inr", 0.0))
    net = gross - cost_inr / (entry_price * qty)

    return {
        "run_id": item["run_id"],
        "ticker": ticker,
        "signal_date": str(item["date"]),
        "entry_date": str(entry_row[DATE].date()),
        "entry_price": entry_price,
        "exit_date": str(exit_date),
        "exit_price": exit_price,
        "exit_reason": reason,
        "sessions_held": held,
        "gross_return": gross,
        "net_return": net,
        "cost_inr": cost_inr,
        "mae": mae,
        "mfe": mfe,
        "composite_score": rec.get("composite_score"),
        "percentile": rec.get("percentile"),
        "config_version": item.get("config_version"),
        "engine_version": item.get("engine_version"),
        "resolved_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def summarise(outcomes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Trade statistics on resolved signals. Returns counts alongside every
    ratio, because a win rate without a sample size is not a measurement."""
    rows = [o for o in outcomes if o.get("net_return") is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    net = np.array([float(o["net_return"]) for o in rows])
    wins, losses = net[net > 0], net[net <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    held = np.array([int(o.get("sessions_held") or 0) for o in rows])
    se = net.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "win_rate": float(len(wins) / n),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(net.mean()),
        "expectancy_se": float(se),
        "expectancy_t": float(net.mean() / se) if n > 1 and se else float("nan"),
        "profit_factor": float(gross_win / gross_loss) if gross_loss else float("inf"),
        "avg_hold_sessions": float(held.mean()),
        "worst": float(net.min()),
        "best": float(net.max()),
        "exit_mix": {r: sum(1 for o in rows if o.get("exit_reason") == r)
                     for r in sorted({o.get("exit_reason") for o in rows})},
    }


def calibration(outcomes: Iterable[Dict[str, Any]], buckets: int = 4) -> List[Dict[str, Any]]:
    """Realised win rate by score bucket.

    The engine emits a rank, not a probability, so this does not test a
    probability claim. It tests the weaker claim the rank does make: that a
    higher score should win more often than a lower one. If the buckets do not
    separate, the score is not carrying information about outcomes.
    """
    rows = [o for o in outcomes
            if o.get("composite_score") is not None and o.get("net_return") is not None]
    if len(rows) < buckets * 2:
        return []
    rows.sort(key=lambda o: float(o["composite_score"]))
    out = []
    size = len(rows) // buckets
    for i in range(buckets):
        lo = i * size
        hi = (i + 1) * size if i < buckets - 1 else len(rows)
        chunk = rows[lo:hi]
        net = np.array([float(o["net_return"]) for o in chunk])
        out.append({
            "bucket": i + 1,
            "score_low": float(chunk[0]["composite_score"]),
            "score_high": float(chunk[-1]["composite_score"]),
            "n": len(chunk),
            "win_rate": float((net > 0).mean()),
            "mean_net": float(net.mean()),
        })
    return out
