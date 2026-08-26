"""Resolve past signals against what the market actually did.

The ledger records every decision but nothing scores it, so a signal issued
today produces no evidence tomorrow. Without resolved outcomes there is no
calibration, no live validation and no paper-trading record -- and every day the
engine runs without one is a day of evidence permanently lost.

Execution follows `backtest._simulate`: entry at the next session's open, and a
bar touching both stop and target counts as the stop, since daily bars cannot
order intraday events.

THE ENGINE'S OWN EXIT. A position also closes when the engine stops holding it.
Stage 6 admits at rank <= entry_rank and holds while the name stays inside
exit_rank, so the book is the exit rule; the stop and the target are the two
ways a position can end EARLY. That rule was missing here, and the record it
produced described a strategy the engine does not run.

Measured on the recorded record before this was added: the simulation held past
the engine's own exit in 94% of trades, by a median of 14 sessions. Median
simulated hold was 15 sessions against a book that had let the name go after 1.
Every figure the History page showed -- win rate, average return, target and
stop counts -- was computed over those phantom sessions.

The book is read from the ledger, which records `signals_generated` for every
run. A session with no recorded run carries no information about the book and
is not treated as an exit; a session that ran and did not name the ticker is.
The exit fills at the NEXT session's open, because the engine decides at the
close and that is the first price the decision could reach.

MODEL VERSIONING. Outcomes carry the `exit_model` they were resolved under and
`load_outcomes` serves only the current one. Changing the exit rule without
that would leave one file holding two strategies' results and average them
together, which is the failure this module exists to detect elsewhere.

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
from .indicators.circuit import band_state, is_untradeable

__all__ = ["Outcome", "EXIT_MODEL", "resolve_pending", "load_outcomes",
           "summarise", "calibration"]

log = get_logger(__name__)

#: Which exit rule produced a row. Bump this whenever the rule changes, so that
#: results from two different strategies can never be pooled into one average.
#:
#:   stop-target-v1  stop, target or the holding-period limit. No book exit,
#:                   so a position ran on after the engine had closed it.
#:   book-band-v2    the above, plus the engine's own exit: the position ends
#:                   when the run stops naming it in `signals_generated`.
#:   book-band-circuit-v3
#:                   the above, plus two execution facts the backtest already
#:                   modelled and this did not: a stop is filled at the OPEN
#:                   when the bar gapped through it, and it is not filled at all
#:                   on a session locked at the price band.
EXIT_MODEL = "book-band-circuit-v3"


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


def load_outcomes(path: Path, *, model: Optional[str] = EXIT_MODEL) -> List[Dict[str, Any]]:
    """Resolved outcomes for one exit model.

    Defaults to the current model. Rows written under an older rule stay in the
    file -- it is append-only and they are the record of what was believed --
    but they are not served, because averaging them with the current ones would
    report two strategies as one. Pass ``model=None`` to read everything.
    """
    rows = _read_jsonl(Path(path))
    if model is None:
        return rows
    # A row with no stamp predates versioning, which means stop-target-v1.
    return [r for r in rows if (r.get("exit_model") or "stop-target-v1") == model]


def book_by_date(ledger_rows) -> Dict[dt.date, set]:
    """What the engine held at the close of each recorded session.

    The last run recorded for a date is the one that stands: a date is often
    run several times and only the final one reflects what the engine finally
    said. Dates with no run are absent rather than empty, and the difference
    matters -- an absent date says nothing about the book, while an empty one
    would say the engine held nothing.
    """
    latest: Dict[dt.date, tuple] = {}
    for row in ledger_rows:
        raw = row.get("date")
        if not raw or row.get("error"):
            continue
        try:
            when = raw if isinstance(raw, dt.date) else dt.date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        stamp = str(row.get("logged_at") or "")
        held = latest.get(when)
        if held is None or stamp >= held[0]:
            latest[when] = (stamp, set(row.get("signals_generated") or []))
    return {when: names for when, (_, names) in latest.items()}


def _closed_by_engine(book: Dict[dt.date, set], when: dt.date, ticker: str) -> bool:
    """Did the run at ``when``'s close stop holding ``ticker``?

    False when no run was recorded for that session. An absent run is missing
    information, not an exit, and treating it as one would close positions on
    days the engine simply was not run.
    """
    names = book.get(when)
    return names is not None and ticker not in names


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
    # Only rows resolved under the CURRENT exit model count as done. A row from
    # an older rule describes a different strategy, so it is re-resolved rather
    # than left to sit in the same averages as the new ones.
    existing = [r for r in _read_jsonl(outcomes_path)
                if (r.get("exit_model") or "stop-target-v1") == EXIT_MODEL]
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

    book = book_by_date(ledger_rows)
    tickers = sorted({p["rec"]["ticker"] for p in pending})
    earliest = min(pd.to_datetime(p["date"]).date() for p in pending)
    bars = store.read_prices(symbols=tickers, start=earliest, end=as_of)
    if bars.empty:
        return {"pending": len(pending), "resolved": 0, "still_open": len(pending)}
    bars[DATE] = pd.to_datetime(bars[DATE]).dt.normalize()
    by_symbol = {s: f.sort_values(DATE) for s, f in bars.groupby("symbol", observed=True)}

    written = 0
    still_open = 0
    refused = 0
    with outcomes_path.open("a", encoding="utf-8") as fh:
        for item in pending:
            res = _resolve_one(item, by_symbol, max_hold, costs, config, as_of,
                               book)
            if res is _REFUSED:
                refused += 1
                continue
            if res is None:
                still_open += 1
                continue
            fh.write(json.dumps(res, default=str) + "\n")
            written += 1
        fh.flush()
        import os
        os.fsync(fh.fileno())

    if refused:
        log.warning("outcomes refused: price basis unreconcilable",
                    extra={"refused": refused})
    log.info("outcomes resolved",
             extra={"resolved": written, "still_open": still_open,
                    "refused": refused})
    return {"pending": len(pending), "resolved": written,
            "still_open": still_open, "refused": refused}


#: How far the recorded close may drift from the stored one before the levels
#: are treated as belonging to a different price basis. Wide enough to absorb
#: rounding in the ledger, far tighter than any real corporate action.
_BASIS_TOLERANCE = 0.005

#: Returned when a trade cannot be priced honestly, as distinct from None,
#: which means the position is still running.
_REFUSED: Dict[str, Any] = {}

#: Beyond this the ratio is not an adjustment, it is a broken record. A 1:10
#: split gives 0.1 and a 10:1 reverse gives 10; anything outside is refused
#: rather than guessed at.
_BASIS_MIN, _BASIS_MAX = 0.001, 1000.0


def _basis_factor(rec, frame, signal_date) -> Optional[float]:
    """Put the recorded decision levels into the basis the store serves TODAY.

    The stop and the targets were computed at run time from the prices as they
    were adjusted THEN, and the ledger stores them as plain numbers. The store
    re-adjusts its whole history whenever a corporate action lands, so after a
    split those levels and these prices are no longer the same currency.

    Measured on the recorded record: BAJFINANCE was signalled on 2025-05-02 with
    a stop of 8195.05 against a close of 8862.50. A 4:1 bonus with a 2:1 face
    split landed on 2025-06-16, and the store now serves that same session at a
    close of 886.25. The stop sat ten times above every subsequent low, so the
    position "stopped out" on its first bar at 8195.05 against an entry of
    887.75 -- a loss recorded as **+823%**. Twenty-nine trades cleared +50% this
    way, and the mean return of the whole record read +53%.

    The factor is the recorded close over the stored close for the signal
    session, which IS the cumulative adjustment applied since the run. It needs
    no corporate-action lookup and it corrects any adjustment, including ones
    the actions table happens to be missing.

    Returns None when the basis cannot be established, and the caller refuses
    the trade rather than scoring it. A trade that cannot be priced honestly is
    not evidence.
    """
    recorded = rec.get("last_close")
    if recorded is None:
        return None
    try:
        recorded = float(recorded)
    except (TypeError, ValueError):
        return None
    if recorded <= 0:
        return None
    row = frame[frame[DATE] == pd.Timestamp(signal_date)]
    if row.empty:
        return None
    stored = float(row.iloc[0]["close"])
    if not np.isfinite(stored) or stored <= 0:
        return None
    factor = stored / recorded
    if not (_BASIS_MIN <= factor <= _BASIS_MAX):
        log.warning("refusing an outcome: price basis is not reconcilable",
                    extra={"ticker": rec.get("ticker"), "factor": factor})
        return None
    return factor


def _resolve_one(item, by_symbol, max_hold, costs, config, as_of,
                 book: Optional[Dict[dt.date, set]] = None) -> Optional[Dict[str, Any]]:
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

    # The levels were recorded in the price basis of the run date; these bars
    # are adjusted to today. Reconcile them or refuse the trade.
    factor = _basis_factor(rec, frame, signal_date)
    if factor is None:
        # Distinct from "still open". A refused trade is not running, it is
        # unscoreable, and collapsing the two would report a shrinking sample
        # as patience.
        return _REFUSED
    stop, t1 = float(rec["stop"]) * factor, float(rec["target_1"]) * factor
    t2 = float(rec["target_2"]) * factor if rec.get("target_2") else t1
    walk = future.iloc[1:].head(max_hold)

    exit_price = exit_date = None
    reason = "open"
    held = 0
    mae = mfe = 0.0
    book = book or {}
    # The last session whose close the engine had already acted on when this
    # bar opened. The book exit is decided at a close and fills at the next
    # open, so it is this date -- not the bar's own -- that is consulted.
    decided_on = entry_row[DATE].date()
    unfilled_stop_sessions = 0
    for held, (_, bar) in enumerate(walk.iterrows(), start=1):
        low, high = float(bar["low"]), float(bar["high"])
        mae = min(mae, (low - entry_price) / entry_price)
        mfe = max(mfe, (high - entry_price) / entry_price)

        # A bar locked at its price band offered exactly one price all session.
        # Filling a stop there records a trade that could not have happened --
        # the seller had no bid to hit. `backtest._simulate` has modelled this
        # since it was written; this record did not, so the live figures assumed
        # a fill the backtest knew was fiction.
        state = band_state(high, low, float(bar["close"]),
                           float(bar.get("prev_close", np.nan)),
                           float(bar.get("volume", np.nan)))
        if is_untradeable(state):
            if low <= stop:
                unfilled_stop_sessions += 1
            continue

        if low <= stop:
            # A gap-down opens below the stop, so the fill is the open, not the
            # stop. Filling at the stop credits a price that was never available
            # and flatters every stopped trade.
            bar_open = float(bar["open"]) if "open" in bar else float("nan")
            exit_price = min(bar_open, stop) if np.isfinite(bar_open) else stop
            reason = "stop_gap" if exit_price < stop else "stop"
        elif high >= t2:
            exit_price, reason = t2, "target_2"
        elif high >= t1:
            exit_price, reason = t1, "target_1"
        if exit_price is not None:
            exit_date = bar[DATE].date()
            break
        # The engine's own exit, checked only after the intraday levels. A stop
        # and a book exit can land on the same bar, and the stop happened
        # DURING the session while the book decision was taken at the previous
        # close -- but the stop is the worse outcome and daily bars cannot
        # order them, so the pessimistic reading stands.
        if _closed_by_engine(book, decided_on, ticker):
            exit_price = float(bar["open"])
            exit_date = bar[DATE].date()
            reason = "book_exit"
            break
        decided_on = bar[DATE].date()
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
        if unfilled_stop_sessions:
            # Breached on a session that could not be traded, then ran to the
            # time exit. Named distinctly so it stays out of the ordinary
            # time-exit population, where it would look like a trade that simply
            # never hit its stop.
            reason = "stop_unfilled_circuit"

    if unfilled_stop_sessions and reason != "stop_unfilled_circuit":
        # Filled eventually, but only after a locked session. The trade is real;
        # the name records that the exit is later than the stop implies.
        reason = f"{reason}_after_circuit"

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
        "exit_model": EXIT_MODEL,
        # 1.0 unless a corporate action re-based the store's prices after the
        # signal. Recorded so a surprising outcome can be checked against it.
        "price_basis_factor": factor,
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
