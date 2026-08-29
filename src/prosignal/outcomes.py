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

__all__ = ["Outcome", "EXIT_MODEL", "PRE_EPOCH", "resolve_pending",
           "load_outcomes", "epochs_present", "summarise_by_epoch",
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
#:   target-t2-v4    T1 STOPS BEING AN EXIT. This record took profit at
#:                   `target_1` (1.5R) while the model is fitted against
#:                   `target_2` (3.0R) through `exits.rules_from_config`, so the
#:                   thing measuring the engine and the thing training it were
#:                   different strategies -- the fourth definition of "how did
#:                   this trade end" that `features/exits.py` was written to
#:                   collapse and did not reach.
#:
#:                   Measured out-of-sample on 49 purged walk-forward dates,
#:                   crossing the label's target against the book's:
#:
#:                       label   book    gross    cost      NET   Sharpe
#:                       3.0R    3.0R   +0.90%   0.54%   +0.36%   +0.21
#:                       1.5R    1.5R   +0.54%   0.66%   -0.12%   -0.06
#:                       3.0R    1.5R   +0.32%   0.54%   -0.22%   -0.18  <- was
#:                       1.5R    3.0R   +1.82%   0.67%   +1.15%   +0.43
#:
#:                   Booking at 1.5R is worse under BOTH labels -- two
#:                   independent comparisons pointing the same way -- and the
#:                   combination this record actually used was the worst of the
#:                   four. 58% of the 716 rows written under v3 exited at T1, so
#:                   that is what the History page had been reporting.
#:
#:                   T1 is retained as a MILESTONE (`touched_t1`), because
#:                   whether a trade reached 1.5R before its stop is real
#:                   information about the trade -- it is just not the moment
#:                   the position ends.
#:   band-time-v5    THE TARGET AND THE INVALIDATION STOP BEING EXITS, and the
#:                   stop moves from a trading rule to a disaster floor.
#:
#:                   Measured on the production configuration, each exit removed
#:                   alone so the case does not rest on a bundle. 258-385 trades,
#:                   7.5 years, net of 40 bps, against the equal-weight eligible
#:                   universe:
#:
#:                     arm                              p_win   alpha   ShExc
#:                     floor only (SHIPPED)             0.578  +20.3%   1.12
#:                     + 3R profit target               0.578  +19.4%   1.08
#:                     + 1.5R profit target             0.571  +15.7%   0.93
#:                     + MA50 - 1.5 ATR invalidation    0.422   +6.0%   0.32
#:                     v4 geometry 2.5 ATR / 3R / 1.5   0.384   +3.1%   0.15
#:
#:                   Two thirds of the return comes from the 39% of positions
#:                   that reach the time limit: those win 69% of the time and
#:                   average +16.1% net, against +3.3% for the ones that leave on
#:                   the rank band. Every rule removed here was a rule that sold
#:                   part of that population early.
#:
#:                   THE VERSION BUMP IS THE POINT. A v4 row and a v5 row are two
#:                   different strategies and `load_outcomes` serves one model at
#:                   a time, so the History page cannot average a trade that was
#:                   sold at 3R against one that was held to the limit. The old
#:                   rows are kept and stay readable under their own model.
#:
#:                   `touched_t1` and a new `touched_t2` remain as MILESTONES.
#:                   Whether a trade reached 1.5R or 3R before it ended is real
#:                   information about the trade; it is just no longer the moment
#:                   the position ends.
EXIT_MODEL = "band-time-v5"


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


def _active_epoch_id(ledger_root: Optional[Path] = None) -> str:
    """The open epoch, or `"unversioned"`.

    Imported lazily and failing soft: outcome resolution must not stop because
    an epoch ledger is missing. An unstamped row is honest about being
    unstamped, which is what `PRE_EPOCH` exists to say.
    """
    try:
        from .validation.epoch import active

        if ledger_root is None:
            from .config.loader import get_config

            ledger_root = Path(get_config().paths.ledger)
        e = active(Path(ledger_root))
        return e.epoch_id if e is not None else PRE_EPOCH
    except Exception:
        return PRE_EPOCH


#: Rows written before epochs existed. They are a real record of what the
#: engine did; they are just not a record of what THIS engine does, because the
#: universe they were produced on is not the one it now trades.
PRE_EPOCH = "pre-epoch"


def load_outcomes(path: Path, *, model: Optional[str] = EXIT_MODEL,
                  epoch: Optional[str] = None,
                  ledger_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Resolved outcomes for one exit model AND one research epoch.

    Defaults to the current model and the open epoch. Rows written under an
    older rule or an older epoch stay in the file -- it is append-only and they
    are the record of what was believed -- but they are not served, because
    averaging them with the current ones reports two strategies as one.

    The exit-model partition was already here. The epoch partition is finding
    C3/C4: the recorded history predates the population and liquidity
    corrections and was produced on a different universe, so pooling it with
    anything current compares two books and calls the result one.

    Pass ``model=None`` and ``epoch="*"`` to read everything, which is what the
    per-epoch report does.
    """
    rows = _read_jsonl(Path(path))
    if model is not None:
        # A row with no stamp predates versioning, which means stop-target-v1.
        rows = [r for r in rows
                if (r.get("exit_model") or "stop-target-v1") == model]
    if epoch == "*":
        return rows
    want = epoch if epoch is not None else _active_epoch_id(ledger_root)
    return [r for r in rows if (r.get("epoch_id") or PRE_EPOCH) == want]


def epochs_present(path: Path) -> Dict[str, int]:
    """Every epoch the outcome record contains, and how many rows each has.

    The point of surfacing this is that the answer is currently lopsided: the
    whole operating history sits under `pre-epoch` and the current experiment
    has none. A record that showed only the current epoch would render as an
    empty page and read as "no trades yet" rather than as "the trades we have
    describe a different engine".
    """
    counts: Dict[str, int] = {}
    for r in _read_jsonl(Path(path)):
        key = str(r.get("epoch_id") or PRE_EPOCH)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
            # A STOP IS REQUIRED; A TARGET IS NOT, since v5 does not exit on
            # one. Requiring `target_1` here would silently drop every row
            # issued by a build that stopped computing targets, and dropping a
            # pending trade makes it invisible rather than open -- the record
            # would simply have fewer trades in it, with nothing to say why.
            if rec.get("stop") is None:
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
    # ONE READER for which exits are armed. `rules_from_config` is what the
    # label geometry, the portfolio simulator and Stage 7 all consult, so the
    # record scores a trade under the same rules the engine traded it under.
    # Reading the hierarchy separately here is how the four different
    # definitions of "how did this trade end" got into this codebase.
    from .features.exits import rules_from_config
    rules = rules_from_config(config.params.stage4_core_score,
                              config.params.stage7_risk)

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
                               book, rules=rules)
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
                 book: Optional[Dict[dt.date, set]] = None,
                 rules=None) -> Optional[Dict[str, Any]]:
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
    # ARMED, from the one reader. `use_target` false means the target levels
    # are milestones and nothing more; the position runs to the rank band, the
    # time limit or the floor.
    use_target = bool(getattr(rules, "use_target", True)) if rules is not None else True
    stop = float(rec["stop"]) * factor
    t1 = float(rec["target_1"]) * factor if rec.get("target_1") else None
    t2 = float(rec["target_2"]) * factor if rec.get("target_2") else t1
    walk = future.iloc[1:].head(max_hold)

    exit_price = exit_date = None
    reason = "open"
    held = 0
    mae = mfe = 0.0
    touched_t1 = False
    touched_t2 = False
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

        if t1 is not None and high >= t1:
            # A MILESTONE, not an exit. See EXIT_MODEL target-t2-v4: booking
            # here measured worse than holding to T2 under both label
            # geometries, and it is not the target the model is fitted against.
            touched_t1 = True
        if t2 is not None and high >= t2:
            # Under v5 this is a milestone too. Recorded because "did it ever
            # reach 3R" is a real fact about the trade and the only way to
            # measure, later and from the record, what booking there WOULD have
            # cost -- which is the comparison that put the target exit here in
            # the first place.
            touched_t2 = True

        if low <= stop:
            # A gap-down opens below the stop, so the fill is the open, not the
            # stop. Filling at the stop credits a price that was never available
            # and flatters every stopped trade.
            bar_open = float(bar["open"]) if "open" in bar else float("nan")
            exit_price = min(bar_open, stop) if np.isfinite(bar_open) else stop
            reason = "stop_gap" if exit_price < stop else "stop"
        elif use_target and t2 is not None and high >= t2:
            exit_price, reason = t2, "target"
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
        # Whether the trade reached T1 before it ended. Real information
        # about the trade, recorded rather than acted on -- see EXIT_MODEL.
        "touched_t1": touched_t1,
        "touched_t2": touched_t2,
        "composite_score": rec.get("composite_score"),
        "percentile": rec.get("percentile"),
        "config_version": item.get("config_version"),
        "engine_version": item.get("engine_version"),
        "exit_model": EXIT_MODEL,
        # WHICH EXPERIMENT THIS TRADE BELONGS TO.
        #
        # `exit_model` already stops two exit rules being averaged as one. It
        # does not stop two UNIVERSES being averaged as one, and that is what
        # happened: the recorded operating history predates the population and
        # liquidity corrections, so it describes a book drawn from a different
        # set of names. It was not comparable to any current figure, and
        # nothing said so -- findings C3 and C4.
        "epoch_id": item.get("epoch_id") or _active_epoch_id(),
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


def summarise_by_epoch(outcomes: Iterable[Dict[str, Any]],
                       *, current: Optional[str] = None,
                       ledger_root: Optional[Path] = None) -> Dict[str, Any]:
    """`summarise` partitioned by the epoch each trade was decided under.

    C3/C4, made arithmetic. The engine's recorded operating history was
    produced on a different universe, under a sizer that would size an
    unmeasured name and a cost model that gave it the cheapest fill in the
    book. Averaging those trades with anything produced after the corrections
    reports two engines as one -- the identical failure `EXIT_MODEL` already
    guards against for the exit rule.

    The alternative that was rejected: dropping the retired rows. They are the
    only operating record that exists, and a page that shows nothing reads as
    "no trades yet" rather than as "the trades we have describe a different
    engine". So they are served, LABELLED, next to the epoch that supersedes
    them.

    ``pooled`` is what a caller that ignored the partition would print, and
    ``pooling_overstates_expectancy_by`` is the size of the error it would make
    -- signed, so a negative value means pooling UNDER-states the current
    epoch. Reporting the mistake next to the correct figure is what stops the
    partition from being re-collapsed by the next person who finds the
    per-epoch samples too small to be interesting.
    """
    rows = [o for o in outcomes if o.get("net_return") is not None]
    if current is None:
        current = _active_epoch_id(ledger_root)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for o in rows:
        buckets.setdefault(str(o.get("epoch_id") or PRE_EPOCH), []).append(o)

    #: `pre-epoch` first because it is oldest; the rest by id, which begins
    #: with the date the epoch was opened and therefore sorts chronologically.
    def _order(k: str):
        return (0 if k == PRE_EPOCH else 1, k)

    epochs: List[Dict[str, Any]] = []
    for key in sorted(buckets, key=_order):
        s = summarise(buckets[key])
        s["epoch_id"] = key
        s["is_current"] = (key == current)
        s["retired"] = (key != current)
        s["note"] = ("" if key == current else
                     "produced under a superseded epoch -- not comparable to "
                     "the current one and not poolable with it")
        epochs.append(s)

    pooled = summarise(rows)
    cur = next((e for e in epochs if e["is_current"]), None)
    gap = float("nan")
    if cur is not None and cur.get("n") and pooled.get("n"):
        gap = float(pooled["expectancy"] - cur["expectancy"])

    return {
        "current_epoch": current,
        "epochs": epochs,
        # Deliberately not merged into `epochs`: a caller iterating the list
        # and summing `n` must not pick this up as another cohort.
        "pooled": pooled,
        "pooling_overstates_expectancy_by": gap,
        "spans_multiple_epochs": len(epochs) > 1,
        "current_epoch_has_no_record": cur is None or not cur.get("n"),
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
