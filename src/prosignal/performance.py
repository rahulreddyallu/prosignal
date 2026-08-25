"""Did following the shortlist beat not following it?

The outcomes module already answers "what happened to each signal". This
answers the question a person actually asks, which is comparative: the same
money, over the same days, in the index instead.

Two things this module refuses to do.

It will not quote a t-statistic without the overlap correction. Signals are
issued daily and held up to a quarter, so a run of trades opened in the same
week shares almost all of its holding window; the naive t on that sample
rejects a true null about a third of the time. validation.significance
carries the measured correction and it is applied here.

It will not report a benchmark it could not compute. If the index series is
missing for a trade's window the trade is counted in the raw return and
excluded from the comparison, and the two counts are reported separately --
a comparison against a benchmark that quietly fell back to zero is worse
than no comparison, because it looks like outperformance.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import weakref

import numpy as np

from .data.types import DATE

__all__ = ["Trade", "performance", "by_ticker", "equity_curve",
           "open_positions", "dedupe", "overlaps", "calls_for", "merge_reentries",
           "holding_profile"]


@dataclass
class Trade:
    ticker: str
    signal_date: str
    entry_date: str
    exit_date: str
    sessions_held: int
    net_return: float
    gross_return: float
    exit_reason: str
    benchmark_return: Optional[float] = None

    @property
    def excess(self) -> Optional[float]:
        if self.benchmark_return is None:
            return None
        return self.net_return - self.benchmark_return


#: The curated index table holds every index NSE publishes -- 176 of them,
#: interleaved in one frame. Selecting the wrong column, or failing to select
#: at all, divides one index's close by another's and yields a benchmark
#: return in the hundreds of percent. It did, before this was pinned.
_NAME_COLUMNS = ("index_name", "symbol", "name")


#: Keyed by the store OBJECT, weakly. id() was wrong: CPython reuses an id
#: once the object behind it is collected, so a freshly built store could be
#: handed the previous store's benchmark frame. A test caught exactly that.
_INDEX_CACHE: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = weakref.WeakKeyDictionary()


def _index_frame(store: Any, symbol: str):
    """The benchmark series, or None. Never a fabricated one, and never a
    frame that still holds more than one index."""
    # performance(), by_ticker() and equity_curve() each want the same
    # benchmark and each rebuilt it from a 224,000-row table, three times a
    # request. Keyed by store identity so a rebuilt store is not served a
    # stale frame.
    try:
        per_store = _INDEX_CACHE.setdefault(store, {})
    except TypeError:
        per_store = None            # not weak-referenceable: just do the work
    if per_store is not None and symbol in per_store:
        return per_store[symbol]
    fn = getattr(store, "read_indices", None)
    if fn is None:
        return None
    try:
        df = fn()
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None

    col = next((c for c in _NAME_COLUMNS if c in df.columns), None)
    if col is None:
        # Cannot prove which rows are the benchmark, so there is no benchmark.
        return None
    want = symbol.strip().casefold()
    sel = df[df[col].astype(str).str.strip().str.casefold() == want]
    if sel.empty:
        if per_store is not None:
            per_store[symbol] = None
        return None
    out = sel.sort_values("date").reset_index(drop=True)
    if per_store is not None:
        per_store[symbol] = out
    return out


def _bench_return(idx, entry_date: str, exit_date: str) -> Optional[float]:
    if idx is None:
        return None
    import pandas as pd
    try:
        a = pd.to_datetime(entry_date).normalize()
        b = pd.to_datetime(exit_date).normalize()
    except Exception:
        return None
    on_or_after = idx[idx["date"] >= a]
    on_or_before = idx[idx["date"] <= b]
    if on_or_after.empty or on_or_before.empty:
        return None
    start = float(on_or_after.iloc[0]["close"])
    end = float(on_or_before.iloc[-1]["close"])
    if not np.isfinite(start) or start <= 0 or not np.isfinite(end):
        return None
    return end / start - 1.0


def _trades(outcomes: Sequence[Dict[str, Any]], idx) -> List[Trade]:
    out: List[Trade] = []
    for o in outcomes:
        if o.get("net_return") is None:
            continue
        t = Trade(
            ticker=str(o.get("ticker", "")),
            signal_date=str(o.get("signal_date", "")),
            entry_date=str(o.get("entry_date", "")),
            exit_date=str(o.get("exit_date", "")),
            sessions_held=int(o.get("sessions_held") or 0),
            net_return=float(o["net_return"]),
            gross_return=float(o.get("gross_return") or 0.0),
            exit_reason=str(o.get("exit_reason") or "open"),
        )
        t.benchmark_return = _bench_return(idx, t.entry_date, t.exit_date)
        out.append(t)
    return out


def _corrected_t(values: np.ndarray, horizon: int, step: int) -> Dict[str, Any]:
    """Overlap-corrected significance, or an honest refusal."""
    n = len(values)
    if n < 2:
        return {"t": None, "note": "Needs at least two closed trades."}
    try:
        from .validation.significance import newey_west_t
        res = newey_west_t(values, horizon_sessions=horizon, step_sessions=step,
                           use_analytic_vif=True)
        t = float(res.adjusted_t)
        return {
            "t": None if not np.isfinite(t) else t,
            "naive_t": None if not np.isfinite(res.naive_t) else float(res.naive_t),
            "effective_n": None if not np.isfinite(res.effective_n)
                           else float(res.effective_n),
            "note": ("Corrected for the overlap between trades held at the "
                     "same time. The uncorrected figure is shown so the size "
                     "of that correction is visible."),
        }
    except Exception:
        return {"t": None, "note": "Overlap correction unavailable."}


def split_cohorts(outcomes: Sequence[Dict[str, Any]], cutoff: Optional[str]):
    """Separate fully observed signals from partially observed ones.

    A signal issued 63 sessions ago has had its whole window: whatever it was
    going to do, it did. A signal issued three days ago has not -- and the
    only members of that cohort visible today are the ones that finished
    fast. Since target_1 sits nearer than a 2.5-ATR stop, finishing fast
    skews to winners, so the recent cohort looks far better than it is.

    Measured on this ledger: complete cohorts gave +1.31% at a 49.3% win rate
    over an 18-session average hold; the incomplete ones gave +5.89% at 60.9%
    over 1.79 sessions. Pooling the two turned a corrected t of 0.41 into
    8.47. They are different populations and are kept apart.
    """
    if not cutoff:
        return list(outcomes), []
    done, partial = [], []
    for o in outcomes:
        d = str(o.get("signal_date") or "")[:10]
        (done if d and d <= cutoff[:10] else partial).append(o)
    return done, partial


def recent_activity(partial: Sequence[Dict[str, Any]], limit: int = 8) -> Dict[str, Any]:
    """Trades that closed inside the incomplete window.

    Real closes, shown so the page moves daily, and deliberately not folded
    into any figure that carries a t-statistic.
    """
    # One name closing once is one close, however many runs issued it.
    rows, seen = [], set()
    for o in sorted((x for x in partial if x.get("net_return") is not None),
                    key=lambda x: str(x.get("exit_date") or ""), reverse=True):
        key = (o.get("ticker"), str(o.get("exit_date") or "")[:10])
        if key in seen:
            continue
        seen.add(key)
        rows.append(o)
    net = np.array([float(o["net_return"]) for o in rows]) if rows else np.array([])
    return {
        "n": len(rows),
        "recent": [{
            "ticker": o.get("ticker"), "net_return": float(o["net_return"]),
            "exit_date": o.get("exit_date"), "exit_reason": o.get("exit_reason"),
            "sessions_held": o.get("sessions_held"),
        } for o in rows[:limit]],
        "avg_return": float(net.mean()) if net.size else None,
        "win_rate": float((net > 0).mean()) if net.size else None,
        "note": ("Closed inside the current window. Only the fast movers in "
                 "this group have finished, so it reads better than it is "
                 "and is kept out of the figures above."),
    }


def performance(outcomes: Sequence[Dict[str, Any]], store: Any = None, *,
                benchmark: str = "Nifty 200", horizon: int = 63,
                step: int = 21) -> Dict[str, Any]:
    """Headline comparison. Every ratio carries the count behind it."""
    idx = _index_frame(store, benchmark) if store is not None else None
    trades = _trades(merge_reentries(outcomes), idx)
    n = len(trades)
    if n == 0:
        return {"n": 0, "benchmark": benchmark,
                "note": "No signal has completed its holding window yet."}

    net = np.array([t.net_return for t in trades], dtype=float)
    paired = [t for t in trades if t.excess is not None]
    exc = np.array([t.excess for t in paired], dtype=float) if paired else np.array([])

    wins = net[net > 0]
    beat = exc[exc > 0] if exc.size else np.array([])

    out: Dict[str, Any] = {
        "n": n,
        "benchmark": benchmark,
        "benchmark_covered": len(paired),
        "benchmark_missing": n - len(paired),
        "total_return": float(net.sum()),
        "avg_return": float(net.mean()),
        "median_return": float(np.median(net)),
        "win_rate": float(len(wins) / n),
        "best": float(net.max()),
        "worst": float(net.min()),
        "avg_hold_sessions": float(np.mean([t.sessions_held for t in trades])),
        "exit_mix": {r: sum(1 for t in trades if t.exit_reason == r)
                     for r in sorted({t.exit_reason for t in trades})},
    }

    if exc.size:
        out.update({
            "avg_benchmark_return": float(
                np.mean([t.benchmark_return for t in paired])),
            "avg_excess": float(exc.mean()),
            "beat_rate": float(len(beat) / len(paired)),
            "significance": _corrected_t(exc, horizon, step),
        })
    else:
        out["comparison_note"] = (
            "The index series does not cover these trades, so no comparison "
            "is offered. The returns above stand on their own.")
    return out


def dedupe(outcomes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One call is one call, however many runs issued it that day.

    A day can produce several ledger runs and each writes its own outcome row
    for the same signal. Counting those separately made SUZLON read as five
    calls returning +70.71% when it was three calls returning +43.17%.
    """
    seen, out = set(), []
    for o in outcomes:
        key = (o.get("ticker"), str(o.get("signal_date") or "")[:10])
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def merge_reentries(outcomes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a name re-signalled while it was still held into one position.

    Stage 6 enters at rank 8 and holds while the name stays inside rank 16,
    so a name that reappears tomorrow is the position being MAINTAINED, not a
    second one being opened. Counting it twice reports a return on two slots
    of capital the strategy never committed -- and it is not rare: on a live
    slate two of twenty-one open positions were re-entries of names already
    held.

    One position runs from the first entry to the last exit. The return is
    recomputed across that whole span rather than summed, because summing
    two overlapping legs is the double-count this exists to remove.
    """
    rows = sorted(dedupe(outcomes),
                  key=lambda o: (str(o.get("ticker") or ""),
                                 str(o.get("entry_date") or "")))
    out: List[Dict[str, Any]] = []
    for o in rows:
        prev = out[-1] if out else None
        same = prev and prev.get("ticker") == o.get("ticker")
        held = same and str(o.get("entry_date") or "") <= str(prev.get("exit_date") or "")
        if not held:
            out.append(dict(o))
            continue
        # Extend the position rather than opening a second one.
        if str(o.get("exit_date") or "") > str(prev.get("exit_date") or ""):
            prev["exit_date"] = o.get("exit_date")
            prev["exit_price"] = o.get("exit_price")
            prev["exit_reason"] = o.get("exit_reason")
        entry = float(prev.get("entry_price") or 0.0)
        exit_p = float(prev.get("exit_price") or 0.0)
        if entry > 0 and np.isfinite(exit_p):
            gross = exit_p / entry - 1.0
            # Costs were charged per leg; one position pays one round trip.
            slip = float(prev.get("gross_return") or 0.0) - float(prev.get("net_return") or 0.0)
            prev["gross_return"] = gross
            prev["net_return"] = gross - slip
        prev["sessions_held"] = int(prev.get("sessions_held") or 0) + \
            int(o.get("sessions_held") or 0)
        prev["merged_legs"] = int(prev.get("merged_legs") or 1) + 1
    return out


def overlaps(trades: Sequence[Dict[str, Any]]) -> int:
    """How many calls on this name were held at the same time as another.

    It matters because the totals add per-call returns. Two calls held
    together cannot both be funded from one slot, so a sum across them is a
    return on twice the capital -- leverage the strategy never took.
    """
    rows = sorted(({"a": str(t.get("entry_date") or "")[:10],
                    "b": str(t.get("exit_date") or "")[:10]} for t in trades),
                  key=lambda r: r["a"])
    n = 0
    for i, r in enumerate(rows):
        if any(q["a"] <= r["b"] and r["a"] <= q["b"] for q in rows[:i]):
            n += 1
    return n


def by_ticker(outcomes: Sequence[Dict[str, Any]], store: Any = None, *,
              benchmark: str = "Nifty 200") -> List[Dict[str, Any]]:
    """Per-name record, worst first -- the losses are the part worth reading."""
    idx = _index_frame(store, benchmark) if store is not None else None
    trades = _trades(merge_reentries(outcomes), idx)
    grouped: Dict[str, List[Trade]] = {}
    for t in trades:
        grouped.setdefault(t.ticker, []).append(t)

    rows = []
    for ticker, ts in grouped.items():
        net = np.array([t.net_return for t in ts], dtype=float)
        exc = [t.excess for t in ts if t.excess is not None]
        rows.append({
            "ticker": ticker,
            "n": len(ts),
            "avg_return": float(net.mean()),
            "total_return": float(net.sum()),
            "win_rate": float((net > 0).mean()),
            "avg_excess": float(np.mean(exc)) if exc else None,
            "best": float(net.max()),
            "worst": float(net.min()),
            "last_exit": max(t.exit_date for t in ts),
            "first_call": min(t.signal_date for t in ts),
            "overlapping": overlaps([{"entry_date": t.entry_date,
                                      "exit_date": t.exit_date} for t in ts]),
            "exit_mix": {r: sum(1 for t in ts if t.exit_reason == r)
                         for r in sorted({t.exit_reason for t in ts})},
        })
    rows.sort(key=lambda r: r["avg_return"])
    return rows


def equity_curve(outcomes: Sequence[Dict[str, Any]], store: Any = None, *,
                 benchmark: str = "Nifty 200") -> List[Dict[str, Any]]:
    """Cumulative return by exit date, against the same days in the index.

    Equal-weighted per trade and NOT compounded across overlapping holds --
    compounding overlapping positions would imply leverage the strategy never
    took. This is the running sum of per-trade returns, which is what an
    equal-slot book actually accumulates.
    """
    idx = _index_frame(store, benchmark) if store is not None else None
    trades = sorted(_trades(merge_reentries(outcomes), idx), key=lambda t: t.exit_date or "")
    curve, run, bench_run = [], 0.0, 0.0
    for t in trades:
        if not t.exit_date:
            continue
        run += t.net_return
        if t.benchmark_return is not None:
            bench_run += t.benchmark_return
        curve.append({
            "date": t.exit_date,
            "ticker": t.ticker,
            "cumulative": run,
            "benchmark_cumulative": bench_run if t.benchmark_return is not None else None,
            "trade_return": t.net_return,
        })
    return curve


# ---------------------------------------------------------------------------
# Positions still running
# ---------------------------------------------------------------------------

def open_positions(ledger_rows, resolved, store, *, max_hold: int = 63,
                   as_of=None) -> Dict[str, Any]:
    """Signals that have not closed yet, marked to the latest close.

    These move every day, which is the point -- without them a page that only
    counts closed trades sits still for weeks and looks broken.

    They are returned SEPARATELY and must never be pooled into the realised
    figures. A mark is not an outcome: it is what the position happens to be
    worth today, it can reverse tomorrow, and letting a favourable mark into
    the win rate or the t-statistic would let the record be improved by
    picking a good afternoon to look at it.
    """
    import pandas as pd

    # A ledger row names the tickers it issued in `signals_generated` and
    # carries their detail in `stocks_scored`. There is no `signals` list --
    # reading one returned nothing, silently, which is exactly how a page
    # ends up reporting zero open positions forever.
    done = {(r.get("run_id"), r.get("ticker")) for r in (resolved or [])}
    seen = set()
    want: Dict[str, list] = {}
    for row in (ledger_rows or []):
        rid = row.get("run_id")
        issued = set(row.get("signals_generated") or [])
        if not issued:
            continue
        for rec in (row.get("stocks_scored") or []):
            tk = rec.get("ticker")
            if not tk or tk not in issued or (rid, tk) in done:
                continue
            if rec.get("stop") is None or rec.get("target_1") is None:
                continue
            # One name called once on one day is one position, however many
            # runs that day issued it. Without this the same holding is
            # listed several times and the average is weighted by how often
            # the engine happened to re-run.
            key = (str(tk), str(row.get("date") or "")[:10])
            if key in seen:
                continue
            seen.add(key)
            want.setdefault(str(tk), []).append({
                "run_id": rid,
                "date": key[1],
            })
    if not want:
        return {"n": 0, "positions": []}

    try:
        prices = store.read_prices(symbols=list(want.keys()))
    except Exception:
        return {"n": 0, "positions": [], "note": "Prices unavailable."}
    if prices is None or getattr(prices, "empty", True):
        return {"n": 0, "positions": []}

    out = []
    for ticker, items in want.items():
        f = prices[prices["symbol"].astype(str) == ticker].sort_values(DATE)
        if f.empty:
            continue
        last = f.iloc[-1]
        last_px, last_dt = float(last["close"]), last[DATE]
        for it in items:
            try:
                sig_dt = pd.to_datetime(it["date"]).normalize()
            except Exception:
                continue
            fut = f[f[DATE] > sig_dt]
            if fut.empty:
                continue
            entry = float(fut.iloc[0]["open"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            held = int((f[DATE] > fut.iloc[0][DATE]).sum())
            if held >= max_hold:
                continue                       # the resolver owns this one
            # The path between entry and today, not just the endpoints. A
            # name up 2% that went to +9% and gave it back is not the same
            # position as one that ground up quietly, and the number alone
            # cannot tell those apart.
            path = [float(v) for v in fut["close"].head(max_hold).tolist()
                    if np.isfinite(v)]
            out.append({
                "ticker": ticker,
                "signal_date": it["date"],
                "entry_price": entry,
                "last_price": last_px,
                "last_date": str(pd.Timestamp(last_dt).date()),
                "unrealised": last_px / entry - 1.0,
                "sessions_held": held,
                "sessions_left": max_hold - held,
                "path": path if len(path) > 1 else [],
            })

    # Same rule for positions still running: the earliest entry is the
    # position, and a later signal on the same name is it being maintained.
    first: Dict[str, Any] = {}
    for r in sorted(out, key=lambda r: (r["ticker"], r["signal_date"])):
        cur = first.get(r["ticker"])
        if cur is None:
            first[r["ticker"]] = r
        else:
            cur["sessions_left"] = min(cur["sessions_left"], r["sessions_left"])
            cur["reaffirmed"] = int(cur.get("reaffirmed") or 0) + 1
    out = list(first.values())
    out.sort(key=lambda r: r["unrealised"])
    marks = np.array([r["unrealised"] for r in out], dtype=float) if out else np.array([])
    return {
        "n": len(out),
        "positions": out,
        "avg_unrealised": float(marks.mean()) if marks.size else None,
        "up": int((marks > 0).sum()) if marks.size else 0,
        "as_of": out[0]["last_date"] if out else None,
        "note": ("Marked to the latest close. These are not results -- they "
                 "can reverse before they close, and they are kept out of "
                 "every figure above."),
    }


def calls_for(ticker: str, outcomes: Sequence[Dict[str, Any]], store: Any = None,
              *, open_rows: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Every call on one name, and two different ways of totalling them.

    The two are not interchangeable and the screen shows both because the
    honest answer to "what did this name return" depends on what you did.

    `taking_every_call` adds the per-call returns. That is what an equal-slot
    book earns IF the calls did not overlap. Where they did, the sum is a
    return on more capital than one slot -- two positions in one name held
    together are two slots -- so the overlap count is reported next to it
    rather than buried.

    `holding_since_first` is the other question entirely: buy on the first
    call, never sell. It ignores every exit the engine asked for, which is
    why it is a comparison and not a result.
    """
    sym = str(ticker).upper()
    mine = merge_reentries([o for o in outcomes
                            if str(o.get("ticker", "")).upper() == sym])
    mine.sort(key=lambda o: str(o.get("signal_date") or ""))
    still = [o for o in (open_rows or [])
             if str(o.get("ticker", "")).upper() == sym]

    calls = [{
        "signal_date": o.get("signal_date"),
        "entry_date": o.get("entry_date"),
        "exit_date": o.get("exit_date"),
        "entry_price": o.get("entry_price"),
        "exit_price": o.get("exit_price"),
        "sessions_held": o.get("sessions_held"),
        "net_return": o.get("net_return"),
        "exit_reason": o.get("exit_reason"),
    } for o in mine]

    total = float(sum(float(c["net_return"] or 0.0) for c in calls)) if calls else None
    over = overlaps(calls)

    hold = None
    if calls and store is not None:
        try:
            # Only the close is needed for the hold comparison; reading the
            # whole row set for one name was most of the panel's latency.
            f = store.read_prices(symbols=[sym],
                                  columns=["date", "symbol", "close"])
            if f is not None and not f.empty:
                f = f.sort_values(DATE)
                first_entry = float(calls[0]["entry_price"] or 0.0)
                last_close = float(f.iloc[-1]["close"])
                if first_entry > 0 and np.isfinite(last_close):
                    hold = last_close / first_entry - 1.0
        except Exception:
            hold = None

    won = [c for c in calls if (c["net_return"] or 0) > 0]
    return {
        "ticker": sym,
        "n_calls": len(calls),
        "n_paid_off": len(won),
        "first_call": calls[0]["signal_date"] if calls else None,
        "last_call": calls[-1]["signal_date"] if calls else None,
        "calls": calls,
        "open": len(still),
        "taking_every_call": total,
        "overlapping_calls": over,
        "holding_since_first": hold,
        "best": max((c["net_return"] for c in calls), default=None),
        "worst": min((c["net_return"] for c in calls), default=None),
    }


def holding_profile(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How long positions actually last, from the record rather than config.

    The card showed "15-63 sessions" on every name, which is the configured
    backstop pair and says nothing about the name it is printed on. Worse, it
    is wrong: across 716 closed trades the median hold is 2 sessions.

    A per-name estimate cannot come from the target distance, because the
    target IS defined in ATR units -- stop = 2.5 ATR and target = 1.5 R, so
    every name's target sits exactly 3.75 ATR away and any first-passage
    estimate returns the same number for all of them. Measured across a live
    slate: target/stop was 1.500 on every pick.

    So this reports the distribution the engine has actually produced, split
    by how positions ended, which is where the variation really is: names
    that reach their target and names that stop out do not last the same
    length of time.
    """
    rows = [o for o in outcomes if o.get("sessions_held")]
    if len(rows) < 20:
        return {"n": len(rows)}
    held = np.array([int(o["sessions_held"]) for o in rows], dtype=float)
    by: Dict[str, Any] = {}
    for o in rows:
        by.setdefault(str(o.get("exit_reason") or "other"), []).append(
            int(o["sessions_held"]))
    return {
        "n": len(rows),
        "median": float(np.median(held)),
        "p10": float(np.percentile(held, 10)),
        "p90": float(np.percentile(held, 90)),
        "by_exit": {k: {"n": len(v), "median": float(np.median(v))}
                    for k, v in sorted(by.items(), key=lambda x: -len(x[1]))},
    }
