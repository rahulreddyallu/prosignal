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

import numpy as np

__all__ = ["Trade", "performance", "by_ticker", "equity_curve"]


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


def _index_frame(store: Any, symbol: str):
    """The benchmark series, or None. Never a fabricated one, and never a
    frame that still holds more than one index."""
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
        return None
    return sel.sort_values("date").reset_index(drop=True)


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


def performance(outcomes: Sequence[Dict[str, Any]], store: Any = None, *,
                benchmark: str = "Nifty 200", horizon: int = 63,
                step: int = 21) -> Dict[str, Any]:
    """Headline comparison. Every ratio carries the count behind it."""
    idx = _index_frame(store, benchmark) if store is not None else None
    trades = _trades(outcomes, idx)
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


def by_ticker(outcomes: Sequence[Dict[str, Any]], store: Any = None, *,
              benchmark: str = "Nifty 200") -> List[Dict[str, Any]]:
    """Per-name record, worst first -- the losses are the part worth reading."""
    idx = _index_frame(store, benchmark) if store is not None else None
    trades = _trades(outcomes, idx)
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
    trades = sorted(_trades(outcomes, idx), key=lambda t: t.exit_date or "")
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
