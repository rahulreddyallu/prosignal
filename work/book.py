"""A book simulator with every layer of the engine as an explicit switch.

`validation.portfolio_sim.simulate` runs the shipped stack and nothing else, so
the only way to ask what one layer costs is to build the stack up one piece at a
time.  That is what section G of the dossier tabulates and what this file
regenerates -- plus the variants the dossier could not run, because each of them
turns on a behaviour the shipped simulator does not have.

Every default here reproduces `portfolio_sim` exactly.  `test_parity` asserts
that: with all switches at their shipped settings the two agree to floating
point on the same rankings.  A variant is therefore a single named departure
from a verified baseline rather than a separate implementation that happens to
give a different number.

THE FOUR DEPARTURES, and why each is a question about measurement rather than
about strategy:

  target_on_high    The shipped call is `resolve_exits(..., high=None)`, so the
                    3R target can only trigger on a CLOSE while the stop
                    triggers on the intraday LOW.  The training label passes
                    `high`.  One module, two callers, different data.

  charge_reentry    Cohorts are `ceil(horizon/step)` rebalances apart, so every
                    position closes before the next opens, and 89% of them close
                    EARLY.  The shipped rule charges a round trip only to names
                    absent from the previous book, so a name that stopped out in
                    week one and is re-bought at the next rebalance trades for
                    free.

  full_investment   Position value is `min(risk_budget/dist, slot, liquidity)`.
                    At 1% risk over 8 slots the risk term binds above an 8% stop
                    distance, so the book holds cash -- most of it in the
                    highest-volatility names -- while the benchmark it is scored
                    against is fully invested.  Normalising to full investment
                    separates the WEIGHTING decision from the EXPOSURE decision;
                    the shipped construction prices them as one number.

  liquidity_unknown A name with no ADTV gets `slot/entry` -- the largest size
                    the slot allows -- and `impact_bps` returns the half-spread
                    alone.  Largest size and cheapest fill, on exactly the names
                    whose liquidity could not be measured.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from prosignal.features.exits import (EXIT_INVALIDATION, EXIT_STOP,
                                      EXIT_TARGET, EXIT_TIMEOUT)


@dataclass(frozen=True)
class Book:
    """The shipped stack, with each layer switchable."""

    capital: float
    max_positions: int
    risk_per_trade_pct: float
    max_participation_of_adtv: float
    stop_atr_multiple: float
    min_stop_distance_pct: float
    max_stop_distance_pct: float
    invalidation_ma_sessions: int
    invalidation_buffer_atr: float
    horizon_sessions: int
    entry_rank: int
    exit_rank: int
    cost_fn: Optional[Callable[[float, float, float], float]] = None
    cost_bps_round_trip: float = 70.0
    target_r_multiple: float = 3.0

    # -- layer switches (all default to the shipped behaviour) --------------
    use_stop: bool = True
    use_target: bool = True
    use_invalidation: bool = True
    use_costs: bool = True
    risk_sizing: bool = True
    #: Refuse a name sitting below its own invalidation level on the decision
    #: date. The shipped simulator does this unconditionally -- the predicate
    #: lives inside `resolve_exits` -- and so do stage 3 and stage 6 live. The
    #: TRAINING PANEL does not, because `build_panel` only reaches
    #: `resolve_exits` when `exit_rules` is not None, and the shipped config's
    #: `triple_barrier: false` makes it None. Turning this off is how the cost
    #: of ranking a population the book cannot buy gets its own line.
    admissible_only: bool = True

    # -- corrections under test (all default OFF = as shipped) -------------
    target_on_high: bool = False
    charge_reentry: bool = False
    full_investment: bool = False
    refuse_unknown_liquidity: bool = False
    #: A name selected but never FILLED -- no ATR, no price, refused by the
    #: admission predicate -- kept its hysteresis slot AND was recorded as
    #: held, so when it finally filled it paid nothing. It was never bought.
    #: Off here so the decomposition's "as shipped" rows keep the behaviour
    #: they are measuring the cost of.
    unfilled_pays: bool = False

    @property
    def slot(self) -> float:
        return self.capital / self.max_positions

    @property
    def risk_budget(self) -> float:
        return self.capital * self.risk_per_trade_pct / 100.0

    def cost_bps(self, price: float, qty: float, adtv: float) -> float:
        if not self.use_costs:
            return 0.0
        if self.cost_fn is None:
            return self.cost_bps_round_trip
        try:
            return float(self.cost_fn(price, qty, adtv))
        except Exception:
            return self.cost_bps_round_trip


def stop_fraction(b: Book, atr: np.ndarray, entry: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = b.stop_atr_multiple * atr / entry * 100.0
    return np.clip(raw, b.min_stop_distance_pct, b.max_stop_distance_pct) / 100.0


def resolve(b: Book, syms: Sequence[str], i: int, px: Dict[str, pd.DataFrame]):
    """Per-symbol outcome for a cohort opened at row ``i``.

    Mirrors `exits.resolve_exits`, including its intra-bar ordering (stop, then
    target, then invalidation, ties to the worst) and its gap fill at the open.
    Returns ``ret``, ``side`` and ``held`` -- the shipped adapter throws the last
    two away, which is the reason `charge_reentry` cannot be asked for there.
    """
    close, low, open_ = px["close"], px["low"], px["open"]
    high, atr, ma = px["high"], px["atr"], px["ma"]
    n = len(close)
    end = min(i + b.horizon_sessions, n - 1)
    cols = list(syms)
    if end <= i or not cols:
        return pd.DataFrame(columns=["ret", "side", "held"], index=cols,
                            dtype="float64")

    sl = slice(i + 1, end + 1)
    cl = close.loc[:, cols].iloc[sl].to_numpy("float64")
    lo = low.loc[:, cols].iloc[sl].to_numpy("float64")
    op = open_.loc[:, cols].iloc[sl].to_numpy("float64")
    # The one departure that is a fix rather than a layer: the shipped call
    # passes high=None, so `resolve_exits` substitutes the close.
    hi = (high.loc[:, cols].iloc[sl].to_numpy("float64")
          if b.target_on_high else cl)
    e = close.loc[:, cols].iloc[i].to_numpy("float64")
    a = atr.loc[:, cols].iloc[i].to_numpy("float64")

    dist = stop_fraction(b, a, e)
    stop_lvl = e * (1.0 - dist)
    target_lvl = e * (1.0 + b.target_r_multiple * dist)
    n_bars = cl.shape[0]
    big = n_bars + 1

    def first(mask):
        return np.where(mask.any(axis=0), mask.argmax(axis=0), big)

    f_stop = first(lo <= stop_lvl) if b.use_stop else np.full(len(e), big)
    f_target = first(hi >= target_lvl) if b.use_target else np.full(len(e), big)
    if b.use_invalidation:
        mv = ma.loc[:, cols].iloc[sl].to_numpy("float64")
        av = atr.loc[:, cols].iloc[sl].to_numpy("float64")
        with np.errstate(invalid="ignore"):
            m = cl < (mv - b.invalidation_buffer_atr * av)
        m = np.where(np.isfinite(mv) & np.isfinite(av) & np.isfinite(cl), m, False)
        f_inval = first(m)
    else:
        f_inval = np.full(len(e), big)

    firsts = np.minimum(np.minimum(f_stop, f_target), f_inval)
    timed_out = firsts > n_bars
    stopped = (~timed_out) & (f_stop == firsts)
    invalidated = (~timed_out) & (~stopped) & (f_inval == firsts)
    took_profit = (~timed_out) & (~stopped) & (~invalidated)

    idx = np.clip(firsts, 0, max(n_bars - 1, 0))
    take = lambda arr: arr[idx, np.arange(arr.shape[1])]
    gap = np.minimum(take(op), stop_lvl)
    stop_fill = np.where(np.isfinite(take(op)), gap, stop_lvl)
    last_close = cl[-1] if n_bars else np.full(len(e), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(timed_out, last_close / e - 1.0, np.nan)
        ret = np.where(stopped, stop_fill / e - 1.0, ret)
        ret = np.where(invalidated, take(cl) / e - 1.0, ret)
        ret = np.where(took_profit, target_lvl / e - 1.0, ret)
    side = np.where(timed_out, EXIT_TIMEOUT,
                    np.where(stopped, EXIT_STOP,
                             np.where(invalidated, EXIT_INVALIDATION, EXIT_TARGET)))
    held = np.where(timed_out, float(n_bars), firsts.astype("float64") + 1.0)

    ok = np.isfinite(e) & (e > 0) & np.isfinite(a)
    if b.admissible_only:
        # `exits.tradeable_at_entry`, reproduced exactly, including its
        # fallback: an UNKNOWN level cannot exclude a row, so a name with no
        # 50-session average yet is admitted rather than refused.
        entry_ma = ma.loc[:, cols].iloc[i].to_numpy("float64")
        level = entry_ma - b.invalidation_buffer_atr * a
        with np.errstate(invalid="ignore"):
            adm = (e >= level) & np.isfinite(level) & np.isfinite(e)
        adm = np.where(np.isfinite(entry_ma) & np.isfinite(a), adm, True)
        ok = ok & adm
    out = pd.DataFrame({"ret": ret, "side": side, "held": held}, index=cols)
    return out.where(pd.Series(ok, index=cols), np.nan)


def size(b: Book, sym, i, px) -> Optional[Tuple[float, float, float]]:
    """(rupees, price, adtv) under the shipped three-way minimum."""
    entry = px["close"][sym].iloc[i]
    a = px["atr"][sym].iloc[i]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(a):
        return None
    liq = px["adtv"][sym].iloc[i]
    liq_known = bool(np.isfinite(liq) and liq > 0)
    if not liq_known and b.refuse_unknown_liquidity:
        return None
    if not b.risk_sizing:
        return float(b.slot), float(entry), float(liq if liq_known else 0.0)
    dist = float(np.clip(b.stop_atr_multiple * a / entry * 100.0,
                         b.min_stop_distance_pct, b.max_stop_distance_pct) / 100.0)
    if dist <= 0:
        return None
    qty_liq = ((liq * b.max_participation_of_adtv) / entry
               if liq_known else b.slot / entry)
    qty = max(min(b.risk_budget / (entry * dist), b.slot / entry, qty_liq), 0.0)
    return float(qty * entry), float(entry), float(liq if liq_known else 0.0)


def simulate(b: Book, rankings, px, *, phase=0, step_sessions=21,
             dates_allowed=None) -> pd.DataFrame:
    close = px["close"]
    index = list(close.index)
    pos = {d: i for i, d in enumerate(index)}
    allowed = set(dates_allowed) if dates_allowed is not None else None
    stride = max(int(np.ceil(b.horizon_sessions / step_sessions)), 1)
    equity = b.capital
    held: Dict[str, float] = {}          # symbol -> side of its last outcome
    rows: List[Dict[str, float]] = []

    for j in range(phase, len(rankings), stride):
        date, scores = rankings[j]
        if date not in pos or (allowed is not None and date not in allowed):
            continue
        i = pos[date]
        if i + b.horizon_sessions >= len(index):
            continue
        rank = {s: r for r, s in enumerate(scores.index, start=1)}
        keep = [s for s in held if rank.get(s, 10 ** 9) <= b.exit_rank]
        room = b.max_positions - len(keep)
        add = [s for s in list(scores.index)[: b.entry_rank]
               if s not in keep][: max(room, 0)]
        book = [s for s in keep + add if s in close.columns]
        if not book:
            continue

        sized = {}
        for s in book:
            v = size(b, s, i, px)
            if v is not None and v[0] > 0:
                sized[s] = v
        if not sized:
            continue
        names = list(sized)
        out = resolve(b, names, i, px)
        names = [s for s in names if np.isfinite(out.loc[s, "ret"])]
        if not names:
            continue

        scale = equity / b.capital
        gross_notional = sum(sized[s][0] for s in names)
        if b.full_investment and gross_notional > 0:
            # Same relative weights, rescaled so the book is as invested as the
            # benchmark it is compared against. Turns absolute risk-budget
            # sizing into a WEIGHTING scheme and leaves the exposure decision
            # to be made, and priced, separately.
            scale *= b.capital / gross_notional

        pnl = deployed = charged = 0.0
        new_names = 0
        for s in names:
            value, price, liq = sized[s]
            value *= scale
            r = float(out.loc[s, "ret"])
            pnl += value * r
            deployed += value
            prior = held.get(s)
            # A name is a new round trip if it was not held, OR if the position
            # held under that name closed before the horizon -- which is what
            # `side != TIMEOUT` means. The shipped rule tests only the first.
            fresh = (prior is None) or (b.charge_reentry and prior != EXIT_TIMEOUT)
            if fresh:
                new_names += 1
                bps = b.cost_bps(price, value / price if price > 0 else 0.0, liq)
                charged += value * bps / 10_000.0
        gross = pnl
        pnl -= charged
        opening = equity
        equity += pnl
        rows.append({
            "date": date, "ret": pnl / opening, "gross_ret": gross / opening,
            "cost_ret": charged / opening, "equity": equity,
            "n_held": len(names), "n_book": len(book), "n_new": new_names,
            "deployed_frac": deployed / opening,
            "stop_share": float((out.loc[names, "side"] == EXIT_STOP).mean()),
            "target_share": float((out.loc[names, "side"] == EXIT_TARGET).mean()),
            "inval_share": float((out.loc[names, "side"] == EXIT_INVALIDATION).mean()),
            "timeout_share": float((out.loc[names, "side"] == EXIT_TIMEOUT).mean()),
            "median_held": float(out.loc[names, "held"].median()),
        })
        # Keyed on the whole BOOK, not just the names that filled -- which is
        # what the shipped simulator does (`held = {s: 1 for s in book}`). A
        # name it could not size still occupies a hysteresis slot next time.
        unfilled = float("nan") if b.unfilled_pays else EXIT_TIMEOUT
        held = {s: float(out.loc[s, "side"]) if s in names else unfilled
                for s in book}

    return pd.DataFrame(rows)


def phases(b: Book, rankings, px, *, step_sessions=21, dates_allowed=None):
    """Every offset of the non-overlapping schedule, pooled AND kept apart.

    `portfolio_sim.phase_summary` reports `max_drawdown` as the MEAN across
    phases. A mean of three schedules' worst moments is not a drawdown anyone
    could have experienced; the worst single schedule is. Both are returned.
    """
    stride = max(int(np.ceil(b.horizon_sessions / step_sessions)), 1)
    frames = [simulate(b, rankings, px, phase=p, step_sessions=step_sessions,
                       dates_allowed=dates_allowed) for p in range(stride)]
    usable = [f for f in frames if len(f) >= 3]
    if not usable:
        return {}
    pooled = pd.concat(usable, ignore_index=True)
    r = pooled["ret"].to_numpy("float64")
    sd = float(r.std(ddof=1))
    ppy = 252.0 / float(b.horizon_sessions)
    dds = []
    for f in usable:
        eq = (1.0 + f["ret"]).cumprod()
        dds.append(float((eq / eq.cummax() - 1.0).min()))
    return {
        "mean_return": float(r.mean()),
        "sd": sd,
        "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        "hit_rate": float((r > 0).mean()),
        "n_periods": int(r.size),
        "n_phases": len(usable),
        "avg_names": float(pooled["n_held"].mean()),
        "avg_book": float(pooled["n_book"].mean()),
        "avg_new": float(pooled["n_new"].mean()),
        "deployed_frac": float(pooled["deployed_frac"].mean()),
        "mean_gross": float(pooled["gross_ret"].mean()),
        "mean_cost": float(pooled["cost_ret"].mean()),
        "drawdown_mean_of_phases": float(np.mean(dds)),
        "drawdown_worst_schedule": float(np.min(dds)),
        "stop_share": float(pooled["stop_share"].mean()),
        "target_share": float(pooled["target_share"].mean()),
        "inval_share": float(pooled["inval_share"].mean()),
        "timeout_share": float(pooled["timeout_share"].mean()),
        "median_held": float(pooled["median_held"].median()),
        "returns": r,
    }
