"""Simulate the book the engine would actually have held.

An IC says the ranking orders names correctly. It says nothing about what a
book built on that ranking returns after position sizing, a stop, an
invalidation level and costs -- and those are not neutral. Sizing here is
``risk_budget / risk_per_share``, so a tighter stop buys a LARGER position for
the same rupee risk; a per-position return comparison silently compares two
different position sizes and attributes the difference to the stop. Every
Stage 6, 7 and 8 finding in this repository turns on that distinction, and this
module is where it is computed.

Two modelling choices that are easy to get wrong and change the answer:

  overlap    Rebalances are ``step`` sessions apart and positions hold up to
             ``horizon``. With horizon > step, several cohorts are open at once
             and compounding every rebalance in sequence implies leverage the
             book never had. :func:`simulate` takes a phase offset and advances
             by whole holding periods, so one cohort closes before the next
             opens. Run every offset and report them together.

  entry cost Only names NEW to the book pay a round trip. Charging every held
             name at every rebalance is what makes a buffer band look free when
             it is the thing doing the work.

  cost size  Round-trip cost is NOT a constant. Impact is a square-root
             function of participation, so the same rupee position costs 86 bps
             against a Rs 20 crore ADTV and 135 bps against Rs 5 crore. A flat
             assumption is optimistic for exactly the thin names a screen
             surfaces, so ``cost_bps`` takes the position and its liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["PortfolioParams", "PortfolioResult", "simulate", "phase_summary"]


@dataclass(frozen=True)
class PortfolioParams:
    """Everything the shipped stages would apply, in one place."""

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
    #: Flat fallback, used when ``cost_fn`` is absent. Tests use it; the CLI
    #: passes the real cost model, which is size- and liquidity-dependent.
    cost_bps_round_trip: float = 70.0
    #: (price, quantity, adtv_inr) -> round-trip bps of the buy value.
    cost_fn: Optional[Callable[[float, float, float], float]] = None

    # -- portfolio-level volatility scaling (Moreira & Muir 2017) -----------
    #: Annualised volatility the BOOK is scaled toward. `None` disables the
    #: overlay entirely and every position keeps its own size.
    #:
    #: Note what this is NOT: position sizing here is already inverse-volatility
    #: through the ATR stop -- `risk_budget / (entry * atr_distance)` gives a
    #: high-ATR name a smaller position by construction. This is the separate,
    #: AGGREGATE decision: how much of the book to have on at all, given how
    #: turbulent the market has been.
    target_vol_annual: Optional[float] = None
    #: Trailing sessions of the equal-weight universe used to read the risk
    #: state. Moreira & Muir use the previous month.
    vol_window_sessions: int = 21
    #: The overlay may not lever the book beyond this or cut it below it. An
    #: uncapped inverse-variance rule takes enormous positions in the calmest
    #: stretch of the sample, which is where a volatility estimate is least
    #: reliable and where a variance-scaled backtest earns most of its result.
    max_vol_scale: float = 1.5
    min_vol_scale: float = 0.5

    def cost_bps(self, price: float, quantity: float, adtv: float) -> float:
        if self.cost_fn is None:
            return self.cost_bps_round_trip
        try:
            return float(self.cost_fn(price, quantity, adtv))
        except Exception:
            # A cost model that cannot price this trade must not silently make
            # it free. The flat fallback is the conservative direction.
            return self.cost_bps_round_trip

    @property
    def slot(self) -> float:
        return self.capital / self.max_positions

    @property
    def risk_budget(self) -> float:
        return self.capital * self.risk_per_trade_pct / 100.0


@dataclass
class PortfolioResult:
    periods: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return self.periods.empty

    def metrics(self, periods_per_year: float = 4.0) -> Dict[str, float]:
        if self.empty or len(self.periods) < 2:
            return {}
        r = self.periods["ret"].to_numpy(dtype="float64")
        equity = (1.0 + self.periods["ret"]).cumprod()
        drawdown = float((equity / equity.cummax() - 1.0).min())
        sd = float(r.std(ddof=1))
        downside = r[r < 0]
        return {
            "mean_return": float(r.mean()),
            "sd": sd,
            "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
            "sortino": (float(r.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))
                        if downside.size > 1 and downside.std(ddof=1) > 0 else float("nan")),
            "max_drawdown": drawdown,
            "hit_rate": float((r > 0).mean()),
            "total_return": float(equity.iloc[-1] - 1.0),
            "n_periods": int(len(r)),
            "avg_names": float(self.periods["n_held"].mean()),
            "avg_turnover": float(self.periods["n_new"].mean()),
            "mean_gross": (float(self.periods["gross_ret"].mean())
                           if "gross_ret" in self.periods else float("nan")),
            "mean_cost": (float(self.periods["cost_ret"].mean())
                          if "cost_ret" in self.periods else float("nan")),
            #: Share of the gross edge handed to the broker and the exchange.
            #: A book whose gross return is real and whose cost share is above
            #: 1.0 is not a strategy, it is a fee-generation scheme.
            "cost_share_of_gross": (
                float(self.periods["cost_ret"].sum() / self.periods["gross_ret"].sum())
                if "gross_ret" in self.periods
                and float(self.periods["gross_ret"].sum()) > 0 else float("nan")),
        }



def _volatility_scale(close: pd.DataFrame, i: int, p: PortfolioParams
                      ) -> Tuple[float, float]:
    """(exposure multiplier, realised annualised vol) for this rebalance.

    Moreira & Muir (2017): scaling a portfolio by the inverse of its recent
    realised variance raises the Sharpe ratio, because volatility is far more
    forecastable at short horizons than return is. The overlay does not try to
    predict direction at all -- it decides SIZE.

    MEASURED HERE, IT DOES NOT. Over 50 out-of-sample rebalances:

        target vol    mean ret      sd    Sharpe   avg scale
        off             +3.12%   7.87%    +0.79      1.00
        10%             +2.69%   7.40%    +0.73      0.74
        15%             +3.37%   9.22%    +0.73      1.01
        20%             +3.83%  10.15%    +0.76      1.21
        25%             +4.22%  11.04%    +0.77      1.35

    A 25% target returns +1.11% more per period than no overlay at all, with a
    t of +2.24 -- and it is not alpha. Average exposure is 1.35x, volatility
    rises from 7.87% to 11.04%, and the SHARPE FALLS. Read on mean return the
    overlay looks like it works; read on the only measure that is invariant to
    leverage, no setting beats switching it off. It ships disabled.

    The risk state is read from the equal-weight universe rather than from the
    book's own history, because the book has no history at its first rebalance
    and a rule that only starts working in period two is not a rule.

    Reads ``close.iloc[: i + 1]`` and nothing after. Returning 1.0 when the
    estimate cannot be formed is deliberate: an unmeasurable risk state is not
    evidence of a calm one.
    """
    if p.target_vol_annual is None or p.target_vol_annual <= 0:
        return 1.0, float("nan")
    window = int(max(p.vol_window_sessions, 2))
    start = max(i + 1 - window - 1, 0)
    hist = close.iloc[start: i + 1]
    if len(hist) < 5:
        return 1.0, float("nan")
    # `fill_method=None`: a padded gap manufactures a zero return, which
    # biases the volatility estimate DOWNWARD and levers the book up on
    # exactly the names whose data is missing.
    ret = hist.pct_change(fill_method=None).iloc[1:]
    if ret.empty:
        return 1.0, float("nan")
    # Equal-weight across names present on each day. A name that listed midway
    # contributes only where it has a return, rather than dragging the mean.
    daily = ret.mean(axis=1, skipna=True).to_numpy("float64")
    daily = daily[np.isfinite(daily)]
    if daily.size < 4:
        return 1.0, float("nan")
    sd = float(np.std(daily, ddof=1)) * np.sqrt(252.0)
    if not np.isfinite(sd) or sd <= 1e-8:
        return 1.0, float("nan")
    raw = float(p.target_vol_annual) / sd
    return float(min(max(raw, p.min_vol_scale), p.max_vol_scale)), sd


def _position(sym: str, i: int, close, atr, adtv, p: PortfolioParams
              ) -> Optional[Tuple[float, float, float]]:
    """(rupees deployed, price, adtv), honouring risk budget, slot and liquidity."""
    entry = close[sym].iloc[i]
    a = atr[sym].iloc[i]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(a):
        return None
    dist = min(max(p.stop_atr_multiple * a / entry * 100.0,
                   p.min_stop_distance_pct), p.max_stop_distance_pct) / 100.0
    risk_per_share = entry * dist
    if risk_per_share <= 0:
        return None
    liquidity = adtv[sym].iloc[i]
    qty_liq = ((liquidity * p.max_participation_of_adtv) / entry
               if np.isfinite(liquidity) and liquidity > 0 else p.slot / entry)
    qty = max(min(p.risk_budget / risk_per_share, p.slot / entry, qty_liq), 0.0)
    return float(qty * entry), float(entry), float(liquidity if np.isfinite(liquidity) else 0.0)


def _hold(sym: str, i: int, close, low, open_, ma, atr, p: PortfolioParams
          ) -> Optional[float]:
    """Realised return of one position: stop, then invalidation, then horizon."""
    entry = close[sym].iloc[i]
    a = atr[sym].iloc[i]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(a):
        return None
    dist = min(max(p.stop_atr_multiple * a / entry * 100.0,
                   p.min_stop_distance_pct), p.max_stop_distance_pct) / 100.0
    stop = entry * (1.0 - dist)
    H = p.horizon_sessions
    lo = low[sym].iloc[i + 1: i + 1 + H].to_numpy(dtype="float64")
    op = open_[sym].iloc[i + 1: i + 1 + H].to_numpy(dtype="float64")
    cl = close[sym].iloc[i + 1: i + 1 + H].to_numpy(dtype="float64")
    mv = ma[sym].iloc[i + 1: i + 1 + H].to_numpy(dtype="float64")
    av = atr[sym].iloc[i + 1: i + 1 + H].to_numpy(dtype="float64")
    finite = np.isfinite(cl)
    if finite.sum() < H * 0.5:
        return None
    for k in range(len(cl)):
        # A gap through the stop fills at the open, not at the stop. Assuming
        # the stop price is the optimistic error, and the optimistic error is
        # the one that matters.
        if np.isfinite(lo[k]) and lo[k] <= stop:
            fill = min(op[k], stop) if np.isfinite(op[k]) else stop
            return float(fill / entry - 1.0)
        if (np.isfinite(cl[k]) and np.isfinite(mv[k]) and np.isfinite(av[k])
                and cl[k] < mv[k] - p.invalidation_buffer_atr * av[k]):
            return float(cl[k] / entry - 1.0)
    return float(cl[finite][-1] / entry - 1.0)


def simulate(
    rankings: Sequence[Tuple[pd.Timestamp, pd.Series]],
    prices: Dict[str, pd.DataFrame],
    params: PortfolioParams,
    *,
    phase: int = 0,
    step_sessions: int = 21,
    dates_allowed: Optional[Sequence[pd.Timestamp]] = None,
) -> PortfolioResult:
    """Run the book across rebalances, one cohort at a time.

    ``rankings`` is (date, score series sorted best first). ``prices`` holds the
    aligned panels: close, low, open, atr, ma, adtv. ``phase`` selects which
    offset of the non-overlapping schedule to walk.
    """
    close, low, open_ = prices["close"], prices["low"], prices["open"]
    atr, ma, adtv = prices["atr"], prices["ma"], prices["adtv"]
    index = list(close.index)
    pos = {d: i for i, d in enumerate(index)}
    allowed = set(dates_allowed) if dates_allowed is not None else None

    stride = max(int(np.ceil(params.horizon_sessions / step_sessions)), 1)
    equity = params.capital
    held: Dict[str, int] = {}
    rows: List[Dict[str, float]] = []

    for j in range(phase, len(rankings), stride):
        date, scores = rankings[j]
        if date not in pos or (allowed is not None and date not in allowed):
            continue
        i = pos[date]
        if i + params.horizon_sessions >= len(index):
            continue
        rank = {sym: r for r, sym in enumerate(scores.index, start=1)}
        # Hysteresis: a held name survives while inside the wider exit band.
        keep = [s for s in held if rank.get(s, 10 ** 9) <= params.exit_rank]
        room = params.max_positions - len(keep)
        add = [s for s in list(scores.index)[: params.entry_rank]
               if s not in keep][: max(room, 0)]
        book = keep + add

        scale = equity / params.capital
        # AGGREGATE exposure, scaled by how turbulent the market has been. The
        # window ends at i and reads only closes at or before the decision date.
        vol_scale, realised_vol = _volatility_scale(close, i, params)
        scale *= vol_scale
        pnl = deployed = charged = 0.0
        filled = 0
        for sym in book:
            if sym not in close.columns:
                continue
            sized = _position(sym, i, close, atr, adtv, params)
            if sized is None or sized[0] <= 0:
                continue
            size, price, liquidity = sized
            ret = _hold(sym, i, close, low, open_, ma, atr, params)
            if ret is None:
                continue
            size *= scale
            pnl += size * ret
            deployed += size
            filled += 1
            if sym not in held:                      # only new names pay entry
                bps = params.cost_bps(price, size / price if price > 0 else 0.0,
                                      liquidity)
                charged += size * bps / 10_000.0
        if filled == 0:
            continue
        gross = pnl
        pnl -= charged
        opening = equity
        equity += pnl
        rows.append({
            "date": date, "ret": pnl / opening, "equity": equity,
            # The cost drag, kept separately. Netting it into `ret` and
            # discarding the parts makes the buy/hold spread unmeasurable: a
            # wider exit band earns its keep by NOT paying entry cost on a name
            # it already holds, and that saving is invisible once the two are
            # added together.
            "gross_ret": gross / opening, "cost_ret": charged / opening,
            "n_held": filled, "n_new": len([s for s in book if s not in held]),
            "deployed_frac": deployed / opening,
            "vol_scale": vol_scale, "realised_vol": realised_vol,
        })
        held = {s: 1 for s in book}

    return PortfolioResult(periods=pd.DataFrame(rows))


def phase_summary(
    rankings: Sequence[Tuple[pd.Timestamp, pd.Series]],
    prices: Dict[str, pd.DataFrame],
    params: PortfolioParams,
    *,
    step_sessions: int = 21,
    dates_allowed: Optional[Sequence[pd.Timestamp]] = None,
) -> Dict[str, float]:
    """Every phase offset, pooled. One offset is one arbitrary schedule."""
    stride = max(int(np.ceil(params.horizon_sessions / step_sessions)), 1)
    results = [
        simulate(rankings, prices, params, phase=p, step_sessions=step_sessions,
                 dates_allowed=dates_allowed)
        for p in range(stride)
    ]
    usable = [r for r in results if not r.empty and len(r.periods) >= 3]
    if not usable:
        return {}
    pooled = pd.concat([r.periods for r in usable], ignore_index=True)
    r = pooled["ret"].to_numpy(dtype="float64")
    sd = float(r.std(ddof=1))
    # Annualise by the horizon actually held, not by a constant. sqrt(4) is
    # correct only at H=63; at H=21 there are twelve periods a year and the
    # factor is sqrt(12), so a fixed 4 understates a short horizon by 1.73x and
    # overstates a long one. That error made Sharpe look like it rose
    # monotonically with horizon; corrected, it peaks near 63 and falls away.
    periods_per_year = 252.0 / float(params.horizon_sessions)
    per_phase = [x.metrics(periods_per_year=periods_per_year) for x in usable]
    return {
        "mean_return": float(r.mean()),
        "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "periods_per_year": periods_per_year,
        "max_drawdown": float(np.mean([m["max_drawdown"] for m in per_phase])),
        "worst_phase_sharpe": float(min(m["sharpe"] for m in per_phase)),
        "hit_rate": float((r > 0).mean()),
        "avg_names": float(pooled["n_held"].mean()),
        "avg_new": float(pooled["n_new"].mean()),
        # Gross and cost carried through pooling, so the buy/hold spread can be
        # priced: a wider exit band buys its edge by NOT paying entry cost on a
        # name it already holds, and that saving is invisible in `mean_return`.
        "mean_gross": (float(pooled["gross_ret"].mean())
                       if "gross_ret" in pooled else float("nan")),
        "mean_cost": (float(pooled["cost_ret"].mean())
                      if "cost_ret" in pooled else float("nan")),
        "cost_share_of_gross": (
            float(pooled["cost_ret"].sum() / pooled["gross_ret"].sum())
            if "gross_ret" in pooled and float(pooled["gross_ret"].sum()) > 0
            else float("nan")),
        "n_periods": int(len(r)),
        "n_phases": len(usable),
    }
