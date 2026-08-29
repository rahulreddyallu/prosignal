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

from ..features.exits import EXIT_TIMEOUT
from ..liquidity import assess

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
    #: Profit target in units of the stop distance (stage7_risk.targets.
    #: t2_r_multiple). The simulator had NO target, so it measured a strategy
    #: that never takes profit while Stage 7 emits a target exit at 3R.
    target_r_multiple: float = 3.0
    #: WHICH EXITS ARE ARMED, from `exit_hierarchy`. These were absent, so the
    #: simulator always took profit at 3R and always sold on invalidation --
    #: which was harmless while both rungs were armed and became a measurement
    #: of a strategy the engine does not run the moment they were disarmed. The
    #: defaults are True so an existing caller that does not pass them measures
    #: what it used to; every shipped caller passes them.
    use_stop: bool = True
    use_target: bool = True
    use_invalidation: bool = True

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
    #: Refuse a name whose ADTV is missing, zero, negative or non-finite,
    #: rather than sizing it at the full capital slot. See `_position`, and
    #: `liquidity.assess` for why "missing" and "zero" are different states.
    #:
    #: Defaults TRUE because the alternative is indefensible. It is a switch at
    #: all so the cost of the old behaviour stays measurable: `work/` prices it,
    #: and a correction whose price can no longer be recomputed is a correction
    #: nobody can check.
    refuse_unknown_liquidity: bool = True
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
            **self._benchmark_block(periods_per_year),
        }

    def _benchmark_block(self, periods_per_year: float) -> Dict[str, float]:
        """What the book earned ABOVE the alternative, or nothing if unknown.

        A Sharpe ratio answers "was this better than cash". It does not answer
        "was this better than owning the universe equal-weighted", and for a
        long-only book selected from that universe the second question is the
        one that decides whether the ranking is worth running. It is reported
        here rather than by the caller so that a summary cannot omit it by not
        computing it.

        The fields are ABSENT, not nan, when no benchmark was supplied.
        `benchmarked` says which case this is. A nan in a metrics dict gets
        formatted, averaged and quietly dropped; a missing key does not, and a
        reader who sees `benchmarked: False` knows the comparison was never
        made rather than that it came out flat.
        """
        if "bench_ret" not in self.periods:
            return {"benchmarked": False}
        b = self.periods["bench_ret"].to_numpy(dtype="float64")
        r = self.periods["ret"].to_numpy(dtype="float64")
        ok = np.isfinite(b) & np.isfinite(r)
        if int(ok.sum()) < 3:
            return {"benchmarked": False}
        return _benchmark_stats(r[ok], b[ok], periods_per_year)


def _benchmark_stats(r: np.ndarray, b: np.ndarray, periods_per_year: float
                     ) -> Dict[str, float]:
    """The single definition of every benchmark-relative figure.

    One function, used by `PortfolioResult.metrics` and by `phase_summary`, so
    a per-phase number and a pooled number cannot come to mean different
    things. That divergence is how the repository ended up with three
    incompatible CPCV results.

    `information_ratio` is mean excess over the standard deviation of excess,
    annualised. `beta_to_benchmark` and `alpha_per_period` come from the same
    regression, so a book that is only long beta shows it.
    """
    ex = r - b
    sd_ex = float(ex.std(ddof=1))
    sd_b = float(b.std(ddof=1))
    var_b = float(b.var(ddof=1))
    beta = float(np.cov(r, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    alpha = float(r.mean() - beta * b.mean()) if np.isfinite(beta) else float("nan")
    return {
        "benchmarked": True,
        "bench_mean_return": float(b.mean()),
        "bench_sharpe": (float(b.mean() / sd_b * np.sqrt(periods_per_year))
                         if sd_b > 0 else float("nan")),
        "mean_excess": float(ex.mean()),
        "information_ratio": (float(ex.mean() / sd_ex * np.sqrt(periods_per_year))
                              if sd_ex > 0 else float("nan")),
        "excess_hit_rate": float((ex > 0).mean()),
        "beta_to_benchmark": beta,
        "alpha_per_period": alpha,
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
    """(rupees deployed, price, adtv), honouring risk budget, slot and liquidity.

    A NAME WHOSE LIQUIDITY CANNOT BE MEASURED IS NOT SIZED, IT IS REFUSED.

    This used to fall back to `qty_liq = slot / entry` -- the largest position
    the capital slot allows -- for exactly the names with no ADTV, while
    `costs.impact_bps` handed the same names the cheapest fill in the model.
    Largest size and best execution, awarded for an absence of information.

    Refusing them is worth +0.17% per 63-session period on both ranking
    constructions and costs about six points of deployed capital, so the names
    it was admitting were on average ones the book was better off without. That
    is a happy accident: the argument for refusing does not rest on it, and
    would stand if the number went the other way.
    """
    entry = close[sym].iloc[i]
    a = atr[sym].iloc[i]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(a):
        return None
    dist = min(max(p.stop_atr_multiple * a / entry * 100.0,
                   p.min_stop_distance_pct), p.max_stop_distance_pct) / 100.0
    risk_per_share = entry * dist
    if risk_per_share <= 0:
        return None

    raw = adtv[sym].iloc[i]
    view = assess(None if not np.isfinite(raw) else float(raw))
    if not (view.tradable if p.refuse_unknown_liquidity else True):
        return None
    if view.adtv_inr is None:
        # Only reachable with the gate switched off, which exists so the cost
        # of the old behaviour can still be measured. Keep the old arithmetic
        # exactly, so that measurement means what it says.
        qty_liq = p.slot / entry
        known = 0.0
    else:
        qty_liq = (view.adtv_inr * p.max_participation_of_adtv) / entry
        known = view.adtv_inr
    qty = max(min(p.risk_budget / risk_per_share, p.slot / entry, qty_liq), 0.0)
    return float(qty * entry), float(entry), float(known)


def _hold(sym: str, i: int, close, low, open_, ma, atr, p: PortfolioParams,
          high=None) -> Optional[Tuple[float, float]]:
    """(realised return, exit side) of one position, from the SHARED resolver.

    This used to carry its own copy of the exit logic -- stop, invalidation,
    horizon, and no profit target at all -- while the training label carried a
    different copy and Stage 7 a third. `features.exits` is the single
    definition now; this is the per-symbol adapter onto it.

    Note that the simulator therefore TAKES PROFIT now, at `target_r_multiple`.
    It did not before, so a position that reached 3R was carried to the horizon
    and whatever happened next was booked. That flattered nothing consistently:
    it overstated the winners that kept running and understated the ones that
    gave it back.

    ``high`` IS NOT OPTIONAL AND USED TO BE PASSED AS None. `resolve_exits`
    substitutes the close when it is missing, so the 3R target could only
    trigger on a CLOSE while the stop still triggered on the intraday LOW. The
    training label passes `high`; this call site did not; every test in
    `test_exit_agreement` passes it on both sides and so could not see the
    difference. One module built to hold ONE definition of what happened to a
    trade, fed different data by its two callers. Measured on the book: the
    target under-triggers and the shipped construction understates the book's
    return by 0.10-0.43% per period.

    Returning the SIDE as well is what makes turnover measurable. The caller
    charges a round trip only to names absent from the previous book, which is
    correct for a position carried through -- and wrong for the 84% that close
    early and are re-bought. Without the side it cannot tell the two apart.
    """
    from ..features.exits import ExitRules, resolve_exits

    rules = ExitRules(
        stop_atr_multiple=p.stop_atr_multiple,
        min_stop_distance_pct=p.min_stop_distance_pct,
        max_stop_distance_pct=p.max_stop_distance_pct,
        target_r_multiple=p.target_r_multiple,
        invalidation_ma_sessions=p.invalidation_ma_sessions,
        invalidation_buffer_atr=p.invalidation_buffer_atr,
        horizon=p.horizon_sessions,
        use_stop=p.use_stop,
        use_target=p.use_target,
        use_invalidation=p.use_invalidation,
    )
    one = [sym]
    out = resolve_exits(close[one], i, rules,
                        high=(high[one] if high is not None else None),
                        low=low[one], open_=open_[one], atr=atr[one], ma=ma[one])
    if not len(out):
        return None
    value = out["ret"].iloc[0]
    side = out["side"].iloc[0]
    return None if not np.isfinite(value) else (float(value), float(side))


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
    high = prices.get("high")
    #: THE ALTERNATIVE. A per-period return series for the equal-weight eligible
    #: universe over the SAME holding window, so every figure this simulator
    #: produces can be read against what doing nothing clever would have paid.
    #: Its absence is why a book returning +1.59% per period was reported as a
    #: positive result for eleven months while the universe it selects from
    #: returned +5.27% over the same windows.
    bench = prices.get("benchmark")
    # The intraday high. Absent, `resolve_exits` falls back to the close and the
    # profit target becomes a close-only instrument while the stop stays
    # intraday -- see `_hold`. A caller that cannot supply it gets the old
    # asymmetry, and is told rather than silently given it.
    high = prices.get("high")
    if high is None:
        import warnings
        warnings.warn(
            "portfolio_sim.simulate: no 'high' panel supplied, so the profit "
            "target can only trigger on a close while the stop still triggers "
            "on the intraday low. The book's return is understated and the "
            "target layer's measured cost is overstated.",
            RuntimeWarning, stacklevel=2)
    index = list(close.index)
    pos = {d: i for i, d in enumerate(index)}
    allowed = set(dates_allowed) if dates_allowed is not None else None

    stride = max(int(np.ceil(params.horizon_sessions / step_sessions)), 1)
    equity = params.capital
    #: symbol -> the side its last position exited on. EXIT_TIMEOUT means the
    #: position was still open at the horizon and a re-selection genuinely costs
    #: nothing; anything else means it closed and re-buying is a new round trip.
    held: Dict[str, float] = {}
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
        filled = new_or_reopened = 0
        outcomes: Dict[str, float] = {}
        for sym in book:
            if sym not in close.columns:
                continue
            sized = _position(sym, i, close, atr, adtv, params)
            if sized is None or sized[0] <= 0:
                continue
            size, price, liquidity = sized
            outcome = _hold(sym, i, close, low, open_, ma, atr, params, high=high)
            if outcome is None:
                continue
            ret, side = outcome
            size *= scale
            pnl += size * ret
            deployed += size
            filled += 1
            outcomes[sym] = side
            # A ROUND TRIP IS OWED WHENEVER A POSITION IS OPENED. That is any
            # name absent from the previous book, and also any name whose
            # previous position CLOSED before the horizon and is being bought
            # again. Rebalances are `ceil(horizon/step)` apart precisely so one
            # cohort finishes before the next opens, and 84% of positions close
            # early, so the second case is most of the book's real turnover. The
            # old test -- `sym not in held` -- charged none of it, and credited
            # the hysteresis band with a saving it does not make.
            reopened = held.get(sym)
            if reopened is None or reopened != EXIT_TIMEOUT:
                bps = params.cost_bps(price, size / price if price > 0 else 0.0,
                                      liquidity)
                charged += size * bps / 10_000.0
                new_or_reopened += 1
        if filled == 0:
            continue
        gross = pnl
        pnl -= charged
        opening = equity
        equity += pnl
        # The benchmark over the SAME window: entry at i, exit at the horizon,
        # equal-weight across whatever the universe held. Computed here rather
        # than annualised afterwards so it lines up period for period.
        bench_ret = float("nan")
        if bench is not None:
            j_exit = min(i + params.horizon_sessions, len(index) - 1)
            try:
                b0 = float(bench.iloc[i]); b1 = float(bench.iloc[j_exit])
                if np.isfinite(b0) and np.isfinite(b1) and b0 > 0:
                    bench_ret = b1 / b0 - 1.0
            except Exception:
                bench_ret = float("nan")
        rows.append({
            "date": date, "ret": pnl / opening, "equity": equity,
            "bench_ret": bench_ret, "excess_ret": (pnl / opening) - bench_ret,
            # The cost drag, kept separately. Netting it into `ret` and
            # discarding the parts makes the buy/hold spread unmeasurable: a
            # wider exit band earns its keep by NOT paying entry cost on a name
            # it already holds, and that saving is invisible once the two are
            # added together.
            "gross_ret": gross / opening, "cost_ret": charged / opening,
            "n_held": filled, "n_new": len([s for s in book if s not in held]),
            #: Positions that actually paid a round trip -- new names plus names
            #: whose previous position closed early and was re-bought. `n_new`
            #: counts only the first and understates real turnover.
            "n_charged": new_or_reopened,
            #: How much of the equity was working. The book is scored against a
            #: FULLY INVESTED benchmark, so cash held here is return given up,
            #: and it is given up under the label "position sizing". At 1% risk
            #: over 8 slots the risk-budget term binds above an 8% stop
            #: distance, which is most names -- measured, the book runs about
            #: three quarters invested.
            "deployed_frac": deployed / opening,
            "vol_scale": vol_scale, "realised_vol": realised_vol,
        })
        # Carry the EXIT SIDE, not a bare 1. A name still open at the horizon
        # costs nothing to keep; one that stopped out and is re-bought is a new
        # round trip.
        #
        # A name in `book` that never FILLED -- no ATR, no price, refused by the
        # admission predicate -- keeps its hysteresis slot exactly as before, and
        # is recorded as NaN rather than as a timeout. NaN != EXIT_TIMEOUT, so if
        # it fills at a later rebalance it pays: it was never bought, so buying
        # it is an opening trade. Recording it as a timeout would have made an
        # unfilled slot into a free entry.
        held = {s: outcomes.get(s, float("nan")) for s in book}

    return PortfolioResult(periods=pd.DataFrame(rows))


def _path_drawdown(usable: Sequence["PortfolioResult"]) -> float:
    """Worst peak-to-trough on any ONE schedule -- not the average of them.

    `phase_summary` used to report the MEAN of the per-phase drawdowns. A phase
    is one rebalance offset and an investor runs exactly one of them, so the
    mean describes a book nobody holds and always a milder one than the schedule
    that got unlucky: -18.6% averaged, against -21.7% on the worst of the three.
    Schedule choice is not a diversifiable risk here; it is a coin the investor
    flips once.

    WHY NOT CONCATENATE THE PHASES. The obvious alternative -- pool every period
    across phases, sort by date, compound -- is wrong, and wrong in the
    dangerous direction. The phases PARTITION the rebalance dates rather than
    running alongside each other, so each date appears in exactly one of them
    and the pooled series is 70 non-overlapping 63-session holds compounded
    end to end: about 17.5 years of compounding laid over a 6-year sample. It
    returns -35.4% here.

    That construction was checked against the thing it cannot be wrong about.
    Run it on the BENCHMARK returns from the same frame and it reports a -62.4%
    drawdown for the equal-weight Indian universe over 2019-2025, which did not
    happen -- the COVID trough was roughly -38%. A drawdown definition that
    fabricates 24 points of benchmark loss fabricates them for the book too.

    So: the worst single schedule. Conservative between the two defensible
    readings, and it is a number an investor could actually have lived through.
    """
    worst = float("nan")
    for x in usable:
        if x.empty:
            continue
        r = x.periods["ret"].to_numpy(dtype="float64")
        r = r[np.isfinite(r)]
        if r.size < 2:
            continue
        equity = np.cumprod(1.0 + r)
        d = float((equity / np.maximum.accumulate(equity) - 1.0).min())
        worst = d if not np.isfinite(worst) else min(worst, d)
    return worst


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
    drawdowns = [m["max_drawdown"] for m in per_phase]
    return {
        "mean_return": float(r.mean()),
        "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "periods_per_year": periods_per_year,
        # A MEAN OF SCHEDULES IS NOT A DRAWDOWN. Each phase is a different,
        # complete rebalance schedule -- one of them is the one that would have
        # been run -- so averaging their worst moments describes an experience
        # nobody could have had, and it is always shallower than the real one.
        #
        # WHICH ONE GETS THE HEADLINE NAME. Two audit passes disagreed here: one
        # kept `max_drawdown` on the mean so older write-ups reconciled, the
        # other moved it to the worst schedule. Both were left in the dict and
        # the later key silently won. Resolved on the rule used everywhere else
        # in this merge -- when two passes disagree, take the reading that
        # cannot flatter the result. `max_drawdown` is what gates and reports
        # quote, and a risk number that is quoted must not be the optimistic of
        # two defensible answers. The mean is not deleted; it keeps its own
        # names so an old report can still be reconciled against a new one.
        # Finding R10 records the move: -13.7% -> -19.1%.
        "max_drawdown": _path_drawdown(usable),
        "max_drawdown_path": _path_drawdown(usable),
        "worst_schedule_drawdown": float(np.min(drawdowns)),
        "max_drawdown_mean_of_phases": float(np.mean(drawdowns)),
        "max_drawdown_period": float(np.mean(drawdowns)),
        "worst_phase_sharpe": float(min(m["sharpe"] for m in per_phase)),
        "hit_rate": float((r > 0).mean()),
        "avg_names": float(pooled["n_held"].mean()),
        "avg_new": float(pooled["n_new"].mean()),
        #: Round trips actually paid for, which is new names PLUS re-entries
        #: after an early exit. It exceeds `avg_new` by however much of the
        #: book closes before the horizon and is bought back.
        "avg_charged": (float(pooled["n_charged"].mean())
                        if "n_charged" in pooled else float("nan")),
        #: Share of equity deployed. The benchmark is fully invested; anything
        #: below 1.0 here is return the book gave up by holding cash, and the
        #: decomposition attributes it to "sizing" unless it is read separately.
        "deployed_frac": (float(pooled["deployed_frac"].mean())
                          if "deployed_frac" in pooled else float("nan")),
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
        # The alternative, pooled the same way the book is. Absent only when no
        # benchmark panel was supplied.
        **({"benchmarked": False} if "bench_ret" not in pooled
           else _pooled_benchmark(pooled, periods_per_year)),
    }


def _pooled_benchmark(pooled: pd.DataFrame, periods_per_year: float
                      ) -> Dict[str, float]:
    """Benchmark-relative figures over the pooled phases, through the SAME
    `_benchmark_stats` a single phase uses."""
    b = pooled["bench_ret"].to_numpy(dtype="float64")
    r = pooled["ret"].to_numpy(dtype="float64")
    ok = np.isfinite(b) & np.isfinite(r)
    if int(ok.sum()) < 3:
        return {"benchmarked": False}
    return _benchmark_stats(r[ok], b[ok], periods_per_year)
