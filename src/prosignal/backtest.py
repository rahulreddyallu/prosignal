"""Walk-forward backtest driver -- connects the strategy to CPCV/PBO/DSR.

Execution model:

    session T close  ->  analysis runs, signal produced
    session T+1 OPEN ->  the trade is entered

Entry at the signal session's close would grant a full session of foresight,
which is enough on its own to make a losing strategy look profitable. Entry is
at the next session's open, which is what acting on an end-of-day signal can
achieve.

Exits are evaluated on subsequent bars in priority order. If a bar's low
touches the stop and its high touches the target, the stop is taken: intraday
sequence is unknowable from daily bars, and assuming the favourable ordering
inflates the win rate.

Costs and slippage come from the same `CostModel` the live engine uses.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config.loader import AppConfig
from .core.calendar import TradingCalendar
from .core.logging import get_logger
from .costs import CostModel
from .data.store import DataStore
from .data.types import DATE, SYMBOL
from .pipeline import PipelineBlocked, run_analysis

__all__ = ["Trade", "BacktestResult", "run_backtest"]

log = get_logger(__name__)


@dataclass
class Trade:
    ticker: str
    signal_date: dt.date
    entry_date: dt.date
    entry_price: float
    stop: float
    target_1: float
    target_2: float
    exit_date: Optional[dt.date] = None
    exit_price: Optional[float] = None
    exit_reason: str = "open"
    holding_sessions: int = 0
    gross_return: float = 0.0
    net_return: float = 0.0
    cost_bps: float = 0.0
    mae: float = 0.0  # maximum adverse excursion, as a fraction
    mfe: float = 0.0  # maximum favourable excursion

    def to_dict(self) -> Dict[str, object]:
        return {k: (v.isoformat() if isinstance(v, dt.date) else v)
                for k, v in self.__dict__.items()}


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    decision_dates: int = 0
    signal_dates: int = 0
    no_trade_dates: int = 0
    blocked_dates: int = 0
    start: Optional[dt.date] = None
    end: Optional[dt.date] = None
    notes: List[str] = field(default_factory=list)

    # -- statistics ---------------------------------------------------------
    def returns(self, net: bool = True) -> np.ndarray:
        vals = [t.net_return if net else t.gross_return for t in self.closed()]
        return np.array(vals, dtype="float64")

    def closed(self) -> List[Trade]:
        return [t for t in self.trades if t.exit_date is not None]

    def stats(self) -> Dict[str, object]:
        closed = self.closed()
        r = self.returns(net=True)
        g = self.returns(net=False)
        if r.size == 0:
            return {
                "n_trades": 0,
                "note": "no closed trades; no statistic is computable",
            }

        wins, losses = r[r > 0], r[r <= 0]
        gross_win = float(wins.sum()) if wins.size else 0.0
        gross_loss = float(-losses.sum()) if losses.size else 0.0
        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        hold = np.array([t.holding_sessions for t in closed], dtype="float64")

        # Per-trade Sharpe. NOT annualised: these are overlapping,
        # non-uniformly-spaced trade returns, and annualising them would imply
        # a compounding schedule the data does not support.
        sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
        sharpe_per_trade = float(r.mean() / sd) if sd > 0 else 0.0
        downside = r[r < 0]
        dsd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
        sortino = float(r.mean() / dsd) if dsd > 0 else 0.0

        return {
            "n_trades": int(r.size),
            "decision_dates": self.decision_dates,
            "signal_dates": self.signal_dates,
            "no_trade_dates": self.no_trade_dates,
            "win_rate": float((r > 0).mean()),
            "mean_return_net": float(r.mean()),
            "median_return_net": float(np.median(r)),
            "mean_return_gross": float(g.mean()),
            "cost_drag_per_trade": float(g.mean() - r.mean()),
            "expectancy": float(r.mean()),
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_win": float(wins.mean()) if wins.size else 0.0,
            "avg_loss": float(losses.mean()) if losses.size else 0.0,
            "largest_win": float(r.max()),
            "largest_loss": float(r.min()),
            "max_drawdown": float(dd.min()),
            "sharpe_per_trade": sharpe_per_trade,
            "sortino_per_trade": sortino,
            "avg_holding_sessions": float(hold.mean()),
            "median_holding_sessions": float(np.median(hold)),
            "avg_mae": float(np.mean([t.mae for t in closed])),
            "avg_mfe": float(np.mean([t.mfe for t in closed])),
            "exit_reasons": {
                reason: sum(1 for t in closed if t.exit_reason == reason)
                for reason in sorted({t.exit_reason for t in closed})
            },
        }


def run_backtest(
    config: AppConfig,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    step_sessions: int = 5,
    max_dates: Optional[int] = None,
    progress: Optional[Callable[[int, int, dt.date], None]] = None,
) -> BacktestResult:
    """Walk forward, generating signals and simulating their outcome.

    ``step_sessions`` samples decision dates rather than running every session.
    The pipeline takes seconds per date, and overlapping daily signals are
    heavily autocorrelated anyway -- weekly sampling costs little independent
    information and makes the run tractable. It is recorded in the result so the
    sample size is never overstated.
    """
    store = DataStore(config.paths.curated, config.paths.snapshots)
    sessions = store.price_sessions()
    calendar = TradingCalendar(sessions)
    need = int(config.params.universe.min_history_sessions.value)

    usable = sessions[need:]
    if start:
        usable = [d for d in usable if d >= start]
    if end:
        usable = [d for d in usable if d <= end]
    decision_dates = usable[::step_sessions]
    if max_dates:
        decision_dates = decision_dates[:max_dates]

    result = BacktestResult(
        decision_dates=len(decision_dates),
        start=decision_dates[0] if decision_dates else None,
        end=decision_dates[-1] if decision_dates else None,
    )
    result.notes.append(
        f"Decision dates sampled every {step_sessions} sessions from "
        f"{len(usable)} eligible dates. Entry is at the NEXT session's OPEN."
    )
    if not decision_dates:
        result.notes.append(
            f"No decision dates: the store holds {len(sessions)} sessions and "
            f"min_history_sessions is {need}."
        )
        return result

    costs = CostModel(config)
    bars = _load_bars(store, sessions[0], sessions[-1])

    for i, day in enumerate(decision_dates):
        if progress:
            progress(i + 1, len(decision_dates), day)
        try:
            run = run_analysis(config, as_of=day)
        except PipelineBlocked as exc:
            result.blocked_dates += 1
            log.debug("backtest date blocked", extra={"date": str(day), "why": exc.reasons})
            continue

        recs = run.output.recommendations
        if not recs:
            result.no_trade_dates += 1
            continue
        result.signal_dates += 1

        for rec in recs:
            trade = _simulate(rec, day, calendar, bars, costs, config)
            if trade is not None:
                result.trades.append(trade)

    return result


# =============================================================================
def _load_bars(store: DataStore, start: dt.date, end: dt.date) -> Dict[str, pd.DataFrame]:
    frame = store.read_prices(start=start, end=end)
    if frame.empty:
        return {}
    frame = frame.copy()
    frame[DATE] = pd.to_datetime(frame[DATE]).dt.normalize()
    out: Dict[str, pd.DataFrame] = {}
    for sym, chunk in frame.groupby(SYMBOL, sort=False, observed=True):
        out[str(sym)] = chunk.sort_values(DATE).reset_index(drop=True)
    return out


def _simulate(rec, signal_date, calendar, bars, costs, config) -> Optional[Trade]:
    """Enter at the next session's open; walk forward to an exit."""
    frame = bars.get(rec.ticker)
    if frame is None or rec.initial_stop is None or rec.target_1 is None:
        return None

    future = frame[frame[DATE] > pd.Timestamp(signal_date)]
    if future.empty:
        return None

    entry_row = future.iloc[0]
    entry_price = float(entry_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    stop = float(rec.initial_stop)
    t1 = float(rec.target_1)
    t2 = float(rec.target_2) if rec.target_2 else t1
    max_hold = int(config.params.stage7_risk.holding_period.max_holding_sessions.value)

    # Size the position the way the live engine would, so costs match.
    slot = float(config.params.capital.position_value_inr())
    qty = max(int(slot / entry_price), 1)

    mae = mfe = 0.0
    walk = future.iloc[1:].head(max_hold)
    exit_price = None
    exit_date = None
    reason = "open"
    held = 0

    for held, (_, bar) in enumerate(walk.iterrows(), start=1):
        low, high = float(bar["low"]), float(bar["high"])
        mae = min(mae, (low - entry_price) / entry_price)
        mfe = max(mfe, (high - entry_price) / entry_price)

        # Pessimistic ordering: a bar touching both stop and target counts as
        # the stop. Daily bars cannot tell us which came first, and assuming
        # the favourable sequence is how a backtest inflates its win rate.
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
        if walk.empty:
            return None
        last = walk.iloc[-1]
        exit_price, exit_date, reason = float(last["close"]), last[DATE].date(), "time_exit"
        held = len(walk)

    gross = exit_price / entry_price - 1.0
    cb = costs.round_trip(entry_price, qty, exit_price=exit_price, adtv_inr=None)
    cost_frac = cb.total_inr / (entry_price * qty)
    return Trade(
        ticker=rec.ticker, signal_date=signal_date,
        entry_date=entry_row[DATE].date(), entry_price=round(entry_price, 2),
        stop=stop, target_1=t1, target_2=t2,
        exit_date=exit_date, exit_price=round(exit_price, 2), exit_reason=reason,
        holding_sessions=held, gross_return=gross, net_return=gross - cost_frac,
        cost_bps=cb.total_bps_of_buy, mae=mae, mfe=mfe,
    )
