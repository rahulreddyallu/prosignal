"""One definition of what happened to a trade. Used by the label AND the book.

WHY THIS EXISTS. The codebase carried three different answers to "how did this
position end", and they disagreed:

    labels.triple_barrier     stop at 0.75 * sigma_H, target at 1.0 * sigma_H,
                              timeout. No invalidation exit.
    portfolio_sim._hold       stop at 2.5 * ATR gap-filled at the open,
                              invalidation, horizon. NO profit target at all.
    stage7_risk               stop at 2.5 * ATR, target at 3.0R, invalidation,
                              trailing stop, signal reversal, regime change.

The model was fitted against the first, the backtest measured the second, and
the engine traded the third. Measured on 156,446 labelled observations, the
first two disagree on 16% of outcomes, and 14% of everything the LABEL called a
winner would have been stopped out by the stop the engine actually places.

The two errors compounded rather than cancelled. The sigma barriers gave the
label a 1.33:1 reward-to-risk profile while the engine trades 3.0:1 by
construction (`t2_r_multiple`), and those two numbers select opposite styles:
1.33:1 rewards high-hit-rate names that grind out small wins, 3.0:1 rewards
low-hit-rate trend names. The model was being trained to find one and the engine
was executing the other.

THE FIX IS DIRECTIONAL. The engine's risk rules are what actually protect
capital and they are calibrated in ATR and R-multiples; the label is a
measurement device. So the measurement is moved to match the execution, not the
other way around -- changing the stop the engine places would change what is
traded, which is a far larger decision than fixing a training label.

WHAT IS STILL NOT MODELLED. Trailing stop, signal reversal, new hard rejection
and severe regime change are all real exits the engine can take, and none of
them is here. Each closes a position EARLIER than this module assumes, so the
labels remain optimistic -- less so than before, and in a stated direction
rather than an unmeasured one. The Stage 6 rank band is likewise absent; the
buy/hold spread measurement covers that separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators.atr import true_range

__all__ = ["ExitRules", "resolve_exits", "rules_from_config",
           "atr_panel", "ma_panel",
           "EXIT_STOP", "EXIT_TARGET", "EXIT_INVALIDATION", "EXIT_TIMEOUT"]

EXIT_STOP = -1.0
EXIT_TIMEOUT = 0.0
EXIT_TARGET = 1.0
#: Thesis invalidation is a LOSS in intent even when the price is above entry:
#: the reason for the trade is gone. It is recorded separately from the stop so
#: the meta-label and the outcome record can tell them apart, and it is scored
#: as "not a win" wherever a binary is needed.
EXIT_INVALIDATION = -2.0


@dataclass(frozen=True)
class ExitRules:
    """The engine's own exit geometry, in the engine's own units.

    Every field maps to a shipped config value, so a change to the traded stop
    moves the training label with it. That coupling is the point.
    """

    #: stage7_risk.stop_loss
    stop_atr_multiple: float = 2.5
    min_stop_distance_pct: float = 2.0
    max_stop_distance_pct: float = 15.0
    #: stage7_risk.targets.t2_r_multiple -- the profit target, in units of the
    #: stop distance. This is what fixes reward-to-risk at 3:1 rather than the
    #: 1.33:1 the sigma barriers implied.
    target_r_multiple: float = 3.0
    #: stage7_risk.thesis_invalidation
    invalidation_ma_sessions: int = 50
    invalidation_buffer_atr: float = 1.5
    #: stage7_risk.atr
    atr_period_sessions: int = 14
    atr_method: str = "wilder"
    #: stage4_core_score.model_horizon_sessions
    horizon: int = 63

    def stop_fraction(self, atr: np.ndarray, entry: np.ndarray) -> np.ndarray:
        """Stop distance as a fraction of entry, clipped exactly as Stage 7 does."""
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = self.stop_atr_multiple * atr / entry * 100.0
        return np.clip(raw, self.min_stop_distance_pct,
                       self.max_stop_distance_pct) / 100.0


def atr_panel(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
              period: int = 14, method: str = "wilder") -> pd.DataFrame:
    """ATR per symbol, from the SAME `true_range` Stage 7 uses.

    Wilder smoothing is a recursion, so it is applied column-wise rather than
    reimplemented as a matrix operation -- a subtly different ATR here would
    reintroduce exactly the divergence this module exists to remove.
    """
    tr = pd.DataFrame(
        {c: true_range(high[c], low[c], close[c]) for c in close.columns},
        index=close.index)
    key = str(method).strip().lower()
    if key == "sma":
        return tr.rolling(window=period, min_periods=period).mean()
    if key != "wilder":
        raise ValueError(f"unknown ATR method {method!r}")
    # Wilder's smoothing is an EWM with alpha = 1/period, seeded on the first
    # `period` true ranges. `adjust=False` is the recursive form.
    return tr.ewm(alpha=1.0 / float(period), adjust=False,
                  min_periods=period).mean()


def ma_panel(close: pd.DataFrame, sessions: int) -> pd.DataFrame:
    return close.rolling(window=int(sessions), min_periods=int(sessions)).mean()


def resolve_exits(
    close: pd.DataFrame,
    i: int,
    rules: ExitRules,
    *,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    open_: Optional[pd.DataFrame] = None,
    atr: Optional[pd.DataFrame] = None,
    ma: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Resolve every symbol's trade opened at row ``i``. Vectorised.

    Reads ``close.iloc[i]`` for the entry and nothing before it for the outcome;
    reads no row past ``i + horizon``. Returns a frame indexed by symbol with
    ``ret``, ``side``, ``held`` and ``t1``.

    ORDER WITHIN A BAR. Stop, then target, then invalidation. Daily bars cannot
    order intraday events, and assuming the favourable one inflates every
    number built on top of this -- the same convention the outcome record uses.
    A bar that gaps through the stop fills at the OPEN, not at the stop price:
    assuming the stop price is the optimistic error, and the optimistic error is
    the one that matters.
    """
    n = len(close)
    end = min(i + rules.horizon, n - 1)
    cols = close.columns
    empty = pd.DataFrame(columns=["ret", "side", "held", "t1"],
                         index=cols, dtype="float64")
    if end <= i:
        return empty

    hi = (high if high is not None else close).iloc[i + 1: end + 1].to_numpy("float64")
    lo = (low if low is not None else close).iloc[i + 1: end + 1].to_numpy("float64")
    cl = close.iloc[i + 1: end + 1].to_numpy("float64")
    op = (open_.iloc[i + 1: end + 1].to_numpy("float64")
          if open_ is not None else cl)
    e = close.iloc[i].to_numpy("float64")

    if atr is None:
        if high is None or low is None:
            raise ValueError(
                "resolve_exits needs high/low to compute ATR, or a precomputed "
                "atr panel. Falling back to a close-only stop would rebuild the "
                "close-only optimism this module exists to remove.")
        atr = atr_panel(high, low, close, rules.atr_period_sessions,
                        rules.atr_method)
    a = atr.iloc[i].reindex(cols).to_numpy("float64")

    if ma is None:
        ma = ma_panel(close, rules.invalidation_ma_sessions)

    dist = rules.stop_fraction(a, e)
    stop_lvl = e * (1.0 - dist)
    target_lvl = e * (1.0 + rules.target_r_multiple * dist)

    n_bars = cl.shape[0]
    hit_stop = lo <= stop_lvl
    hit_target = hi >= target_lvl

    mv = ma.iloc[i + 1: end + 1].reindex(columns=cols).to_numpy("float64")
    av = atr.iloc[i + 1: end + 1].reindex(columns=cols).to_numpy("float64")
    with np.errstate(invalid="ignore"):
        hit_inval = cl < (mv - rules.invalidation_buffer_atr * av)
    hit_inval = np.where(np.isfinite(mv) & np.isfinite(av) & np.isfinite(cl),
                         hit_inval, False)

    def _first(mask: np.ndarray) -> np.ndarray:
        return np.where(mask.any(axis=0), mask.argmax(axis=0), n_bars + 1)

    f_stop, f_target, f_inval = _first(hit_stop), _first(hit_target), _first(hit_inval)
    first = np.minimum(np.minimum(f_stop, f_target), f_inval)
    timed_out = first > n_bars

    # Ties inside one bar resolve to the WORST outcome available.
    stopped = (~timed_out) & (f_stop == first)
    invalidated = (~timed_out) & (~stopped) & (f_inval == first)
    took_profit = (~timed_out) & (~stopped) & (~invalidated)

    idx = np.clip(first, 0, max(n_bars - 1, 0))
    take = lambda arr: arr[idx, np.arange(arr.shape[1])]

    # A gap through the stop fills at the open when the open is already below.
    gap_fill = np.minimum(take(op), stop_lvl)
    stop_fill = np.where(np.isfinite(take(op)), gap_fill, stop_lvl)

    last_close = cl[-1] if n_bars else np.full(len(e), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(timed_out, last_close / e - 1.0, np.nan)
        ret = np.where(stopped, stop_fill / e - 1.0, ret)
        ret = np.where(invalidated, take(cl) / e - 1.0, ret)
        ret = np.where(took_profit, target_lvl / e - 1.0, ret)

    side = np.where(timed_out, EXIT_TIMEOUT,
                    np.where(stopped, EXIT_STOP,
                             np.where(invalidated, EXIT_INVALIDATION, EXIT_TARGET)))
    held = np.where(timed_out, float(n_bars), first.astype("float64") + 1.0)
    t1 = np.where(timed_out, float(end), (i + 1 + first).astype("float64"))

    # A NAME ALREADY BELOW ITS INVALIDATION LEVEL ON THE DECISION DATE IS NOT A
    # TRADE. Stage 6 would never trigger an entry on it, and labelling it as a
    # position that invalidates on day one puts a trend filter inside the label:
    # measured across the panel it made invalidation 52% of all outcomes at a
    # median hold of THREE sessions, and momentum's coefficient collapsed
    # because the label had started to contain what momentum was meant to
    # predict. Excluding them leaves invalidation at 37-40% with a median hold
    # of 20, which is the engine's real experience.
    entry_ma = ma.iloc[i].reindex(cols).to_numpy("float64")
    entry_atr = a
    with np.errstate(invalid="ignore"):
        invalid_at_entry = e < (entry_ma - rules.invalidation_buffer_atr * entry_atr)
    invalid_at_entry = np.where(np.isfinite(entry_ma) & np.isfinite(entry_atr),
                                invalid_at_entry, False)

    usable = (np.isfinite(e) & (e > 0) & np.isfinite(a) & ~invalid_at_entry)
    out = pd.DataFrame({"ret": ret, "side": side, "held": held, "t1": t1},
                       index=cols)
    return out.where(pd.Series(usable, index=cols), np.nan)


def rules_from_config(c4, c7) -> ExitRules:
    """Build the exit geometry from the shipped config. ONE reader.

    `c4` is stage4_core_score and `c7` is stage7_risk. Every caller -- the
    refit, the research panel, the portfolio simulator -- goes through here, so
    a change to the traded stop cannot reach one of them and miss another.
    """
    from ..stages._cfg import fv, iv

    return ExitRules(
        stop_atr_multiple=fv(c7.stop_loss.atr_multiple),
        min_stop_distance_pct=fv(c7.stop_loss.min_stop_distance_pct),
        max_stop_distance_pct=fv(c7.stop_loss.max_stop_distance_pct),
        target_r_multiple=fv(c7.targets.t2_r_multiple),
        invalidation_ma_sessions=iv(c7.thesis_invalidation.structure_ma_sessions),
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        atr_period_sessions=iv(c7.atr.period_sessions),
        atr_method=str(c7.atr.method.value),
        horizon=iv(c4.model_horizon_sessions),
    )
