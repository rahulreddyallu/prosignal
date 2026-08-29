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

WHAT IS STILL NOT MODELLED, AND HOW BIG IT IS. Trailing stop, signal reversal,
new hard rejection and severe regime change are all real exits the engine can
take, and none of them is here. Each closes a position EARLIER than this module
assumes, so the labels remain optimistic.

The largest omission by far is the Stage 6 RANK BAND. Measured on the 118
outcomes resolved under `outcomes.EXIT_MODEL = target-t2-v4`:

    book_exit (the rank band)   105   89%
    stop                         12   10%
    stop_gap                      1    1%
    target (T2, 3.0R)             0    0%   <- what this label rewards
    reached T1 at any point       7    6%
    median sessions held          3         <- against a 63-session horizon

So the label scores an outcome the book never experiences: not one trade in 118
reached the target the model is fitted against, and nine in ten ended for a
reason this module does not represent.

SHORTENING THE HORIZON IS NOT THE FIX, which is worth stating because it is the
obvious move. Refitting at 21, 42 and 63 sessions and putting each through the
same book simulation, out-of-sample over purged folds:

    horizon    net     Sharpe    mom   reversal  lottery   risk  delivery
    21       +0.31%    +0.39    +0.70    -2.96*   -4.64*   +1.78   +4.21*
    42       +0.22%    +0.14    +0.36    -4.81*   -5.36*   +2.20*  +5.37*
    63       +0.57%    +0.30    +0.04    -4.79*   -5.02*   +2.20*  +5.09*

63 earns the most and the theme structure is weaker at 21, where `risk` falls
below the significance floor. The horizon is a TIMEOUT, not a holding period --
shortening it truncates trades that would have resolved, which is why it costs
return. The mismatch is real and it is not the horizon's fault.

Modelling the band inside the label is circular: the band depends on the
ranking, which depends on the model, which depends on the label. It would need
an iterated fit. Until then the size of the gap is stated above rather than
described as "optimistic".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators.atr import true_range

__all__ = ["ExitRules", "resolve_exits", "rules_from_config",
           "tradeable_at_entry", "invalidation_level",
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

    #: stage7_risk.exit_hierarchy -- WHICH EXITS ARE ARMED AT ALL.
    #:
    #: These existed in `config/parameters.yaml` and were read by exactly one
    #: place: `stage7_risk._exit_hierarchy`, which decides what the CARD prints.
    #: Nothing in the measurement path read them -- not `rules_from_config`, not
    #: `resolve_exits`, not the label, not the portfolio simulator. So setting
    #: `stop_loss_breach: false` removed the stop from the card and left it in
    #: every backtest, every label and every validation number, and an operator
    #: who turned it off would have seen the measurements not move and concluded
    #: the stop was free.
    #:
    #: All three ship armed, so honouring them here leaves
    #: `baseline-v1@127d8a314ec49aa2` byte-identical. It is a config value
    #: meaning what it says, not a change to what is traded.
    use_stop: bool = True
    use_target: bool = True
    use_invalidation: bool = True

    def stop_fraction(self, atr: np.ndarray, entry: np.ndarray) -> np.ndarray:
        """Stop distance as a fraction of entry, clipped exactly as Stage 7 does."""
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = self.stop_atr_multiple * atr / entry * 100.0
        return np.clip(raw, self.min_stop_distance_pct,
                       self.max_stop_distance_pct) / 100.0


def invalidation_level(ma, atr, rules: ExitRules):
    """Where the thesis is dead. `MA(n) - buffer * ATR`, the engine's own rule.

    One definition, because it decides two different things that MUST agree:
    whether an open position is closed, and whether a new one may be opened at
    all. Stage 7 emits it as exit condition #1.
    """
    return ma - float(rules.invalidation_buffer_atr) * atr


def tradeable_at_entry(close, ma, atr, rules: ExitRules):
    """Whether a bar is a valid ENTRY under the engine's own invalidation rule.

    THE POPULATION DEFINITION. A name already below its invalidation level is
    not a trade: it satisfies its own first exit condition at the moment it is
    opened, and the card would print "close below X means the thesis is dead"
    with X sitting above the entry price.

    This existed only inside `resolve_exits`, as an inline mask excluding such
    rows from the LABEL, justified there by "Stage 6 would never trigger an
    entry on it". That was true when Stage 6 required a pullback, a reclaim or
    a breakout. Stage 6 now admits on rank alone, so nothing enforced it on the
    live path -- and because the exclusion also removes those rows from the
    training panel, every validation that derives rankings from that panel
    inherits it too. Measured on the eligible universe, 21.8% of the selection
    period and 26.9% of the holdout sits below the level, so the validated
    strategy and the live one were drawing from populations that differ by
    roughly a fifth, and no harness could see it: restricting an
    already-restricted panel finds 1.8% left to remove.

    Returns a boolean aligned to whatever was passed -- a Series for one bar,
    a frame for a panel. NaN inputs read as NOT tradeable, because an unknown
    invalidation level is not a cleared one.
    """
    level = invalidation_level(ma, atr, rules)
    ok = close >= level
    try:
        return ok & level.notna() & close.notna()
    except AttributeError:            # numpy inputs
        return np.asarray(ok) & np.isfinite(level) & np.isfinite(close)


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

    never = np.full(len(e), n_bars + 1)
    # A DISARMED EXIT NEVER FIRES, rather than firing and being ignored. See
    # `ExitRules.use_stop`: these switches come from `exit_hierarchy`, which
    # until now reached the card and nothing else.
    f_stop = _first(hit_stop) if rules.use_stop else never
    f_target = _first(hit_target) if rules.use_target else never
    f_inval = _first(hit_inval) if rules.use_invalidation else never
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
    # THE SAME PREDICATE THE LIVE ADMISSION USES. See `tradeable_at_entry`:
    # this was an inline mask here and nothing enforced it on the live path,
    # so the label and the engine drew from populations differing by ~22%.
    entry_ma = ma.iloc[i].reindex(cols).to_numpy("float64")
    with np.errstate(invalid="ignore"):
        admissible = tradeable_at_entry(e, entry_ma, a, rules)
    # An unknown level cannot exclude a row from the LABEL -- the older bars
    # have no 50-session average yet and dropping them would shorten the panel
    # for everyone. Live admission is stricter and refuses the unknown.
    admissible = np.where(np.isfinite(entry_ma) & np.isfinite(a),
                          admissible, True)

    usable = (np.isfinite(e) & (e > 0) & np.isfinite(a) & admissible)
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

    def _armed(node) -> bool:
        return bool(getattr(node, "value", node))

    h = c7.exit_hierarchy
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
        # `exit_hierarchy` decided what the CARD printed and nothing else. A
        # stop switched off in the config stayed switched on in every backtest.
        use_stop=_armed(h.stop_loss_breach),
        use_target=_armed(h.target_achieved),
        use_invalidation=_armed(h.thesis_invalidation),
    )
