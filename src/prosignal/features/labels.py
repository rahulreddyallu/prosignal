"""Triple-barrier labelling and the uniqueness weights it forces on you.

WHY THE HORIZON RETURN IS THE WRONG LABEL. The engine promises a stop and a
holding period, and then fits against the return h sessions later as though
neither existed. That label is blind to the path: a name that drifts up quietly
and a name that falls 20% and claws back by day 63 get the same number, and the
second was never a position anyone held -- it was stopped out in week two.
Fitting against it teaches the model to like trades the engine would have closed
at a loss.

A triple-barrier label asks the question the engine actually asks. Set a profit
barrier, a stop barrier and a time barrier, and record whichever is touched
FIRST (Lopez de Prado, Advances in Financial Machine Learning, ch. 3). What
comes back is the return the trade would have realised, and a side saying which
barrier ended it.

THE BARRIERS ARE VOLATILITY-SCALED. A flat 8% means something different to a
large cap at 1.2% daily sigma and a midcap at 4%. Scaled in units of the name's
own volatility they mean the same thing everywhere. Calibration matters in both
directions: too tight against the noise and the labels are close to random, too
wide and everything times out and the label collapses back to the horizon
return it was meant to replace.

OVERLAP IS NOT OPTIONAL TO HANDLE. A label spanning 63 sessions, sampled every
21, shares two thirds of its window with its neighbour. Consecutive rows are not
independent draws, and an estimator that assumes they are counts one market
shock many times: the panel here has ~33,000 rows and nothing like 33,000
independent observations. `average_uniqueness` measures how much of each label's
span it holds alone, and the fit weights by it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "BarrierSpec", "triple_barrier", "average_uniqueness", "concurrency",
]


class BarrierSpec:
    """Barrier widths in units of the name's own daily volatility.

    ``upper`` and ``lower`` multiply the trailing daily sigma scaled to the
    horizon, so a 63-session label with ``upper=2.0`` takes profit at two
    horizon-sigmas. ``lower`` is given POSITIVE and applied downward.
    """

    __slots__ = ("upper", "lower", "horizon", "vol_window")

    def __init__(self, upper: float = 2.0, lower: float = 1.5,
                 horizon: int = 63, vol_window: int = 60) -> None:
        if upper <= 0 or lower <= 0:
            raise ValueError("barrier widths must be positive")
        if horizon <= 0:
            raise ValueError("the time barrier must be a positive number of sessions")
        self.upper = float(upper)
        self.lower = float(lower)
        self.horizon = int(horizon)
        self.vol_window = int(vol_window)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"BarrierSpec(upper={self.upper}, lower={self.lower}, "
                f"horizon={self.horizon}, vol_window={self.vol_window})")


def _horizon_sigma(close: pd.DataFrame, i: int, spec: BarrierSpec) -> pd.Series:
    """Trailing daily sigma scaled to the horizon, per symbol.

    Uses only rows at or before ``i``. sqrt-time scaling is the usual
    approximation and is wrong under autocorrelation, which is exactly what
    momentum is -- so the barrier is approximate and is documented as such
    rather than presented as a probability.
    """
    window = close.iloc[max(0, i - spec.vol_window): i + 1]
    rets = window.pct_change(fill_method=None)
    sigma = rets.std(ddof=1) * np.sqrt(spec.horizon)
    # A name with no measured dispersion has no barriers: upper and lower
    # collapse onto the entry price and EVERY bar touches both, which the
    # both-touched rule then books as a stop at zero. That is a label
    # manufactured out of a degenerate estimate, so it is refused instead.
    return sigma.where(sigma > 1e-6)


def triple_barrier(
    close: pd.DataFrame,
    i: int,
    spec: BarrierSpec,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Label every symbol at row ``i``. Never reads a row at or before ``i``
    for the outcome, and never reads past ``i + horizon``.

    Returns a frame indexed by symbol with:

    ``ret``       the return the trade realised, at whichever barrier hit first
    ``side``      +1 profit barrier, -1 stop barrier, 0 timed out
    ``held``      sessions from entry to the touch
    ``t1``        row index of the touch, for the uniqueness weights

    ``high``/``low`` make the touch test intraday, which is what a real stop
    is. Without them the test falls back to closes, which UNDERSTATES how often
    a stop is hit and therefore flatters the label -- the same optimism the
    outcome record was corrected for.
    """
    n = len(close)
    entry = close.iloc[i]
    sigma = _horizon_sigma(close, i, spec)
    end = min(i + spec.horizon, n - 1)
    if end <= i:
        return pd.DataFrame(columns=["ret", "side", "held", "t1"],
                            index=close.columns, dtype="float64")

    up = entry * (1.0 + spec.upper * sigma)
    dn = entry * (1.0 - spec.lower * sigma)

    hi = (high if high is not None else close).iloc[i + 1: end + 1]
    lo = (low if low is not None else close).iloc[i + 1: end + 1]
    cl = close.iloc[i + 1: end + 1]

    # Vectorised. The per-symbol Python loop this replaces ran ~3,500 symbols x
    # ~90 dates on a refit and did not finish in twelve minutes.
    e = entry.to_numpy("float64")
    s = sigma.reindex(close.columns).to_numpy("float64")
    up_lvl = e * (1.0 + spec.upper * s)
    dn_lvl = e * (1.0 - spec.lower * s)

    hit_up = hi.to_numpy("float64") >= up_lvl          # (bars, symbols)
    hit_dn = lo.to_numpy("float64") <= dn_lvl
    n_bars = hit_up.shape[0]

    # argmax on a boolean gives the first True, or 0 when there is none --
    # so `any` decides whether that 0 means "bar 0" or "never".
    first_up = np.where(hit_up.any(axis=0), hit_up.argmax(axis=0), n_bars + 1)
    first_dn = np.where(hit_dn.any(axis=0), hit_dn.argmax(axis=0), n_bars + 1)

    usable = np.isfinite(e) & (e > 0) & np.isfinite(s)
    timed_out = (first_up > n_bars) & (first_dn > n_bars)
    # Ties go to the stop: daily bars cannot order intraday events and assuming
    # the favourable one inflates every result built on the label.
    stopped = (~timed_out) & (first_dn <= first_up)
    took_profit = (~timed_out) & (~stopped)

    last_close = cl.to_numpy("float64")[-1] if n_bars else np.full(len(e), np.nan)

    ret = np.full(len(e), np.nan)
    side = np.full(len(e), np.nan)
    held = np.full(len(e), np.nan)
    t1 = np.full(len(e), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(timed_out, last_close / e - 1.0, ret)
        ret = np.where(stopped, dn_lvl / e - 1.0, ret)
        ret = np.where(took_profit, up_lvl / e - 1.0, ret)
    side = np.where(timed_out, 0.0, np.where(stopped, -1.0, 1.0))
    held = np.where(timed_out, float(n_bars),
                    np.where(stopped, first_dn + 1.0, first_up + 1.0))
    t1 = np.where(timed_out, float(end),
                  np.where(stopped, i + 1.0 + first_dn, i + 1.0 + first_up))

    ret = np.where(usable, ret, np.nan)
    side = np.where(usable, side, np.nan)
    held = np.where(usable, held, np.nan)
    t1 = np.where(usable, t1, np.nan)

    return pd.DataFrame({"ret": ret, "side": side, "held": held, "t1": t1},
                        index=close.columns)


def concurrency(t0: np.ndarray, t1: np.ndarray, n_rows: int) -> np.ndarray:
    """How many labels are live at each bar."""
    counts = np.zeros(int(n_rows) + 1, dtype="float64")
    for a, b in zip(t0, t1):
        if not (np.isfinite(a) and np.isfinite(b)) or b < a:
            continue
        counts[int(a): int(b) + 1] += 1.0
    return counts


def average_uniqueness(t0: np.ndarray, t1: np.ndarray,
                       n_rows: int) -> np.ndarray:
    """Mean of 1/concurrency over each label's span (Lopez de Prado, ch. 4).

    A label that has its window to itself weighs 1. One that shares every bar
    with two others weighs about a third. Fitting without this counts one
    market shock once per overlapping row, which is why an unweighted panel of
    33,000 rows reports a t-statistic it has not earned.

    Computed WITHIN a symbol: overlap means a label sharing its outcome window
    with the same name's other labels. Thirty names on one date are thirty
    correlated observations, not a thirtieth of one.
    """
    counts = concurrency(t0, t1, n_rows)
    safe = np.where(counts > 0, counts, 1.0)
    out = np.zeros(len(t0), dtype="float64")
    for k, (a, b) in enumerate(zip(t0, t1)):
        if not (np.isfinite(a) and np.isfinite(b)) or b < a:
            out[k] = np.nan
            continue
        out[k] = float(np.mean(1.0 / safe[int(a): int(b) + 1]))
    return out
