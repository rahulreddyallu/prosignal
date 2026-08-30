"""Live monitoring for the v2 scorer.

Two questions the engine could not previously answer about itself:

  1. Is each factor still doing what the holdout measured it doing? A composite
     can hold up in aggregate while half of it has quietly inverted, and the
     composite's own IC will not say which half. So the information coefficient
     is tracked PER FACTOR on a rolling window, against the factor's shipped
     sign.

  2. Is the book in a drawdown deeper than anything the evidence describes?

Neither ever disables anything. A monitor that silently switches a factor off
changes the model without a decision being taken, and the next person to read
the config sees a model that is not the one running. Both of these FLAG.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features.v2 import V2_FACTORS

__all__ = ["FactorHealth", "DrawdownFlag", "rolling_factor_ic",
           "review_factors", "review_drawdown", "MIN_PERIODS", "IC_ALERT_T"]

#: Fewer than this many scored periods and the rolling IC is not worth reading.
#: At the shipped weekly cadence and a 42-session label this is roughly a year
#: of signal dates and about six independent observations -- which is thin, and
#: is exactly why the alert below is a flag and not a switch.
MIN_PERIODS = 40

#: A factor is flagged when its rolling IC has the WRONG SIGN at this
#: confidence. Not when it is merely weak: a factor at zero is uninformative and
#: costs the composite a tenth of its weight, while a factor that has inverted is
#: actively subtracting, and only the second is worth waking someone for.
IC_ALERT_T = 2.0

#: Book drawdown past which the run carries a loud flag. Set at the deepest
#: drawdown the sealed holdout produced (-14.0%), rounded out to -15%: past this
#: the book is outside the range the shipped evidence describes, which is a
#: statement about evidence, not a prediction.
DRAWDOWN_FLAG = -0.15


@dataclass
class FactorHealth:
    name: str
    shipped_sign: int
    n_periods: int
    ic_mean: Optional[float] = None
    ic_t: Optional[float] = None
    inverted: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"factor": self.name, "shipped_sign": self.shipped_sign,
                "n_periods": self.n_periods, "ic_mean": self.ic_mean,
                "ic_t": self.ic_t, "inverted": self.inverted, "note": self.note}


@dataclass
class DrawdownFlag:
    flagged: bool
    drawdown: float
    threshold: float = DRAWDOWN_FLAG
    peak_date: Optional[dt.date] = None
    trough_date: Optional[dt.date] = None
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"flagged": self.flagged, "drawdown": self.drawdown,
                "threshold": self.threshold,
                "peak_date": self.peak_date.isoformat() if self.peak_date else None,
                "trough_date": self.trough_date.isoformat() if self.trough_date else None,
                "note": self.note}


def _rank_ic(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rolling_factor_ic(panel: pd.DataFrame, label_col: str,
                      factors: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-date rank IC for each factor's SHIPPED-SIGN-ORIENTED rank column.

    ``panel`` needs a ``date`` column, one ``<factor>_r`` column per factor and
    the realised forward return in ``label_col``. Rows whose label has not
    resolved yet are simply absent -- the monitor never waits on an outcome it
    cannot see, and never fills one in.
    """
    names = list(factors) if factors else [f.name for f in V2_FACTORS]
    signs = {f.name: f.sign for f in V2_FACTORS}
    rows = []
    for d, g in panel.groupby("date", sort=True):
        y = g[label_col].to_numpy("float64")
        rec: Dict[str, object] = {"date": d, "n": int(np.isfinite(y).sum())}
        for n in names:
            col = n + "_r"
            if col not in g.columns:
                rec[n] = np.nan
                continue
            ic = _rank_ic(g[col].to_numpy("float64") * signs.get(n, 1), y)
            rec[n] = np.nan if ic is None else ic
        rows.append(rec)
    return pd.DataFrame(rows)


def review_factors(ic_frame: pd.DataFrame, window: int = 52,
                   min_periods: int = MIN_PERIODS,
                   alert_t: float = IC_ALERT_T) -> List[FactorHealth]:
    """Flag any factor whose recent oriented IC is significantly NEGATIVE.

    The t-statistic is naive -- overlapping labels make consecutive ICs share
    most of their window, which inflates it by roughly sqrt(horizon / stride).
    That inflation is stated rather than corrected because this is a screen for
    a human to look at, and the honest correction at these sample sizes costs
    more power than the screen has to spare. Read a flag as "go and check",
    never as "this factor is dead".
    """
    out: List[FactorHealth] = []
    tail = ic_frame.tail(int(window))
    for f in V2_FACTORS:
        if f.name not in tail.columns:
            out.append(FactorHealth(f.name, f.sign, 0, note="not present in the panel"))
            continue
        v = tail[f.name].to_numpy("float64")
        v = v[np.isfinite(v)]
        if len(v) < min_periods:
            out.append(FactorHealth(f.name, f.sign, len(v),
                                    note=f"under the {min_periods}-period floor; "
                                         f"no verdict"))
            continue
        m = float(v.mean())
        t = float(m / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 0 else 0.0
        inverted = t <= -alert_t
        note = ("oriented IC is significantly negative -- this factor has been "
                "subtracting from the composite over this window; the naive t "
                "is inflated by label overlap, so check before acting"
                if inverted else "consistent with its shipped sign")
        out.append(FactorHealth(f.name, f.sign, len(v), m, t, inverted, note))
    return out


def review_drawdown(equity: Sequence[float],
                    dates: Optional[Sequence[dt.date]] = None,
                    threshold: float = DRAWDOWN_FLAG) -> DrawdownFlag:
    """Current drawdown from the running peak, and whether it is past the flag.

    FLAGS, NEVER DISABLES. A breaker that stops new entries in a drawdown is a
    market-timing rule, and every market-timing rule this engine measured cost
    8-13 points of annual excess without reducing the drawdown it was there to
    avoid. So this reports; the person decides.
    """
    e = np.asarray(list(equity), dtype="float64")
    if e.size == 0 or not np.isfinite(e).any():
        return DrawdownFlag(False, float("nan"),
                            note="no equity curve to measure")
    peaks = np.maximum.accumulate(e)
    dd = e / np.where(peaks > 0, peaks, np.nan) - 1.0
    cur = float(dd[-1])
    i_trough = int(np.nanargmin(dd))
    i_peak = int(np.nanargmax(e[: i_trough + 1])) if i_trough >= 0 else 0
    d_peak = d_trough = None
    if dates is not None and len(dates) == len(e):
        d_peak, d_trough = dates[i_peak], dates[i_trough]
    flagged = np.isfinite(cur) and cur <= threshold
    note = (f"book is {cur:.1%} below its peak, past the {threshold:.0%} flag. "
            f"The deepest drawdown in the sealed holdout was -14.0%; this is "
            f"outside the range the shipped evidence describes. Nothing has "
            f"been disabled -- review the position sizes and the rolling factor "
            f"IC below before the next entry."
            if flagged else
            f"book is {cur:.1%} below its peak, inside the {threshold:.0%} flag")
    return DrawdownFlag(bool(flagged), cur, threshold, d_peak, d_trough, note)
