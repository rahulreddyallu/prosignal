"""Is a theme still working, or did it used to work?

WHY A MONITOR AND NOT JUST A REFIT. The gated estimator already sets a theme to
zero when it cannot clear |t| >= 2 on its own training window, so a theme that
stops working stops being traded. That is a control, not a monitor: it acts, and
then says nothing about WHY, and a theme that flickers in and out of the book
across refits looks identical to one that is quietly dying.

The distinction matters because the two call for opposite responses. A theme
with a small coefficient this quarter is noise and should be left alone. A theme
whose trailing coefficient has walked monotonically to zero over three years is
not a small number, it is a dead factor, and continuing to refit it every 21
sessions in the hope it comes back is how a strategy is kept alive past the
point where its edge was arbitraged away.

THE HAIRCUT. McLean & Pontiff (2016) measured 97 published anomalies and found
returns fall roughly 58% out of sample after publication -- about a third from
statistical bias in the original result, the rest from real arbitrage once the
paper was read. Every theme in this engine comes from a published paper. So the
honest expectation for any of them is NOT the in-sample coefficient; it is a
haircut version of it, and a theme that is merely meeting its haircut
expectation is behaving exactly as the literature predicts rather than
underperforming.

PRE-COMMITMENT. The kill criterion is declared in the config, before the numbers
are looked at, and this module only evaluates it. A rule chosen after seeing
which themes it would remove is not a rule -- it is the selection it exists to
prevent, wearing a lab coat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..features.famamacbeth import fama_macbeth, newey_west_se

__all__ = ["ThemeHealth", "DecayVerdict", "HAIRCUT_MCLEAN_PONTIFF",
           "assess_decay"]

#: McLean & Pontiff (2016), table 3: post-publication decay across 97 anomalies.
HAIRCUT_MCLEAN_PONTIFF = 0.58


@dataclass
class ThemeHealth:
    theme: str
    full_lambda: float
    full_t: float
    recent_lambda: float
    recent_t: float
    recent_dates: int
    #: What the literature would expect after publication decay.
    expected_lambda: float
    #: Trailing means, for the shape rather than the level.
    trail: pd.Series = field(default_factory=pd.Series)
    breaches: int = 0
    killed: bool = False
    reason: str = ""

    #: Below this the full-sample estimate is not distinguishable from zero, so
    #: there is no expectation to fall short of. `reversal` read "837% of
    #: expected" and `lottery` "4341%" -- both dividing by a coefficient that
    #: was itself noise, and both printing as though the theme were thriving.
    MIN_T_FOR_EXPECTATION = 1.0

    @property
    def share_of_expected(self) -> float:
        """Recent coefficient against the haircut expectation, or NaN.

        Only meaningful when the full-sample estimate is itself measurable AND
        the two share a sign. A negative expectation divided into a more
        negative recent value gives a large positive ratio that reads as health.
        """
        if not np.isfinite(self.expected_lambda) or abs(self.expected_lambda) < 1e-12:
            return float("nan")
        if not np.isfinite(self.full_t) or abs(self.full_t) < self.MIN_T_FOR_EXPECTATION:
            return float("nan")
        if not np.isfinite(self.recent_lambda):
            return float("nan")
        return float(self.recent_lambda / self.expected_lambda)


@dataclass
class DecayVerdict:
    themes: List[ThemeHealth] = field(default_factory=list)
    window: int = 0
    kill_t: float = 0.0
    required_breaches: int = 0
    haircut: float = HAIRCUT_MCLEAN_PONTIFF
    notes: List[str] = field(default_factory=list)

    @property
    def killed(self) -> List[str]:
        return [t.theme for t in self.themes if t.killed]


def _rolling_t(slopes: pd.Series, window: int, lags: int = 2) -> pd.Series:
    """Trailing Newey-West t of the mean slope, ending at each date."""
    values = slopes.to_numpy("float64")
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        if i + 1 < window:
            continue
        block = values[i + 1 - window: i + 1]
        block = block[np.isfinite(block)]
        if len(block) < max(4, window // 2):
            continue
        se = newey_west_se(block, lags)
        if se and np.isfinite(se) and se > 0:
            out[i] = float(block.mean() / se)
    return pd.Series(out, index=slopes.index)


def assess_decay(
    panel: pd.DataFrame,
    features: Sequence[str],
    *,
    window: int,
    kill_t: float,
    required_breaches: int,
    haircut: float = HAIRCUT_MCLEAN_PONTIFF,
    horizon: int = 63,
    step: int = 21,
    target: str = "label_rank",
) -> Optional[DecayVerdict]:
    """Evaluate the pre-committed kill criterion against every theme.

    A theme is killed when its trailing ``window``-date Newey-West t has sat at
    or below ``kill_t`` for ``required_breaches`` consecutive checks. Consecutive
    is the point: one bad window is a quarter, and killing on it would make the
    monitor a slower, noisier version of the refit gate that already exists.
    """
    full = fama_macbeth(panel, features, target=target, horizon=horizon, step=step)
    if full is None:
        return None
    verdict = DecayVerdict(window=window, kill_t=kill_t,
                           required_breaches=required_breaches, haircut=haircut)
    if len(full.slopes) < window + required_breaches:
        verdict.notes.append(
            f"{len(full.slopes)} cross-sections cannot support a {window}-date "
            f"trailing window plus {required_breaches} consecutive checks; the "
            f"kill criterion is not evaluable and NOTHING is killed on it")
        required = 0
    else:
        required = required_breaches

    for col in full.features:
        slopes = full.slopes[col]
        trail = slopes.rolling(window, min_periods=max(4, window // 2)).mean()
        t_series = _rolling_t(slopes, window)
        recent = t_series.dropna()
        # Consecutive breaches ending at the most recent evaluable check.
        breaches = 0
        for value in reversed(recent.to_numpy("float64")):
            if value <= kill_t:
                breaches += 1
            else:
                break
        block = slopes.to_numpy("float64")[-window:]
        block = block[np.isfinite(block)]
        recent_lam = float(block.mean()) if block.size else float("nan")
        se = newey_west_se(block, 2) if block.size else float("nan")
        recent_t = (float(recent_lam / se)
                    if se and np.isfinite(se) and se > 0 else float("nan"))
        health = ThemeHealth(
            theme=col, full_lambda=full.lam[col], full_t=full.t_stat[col],
            recent_lambda=recent_lam, recent_t=recent_t,
            recent_dates=int(block.size),
            expected_lambda=full.lam[col] * (1.0 - haircut),
            trail=trail, breaches=breaches,
        )
        if required and breaches >= required:
            health.killed = True
            health.reason = (
                f"trailing {window}-date t has been at or below {kill_t:+.2f} "
                f"for {breaches} consecutive checks")
        verdict.themes.append(health)
    return verdict
