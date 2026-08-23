"""Is the top decile alpha, or is it factor exposure wearing a different name?

Every factor this engine fits is a documented premium -- Jegadeesh-Titman
momentum, Blitz residual momentum, George-Hwang 52-week proximity, Amihud
illiquidity, book-to-price, earnings yield, betting-against-beta. That gives
the strategy the economic rationale a sceptical reviewer asks for, and it
carries a corollary that is rarely stated alongside it: a portfolio built from
published premia should be expected to LOAD on those premia, and its returns
are then available more cheaply elsewhere.

The test is a regression of the strategy's excess return on long-short factor
portfolios built from the same universe on the same dates. If the intercept
survives, there is something here the factors do not explain. If it does not,
the engine is a factor portfolio and should be priced and risk-managed as one.

The factors are constructed here rather than taken from a vendor because no
Indian equivalent of the Fama-French research files covers this universe at
this frequency. That is a limitation and it cuts both ways: these are the
engine's own factor definitions, so the regression is if anything BIASED
TOWARD explaining the strategy away. An intercept that survives this test has
survived a hostile one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["FactorLoad", "Attribution", "build_factor_returns", "attribute"]


@dataclass(frozen=True)
class FactorLoad:
    name: str
    beta: float
    t: float
    #: Share of the strategy's mean return this factor accounts for.
    contribution: float


@dataclass(frozen=True)
class Attribution:
    alpha_per_period: float
    alpha_t: float
    alpha_t_adjusted: float
    r_squared: float
    n: int
    loads: List[FactorLoad] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def alpha_survives(self) -> bool:
        return self.alpha_t_adjusted >= 2.0

    @property
    def explained_share(self) -> float:
        """How much of the raw mean the factor loadings account for."""
        total = self.alpha_per_period + sum(l.contribution for l in self.loads)
        if total == 0:
            return float("nan")
        return 1.0 - (self.alpha_per_period / total)


def _long_short(
    scores: pd.Series,
    forward: pd.Series,
    *,
    decile: float = 0.30,
) -> Optional[float]:
    """Top minus bottom, equal weighted.

    A 30% tail rather than a decile: with a few hundred names per date the
    decile is thin enough that one or two prints move the factor, and a noisy
    regressor biases its own coefficient toward zero, which would understate
    how much the factor explains.
    """
    joined = pd.concat([scores.rename("s"), forward.rename("f")], axis=1).dropna()
    if len(joined) < 30:
        return None
    lo, hi = joined["s"].quantile(decile), joined["s"].quantile(1.0 - decile)
    top = joined.loc[joined["s"] >= hi, "f"]
    bot = joined.loc[joined["s"] <= lo, "f"]
    if top.empty or bot.empty:
        return None
    return float(top.mean() - bot.mean())


#: Factor name -> the panel column that ranks it, and whether high is the long
#: leg. These mirror the engine's own definitions deliberately.
FACTOR_SPEC: Dict[str, tuple] = {
    "MOM":     ("mom_6_1_r", True),
    "VALUE":   ("book_to_price_r", True),
    "QUALITY": ("earnings_yield_r", True),
    "LOWVOL":  ("downside_vol_r", False),
    "LIQ":     ("amihud_r", True),
    "SIZE":    ("turnover_ratio_r", False),
}


def build_factor_returns(
    panel: pd.DataFrame,
    *,
    forward_col: str = "fwd",
    date_col: str = "date",
    market_col: Optional[str] = None,
) -> pd.DataFrame:
    """One row per date: the market return and each long-short factor.

    MKT is the equal-weighted mean forward return across the eligible
    universe, which is the return of holding everything the engine could have
    chosen from. That is the right benchmark for a strategy that picks within
    that universe -- a cap-weighted index would introduce a size tilt the
    strategy never had the option to take.
    """
    if forward_col not in panel.columns:
        raise ValueError(f"panel has no {forward_col!r} column")

    rows: List[Dict[str, float]] = []
    for date, block in panel.groupby(date_col, observed=True):
        fwd = block[forward_col]
        if fwd.notna().sum() < 30:
            continue
        row: Dict[str, float] = {date_col: date}
        row["MKT"] = (float(block[market_col].mean()) if market_col
                      else float(fwd.mean()))
        for name, (col, high_is_long) in FACTOR_SPEC.items():
            if col not in block.columns:
                continue
            scores = block[col] if high_is_long else -block[col]
            value = _long_short(scores, fwd)
            if value is not None:
                row[name] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(date_col).reset_index(drop=True) if not out.empty else out


def attribute(
    strategy_excess: Sequence[float],
    factors: pd.DataFrame,
    *,
    horizon_sessions: int,
    step_sessions: int,
    factor_names: Optional[Sequence[str]] = None,
) -> Attribution:
    """Regress the strategy's excess on the factor returns.

    `strategy_excess` must already be excess of the same benchmark used for
    MKT, and aligned to the factor frame's dates. The intercept is the alpha
    and its t is reported both naively and corrected for the overlap in the
    sampling scheme, because the same inflation applies here as anywhere else.
    """
    from .significance import analytic_vif

    y = np.asarray(strategy_excess, dtype="float64")
    cols = [c for c in (factor_names or FACTOR_SPEC.keys()) if c in factors.columns]
    if not cols:
        raise ValueError("no usable factor columns")
    X = factors[cols].to_numpy("float64")

    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n = y.size
    notes: List[str] = []
    if n <= len(cols) + 2:
        raise ValueError(
            f"{n} usable observations against {len(cols)} factors -- "
            f"the regression has no degrees of freedom left"
        )
    if n < 3 * len(cols):
        notes.append(
            f"{n} observations for {len(cols)} factors. Coefficients are "
            f"poorly determined and the intercept absorbs whatever they miss."
        )

    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = n - design.shape[1]
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)

    vif = analytic_vif(horizon_sessions, step_sessions, n)
    alpha, alpha_se = float(beta[0]), float(se[0])
    alpha_t = alpha / alpha_se if alpha_se else float("nan")

    means = X.mean(axis=0)
    loads = [
        FactorLoad(
            name=cols[i],
            beta=float(beta[i + 1]),
            t=float(beta[i + 1] / se[i + 1]) if se[i + 1] else float("nan"),
            contribution=float(beta[i + 1] * means[i]),
        )
        for i in range(len(cols))
    ]
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return Attribution(
        alpha_per_period=alpha,
        alpha_t=alpha_t,
        alpha_t_adjusted=alpha_t / float(np.sqrt(vif)),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        n=n,
        loads=sorted(loads, key=lambda l: -abs(l.contribution)),
        notes=notes,
    )
