"""Ridge and Elastic Net for cross-sectional return ranking, plus the purged
walk-forward used to validate them.

Ridge and Elastic Net are implemented directly rather than pulled from sklearn.
Ridge is a closed-form solve and Elastic Net is coordinate descent on
standardised columns; together that is about sixty lines against roughly 100 MB
of dependency on a 512 MB instance. The baseline has to be permanent, so it
should not be the thing that breaks the deployment.

Purging and embargo follow Lopez de Prado. With a 21-session forward label, an
observation dated within 21 sessions of a test block still encodes part of that
block's outcome, so it cannot sit in training. The embargo removes the residual
serial correlation immediately after the block.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ridge_fit", "elastic_net_fit", "predict",
    "purged_walk_forward", "rank_ic", "diebold_mariano",
]


def _standardise(x: np.ndarray, w: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if w is None:
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=0)
    else:
        tw = w.sum()
        mu = (x * w[:, None]).sum(axis=0) / tw
        sd = np.sqrt((((x - mu) ** 2) * w[:, None]).sum(axis=0) / tw)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd, mu, sd


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0,
              weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Closed-form ridge on standardised columns. The intercept is not penalised.

    ``weights`` are per-observation and exist for label overlap. A 63-session
    label sampled every 21 shares two thirds of its window with its neighbour,
    so consecutive rows are not independent draws and an unweighted fit counts
    one market shock once per overlapping row. Weighting by average uniqueness
    (Lopez de Prado, ch. 4) charges each row only for the part of its span it
    holds alone: on this panel's geometry that is 0.395, so 33,000 rows carry
    about 13,000 independent-equivalent observations.

    Scaled to mean 1 so ``alpha`` keeps its meaning -- otherwise down-weighting
    the sample would silently strengthen the penalty.
    """
    if weights is not None:
        w = np.asarray(weights, dtype="float64")
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() <= 0:
            w = None
        else:
            w = w * (len(w) / w.sum())
    else:
        w = None

    xs, mu, sd = _standardise(x, w)
    n, p = xs.shape
    if w is None:
        a = xs.T @ xs + alpha * np.eye(p)
        b = xs.T @ (y - y.mean())
        intercept = float(y.mean())
    else:
        yw = float((y * w).sum() / w.sum())
        xw = xs * w[:, None]
        a = xs.T @ xw + alpha * np.eye(p)
        b = xw.T @ (y - yw)
        intercept = yw
    coef = np.linalg.solve(a, b)
    return {"coef": coef, "mu": mu, "sd": sd, "intercept": intercept}


def elastic_net_fit(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.01, l1_ratio: float = 0.5,
    max_iter: int = 500, tol: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Coordinate descent with soft thresholding (Friedman, Hastie & Tibshirani 2010)."""
    xs, mu, sd = _standardise(x)
    n, p = xs.shape
    yc = y - y.mean()
    coef = np.zeros(p)
    norms = (xs ** 2).sum(axis=0) / n
    norms[norms < 1e-12] = 1.0
    l1, l2 = alpha * l1_ratio, alpha * (1.0 - l1_ratio)
    for _ in range(max_iter):
        prev = coef.copy()
        for j in range(p):
            resid = yc - xs @ coef + xs[:, j] * coef[j]
            rho = float(xs[:, j] @ resid) / n
            coef[j] = np.sign(rho) * max(abs(rho) - l1, 0.0) / (norms[j] + l2)
        if np.max(np.abs(coef - prev)) < tol:
            break
    return {"coef": coef, "mu": mu, "sd": sd, "intercept": float(y.mean())}


def predict(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (x - model["mu"]) / model["sd"]
    return xs @ model["coef"] + model["intercept"]


def rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman correlation, computed from ranks with numpy."""
    ok = np.isfinite(pred) & np.isfinite(actual)
    if ok.sum() < 8:
        return np.nan
    a = pd.Series(pred[ok]).rank().to_numpy()
    b = pd.Series(actual[ok]).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> Tuple[float, float]:
    """Diebold-Mariano on squared-error loss.

    Returns (statistic, two-sided p). Positive means model 1 has the larger
    loss, so model 2 forecasts better. p from the normal approximation, which
    is what the small sample here supports.
    """
    d = (e1 ** 2) - (e2 ** 2)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    mean = d.mean()
    var = d.var(ddof=1)
    if var <= 0:
        return np.nan, np.nan
    stat = mean / np.sqrt(var / n)
    p = 2.0 * (1.0 - _norm_cdf(abs(stat)))
    return float(stat), float(p)


def _norm_cdf(z: float) -> float:
    """Normal CDF via erf, avoiding a scipy dependency."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def purged_walk_forward(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    n_splits: int = 6,
    embargo: int = 5,
    target: str = "label_rank",
) -> List[Dict[str, object]]:
    """Expanding-window walk-forward with purge and embargo.

    Splits are chronological. For each test block, training keeps only dates
    whose label window ends before the test block starts, minus an embargo. The
    purge distance is the label horizon itself, expressed in dates rather than
    rows because the panel is sampled every ``step`` sessions.
    """
    dates = np.array(sorted(panel["date"].unique()))
    if len(dates) < n_splits * 3:
        return []
    blocks = np.array_split(dates, n_splits + 1)
    folds: List[Dict[str, object]] = []
    for k in range(1, len(blocks)):
        test_dates = set(blocks[k])
        test_start = blocks[k][0]
        # Purge: a training label finishing on or after the test block's first
        # date overlaps it. Sampling is every `step` sessions, so one sampled
        # date is one step; convert the horizon into sampled dates.
        purge_steps = max(1, int(np.ceil(horizon / 21)))
        train_pool = [d for d in dates if d < test_start]
        if len(train_pool) <= purge_steps + embargo:
            continue
        keep = train_pool[: len(train_pool) - (purge_steps + embargo)]
        if len(keep) < 8:
            continue
        tr = panel[panel["date"].isin(set(keep))]
        te = panel[panel["date"].isin(test_dates)]
        if len(tr) < 200 or len(te) < 40:
            continue
        folds.append({
            "fold": k,
            "train_dates": (keep[0], keep[-1]),
            "test_dates": (blocks[k][0], blocks[k][-1]),
            "n_train": len(tr), "n_test": len(te),
            "purged_steps": purge_steps, "embargo_steps": embargo,
            "train": tr, "test": te,
        })
    return folds
