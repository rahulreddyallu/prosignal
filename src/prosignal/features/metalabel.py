"""Meta-labelling: a second model that decides whether to act on the first.

WHAT THE PRIMARY MODEL CANNOT ANSWER. The cross-sectional fit ranks the
universe. It is very good at saying which of two names is more attractive and
says nothing at all about whether the better of them is worth buying, because a
rank is relative by construction -- somebody is always top of the list, on the
best day of the decade and on the worst. The engine then buys the top of that
list every single rebalance.

Lopez de Prado (ch. 3) splits the two questions. The primary model picks the
trade; a SECONDARY binary model, fitted only on the trades the primary would
actually have taken, predicts whether a trade like this reaches its profit
barrier before its stop. The second model cannot pick a name the first did not
-- it has no long side of its own -- so its only power is to VETO. That is
exactly the shape the NO TRADE gate needs.

WHY IT IS FITTED ON THE SHORTLIST, NOT THE UNIVERSE. A classifier fitted on
every name in the panel learns what separates a good stock from a bad one, which
is the primary model's job and which it already does. Fitted on the top of the
primary's own ranking it learns something different and much narrower: among the
names this engine likes, which ones does it tend to be WRONG about. Those are
not the same question and the second one is the only one worth a second model.

THE GROUND TRUTH IS ALREADY THERE. Triple-barrier labelling records which
barrier was touched first, so `barrier_side` is the meta-label: +1 the trade
reached target, -1 it was stopped, 0 it timed out. No new data is needed, which
is the point -- meta-labelling is a re-use of the label the engine already fits
on, not another source of things to overfit to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MetaModel", "logistic_fit", "logistic_predict", "meta_label",
    "fit_meta", "reliability", "auc", "MIN_META_ROWS", "TIMEOUT_IS",
]

#: Below this the classifier is fitting noise and is refused. A shortlist of
#: eight names on seventy dates is 560 rows before any dropna.
MIN_META_ROWS = 300

#: How a timed-out trade is scored. A trade that touched neither barrier in 63
#: sessions is NOT a win: the engine tied up a slot, paid the spread and got
#: whatever drift was going. Counting it as a loss would overstate the damage
#: -- it usually ends slightly positive -- so it is excluded from the fit
#: entirely and the classifier answers the clean question, target or stop.
TIMEOUT_IS = "excluded"


@dataclass
class MetaModel:
    """A fitted binary classifier plus the provenance to reproduce it."""

    features: List[str]
    coef: np.ndarray
    intercept: float
    mu: np.ndarray
    sd: np.ndarray
    n_train: int
    base_rate: float
    #: Out-of-fold discrimination, measured while fitting. `None` when the
    #: training panel could not support an honest estimate.
    oof_auc: Optional[float] = None
    converged: bool = True
    notes: List[str] = field(default_factory=list)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy("float64")
        return logistic_predict(
            {"coef": self.coef, "intercept": self.intercept,
             "mu": self.mu, "sd": self.sd}, x)

    def summary(self) -> str:
        auc_s = f"{self.oof_auc:.3f}" if self.oof_auc is not None else "n/a"
        return (f"meta-label classifier on {len(self.features)} features, "
                f"{self.n_train} shortlist rows, base rate "
                f"{self.base_rate:.1%}, out-of-fold AUC {auc_s}")


def _standardise(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd, mu, sd


def logistic_fit(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0,
                 weights: Optional[np.ndarray] = None,
                 max_iter: int = 50, tol: float = 1e-8) -> Dict[str, object]:
    """Ridge-penalised logistic regression by IRLS.

    Written out rather than pulled from sklearn for the same reason `ridge_fit`
    is: this is forty lines against ~100 MB of dependency on a 512 MB instance,
    and the baseline must not be the thing that breaks the deployment.

    ``weights`` are per-observation and carry label uniqueness through, exactly
    as the primary fit does -- overlapping shortlist rows are no more
    independent here than they are there.
    """
    xs, mu, sd = _standardise(np.asarray(x, dtype="float64"))
    y = np.asarray(y, dtype="float64")
    n, p = xs.shape
    design = np.column_stack([np.ones(n), xs])
    w_obs = np.ones(n) if weights is None else np.asarray(weights, dtype="float64")
    w_obs = np.where(np.isfinite(w_obs) & (w_obs > 0), w_obs, 0.0)
    if w_obs.sum() <= 0:
        w_obs = np.ones(n)
    w_obs = w_obs * (n / w_obs.sum())

    beta = np.zeros(p + 1)
    # The intercept is not penalised: shrinking it drags every prediction toward
    # 0.5 regardless of the base rate, which is a different claim entirely.
    penalty = np.eye(p + 1) * float(l2)
    penalty[0, 0] = 0.0
    converged = False
    for _ in range(max_iter):
        eta = design @ beta
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        # IRLS weight. Floored because a saturated probability drives it to zero
        # and the normal equations become singular.
        s = np.clip(prob * (1.0 - prob), 1e-6, None) * w_obs
        z = eta + (y - prob) / np.clip(prob * (1.0 - prob), 1e-6, None)
        a = design.T @ (design * s[:, None]) + penalty
        b = design.T @ (s * z)
        try:
            new = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(new).all():
            break
        step = float(np.max(np.abs(new - beta)))
        beta = new
        if step < tol:
            converged = True
            break
    return {"coef": beta[1:], "intercept": float(beta[0]), "mu": mu, "sd": sd,
            "converged": converged}


def logistic_predict(fit, x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype="float64") - fit["mu"]) / fit["sd"]
    eta = xs @ fit["coef"] + fit["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))


def meta_label(barrier_side: pd.Series) -> pd.Series:
    """1 the trade reached target, 0 it was stopped, NaN it timed out.

    See TIMEOUT_IS. A timeout is neither outcome and is dropped rather than
    forced into one, which would teach the classifier that "nothing happened"
    looks like whichever class it was assigned to.
    """
    side = pd.to_numeric(barrier_side, errors="coerce")
    out = pd.Series(np.nan, index=side.index, dtype="float64")
    out[side > 0] = 1.0
    out[side < 0] = 0.0
    return out


def auc(y: np.ndarray, prob: np.ndarray) -> float:
    """Area under the ROC curve, by rank. 0.5 is a coin.

    Computed from the Mann-Whitney identity, so ties are handled by midranks
    rather than silently favouring one class.
    """
    y = np.asarray(y, dtype="float64")
    prob = np.asarray(prob, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(prob)
    y, prob = y[ok], prob[ok]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(prob).rank().to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def reliability(y: np.ndarray, prob: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Predicted probability against realised frequency, by bucket.

    A classifier can discriminate well and still be badly calibrated, and a NO
    TRADE gate reads the LEVEL of the probability, not its ordering -- so
    discrimination alone is not enough to justify a threshold.
    """
    y = np.asarray(y, dtype="float64")
    prob = np.asarray(prob, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(prob)
    frame = pd.DataFrame({"y": y[ok], "p": prob[ok]})
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "n", "predicted", "realised"])
    frame["bucket"] = pd.qcut(frame["p"], min(bins, frame["p"].nunique()),
                              labels=False, duplicates="drop")
    grouped = frame.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), realised=("y", "mean"))
    return grouped.reset_index()


def shortlist(panel: pd.DataFrame, score: pd.Series, top_k: int) -> pd.DataFrame:
    """The rows the primary model would actually have bought, per date.

    ``score`` must be aligned to ``panel``'s index and must come from a model
    that did NOT see these dates -- an in-sample ranking selects the names the
    primary got right, and a classifier fitted on that learns to approve
    everything.
    """
    work = panel.copy()
    work["_meta_score"] = np.asarray(score, dtype="float64")
    work["_meta_rank"] = work.groupby("date", observed=True)["_meta_score"] \
        .rank(ascending=False, method="first")
    return work[work["_meta_rank"] <= int(top_k)]


def fit_meta(
    rows: pd.DataFrame,
    features: Sequence[str],
    *,
    l2: float = 1.0,
    weights: Optional[np.ndarray] = None,
    min_rows: int = MIN_META_ROWS,
) -> Tuple[Optional[MetaModel], Optional[str]]:
    """Fit the veto model on shortlist rows. Returns (model, reason_unavailable)."""
    cols = [c for c in features if c in rows.columns]
    if not cols:
        return None, "no usable feature columns for the meta model"
    if "barrier_side" not in rows.columns:
        return None, ("the panel carries no barrier outcome, so there is no "
                      "meta-label to fit -- triple-barrier labelling is off")
    work = rows.copy()
    work["_meta_y"] = meta_label(work["barrier_side"])
    work = work.dropna(subset=cols + ["_meta_y"])
    if len(work) < min_rows:
        return None, (f"{len(work)} shortlist rows with a decided outcome; "
                      f"{min_rows} required")
    y = work["_meta_y"].to_numpy("float64")
    if y.min() == y.max():
        return None, ("every shortlisted trade had the same outcome; there is "
                      "nothing for a classifier to separate")
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype="float64")
    elif "uniqueness" in work.columns:
        w = work["uniqueness"].to_numpy("float64")
    x = work[cols].to_numpy("float64")
    fit = logistic_fit(x, y, l2=l2, weights=w)
    model = MetaModel(
        features=cols, coef=np.asarray(fit["coef"], dtype="float64"),
        intercept=float(fit["intercept"]), mu=fit["mu"], sd=fit["sd"],
        n_train=len(work), base_rate=float(y.mean()),
        converged=bool(fit["converged"]),
    )
    if not model.converged:
        model.notes.append("IRLS did not converge; coefficients are the last iterate")
    return model, None


#: Fraction of the training dates used to FIT the primary when generating
#: shortlists for the meta model. The rest are scored by a primary that never
#: saw them, and only those rows are eligible to train the veto.
INNER_FIT_FRACTION = 0.6

#: How far down the primary's ranking counts as "a trade this engine would
#: consider". The book holds eight names, but eight rows a date is ~370 decided
#: trades across the entire history -- not a sample a classifier can be fitted
#: on. The top 50 of ~600 eligible names is the region the book is drawn from
#: after the Stage 3/5 gates have had their say.
DEFAULT_SHORTLIST = 50


def fit_meta_out_of_sample(
    panel: pd.DataFrame,
    features: Sequence[str],
    fit_primary,
    *,
    top_k: int = DEFAULT_SHORTLIST,
    l2: float = 1.0,
    inner_fraction: float = INNER_FIT_FRACTION,
    min_rows: int = MIN_META_ROWS,
) -> Tuple[Optional[MetaModel], Optional[str]]:
    """Fit the veto on shortlists the primary produced WITHOUT seeing them.

    ``fit_primary(train, frame)`` returns primary scores for ``frame`` from a
    model fitted on ``train``.

    The training dates are cut in two. The primary is fitted on the earlier
    part and used to rank the later part; the shortlist from that later part is
    what the meta model learns on. Skipping this and shortlisting with an
    in-sample primary selects the names the primary already got right, and a
    classifier fitted on those learns to approve everything -- it would report a
    high accuracy and veto nothing.
    """
    if "date" not in panel.columns:
        return None, "the panel has no date column"
    dates = sorted(panel["date"].unique())
    if len(dates) < 10:
        return None, (f"{len(dates)} training dates; the inner split needs at "
                      f"least 10 to leave anything out of sample")
    cut = set(dates[: max(int(len(dates) * inner_fraction), 3)])
    inner_train = panel[panel["date"].isin(cut)]
    inner_test = panel[~panel["date"].isin(cut)]
    if inner_test.empty or inner_train.empty:
        return None, "the inner split left one side empty"
    scores = fit_primary(inner_train, inner_test)
    if scores is None:
        return None, "the primary model could not be fitted on the inner block"
    rows = shortlist(inner_test, scores, top_k)
    return fit_meta(rows, features, l2=l2, min_rows=min_rows)
