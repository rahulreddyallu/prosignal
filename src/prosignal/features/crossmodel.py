"""Cross-sectional return-rank model used by Stage 4.

A ridge regression on fourteen cross-sectionally ranked features, refitted on
every run from history strictly before the decision date, predicting the rank of
the forward 21-session return.

It is here rather than in a research folder because it was measured against the
incumbent composite under purged walk-forward on 8.8 years and won:

    composite (momentum + sector RS)  IC +0.025  t 1.28   excess +0.14%/mo t 0.30
    ridge                             IC +0.052  t 3.64   excess +1.11%/mo t 3.44

The composite's excess return over an equal-weight benchmark is not
distinguishable from zero. Elastic Net additionally beats the composite
head-to-head on a scale-invariant loss at p = 0.045.

Training uses only dates whose 21-session label window closed before ``as_of``,
so no observation used in the fit can know anything about the decision date.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import json

from ..core.logging import get_logger
from ..core.memory import release_memory
from .crosssec import FEATURES, MIN_LOOKBACK, build_panel
from .linear import predict, ridge_fit

__all__ = ["CrossSectionalModel", "fit_predict", "load_cached", "save_cache",
           "score_with", "today_features"]

log = get_logger(__name__)

FEATURE_COLUMNS = [f + "_r" for f in FEATURES]

#: Ridge penalty. Fixed rather than searched: tuning it per run against the
#: same history that scores it is how a validated result becomes an overfit one.
ALPHA = 10.0

#: Label horizon in sessions, matching the holding period the engine plans for.
HORIZON = 21

#: Minimum training rows. Below this the fit is noise and the model abstains
#: rather than returning a confident-looking number from nothing.
MIN_TRAIN_ROWS = 600

#: Training window in sessions. Bounded rather than "all history" for memory:
#: the full 2,206-session panel peaked at 615 MB on a 512 MB instance. 1,500
#: sessions still yields ~45 training periods for 14 features, which is the
#: span the earliest validated walk-forward folds trained on.
MAX_TRAIN_SESSIONS = 3000

#: Refit cadence. The coefficients are a slow-moving object -- refitting daily
#: costs 17 seconds and 300 MB to move them marginally, and invites the fit to
#: chase noise. Cached coefficients are reused until this many sessions pass.
REFIT_EVERY_SESSIONS = 21


class CrossSectionalModel:
    """A fitted ridge model plus the provenance needed to reproduce it."""

    def __init__(self, coef: Dict[str, float], n_train: int, train_end: dt.date) -> None:
        self.coef = coef
        self.n_train = n_train
        self.train_end = train_end

    def summary(self) -> str:
        top = sorted(self.coef.items(), key=lambda kv: -abs(kv[1]))[:4]
        parts = ", ".join(f"{k.replace('_r','')} {v:+.3f}" for k, v in top)
        return (
            f"ridge on {len(self.coef)} cross-sectional features, "
            f"{self.n_train} training rows to {self.train_end}; strongest: {parts}"
        )


def load_cached(path, as_of: dt.date) -> Optional[CrossSectionalModel]:
    """Coefficients from a recent fit, or None when absent or stale."""
    try:
        if not path.is_file():
            return None
        blob = json.loads(path.read_text(encoding="utf-8"))
        fitted = dt.date.fromisoformat(blob["fitted_for"])
        if (as_of - fitted).days > REFIT_EVERY_SESSIONS * 2:
            return None
        if sorted(blob["coef"]) != sorted(FEATURE_COLUMNS):
            return None                      # feature set changed; refit
        m = CrossSectionalModel(
            coef=blob["coef"], n_train=int(blob["n_train"]),
            train_end=dt.date.fromisoformat(blob["train_end"]),
        )
        m.mu = np.array(blob["mu"], dtype="float64")
        m.sd = np.array(blob["sd"], dtype="float64")
        m.intercept = float(blob["intercept"])
        return m
    except (OSError, ValueError, KeyError):
        return None


def save_cache(path, model: CrossSectionalModel, as_of: dt.date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fitted_for": as_of.isoformat(),
        "train_end": model.train_end.isoformat(),
        "n_train": model.n_train,
        "coef": model.coef,
        "mu": list(map(float, model.mu)),
        "sd": list(map(float, model.sd)),
        "intercept": model.intercept,
    }), encoding="utf-8")


def score_with(model: CrossSectionalModel, features: pd.DataFrame) -> pd.Series:
    """Apply stored coefficients to today's features."""
    x = features[FEATURE_COLUMNS].to_numpy("float64")
    coef = np.array([model.coef[c] for c in FEATURE_COLUMNS], dtype="float64")
    raw = ((x - model.mu) / model.sd) @ coef + model.intercept
    s = pd.Series(raw, index=features["symbol"].to_numpy())
    return ((s.rank(pct=True) - 0.5) * 2.0).sort_values(ascending=False)


def fit_predict(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    as_of: dt.date,
) -> Tuple[Optional[pd.Series], Optional[CrossSectionalModel], Optional[str]]:
    """Rank every symbol by predicted forward return.

    Returns ``(scores, model, reason_unavailable)``. Scores are in [-1, 1] and
    comparable only within this run, since the fit is refitted each time.
    """
    ts = pd.Timestamp(as_of)
    hist = close[close.index <= ts]
    if len(hist) > MAX_TRAIN_SESSIONS + HORIZON:
        hist = hist.iloc[-(MAX_TRAIN_SESSIONS + HORIZON):]
    if len(hist) < MIN_LOOKBACK + HORIZON + 60:
        return None, None, (
            f"{len(hist)} sessions of history; the cross-sectional model needs "
            f"{MIN_LOOKBACK + HORIZON + 60}"
        )

    # Training stops one full label horizon before as_of. A row dated later
    # would have a label running past the decision date, which is the leak this
    # model exists to avoid.
    train_close = hist.iloc[: len(hist) - HORIZON]
    train_turnover = turnover.reindex(train_close.index)
    panel = build_panel(train_close, train_turnover, horizon=HORIZON, step=21)
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["label_rank"]) if not panel.empty else panel
    if panel.empty or len(panel) < MIN_TRAIN_ROWS:
        return None, None, (
            f"{0 if panel.empty else len(panel)} usable training rows; "
            f"{MIN_TRAIN_ROWS} required"
        )

    x = panel[FEATURE_COLUMNS].to_numpy("float64")
    y = panel["label_rank"].to_numpy("float64")
    fit = ridge_fit(x, y, alpha=ALPHA)

    # Features for the decision date itself, from the same builder, so training
    # and inference cannot drift apart in definition.
    live = build_panel(hist.tail(MIN_LOOKBACK + 5),
                       turnover.reindex(hist.index).tail(MIN_LOOKBACK + 5),
                       horizon=1, step=21)
    if live.empty:
        return None, None, "features could not be computed for the decision date"
    latest = live[live["date"] == live["date"].max()]
    latest = latest.dropna(subset=FEATURE_COLUMNS)
    if latest.empty:
        return None, None, "no symbol had a complete feature set on the decision date"

    n_train = len(panel)
    train_end = pd.Timestamp(panel["date"].max()).date()
    del panel, x, y
    raw = predict(fit, latest[FEATURE_COLUMNS].to_numpy("float64"))
    scores = pd.Series(raw, index=latest["symbol"].to_numpy())
    ranked = (scores.rank(pct=True) - 0.5) * 2.0

    model = CrossSectionalModel(
        coef=dict(zip(FEATURE_COLUMNS, fit["coef"].tolist())),
        n_train=n_train,
        train_end=train_end,
    )
    model.mu, model.sd, model.intercept = fit["mu"], fit["sd"], fit["intercept"]
    release_memory()
    log.info(
        "cross-sectional model fitted",
        extra={"n_train": n_train, "train_end": str(model.train_end),
               "scored": len(ranked)},
    )
    return ranked.sort_values(ascending=False), model, None


def today_features(close: pd.DataFrame, turnover: pd.DataFrame, as_of: dt.date):
    """Features for the decision date only.

    The cheap path: one date rather than a full training panel, so a cached
    model scores today without the large historical read.
    """
    ts = pd.Timestamp(as_of)
    hist = close[close.index <= ts].tail(MIN_LOOKBACK + 5)
    if len(hist) < MIN_LOOKBACK:
        return None
    live = build_panel(hist, turnover.reindex(hist.index), horizon=1, step=21)
    if live.empty:
        return None
    latest = live[live["date"] == live["date"].max()].dropna(subset=FEATURE_COLUMNS)
    return latest if not latest.empty else None
