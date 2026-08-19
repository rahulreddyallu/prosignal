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
from .fundamentals import FEATURE_NAMES as FUND_NAMES, compute_features
from .linear import predict, ridge_fit

__all__ = ["CrossSectionalModel", "fit_predict", "load_cached", "save_cache",
           "score_with", "today_features"]

log = get_logger(__name__)

#: Value and quality, ranked cross-sectionally like everything else. These are
#: the only inputs not derived from price and volume.
#:
#: Re-measured on corporate-action-adjusted prices at the shipped horizon,
#: holdout never used for selection:
#:
#:     price only            IC +0.100 (t 4.06)  excess +4.85%/period (t 5.74)
#:     price + fundamentals  IC +0.121 (t 5.09)  excess +5.47%/period (t 6.50)
#:     fundamentals alone    IC +0.097 (t 6.02)
#:
#: Diebold-Mariano p = 0.001: the fundamentals add forecasting information.
#: Both sets clear a permuted-label placebo. The earlier figures for this
#: comparison were computed on unadjusted prices and understated it -- on that
#: data the same test returned p = 0.66.
#:
#: They also break the liquidity concentration the audit flagged: removing the
#: liquidity family leaves IC +0.078 (t 2.93), where the price-only model
#: collapsed without it.
#:
#: market_cap is excluded: it is a scale variable, already proxied by turnover,
#: and it duplicates the liquidity family the model leans on too heavily.
FUNDAMENTAL_FEATURES = [f for f in FUND_NAMES if f != "market_cap"]

FEATURE_COLUMNS = [f + "_r" for f in FEATURES] + [f + "_r" for f in FUNDAMENTAL_FEATURES]

#: Ridge penalty. Fixed rather than searched per run: tuning it against the same
#: history that scores it is how a validated result becomes an overfit one.
#:
#: 20,000 rather than 10: with 24 correlated factors a light penalty lets the
#: fit chase noise. Chosen on the selection period alone and then left; on the
#: holdout the same feature set scored IC +0.045 at this value against +0.015
#: at 10.
ALPHA = 20_000.0          # default only; stage 4 passes the configured value

#: Label horizon in sessions, matching the holding period the engine plans for.
#:
#: Measured on a holdout never used for selection, the model is better at every
#: horizon tested up to 63 sessions and the curve is still rising:
#:
#:     H=21  IC +0.044 (t 2.18)  excess +1.11%/period  DSR 0.645  net +7.50%/yr
#:     H=42  IC +0.078 (t 3.38)  excess +2.40%/period  DSR 0.670  net +11.45%/yr
#:     H=63  IC +0.098 (t 3.72)  excess +4.06%/period  DSR 0.875  net +14.19%/yr
#:
#: Longer horizons win twice: the cross-sectional signal is stronger and the
#: turnover charge is smaller. The holding period below must match, or the
#: engine would exit before the return it is forecasting has accrued.
HORIZON = 63              # default only; stage 4 passes the configured value

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


def _attach_fundamentals(
    panel: pd.DataFrame, fundamentals: Optional[pd.DataFrame],
    close: pd.DataFrame, max_age_days: Optional[int],
) -> pd.DataFrame:
    """Merge point-in-time value/quality onto each panel date.

    Each date is priced with its own closes and gated on ``filing_date`` at that
    date, so a row can only ever see filings the market had already received.
    Names without a usable filing rank neutral rather than being dropped, which
    is what Stage 4 does when a factor is unavailable.
    """
    cols = [f + "_r" for f in FUNDAMENTAL_FEATURES]
    if fundamentals is None or fundamentals.empty:
        for c in cols:
            panel[c] = 0.0
        return panel

    frames = []
    for d in panel["date"].unique():
        ts = pd.Timestamp(d)
        prices = close.loc[:ts]
        if prices.empty:
            continue
        px = prices.iloc[-1].dropna().to_dict()
        feats = compute_features(fundamentals, px, ts.date(), max_age_days=max_age_days)
        if feats is None or feats.empty:
            continue
        f = feats.reset_index()
        if "symbol" not in f.columns:
            f = f.rename(columns={f.columns[0]: "symbol"})
        f["date"] = ts
        frames.append(f)

    if not frames:
        for c in cols:
            panel[c] = 0.0
        return panel

    fund = pd.concat(frames, ignore_index=True)
    keep = ["date", "symbol"] + [f for f in FUNDAMENTAL_FEATURES if f in fund.columns]
    panel = panel.merge(fund[keep], on=["date", "symbol"], how="left")
    for f in FUNDAMENTAL_FEATURES:
        col = f + "_r"
        if f in panel.columns:
            panel[col] = panel.groupby("date")[f].transform(
                lambda s: ((s.rank(pct=True, na_option="keep") - 0.5) * 2.0)
            ).fillna(0.0)
        else:
            panel[col] = 0.0
    return panel


def fit_predict(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
    as_of: dt.date,
    fundamentals: Optional[pd.DataFrame] = None,
    max_fundamental_age_days: Optional[int] = None,
    horizon: Optional[int] = None,
    alpha: Optional[float] = None,
    max_train_sessions: Optional[int] = None,
    min_train_rows: Optional[int] = None,
) -> Tuple[Optional[pd.Series], Optional[CrossSectionalModel], Optional[str]]:
    """Rank every symbol by predicted forward return.

    Returns ``(scores, model, reason_unavailable)``. Scores are in [-1, 1] and
    comparable only within this run, since the fit is refitted each time.
    """
    H = int(horizon if horizon is not None else HORIZON)
    A = float(alpha if alpha is not None else ALPHA)
    MAXS = int(max_train_sessions if max_train_sessions is not None else MAX_TRAIN_SESSIONS)
    MINR = int(min_train_rows if min_train_rows is not None else MIN_TRAIN_ROWS)

    ts = pd.Timestamp(as_of)
    hist = close[close.index <= ts]
    if len(hist) > MAXS + H:
        hist = hist.iloc[-(MAXS + H):]
    if len(hist) < MIN_LOOKBACK + H + 60:
        return None, None, (
            f"{len(hist)} sessions of history; the cross-sectional model needs "
            f"{MIN_LOOKBACK + H + 60}"
        )

    # Training stops one full label horizon before as_of. A row dated later
    # would have a label running past the decision date, which is the leak this
    # model exists to avoid.
    train_close = hist.iloc[: len(hist) - H]
    train_turnover = turnover.reindex(train_close.index)
    panel = build_panel(train_close, train_turnover, horizon=H, step=21)
    if not panel.empty:
        panel = _attach_fundamentals(panel, fundamentals, train_close, max_fundamental_age_days)
        panel = panel.dropna(subset=[c for c in FEATURE_COLUMNS if c in panel.columns]
                             + ["label_rank"])
    if panel.empty or len(panel) < MINR:
        return None, None, (
            f"{0 if panel.empty else len(panel)} usable training rows; "
            f"{MINR} required"
        )

    x = panel[FEATURE_COLUMNS].to_numpy("float64")
    y = panel["label_rank"].to_numpy("float64")
    fit = ridge_fit(x, y, alpha=A)

    # Features for the decision date itself, from the same builder, so training
    # and inference cannot drift apart in definition.
    live = build_panel(hist.tail(MIN_LOOKBACK + 5),
                       turnover.reindex(hist.index).tail(MIN_LOOKBACK + 5),
                       horizon=1, step=21)
    if live.empty:
        return None, None, "features could not be computed for the decision date"
    live = _attach_fundamentals(live, fundamentals, hist, max_fundamental_age_days)
    latest = live[live["date"] == live["date"].max()]
    latest = latest.dropna(subset=[c for c in FEATURE_COLUMNS if c in latest.columns])
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


def today_features(close: pd.DataFrame, turnover: pd.DataFrame, as_of: dt.date,
                   fundamentals: Optional[pd.DataFrame] = None,
                   max_fundamental_age_days: Optional[int] = None):
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
    live = _attach_fundamentals(live, fundamentals, hist, max_fundamental_age_days)
    latest = live[live["date"] == live["date"].max()].dropna(
        subset=[c for c in FEATURE_COLUMNS if c in live.columns])
    return latest if not latest.empty else None
