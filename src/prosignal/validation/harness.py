"""Run the cross-sectional model through CPCV and report the distribution.

Walk-forward tests one path. Every number this repository has quoted came from
one sequence of train/test windows, and the honest limitation was always the
same: 17 non-overlapping periods is a thin sample, and a Sharpe drawn from it
carries error bars wider than most of the differences being argued about.

CPCV answers that without needing more data. Splitting history into ``N``
groups and testing every combination of ``k`` of them yields ``C(N, k)`` fits
which weave into ``C(N-1, k-1)`` complete out-of-sample paths. Each path is a
full backtest; the spread across paths is the thing walk-forward cannot show.
Arian, Norouzi & Seco (2024) find CPCV better than walk-forward on both PBO and
DSR for false-discovery control, which is why validation/cpcv.py was written.
It was never called by anything until this module.

Purging and embargo are not optional here. With a 63-session label, a training
row dated 40 sessions before a test block still encodes part of that block's
outcome, and leaving it in flatters every number computed downstream.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..features.linear import predict, ridge_fit
from .cpcv import CombinatorialPurgedCV
from .metrics import compute_pbo, deflated_sharpe_ratio, sharpe_ratio

__all__ = ["CpcvResult", "run_cpcv"]

log = get_logger(__name__)


@dataclass
class CpcvResult:
    """The distribution of out-of-sample estimates, and what it implies."""

    n_splits: int
    n_paths: int
    #: Rank IC per test date, pooled across every split.
    ic: List[float] = field(default_factory=list)
    #: Top-decile excess return per test date, pooled.
    excess: List[float] = field(default_factory=list)
    #: One Sharpe per woven path -- the distribution walk-forward cannot show.
    path_sharpes: List[float] = field(default_factory=list)
    path_ics: List[float] = field(default_factory=list)
    purged_total: int = 0
    embargoed_total: int = 0
    notes: List[str] = field(default_factory=list)

    # -- summary ------------------------------------------------------------
    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic)) if self.ic else float("nan")

    @property
    def t_ic(self) -> float:
        if len(self.ic) < 3:
            return float("nan")
        a = np.asarray(self.ic, dtype="float64")
        return float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size)))

    def path_spread(self) -> Dict[str, float]:
        """Where the paths actually landed. The point of running CPCV at all."""
        if not self.path_sharpes:
            return {}
        a = np.asarray(self.path_sharpes, dtype="float64")
        return {
            "min": float(a.min()),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "share_negative": float((a < 0).mean()),
        }

    def deflated(self, n_trials: int):
        """DSR on the pooled per-period excess, charging the trial count."""
        return deflated_sharpe_ratio(self.excess, n_trials=n_trials)


def _rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(actual)
    if ok.sum() < 10:
        return float("nan")
    p, a = pred[ok], actual[ok]
    if p.std() == 0 or a.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, a)[0, 1])


def run_cpcv(
    panel: pd.DataFrame,
    features: Sequence[str],
    *,
    horizon_sessions: int,
    step_sessions: int,
    alpha: float,
    n_groups: int,
    n_test_groups: int,
    purge_sessions: int,
    embargo_sessions: int,
    min_train_rows: int = 2000,
    top_decile: float = 0.90,
    progress: Optional[Callable[[int, int], None]] = None,
) -> CpcvResult:
    """Fit and score the ridge model across every CPCV split.

    ``panel`` carries one row per (date, symbol) with the ranked feature
    columns, ``label_rank`` and ``label``. Groups are formed over DATES, not
    rows: a date is one observation as far as leakage is concerned, and
    splitting rows would put the same day on both sides of the partition.
    """
    cols = [c for c in features if c in panel.columns]
    work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
    dates = sorted(work["date"].unique())
    if len(dates) < n_groups * 2:
        raise ValueError(
            f"{len(dates)} panel dates cannot support {n_groups} CPCV groups; "
            f"fit fewer groups or build a longer panel"
        )

    # Purge and embargo are quoted in SESSIONS; the panel is sampled every
    # step_sessions, so they convert to whole observations by division. Rounding
    # up is deliberate -- a partial observation of overlap is still overlap.
    purge_obs = int(np.ceil(purge_sessions / step_sessions))
    embargo_obs = int(np.ceil(embargo_sessions / step_sessions))

    cv = CombinatorialPurgedCV(
        n_groups=n_groups, n_test_groups=n_test_groups,
        label_horizon=purge_obs, embargo=embargo_obs,
    )
    by_date = {d: g for d, g in work.groupby("date")}
    result = CpcvResult(n_splits=cv.n_splits, n_paths=cv.paths_per_observation())
    # path_id -> per-date results, so each path can be scored as one backtest
    paths: Dict[int, List[Dict[str, float]]] = {}
    seen: Dict[int, int] = {}

    for n, split in enumerate(cv.split(len(dates)), start=1):
        if progress:
            progress(n, cv.n_splits)
        train_dates = [dates[i] for i in split.train_idx]
        test_dates = [dates[i] for i in split.test_idx]
        train = work[work["date"].isin(train_dates)]
        if len(train) < min_train_rows:
            result.notes.append(
                f"split {split.split_id}: {len(train)} training rows, below the "
                f"{min_train_rows} floor; skipped"
            )
            continue
        result.purged_total += split.purged_count
        result.embargoed_total += split.embargoed_count

        fit = ridge_fit(train[cols].to_numpy("float64"),
                        train["label_rank"].to_numpy("float64"), alpha=alpha)
        for d in test_dates:
            te = by_date[d]
            pred = predict(fit, te[cols].to_numpy("float64"))
            ic = _rank_ic(pred, te["label_rank"].to_numpy("float64"))
            if np.isfinite(ic):
                result.ic.append(ic)
            lab = te["label"].to_numpy("float64")
            ok = np.isfinite(pred) & np.isfinite(lab)
            if ok.sum() >= 40:
                r = pd.Series(pred[ok]).rank(pct=True).to_numpy()
                ex = float(lab[ok][r >= top_decile].mean() - lab[ok].mean())
                result.excess.append(ex)
                # Weave: the k-th time a date is tested it belongs to path k.
                pid = seen.get(hash(d), 0)
                seen[hash(d)] = pid + 1
                paths.setdefault(pid, []).append({"date": d, "excess": ex, "ic": ic})

    for pid, rows in sorted(paths.items()):
        vals = np.asarray([r["excess"] for r in rows], dtype="float64")
        ics = np.asarray([r["ic"] for r in rows], dtype="float64")
        if vals.size < 4:
            continue
        result.path_sharpes.append(
            float(vals.mean() / vals.std(ddof=1)) if vals.std(ddof=1) > 0 else 0.0
        )
        result.path_ics.append(float(np.nanmean(ics)))

    log.info("cpcv complete",
             extra={"splits": result.n_splits, "paths": len(result.path_sharpes),
                    "mean_ic": round(result.mean_ic, 5)})
    return result


def configuration_matrix(
    panel: pd.DataFrame,
    configurations: Dict[str, Sequence[str]],
    *,
    step_sessions: int,
    alpha: float,
    purge_sessions: int,
    min_train_dates: int = 30,
    min_train_rows: int = 2000,
    top_decile: float = 0.90,
) -> pd.DataFrame:
    """Per-period performance of every configuration, on one common index.

    Feeds :func:`prosignal.validation.metrics.compute_pbo`, which asks a
    question no single backtest can: across the configurations actually tried,
    how often does the in-sample winner land below the out-of-sample median?
    That is the number that says whether a selection was skill or shopping.

    Every configuration is scored on identical dates with an identical purged
    expanding window, so the columns are comparable by construction.
    """
    purge_obs = int(np.ceil(purge_sessions / step_sessions))
    out: Dict[str, Dict[pd.Timestamp, float]] = {}

    for name, features in configurations.items():
        cols = [c for c in features if c in panel.columns]
        work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
        dates = sorted(work["date"].unique())
        by_date = {d: g for d, g in work.groupby("date")}
        series: Dict[pd.Timestamp, float] = {}
        for i in range(min_train_dates + purge_obs, len(dates)):
            train = work[work["date"].isin(dates[: i - purge_obs])]
            if len(train) < min_train_rows:
                continue
            te = by_date[dates[i]]
            fit = ridge_fit(train[cols].to_numpy("float64"),
                            train["label_rank"].to_numpy("float64"), alpha=alpha)
            pred = predict(fit, te[cols].to_numpy("float64"))
            lab = te["label"].to_numpy("float64")
            ok = np.isfinite(pred) & np.isfinite(lab)
            if ok.sum() < 40:
                continue
            r = pd.Series(pred[ok]).rank(pct=True).to_numpy()
            series[dates[i]] = float(lab[ok][r >= top_decile].mean() - lab[ok].mean())
        out[name] = series

    frame = pd.DataFrame(out).dropna()
    if frame.empty:
        raise ValueError("no date was scored by every configuration")
    return frame
