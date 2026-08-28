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

#: Every fit in this module goes through `crossmodel.fit_coefficients`, so what
#: CPCV validates is what stage 4 trades. Six separate `ridge_fit` calls used to
#: live here, and switching the production estimator would have left every
#: number in this file describing a model no longer in use.
_ESTIMATOR_DEFAULT = "fama_macbeth"


def _fit(train, cols, alpha, estimator, horizon_sessions, step_sessions,
         significance_floor=None):
    from ..features.crossmodel import fit_coefficients
    w = (train["uniqueness"].to_numpy("float64")
         if "uniqueness" in train.columns else None)
    fit, _fm, _why = fit_coefficients(
        train, cols, estimator=estimator, alpha=alpha,
        horizon=horizon_sessions, step=step_sessions,
        significance_floor=significance_floor, weights=w)
    return fit
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
    #: Excess keyed by TEST DATE, so the duplication `excess` carries can be
    #: collapsed before any statistic is computed on it. A date appears in
    #: C(N-1,k-1) splits, so the pooled list counts each one that many times and
    #: anything scaling with sqrt(n) reads inflated off it.
    excess_by_date: Dict[object, List[float]] = field(default_factory=dict)
    #: Label geometry, carried so the overlap correction can be derived here
    #: rather than guessed by the caller.
    horizon_sessions: int = 0
    step_sessions: int = 21
    #: WHAT THE IC AND THE EXCESS ARE MEASURED AGAINST. Not decoration. The
    #: ranker is fitted against the h-session forward return; the book earns
    #: whatever `resolve_exits` produces under a 2.5xATR stop, a 3R target and
    #: a 50-session invalidation level. Measured across 35,643 rows on 86
    #: dates, the within-date rank correlation between the two is 0.531 -- the
    #: label explains about 28% of the variance of what the book actually
    #: earns, and the book's own positions leave by invalidation 39.3% of the
    #: time, by stop 32.1%, by target 17.9% and by timeout 10.7%.
    #:
    #: Every IC and every top-decile excess in this object is therefore a
    #: statement about the LABEL, not about the book. Carrying the target's
    #: name on the result is what stops the two being read as one number.
    target: str = "label_rank"
    #: Rank correlation between the label and the realised book outcome, where
    #: the caller measured it. None means it was not measured, which is itself
    #: worth printing -- it is not the same as "they agree".
    label_book_rank_corr: Optional[float] = None

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

    # -- what the DSR is actually entitled to run on -------------------------
    def excess_per_date(self) -> List[float]:
        """One excess per DISTINCT test date, averaged over the splits that
        scored it. This is the series a statistic may be computed on.

        `excess` holds one entry per (split, date) pair. On the shipped geometry
        that is ~639 entries over 71 dates -- nine copies of each. Anything
        scaling with sqrt(n) read off the pooled list is inflated threefold
        before the label overlap is even counted.
        """
        if not self.excess_by_date:
            return list(self.excess)
        return [float(np.mean(v)) for _, v in sorted(
            self.excess_by_date.items(), key=lambda kv: str(kv[0]))]

    def effective_observations(self) -> float:
        """Independent observations behind `excess_per_date`.

        Two deductions, both arithmetic rather than estimated. Distinct dates
        removes the CPCV duplication; the analytic variance inflation of an
        h-session label sampled every s removes the overlap between neighbouring
        dates. At h=63, s=21 the second costs a factor of about three.
        """
        from .significance import analytic_vif

        n = len(self.excess_per_date())
        if n < 3:
            return float(n)
        if self.horizon_sessions <= 0 or self.step_sessions <= 0:
            return float(n)
        vif = analytic_vif(self.horizon_sessions, self.step_sessions, n)
        return float(n / vif) if vif > 0 else float(n)

    def deflated(self, n_trials: int):
        """DSR on the per-date excess, charged for the trial count, the CPCV
        duplication and the label overlap.

        Three things this call gets right that the previous one did not, all of
        which pushed the same way:

        * it runs on DISTINCT dates rather than (split, date) pairs;
        * it declares how many of those dates are independent, so the PSR's
          sqrt(n-1) term stops reading nine overlapping copies as evidence;
        * it hands over the woven path Sharpes, which ARE Bailey & Lopez de
          Prado's Var[SR] -- the cross-sectional dispersion across trials --
          instead of falling through to a null approximation.

        The previous version returned 1.000 on this data and still passed at
        100,000 trials.
        """
        return deflated_sharpe_ratio(
            self.excess_per_date(), n_trials=n_trials,
            trial_sharpes=(self.path_sharpes if len(self.path_sharpes) > 1 else None),
            effective_n=self.effective_observations(),
        )


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
    estimator: str = _ESTIMATOR_DEFAULT,
) -> CpcvResult:
    """Fit and score the ridge model across every CPCV split.

    ``panel`` carries one row per (date, symbol) with the ranked feature
    columns, ``label_rank`` and ``label``. Groups are formed over DATES, not
    rows: a date is one observation as far as leakage is concerned, and
    splitting rows would put the same day on both sides of the partition.
    """
    cols = [c for c in features if c in panel.columns]
    if purge_sessions < 0 or embargo_sessions < 0:
        raise ValueError(
            f"purge_sessions ({purge_sessions}) and embargo_sessions "
            f"({embargo_sessions}) must be non-negative. A negative value is "
            f"silently rounded to zero by the session-to-observation "
            f"conversion below, which disables the leakage guard while the run "
            f"reports normally."
        )
    if not cols:
        raise ValueError(
            "no usable feature columns: none of the requested features are "
            "present in the panel. A fit with no features returns an intercept, "
            "which scores an IC near zero and looks like a weak model rather "
            "than a broken call."
        )
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
    result = CpcvResult(n_splits=cv.n_splits, n_paths=cv.paths_per_observation(),
                        horizon_sessions=int(horizon_sessions),
                        step_sessions=int(step_sessions))
    # path_id -> per-date results, so each path can be scored as one backtest
    paths: Dict[int, List[Dict[str, float]]] = {}
    seen: Dict[int, int] = {}
    degenerate = 0

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

        fit = _fit(train, cols, alpha, estimator, horizon_sessions, step_sessions)
        if fit is None:
            result.notes.append(
                f"split {split.split_id}: the estimator produced no tradeable "
                f"coefficients; skipped")
            continue
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
                top = r >= top_decile
                # A model that predicted the same number for every name has no
                # top decile: `rank(pct=True)` gives every tied element the
                # midrank, so the mask selects NOTHING and the mean of an empty
                # slice is a NaN. That NaN then poisoned the pooled figure --
                # the headline economic number printed as "+nan%" while every
                # other line in the table read normally.
                if not top.any():
                    degenerate += 1
                    continue
                ex = float(lab[ok][top].mean() - lab[ok].mean())
                if not np.isfinite(ex):
                    degenerate += 1
                    continue
                result.excess.append(ex)
                # Keyed by date as well as pooled, so the duplication can be
                # collapsed before anything is inferred from it.
                result.excess_by_date.setdefault(d, []).append(ex)
                # Weave: the k-th time a date is tested it belongs to path k.
                pid = seen.get(hash(d), 0)
                seen[hash(d)] = pid + 1
                paths.setdefault(pid, []).append({"date": d, "excess": ex, "ic": ic})

    if degenerate:
        result.notes.append(
            f"{degenerate} test date(s) produced no top decile -- the model "
            f"predicted the same value for every name -- and were excluded "
            f"rather than averaged in as NaN")

    for pid, rows in sorted(paths.items()):
        vals = np.asarray([r["excess"] for r in rows], dtype="float64")
        ics = np.asarray([r["ic"] for r in rows], dtype="float64")
        ics = ics[np.isfinite(ics)]
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
    min_dates: int = 8,
    estimator: str = _ESTIMATOR_DEFAULT,
) -> pd.DataFrame:
    """Per-period performance of every configuration, on one common index.

    Feeds :func:`prosignal.validation.metrics.compute_pbo`, which asks a
    question no single backtest can: across the configurations actually tried,
    how often does the in-sample winner land below the out-of-sample median?
    That is the number that says whether a selection was skill or shopping.

    Every configuration is scored on identical dates with an identical purged
    expanding window, so the columns are comparable by construction.
    """
    if purge_sessions < 0:
        raise ValueError(
            f"purge_sessions ({purge_sessions}) must be non-negative; a negative "
            f"value rounds to zero and disables purging without reporting it"
        )
    purge_obs = int(np.ceil(purge_sessions / step_sessions))
    out: Dict[str, Dict[pd.Timestamp, float]] = {}

    single_theme_ungated = []
    for name, features in configurations.items():
        cols = [c for c in features if c in panel.columns]
        # A SINGLE-THEME ARM IS RUN UNGATED, AND SAID SO.
        #
        # Under the gated Fama-MacBeth a one-theme arm is not a strategy scored
        # on every split -- it is a strategy scored on the splits where that
        # theme happened to clear |t| >= 2, and it holds cash on the rest. Its
        # series is therefore conditioned on its own in-sample significance
        # while a multi-theme arm's is not, and comparing them compares two
        # different selection regimes, not two feature sets.
        #
        # Measured at audit time: `beta` alone traded on a minority of splits
        # and its scored dates were exactly the ones where it measured, which
        # is what carried the README's estimator-comparison conclusion.
        #
        # THIS CORRECTION RUNS IN THE FLATTERING DIRECTION and is flagged here
        # for that reason. The artifact made the single-theme CONTROLS look
        # good, so removing it can only help the production model. Every
        # conclusion drawn from a matrix containing single-theme arms must be
        # re-derived rather than inherited.
        arm_floor = 0.0 if len(cols) == 1 else None
        if arm_floor is not None:
            single_theme_ungated.append(name)
        work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
        dates = sorted(work["date"].unique())
        by_date = {d: g for d, g in work.groupby("date")}
        series: Dict[pd.Timestamp, float] = {}
        for i in range(min_train_dates + purge_obs, len(dates)):
            train = work[work["date"].isin(dates[: i - purge_obs])]
            if len(train) < min_train_rows:
                continue
            te = by_date[dates[i]]
            # `purge_sessions` IS the label horizon here -- that is what the
            # purge is sized from -- and it is what sets the Newey-West lag.
            fit = _fit(train, cols, alpha, estimator, purge_sessions,
                       step_sessions, significance_floor=arm_floor)
            if fit is None:
                continue
            pred = predict(fit, te[cols].to_numpy("float64"))
            lab = te["label"].to_numpy("float64")
            ok = np.isfinite(pred) & np.isfinite(lab)
            if ok.sum() < 40:
                continue
            r = pd.Series(pred[ok]).rank(pct=True).to_numpy()
            top = r >= top_decile
            # No top decile means every prediction tied. The `dropna` below
            # would drop that date for EVERY configuration, not just this one,
            # so a single degenerate arm silently shortens the shared index.
            if not top.any():
                continue
            series[dates[i]] = float(lab[ok][top].mean() - lab[ok].mean())
        out[name] = series

    frame = pd.DataFrame(out)
    if single_theme_ungated:
        frame.attrs["single_theme_ungated"] = list(single_theme_ungated)
        log.info("single-theme arms scored with the significance floor at zero",
                 extra={"arms": list(single_theme_ungated)})
    # A configuration the ESTIMATOR REFUSES scores nothing at all -- under the
    # gated Fama-MacBeth a single-theme arm that never clears |t| >= 2 produces
    # no coefficients on any split. Its column is entirely empty, and the row
    # `dropna` below would then delete every date for every OTHER configuration
    # too, so one unusable arm made PBO uncomputable rather than merely absent.
    #
    # A configuration that cannot be traded is not a configuration that could
    # have been chosen, so it is dropped from the comparison and named.
    kept = [c for c in frame.columns if frame[c].notna().sum() >= min_dates]
    dropped = [c for c in frame.columns if c not in kept]
    frame = frame[kept].dropna()
    if frame.empty or len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} configuration(s) produced a tradeable score on "
            f"{min_dates}+ dates"
            + (f"; {', '.join(dropped)} produced none" if dropped else "")
        )
    if dropped:
        frame.attrs["dropped_configurations"] = dropped
    return frame


@dataclass
class PortfolioCpcvResult:
    """Book-level performance across CPCV splits, not just ranking quality."""

    n_splits: int
    split_metrics: List[Dict[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def spread(self, key: str = "sharpe") -> Dict[str, float]:
        vals = np.asarray([m[key] for m in self.split_metrics if key in m],
                          dtype="float64")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {}
        return {
            "min": float(vals.min()),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "share_negative": float((vals < 0).mean()),
            "n": int(vals.size),
        }


def run_portfolio_cpcv(
    panel: pd.DataFrame,
    features: Sequence[str],
    prices: Dict[str, pd.DataFrame],
    portfolio_params,
    *,
    step_sessions: int,
    alpha: float,
    n_groups: int,
    n_test_groups: int,
    purge_sessions: int,
    embargo_sessions: int,
    min_train_rows: int = 2000,
    progress: Optional[Callable[[int, int], None]] = None,
    estimator: str = _ESTIMATOR_DEFAULT,
) -> PortfolioCpcvResult:
    """Fit on each CPCV training set, then trade the test blocks.

    The ranking is what CPCV normally scores. This scores the BOOK: the same
    splits, but each test block is walked with position sizing, the stop, the
    invalidation level, the buffer bands and costs. That is the only level at
    which risk-based sizing is visible, and it is where two of this audit's
    conclusions reversed.

    Each split contributes one set of portfolio metrics. Test blocks inside a
    split may not be contiguous; the simulator is restricted to the dates the
    split actually holds out, so a cohort never trades through a training block.
    """
    from .portfolio_sim import phase_summary

    cols = [c for c in features if c in panel.columns]
    if purge_sessions < 0 or embargo_sessions < 0:
        raise ValueError(
            f"purge_sessions ({purge_sessions}) and embargo_sessions "
            f"({embargo_sessions}) must be non-negative. A negative value is "
            f"silently rounded to zero by the session-to-observation "
            f"conversion below, which disables the leakage guard while the run "
            f"reports normally."
        )
    if not cols:
        raise ValueError(
            "no usable feature columns: none of the requested features are "
            "present in the panel. A fit with no features returns an intercept, "
            "which scores an IC near zero and looks like a weak model rather "
            "than a broken call."
        )
    work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
    dates = sorted(work["date"].unique())
    if len(dates) < n_groups * 2:
        raise ValueError(
            f"{len(dates)} panel dates cannot support {n_groups} CPCV groups"
        )
    purge_obs = int(np.ceil(purge_sessions / step_sessions))
    embargo_obs = int(np.ceil(embargo_sessions / step_sessions))
    cv = CombinatorialPurgedCV(
        n_groups=n_groups, n_test_groups=n_test_groups,
        label_horizon=purge_obs, embargo=embargo_obs,
    )
    by_date = {d: g for d, g in work.groupby("date")}
    result = PortfolioCpcvResult(n_splits=cv.n_splits)

    for n, split in enumerate(cv.split(len(dates)), start=1):
        if progress:
            progress(n, cv.n_splits)
        train = work[work["date"].isin([dates[i] for i in split.train_idx])]
        if len(train) < min_train_rows:
            continue
        # `purge_sessions` IS the label horizon -- the purge is sized from it --
        # and it is what sets the Newey-West lag.
        fit = _fit(train, cols, alpha, estimator, purge_sessions, step_sessions)
        if fit is None:
            result.notes.append(
                f"split {split.split_id}: the estimator produced no tradeable "
                f"coefficients; skipped")
            continue
        test_dates = [dates[i] for i in split.test_idx]
        rankings = []
        for d in test_dates:
            te = by_date[d]
            pred = predict(fit, te[cols].to_numpy("float64"))
            s = pd.Series(pred, index=te["symbol"].to_numpy()).sort_values(ascending=False)
            rankings.append((pd.Timestamp(d), s))
        if len(rankings) < 4:
            continue
        metrics = phase_summary(rankings, prices, portfolio_params,
                                step_sessions=step_sessions,
                                dates_allowed=[pd.Timestamp(d) for d in test_dates])
        if metrics:
            metrics["split_id"] = split.split_id
            result.split_metrics.append(metrics)

    log.info("portfolio cpcv complete",
             extra={"splits": cv.n_splits, "scored": len(result.split_metrics)})
    return result


@dataclass
class NestedResult:
    """Outer-loop performance WITH the cost of parameter selection included.

    A parameter chosen on the same data that reports the result is not a
    parameter, it is a fitted value, and the number it produces is in-sample
    however many folds surround it. Nested validation is the only construction
    that prices selection: the inner loop chooses, the outer loop reports, and
    the outer never sees the choice being made.
    """

    outer_splits: int
    #: one row per outer split: chosen parameters and the outer-test metrics
    rows: List[Dict[str, object]] = field(default_factory=list)
    #: why splits were skipped. An empty `rows` with nothing here is
    #: indistinguishable from a search that ran and found nothing, and a caller
    #: reading `len(rows) == 0` cannot tell "no model" from "no edge".
    notes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def spread(self, key: str) -> Dict[str, float]:
        vals = np.asarray([r[key] for r in self.rows if key in r and
                           np.isfinite(r[key])], dtype="float64")
        if vals.size == 0:
            return {}
        return {"min": float(vals.min()), "median": float(np.median(vals)),
                "max": float(vals.max()), "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "share_negative": float((vals < 0).mean()), "n": int(vals.size)}

    def chosen_counts(self, key: str) -> Dict[object, int]:
        """How often each parameter value won. A stable winner is evidence; a
        scatter across the grid means the inner loop is reading noise."""
        out: Dict[object, int] = {}
        for r in self.rows:
            out[r.get(key)] = out.get(r.get(key), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def nested_band_search(
    panel: pd.DataFrame,
    features: Sequence[str],
    prices: Dict[str, pd.DataFrame],
    make_params: Callable[[int, int], object],
    grid: Sequence[tuple],
    *,
    step_sessions: int,
    alpha: float,
    n_groups: int,
    n_test_groups: int,
    purge_sessions: int,
    embargo_sessions: int,
    inner_fraction: float = 0.3,
    min_train_rows: int = 2000,
    progress: Optional[Callable[[int, int], None]] = None,
    estimator: str = _ESTIMATOR_DEFAULT,
) -> NestedResult:
    """Choose Stage 6 bands inside each outer training set, report outside it.

    ``grid`` is a sequence of (entry_rank, exit_rank) pairs, fixed in advance.
    Within each outer split the training dates are cut again: the earlier part
    fits the model, the later ``inner_fraction`` selects the band. The chosen
    band is then applied to the outer test block, which nothing in the
    selection has touched.
    """
    from .portfolio_sim import phase_summary

    cols = [c for c in features if c in panel.columns]
    if purge_sessions < 0 or embargo_sessions < 0:
        raise ValueError("purge_sessions and embargo_sessions must be non-negative")
    if not cols:
        raise ValueError("no usable feature columns")
    if not grid:
        raise ValueError("the parameter grid is empty; nothing to select")
    work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
    dates = sorted(work["date"].unique())
    purge_obs = int(np.ceil(purge_sessions / step_sessions))
    cv = CombinatorialPurgedCV(
        n_groups=n_groups, n_test_groups=n_test_groups,
        label_horizon=purge_obs,
        embargo=int(np.ceil(embargo_sessions / step_sessions)),
    )
    by_date = {d: g for d, g in work.groupby("date")}
    result = NestedResult(outer_splits=cv.n_splits)

    def _rank(fit, ds):
        out = []
        for d in ds:
            te = by_date[d]
            p = predict(fit, te[cols].to_numpy("float64"))
            s = pd.Series(p, index=te["symbol"].to_numpy()).sort_values(ascending=False)
            out.append((pd.Timestamp(d), s))
        return out

    for n, split in enumerate(cv.split(len(dates)), start=1):
        if progress:
            progress(n, cv.n_splits)
        train_dates = [dates[i] for i in split.train_idx]
        test_dates = [dates[i] for i in split.test_idx]
        cut = int(len(train_dates) * (1.0 - inner_fraction))
        # The inner validation block is purged from the inner fit for the same
        # reason the outer one is: its labels reach backwards.
        fit_dates = train_dates[: max(cut - purge_obs, 0)]
        inner_dates = train_dates[cut:]
        if len(fit_dates) < 20 or len(inner_dates) < 4 or len(test_dates) < 4:
            continue
        inner_fit_rows = work[work["date"].isin(fit_dates)]
        if len(inner_fit_rows) < min_train_rows:
            continue
        inner_fit = _fit(inner_fit_rows, cols, alpha, estimator,
                         purge_sessions, step_sessions)
        if inner_fit is None:
            result.notes.append(
                f"split {n}: the inner fit produced no tradeable coefficients")
            continue
        inner_rank = _rank(inner_fit, inner_dates)

        best, best_sharpe = None, -np.inf
        for entry, exit_ in grid:
            m = phase_summary(inner_rank, prices, make_params(entry, exit_),
                              step_sessions=step_sessions,
                              dates_allowed=[pd.Timestamp(d) for d in inner_dates])
            if m and np.isfinite(m.get("sharpe", np.nan)) and m["sharpe"] > best_sharpe:
                best, best_sharpe = (entry, exit_), m["sharpe"]
        if best is None:
            continue

        outer_rows = work[work["date"].isin(train_dates)]
        if len(outer_rows) < min_train_rows:
            continue
        outer_fit = _fit(outer_rows, cols, alpha, estimator,
                         purge_sessions, step_sessions)
        if outer_fit is None:
            result.notes.append(
                f"split {n}: the outer fit produced no tradeable coefficients")
            continue
        outer = phase_summary(_rank(outer_fit, test_dates), prices,
                              make_params(*best), step_sessions=step_sessions,
                              dates_allowed=[pd.Timestamp(d) for d in test_dates])
        if not outer:
            continue
        result.rows.append({
            "split_id": split.split_id, "entry_rank": best[0], "exit_rank": best[1],
            "inner_sharpe": float(best_sharpe), "sharpe": outer["sharpe"],
            "mean_return": outer["mean_return"], "max_drawdown": outer["max_drawdown"],
        })

    log.info("nested band search complete",
             extra={"outer_splits": cv.n_splits, "scored": len(result.rows)})
    return result
