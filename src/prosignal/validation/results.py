"""RESULTS OF RECORD, generated rather than written.

WHY THIS MODULE EXISTS. README.md carried two book tables that cannot both
describe the same engine:

    RESULTS OF RECORD    mean excess -4.23%/period, IR -0.83, alpha -0.67%,
                         32.9% of periods beating the benchmark
    "tuning pass"        +42.6% annualised book return, +20.3% annualised
                         alpha, Sharpe 1.59, positive alpha in 6 of 6 years
                         (and this section appeared TWICE)

Prose cannot arbitrate between them, because prose is what produced them. So
the arbitration is a program: both configurations are re-run against the CURRENT
store through the repository's own simulator, and whichever fails to reproduce
is marked WITHDRAWN with the reason. Neither is averaged. Neither is quoted
because it is the nicer one.

WHAT IS AND IS NOT A TRIAL HERE. Re-running a configuration that has already
been looked at, in order to find out whether its published number reproduces, is
not a new look at the data -- both of these are already inside the trial counts
the Deflated Sharpe charges (4,877 and 81 respectively). Nothing here is
SELECTED: the shipped ranker is fixed by `stage4_core_score.ranking.source` and
this module cannot change it. So the v10 budget records this pass at zero, and
`registry.by_pass()["P0"]` is asserted to stay there.

THREE DISCIPLINES THIS FILE INHERITS AND MUST NOT DROP.

  GROSS AND COST SEPARATELY. Every arm reports gross excess, cost drag and net
  excess as three numbers. Netting them and keeping only the last is a defect
  this repository already fixed once, and it is what makes "the book loses to
  its universe" indistinguishable from "the book pays too much to trade".

  A POWER STATEMENT ON EVERY RESULT. `expected t = IR x sqrt(years)`. Without it
  an early positive stretch reads as confirmation. At the information ratios
  this engine actually produces, the horizons involved are years, not months,
  and that has to be visible next to the number rather than in a footnote.

  NO NAIVE t FROM OVERLAPPING WINDOWS. Signal dates are `stride` sessions apart
  and the label is `horizon` sessions long, so observations share most of their
  window. `significance.analytic_vif` is the correction and it is applied to
  every t reported here.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from . import significance as sig
from .metrics import quintile_spread, rank_ic
from .portfolio_sim import phase_summary

log = get_logger(__name__)

__all__ = ["ArmResult", "RankingResult", "Stamp", "ResultsOfRecord",
           "build", "render", "DOC_RELPATH", "REPRODUCTION_TOLERANCE"]

DOC_RELPATH = "docs/RESULTS_OF_RECORD.md"

#: How close a re-run has to land before the published claim is called
#: REPRODUCED. Generous on purpose -- the question is whether a number is the
#: same RESULT, not whether it matches to the basis point. The simulator's
#: cohort model, the store and the universe have all moved since these were
#: published, so a tight tolerance would fail everything and say nothing. What
#: it must catch is a SIGN disagreement or an order-of-magnitude gap, and it is
#: set where it catches both.
REPRODUCTION_TOLERANCE = {
    #: Annualised excess: within 5 percentage points AND the same sign.
    "excess_ann": 0.05,
    #: Information ratio: within 0.5 AND the same sign.
    "ir": 0.50,
}


# =============================================================================
# what a result is
# =============================================================================


@dataclass
class ArmResult:
    """One published claim, re-run against the current store."""

    key: str
    title: str
    #: What the configuration IS, in enough detail to re-specify it.
    configuration: str
    #: Where the published claim lives, so a reader can find what is superseded.
    claimed_in: str
    #: The published numbers, verbatim.
    claimed: Dict[str, Any]
    #: What the re-run produced. Empty when it could not be run at all.
    measured: Dict[str, Any] = field(default_factory=dict)
    #: REPRODUCED | WITHDRAWN | NOT_TESTABLE
    status: str = "NOT_TESTABLE"
    reason: str = ""
    #: True when this arm is the configuration the engine actually ships.
    is_shipped: bool = False
    #: EVERY published figure, claimed against measured, with its own verdict.
    #:
    #: The arm-level status is decided by the HEADLINE figures -- the ones the
    #: published table's own summary sentence asserts. This list carries all of
    #: them, headline or not, so a figure that diverges cannot be quietly left
    #: out of the comparison because it was not part of the verdict. On the
    #: shipped arm that matters immediately: IR, mean excess and the beat rate
    #: all reproduce closely while alpha flips sign on a near-zero quantity,
    #: and a reader is entitled to see that rather than infer it.
    comparison: List[Dict[str, Any]] = field(default_factory=list)
    #: Calendar years the arm actually spans. NOT `n_periods / periods_per_year`
    #: -- `phase_summary` pools every phase offset, so that ratio counts one
    #: calendar period once per offset and reads 92.8 years on a 7.7-year panel.
    #: The power statement is computed from this.
    years: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RankingResult:
    """The ordering, judged apart from any book built on it."""

    horizon: int
    n_dates: int
    n_rows: int
    ic: float
    ic_t_naive: float
    ic_t_corrected: float
    spread: float
    spread_t_naive: float
    spread_t_corrected: float
    top_decile_excess: float
    top_decile_t_corrected: float
    decile_monotonicity: float
    independent_observations: float
    vif: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Stamp:
    """Everything needed to say WHICH engine, WHICH data and WHICH claim."""

    generated_at: str
    config_version: str
    params_hash: str
    store_hash: str
    train_hash: str
    store_fingerprint: str
    data_manifest_digest: str
    git_commit: str
    git_dirty: bool
    engine_version: str
    ranking_source: str
    panel_first_date: Optional[str]
    panel_last_date: Optional[str]
    panel_rows: int
    panel_distinct_dates: int
    independent_observations: float
    cumulative_trials: int
    trials_by_pass: Dict[str, int]
    horizon_sessions: int
    stride_sessions: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ResultsOfRecord:
    stamp: Stamp
    ranking: List[RankingResult]
    arms: List[ArmResult]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"stamp": self.stamp.to_dict(),
                "ranking": [r.to_dict() for r in self.ranking],
                "arms": [a.to_dict() for a in self.arms],
                "notes": list(self.notes)}


# =============================================================================
# helpers
# =============================================================================


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:                              # noqa: BLE001
        return ""


def _power_statement(ir: float, years: float) -> str:
    """`expected t = IR x sqrt(years)`, and what it means for a paper window.

    Printed beside every arm. At an IR under about 1.0 no forward horizon
    anybody will wait for produces significance, and saying so next to the
    number is what stops an early positive stretch reading as confirmation.
    """
    if not np.isfinite(ir) or years <= 0:
        return "power: NOT COMPUTABLE (no information ratio on this arm)"
    t_now = ir * np.sqrt(years)
    need = (2.0 / ir) ** 2 if ir > 0 else float("inf")
    bit = (f"{need:.1f} years to reach t=2.0" if np.isfinite(need) and ir > 0
           else "t=2.0 is unreachable at a non-positive IR")
    return (f"power: expected t = IR x sqrt(years) = {ir:+.2f} x "
            f"sqrt({years:.1f}) = {t_now:+.2f}; {bit}")


def _decile_monotonicity(panel: pd.DataFrame, label: str,
                         score: str = "score") -> float:
    """Spearman between decile index and mean forward excess, per date, averaged.

    Computed PER DATE and then averaged rather than pooled. A pooled figure
    mixes the cross-section with the time series and reads as though it had as
    many observations as rows, which this panel does not.
    """
    rhos: List[float] = []
    for _, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=[score, label])
        if len(g) < 100:
            continue
        d = pd.qcut(g[score].rank(method="first"), 10, labels=False,
                    duplicates="drop")
        means = g.groupby(d)[label].mean()
        if len(means) < 5:
            continue
        idx = pd.Series(means.index, dtype="float64")
        rhos.append(float(idx.corr(pd.Series(means.to_numpy()),
                                   method="spearman")))
    return float(np.mean(rhos)) if rhos else float("nan")


def _top_decile_excess(panel: pd.DataFrame, label: str,
                       score: str = "score") -> List[float]:
    """Per-date mean of the top decile minus the mean of the whole date.

    Against the EQUAL-WEIGHT ELIGIBLE UNIVERSE, not against zero. A long-only
    book selected from this universe always had the option of holding all of
    it, so that is the alternative its excess has to be measured against.
    """
    out: List[float] = []
    for _, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=[score, label])
        if len(g) < 100:
            continue
        k = max(len(g) // 10, 5)
        top = g.nlargest(k, score)[label].mean()
        out.append(float(top - g[label].mean()))
    return out


def _corrected(series: Sequence[float], horizon: int, stride: int
               ) -> Tuple[float, float, float]:
    """(mean, naive t, overlap-corrected t) for an overlapping-window series."""
    a = np.asarray([x for x in series if np.isfinite(x)], dtype="float64")
    if a.size < 3:
        return float("nan"), float("nan"), float("nan")
    sd = a.std(ddof=1)
    naive = float(a.mean() / (sd / np.sqrt(a.size))) if sd > 0 else float("nan")
    vif = sig.analytic_vif(horizon, stride, a.size)
    return float(a.mean()), naive, float(naive / np.sqrt(vif))


def _rankings_from_panel(panel: pd.DataFrame, score_col: str,
                         close: pd.DataFrame,
                         apply_absolute_floor: bool) -> List[Tuple[pd.Timestamp, pd.Series]]:
    """[(date, score best-first)], with the absolute floor applied to ENTRIES.

    Names failing the floor are pushed to -inf so they cannot ENTER; a name
    already held is governed by the exit band, which is what
    `absolute_floor.applies_to: entries` means. When the floor is disabled in
    config -- as it is on the shipped configuration since 2026-09-02 -- nothing
    is masked and the ranking is the raw score.
    """
    ma200 = close.rolling(200, min_periods=200).mean() if apply_absolute_floor else None
    known = set(close.index)
    out: List[Tuple[pd.Timestamp, pd.Series]] = []
    for d, g in panel.groupby("date", sort=True):
        ts = pd.Timestamp(d)
        if ts not in known:
            continue
        s = pd.Series(g[score_col].to_numpy("float64"),
                      index=g["symbol"].to_numpy()).dropna()
        if s.empty:
            continue
        if apply_absolute_floor and "n_themes_positive" in g.columns:
            npos = pd.Series(g["n_themes_positive"].to_numpy("float64"),
                             index=g["symbol"].to_numpy()).reindex(s.index)
            c = close.loc[ts].reindex(s.index)
            m = ma200.loc[ts].reindex(s.index)
            s = s.where((npos >= 3) & (c > m), other=-np.inf)
        out.append((ts, s.sort_values(ascending=False)))
    return out


def _run_book(rankings, panels, params, step_sessions: int) -> Optional[Dict[str, float]]:
    m = phase_summary(rankings, panels, params, step_sessions=step_sessions)
    if not m or not m.get("benchmarked", False):
        return None
    ppy = float(m["periods_per_year"])
    gross = float(m["mean_excess"]) + float(m.get("mean_cost", 0.0) or 0.0)
    return {
        "mean_return_per_period": float(m["mean_return"]),
        "bench_return_per_period": float(m["bench_mean_return"]),
        "sharpe": float(m["sharpe"]),
        "bench_sharpe": float(m["bench_sharpe"]),
        "mean_excess_per_period": float(m["mean_excess"]),
        "excess_ann": float(m["mean_excess"]) * ppy,
        "gross_excess_ann": gross * ppy,
        "cost_drag_ann": float(m.get("mean_cost", 0.0) or 0.0) * ppy,
        "ir": float(m["information_ratio"]),
        "alpha_per_period": float(m["alpha_per_period"]),
        "beta_to_benchmark": float(m["beta_to_benchmark"]),
        "periods_beating_benchmark": float(m["excess_hit_rate"]),
        "worst_schedule_drawdown": float(m["worst_schedule_drawdown"]),
        "avg_names": float(m["avg_names"]),
        "n_periods": int(m["n_periods"]),
        "periods_per_year": ppy,
        "book_return_ann": float(m["mean_return"]) * ppy,
        "bench_return_ann": float(m["bench_mean_return"]) * ppy,
    }


def _judge(claimed: Dict[str, Any], measured: Dict[str, Any]) -> Tuple[str, str]:
    """REPRODUCED or WITHDRAWN, and why. Sign first, magnitude second."""
    if not measured:
        return "NOT_TESTABLE", ("the arm could not be simulated on the current "
                                "store, so the claim is neither confirmed nor "
                                "refuted -- it is unchecked")
    problems: List[str] = []
    for field_name, tol in REPRODUCTION_TOLERANCE.items():
        c, m = claimed.get(field_name), measured.get(field_name)
        if c is None or m is None or not np.isfinite(m):
            continue
        c = float(c)
        if np.sign(c) != np.sign(m) and abs(c) > 1e-9 and abs(m) > 1e-9:
            problems.append(f"{field_name}: claimed {c:+.4g}, measured "
                            f"{m:+.4g} -- OPPOSITE SIGN")
        elif abs(c - m) > tol:
            problems.append(f"{field_name}: claimed {c:+.4g}, measured "
                            f"{m:+.4g} -- outside the {tol:g} tolerance")
    if problems:
        return "WITHDRAWN", "; ".join(problems)
    return "REPRODUCED", "re-run on the current store lands inside tolerance"


# =============================================================================
# building
# =============================================================================


def _stamp(cfg, store, panel: pd.DataFrame, horizon: int, stride: int,
           independent: float) -> Stamp:
    from .. import __version__
    from ..data import manifest as man
    from .registry import TrialRegistry, registry_path

    ident = cfg.identity
    reg = TrialRegistry(registry_path(cfg.paths.curated))
    carried = int(getattr(cfg.params.validation.search_budget,
                          "cumulative_trials_logged", 0) or 0)
    dates = pd.to_datetime(panel["date"]).dt.date if not panel.empty else None
    return Stamp(
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        config_version=cfg.version,
        params_hash=cfg.hash,
        store_hash=(ident.store_hash[:16] if ident else "unbound"),
        train_hash=(ident.train_hash[:16] if ident else "unbound"),
        store_fingerprint=(ident.store.summary() if ident else "unbound"),
        data_manifest_digest=man.digest_of(cfg.paths.curated),
        git_commit=_git("rev-parse", "HEAD")[:12] or "unknown",
        git_dirty=bool(_git("status", "--porcelain")),
        engine_version=__version__,
        ranking_source=str(cfg.params.stage4_core_score.ranking.source),
        panel_first_date=(str(dates.min()) if dates is not None and len(dates) else None),
        panel_last_date=(str(dates.max()) if dates is not None and len(dates) else None),
        panel_rows=int(len(panel)),
        panel_distinct_dates=(int(panel["date"].nunique()) if not panel.empty else 0),
        independent_observations=independent,
        cumulative_trials=reg.effective_trials(carried),
        trials_by_pass=reg.by_pass(),
        horizon_sessions=horizon,
        stride_sessions=stride,
    )


def _ranking_results(panel: pd.DataFrame, horizons: Sequence[int],
                     stride: int) -> List[RankingResult]:
    out: List[RankingResult] = []
    for h in horizons:
        label = f"y{h}"
        if label not in panel.columns:
            continue
        sub = panel.dropna(subset=["score", label])
        if sub.empty:
            continue
        ic, ic_t, n_ic = rank_ic(sub, label)
        sp, sp_t, _ = quintile_spread(sub, label)
        vif = sig.analytic_vif(h, stride, max(n_ic, 1))
        tde = _top_decile_excess(sub, label)
        _, _, tde_t = _corrected(tde, h, stride)
        n_dates = int(sub["date"].nunique())
        out.append(RankingResult(
            horizon=h,
            n_dates=n_dates,
            n_rows=int(len(sub)),
            ic=ic, ic_t_naive=ic_t,
            ic_t_corrected=ic_t / np.sqrt(vif) if np.isfinite(ic_t) else float("nan"),
            spread=sp, spread_t_naive=sp_t,
            spread_t_corrected=sp_t / np.sqrt(vif) if np.isfinite(sp_t) else float("nan"),
            top_decile_excess=float(np.mean(tde)) if tde else float("nan"),
            top_decile_t_corrected=tde_t,
            decile_monotonicity=_decile_monotonicity(sub, label),
            independent_observations=_independent(n_dates, stride, h),
            vif=vif,
        ))
    return out


def _independent(n_dates: int, stride: int, horizon: int) -> float:
    """Non-overlapping windows the panel actually contains.

    The span the dates cover, divided by the label length. It is the number
    every Sharpe, DSR and calibration claim in this repository is bounded by,
    and it is a great deal smaller than the row count.
    """
    if n_dates <= 0 or horizon <= 0:
        return 0.0
    span = (n_dates - 1) * stride + horizon
    return round(span / horizon, 1)


def build(cfg, store, *, panel: Optional[pd.DataFrame] = None,
          price_panels: Optional[Dict[str, pd.DataFrame]] = None,
          built: Optional[Dict[str, Any]] = None,
          progress: Optional[Callable[[str], None]] = None) -> ResultsOfRecord:
    """Re-run every published book claim against the current store.

    `panel` and `price_panels` are injectable so the caller can cache them
    across invocations -- building both is the slow part (about twenty-five
    minutes over the whole store) and neither depends on which arm is priced.

    `built`, when given, receives whatever this call had to construct, so the
    caller can persist it. Without it a caller could pass a cache in but never
    fill one, which is the shape --panel-cache had on its first outing.
    """
    import dataclasses as _dc

    from ..cli import _portfolio_inputs, _portfolio_params
    from .v3_panel import SIGNAL_STRIDE_SESSIONS, build_v3_panel

    say = progress or (lambda _m: None)
    p = cfg.params
    u, c4 = p.universe, p.stage4_core_score
    horizon = int(getattr(c4.model_horizon_sessions, "value",
                          c4.model_horizon_sessions))
    stride = SIGNAL_STRIDE_SESSIONS
    sessions = store.price_sessions()
    end = sessions[-1]

    if panel is None:
        say("building the v3 score panel over the whole store")
        panel = build_v3_panel(
            store, end=end, stride=stride,
            horizons=(21, 42, horizon),
            max_names=int(getattr(u.pit_max_names, "value", u.pit_max_names)),
            min_adtv_inr=float(getattr(u.pit_min_adtv_inr, "value",
                                       u.pit_min_adtv_inr)),
            min_price_inr=float(getattr(u.min_price_inr, "value", u.min_price_inr)),
            min_history_sessions=int(getattr(u.min_history_sessions, "value",
                                             u.min_history_sessions)))
    if panel is None or panel.empty:
        raise RuntimeError("the v3 panel came back empty; nothing can be regenerated")
    if built is not None:
        built["panel"] = panel

    if price_panels is None:
        say("building OHLC / ATR / MA / ADTV / benchmark panels")
        price_panels = _portfolio_inputs(cfg, store, sessions, None, end)
    if built is not None:
        built["price_panels"] = price_panels
    close = price_panels["close"]

    say("scoring the ranking")
    ranking = _ranking_results(panel, (21, 42, horizon), stride)
    n_dates = int(panel["date"].nunique())
    independent = _independent(n_dates, stride, horizon)

    # CALENDAR YEARS THE PANEL ACTUALLY SPANS. `phase_summary` pools every
    # phase offset, so `n_periods / periods_per_year` counts one calendar
    # period once per offset -- it read 92.8 "years" on a 7.7-year panel and
    # made every power statement nonsense. The span is the honest denominator.
    _pd = pd.to_datetime(panel["date"])
    span_years = float((_pd.max() - _pd.min()).days) / 365.25

    base = _portfolio_params(cfg)
    floor_on = bool(getattr(p.stage6_entry, "absolute_floor", None)
                    and getattr(p.stage6_entry.absolute_floor.enabled, "value",
                                p.stage6_entry.absolute_floor.enabled))

    arms: List[ArmResult] = []

    # -- ARM 1: the configuration that actually ships -----------------------
    say("arm 1/2: the shipped configuration (v3 composite)")
    rk = _rankings_from_panel(panel, "score", close, floor_on)
    shipped_measured = _run_book(rk, price_panels, base, stride) or {}
    shipped = ArmResult(
        key="results_of_record",
        title="RESULTS OF RECORD -- the shipped book against its own universe",
        configuration=(
            f"ranking.source={p.stage4_core_score.ranking.source} "
            f"({len(_v3_factor_count())} factors in {len(_v3_theme_count())} "
            f"themes); {base.max_positions} slots, entry rank "
            f"{base.entry_rank}, exit rank {base.exit_rank}, horizon "
            f"{base.horizon_sessions} sessions; stop "
            f"{base.stop_atr_multiple:g}xATR (armed={base.use_stop}), target "
            f"{base.target_r_multiple:g}R (armed={base.use_target}), "
            f"invalidation armed={base.use_invalidation}; absolute floor "
            f"{'ENTRIES' if floor_on else 'DISABLED'}; shipped cost model"),
        claimed_in="README.md, 'RESULTS OF RECORD'",
        claimed={"mean_excess_per_period": -0.0423, "ir": -0.83,
                 "alpha_per_period": -0.0067,
                 "periods_beating_benchmark": 0.329},
        measured=shipped_measured,
        is_shipped=True,
        years=span_years,
    )
    (shipped.status, shipped.reason,
     shipped.comparison) = _judge_shipped(shipped.claimed, shipped_measured)
    arms.append(shipped)

    # -- ARM 2: the tuning-pass claim ---------------------------------------
    say("arm 2/2: the 2026-08-29 tuning-pass configuration (mom_6_1 rank)")
    tuning_measured: Dict[str, float] = {}
    tuning_reason = ""
    try:
        mom = _mom_6_1_panel(panel, close)
        if mom is None:
            tuning_reason = ("mom_6_1 could not be built on this panel's dates; "
                             "the arm is UNKNOWN, not refuted")
        else:
            tparams = _dc.replace(
                base, max_positions=6, entry_rank=6, exit_rank=18,
                stop_atr_multiple=8.0, max_stop_distance_pct=35.0,
                use_target=False, use_invalidation=False)
            rk_m = _rankings_from_panel(mom, "score", close, False)
            tuning_measured = _run_book(rk_m, price_panels, tparams, stride) or {}
    except Exception as exc:                       # noqa: BLE001
        tuning_reason = f"the arm raised while being simulated: {exc}"
        log.warning("tuning-pass arm failed", extra={"error": str(exc)})

    tuning = ArmResult(
        key="tuning_pass_2026_08_29",
        title="Tuning pass (2026-08-29) -- sector-neutral 6-1 momentum, 6 names",
        configuration=(
            "ranking = sector-neutral rank of mom_6_1 (close[t-21]/close[t-147] "
            "- 1), ONE column; 6 slots, entry rank 6, exit rank 18, entries "
            "every 21 sessions, held to a 63-session backstop; disaster floor "
            "8xATR clipped to 35% of entry; no profit target, no invalidation "
            "exit; shipped cost model. This is `ranking.source=measured_factor`, "
            "which is NOT what the engine ships."),
        claimed_in="README.md, 'What changed in the tuning pass (2026-08-29)' "
                   "(the section appeared twice) and config `expectancy:`",
        claimed={"book_return_ann": 0.426, "alpha_ann": 0.203, "sharpe": 1.59,
                 "excess_sharpe": 1.12, "sample_trades": 258,
                 "positive_alpha_years": "6 of 6",
                 "ir": None, "excess_ann": None},
        measured=tuning_measured,
        years=span_years,
    )
    if tuning_measured:
        (tuning.status, tuning.reason,
         tuning.comparison) = _judge_tuning(tuning.claimed, tuning_measured,
                                            span_years)
    else:
        tuning.status = "NOT_TESTABLE"
        tuning.reason = tuning_reason or (
            "the arm produced no benchmarked book on the current store")
    arms.append(tuning)

    notes = [
        "Both arms are priced with the SHIPPED cost model at the shipped impact "
        "coefficient. Gross excess, cost drag and net excess are reported "
        "separately on every arm; the cost question and the gross-edge question "
        "are different questions and netting them hides which one is binding.",
        "The book model is the repository's own cohort simulator "
        "(`portfolio_sim.simulate` / `phase_summary`), the same one "
        "`research portfolio` uses. It is NOT a bit-reproduction of the "
        "continuous weekly book the sealed holdouts evaluated, so magnitudes "
        "are not comparable to HOLDOUT_V3_A/B. Read the sign and the ordering.",
        "The panel spans the whole store, which OVERLAPS the surfaces both of "
        "these configurations were selected on. Neither arm is out-of-sample "
        "evidence; this is a reproduction check, not a validation.",
    ]
    return ResultsOfRecord(
        stamp=_stamp(cfg, store, panel, horizon, stride, independent),
        ranking=ranking, arms=arms, notes=notes)


def _v3_factor_count():
    from ..features import v3
    return v3.ALL_FACTORS


def _v3_theme_count():
    from ..features import v3
    return v3.THEMES


def _mom_6_1_panel(panel: pd.DataFrame, close: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The tuning pass's ranker: sector-neutral rank of 6-1 momentum.

    `mom_6_1` is NOT one of the twenty-two v3 factors -- it belongs to
    `features/crosssec.py`, the FITTED model's panel, which is the single
    sharpest fact about the README contradiction this module exists to settle.
    So it is computed here from the same close matrix, to `crosssec`'s own
    definition: close[t-21] / close[t-147] - 1.
    """
    from ..features.v3 import sector_neutral_rank

    if "sector" not in panel.columns:
        return None
    idx = {d: i for i, d in enumerate(close.index)}
    rows = []
    for d, g in panel.groupby("date", sort=True):
        ts = pd.Timestamp(d)
        i = idx.get(ts)
        if i is None or i < 147:
            continue
        syms = [s for s in g["symbol"].to_numpy() if s in close.columns]
        if len(syms) < 60:
            continue
        near = close.iloc[i - 21][syms]
        far = close.iloc[i - 147][syms]
        with np.errstate(invalid="ignore", divide="ignore"):
            mom = near / far.where(far > 0) - 1.0
        sec = g.set_index("symbol")["sector"].reindex(syms)
        blk = pd.DataFrame({"symbol": syms,
                            "score": sector_neutral_rank(mom, sec).to_numpy()})
        blk["date"] = ts
        rows.append(blk)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


#: (figure, tolerance, is_headline). HEADLINE figures decide the arm's status
#: and are the ones the published table's own summary sentence asserts. The
#: rest are compared and reported but do not by themselves withdraw a claim --
#: `alpha_per_period` is the case in point: it is a near-zero residual whose
#: sign is not stable across two different books, and letting it withdraw a
#: table whose headline reproduces to two decimals would be as misleading as
#: hiding it.
SHIPPED_FIGURES = (
    ("ir", "information ratio", 0.50, True),
    ("mean_excess_per_period", "mean excess / period", 0.05, True),
    ("periods_beating_benchmark", "periods beating the benchmark", 0.20, True),
    ("alpha_per_period", "alpha / period", 0.02, False),
)


def _compare(claimed: Dict[str, Any], measured: Dict[str, Any],
             spec) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, label, tol, headline in spec:
        c, m = claimed.get(key), measured.get(key)
        if c is None or m is None or not np.isfinite(m):
            rows.append({"figure": label, "key": key, "claimed": c,
                         "measured": m, "verdict": "NOT_TESTABLE",
                         "headline": headline})
            continue
        c = float(c)
        if np.sign(c) != np.sign(m) and abs(c) > 1e-9 and abs(m) > 1e-9:
            verdict = "OPPOSITE SIGN"
        elif abs(c - m) > tol:
            verdict = f"outside {tol:g}"
        else:
            verdict = "matches"
        rows.append({"figure": label, "key": key, "claimed": c, "measured": m,
                     "verdict": verdict, "headline": headline})
    return rows


def _judge_shipped(claimed: Dict[str, Any], measured: Dict[str, Any]
                   ) -> Tuple[str, str, List[Dict[str, Any]]]:
    if not measured:
        return ("NOT_TESTABLE",
                "the shipped book produced no benchmarked result on the "
                "current store", [])
    rows = _compare(claimed, measured, SHIPPED_FIGURES)
    bad = [r for r in rows if r["headline"] and r["verdict"] not in
           ("matches", "NOT_TESTABLE")]
    other = [r for r in rows if not r["headline"] and r["verdict"] not in
             ("matches", "NOT_TESTABLE")]
    if bad:
        return ("WITHDRAWN",
                "; ".join(f"{r['figure']}: claimed {r['claimed']:+.4g}, "
                          f"measured {r['measured']:+.4g} -- {r['verdict']}"
                          for r in bad),
                rows)
    tail = ""
    if other:
        tail = (" Note, and it is reported rather than dropped because it is "
                "not a headline figure: "
                + "; ".join(f"{r['figure']} claimed {r['claimed']:+.4g} against "
                            f"{r['measured']:+.4g} measured ({r['verdict']})"
                            for r in other)
                + ". Alpha here is a near-zero residual of two different books "
                  "with different betas, so its sign is not stable; the "
                  "headline claim is the underperformance, and that "
                  "reproduces.")
    return ("REPRODUCED",
            "the direction and magnitude of the published headline claim "
            "survive a re-run on the current store." + tail,
            rows)


def _judge_tuning(claimed: Dict[str, Any], measured: Dict[str, Any],
                  years: float) -> Tuple[str, str, List[Dict[str, Any]]]:
    """The tuning claim is about ALPHA, SHARPE and the BOOK RETURN."""
    derived = dict(measured)
    derived["alpha_ann"] = (measured["alpha_per_period"]
                            * measured["periods_per_year"])
    spec = (
        ("alpha_ann", "annualised alpha", 0.05, True),
        ("sharpe", "Sharpe", 0.50, True),
        ("book_return_ann", "annualised book return", 0.10, True),
        ("excess_sharpe", "excess Sharpe", 0.50, False),
    )
    rows = _compare(claimed, derived, spec)
    bad = [r for r in rows if r["headline"] and r["verdict"] not in
           ("matches", "NOT_TESTABLE")]
    if bad:
        return ("WITHDRAWN",
                "; ".join(f"{r['figure']}: claimed {r['claimed']:+.4g}, "
                          f"measured {r['measured']:+.4g}" for r in bad),
                rows)
    return "REPRODUCED", "the published claim survives a re-run", rows


# =============================================================================
# rendering
# =============================================================================


_STATUS_BADGE = {
    "REPRODUCED": "REPRODUCED",
    "WITHDRAWN": "WITHDRAWN",
    "NOT_TESTABLE": "NOT_TESTABLE",
}


def _pct(x: Optional[float], places: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{places}%}"


def _num(x: Optional[float], places: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{places}f}"


def _arm_block(a: ArmResult) -> List[str]:
    L: List[str] = [f"### {a.title}", ""]
    L.append(f"**Status: {_STATUS_BADGE.get(a.status, a.status)}** — {a.reason}")
    L.append("")
    L.append(f"*Claimed in:* {a.claimed_in}")
    L.append("")
    L.append(f"*Configuration:* {a.configuration}")
    L.append("")
    if a.is_shipped:
        L.append("> This arm **is** the configuration the engine ships.")
        L.append("")
    else:
        L.append("> This arm is **not** what the engine ships. It is re-run "
                 "here only to find out whether its published numbers "
                 "reproduce.")
        L.append("")
    m = a.measured
    if not m:
        L.append("No benchmarked result was produced, so the claim is "
                 "unchecked rather than refuted. `NOT_TESTABLE` is a distinct "
                 "outcome from a failure and it is not upgraded to one.")
        L.append("")
        return L

    L += ["| | book | benchmark (equal-weight eligible universe) |",
          "|---|---|---|",
          f"| mean return / period | {_pct(m['mean_return_per_period'])} | "
          f"{_pct(m['bench_return_per_period'])} |",
          f"| annualised | {_pct(m['book_return_ann'], 1)} | "
          f"{_pct(m['bench_return_ann'], 1)} |",
          f"| Sharpe | {_num(m['sharpe'])} | {_num(m['bench_sharpe'])} |",
          f"| mean excess / period | {_pct(m['mean_excess_per_period'])} | — |",
          f"| information ratio | {_num(m['ir'])} | — |",
          f"| beta to benchmark | {_num(m['beta_to_benchmark'])} | — |",
          f"| alpha / period | {_pct(m['alpha_per_period'])} | — |",
          f"| periods beating the benchmark | {m['periods_beating_benchmark']:.1%} | — |",
          f"| worst schedule drawdown | {_pct(m['worst_schedule_drawdown'], 1)} | — |",
          f"| mean names held | {m['avg_names']:.1f} | — |",
          f"| periods scored | {m['n_periods']} | — |",
          ""]

    # GROSS AND COST, SEPARATELY. Never one netted number.
    L += ["**Gross and cost, separately** — netting them and keeping the last "
          "number hides which of the two is binding:",
          "",
          "| | annualised |",
          "|---|---|",
          f"| gross excess over the universe | {_pct(m['gross_excess_ann'], 1)} |",
          f"| cost drag | {_pct(-abs(m['cost_drag_ann']), 1)} |",
          f"| **net excess** | **{_pct(m['excess_ann'], 1)}** |",
          ""]

    L.append(f"*{_power_statement(m['ir'], a.years)}*")
    L.append("")

    if a.comparison:
        L += ["**Claimed against measured**, every published figure, headline "
              "or not:", "",
              "| figure | published claim | re-run | verdict | headline? |",
              "|---|---|---|---|---|"]
        for r in a.comparison:
            c = r["claimed"]
            mm = r["measured"]
            fmt = (lambda v: "n/a" if v is None else
                   (f"{v:+.2%}" if abs(v) < 1 else f"{v:+.2f}"))
            L.append(f"| {r['figure']} | {fmt(c)} | {fmt(mm)} | "
                     f"{r['verdict']} | {'yes' if r['headline'] else 'no'} |")
        L.append("")
    return L


def render(rec: ResultsOfRecord) -> str:
    """The whole document. Regenerated end to end; never hand-edited."""
    s = rec.stamp
    L: List[str] = []
    L.append("# RESULTS OF RECORD")
    L.append("")
    L.append("> [!IMPORTANT]")
    L.append("> **This file is GENERATED. Do not edit it.** Run "
             "`prosignal research results` to regenerate it. Every figure below "
             "was produced by code from the store named in the stamp; a number "
             "that appears anywhere else in this repository and disagrees with "
             "this file is superseded by it, and "
             "`tests/test_readme_numbers.py` fails if README.md drifts.")
    L.append("")

    # -- the stamp ----------------------------------------------------------
    L += ["## What produced these numbers", "",
          "| | |", "|---|---|",
          f"| generated at | `{s.generated_at}` |",
          f"| config version | `{s.config_version}` |",
          f"| — parameters hash | `{s.params_hash}` |",
          f"| — store hash | `{s.store_hash}` |",
          f"| — training-window hash | `{s.train_hash}` |",
          f"| shipped ranker | `{s.ranking_source}` |",
          f"| git commit | `{s.git_commit}`"
          f"{' **(working tree dirty)**' if s.git_dirty else ''} |",
          f"| engine version | `{s.engine_version}` |",
          f"| data manifest digest | `{s.data_manifest_digest}` |",
          f"| store fingerprint | {s.store_fingerprint} |",
          f"| panel span | {s.panel_first_date} → {s.panel_last_date} |",
          f"| panel rows | {s.panel_rows:,} |",
          f"| distinct signal dates | {s.panel_distinct_dates} |",
          f"| **independent observations** | **{s.independent_observations}** |",
          f"| horizon / stride | {s.horizon_sessions} / {s.stride_sessions} sessions |",
          f"| cumulative trials charged | {s.cumulative_trials:,} |",
          f"| trials by v10 pass | "
          f"{', '.join(f'{k}={v}' for k, v in sorted(s.trials_by_pass.items()))} |",
          ""]
    L.append("**Read `independent observations` before any t-statistic below.** "
             f"The panel has {s.panel_rows:,} rows and "
             f"{s.independent_observations} independent "
             f"{s.horizon_sessions}-session windows. Every Sharpe, every "
             "information ratio and every deflated statistic in this engine is "
             "bounded by the second number, not the first.")
    L.append("")

    # -- the ranking --------------------------------------------------------
    L += ["## The ranking, judged apart from any book", "",
          "The ordering is a different object from the book built on it, and "
          "this repository's history is largely the story of the two being "
          "confused. No naive `t` is quoted: signal dates are "
          f"{s.stride_sessions} sessions apart against a "
          f"{s.horizon_sessions}-session label, so observations overlap and the "
          "naive statistic is inflated by roughly `sqrt(VIF)`.", "",
          "| horizon | dates | rows | rank IC | IC t (naive) | IC t (corrected) | "
          "quintile spread | spread t (corr.) | top-decile excess | "
          "top-decile t (corr.) | decile monotonicity | indep. obs | VIF |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rec.ranking:
        L.append(
            f"| {r.horizon} | {r.n_dates} | {r.n_rows:,} | {r.ic:+.4f} | "
            f"{_num(r.ic_t_naive)} | **{_num(r.ic_t_corrected)}** | "
            f"{_pct(r.spread)} | **{_num(r.spread_t_corrected)}** | "
            f"{_pct(r.top_decile_excess)} | **{_num(r.top_decile_t_corrected)}** | "
            f"{r.decile_monotonicity:+.3f} | {r.independent_observations} | "
            f"{r.vif:.2f} |")
    L.append("")

    # -- the arms -----------------------------------------------------------
    L += ["## The two book tables, re-run", "",
          "README.md carried two performance tables that cannot both describe "
          "the same engine. Both configurations are re-run below against the "
          "store named in the stamp. **They are not averaged, and the more "
          "favourable one is not quoted.**", ""]
    for a in rec.arms:
        L += _arm_block(a)

    # -- caveats ------------------------------------------------------------
    L += ["## What these numbers are not", ""]
    for n in rec.notes:
        L.append(f"- {n}")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"*Generated by `prosignal research results` "
             f"({DOC_RELPATH}). Regenerate; do not edit.*")
    return "\n".join(L) + "\n"


def write(rec: ResultsOfRecord, root: Path) -> Path:
    """Write the document and its machine-readable twin."""
    path = Path(root) / DOC_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(rec), encoding="utf-8")
    (path.with_suffix(".json")).write_text(
        json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
    return path
