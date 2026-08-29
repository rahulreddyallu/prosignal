"""Section G, regenerated -- and then each measurement defect priced against it.

Two ranking constructions, because the choice matters and hiding it would be the
kind of thing this audit is about:

  weave     CPCV path 0. Every panel date scored once, by the first split that
            held it out. 70 dates. This is the repo's own construction and it
            is what section G's "70 out-of-sample dates" means. CPCV training
            blocks may sit AFTER the test block; purge and embargo handle the
            label overlap but the fit still sees a later calendar.

  forward   Purged expanding walk-forward. Fit on everything up to `i - purge`,
            rank date `i`. Fewer dates, but no fit ever sees a later session --
            which is the only construction a live engine can actually run.

Every figure is reported on both. A finding that holds on one and not the other
is reported as that, not as a finding.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.costs import CostModel
from prosignal.features.linear import predict
from prosignal.stages._cfg import fv, iv
from prosignal.validation.cpcv import CombinatorialPurgedCV
from prosignal.validation.harness import _fit
from prosignal.validation.significance import newey_west_t

from book import Book, phases, simulate

CACHE = Path(__file__).resolve().parent / "cache"
STEP = 21


# =============================================================================
# rankings
# =============================================================================
def build_rankings(panel, features, cfg):
    c4, val = cfg.params.stage4_core_score, cfg.params.validation
    alpha, est = fv(c4.model_ridge_alpha), str(c4.estimator.method)
    horizon = iv(c4.model_horizon_sessions)
    purge, embargo = iv(val.cpcv.purge_sessions), iv(val.cpcv.embargo_sessions)
    cols = [c for c in features if c in panel.columns]
    work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
    dates = sorted(work["date"].unique())
    by_date = {d: g for d, g in work.groupby("date")}
    purge_obs = int(np.ceil(purge / STEP))

    def rank_on(fit, d):
        te = by_date[d]
        p = predict(fit, te[cols].to_numpy("float64"))
        return pd.Series(p, index=te["symbol"].to_numpy()).sort_values(ascending=False)

    # -- weave: first split to hold a date out owns it ----------------------
    cv = CombinatorialPurgedCV(n_groups=iv(val.cpcv.n_groups), n_test_groups=2,
                               label_horizon=purge_obs,
                               embargo=int(np.ceil(embargo / STEP)))
    weave = {}
    for n, split in enumerate(cv.split(len(dates)), start=1):
        train = work[work["date"].isin([dates[i] for i in split.train_idx])]
        if len(train) < 2000:
            continue
        fit = _fit(train, cols, alpha, est, horizon, STEP)
        if fit is None:
            continue
        for d in [dates[i] for i in split.test_idx]:
            if d not in weave:
                weave[d] = rank_on(fit, d)
        if n % 10 == 0:
            print(f"  weave split {n}/{cv.n_splits}", flush=True)

    # -- forward: expanding, purged ----------------------------------------
    forward = {}
    for i in range(30 + purge_obs, len(dates)):
        train = work[work["date"].isin(dates[: i - purge_obs])]
        if len(train) < 2000:
            continue
        fit = _fit(train, cols, alpha, est, horizon, STEP)
        if fit is None:
            continue
        forward[dates[i]] = rank_on(fit, dates[i])
    print(f"  weave {len(weave)} dates, forward {len(forward)} dates")

    fmt = lambda d: [(pd.Timestamp(k), v) for k, v in sorted(d.items())]
    return fmt(weave), fmt(forward), work


def benchmark(work, rankings, b: Book, px):
    """Equal-weight eligible universe, on exactly the dates the book traded.

    Layer 0. The book is measured against the alternative of holding everything
    the screen admitted, which is the comparison the engine had none of.
    """
    close = px["close"]
    pos = {d: i for i, d in enumerate(close.index)}
    lab = work.groupby("date")["label"].mean()
    lab.index = pd.to_datetime(lab.index)
    stride = max(int(np.ceil(b.horizon_sessions / STEP)), 1)
    out = []
    for p in range(stride):
        got = []
        for j in range(p, len(rankings), stride):
            d = rankings[j][0]
            if d not in pos or pos[d] + b.horizon_sessions >= len(close.index):
                continue
            if d in lab.index:
                got.append(float(lab.loc[d]))
        if len(got) >= 3:
            out.extend(got)
    return np.asarray(out, dtype="float64")


# =============================================================================
def stat(r, bench, horizon):
    r = np.asarray(r, dtype="float64")
    ex = r - float(np.mean(bench)) if bench is not None else r
    try:
        t = newey_west_t(ex, horizon_sessions=horizon, step_sessions=STEP).adjusted_t
    except ValueError:
        t = float("nan")
    return t


def row(name, m, bench_mean, horizon, prev=None):
    r = m["returns"]
    vs = m["mean_return"] - bench_mean
    try:
        t = newey_west_t(r - bench_mean, horizon_sessions=horizon,
                         step_sessions=STEP).adjusted_t
    except ValueError:
        t = float("nan")
    d = "" if prev is None else f"{m['mean_return'] - prev:+.2%}"
    return (f"{name:<34}{m['mean_return']:>8.2%}{vs:>10.2%}{t:>8.2f}"
            f"{m['hit_rate']:>7.0%}{m['deployed_frac']:>9.0%}{d:>9}")


HDR = (f"{'layer':<34}{'mean':>8}{'vs bench':>10}{'t':>8}{'hit':>7}"
       f"{'invested':>9}{'delta':>9}")


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    with open(CACHE / "research.pkl", "rb") as fh:
        rp = pickle.load(fh)
    with open(CACHE / "panels.pkl", "rb") as fh:
        px = pickle.load(fh)
    panel, features, horizon = rp["panel"], rp["features"], rp["horizon"]
    print(f"config {cfg.version}; panel {len(panel):,} rows / "
          f"{panel['date'].nunique()} dates; horizon {horizon}")

    cache = CACHE / "rankings.pkl"
    if cache.is_file():
        with open(cache, "rb") as fh:
            weave, forward, work = pickle.load(fh)
        print(f"  rankings from cache: weave {len(weave)}, forward {len(forward)}")
    else:
        weave, forward, work = build_rankings(panel, features, cfg)
        with open(cache, "wb") as fh:
            pickle.dump((weave, forward, work), fh, protocol=4)

    cap, c6, c7 = cfg.params.capital, cfg.params.stage6_entry, cfg.params.stage7_risk
    model = CostModel(cfg)

    def cost_bps(price, qty, adtv):
        if price <= 0 or qty <= 0:
            return 0.0
        return float(model.round_trip(price, int(max(qty, 1)),
                                      adtv_inr=adtv if adtv > 0 else None
                                      ).total_bps_of_buy)

    shipped = Book(
        capital=fv(cap.total_capital_inr),
        max_positions=iv(cap.max_open_positions),
        risk_per_trade_pct=fv(cap.risk_per_trade_pct),
        max_participation_of_adtv=fv(cap.max_participation_of_adtv),
        stop_atr_multiple=fv(c7.stop_loss.atr_multiple),
        min_stop_distance_pct=fv(c7.stop_loss.min_stop_distance_pct),
        max_stop_distance_pct=fv(c7.stop_loss.max_stop_distance_pct),
        invalidation_ma_sessions=iv(c7.thesis_invalidation.structure_ma_sessions),
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        horizon_sessions=horizon,
        entry_rank=iv(c6.admission.entry_rank),
        exit_rank=iv(c6.admission.exit_rank),
        target_r_multiple=fv(c7.targets.t2_r_multiple),
        cost_fn=cost_bps,
    )
    print(f"  book: {shipped.max_positions} slots, entry {shipped.entry_rank}, "
          f"exit {shipped.exit_rank}, stop {shipped.stop_atr_multiple:g}xATR, "
          f"target {shipped.target_r_multiple:g}R, "
          f"risk Rs {shipped.risk_budget:,.0f} vs slot Rs {shipped.slot:,.0f}")
    print(f"  risk budget binds above a stop distance of "
          f"{shipped.risk_budget / shipped.slot:.1%}\n")

    results = {}
    for label, rankings in (("weave", weave), ("forward", forward)):
        print("=" * 96)
        print(f"{label.upper()}  --  {len(rankings)} out-of-sample dates")
        print("=" * 96)
        bench = benchmark(work, rankings, shipped, px)
        bm = float(bench.mean())
        print(f"{'0  equal-weight eligible universe':<34}{bm:>8.2%}"
              f"{0.0:>10.2%}{'--':>8}{'--':>7}{'100%':>9}\n")
        print(HDR)

        layers = [
            ("1  top 8, equal weight, no exits",
             dict(risk_sizing=False, use_stop=False, use_target=False,
                  use_invalidation=False, use_costs=False,
                  admissible_only=False)),
            ("1a + refuse the unbuyable names",
             dict(risk_sizing=False, use_stop=False, use_target=False,
                  use_invalidation=False, use_costs=False)),
            ("2  + risk-budget sizing",
             dict(use_stop=False, use_target=False, use_invalidation=False,
                  use_costs=False)),
            ("3  + 2.5x ATR stop",
             dict(use_target=False, use_invalidation=False, use_costs=False)),
            ("4  + 3R target",
             dict(use_invalidation=False, use_costs=False)),
            ("5  + invalidation exit",
             dict(use_costs=False)),
            ("6  + costs -- SHIPPED BOOK", dict()),
        ]
        prev, got = None, {}
        for name, kw in layers:
            m = phases(replace(shipped, **kw), rankings, px)
            got[name] = m
            print(row(name, m, bm, horizon, prev))
            prev = m["mean_return"]

        shipped_m = got["6  + costs -- SHIPPED BOOK"]
        print()
        print(f"  exit mix: stop {shipped_m['stop_share']:.0%}  "
              f"target {shipped_m['target_share']:.0%}  "
              f"invalidation {shipped_m['inval_share']:.0%}  "
              f"timeout {shipped_m['timeout_share']:.0%}  "
              f"median hold {shipped_m['median_held']:.0f} sessions")
        print(f"  drawdown: mean-of-phases {shipped_m['drawdown_mean_of_phases']:+.1%}  "
              f"WORST SCHEDULE {shipped_m['drawdown_worst_schedule']:+.1%}")
        print(f"  turnover: {shipped_m['avg_new']:.1f} new names per rebalance "
              f"of {shipped_m['avg_names']:.1f} held")
        print(f"  slots:    {shipped_m['avg_names']:.2f} filled of "
              f"{shipped_m['avg_book']:.2f} selected -- the gap is names the "
              f"ranking chose and the book refused")

        print(f"\n{'-- corrections, one at a time, against layer 6 --':<34}")
        print(HDR)
        base = shipped_m["mean_return"]
        fixes = [
            ("A  target read on the HIGH", dict(target_on_high=True)),
            ("B  re-entry pays its round trip", dict(charge_reentry=True)),
            ("C  book held fully invested", dict(full_investment=True)),
            ("D  unknown liquidity refused", dict(refuse_unknown_liquidity=True)),
        ]
        for name, kw in fixes:
            m = phases(replace(shipped, **kw), rankings, px)
            got[name] = m
            print(row(name, m, bm, horizon, base))
        allfix = phases(replace(shipped, target_on_high=True, charge_reentry=True,
                                refuse_unknown_liquidity=True), rankings, px)
        got["E  A+B+D together"] = allfix
        print(row("E  A+B+D together", allfix, bm, horizon, base))
        allfix2 = phases(replace(shipped, target_on_high=True, charge_reentry=True,
                                 refuse_unknown_liquidity=True,
                                 full_investment=True), rankings, px)
        got["F  A+B+C+D together"] = allfix2
        print(row("F  A+B+C+D together", allfix2, bm, horizon, base))

        results[label] = {"bench": bm, "n_dates": len(rankings),
                          "layers": {k: {kk: vv for kk, vv in v.items()
                                         if kk != "returns"}
                                     for k, v in got.items()}}
        print()

    with open(CACHE / "decomposition.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
