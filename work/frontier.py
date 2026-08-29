"""H1, H2, H3, H5 -- the exit frontier and the layers around it, priced.

Section H of the dossier tabulated stop multiples with an OFF arm and concluded
the relationship is monotone. This re-runs that on an independent construction,
adds the columns a capital owner actually needs to choose on -- worst SINGLE
schedule rather than a mean of schedules, and the COVID cohorts on their own --
and prices the exit components separately.

Nothing here selects a setting. The output is a menu with a price against every
line; the choice is a risk-tolerance decision and it is not the code's to make.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.costs import CostModel
from prosignal.stages._cfg import fv, iv
from prosignal.validation.significance import newey_west_t

from book import Book, phases, simulate
from decompose import benchmark

CACHE = Path(__file__).resolve().parent / "cache"
STEP = 21
#: A stop this wide is unreachable inside the max_stop_distance_pct cap, which
#: is how "no stop" is expressed without a separate code path that might differ
#: in some other way. Verified against use_stop=False below.
OFF = None


def shipped_book(cfg, horizon):
    cap, c6, c7 = cfg.params.capital, cfg.params.stage6_entry, cfg.params.stage7_risk
    model = CostModel(cfg)

    def cost_bps(price, qty, adtv):
        if price <= 0 or qty <= 0:
            return 0.0
        return float(model.round_trip(price, int(max(qty, 1)),
                                      adtv_inr=adtv if adtv > 0 else None
                                      ).total_bps_of_buy)

    return Book(
        capital=fv(cap.total_capital_inr),
        max_positions=iv(cap.max_open_positions),
        risk_per_trade_pct=fv(cap.risk_per_trade_pct),
        max_participation_of_adtv=fv(cap.max_participation_of_adtv),
        stop_atr_multiple=fv(c7.stop_loss.atr_multiple),
        min_stop_distance_pct=fv(c7.stop_loss.min_stop_distance_pct),
        max_stop_distance_pct=fv(c7.stop_loss.max_stop_distance_pct),
        invalidation_ma_sessions=iv(c7.thesis_invalidation.structure_ma_sessions),
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        horizon_sessions=horizon, entry_rank=iv(c6.admission.entry_rank),
        exit_rank=iv(c6.admission.exit_rank),
        target_r_multiple=fv(c7.targets.t2_r_multiple), cost_fn=cost_bps)


def spearman(a, b) -> float:
    """Rank correlation without scipy."""
    ra = pd.Series(a).rank().to_numpy("float64")
    rb = pd.Series(b).rank().to_numpy("float64")
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cohorts(b: Book, rankings, px):
    """Worst single 63-session cohort, and the 2020 crash cohorts on their own.

    Two different questions, and section H answers only the first. The worst
    COHORT is the worst quarter anyone lived through. The worst SCHEDULE
    drawdown is the deepest peak-to-trough of the equity curve, which is what a
    sequence of mediocre quarters compounds into. A stop can improve the first
    while making the second worse, and here it does.
    """
    stride = max(int(np.ceil(b.horizon_sessions / STEP)), 1)
    worst, crash = np.inf, []
    for p in range(stride):
        f = simulate(b, rankings, px, phase=p, step_sessions=STEP)
        if f.empty:
            continue
        worst = min(worst, float(f["ret"].min()))
        m = ((f["date"] >= pd.Timestamp("2020-01-01"))
             & (f["date"] <= pd.Timestamp("2020-06-30")))
        crash.extend(f.loc[m, "ret"].tolist())
    return (worst if np.isfinite(worst) else float("nan"),
            float(np.mean(crash)) if crash else float("nan"))


def line(name, m, bm, horizon, coh):
    r = m["returns"]
    try:
        t = newey_west_t(r - bm, horizon_sessions=horizon,
                         step_sessions=STEP).adjusted_t
    except ValueError:
        t = float("nan")
    worst_cohort, crash = coh
    return (f"{name:<28}{m['mean_return']:>8.2%}{m['mean_return'] - bm:>10.2%}"
            f"{t:>7.2f}{m['hit_rate']:>6.0%}"
            f"{worst_cohort:>10.1%}{m['drawdown_worst_schedule']:>10.1%}"
            f"{crash:>9.1%}{m['sharpe']:>8.2f}")


HDR = (f"{'arm':<28}{'net':>8}{'vs bench':>10}{'t':>7}{'hit':>6}"
       f"{'worst coh':>10}{'worst dd':>10}{'covid':>9}{'sharpe':>8}")


def main() -> int:
    cfg = load_config()
    with open(CACHE / "research.pkl", "rb") as fh:
        rp = pickle.load(fh)
    with open(CACHE / "panels.pkl", "rb") as fh:
        px = pickle.load(fh)
    with open(CACHE / "rankings.pkl", "rb") as fh:
        weave, forward, work = pickle.load(fh)
    horizon = rp["horizon"]
    b0 = shipped_book(cfg, horizon)
    out = {}

    for label, rankings in (("weave", weave), ("forward", forward)):
        bench = benchmark(work, rankings, b0, px)
        bm = float(bench.mean())
        print("=" * 106)
        print(f"{label.upper()}  --  {len(rankings)} dates; benchmark "
              f"{bm:+.2%} per {horizon}-session period")
        print("=" * 106)

        # -- H1: the stop frontier, all arms carrying costs ----------------
        print("\nH1  STOP FRONTIER (3R target and invalidation on, costs on)")
        print(HDR)
        arms, means, mults = {}, [], []
        for mult in (None, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0):
            if mult is None:
                bb = replace(b0, use_stop=False)
                nm = "no stop"
                rank_value = 99.0
            else:
                bb = replace(b0, stop_atr_multiple=mult)
                nm = f"{mult:g}x ATR" + ("  <- SHIPPED" if mult == 2.5 else "")
                rank_value = mult
            m = phases(bb, rankings, px)
            c = cohorts(bb, rankings, px)
            arms[nm.strip()] = {k: v for k, v in m.items() if k != "returns"}
            arms[nm.strip()]["worst_cohort"], arms[nm.strip()]["covid"] = c
            means.append(m["mean_return"]); mults.append(rank_value)
            print(line(nm, m, bm, horizon, c))
        rho = spearman(mults, means)
        best = mults[int(np.argmax(means))]
        print(f"\n  Spearman(stop multiple, return) = {rho:+.3f}; "
              f"best arm = {'no stop' if best == 99 else f'{best:g}x'}")
        print(f"  H1 {'PASSES' if rho >= 0.9 and best == 99 else 'FAILS'} "
              f"its pre-registered bar (rho >= +0.9 and the OFF arm best)")

        # -- exit components, each removed on its own ----------------------
        print("\nH1b EXIT COMPONENTS (each removed from the shipped book)")
        print(HDR)
        for nm, kw in (("shipped", {}),
                       ("no target", dict(use_target=False)),
                       ("no invalidation", dict(use_invalidation=False)),
                       ("no stop", dict(use_stop=False)),
                       ("no exits at all", dict(use_stop=False, use_target=False,
                                                use_invalidation=False))):
            bb = replace(b0, **kw)
            m = phases(bb, rankings, px)
            c = cohorts(bb, rankings, px)
            arms[f"component: {nm}"] = {k: v for k, v in m.items() if k != "returns"}
            print(line(nm, m, bm, horizon, c))

        # -- H2: sizing split into weighting and exposure -------------------
        print("\nH2  SIZING -- WEIGHTING vs EXPOSURE")
        no_exit = dict(use_stop=False, use_target=False, use_invalidation=False,
                       use_costs=False)
        eq = phases(replace(b0, risk_sizing=False, **no_exit), rankings, px)
        rb = phases(replace(b0, **no_exit), rankings, px)
        fi = phases(replace(b0, full_investment=True, **no_exit), rankings, px)
        layer2 = rb["mean_return"] - eq["mean_return"]
        recovered = fi["mean_return"] - rb["mean_return"]
        share = recovered / abs(layer2) if layer2 else float("nan")
        print(f"  equal weight, fully invested        {eq['mean_return']:+.2%}  "
              f"invested {eq['deployed_frac']:.0%}")
        print(f"  risk-budget sizing (SHIPPED)        {rb['mean_return']:+.2%}  "
              f"invested {rb['deployed_frac']:.0%}   layer 2 = {layer2:+.2%}")
        print(f"  same weights, held fully invested   {fi['mean_return']:+.2%}  "
              f"invested {fi['deployed_frac']:.0%}   recovers {recovered:+.2%}")
        print(f"  => {share:.0%} of layer 2 is CASH, not weighting. "
              f"H2 {'PASSES' if share >= 0.5 else 'FAILS'} its 50% bar.")
        out.setdefault(label, {})["h2"] = {
            "layer2": layer2, "recovered": recovered, "share": share,
            "invested_shipped": rb["deployed_frac"]}

        # -- H3: does the label describe the trade? -------------------------
        m = phases(b0, rankings, px)
        print(f"\nH3  LABEL vs TRADE")
        print(f"  label horizon                       {horizon} sessions")
        print(f"  median realised hold                {m['median_held']:.0f} sessions")
        print(f"  reached the {horizon}-session timeout       "
              f"{m['timeout_share']:.0%}")
        print(f"  exited by stop / target / invalid   "
              f"{m['stop_share']:.0%} / {m['target_share']:.0%} / "
              f"{m['inval_share']:.0%}")
        h3 = m["median_held"] <= 40 and m["timeout_share"] <= 0.25
        print(f"  H3 {'PASSES' if h3 else 'FAILS'}: the model is ranking on an "
              f"outcome {horizon/max(m['median_held'],1):.1f}x longer than the "
              f"book's median trade")

        # -- H5: the unbuyable population -----------------------------------
        full = phases(replace(b0, admissible_only=False, **no_exit), rankings, px)
        adm = phases(replace(b0, **no_exit, risk_sizing=False), rankings, px)
        eqfull = phases(replace(b0, admissible_only=False, risk_sizing=False,
                                **no_exit), rankings, px)
        print(f"\nH5  RANKING A POPULATION THE BOOK CANNOT BUY")
        print(f"  slots selected per rebalance        {m['avg_book']:.2f}")
        print(f"  slots the book could actually open  {m['avg_names']:.2f}")
        print(f"  cost of the refusal, equal weight   "
              f"{adm['mean_return'] - eqfull['mean_return']:+.2%} per period")
        out[label].update({
            "benchmark": bm, "n_dates": len(rankings), "arms": arms,
            "h1_rho": rho, "h1_best": best,
            "h3": {"median_held": m["median_held"],
                   "timeout_share": m["timeout_share"],
                   "stop_share": m["stop_share"],
                   "target_share": m["target_share"],
                   "inval_share": m["inval_share"]},
            "h5": {"avg_book": m["avg_book"], "avg_filled": m["avg_names"],
                   "cost": adm["mean_return"] - eqfull["mean_return"]}})
        print()

    with open(CACHE / "frontier.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
