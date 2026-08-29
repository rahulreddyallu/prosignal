"""Price every v2 candidate on return AND on risk, before proposing any of it.

A change that raises return and raises drawdown is a risk-tolerance decision and
belongs to whoever holds the capital. A change that raises return and LOWERS
drawdown on both constructions is not a preference, it is a correction. The two
have to be told apart, and the only way to do that is to print both columns for
every arm -- which the shipped decomposition does not.
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

from prosignal.config.loader import load_config
from prosignal.stages._cfg import iv
from prosignal.validation.significance import newey_west_t

from book import phases
from decompose import benchmark
from frontier import cohorts, shipped_book

CACHE = Path(__file__).resolve().parent / "cache"
STEP = 21

HDR = (f"{'arm':<40}{'net':>8}{'vs bench':>10}{'t':>7}{'sharpe':>8}"
       f"{'worst coh':>11}{'worst dd':>10}{'invested':>10}")


def line(name, m, bm, horizon, coh):
    try:
        t = newey_west_t(m["returns"] - bm, horizon_sessions=horizon,
                         step_sessions=STEP).adjusted_t
    except ValueError:
        t = float("nan")
    return (f"{name:<40}{m['mean_return']:>8.2%}{m['mean_return'] - bm:>10.2%}"
            f"{t:>7.2f}{m['sharpe']:>8.2f}{coh[0]:>11.1%}"
            f"{m['drawdown_worst_schedule']:>10.1%}{m['deployed_frac']:>10.0%}")


CANDIDATES = [
    ("shipped (config v1)", {}),
    ("v2a  book held fully invested", dict(full_investment=True)),
    ("v2b  stop 3.5x ATR", dict(stop_atr_multiple=3.5)),
    ("v2c  stop 5.0x ATR", dict(stop_atr_multiple=5.0)),
    ("v2d  no stop, invalidation kept", dict(use_stop=False)),
    ("v2e  no stop, no target, no invalidation",
     dict(use_stop=False, use_target=False, use_invalidation=False)),
    ("v2a+v2b  invested + 3.5x",
     dict(full_investment=True, stop_atr_multiple=3.5)),
    ("v2a+v2d  invested + no stop",
     dict(full_investment=True, use_stop=False)),
    ("v2a+v2e  invested + no exits",
     dict(full_investment=True, use_stop=False, use_target=False,
          use_invalidation=False)),
]


def main() -> int:
    cfg = load_config()
    horizon = iv(cfg.params.stage4_core_score.model_horizon_sessions)
    with open(CACHE / "panels.pkl", "rb") as fh:
        px = pickle.load(fh)
    with open(CACHE / "rankings.pkl", "rb") as fh:
        weave, forward, work = pickle.load(fh)
    b0 = shipped_book(cfg, horizon)
    out = {}

    for label, rankings in (("weave", weave), ("forward", forward)):
        bm = float(benchmark(work, rankings, b0, px).mean())
        print("=" * 104)
        print(f"{label.upper()}  --  {len(rankings)} dates; benchmark {bm:+.2%}")
        print("=" * 104)
        print(HDR)
        base = None
        for name, kw in CANDIDATES:
            m = phases(replace(b0, **kw), rankings, px)
            coh = cohorts(replace(b0, **kw), rankings, px)
            print(line(name, m, bm, horizon, coh))
            row = {k: v for k, v in m.items() if k != "returns"}
            row["worst_cohort"], row["covid"] = coh
            row["vs_bench"] = m["mean_return"] - bm
            out.setdefault(label, {})[name] = row
            if base is None:
                base = row
        print()
        # Dominance: better on return AND on both risk measures.
        print("  arms that DOMINATE the shipped book -- higher return, shallower")
        print("  worst cohort AND shallower worst schedule drawdown:")
        any_dom = False
        for name, row in out[label].items():
            if name.startswith("shipped"):
                continue
            if (row["mean_return"] > base["mean_return"]
                    and row["worst_cohort"] > base["worst_cohort"]
                    and row["drawdown_worst_schedule"]
                    > base["drawdown_worst_schedule"]):
                print(f"    {name}")
                any_dom = True
        if not any_dom:
            print("    none -- every improvement here costs something in risk")
        print()

    # Cross-construction: dominance has to hold on BOTH to be worth stating.
    both = []
    for name in out["weave"]:
        if name.startswith("shipped"):
            continue
        ok = True
        for label in ("weave", "forward"):
            b = out[label]["shipped (config v1)"]
            r = out[label][name]
            ok = ok and (r["mean_return"] > b["mean_return"]
                         and r["worst_cohort"] > b["worst_cohort"]
                         and r["drawdown_worst_schedule"] > b["drawdown_worst_schedule"])
        if ok:
            both.append(name)
    print("DOMINATES THE SHIPPED BOOK ON BOTH CONSTRUCTIONS:")
    for n in both or ["  none"]:
        print(f"  {n}")

    with open(CACHE / "v2.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
