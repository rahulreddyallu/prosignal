"""H4c, on ONE benchmark.

The first run scored each arm against the equal-weight mean of ITS OWN label,
so the R-multiple arm was compared with the universe's mean R-multiple and read
-20.87%. That is not a benchmark, it is a unit error, and it is exactly the
class of defect this whole review is about -- a number that looks like a
comparison and is not one.

Every arm here is scored against the SAME thing: the equal-weight forward
return of the eligible universe over the same holding windows, in rupees. That
is the alternative a person actually has, and it does not change when the
model's training label changes.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.features.crosssec import cross_sectional_rank
from prosignal.stages._cfg import iv
from prosignal.validation.significance import newey_west_t

from book import phases
from decompose import build_rankings
from frontier import shipped_book

CACHE = Path(__file__).resolve().parent / "cache"
STEP = 21


def rupee_benchmark(horizon_panel, rankings, b, px):
    """Equal-weight eligible universe, in rupees, on the dates the book traded.

    Read from the horizon-return panel for every arm, whatever label the arm
    was fitted on.
    """
    close = px["close"]
    pos = {d: i for i, d in enumerate(close.index)}
    lab = horizon_panel.groupby("date")["label"].mean()
    lab.index = pd.to_datetime(lab.index)
    stride = max(int(np.ceil(b.horizon_sessions / STEP)), 1)
    got = []
    for p in range(stride):
        for j in range(p, len(rankings), stride):
            d = rankings[j][0]
            if d not in pos or pos[d] + b.horizon_sessions >= len(close.index):
                continue
            if d in lab.index:
                got.append(float(lab.loc[d]))
    return float(np.mean(got)) if got else float("nan")


def main() -> int:
    cfg = load_config()
    horizon = iv(cfg.params.stage4_core_score.model_horizon_sessions)
    with open(CACHE / "panels.pkl", "rb") as fh:
        px = pickle.load(fh)
    with open(CACHE / "engine_panel.pkl", "rb") as fh:
        ep, features, rules = pickle.load(fh)
    with open(CACHE / "research.pkl", "rb") as fh:
        hp = pickle.load(fh)["panel"]

    b0 = shipped_book(cfg, horizon)
    variants = {
        "horizon return (SHIPPED)": (hp, "label"),
        "engine geometry, rupee": (ep, "label"),
        "engine geometry, R-multiple": (ep, "label_r"),
    }
    prepared = {}
    for k, (p, col) in variants.items():
        q = p.copy()
        q["label"] = q[col]
        q["label_rank"] = q.groupby("date")["label"].transform(cross_sectional_rank)
        prepared[k] = q[np.isfinite(q["label"])]

    out = {}
    hdr = (f"  {'label the model was fitted on':<32}{'dates':>6}{'net':>9}"
           f"{'bench':>8}{'vs bench':>10}{'t':>7}{'sharpe':>8}{'worst dd':>10}")
    for construction in (0, 1):
        name = ("weave", "forward")[construction]
        print(f"\n--- {name} ---")
        print(hdr)
        for k, q in prepared.items():
            rk = build_rankings(q, features, cfg)
            rankings = rk[construction]
            if len(rankings) < 6:
                print(f"  {k:<32}{len(rankings):>6}   too few dates")
                continue
            bm = rupee_benchmark(hp, rankings, b0, px)
            m = phases(b0, rankings, px)
            try:
                t = newey_west_t(m["returns"] - bm, horizon_sessions=horizon,
                                 step_sessions=STEP).adjusted_t
            except ValueError:
                t = float("nan")
            print(f"  {k:<32}{len(rankings):>6}{m['mean_return']:>9.2%}"
                  f"{bm:>8.2%}{m['mean_return'] - bm:>10.2%}{t:>7.2f}"
                  f"{m['sharpe']:>8.2f}{m['drawdown_worst_schedule']:>10.1%}")
            # The same arm with the exits switched OFF, since the whole point of
            # matching the label to the geometry is to justify the geometry.
            mno = phases(b0.__class__(**{**b0.__dict__, "use_stop": False,
                                         "use_target": False,
                                         "use_invalidation": False}),
                         rankings, px)
            print(f"  {'   ^ same ranking, no exits':<32}{'':>6}"
                  f"{mno['mean_return']:>9.2%}{bm:>8.2%}"
                  f"{mno['mean_return'] - bm:>10.2%}{'':>7}"
                  f"{mno['sharpe']:>8.2f}"
                  f"{mno['drawdown_worst_schedule']:>10.1%}")
            out.setdefault(name, {})[k] = {
                "n_dates": len(rankings), "bench": bm,
                "mean": m["mean_return"], "vs_bench": m["mean_return"] - bm,
                "t": t, "sharpe": m["sharpe"],
                "dd": m["drawdown_worst_schedule"],
                "no_exits_mean": mno["mean_return"],
                "no_exits_vs_bench": mno["mean_return"] - bm,
                "no_exits_sharpe": mno["sharpe"]}

    with open(CACHE / "h4c.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
