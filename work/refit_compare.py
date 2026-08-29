"""What the R9 correction does to the coefficients the engine trades.

The audit's instruction was explicit: enable it, then refit, and do not
preserve the current coefficients merely because the corrected panel moves
them. This measures the move rather than asserting it -- one fit on the wide
panel, one on the admissible panel, same estimator, same config, same dates.

It also runs W2's selection correction over BOTH fits. A theme that clears the
gate on the wide panel and misses it on the admissible one was selected by
names the book cannot buy.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.features.crossmodel import fit_coefficients
from prosignal.stages._cfg import fv, iv
from prosignal.validation.selection import correct_t

CACHE = Path(__file__).resolve().parent / "cache"
STEP = 21


def fit(panel, features, cfg):
    c4 = cfg.params.stage4_core_score
    cols = [c for c in features if c in panel.columns]
    work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
    _fit, fm, why = fit_coefficients(
        work, cols, alpha=fv(c4.model_ridge_alpha),
        estimator=str(c4.estimator.method),
        horizon=iv(c4.model_horizon_sessions), step=STEP,
        significance_floor=fv(c4.estimator.significance_floor),
        shrink_toward=str(c4.estimator.shrink_toward))
    return _fit, fm, why, work


def main() -> int:
    cfg = load_config()
    floor = fv(cfg.params.stage4_core_score.estimator.significance_floor)
    print(f"config {cfg.version}; significance floor {floor}")

    out = {}
    for tag, path in (("v1 wide", CACHE.parent / "cache_v1" / "research.pkl"),
                      ("v2 admissible", CACHE / "research.pkl")):
        if not path.is_file():
            print(f"  {tag}: no panel at {path}")
            continue
        blob = pickle.load(open(path, "rb"))
        f, fm, why, work = fit(blob["panel"], blob["features"], cfg)
        if fm is None:
            print(f"  {tag}: no fit -- {why}")
            continue
        out[tag] = (fm, len(work), work["date"].nunique())
        print(f"  {tag}: {len(work):,} rows / {work['date'].nunique()} dates, "
              f"{fm.n_dates} cross-sections")

    if len(out) < 2:
        return 1

    (a, na, da), (b, nb, db) = out["v1 wide"], out["v2 admissible"]
    themes = sorted(set(a.lam) | set(b.lam))

    print(f"\n{'='*104}")
    print("FAMA-MACBETH COEFFICIENTS  --  wide panel vs the population the book can buy")
    print(f"{'='*104}")
    print(f"{'theme':<16}{'v1 lambda':>11}{'v2 lambda':>11}{'delta':>9}"
          f"{'v1 t':>8}{'v2 t':>8}{'v1 W2':>8}{'v2 W2':>8}   verdict")
    for th in themes:
        la, lb = a.lam.get(th, float('nan')), b.lam.get(th, float('nan'))
        ta, tb = a.t_stat.get(th, float('nan')), b.t_stat.get(th, float('nan'))
        ca = correct_t(ta, floor) if abs(ta) >= floor else float('nan')
        cb = correct_t(tb, floor) if abs(tb) >= floor else float('nan')
        pa, pb = abs(ta) >= floor, abs(tb) >= floor
        if pa and pb:
            verdict = "traded in both"
        elif pa and not pb:
            verdict = "DROPS OUT -- was selected by names the book cannot buy"
        elif pb and not pa:
            verdict = "ENTERS -- only visible on the tradable population"
        else:
            verdict = "zeroed in both"
        f = lambda x: "     --" if not np.isfinite(x) else f"{x:+.4f}"
        g = lambda x: "    --" if not np.isfinite(x) else f"{x:+.2f}"
        print(f"{th:<16}{f(la):>11}{f(lb):>11}{f(lb-la):>9}"
              f"{g(ta):>8}{g(tb):>8}{g(ca):>8}{g(cb):>8}   {verdict}")

    live_a = [t for t in themes if abs(a.t_stat.get(t, 0)) >= floor]
    live_b = [t for t in themes if abs(b.t_stat.get(t, 0)) >= floor]
    print(f"\n  traded on the wide panel        : {', '.join(live_a) or 'none'}")
    print(f"  traded on the admissible panel  : {', '.join(live_b) or 'none'}")
    moved = [t for t in themes
             if np.isfinite(a.lam.get(t, np.nan)) and np.isfinite(b.lam.get(t, np.nan))
             and abs(b.lam[t] - a.lam[t]) > 1e-9]
    print(f"  coefficients that moved at all  : {len(moved)} of {len(themes)}")
    print(f"\n  Every number below the section D table was computed on the wide")
    print(f"  panel. The coefficients above are why that measurement is retired")
    print(f"  rather than adjusted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
