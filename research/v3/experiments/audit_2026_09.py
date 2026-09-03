"""Quant-audit experiments (2026-09), run against the SHIPPED scorer.

These resolve the factor-level findings of the 2026-09 adversarial review WITHOUT
touching the frozen model: every number here comes from `features.v3` /
`features.v3_factors` applied to the local curated store through
`validation.v3_panel.build_v3_panel`, the same panel the quarterly re-check uses.

    EXP-A  sign stability (K-4)      -- does each theme (esp. "quality") hold its
                                        IC sign across sub-periods, or is a sign a
                                        one-window artifact? The repo's own screen
                                        kills a factor whose sign flips between
                                        halves; this applies that test to the
                                        shipped themes on all available data.
    EXP-B  delivery incremental (K-3)-- does the ownership/delivery theme add
                                        rank information AFTER the other themes,
                                        and after controlling for liquidity/vol?
    EXP-C  blend vs single (K-2 lite)-- does the 5-theme composite beat the best
                                        single theme on the ranking statistics?
                                        (The book-level, net-of-cost K-2 needs the
                                        portfolio simulator; this is the IC-level
                                        precursor.)

WHAT THIS CANNOT DO. It measures the RANKING, not the traded book, and the
ranking's signs/weights were fit on 2018-2024, so full-window IC is partly
in-sample. Sign STABILITY across sub-periods is meaningful regardless; IC LEVELS
on 2018-2024 are not out-of-sample and are labelled as such. The clean book test
(K-1) and cost break-even (K-6) live in their own scripts.

Usage:
    python research/v3/experiments/audit_2026_09.py            # build + run all
    python research/v3/experiments/audit_2026_09.py --rebuild  # ignore panel cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.config.loader import load_config          # noqa: E402
from prosignal.data.store import DataStore                # noqa: E402
from prosignal.features import v3                          # noqa: E402
from prosignal.validation.v3_panel import build_v3_panel   # noqa: E402
from prosignal.validation.v2_panel import rank_ic, quintile_spread  # noqa: E402
from prosignal import v3_monitor as vm                     # noqa: E402

OUT = Path(__file__).resolve().parent
PANEL_CACHE = OUT / "panel_2026_09.parquet"
LABEL = "y21"          # the horizon the composite blend was judged on
THEMES = list(v3.THEMES)


# --------------------------------------------------------------------------- IC
def _agg_ic(ic_series: np.ndarray) -> tuple:
    """mean IC and its t across dates (per-date, never pooled)."""
    x = np.asarray([v for v in ic_series if np.isfinite(v)], dtype="float64")
    if len(x) < 5:
        return float("nan"), float("nan"), len(x)
    t = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std(ddof=1) > 0 else float("nan")
    return float(x.mean()), t, len(x)


def _theme_ic_table(panel: pd.DataFrame, label: str = LABEL) -> dict:
    tic = vm.rolling_theme_ic(panel, label)
    out = {}
    for t in THEMES:
        if t in tic.columns:
            m, tt, n = _agg_ic(tic[t].to_numpy("float64"))
            out[t] = {"ic": m, "t": tt, "dates": n}
    return out


# ---------------------------------------------------------------- EXP-A: signs
def exp_a_sign_stability(panel: pd.DataFrame) -> dict:
    """Per-theme and per-quality-factor IC over the whole window, halves, thirds.

    A theme whose IC changes sign across sub-periods fails the repo's own
    sign-stability screen and cannot be trusted at its shipped sign.
    """
    dates = np.array(sorted(pd.to_datetime(panel["date"]).unique()))
    def _sub(a, b):
        m = (pd.to_datetime(panel["date"]) >= a) & (pd.to_datetime(panel["date"]) < b)
        return panel[m]

    halves, thirds = [], []
    h = dates[len(dates) // 2]
    halves = [(dates[0], h), (h, dates[-1] + np.timedelta64(1, "D"))]
    q1, q2 = dates[len(dates) // 3], dates[2 * len(dates) // 3]
    thirds = [(dates[0], q1), (q1, q2), (q2, dates[-1] + np.timedelta64(1, "D"))]

    res = {"whole": _theme_ic_table(panel),
           "halves": [], "thirds": [], "quality_factors": {}}
    for a, b in halves:
        res["halves"].append({"start": str(pd.Timestamp(a).date()),
                              "end": str(pd.Timestamp(b).date()),
                              "themes": _theme_ic_table(_sub(a, b))})
    for a, b in thirds:
        res["thirds"].append({"start": str(pd.Timestamp(a).date()),
                             "end": str(pd.Timestamp(b).date()),
                             "themes": _theme_ic_table(_sub(a, b))})

    # The two quality factors, at their SHIPPED sign, across halves.
    fic = vm.rolling_factor_ic(panel, LABEL)
    for f in ("net_margin", "margin_stability"):
        if f in fic.columns:
            whole = _agg_ic(fic[f].to_numpy("float64"))
            hh = []
            for a, b in halves:
                m = (pd.to_datetime(fic["date"]) >= a) & (pd.to_datetime(fic["date"]) < b)
                hh.append(_agg_ic(fic[m][f].to_numpy("float64")))
            res["quality_factors"][f] = {
                "whole": {"ic": whole[0], "t": whole[1], "dates": whole[2]},
                "half1": {"ic": hh[0][0], "t": hh[0][1], "dates": hh[0][2]},
                "half2": {"ic": hh[1][0], "t": hh[1][1], "dates": hh[1][2]},
                "sign_flips": bool(np.isfinite(hh[0][0]) and np.isfinite(hh[1][0])
                                   and np.sign(hh[0][0]) != np.sign(hh[1][0])),
            }

    # Sign-flip verdicts per theme, across thirds.
    flips = {}
    for t in THEMES:
        signs = [np.sign(third["themes"].get(t, {}).get("ic", np.nan))
                 for third in res["thirds"]]
        signs = [s for s in signs if np.isfinite(s)]
        flips[t] = bool(len(set(signs)) > 1)
    res["theme_sign_flips_across_thirds"] = flips
    return res


# -------------------------------------------------------- EXP-B: delivery value
def _fm_ols(panel: pd.DataFrame, ycol: str, xcols: list) -> dict:
    """Fama-MacBeth: per-date cross-sectional OLS of rank(y) on xcols, then the
    slope series is the sample. Returns each coef's mean and t across dates."""
    slopes = {c: [] for c in xcols}
    for _, g in panel.groupby("date", sort=True):
        sub = g[[ycol] + xcols].to_numpy("float64")
        m = np.isfinite(sub).all(axis=1)
        if m.sum() < 60:
            continue
        y = pd.Series(g[ycol].to_numpy("float64")[m]).rank().to_numpy()
        X = g[xcols].to_numpy("float64")[m]
        X = np.column_stack([np.ones(len(y)), X])
        try:
            beta, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
        except Exception:
            continue
        for i, c in enumerate(xcols):
            slopes[c].append(beta[i + 1])
    out = {}
    for c in xcols:
        s = np.asarray(slopes[c], dtype="float64")
        if len(s) >= 5 and s.std(ddof=1) > 0:
            out[c] = {"coef": float(s.mean()),
                      "t": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))),
                      "dates": len(s)}
        else:
            out[c] = {"coef": float("nan"), "t": float("nan"), "dates": len(s)}
    return out


def exp_b_delivery(panel: pd.DataFrame) -> dict:
    """Does delivery add rank info AFTER the other themes, and after liquidity?"""
    subs = [t + "_sub" for t in THEMES if t + "_sub" in panel.columns]
    res = {"multivariate_fm": _fm_ols(panel, LABEL, subs)}

    # Delivery residualised on liquidity (adtv rank) and downside-vol rank, then
    # IC of the residual vs forward return. If delivery is "just liquidity/vol",
    # the residual IC collapses toward zero.
    p = panel.copy()
    p["adtv_rank_x"] = p.groupby("date")["adtv"].rank(pct=True)
    dv_ic_raw, dv_ic_resid = [], []
    for _, g in p.groupby("date", sort=True):
        if "ownership_sub" not in g or LABEL not in g:
            continue
        y = g[LABEL].to_numpy("float64")
        dv = g["ownership_sub"].to_numpy("float64")
        ctrl_cols = [c for c in ("adtv_rank_x", "downside_vol_60_r") if c in g]
        m = np.isfinite(y) & np.isfinite(dv)
        for c in ctrl_cols:
            m &= np.isfinite(g[c].to_numpy("float64"))
        if m.sum() < 60:
            continue
        yy = pd.Series(y[m]).rank().to_numpy()
        dvv = dv[m]
        dv_ic_raw.append(np.corrcoef(pd.Series(dvv).rank(), yy)[0, 1])
        if ctrl_cols:
            C = np.column_stack([np.ones(m.sum())] +
                                [g[c].to_numpy("float64")[m] for c in ctrl_cols])
            beta, *_ = np.linalg.lstsq(C, dvv - dvv.mean(), rcond=None)
            resid = (dvv - dvv.mean()) - C @ beta
            dv_ic_resid.append(np.corrcoef(pd.Series(resid).rank(), yy)[0, 1])
    res["delivery_ic_raw"] = dict(zip(("ic", "t", "dates"), _agg_ic(np.array(dv_ic_raw))))
    res["delivery_ic_resid_on_liquidity_vol"] = dict(
        zip(("ic", "t", "dates"), _agg_ic(np.array(dv_ic_resid))))
    return res


# ------------------------------------------------- EXP-C: blend vs single theme
def exp_c_blend_vs_single(panel: pd.DataFrame) -> dict:
    """Composite vs each theme sub-score alone, on IC and quintile spread."""
    res = {"composite": {}, "themes": {}}
    ic, ic_t, n = rank_ic(panel, LABEL, "score")
    sp, sp_t, _ = quintile_spread(panel, LABEL, score="score")
    res["composite"] = {"ic": ic, "ic_t": ic_t, "spread": sp, "spread_t": sp_t, "dates": n}
    for t in THEMES:
        col = t + "_sub"
        if col not in panel.columns:
            continue
        ic, ic_t, n = rank_ic(panel, LABEL, col)
        sp, sp_t, _ = quintile_spread(panel, LABEL, score=col)
        res["themes"][t] = {"ic": ic, "ic_t": ic_t, "spread": sp,
                            "spread_t": sp_t, "dates": n}
    best = max((v["spread"] for v in res["themes"].values()
               if np.isfinite(v.get("spread", np.nan))), default=float("nan"))
    res["blend_beats_best_single_on_spread"] = bool(
        np.isfinite(best) and res["composite"]["spread"] > best)
    res["best_single_spread"] = best
    return res


# ------------------------------------------- EXP-D: equal-weight vs frozen wts
def exp_d_equal_weight(panel: pd.DataFrame) -> dict:
    """Re-blend the SAME theme sub-scores with equal weights and compare to the
    frozen performance-proportional composite.

    Finding #4: the shipped weights are proportional to in-sample top-decile
    excess -- the scheme `weighting_mode` warns overfits. If an equal-weight
    blend of the identical sub-scores ranks as well or better, the fitted
    weights bought nothing out of sample. This changes NO shipped code: it
    recomputes the composite from columns already in the panel.
    """
    sub_cols = [t + "_sub" for t in THEMES if t + "_sub" in panel.columns]
    p = panel.copy()
    # equal weight over the themes a name actually has (mirrors score_frame's
    # renormalisation, just with equal instead of frozen weights).
    M = p[sub_cols].to_numpy("float64")
    ok = np.isfinite(M)
    n = ok.sum(axis=1)
    eq = np.where(n >= v3.MIN_THEMES,
                  np.nansum(np.where(ok, M, 0.0), axis=1) / np.maximum(n, 1), np.nan)
    p["score_eq"] = eq
    ic_f, ic_ft, _ = rank_ic(panel, LABEL, "score")
    sp_f, sp_ft, _ = quintile_spread(panel, LABEL, score="score")
    ic_e, ic_et, _ = rank_ic(p, LABEL, "score_eq")
    sp_e, sp_et, _ = quintile_spread(p, LABEL, score="score_eq")
    return {
        "frozen_weight": {"ic": ic_f, "ic_t": ic_ft, "spread": sp_f, "spread_t": sp_ft},
        "equal_weight": {"ic": ic_e, "ic_t": ic_et, "spread": sp_e, "spread_t": sp_et},
        "equal_weight_at_least_as_good": bool(
            np.isfinite(sp_e) and np.isfinite(sp_f) and sp_e >= sp_f - 1e-4),
    }


# --------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    if PANEL_CACHE.exists() and not args.rebuild:
        print(f"[panel] loading cache {PANEL_CACHE.name}", flush=True)
        panel = pd.read_parquet(PANEL_CACHE)
    else:
        cfg = load_config()
        store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
        print("[panel] building from the shipped scorer over the local store "
              "(this reads years of prices; ~1-3 min)...", flush=True)
        panel = build_v3_panel(
            store,
            start=pd.Timestamp(args.start).date(),
            end=pd.Timestamp(args.end).date() if args.end else None)
        if panel.empty:
            print("PANEL EMPTY -- the store served no usable prices. Check "
                  "data/curated.", flush=True)
            sys.exit(2)
        panel.to_parquet(PANEL_CACHE)
        print(f"[panel] cached {len(panel):,} rows -> {PANEL_CACHE.name}", flush=True)

    dts = pd.to_datetime(panel["date"])
    print(f"[panel] {len(panel):,} rows, {dts.nunique()} signal dates, "
          f"{dts.min().date()}..{dts.max().date()}", flush=True)

    results = {
        "window": {"start": str(dts.min().date()), "end": str(dts.max().date()),
                   "signal_dates": int(dts.nunique()), "rows": int(len(panel))},
        "label": LABEL,
        "caveat": ("ranking signs/weights fit 2018-2024; full-window IC is "
                   "partly in-sample. Sign STABILITY across sub-periods is the "
                   "trustworthy read here."),
        "EXP_A_sign_stability": exp_a_sign_stability(panel),
        "EXP_B_delivery_incremental": exp_b_delivery(panel),
        "EXP_C_blend_vs_single": exp_c_blend_vs_single(panel),
        "EXP_D_equal_vs_frozen_weights": exp_d_equal_weight(panel),
    }
    out_json = OUT / "results_2026_09.json"
    out_json.write_text(json.dumps(results, indent=1, default=str))
    print(f"\n[results] written -> {out_json}", flush=True)
    _summary(results)


def _summary(r: dict) -> None:
    print("\n" + "=" * 68)
    print("EXP-A  THEME IC SIGN STABILITY (whole / thirds)")
    w = r["EXP_A_sign_stability"]["whole"]
    for t in THEMES:
        if t in w:
            print(f"  {t:10s} whole IC {w[t]['ic']:+.4f} (t {w[t]['t']:+.2f})")
    flips = r["EXP_A_sign_stability"]["theme_sign_flips_across_thirds"]
    flipped = [t for t, f in flips.items() if f]
    print(f"  sign-flips across thirds: {flipped or 'none'}")
    qf = r["EXP_A_sign_stability"]["quality_factors"]
    for f, d in qf.items():
        print(f"  quality/{f}: half1 IC {d['half1']['ic']:+.4f}  "
              f"half2 IC {d['half2']['ic']:+.4f}  flips={d['sign_flips']}")

    print("\nEXP-B  DELIVERY INCREMENTAL VALUE")
    fm = r["EXP_B_delivery_incremental"]["multivariate_fm"]
    if "ownership_sub" in fm:
        o = fm["ownership_sub"]
        print(f"  ownership coef after all themes: {o['coef']:+.4f} (t {o['t']:+.2f})")
    raw = r["EXP_B_delivery_incremental"]["delivery_ic_raw"]
    rd = r["EXP_B_delivery_incremental"]["delivery_ic_resid_on_liquidity_vol"]
    print(f"  delivery IC raw            {raw.get('ic', float('nan')):+.4f} (t {raw.get('t', float('nan')):+.2f})")
    print(f"  delivery IC resid liq/vol  {rd.get('ic', float('nan')):+.4f} (t {rd.get('t', float('nan')):+.2f})")

    print("\nEXP-C  BLEND vs BEST SINGLE THEME (quintile spread)")
    c = r["EXP_C_blend_vs_single"]
    print(f"  composite spread {c['composite']['spread']:+.4f} (t {c['composite']['spread_t']:+.2f})")
    for t, d in c["themes"].items():
        print(f"    {t:10s} {d['spread']:+.4f} (t {d['spread_t']:+.2f})")
    print(f"  blend beats best single: {c['blend_beats_best_single_on_spread']}"
          f"  (best single {c['best_single_spread']:+.4f})")

    d = r.get("EXP_D_equal_vs_frozen_weights", {})
    if d:
        print("\nEXP-D  EQUAL-WEIGHT vs FROZEN (perf-proportional) WEIGHTS")
        fz, eq = d["frozen_weight"], d["equal_weight"]
        print(f"  frozen  spread {fz['spread']:+.4f} (t {fz['spread_t']:+.2f}) | IC {fz['ic']:+.4f}")
        print(f"  equal   spread {eq['spread']:+.4f} (t {eq['spread_t']:+.2f}) | IC {eq['ic']:+.4f}")
        print(f"  equal-weight at least as good: {d['equal_weight_at_least_as_good']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
