"""Window B: the whole pipeline re-run on pre-2021-07 data, then ONE evaluation.

This is the only genuinely uncontaminated read available. Window A is the same
18 months v2 was scored on. Window B (2021-07 to 2022-12) has never been touched
by any search -- but the frozen v3 configuration was fitted on data that
INCLUDES it, so applying that configuration to B would be in-sample and worth
nothing.

So the procedure is re-run from scratch on TRAIN2_PRE_B, which ends 2021-02-17:
screen, stability, admission, theme orientation, theme weights. Whatever that
produces is what someone running this method in mid-2021 would have shipped, and
it is evaluated once on the eighteen months that followed -- which contain the
2022 drawdown and the global rate shock.

THE ADMISSION RULE SCALES WITH THE WINDOW. `survivors.MIN_DATES` is 120 on the
full training panel; stated as a fraction that is 40% of its 293 dates, and the
same fraction of the 111 dates available here is 60. Holding the absolute number
would admit nothing and would be testing the calendar rather than the method.
"""
from __future__ import annotations
import sys, json, hashlib, os, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, themes as TH, guard as G, composite as CO, screen as SC
import stability as ST, survivors as SV, search3 as S3, holdout3 as HO

CACHE = "/home/claude/psr/cache"
CAP, WFLOOR, MIN_THEMES = 0.40, 0.06, 3
BOOK = dict(slots=10, entry_rank=20, exit_rank=30, rebalance_every=2,
            max_per_sector=3, weighting="equal")
OUT_CFG = f"{CACHE}/FROZEN_V3_PRE_B.json"


def build_cfg():
    tr = core.load_train2(pre_b=True)
    cols = G.factor_columns(tr)
    nd = tr["date"].nunique()
    min_dates = max(60, int(0.40 * nd))
    print(f"pre-B training: {nd} dates, {len(tr):,} rows, {len(cols)} factors, "
          f"admission floor {min_dates} dates", flush=True)

    sc = SC.run(horizons=(10, 21, 42, 63),
                panel=tr[["date", "symbol", "sector", "y10", "y21", "y42", "y63"] + cols],
                tag="preb")
    sc["theme"] = sc["factor"].map(TH.FACTOR_THEME)
    sc.to_csv(f"{CACHE}/screen_preb.csv", index=False)
    print(f"  screen: {int(sc.keep.sum())} factor-horizons cleared", flush=True)

    rows = []
    for f in sorted(set(sc[sc.keep].factor)):
        if f not in tr.columns:
            continue
        r = ST.split_ic(tr, f)
        if r:
            r.update({"factor": f, "theme": TH.FACTOR_THEME.get(f)})
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{CACHE}/stability_preb.csv", index=False)

    R = CO.rank_block(tr, [f for f in cols if f in set(sc[sc.keep].factor)])
    R.corr(min_periods=1000).to_csv(f"{CACHE}/corr_preb.csv")

    by_theme, cut = SV.admitted(screen_csv=f"{CACHE}/screen_preb.csv",
                                stability_csv=f"{CACHE}/stability_preb.csv",
                                corr_csv=f"{CACHE}/corr_preb.csv",
                                min_dates=min_dates, verbose=True)
    by_theme = {t: fs for t, fs in by_theme.items() if fs}
    if not by_theme:
        raise SystemExit("nothing was admitted on the pre-B window")

    fr = S3.Frame(tr, by_theme)
    allrows = np.arange(len(fr.df))
    signs, raw_w, cov = {}, {}, {}
    for t, fs in fr.by_theme.items():
        h = S3.THEME_HORIZON[t]
        sg, _, _ = S3._fit_theme(fr, fs, allrows, "equal", h)
        signs[t] = sg
        s = CO.theme_subscore(fr.R, fs, sg, fr.dates_col)
        raw_w[t] = S3._theme_contribution(fr, s.to_numpy("float64"), allrows, 21)
        cov[t] = float(np.isfinite(s.to_numpy()).mean())
    w = CO.cap_weights({t: max(v["topk"], 0.0) for t, v in raw_w.items()},
                       CAP, floor=WFLOOR, coverage=cov)
    cfg = {
        "name": "prosignal-v3-pre-B",
        "fitted_on": "2018-11-27 .. 2021-02-17",
        "themes": {t: {"weight": round(w[t], 5),
                       "orientation_horizon_sessions": S3.THEME_HORIZON[t],
                       "coverage": round(cov[t], 4),
                       "validated_topk_contribution": round(raw_w[t]["topk"], 6),
                       "validated_ic_t": round(raw_w[t]["ic_t"], 3),
                       "factors": [{"name": f, "sign": int(signs[t][f])}
                                   for f in fr.by_theme[t]]}
                   for t in fr.by_theme},
        "min_themes": MIN_THEMES, "book": BOOK, "gate": "trend_npos3",
        "universe": {"max_names": 750},
        "theme_weighting": {"cap": CAP, "floor": WFLOOR, "coverage_cap": True},
        "admission_min_dates": min_dates,
    }
    cfg["config_sha256"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    json.dump(cfg, open(OUT_CFG, "w"), indent=1, default=str)
    print("\nPRE-B CONFIG")
    for t, v in sorted(cfg["themes"].items(), key=lambda x: -x[1]["weight"]):
        print(f"  {t:11s} w={v['weight']:.3f} cov={v['coverage']:.2f} "
              f"h={v['orientation_horizon_sessions']:2d} "
              f"{len(v['factors'])} factors: "
              f"{', '.join(f['name'] for f in v['factors'])}")
    print("  sha", cfg["config_sha256"])
    return cfg


if __name__ == "__main__":
    out = f"{CACHE}/HOLDOUT_V3_B.json"
    if os.path.exists(out):
        print("WINDOW B ALREADY RUN -- the result stands.")
        print(open(out).read()); sys.exit(0)
    cfg = json.load(open(OUT_CFG)) if os.path.exists(OUT_CFG) else build_cfg()
    seal = json.load(open(f"{CACHE}/SEAL2.json"))
    assert hashlib.sha256(open(f"{CACHE}/SEALED_B_ERA.parquet", "rb").read()).hexdigest() \
        == seal["B_era"]["sha256"], "the sealed file changed"
    panel = pd.read_parquet(f"{CACHE}/SEALED_B_ERA.parquet")
    print(f"\nevaluating on {panel['date'].nunique()} sealed dates "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}", flush=True)
    res = HO.evaluate(panel, cfg, 21, "B")
    res["seal"] = seal["B_era"]
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("window", "signal_dates", "book_net_of_costs",
                               "absolute_floor", "ranking_h21", "ranking_h42",
                               "theme_ic_h21", "shuffled_score_null")}, indent=1))
