"""Round 2: methods (incl. gradient boosting) x feature set x horizon x universe.
Selection metric is annualised EXCESS over the investable equal-weight
benchmark, guarded by the information ratio and by how many calendar years the
excess is positive in. TRAIN only."""
import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, search as S

tr = core.load_train().sort_values(["date", "symbol"]).reset_index(drop=True)
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close", "atr_pct"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
tr = pd.concat([tr, core.sector_neutral_ranks(tr, fcols)], axis=1)

METHODS = ["equal", "ic_weighted", "ridge_20.0", "ridge_200.0", "lasso_0.0005",
           "pcr_8", "xgb_3_300_0.05_0.8", "xgb_4_400_0.05_0.8", "xgb_2_500_0.05_0.9"]
HORIZONS = [5, 10, 21, 42]
UNIVERSES = [100, 200, 500, 750]

rows, cache = [], {}
t0 = time.time()
for h in HORIZONS:
    kh = S.kept(h)
    fsets = {"kept_h": kh,
             "decorr_h": S.decorrelated(tr, kh, None),
             "swing_union": S.FSETS["swing_union"],
             "incumbent": S.FSETS["incumbent"]}
    for fname, feats in fsets.items():
        feats = [f for f in feats if f + "_r" in tr.columns]
        if len(feats) < 3:
            continue
        for m in METHODS:
            oos = S.walk_forward(tr, feats, m, h)
            if oos is None:
                continue
            cache[(h, fname, m)] = oos
            for u in UNIVERSES:
                r = S.evaluate(oos, h, universe=u)
                if r is None:
                    continue
                r.update({"h": h, "fset": fname, "nf": len(feats), "method": m, "univ": u})
                rows.append(r)
        print(f"  h={h} {fname} done {time.time()-t0:.0f}s")
    pd.DataFrame(rows).to_csv("/home/claude/psr/cache/r2.csv", index=False)

with open("/home/claude/psr/cache/r2_oos.pkl", "wb") as f:
    pickle.dump(cache, f)
df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/r2.csv", index=False)
pd.set_option("display.width", 250)
cols = ["h", "fset", "nf", "method", "univ", "ic", "ic_t", "ann", "bench_ann",
        "excess_ann", "ir", "sharpe", "maxdd", "cost_drag", "yr_pos", "yr_min"]
print("\n=== TOP 30 by validation EXCESS over benchmark ===")
print(df.sort_values("excess_ann", ascending=False).head(30)[cols].round(4).to_string(index=False))
print("\n=== TOP 20 by information ratio ===")
print(df.sort_values("ir", ascending=False).head(20)[cols].round(4).to_string(index=False))
