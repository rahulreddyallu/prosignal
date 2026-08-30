"""Round 1: feature set x method x horizon, at the shipped universe. TRAIN only."""
import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, search as S

tr = core.load_train().sort_values(["date", "symbol"]).reset_index(drop=True)
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close", "atr_pct"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
R = core.sector_neutral_ranks(tr, fcols)
tr = pd.concat([tr, R], axis=1)
print("panel", tr.shape)

METHODS = ["equal", "ic_weighted", "ridge_1.0", "ridge_20.0", "ridge_200.0",
           "lasso_0.0005", "enet_0.001_0.5", "pcr_5", "pcr_10"]
HORIZONS = [5, 10, 21, 42, 63]

rows = []
t0 = time.time()
for h in HORIZONS:
    kh = S.kept(h)
    if not kh:
        continue
    fsets = {"kept_h": kh,
             "decorr_h": S.decorrelated(tr, kh, None),
             "swing_union": S.FSETS["swing_union"],
             "any_union": S.FSETS["any_union"],
             "incumbent": S.FSETS["incumbent"]}
    for fname, feats in fsets.items():
        feats = [f for f in feats if f + "_r" in tr.columns]
        if len(feats) < 3:
            continue
        for m in METHODS:
            oos = S.walk_forward(tr, feats, m, h)
            if oos is None:
                continue
            r = S.evaluate(oos, h)
            if r is None:
                continue
            r.update({"h": h, "fset": fname, "nf": len(feats), "method": m})
            rows.append(r)
    print(f"h={h} done  {time.time()-t0:.0f}s  ({len(rows)} configs)")

df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/r1.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== TOP 25 by validation net Sharpe ===")
print(df.sort_values("sharpe", ascending=False).head(25)[
    ["h", "fset", "nf", "method", "ic", "ic_t", "topk_excess", "ann", "bench_ann",
     "sharpe", "maxdd", "cost_drag"]].round(4).to_string(index=False))
print("\n=== TOP 15 by validation net annualised ===")
print(df.sort_values("ann", ascending=False).head(15)[
    ["h", "fset", "method", "ic", "ann", "bench_ann", "sharpe", "maxdd"]].round(4).to_string(index=False))
