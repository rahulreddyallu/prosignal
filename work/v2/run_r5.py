"""Round 5: finalist bake-off on a validation window containing the 2020 crash.
TRAIN only."""
import sys, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL, finalists as F, regime as RG

tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols, horizons=(10, 21, 42), start_frac=0.20, n_blocks=8)
reg = pd.read_parquet("/home/claude/psr/cache/regime.parquet")
val_lo = fr.dates[fr.folds[21][0][1].min() and 0] if False else None
first_val = tr["date"].to_numpy()[fr.folds[21][0][1]].min()
print("validation starts", pd.Timestamp(first_val).date(), "| folds", len(fr.folds[21]))

surv21 = F.decorrelate(fr, F.survivors(21), top=12)
surv42 = F.decorrelate(fr, F.survivors(42), top=12)
surv10 = F.decorrelate(fr, F.survivors(10), top=12)
print("survivors h21:", surv21)
print("survivors h42:", surv42)
print("survivors h10:", surv10)

CANDS = {
    "A_surv21_equal":   (surv21, 21, "equal"),
    "B_surv21_icw":     (surv21, 21, "icw"),
    "C_surv21_ridge":   (surv21, 21, "ridge"),
    "D_greedy21_ridge": (F.GREEDY_21, 21, "ridge"),
    "E_incumbent_ridge": (F.INCUMBENT, 21, "ridge"),
    "F_trendcore_equal": (F.TREND_CORE, 21, "equal"),
    "G_surv42_equal":   (surv42, 42, "equal"),
    "H_surv10_equal":   (surv10, 10, "equal"),
    "I_trendcore_ridge": (F.TREND_CORE, 21, "ridge"),
    "J_surv42_ridge":   (surv42, 42, "ridge"),
}


def build_scores(feats, h, how, alpha=2000.0):
    cols = [fr.col[f] for f in feats]
    out = np.full(len(fr.df), np.nan)
    for trn, te in fr.folds[h]:
        m = np.isfinite(fr.Y[h][trn]); trn = trn[m]
        A = fr.X[np.ix_(trn, cols)].astype("float64")
        y = fr.Yr[h][trn]
        B = fr.X[np.ix_(te, cols)].astype("float64")
        if how == "equal":
            sg = np.sign([np.corrcoef(A[:, j], y)[0, 1] for j in range(A.shape[1])])
            sg = np.nan_to_num(sg, nan=1.0); sg[sg == 0] = 1.0
            out[te] = (B * sg).mean(axis=1)
        elif how == "icw":
            ic = np.nan_to_num([np.corrcoef(A[:, j], y)[0, 1] for j in range(A.shape[1])])
            w = ic / (np.abs(ic).sum() + 1e-12)
            out[te] = B @ w
        else:
            G = A.T @ A + alpha * np.eye(A.shape[1])
            out[te] = B @ np.linalg.solve(G, A.T @ y)
    return out


BOOKS = [dict(slots=s, entry=int(s * e), exit_=int(s * x), rebal=r, cap=c, wt="equal", univ=u)
         for s in (8, 10, 12) for (e, x) in ((2, 4), (3, 6)) for r in (2, 4, 8)
         for c in (2, 3) for u in (500, 750)]
GATES = ["always", "ma200", "dd_shallow", "vol_calm", "ma200_breadth"]
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS

rows = []
scores_cache = {}
for name, (feats, h, how) in CANDS.items():
    feats = [f for f in feats if f in fr.col]
    s = build_scores(feats, h, how)
    scores_cache[name] = s
    ex, t, n = fr.topk_excess(s, h, 10)
    ic, ict = fr.ic(s, h)
    d = fr.df[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5"]].copy()
    d["score"] = s
    d = d.dropna(subset=["score"])
    prep = {u: core.prepare(d[d.adtv_rank <= u]) for u in (500, 750)}
    gates = {g: RG.gate_series(reg, g, prep[750]["dates"]) for g in GATES}
    yrs = {u: pd.DatetimeIndex(prep[u]["dates"][1:]).year.to_numpy() for u in (500, 750)}
    for bk in BOOKS:
        for g in GATES:
            u = bk["univ"]
            sim = core.simulate(prep[u], slots=bk["slots"], entry_rank=bk["entry"],
                                exit_rank=bk["exit_"], rebalance_every=bk["rebal"],
                                weighting=bk["wt"], max_per_sector=bk["cap"],
                                regime_ok=gates[g])
            c, b = sim["curve"], sim["bench_curve"]
            exr = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
            ir = exr.mean() / (exr.std(ddof=1) + 1e-12) * np.sqrt(PER_YR)
            yy = yrs[u]; uy = np.unique(yy)
            ye = {int(y): float(np.prod(1 + exr[yy == y]) - 1) for y in uy}
            rows.append(dict(cand=name, h=h, how=how, nf=len(feats), topk=ex, topk_t=t,
                             ic=ic, ic_t=ict, gate=g, **{k: v for k, v in bk.items()},
                             ann=sim["ann"], bench=sim["bench_ann"],
                             excess=sim["ann"] - sim["bench_ann"], ir=float(ir),
                             sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                             cost=sim["cost_drag_ann"], cash=sim["cash_share"],
                             yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                             yr_min=float(min(ye.values())),
                             **{f"y{y}": v for y, v in ye.items()}))
    print(f"  {name}: topk={ex:.4f} t={t:.2f} ic={ic:.4f} ({ict:.2f})  rows={len(rows)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/r5.csv", index=False)
np.savez_compressed("/home/claude/psr/cache/r5_scores.npz", **scores_cache)
print("done", df.shape)
