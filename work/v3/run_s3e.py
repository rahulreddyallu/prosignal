"""Final validation pass, with the coverage cap live. TRAIN ONLY."""
import sys, pickle, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core, composite as CO

blob = pickle.load(open("/home/claude/psr/cache/stageA2.pkl", "rb"))
folds = blob["folds"]
tr = core.load_train2().sort_values(["date", "symbol"]).reset_index(drop=True)
H, PER_YR = 21, core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
dates_col, yv = tr["date"], tr["y21"].to_numpy("float64")
n = len(tr)
base = tr[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
           "dist_200dma"]].copy()

GATES = {
    "none": lambda d: pd.Series(True, index=d.index),
    "npos3": lambda d: d["npos"] >= 3,
    "trend_npos3": lambda d: (d["dist_200dma"] > 0) & (d["npos"] >= 3),
}


def metrics(score):
    d = pd.DataFrame({"d": dates_col, "s": score, "y": yv}).dropna()
    ics, ex, q = [], [], []
    for _, g in d.groupby("d", sort=True):
        if len(g) < 60:
            continue
        a, b = g["s"].rank().to_numpy(), g["y"].rank().to_numpy()
        if a.std() > 1e-9 and b.std() > 1e-9:
            ics.append(np.corrcoef(a, b)[0, 1])
        ss, yy = g["s"].to_numpy(), g["y"].to_numpy()
        k = max(len(g) // 10, 5)
        ex.append(yy[np.argpartition(-ss, k)[:k]].mean() - yy.mean())
        kk = max(len(g) // 5, 5); o = np.argsort(-ss)
        q.append(yy[o[:kk]].mean() - yy[o[-kk:]].mean())
    t = lambda z: float(np.mean(z) / (np.std(z, ddof=1) / np.sqrt(len(z)))) if len(z) > 5 else np.nan
    return (float(np.mean(ics)), t(np.array(ics)), float(np.mean(ex)),
            t(np.array(ex)), float(np.mean(q)), t(np.array(q)))


rows, keep_scores = [], {}
for method, blk in blob["m"].items():
    if method != "equal":
        continue
    sub = {t: np.asarray(v, dtype="float64") for t, v in blk["sub"].items()}
    themes = list(sub)
    M = np.column_stack([sub[t] for t in themes])
    for cap, use_cov in itertools.product((0.35, 0.40, 0.45), (True, False)):
        score = np.full(n, np.nan); npos = np.zeros(n); wlog = []
        for (trn, te), wrow in zip(folds, blk["fold_w"]):
            cov = {t: float(np.isfinite(M[trn, i]).mean()) for i, t in enumerate(themes)}
            raw = {t: max(wrow[t]["topk"], 0.0) for t in themes}
            w = CO.cap_weights(raw, cap, floor=0.06,
                               coverage=cov if use_cov else None)
            wlog.append(w)
            W = np.array([w[t] for t in themes])
            blkte = M[te]; ok = np.isfinite(blkte)
            num = np.nansum(np.where(ok, blkte * W, 0.0), axis=1)
            den = np.where(ok, W, 0.0).sum(axis=1)
            cnt = ok.sum(axis=1)
            score[te] = np.where((den > 0) & (cnt >= 3), num / np.maximum(den, 1e-12), np.nan)
            npos[te] = ((blkte > 0) & ok).sum(axis=1)
        mw = {t: float(np.mean([w[t] for w in wlog])) for t in themes}
        ic, ic_t, ex, ex_t, q, q_t = metrics(score)
        d0 = base.copy()
        d0["score"] = score; d0["npos"] = npos.astype("int16")
        d0 = d0.dropna(subset=["score"])
        keep_scores[(cap, use_cov)] = (score.astype("float32"), npos.astype("int8"))
        for u, gname in itertools.product((500, 750), GATES):
            du = d0[d0.adtv_rank <= u]
            dd = du[GATES[gname](du).fillna(False).to_numpy()]
            if len(dd) < 2000:
                continue
            per_date = dd.groupby("date").size().reindex(
                sorted(du["date"].unique()), fill_value=0)
            P = core.prepare(dd)
            for slots, e, x, rb in ((10, 20, 30, 2), (10, 15, 30, 2), (12, 24, 36, 2),
                                    (10, 20, 30, 1), (8, 16, 24, 2)):
                sim = core.simulate(P, slots=slots, entry_rank=e, exit_rank=x,
                                    rebalance_every=rb, max_per_sector=3)
                c, b = sim["curve"], sim["bench_curve"]
                exr = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
                ir = float(exr.mean() / (exr.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
                yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
                ye = {int(v): float(np.prod(1 + exr[yy == v]) - 1) for v in np.unique(yy)}
                rows.append(dict(cap=cap, cov_cap=use_cov, univ=u, gate=gname,
                                 slots=slots, entry=e, exit=x, rebal=rb,
                                 ic=ic, ic_t=ic_t, topk=ex, topk_t=ex_t,
                                 quint=q, quint_t=q_t, ann=sim["ann"],
                                 bench=sim["bench_ann"],
                                 excess=sim["ann"] - sim["bench_ann"], ir=ir,
                                 sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                                 cost=sim["cost_drag_ann"], cash=sim["cash_share"],
                                 hold=sim["median_hold_sessions"],
                                 min_names=int(per_date.min()),
                                 dates_short=int((per_date < slots).sum()),
                                 yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                                 yr_min=float(min(ye.values())),
                                 **{f"w_{k}": round(v, 3) for k, v in mw.items()}))
    print(f"  {method} done ({len(rows)})", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/s3e.csv", index=False)
with open("/home/claude/psr/cache/s3e_scores.pkl", "wb") as f:
    pickle.dump(keep_scores, f)
pd.set_option("display.width", 300)
c = ["cap", "cov_cap", "univ", "gate", "slots", "entry", "exit", "rebal", "ic", "ic_t",
     "topk_t", "quint", "quint_t", "ann", "excess", "ir", "sharpe", "maxdd", "cash",
     "min_names", "dates_short", "hold", "yr_pos", "yr_min"]
wc = [x for x in df.columns if x.startswith("w_")]
print("\n=== coverage cap ON vs OFF, on the identical grid ===")
print(df.groupby("cov_cap")[["ic", "ic_t", "quint", "quint_t", "excess", "ir",
                             "sharpe", "maxdd"]].median().round(4).to_string())
print("\n=== top 18 by IR ===")
print(df.sort_values("ir", ascending=False).head(18)[c + wc].round(3).to_string(index=False))
