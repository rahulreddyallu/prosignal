"""Stage B: theme weighting, capping, and the absolute floor. TRAIN ONLY.

Weights are applied FOLD BY FOLD from that fold's own training rows, so nothing
in a blend has seen the block it is scoring.
"""
import sys, pickle, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core, composite as CO, themes as TH

blob = pickle.load(open("/home/claude/psr/cache/stageA2.pkl", "rb"))
folds = blob["folds"]
tr = core.load_train2().sort_values(["date", "symbol"]).reset_index(drop=True)
H = 21
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
dates_col = tr["date"]
yv = tr[f"y{H}"].to_numpy("float64")
base = tr[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
           "mcap", f"y{H}"]].copy()
n = len(tr)


def blend_folds(sub, fold_w, scheme, cap, min_themes, floor=0.0):
    """Blend each fold's test rows with that fold's own weights."""
    score = np.full(n, np.nan)
    nth = np.zeros(n)
    npos = np.zeros(n)
    wlog = []
    themes = list(sub)
    M = np.column_stack([sub[t].astype("float64") for t in themes])
    for (trn, te), wrow in zip(folds, fold_w):
        if scheme == "equal":
            raw = {t: 1.0 for t in themes}
        elif scheme == "topk":
            raw = {t: max(wrow[t]["topk"], 0.0) for t in themes}
        elif scheme == "ic_t":
            raw = {t: max(wrow[t]["ic_t"], 0.0) for t in themes}
        else:
            raise ValueError(scheme)
        w = CO.cap_weights(raw, cap, floor=floor)
        wlog.append(w)
        W = np.array([w[t] for t in themes])
        blk = M[te]
        ok = np.isfinite(blk)
        num = np.nansum(np.where(ok, blk * W, 0.0), axis=1)
        den = np.where(ok, W, 0.0).sum(axis=1)
        cnt = ok.sum(axis=1)
        s = np.where((den > 0) & (cnt >= min_themes), num / np.maximum(den, 1e-12), np.nan)
        score[te] = s
        nth[te] = cnt
        npos[te] = ((blk > 0) & ok).sum(axis=1)
    mean_w = {t: float(np.mean([w[t] for w in wlog])) for t in themes}
    return score, nth, npos, mean_w


def metrics(score):
    d = pd.DataFrame({"d": dates_col, "s": score, "y": yv}).dropna()
    ics, ex, q = [], [], []
    for _, g in d.groupby("d", sort=True):
        if len(g) < 60:
            continue
        a = g["s"].rank().to_numpy(); b = g["y"].rank().to_numpy()
        if a.std() > 1e-9 and b.std() > 1e-9:
            ics.append(np.corrcoef(a, b)[0, 1])
        ss, yy = g["s"].to_numpy(), g["y"].to_numpy()
        k = max(len(g) // 10, 5)
        ex.append(yy[np.argpartition(-ss, k)[:k]].mean() - yy.mean())
        kk = max(len(g) // 5, 5); o = np.argsort(-ss)
        q.append(yy[o[:kk]].mean() - yy[o[-kk:]].mean())
    t = lambda z: float(np.mean(z) / (np.std(z, ddof=1) / np.sqrt(len(z)))) if len(z) > 5 else np.nan
    return (float(np.mean(ics)), t(np.array(ics)), float(np.mean(ex)),
            t(np.array(ex)), float(np.mean(q)), t(np.array(q)), len(q))


def book(score, npos=None, min_pos=None, univ=750, slots=10, entry=15, exit_=25,
         rebal=1, cap=3):
    d = base.copy(); d["score"] = score
    if min_pos is not None and npos is not None:
        d = d[npos >= min_pos]
    d = d.dropna(subset=["score"])
    d = d[d.adtv_rank <= univ]
    if len(d) < 2000:
        return None
    P = core.prepare(d)
    sim = core.simulate(P, slots=slots, entry_rank=entry, exit_rank=exit_,
                        rebalance_every=rebal, max_per_sector=cap)
    c, b = sim["curve"], sim["bench_curve"]
    if len(c) < 20:
        return None
    ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
    ir = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
    yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
    ye = {int(v): float(np.prod(1 + ex[yy == v]) - 1) for v in np.unique(yy)}
    return dict(ann=sim["ann"], bench=sim["bench_ann"],
                excess=sim["ann"] - sim["bench_ann"], ir=ir, sharpe=sim["sharpe"],
                maxdd=sim["maxdd"], cost=sim["cost_drag_ann"], cash=sim["cash_share"],
                hold=sim["median_hold_sessions"], ntr=sim["n_closed"],
                yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                yr_min=float(min(ye.values())))


rows, cache = [], {}
for method, blk in blob["m"].items():
    sub = {t: np.asarray(v, dtype="float64") for t, v in blk["sub"].items()}
    for scheme in ("topk", "ic_t", "equal"):
        for cap in (1.0, 0.50, 0.45, 0.40, 0.35, 0.30):
            for floor in (0.0, 0.06, 0.10, 0.14):
                min_themes = 3
                sc, nth, npos, mw = blend_folds(sub, blk["fold_w"], scheme, cap,
                                                min_themes, floor=floor)
                if np.isfinite(sc).sum() < 20000:
                    continue
                ic, ic_t, ex, ex_t, q, q_t, nd = metrics(sc)
                bk = book(sc)
                key = (method, scheme, cap, floor, min_themes)
                cache[key] = (sc.astype("float32"), npos.astype("int8"))
                r = dict(method=method, scheme=scheme, cap=cap, floor=floor,
                         min_themes=min_themes,
                         ic=ic, ic_t=ic_t, topk=ex, topk_t=ex_t, quint=q, quint_t=q_t,
                         n_dates=nd, max_w=max(mw.values()),
                         **{f"w_{k}": round(v, 3) for k, v in mw.items()})
                if bk:
                    r.update(bk)
                rows.append(r)
    print(f"  {method}: {len(rows)} configs", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/s3b3.csv", index=False)
with open("/home/claude/psr/cache/s3b3_scores.pkl", "wb") as f:
    pickle.dump(cache, f)
pd.set_option("display.width", 280)
c = ["method", "scheme", "cap", "floor", "min_themes", "ic", "ic_t", "topk", "topk_t",
     "quint", "quint_t", "excess", "ir", "sharpe", "maxdd", "max_w"]
wc = [x for x in df.columns if x.startswith("w_")]
print("\n=== top 18 by quintile-spread t ===")
print(df.sort_values("quint_t", ascending=False).head(18)[c + wc].round(4).to_string(index=False))
print("\n=== top 12 by book excess ===")
print(df.sort_values("excess", ascending=False).head(12)[c + wc].round(4).to_string(index=False))
print("\n=== marginals ===")
for k in ("method", "scheme", "cap", "floor"):
    print("\n--", k)
    print(df.groupby(k)[["quint", "quint_t", "topk", "topk_t", "excess", "ir",
                         "sharpe", "maxdd"]].mean().round(4).to_string())
