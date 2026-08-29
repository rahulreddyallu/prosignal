"""Round 6, on TRAIN only:
  (1) integrity check -- the whole pipeline run on SHUFFLED labels must produce
      no excess. If it does, something reads the future.
  (2) horizon x feature-set isolation.
  (3) the NO-TRADE mechanism: a bottom-up absolute admission filter, measured
      against a top-down regime gate and against neither.
"""
import sys, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL, finalists as F, regime as RG

RNG = np.random.default_rng(11)
tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols, horizons=(10, 21, 42), start_frac=0.20, n_blocks=8)
reg = pd.read_parquet("/home/claude/psr/cache/regime.parquet")
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS

SURV = {h: F.decorrelate(fr, F.survivors(h), top=12) for h in (10, 21, 42)}
print({h: SURV[h] for h in SURV})


def scores(feats, h, how="equal", shuffle=False):
    cols = [fr.col[f] for f in feats]
    out = np.full(len(fr.df), np.nan)
    Yr = fr.Yr[h].copy()
    if shuffle:
        for a, b in zip(fr.starts, fr.ends):
            Yr[a:b] = RNG.permutation(Yr[a:b])
    for trn, te in fr.folds[h]:
        m = np.isfinite(fr.Y[h][trn]); trn = trn[m]
        A = fr.X[np.ix_(trn, cols)].astype("float64"); y = Yr[trn]
        B = fr.X[np.ix_(te, cols)].astype("float64")
        if how == "equal":
            sg = np.nan_to_num(np.sign([np.corrcoef(A[:, j], y)[0, 1]
                                        for j in range(A.shape[1])]), nan=1.0)
            sg[sg == 0] = 1.0
            out[te] = (B * sg).mean(axis=1)
        else:
            G = A.T @ A + 2000.0 * np.eye(A.shape[1])
            out[te] = B @ np.linalg.solve(G, A.T @ y)
    return out


base = fr.df[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
              "dist_50dma", "prox_52w_now"]].copy()
FILTERS = {
    "none":       None,
    "above50dma": lambda d: d["dist_50dma"] > 0,
    "trend_abs":  lambda d: (d["dist_50dma"] > 0) & (d["prox_52w_now"] > -0.25),
}
GATES = ["always", "ma200", "vol_calm"]
BOOK = dict(slots=10, entry_rank=30, exit_rank=60, rebalance_every=2,
            max_per_sector=2, weighting="equal")

rows = []
combos = [(f"surv{fh}", fh, h, how) for fh in (10, 21, 42) for h in (10, 21, 42)
          for how in ("equal", "ridge")]
combos += [("shuffled_surv42", 42, 42, "equal")]
for name, fh, h, how in combos:
    feats = [f for f in SURV[fh] if f in fr.col]
    sh = name.startswith("shuffled")
    s = scores(feats, h, how, shuffle=sh)
    ex, t, _ = fr.topk_excess(s, h, 10)
    d = base.copy(); d["score"] = s
    d = d.dropna(subset=["score"])
    for filt, fn in FILTERS.items():
        dd = d.copy()
        dd["name_ok"] = True if fn is None else fn(dd).fillna(False)
        for u in (500, 750):
            P = core.prepare(dd[dd.adtv_rank <= u])
            gs = {g: RG.gate_series(reg, g, P["dates"]) for g in GATES}
            yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
            for g in GATES:
                sim = core.simulate(P, regime_ok=gs[g],
                                    require_name_ok=(filt != "none"), **BOOK)
                c, b = sim["curve"], sim["bench_curve"]
                exr = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
                ir = exr.mean() / (exr.std(ddof=1) + 1e-12) * np.sqrt(PER_YR)
                ye = {int(y): float(np.prod(1 + exr[yy == y]) - 1) for y in np.unique(yy)}
                rows.append(dict(cand=name, fset=f"surv{fh}", h=h, how=how, filt=filt,
                                 gate=g, univ=u, topk=ex, topk_t=t,
                                 ann=sim["ann"], bench=sim["bench_ann"],
                                 excess=sim["ann"] - sim["bench_ann"], ir=float(ir),
                                 sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                                 cost=sim["cost_drag_ann"], cash=sim["cash_share"],
                                 hold=sim["median_hold_sessions"],
                                 yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                                 **{f"y{y}": v for y, v in ye.items()}))
    print(f"  {name} h={h} {how}: topk={ex:.4f} t={t:.2f}", flush=True)
pd.DataFrame(rows).to_csv("/home/claude/psr/cache/r6.csv", index=False)
print("done", len(rows))
