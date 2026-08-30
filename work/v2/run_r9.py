"""Round 9: final book knee. TRAIN only. After this the configuration is frozen."""
import sys, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL, finalists as F

tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols, horizons=(10, 21, 42), start_frac=0.20, n_blocks=8)
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
H = 42
SURV = F.decorrelate(fr, F.survivors(H), top=12)
print("FACTORS:", SURV)

cols = [fr.col[f] for f in SURV]
s = np.full(len(fr.df), np.nan)
sign_log = []
for trn, te in fr.folds[H]:
    m = np.isfinite(fr.Y[H][trn]); trn = trn[m]
    A = fr.X[np.ix_(trn, cols)].astype("float64"); y = fr.Yr[H][trn]
    sg = np.nan_to_num(np.sign([np.corrcoef(A[:, j], y)[0, 1] for j in range(A.shape[1])]), nan=1.0)
    sg[sg == 0] = 1.0
    sign_log.append(sg.tolist())
    s[te] = (fr.X[np.ix_(te, cols)].astype("float64") * sg).mean(axis=1)
sg_arr = np.array(sign_log)
print("sign stability across folds:",
      {f: int(sg_arr[:, j].sum()) for j, f in enumerate(SURV)}, f"(of {len(sign_log)} folds)")

base = fr.df[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5"]].copy()
base["score"] = s
d = base.dropna(subset=["score"])
rows = []
for slots, e, x, rb, cap, u in itertools.product(
        (8, 10, 12, 15), (1.25, 1.5, 2.0), (2.0, 2.25, 2.5, 3.0), (1, 2), (2, 3), (500, 750)):
    en, ex_ = int(round(slots * e)), int(round(slots * x))
    if ex_ <= en:
        continue
    P = core.prepare(d[d.adtv_rank <= u])
    sim = core.simulate(P, slots=slots, entry_rank=en, exit_rank=ex_,
                        rebalance_every=rb, max_per_sector=cap)
    c, b = sim["curve"], sim["bench_curve"]
    exr = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
    ir = exr.mean() / (exr.std(ddof=1) + 1e-12) * np.sqrt(PER_YR)
    yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
    ye = {int(y): float(np.prod(1 + exr[yy == y]) - 1) for y in np.unique(yy)}
    rows.append(dict(slots=slots, entry=en, exit=ex_, rebal=rb, cap=cap, univ=u,
                     ann=sim["ann"], bench=sim["bench_ann"],
                     excess=sim["ann"] - sim["bench_ann"], ir=float(ir),
                     sharpe=sim["sharpe"], maxdd=sim["maxdd"], cost=sim["cost_drag_ann"],
                     hold=sim["median_hold_sessions"], mhold=sim["mean_hold_sessions"],
                     ntr=sim["n_closed"], cash=sim["cash_share"],
                     yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                     yr_min=float(min(ye.values())), **{f"y{y}": v for y, v in ye.items()}))
df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/r9.csv", index=False)
pd.set_option("display.width", 260)
sw = df[(df.hold >= 10) & (df.hold <= 25)]
print(f"\n=== swing band (median hold 2-5 weeks): {len(sw)} of {len(df)} ===")
print(sw.sort_values("ir", ascending=False).head(18)[
    ["slots", "entry", "exit", "rebal", "cap", "univ", "ann", "excess", "ir", "sharpe",
     "maxdd", "cost", "hold", "mhold", "ntr", "yr_pos", "yr_min"]].round(3).to_string(index=False))
