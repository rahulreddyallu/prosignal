"""Round 7, TRAIN only.
 (1) Integrity: the ENTIRE pipeline -- screen, sign orientation, composite,
     book -- re-run on labels permuted within each cross-section, 20 draws.
     A clean pipeline puts the real result outside that distribution.
 (2) Book sweep with the holding period reported, so the shipped book can be
     required to sit in the 1-4 week band the product promises.
 (3) Regime mechanisms that scale rather than switch.
"""
import sys, json, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL, finalists as F, regime as RG, screen as SC

RNG = np.random.default_rng(2026)
tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols, horizons=(10, 21, 42), start_frac=0.20, n_blocks=8)
reg = pd.read_parquet("/home/claude/psr/cache/regime.parquet")
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
H = 42

base = fr.df[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
              "dist_50dma"]].copy()


def composite(feats, h, Yr):
    cols = [fr.col[f] for f in feats]
    out = np.full(len(fr.df), np.nan)
    for trn, te in fr.folds[h]:
        m = np.isfinite(fr.Y[h][trn]); trn = trn[m]
        A = fr.X[np.ix_(trn, cols)].astype("float64"); y = Yr[trn]
        sg = np.nan_to_num(np.sign([np.corrcoef(A[:, j], y)[0, 1]
                                    for j in range(A.shape[1])]), nan=1.0)
        sg[sg == 0] = 1.0
        out[te] = (fr.X[np.ix_(te, cols)].astype("float64") * sg).mean(axis=1)
    return out


def book(scores, **kw):
    d = base.copy(); d["score"] = scores
    d = d.dropna(subset=["score"])
    u = kw.pop("univ", 750)
    P = core.prepare(d[d.adtv_rank <= u])
    sim = core.simulate(P, **kw)
    c, b = sim["curve"], sim["bench_curve"]
    ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
    ir = ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR)
    yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
    ye = {int(y): float(np.prod(1 + ex[yy == y]) - 1) for y in np.unique(yy)}
    return sim, float(ir), ye


# ---------------------------------------------------------------- (1)
print("=== integrity: 20 permuted-label pipelines ===", flush=True)
SURV42 = F.decorrelate(fr, F.survivors(42), top=12)
real_s = composite(SURV42, H, fr.Yr[H])
sim, ir, ye = book(real_s, slots=10, entry_rank=30, exit_rank=60, rebalance_every=2,
                   max_per_sector=2, univ=750)
real = dict(excess=sim["ann"] - sim["bench_ann"], ir=ir, sharpe=sim["sharpe"])
print(f"  REAL   excess={real['excess']:.4f} ir={real['ir']:.3f} sharpe={real['sharpe']:.3f}")

null = []
panel = fr.df
for b_ in range(20):
    Yp = fr.Y[H].copy(); Yrp = fr.Yr[H].copy()
    for a, e in zip(fr.starts, fr.ends):
        p = RNG.permutation(e - a)
        Yp[a:e] = Yp[a:e][p]; Yrp[a:e] = Yrp[a:e][p]
    # re-screen against the permuted labels, exactly as the real pipeline does
    tmp = panel.copy(); tmp[f"y{H}"] = Yp
    sc = SC.run(horizons=(H,), panel=tmp, tag=f"null{b_}")
    kp = list(sc[sc.keep].sort_values("t", key=abs, ascending=False).factor)
    kp = F.decorrelate(fr, [f for f in kp if f in fr.col], top=12)
    if len(kp) < 3:
        null.append(dict(excess=0.0, ir=0.0, sharpe=0.0, nf=len(kp)))
        continue
    s = composite(kp, H, Yrp)
    sim_n, ir_n, _ = book(s, slots=10, entry_rank=30, exit_rank=60,
                          rebalance_every=2, max_per_sector=2, univ=750)
    null.append(dict(excess=sim_n["ann"] - sim_n["bench_ann"], ir=ir_n,
                     sharpe=sim_n["sharpe"], nf=len(kp)))
    print(f"  null {b_:2d}: nf={len(kp):2d} excess={null[-1]['excess']:+.4f} "
          f"ir={ir_n:+.3f}", flush=True)
nd = pd.DataFrame(null)
json.dump({"real": real, "null_mean": nd.mean().to_dict(),
           "null_p95_excess": float(np.percentile(nd.excess, 95)),
           "null_p95_ir": float(np.percentile(nd.ir, 95)),
           "null_max_excess": float(nd.excess.max()),
           "n_null": len(nd)},
          open("/home/claude/psr/cache/integrity.json", "w"), indent=1)
print(f"  NULL   excess mean={nd.excess.mean():.4f} p95={np.percentile(nd.excess,95):.4f} "
      f"max={nd.excess.max():.4f} | ir mean={nd.ir.mean():.3f} p95={np.percentile(nd.ir,95):.3f}",
      flush=True)

# ---------------------------------------------------------------- (2)
print("\n=== book sweep, holding period reported ===", flush=True)
rows = []
for slots, em, xm, rb, cap, u, wt in itertools.product(
        (8, 10, 12), (1.5, 2, 3, 4), (2, 3, 4, 6), (1, 2, 4), (2, 3), (500, 750),
        ("equal",)):
    e, x = int(slots * em), int(slots * xm)
    if x <= e:
        continue
    sim, ir, ye = book(real_s, slots=slots, entry_rank=e, exit_rank=x,
                       rebalance_every=rb, max_per_sector=cap, weighting=wt, univ=u)
    rows.append(dict(slots=slots, entry=e, exit=x, rebal=rb, cap=cap, univ=u, wt=wt,
                     ann=sim["ann"], bench=sim["bench_ann"],
                     excess=sim["ann"] - sim["bench_ann"], ir=ir, sharpe=sim["sharpe"],
                     maxdd=sim["maxdd"], cost=sim["cost_drag_ann"],
                     hold=sim["median_hold_sessions"], mhold=sim["mean_hold_sessions"],
                     ntr=sim["n_closed"],
                     yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                     **{f"y{y}": v for y, v in ye.items()}))
pd.DataFrame(rows).to_csv("/home/claude/psr/cache/r7_book.csv", index=False)
print("book rows", len(rows), flush=True)
print("done")
