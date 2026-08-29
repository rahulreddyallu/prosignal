"""Round 8: the decisive power test. TRAIN only.

The permuted-label null showed a 10-name book's five-year excess is so noisy
that the real result sits inside it. That is a statement about the ESTIMATOR,
not necessarily about the signal. So the same real-vs-null comparison is run on
statistics with progressively more power:

  quintile spread   every name in the cross-section contributes
  25-slot book      idiosyncratic variance cut by ~2.5x
  10-slot book      the shortlist as the product actually presents it

Whichever statistic separates real from null is the one the configuration may
be selected on; the ones that do not separate are reported as noise.
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL, finalists as F, screen as SC

RNG = np.random.default_rng(4242)
NB = 40
H = 42
tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols, horizons=(10, 21, 42), start_frac=0.20, n_blocks=8)
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
base = fr.df[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5"]].copy()


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


def quintile_spread(s, Y, q=5):
    sp = []
    for a, b in zip(fr.starts, fr.ends):
        ss, yy = s[a:b], Y[a:b]
        m = np.isfinite(ss) & np.isfinite(yy)
        if m.sum() < 100:
            continue
        ss, yy = ss[m], yy[m]
        n = len(ss); k = max(n // q, 5)
        o = np.argsort(-ss)
        sp.append(yy[o[:k]].mean() - yy[o[-k:]].mean())
    sp = np.array(sp)
    return float(sp.mean()), float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))), len(sp)


def book_excess(s, slots, entry_m=3, exit_m=6, rebal=2, univ=750):
    d = base.copy(); d["score"] = s
    d = d.dropna(subset=["score"])
    P = core.prepare(d[d.adtv_rank <= univ])
    sim = core.simulate(P, slots=slots, entry_rank=int(slots * entry_m),
                        exit_rank=int(slots * exit_m), rebalance_every=rebal,
                        max_per_sector=max(2, slots // 5))
    return sim["ann"] - sim["bench_ann"], sim


SURV = F.decorrelate(fr, F.survivors(H), top=12)
real_s = composite(SURV, H, fr.Yr[H])
qs, qt, qn = quintile_spread(real_s, fr.Y[H])
r10, s10 = book_excess(real_s, 10)
r25, s25 = book_excess(real_s, 25)
r40, s40 = book_excess(real_s, 40)
real = {"quintile_spread": qs, "quintile_t": qt, "n_dates": qn,
        "ex10": r10, "ex25": r25, "ex40": r40,
        "sharpe10": s10["sharpe"], "sharpe25": s25["sharpe"], "sharpe40": s40["sharpe"],
        "hold10": s10["median_hold_sessions"], "hold25": s25["median_hold_sessions"]}
print("REAL:", json.dumps({k: round(v, 4) for k, v in real.items()}), flush=True)

panel = fr.df
null = []
for b_ in range(NB):
    Yp = fr.Y[H].copy(); Yrp = fr.Yr[H].copy()
    for a, e in zip(fr.starts, fr.ends):
        p = RNG.permutation(e - a)
        Yp[a:e] = Yp[a:e][p]; Yrp[a:e] = Yrp[a:e][p]
    tmp = panel.copy(); tmp[f"y{H}"] = Yp
    sc = SC.run(horizons=(H,), panel=tmp, tag="null")
    kp = list(sc[sc.keep].sort_values("t", key=abs, ascending=False).factor)
    kp = F.decorrelate(fr, [f for f in kp if f in fr.col], top=12)
    if len(kp) < 3:
        kp = list(sc.sort_values("t", key=abs, ascending=False).factor)[:6]
        kp = F.decorrelate(fr, [f for f in kp if f in fr.col], top=12)
    s = composite(kp, H, Yrp)
    a1, at, _ = quintile_spread(s, Yp)
    e10, _ = book_excess(s, 10)
    e25, _ = book_excess(s, 25)
    e40, _ = book_excess(s, 40)
    null.append(dict(nf=len(kp), qs=a1, qt=at, ex10=e10, ex25=e25, ex40=e40))
    print(f"  null {b_:2d} nf={len(kp):2d} qs={a1:+.4f}(t{at:+.2f}) "
          f"ex10={e10:+.3f} ex25={e25:+.3f} ex40={e40:+.3f}", flush=True)

nd = pd.DataFrame(null)
res = {"real": real, "n_null": len(nd)}
for k, rv in (("qs", qs), ("qt", qt), ("ex10", r10), ("ex25", r25), ("ex40", r40)):
    v = nd[k].to_numpy()
    res[k] = {"real": float(rv), "null_mean": float(v.mean()),
              "null_sd": float(v.std(ddof=1)), "null_p95": float(np.percentile(v, 95)),
              "null_max": float(v.max()),
              "p_value": float((v >= rv).mean()),
              "z": float((rv - v.mean()) / (v.std(ddof=1) + 1e-12))}
json.dump(res, open("/home/claude/psr/cache/power.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "real"}, indent=1))
