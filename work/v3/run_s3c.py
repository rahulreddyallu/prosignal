"""Stage C: universe, the absolute quality floor, and book construction.
TRAIN ONLY. Nothing is refitted -- the cached level-2 scores are re-simulated."""
import sys, pickle, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core

cache = pickle.load(open("/home/claude/psr/cache/s3b3_scores.pkl", "rb"))
tr = core.load_train2().sort_values(["date", "symbol"]).reset_index(drop=True)
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
base = tr[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
           "mcap", "book_to_price"]].copy()

CONFIGS = [("equal", "topk", 0.35, 0.06), ("equal", "topk", 0.40, 0.0),
           ("equal", "topk", 0.35, 0.0), ("equal", "topk", 0.45, 0.06)]
UNIV = [100, 200, 500, 750]
MINPOS = [0, 2, 3]
rows = []
for cfg in CONFIGS:
    key = (cfg[0], cfg[1], cfg[2], cfg[3], 3)
    if key not in cache:
        print("missing", key); continue
    sc, npos = cache[key]
    d0 = base.copy()
    d0["score"] = np.asarray(sc, dtype="float64")
    d0["npos"] = np.asarray(npos, dtype="int16")
    d0 = d0.dropna(subset=["score"])
    for u in UNIV:
        du = d0[d0.adtv_rank <= u]
        for mp in MINPOS:
            dd = du[du.npos >= mp] if mp else du
            if len(dd) < 2000:
                continue
            P = core.prepare(dd)
            # how often the floor empties the shortlist
            per_date = dd.groupby("date").size()
            all_dates = du["date"].nunique()
            empty = all_dates - (per_date >= 1).sum()
            yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
            for slots, em, xm, rb, sc_cap in itertools.product(
                    (8, 10, 12), (1.5, 2.0, 3.0), (2.5, 3.0, 4.0), (1, 2), (2, 3)):
                e, x = int(round(slots * em)), int(round(slots * xm))
                if x <= e:
                    continue
                sim = core.simulate(P, slots=slots, entry_rank=e, exit_rank=x,
                                    rebalance_every=rb, max_per_sector=sc_cap)
                c, b = sim["curve"], sim["bench_curve"]
                if len(c) < 20:
                    continue
                ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
                ir = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
                ye = {int(v): float(np.prod(1 + ex[yy == v]) - 1) for v in np.unique(yy)}
                rows.append(dict(cap=cfg[2], floor=cfg[3], univ=u, min_pos=mp,
                                 slots=slots, entry=e, exit=x, rebal=rb,
                                 sec_cap=sc_cap, ann=sim["ann"], bench=sim["bench_ann"],
                                 excess=sim["ann"] - sim["bench_ann"], ir=ir,
                                 sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                                 cost=sim["cost_drag_ann"], cash=sim["cash_share"],
                                 hold=sim["median_hold_sessions"], ntr=sim["n_closed"],
                                 no_trade_dates=int(empty),
                                 yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                                 yr_min=float(min(ye.values()))))
    print(f"  cap={cfg[2]} floor={cfg[3]} done ({len(rows)})", flush=True)
    pd.DataFrame(rows).to_csv("/home/claude/psr/cache/s3c.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/s3c.csv", index=False)
pd.set_option("display.width", 280)
c = ["cap", "floor", "univ", "min_pos", "slots", "entry", "exit", "rebal", "sec_cap",
     "ann", "bench", "excess", "ir", "sharpe", "maxdd", "cost", "hold", "yr_pos", "yr_min"]
print("\n=== top 20 by information ratio ===")
print(df.sort_values("ir", ascending=False).head(20)[c].round(3).to_string(index=False))
print("\n=== marginals ===")
for k in ("univ", "min_pos", "cap", "floor", "slots", "rebal", "entry"):
    print("\n--", k)
    print(df.groupby(k)[["excess", "ir", "sharpe", "maxdd", "cost", "hold"]]
          .median().round(3).to_string())
