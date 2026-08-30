"""The absolute quality floor, and how often it produces NO TRADE. TRAIN ONLY.

A floor on a CROSS-SECTIONAL RANK cannot fire. The composite is a rank blend, so
somebody is top of the list every day however weak the day is -- and "at least
three themes above the median" is satisfied by roughly half the universe by
construction. Measured over 235 validation dates it emptied the shortlist on
exactly ZERO of them.

So the floor has to be measured against the stock itself rather than against its
peers: an absolute trend condition, a floor on the momentum theme's own raw
sign, and a minimum number of themes on the right side. On a broad decline few
names clear it and the book holds cash, which is the NO TRADE state the product
is required to be able to reach.
"""
import sys, pickle, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core

cache = pickle.load(open("/home/claude/psr/cache/s3b3_scores.pkl", "rb"))
tr = core.load_train2().sort_values(["date", "symbol"]).reset_index(drop=True)
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS

FLOORS = {
    "none":        lambda d: pd.Series(True, index=d.index),
    "npos3":       lambda d: d["npos"] >= 3,
    "above200":    lambda d: d["dist_200dma"] > 0,
    "above200_50": lambda d: (d["dist_200dma"] > 0) & (d["dist_50dma"] > 0),
    "trend_npos3": lambda d: (d["dist_200dma"] > 0) & (d["npos"] >= 3),
    "trend_npos4": lambda d: (d["dist_200dma"] > 0) & (d["npos"] >= 4),
    "strict":      lambda d: ((d["dist_200dma"] > 0) & (d["dist_50dma"] > 0)
                              & (d["npos"] >= 4) & (d["prox_52w_now"] > -0.20)),
}

base = tr[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
           "dist_200dma", "dist_50dma", "prox_52w_now"]].copy()
CONFIGS = [("equal", "topk", 0.45, 0.06), ("equal", "topk", 0.35, 0.06)]
rows = []
for cfg in CONFIGS:
    sc, npos = cache[(cfg[0], cfg[1], cfg[2], cfg[3], 3)]
    d0 = base.copy()
    d0["score"] = np.asarray(sc, dtype="float64")
    d0["npos"] = np.asarray(npos, dtype="int16")
    d0 = d0.dropna(subset=["score"])
    for u in (500, 750):
        du = d0[d0.adtv_rank <= u]
        all_dates = du["date"].nunique()
        for fname, fn in FLOORS.items():
            dd = du[fn(du).fillna(False).to_numpy()]
            if len(dd) < 2000:
                continue
            per_date = dd.groupby("date").size().reindex(
                sorted(du["date"].unique()), fill_value=0)
            for slots in (8, 10, 12):
                short = int((per_date < slots).sum())
                empty = int((per_date == 0).sum())
                P = core.prepare(dd)
                for e_m, x_m, rb in itertools.product((1.5, 2.0), (2.5, 3.0), (1, 2)):
                    e, x = int(slots * e_m), int(slots * x_m)
                    sim = core.simulate(P, slots=slots, entry_rank=e, exit_rank=x,
                                        rebalance_every=rb, max_per_sector=3)
                    c, b = sim["curve"], sim["bench_curve"]
                    if len(c) < 20:
                        continue
                    ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
                    ir = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
                    yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
                    ye = {int(v): float(np.prod(1 + ex[yy == v]) - 1)
                          for v in np.unique(yy)}
                    rows.append(dict(cap=cfg[2], floor_w=cfg[3], univ=u, gate=fname,
                                     slots=slots, entry=e, exit=x, rebal=rb,
                                     ann=sim["ann"], bench=sim["bench_ann"],
                                     excess=sim["ann"] - sim["bench_ann"], ir=ir,
                                     sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                                     cost=sim["cost_drag_ann"], cash=sim["cash_share"],
                                     hold=sim["median_hold_sessions"],
                                     pass_rate=float(len(dd) / len(du)),
                                     dates_no_name=empty,
                                     dates_short=short, n_dates=all_dates,
                                     yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                                     yr_min=float(min(ye.values()))))
    print(f"  cap={cfg[2]} done ({len(rows)})", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/s3d.csv", index=False)
pd.set_option("display.width", 280)
print("\n=== does the floor ever fire? ===")
g = df.groupby("gate").agg(pass_rate=("pass_rate", "median"),
                           dates_no_name=("dates_no_name", "median"),
                           dates_short=("dates_short", "median"),
                           n_dates=("n_dates", "median"),
                           cash=("cash", "median"), excess=("excess", "median"),
                           ir=("ir", "median"), sharpe=("sharpe", "median"),
                           maxdd=("maxdd", "median"))
print(g.round(3).to_string())
c = ["cap", "univ", "gate", "slots", "entry", "exit", "rebal", "ann", "excess",
     "ir", "sharpe", "maxdd", "cash", "dates_short", "hold", "yr_pos", "yr_min"]
print("\n=== top 20 by IR ===")
print(df.sort_values("ir", ascending=False).head(20)[c].round(3).to_string(index=False))
