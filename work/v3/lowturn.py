"""A book whose costs the edge can actually pay for. TRAIN ONLY.

The criterion is stated BEFORE the search and is not a performance target:
modelled cost drag <= 4% a year. Both sealed windows are spent, so whatever
comes out of this is shipped as NOT YET HOLDOUT-TESTED and the quarterly
re-check is what will settle it on fresh data.

Turnover is a mechanical property of a book -- it needs no labels and no
outcome -- so choosing on it uses nothing the holdout told me.
"""
from __future__ import annotations
import sys, pickle, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core

COST_CAP = 0.04
scores = pickle.load(open("/home/claude/psr/cache/s3e_scores.pkl", "rb"))
sc, npos = scores[(0.40, True)]
tr = core.load_train2().sort_values(["date", "symbol"]).reset_index(drop=True)
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
d0 = tr[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
         "dist_200dma"]].copy()
d0["score"] = np.asarray(sc, dtype="float64")
d0["npos"] = np.asarray(npos, dtype="int16")
d0 = d0.dropna(subset=["score"])

rows = []
for u in (500, 750):
    du = d0[d0.adtv_rank <= u]
    for gate in ("trend_npos3", "none"):
        dd = du[((du.dist_200dma > 0) & (du.npos >= 3)).fillna(False).to_numpy()] \
            if gate == "trend_npos3" else du
        per_date = dd.groupby("date").size().reindex(
            sorted(du["date"].unique()), fill_value=0)
        P = core.prepare(dd)
        yy = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
        for slots, em, xm, rb in itertools.product(
                (10, 12, 15, 20), (1.5, 2.0, 3.0), (3.0, 4.0, 6.0, 8.0), (2, 4, 6, 8)):
            e, x = int(slots * em), int(slots * xm)
            if x <= e:
                continue
            sim = core.simulate(P, slots=slots, entry_rank=e, exit_rank=x,
                                rebalance_every=rb, max_per_sector=3)
            c, b = sim["curve"], sim["bench_curve"]
            if len(c) < 20:
                continue
            ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
            ir = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
            ye = {int(v): float(np.prod(1 + ex[yy == v]) - 1) for v in np.unique(yy)}
            rows.append(dict(univ=u, gate=gate, slots=slots, entry=e, exit=x, rebal=rb,
                             ann=sim["ann"], bench=sim["bench_ann"],
                             excess=sim["ann"] - sim["bench_ann"], ir=ir,
                             sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                             cost=sim["cost_drag_ann"], gross=sim["ann"] + sim["cost_drag_ann"],
                             cash=sim["cash_share"], hold=sim["median_hold_sessions"],
                             ntr=sim["n_closed"], min_names=int(per_date.min()),
                             short=int((per_date < slots).sum()),
                             yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                             yr_min=float(min(ye.values()))))
df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/lowturn.csv", index=False)
pd.set_option("display.width", 280)
c = ["univ", "gate", "slots", "entry", "exit", "rebal", "ann", "bench", "gross",
     "cost", "excess", "ir", "sharpe", "maxdd", "hold", "ntr", "min_names",
     "short", "yr_pos", "yr_min"]
ok = df[df.cost <= COST_CAP]
print(f"{len(ok)} of {len(df)} book settings hold modelled cost at or under "
      f"{COST_CAP:.0%} a year\n")
print("=== of those, the top 20 by information ratio ===")
print(ok.sort_values("ir", ascending=False).head(20)[c].round(3).to_string(index=False))
print("\n=== cost vs everything else, across the whole sweep ===")
df["cost_band"] = pd.cut(df.cost, [0, .02, .03, .04, .06, .09, 1.0])
print(df.groupby("cost_band", observed=True)[["gross", "excess", "ir", "sharpe",
                                              "maxdd", "hold"]].median().round(3).to_string())
