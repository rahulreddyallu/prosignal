"""Apply the absolute floor to ENTRIES ONLY. TRAIN ONLY.

The floor was filtering the whole population, so a name that slipped below its
200-session average was EJECTED even while it was still ranked third. That is
not what an entry filter is for, and it is most of why the book turned over:
gated books cost 9-10% a year against 7% ungated, and the difference is forced
exits rather than chosen ones.

Requiring quality to BUY and exiting on rank or stop is both the ordinary
discretionary discipline and, mechanically, far cheaper. Turnover needs no
labels to measure, so this is chosen on the training window against a stated
cost target and nothing here has seen a holdout.
"""
from __future__ import annotations
import sys, pickle, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core
from prosignal.data.instruments import non_equity_symbols

scores = pickle.load(open("/home/claude/psr/cache/s3e_scores.pkl", "rb"))
sc, npos = scores[(0.40, True)]
tr = core.load_train2().sort_values(["date", "symbol"]).reset_index(drop=True)
import data as D_
m = D_.build()
close = pd.DataFrame(m["close"], index=pd.DatetimeIndex(m["dates"]),
                     columns=list(m["symbols"]))
em = pd.read_parquet("/mnt/user-data/uploads/Pro Stock Signal BOT/data/curated/equity_master.parquet")
drop = non_equity_symbols(sorted(tr["symbol"].unique()), equity_master=em, close=close)
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS

d0 = tr[["date", "symbol", "sector", "adtv", "adtv_rank", "atr_pct", "y5", "b5",
         "dist_200dma"]].copy()
d0["score"] = np.asarray(sc, dtype="float64")
d0["npos"] = np.asarray(npos, dtype="int16")
d0 = d0.dropna(subset=["score"])
d0 = d0[~d0["symbol"].isin(drop)]
d0["name_ok"] = ((d0["dist_200dma"] > 0) & (d0["npos"] >= 3)).fillna(False)
print(f"equity-only training rows {len(d0):,}; the floor admits "
      f"{d0['name_ok'].mean():.1%} of them", flush=True)

rows = []
for u in (500, 750):
    du = d0[d0.adtv_rank <= u]
    P_all = core.prepare(du)                       # gate as an ENTRY filter
    gated = du[du.name_ok.to_numpy()]
    P_gate = core.prepare(gated)                   # gate as a POPULATION filter
    per_date = gated.groupby("date").size().reindex(
        sorted(du["date"].unique()), fill_value=0)
    yy = pd.DatetimeIndex(P_all["dates"][1:]).year.to_numpy()
    for mode, P in (("entry_only", P_all), ("population", P_gate)):
        for slots, em_, xm, rb in itertools.product(
                (10, 12, 15), (1.5, 2.0), (2.5, 3.0, 4.0, 6.0), (1, 2, 4)):
            e, x = int(slots * em_), int(slots * xm)
            sim = core.simulate(P, slots=slots, entry_rank=e, exit_rank=x,
                                rebalance_every=rb, max_per_sector=3,
                                require_name_ok=(mode == "entry_only"))
            c, b = sim["curve"], sim["bench_curve"]
            if len(c) < 20:
                continue
            ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
            ir = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
            y2 = pd.DatetimeIndex(P["dates"][1:]).year.to_numpy()
            ye = {int(v): float(np.prod(1 + ex[y2 == v]) - 1) for v in np.unique(y2)}
            rows.append(dict(mode=mode, univ=u, slots=slots, entry=e, exit=x, rebal=rb,
                             ann=sim["ann"], bench=sim["bench_ann"],
                             excess=sim["ann"] - sim["bench_ann"], ir=ir,
                             sharpe=sim["sharpe"], maxdd=sim["maxdd"],
                             cost=sim["cost_drag_ann"],
                             gross=sim["ann"] + sim["cost_drag_ann"],
                             cash=sim["cash_share"], hold=sim["median_hold_sessions"],
                             ntr=sim["n_closed"], min_floor=int(per_date.min()),
                             yr_pos=float(np.mean([v > 0 for v in ye.values()])),
                             yr_min=float(min(ye.values()))))
df = pd.DataFrame(rows)
df.to_csv("/home/claude/psr/cache/entryonly.csv", index=False)
pd.set_option("display.width", 260)
print("\n=== entry-only vs population filter, medians over the same grid ===")
print(df.groupby("mode")[["cost", "gross", "ann", "excess", "ir", "sharpe",
                          "maxdd", "hold", "cash", "ntr"]].median().round(3).to_string())
c = ["mode", "univ", "slots", "entry", "exit", "rebal", "ann", "bench", "gross",
     "cost", "excess", "ir", "sharpe", "maxdd", "cash", "hold", "ntr", "yr_pos", "yr_min"]
print("\n=== top 16 by IR ===")
print(df.sort_values("ir", ascending=False).head(16)[c].round(3).to_string(index=False))
print("\n=== entry-only, cost at or under 5%, best by IR ===")
ok = df[(df["mode"] == "entry_only") & (df.cost <= 0.05)]
print(ok.sort_values("ir", ascending=False).head(12)[c].round(3).to_string(index=False))
