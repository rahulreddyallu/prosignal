"""Round 3: book construction, re-simulating cached out-of-sample scores.
Nothing is refitted here, so no label is fitted twice."""
import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, regime as RG

COLS = ["h", "fset", "method", "slots", "entry", "exit", "rebal", "cap", "wt",
        "gate", "univ", "ann", "bench", "excess", "ir", "sharpe", "maxdd",
        "cost", "cash", "yr_pos", "yr_min", "yr_med"]

oos = pickle.load(open("/home/claude/psr/cache/r2_oos.pkl", "rb"))
reg = pd.read_parquet("/home/claude/psr/cache/regime.parquet")
r2 = pd.read_csv("/home/claude/psr/cache/r2.csv")
base = r2[r2.univ == 750].sort_values("topk_t", ascending=False)
cands = list(dict.fromkeys([(int(r.h), r.fset, r.method) for r in base.itertuples()]))[:14]

SLOTS = [5, 8, 12, 20]
BANDS = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (2.0, 6.0)]
REBAL = [1, 2, 4, 8]
CAPS = [2, 3, 0]
GATES = ["always", "ma200", "ma200_breadth"]
UNIVS = [200, 500, 750]
WTS = ["equal", "invvol"]
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS

rows = []
for ci, (h, fset, method) in enumerate(cands):
    d0 = oos[(h, fset, method)]
    PREP, YRS = {}, {}
    for u in UNIVS:
        PREP[u] = core.prepare(d0[d0["adtv_rank"] <= u])
        dts = pd.DatetimeIndex(PREP[u]["dates"][1:])
        yrs = dts.year.to_numpy()
        YRS[u] = (yrs, np.unique(yrs))
    gates = {g: RG.gate_series(reg, g, PREP[750]["dates"]) for g in GATES}
    for slots, band, rebal, cap, wt, g, u in itertools.product(
            SLOTS, BANDS, REBAL, CAPS, WTS, GATES, UNIVS):
        e = int(round(slots * band[0])); x = int(round(slots * band[1]))
        sim = core.simulate(PREP[u], slots=slots, entry_rank=e, exit_rank=x,
                            rebalance_every=rebal, weighting=wt,
                            max_per_sector=(cap or None), regime_ok=gates[g])
        c, b = sim["curve"], sim["bench_curve"]
        if len(c) < 20:
            continue
        ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
        ir = ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(PER_YR)
        yrs, uy = YRS[u]
        ye = np.array([np.prod(1 + ex[yrs == y]) - 1 for y in uy])
        rows.append((h, fset, method, slots, e, x, rebal, cap, wt, g, u,
                     sim["ann"], sim["bench_ann"], sim["ann"] - sim["bench_ann"],
                     float(ir), sim["sharpe"], sim["maxdd"], sim["cost_drag_ann"],
                     sim["cash_share"], float((ye > 0).mean()), float(ye.min()),
                     float(np.median(ye))))
    print(f"  {ci+1}/{len(cands)} h={h} {fset} {method}: {len(rows)} rows", flush=True)
    pd.DataFrame(rows, columns=COLS).to_csv("/home/claude/psr/cache/r3.csv", index=False)
print("done", len(rows))
