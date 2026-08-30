"""THE SEALED HOLDOUT. One run, one configuration, no parameters after this.

Reads FROZEN_CONFIG.json (sha recorded before this file was ever executed),
applies it blind to SEALED_HOLDOUT.parquet, and writes the result. A permuted-
label null is run on the same window purely as a reference distribution -- it
changes nothing and is not a selection step.
"""
import sys, json, hashlib, os, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core

CACHE = "/home/claude/psr/cache"
OUT = f"{CACHE}/HOLDOUT_RESULT.json"
if os.path.exists(OUT):
    print("HOLDOUT ALREADY RUN -- the result stands. Refusing to run again.")
    print(open(OUT).read()); sys.exit(0)

cfg = json.load(open(f"{CACHE}/FROZEN_CONFIG.json"))
seal = json.load(open(f"{CACHE}/SEAL.json"))
ho = pd.read_parquet(f"{CACHE}/SEALED_HOLDOUT.parquet")
print("holdout", seal["holdout_start"], "->", seal["holdout_end"],
      "| dates", ho.date.nunique(), "| rows", len(ho))
assert hashlib.sha256(open(f"{CACHE}/SEALED_HOLDOUT.parquet", "rb").read()).hexdigest() \
    == seal["sha256_holdout"], "holdout file changed since sealing"

FE = [f["name"] for f in cfg["factors"]]
SG = np.array([f["sign"] for f in cfg["factors"]], dtype="float64")
WT = np.array([f["weight"] for f in cfg["factors"]], dtype="float64")

ho = ho.sort_values(["date", "symbol"]).reset_index(drop=True)
R = core.sector_neutral_ranks(ho, FE)
X = np.nan_to_num(R.to_numpy("float64"))
ho["score"] = (X * SG * WT).sum(axis=1) / WT.sum()

B = cfg["book"]
U = cfg["universe"]["max_names"]
d = ho[ho.adtv_rank <= U].dropna(subset=["score"])
P = core.prepare(d)
sim = core.simulate(P, slots=B["slots"], entry_rank=B["entry_rank"],
                    exit_rank=B["exit_rank"],
                    rebalance_every=B["rebalance_every_signal_dates"],
                    max_per_sector=B["max_per_sector"], weighting=B["weighting"],
                    cash_rate=B["cash_rate"])
c, b = sim["curve"], sim["bench_curve"]
exr = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
PER_YR = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
ir = float(exr.mean() / (exr.std(ddof=1) + 1e-12) * np.sqrt(PER_YR))
dts = pd.DatetimeIndex(P["dates"][1:])
ye = {int(y): float(np.prod(1 + exr[dts.year == y]) - 1) for y in np.unique(dts.year)}

H = cfg["label_horizon_sessions"]
ic, ic_t, nd = core.rank_ic(d, "score", f"y{H}")
ic21, ic21_t, _ = core.rank_ic(d, "score", "y21")
tk, tk_t, _ = core.topk_excess(d, "score", f"y{H}", k=B["slots"])
tk21, tk21_t, _ = core.topk_excess(d, "score", "y21", k=B["slots"])


def quintile(dd, ycol, q=5):
    sp = []
    for _, g in dd.groupby("date", sort=False):
        g = g.dropna(subset=["score", ycol])
        if len(g) < 100:
            continue
        k = max(len(g) // q, 5)
        o = g.sort_values("score", ascending=False)
        sp.append(float(o[ycol].head(k).mean() - o[ycol].tail(k).mean()))
    sp = np.array(sp)
    return float(sp.mean()), float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))), len(sp)


qs, qt, qn = quintile(d, f"y{H}")
qs21, qt21, _ = quintile(d, "y21")

# reference null on the same window -- not a selection step
rng = np.random.default_rng(99)
null_ex, null_q = [], []
sc = d["score"].to_numpy().copy()
dd = d.copy()
starts = dd.groupby("date", sort=False).indices
for _ in range(200):
    s2 = sc.copy()
    for idx in starts.values():
        s2[idx] = rng.permutation(s2[idx])
    dd["score"] = s2
    P2 = core.prepare(dd)
    sm = core.simulate(P2, slots=B["slots"], entry_rank=B["entry_rank"],
                       exit_rank=B["exit_rank"],
                       rebalance_every=B["rebalance_every_signal_dates"],
                       max_per_sector=B["max_per_sector"])
    null_ex.append(sm["ann"] - sm["bench_ann"])
    if len(null_q) < 60:
        null_q.append(quintile(dd, f"y{H}")[0])
null_ex = np.array(null_ex); null_q = np.array(null_q)

res = {
    "config_sha256": cfg["config_sha256"],
    "holdout": {"start": seal["holdout_start"], "end": seal["holdout_end"],
                "signal_dates": int(ho.date.nunique()), "rows": int(len(ho)),
                "sha256": seal["sha256_holdout"]},
    "book_net_of_costs": {
        "annualised_return": sim["ann"], "benchmark_annualised": sim["bench_ann"],
        "excess_annualised": sim["ann"] - sim["bench_ann"],
        "information_ratio": ir, "sharpe": sim["sharpe"],
        "max_drawdown": sim["maxdd"], "cost_drag_annualised": sim["cost_drag_ann"],
        "cash_share": sim["cash_share"],
        "median_hold_sessions": sim["median_hold_sessions"],
        "mean_hold_sessions": sim["mean_hold_sessions"],
        "closed_positions": sim["n_closed"], "periods": sim["n_periods"],
        "total_return": sim["final"] - 1.0,
        "benchmark_total_return": float(b[-1] - 1.0),
        "excess_by_calendar_year": ye},
    "ranking": {"rank_ic_h42": ic, "rank_ic_t_h42": ic_t, "dates": nd,
                "rank_ic_h21": ic21, "rank_ic_t_h21": ic21_t,
                "top10_excess_h42": tk, "top10_excess_t_h42": tk_t,
                "top10_excess_h21": tk21, "top10_excess_t_h21": tk21_t,
                "quintile_spread_h42": qs, "quintile_spread_t_h42": qt,
                "quintile_spread_h21": qs21, "quintile_spread_t_h21": qt21},
    "shuffled_score_null": {
        "draws_book": int(len(null_ex)),
        "excess_mean": float(null_ex.mean()), "excess_sd": float(null_ex.std(ddof=1)),
        "excess_p95": float(np.percentile(null_ex, 95)),
        "p_value_excess": float((null_ex >= (sim["ann"] - sim["bench_ann"])).mean()),
        "draws_quintile": int(len(null_q)),
        "quintile_mean": float(null_q.mean()), "quintile_sd": float(null_q.std(ddof=1)),
        "p_value_quintile": float((null_q >= qs).mean())},
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
