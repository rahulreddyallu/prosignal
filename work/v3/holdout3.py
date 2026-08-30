"""THE SEALED HOLDOUT for v3. One run per window, no parameters after this.

Applies FROZEN_V3.json blind. Nothing is fitted here -- signs, theme weights,
the cap, the floor and the book all come out of the frozen file, whose sha256
was written before this script was ever executed.

Window A is the recent 18 months and is the SECOND configuration scored on it
(v2 was the first), so the multiple-testing count is 2 and it is stated.
Window B is the 2021-07 to 2022-12 era window, which no search has touched.
"""
from __future__ import annotations
import sys, json, hashlib, os, argparse, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, composite as CO

CACHE = "/home/claude/psr/cache"


def score_frozen(df, cfg):
    themes = cfg["themes"]
    cols = [f["name"] for t in themes.values() for f in t["factors"]]
    R = CO.rank_block(df, cols)
    sub, names = {}, []
    for tname, spec in themes.items():
        fs = [f["name"] for f in spec["factors"]]
        sg = {f["name"]: float(f["sign"]) for f in spec["factors"]}
        s = CO.theme_subscore(R, fs, sg, df["date"])
        sub[tname] = s
        names.append(tname)
    M = np.column_stack([sub[t].to_numpy("float64") for t in names])
    W = np.array([themes[t]["weight"] for t in names])
    ok = np.isfinite(M)
    num = np.nansum(np.where(ok, M * W, 0.0), axis=1)
    den = np.where(ok, W, 0.0).sum(axis=1)
    cnt = ok.sum(axis=1)
    score = np.where((den > 0) & (cnt >= cfg["min_themes"]),
                     num / np.maximum(den, 1e-12), np.nan)
    npos = ((M > 0) & ok).sum(axis=1)
    return score, npos, cnt, {t: sub[t] for t in names}


def evaluate(panel, cfg, label, tag, n_null=200, seed=17):
    d = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    score, npos, nth, sub = score_frozen(d, cfg)
    d = d.assign(score=score, npos=npos, n_themes=nth)
    d = d.dropna(subset=["score"])
    u = cfg["universe"]["max_names"]
    du = d[d.adtv_rank <= u]
    gate = (du["dist_200dma"] > 0) & (du["npos"] >= 3)
    dd = du[gate.fillna(False).to_numpy()]
    per_date = dd.groupby("date").size().reindex(
        sorted(du["date"].unique()), fill_value=0)

    B = cfg["book"]
    P = core.prepare(dd)
    sim = core.simulate(P, slots=B["slots"], entry_rank=B["entry_rank"],
                        exit_rank=B["exit_rank"],
                        rebalance_every=B["rebalance_every"],
                        max_per_sector=B["max_per_sector"],
                        weighting=B["weighting"])
    c, b = sim["curve"], sim["bench_curve"]
    ex = np.diff(c) / c[:-1] - np.diff(b) / b[:-1]
    per_yr = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
    ir = float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(per_yr))
    dts = pd.DatetimeIndex(P["dates"][1:])
    ye = {int(y): float(np.prod(1 + ex[dts.year == y]) - 1) for y in np.unique(dts.year)}

    def stats(frame, col):
        ics, tk, qs = [], [], []
        for _, g in frame.groupby("date", sort=True):
            g = g.dropna(subset=["score", col])
            if len(g) < 60:
                continue
            a = g["score"].rank().to_numpy(); y = g[col].rank().to_numpy()
            if a.std() > 1e-9 and y.std() > 1e-9:
                ics.append(float(np.corrcoef(a, y)[0, 1]))
            ss, yy = g["score"].to_numpy(), g[col].to_numpy()
            k = min(B["slots"], len(g) - 1)
            tk.append(float(yy[np.argpartition(-ss, k)[:k]].mean() - yy.mean()))
            kk = max(len(g) // 5, 5); o = np.argsort(-ss)
            qs.append(float(yy[o[:kk]].mean() - yy[o[-kk:]].mean()))
        t = lambda z: float(np.mean(z) / (np.std(z, ddof=1) / np.sqrt(len(z)))) \
            if len(z) > 5 else float("nan")
        return {"ic": float(np.mean(ics)), "ic_t": t(np.array(ics)),
                "topk": float(np.mean(tk)), "topk_t": t(np.array(tk)),
                "quintile": float(np.mean(qs)), "quintile_t": t(np.array(qs)),
                "dates": len(qs)}

    rank21 = stats(du, "y21")
    rank42 = stats(du, "y42")

    rng = np.random.default_rng(seed)
    idx = dd.groupby("date", sort=False).indices
    base_sc = dd["score"].to_numpy("float64").copy()
    tmp = dd.copy()
    null_ex, null_q = [], []
    for i in range(n_null):
        s2 = base_sc.copy()
        for ii in idx.values():
            s2[ii] = rng.permutation(s2[ii])
        tmp["score"] = s2
        P2 = core.prepare(tmp)
        sm = core.simulate(P2, slots=B["slots"], entry_rank=B["entry_rank"],
                           exit_rank=B["exit_rank"],
                           rebalance_every=B["rebalance_every"],
                           max_per_sector=B["max_per_sector"])
        null_ex.append(sm["ann"] - sm["bench_ann"])
        if i < 60:
            null_q.append(stats(tmp, "y21")["quintile"])
    null_ex = np.asarray(null_ex); null_q = np.asarray(null_q)
    real_ex = sim["ann"] - sim["bench_ann"]

    theme_ic = {}
    for t, s in sub.items():
        g = pd.DataFrame({"date": panel.sort_values(["date", "symbol"])
                          .reset_index(drop=True)["date"],
                          "s": s.to_numpy(),
                          "y": panel.sort_values(["date", "symbol"])
                          .reset_index(drop=True)["y21"]}).dropna()
        ics = []
        for _, gg in g.groupby("date", sort=True):
            if len(gg) < 60:
                continue
            a, y = gg["s"].rank().to_numpy(), gg["y"].rank().to_numpy()
            if a.std() > 1e-9 and y.std() > 1e-9:
                ics.append(float(np.corrcoef(a, y)[0, 1]))
        ics = np.array(ics)
        theme_ic[t] = {"ic": float(ics.mean()) if len(ics) else None,
                       "ic_t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))))
                       if len(ics) > 5 else None,
                       "weight": cfg["themes"][t]["weight"]}

    return {
        "window": tag, "label_horizon": label,
        "config_sha256": cfg["config_sha256"],
        "signal_dates": int(panel["date"].nunique()),
        "book_net_of_costs": {
            "annualised_return": sim["ann"], "benchmark_annualised": sim["bench_ann"],
            "excess_annualised": real_ex, "information_ratio": ir,
            "sharpe": sim["sharpe"], "max_drawdown": sim["maxdd"],
            "cost_drag_annualised": sim["cost_drag_ann"],
            "cash_share": sim["cash_share"],
            "median_hold_sessions": sim["median_hold_sessions"],
            "closed_positions": sim["n_closed"], "periods": sim["n_periods"],
            "total_return": sim["final"] - 1.0,
            "benchmark_total_return": float(b[-1] - 1.0),
            "excess_by_calendar_year": ye},
        "absolute_floor": {
            "median_names_clearing": float(per_date.median()),
            "min_names_clearing": int(per_date.min()),
            "dates_short_of_a_full_book": int((per_date < B["slots"]).sum()),
            "dates_with_no_name": int((per_date == 0).sum()),
            "dates": int(len(per_date))},
        "ranking_h21": rank21, "ranking_h42": rank42,
        "theme_ic_h21": theme_ic,
        "shuffled_score_null": {
            "draws_book": int(len(null_ex)), "excess_mean": float(null_ex.mean()),
            "excess_sd": float(null_ex.std(ddof=1)),
            "excess_p95": float(np.percentile(null_ex, 95)),
            "p_value_excess": float((null_ex >= real_ex).mean()),
            "draws_quintile": int(len(null_q)),
            "quintile_mean": float(null_q.mean()),
            "p_value_quintile": float((null_q >= rank21["quintile"]).mean())},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["A", "B"], required=True)
    a = ap.parse_args()
    out = f"{CACHE}/HOLDOUT_V3_{a.window}.json"
    if os.path.exists(out):
        print(f"WINDOW {a.window} ALREADY RUN -- the result stands.")
        print(open(out).read()); sys.exit(0)
    cfg = json.load(open(f"{CACHE}/FROZEN_V3.json"))
    seal = json.load(open(f"{CACHE}/SEAL2.json"))
    fn = "SEALED_A_RECENT" if a.window == "A" else "SEALED_B_ERA"
    assert hashlib.sha256(open(f"{CACHE}/{fn}.parquet", "rb").read()).hexdigest() \
        == seal["A_recent" if a.window == "A" else "B_era"]["sha256"], \
        "the sealed file changed since it was sealed"
    panel = pd.read_parquet(f"{CACHE}/{fn}.parquet")
    res = evaluate(panel, cfg, 21, a.window)
    res["seal"] = seal["A_recent" if a.window == "A" else "B_era"]
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
