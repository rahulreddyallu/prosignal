"""Combination search. TRAIN ONLY -- the sealed holdout is not importable here.

Expanding walk-forward: fit on everything up to a block, predict the block,
never the other way round. Purge = label horizon, embargo = 21 sessions, both
measured in signal steps. Out-of-sample predictions are pooled and scored by
rank IC, top-k excess and a net-of-cost portfolio simulation.
"""
from __future__ import annotations
import sys, json, itertools, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.decomposition import PCA
import xgboost as xgb

RNG = np.random.default_rng(7)
EMBARGO = 21
STEP = 5

SCREEN = pd.read_csv("/home/claude/psr/cache/screen_train.csv")


def kept(h):
    return list(SCREEN[(SCREEN.h == h) & SCREEN.keep].sort_values("t", key=abs, ascending=False).factor)


FSETS = {
    "kept_h": None,                                    # resolved per horizon
    "swing_union": sorted(set(SCREEN[(SCREEN.h.isin([10, 21])) & SCREEN.keep].factor)),
    "any_union": sorted(set(SCREEN[SCREEN.keep].factor)),
    "incumbent": ["mom_6_1", "resid_rev_21", "downside_vol_60", "beta_126", "amihud_60",
                  "log_adtv_60", "max_dd_120", "prox_52w", "max5_21", "resid_mom_252_21",
                  "idio_vol_126", "idio_skew_126", "deliv_pct_60", "deliv_trend"],
}


def decorrelated(df, cands, rcols, max_r=0.75, top=14):
    """Greedy: take factors in |t| order, skip one that correlates above max_r
    with anything already taken."""
    keep = []
    M = df[[c + "_r" for c in cands]].to_numpy("float32")
    for j, c in enumerate(cands):
        ok = True
        for k in keep:
            i = cands.index(k)
            a, b = M[:, j], M[:, i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 1000 and abs(np.corrcoef(a[m], b[m])[0, 1]) > max_r:
                ok = False
                break
        if ok:
            keep.append(c)
        if len(keep) >= top:
            break
    return keep


# ------------------------------------------------------------------ models
def fit_predict(method, Xtr, ytr, wtr, Xte, seed=0):
    if method == "equal":
        # SIGN-ORIENTED. Averaging raw ranks across factors whose ICs point in
        # opposite directions (volatility down, momentum up) cancels them and
        # produces a composite with no defined direction -- which is what the
        # first pass of this search was scoring.
        sgn = np.sign([np.corrcoef(Xtr[:, j], ytr)[0, 1] for j in range(Xtr.shape[1])])
        sgn = np.nan_to_num(sgn, nan=1.0)
        sgn[sgn == 0] = 1.0
        return (Xte * sgn).mean(axis=1)
    if method == "ic_weighted":
        ic = np.array([np.corrcoef(Xtr[:, j], ytr)[0, 1] if np.isfinite(Xtr[:, j]).all() else 0.0
                       for j in range(Xtr.shape[1])])
        ic = np.nan_to_num(ic)
        w = ic / (np.abs(ic).sum() + 1e-12)
        return Xte @ w
    if method.startswith("ridge"):
        a = float(method.split("_")[1])
        m = Ridge(alpha=a).fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    if method.startswith("lasso"):
        a = float(method.split("_")[1])
        m = Lasso(alpha=a, max_iter=4000).fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    if method.startswith("enet"):
        _, a, l1 = method.split("_")
        m = ElasticNet(alpha=float(a), l1_ratio=float(l1), max_iter=4000).fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    if method.startswith("pcr"):
        k = int(method.split("_")[1])
        p = PCA(n_components=min(k, Xtr.shape[1])).fit(Xtr)
        m = LinearRegression().fit(p.transform(Xtr), ytr, sample_weight=wtr)
        return m.predict(p.transform(Xte))
    if method.startswith("xgb"):
        parts = method.split("_")
        depth, n_est, lr = int(parts[1]), int(parts[2]), float(parts[3])
        sub = float(parts[4]) if len(parts) > 4 else 0.8
        m = xgb.XGBRegressor(max_depth=depth, n_estimators=n_est, learning_rate=lr,
                             subsample=sub, colsample_bytree=0.7, reg_lambda=5.0,
                             min_child_weight=50, tree_method="hist", n_jobs=2,
                             random_state=seed, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    raise ValueError(method)


# ------------------------------------------------------------------ WF loop
def walk_forward(df, feats, method, h, n_blocks=6, target="rank", seed=0):
    dates = np.array(sorted(df["date"].unique()))
    D = len(dates)
    gap = int(np.ceil((h + EMBARGO) / STEP))
    start = int(D * 0.35)
    bounds = np.linspace(start, D, n_blocks + 1).astype(int)
    cols = [f + "_r" for f in feats]
    y_col = f"y{h}"
    preds = []
    for bi in range(n_blocks):
        lo, hi = bounds[bi], bounds[bi + 1]
        te_d = dates[lo:hi]
        tr_d = dates[: max(lo - gap, 0)]
        if len(tr_d) < 40 or len(te_d) < 3:
            continue
        tr = df[df["date"].isin(tr_d)]
        te = df[df["date"].isin(te_d)]
        tr = tr[np.isfinite(tr[y_col])]
        if len(tr) < 3000 or len(te) == 0:
            continue
        Xtr = np.nan_to_num(tr[cols].to_numpy("float32"))
        Xte = np.nan_to_num(te[cols].to_numpy("float32"))
        ytr = (tr.groupby("date")[y_col].rank(pct=True).to_numpy("float32") - 0.5) * 2 \
            if target == "rank" else tr[y_col].to_numpy("float32")
        w = np.full(len(tr), 1.0, "float32")
        p = fit_predict(method, Xtr, ytr, w, Xte, seed=seed)
        keep_cols = list(dict.fromkeys(["date", "symbol", "sector", "adtv", "adtv_rank",
                                        "atr_pct", "y5", "b5", y_col, f"b{h}"]))
        o = te[keep_cols].copy()
        o["score"] = p
        preds.append(o)
    if not preds:
        return None
    return pd.concat(preds, ignore_index=True)


def evaluate(oos, h, *, slots=8, entry=8, exit_=16, rebal=None, universe=750,
             weighting="equal", score_floor=None, max_per_sector=2, regime_ok=None):
    d = oos[oos["adtv_rank"] <= universe].copy() if universe else oos
    if len(d) < 1000:
        return None
    ic, ic_t, nd = core.rank_ic(d, "score", f"y{h}")
    ex, ex_t, _ = core.topk_excess(d, "score", f"y{h}", k=slots)
    rb = rebal if rebal else max(int(round(h / STEP)), 1)
    sim = core.simulate(d, slots=slots, entry_rank=entry, exit_rank=exit_,
                        rebalance_every=rb, weighting=weighting,
                        score_floor=score_floor, max_per_sector=max_per_sector,
                        regime_ok=regime_ok)
    c, b = sim["curve"], sim["bench_curve"]
    rb_ = np.diff(c) / c[:-1]
    rbm = np.diff(b) / b[:-1]
    exr = rb_ - rbm
    per_yr = core.SESSIONS_PER_YEAR / core.STEP_SESSIONS
    ir = (exr.mean() / (exr.std(ddof=1) + 1e-12)) * np.sqrt(per_yr) if len(exr) > 3 else np.nan
    # per-calendar-year excess, as a consistency check
    dts = np.array(sorted(d["date"].unique()))[1:]
    yr = pd.Series(exr, index=pd.DatetimeIndex(dts)).groupby(
        pd.DatetimeIndex(dts).year).apply(lambda s: float((1 + s).prod() - 1))
    return {"ic": ic, "ic_t": ic_t, "n_dates": nd, "topk_excess": ex, "topk_t": ex_t,
            "ann": sim["ann"], "bench_ann": sim["bench_ann"],
            "excess_ann": float(sim["ann"] - sim["bench_ann"]),
            "ir": float(ir), "sharpe": sim["sharpe"],
            "maxdd": sim["maxdd"], "cost_drag": sim["cost_drag_ann"],
            "cash": sim["cash_share"], "n_per": sim["n_periods"],
            "yr_pos": float((yr > 0).mean()), "yr_min": float(yr.min()),
            "yr_med": float(yr.median()), "n_years": int(len(yr))}
