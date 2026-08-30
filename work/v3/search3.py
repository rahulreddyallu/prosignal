"""Two-level composite search. TRAIN ONLY.

EACH THEME IS COMBINED AGAINST THE HORIZON IT ACTUALLY WORKS AT. Forcing every
theme onto one label was the first version's mistake and it was expensive:
oriented against a 42-session label, the reversal sub-score came out
ANTI-predictive (IC -0.007, top-decile excess -0.79% at t -3.96), because
reversal is a two-week effect and at forty-two sessions its sign has already
turned over. The screen says so directly -- reversal clears at h=10 six times
and at h=42 not once.

Sub-scores are within-date RANKS, so they are commensurable whatever horizon
oriented them, and the book's holding period is set by the rank band rather
than by any label. The blend and its weights are judged at h=21, which is what
the book actually holds for.

LEVEL-2 WEIGHTS ARE FOLD-LOCAL. Computing them once on the pooled out-of-sample
series and then applying them to that same series is a leak -- small, but it is
the kind that makes a composite look better than it will be.
"""
from __future__ import annotations
import sys, json, pickle, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
warnings.filterwarnings("ignore")
import core, themes as TH, guard as G, composite as CO, survivors as SV
import xgboost as xgb

EMBARGO = 21
STEP = 5
H_BLEND = 21           # what the book holds for, and what level 2 is judged at
N_BLOCKS = 8
START_FRAC = 0.20

#: The horizon each theme is oriented and combined against, taken from where it
#: clears the placebo screen rather than chosen: reversal and ownership are
#: short-horizon effects, momentum is a long one, risk and quality sit between.
THEME_HORIZON = {"momentum": 42, "reversal": 10, "risk": 21,
                 "ownership": 10, "quality": 21, "value": 63,
                 "liquidity": 21, "seasonality": 63}


class Frame:
    def __init__(self, tr, by_theme, horizons=(10, 21, 42, 63)):
        tr = tr.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.df = tr
        self.by_theme = {t: fs for t, fs in by_theme.items() if fs}
        self.cols = [f for fs in self.by_theme.values() for f in fs]
        G.assert_no_lookahead(self.cols)
        self.R = CO.rank_block(tr, self.cols)
        self.dates_col = tr["date"]
        self.y = {h: tr[f"y{h}"] for h in horizons}
        self.yr = {h: tr.groupby("date")[f"y{h}"].rank(pct=True) * 2 - 1
                   for h in horizons}
        d = np.array(sorted(tr["date"].unique()))
        self.dates = d
        self.di = pd.Series(np.arange(len(d)), index=pd.DatetimeIndex(d)) \
            .reindex(tr["date"]).to_numpy()
        gap = int(np.ceil((max(horizons) + EMBARGO) / STEP))
        b = np.linspace(int(len(d) * START_FRAC), len(d), N_BLOCKS + 1).astype(int)
        self.folds = []
        for i in range(N_BLOCKS):
            lo, hi = b[i], b[i + 1]
            trn = np.where(self.di < max(lo - gap, 0))[0]
            te = np.where((self.di >= lo) & (self.di < hi))[0]
            if len(trn) > 5000 and len(te) > 500:
                self.folds.append((trn, te))
        self.val_start = pd.Timestamp(d[b[0]])
        print(f"{len(self.cols)} factors | {len(self.by_theme)} themes | "
              f"{len(d)} dates | {len(self.folds)} folds | validation from "
              f"{self.val_start.date()}", flush=True)


def _fit_theme(fr, cols, trn, method, h):
    """Signs and within-theme weights, from the training rows only."""
    Rtr = fr.R.iloc[trn][cols]
    yr = fr.yr[h].iloc[trn]
    ok = np.isfinite(yr.to_numpy("float64"))
    Rtr, yr = Rtr[ok], yr[ok]
    signs = CO.orient(Rtr, yr, cols)
    if method == "equal":
        return signs, None, None
    y = yr.to_numpy("float64")
    if method == "icw":
        w = {}
        for c in cols:
            v = Rtr[c].to_numpy("float64")
            m = np.isfinite(v)
            ic = np.corrcoef(v[m], y[m])[0, 1] if m.sum() > 500 else 0.0
            w[c] = abs(ic) if np.isfinite(ic) else 0.0
        return signs, w, None
    if method == "ridge":
        A = np.nan_to_num(Rtr.to_numpy("float64"))
        coef = np.linalg.solve(A.T @ A + 2000.0 * np.eye(A.shape[1]), A.T @ y)
        return ({c: (1.0 if coef[i] >= 0 else -1.0) for i, c in enumerate(cols)},
                {c: abs(coef[i]) for i, c in enumerate(cols)}, None)
    if method == "xgb":
        if len(cols) < 5:
            return signs, None, None
        A = np.nan_to_num(Rtr.to_numpy("float32"))
        mdl = xgb.XGBRegressor(max_depth=2, n_estimators=200, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
                               min_child_weight=100, tree_method="hist", n_jobs=2,
                               random_state=0, verbosity=0)
        mdl.fit(A, y)
        return None, None, mdl
    raise ValueError(method)


def _theme_contribution(fr, sub_tr, idx, h, k_frac=0.10):
    """A theme's standalone top-k excess and IC t on the TRAINING rows."""
    v = sub_tr[idx]
    y = fr.y[h].to_numpy("float64")[idx]
    d = fr.dates_col.to_numpy()[idx]
    ex, ics = [], []
    for dt_ in np.unique(d):
        m = d == dt_
        a, b = v[m], y[m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        a, b = a[ok], b[ok]
        kk = max(int(len(a) * k_frac), 5)
        top = np.argpartition(-a, kk)[:kk]
        ex.append(b[top].mean() - b.mean())
        ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        if ra.std() > 1e-9 and rb.std() > 1e-9:
            ics.append(np.corrcoef(ra, rb)[0, 1])
    ex, ics = np.array(ex), np.array(ics)
    t = lambda z: float(z.mean() / (z.std(ddof=1) / np.sqrt(len(z)))) if len(z) > 5 else 0.0
    return {"topk": float(ex.mean()) if len(ex) else 0.0,
            "topk_t": t(ex), "ic_t": t(ics)}


def stage_a(fr, method, theme_horizon=THEME_HORIZON, h_blend=H_BLEND):
    """Out-of-sample theme sub-scores, plus FOLD-LOCAL level-2 raw weights."""
    n = len(fr.df)
    sub = {t: np.full(n, np.nan) for t in fr.by_theme}
    fold_w = []
    for fi, (trn, te) in enumerate(fr.folds):
        wrow = {}
        for t, cols in fr.by_theme.items():
            h = theme_horizon.get(t, h_blend)
            signs, wts, mdl = _fit_theme(fr, cols, trn, method, h)
            if mdl is not None:
                A = np.nan_to_num(fr.R.iloc[te][cols].to_numpy("float32"))
                raw = mdl.predict(A).astype("float64")
                cnt = np.isfinite(fr.R.iloc[te][cols].to_numpy("float64")).sum(axis=1)
                raw = np.where(cnt >= CO.MIN_FACTORS_PER_THEME, raw, np.nan)
                s_te = pd.Series(raw, index=fr.R.index[te]).groupby(
                    fr.dates_col.iloc[te]).transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
                A2 = np.nan_to_num(fr.R.iloc[trn][cols].to_numpy("float32"))
                raw2 = mdl.predict(A2).astype("float64")
                s_tr = pd.Series(raw2, index=fr.R.index[trn]).groupby(
                    fr.dates_col.iloc[trn]).transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
            else:
                s_te = CO.theme_subscore(fr.R.iloc[te], cols, signs,
                                         fr.dates_col.iloc[te], weights=wts)
                s_tr = CO.theme_subscore(fr.R.iloc[trn], cols, signs,
                                         fr.dates_col.iloc[trn], weights=wts)
            sub[t][te] = s_te.to_numpy()
            full = np.full(n, np.nan)
            full[trn] = s_tr.to_numpy()
            wrow[t] = _theme_contribution(fr, full, trn, h_blend)
        fold_w.append(wrow)
    for t in sub:
        s = pd.Series(sub[t], index=fr.df.index)
        sub[t] = s.groupby(fr.dates_col).transform(
            lambda x: (x.rank(pct=True) - 0.5) * 2.0).to_numpy()
    return sub, fold_w


if __name__ == "__main__":
    by_theme, cut = SV.admitted(verbose=False)
    tr = core.load_train2()
    fr = Frame(tr, by_theme)
    print("theme horizons:", {t: THEME_HORIZON[t] for t in fr.by_theme}, flush=True)
    out = {}
    for method in ("equal", "icw", "ridge", "xgb"):
        sub, fold_w = stage_a(fr, method)
        out[method] = {"sub": {t: v.astype("float32") for t, v in sub.items()},
                       "fold_w": fold_w}
        print(f"  stage A {method}", flush=True)
    with open("/home/claude/psr/cache/stageA2.pkl", "wb") as f:
        pickle.dump({"m": out, "by_theme": by_theme,
                     "folds": [(a.astype("int32"), b.astype("int32"))
                               for a, b in fr.folds],
                     "theme_horizon": THEME_HORIZON}, f)
    print("cached stageA2")
