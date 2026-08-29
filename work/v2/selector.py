"""Fast greedy forward selection, scored by walk-forward TOP-K excess.

An eight-to-twelve name book lives in the tail of the ranking, so the objective
is what the top k actually earn against the cross-section, not pooled IC. All
matrices are prepared once; each candidate costs one ridge solve.
"""
from __future__ import annotations
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core

EMBARGO = 21
STEP = 5


class Frame:
    def __init__(self, df, fcols, horizons=(10, 21, 42), start_frac=0.20, n_blocks=8):
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.df = df
        self.fcols = fcols
        R = core.sector_neutral_ranks(df, fcols)
        self.X = np.nan_to_num(R.to_numpy("float32"))
        self.col = {c: i for i, c in enumerate(fcols)}
        d = df["date"].to_numpy()
        self.dates = np.array(sorted(df["date"].unique()))
        self.di = pd.Series(np.arange(len(self.dates)),
                            index=pd.DatetimeIndex(self.dates)).reindex(df["date"]).to_numpy()
        _, s = np.unique(self.di, return_index=True)
        self.starts = np.sort(s); self.ends = np.append(self.starts[1:], len(df))
        self.Y = {h: df[f"y{h}"].to_numpy("float64") for h in horizons}
        # within-date label rank, the regression target
        self.Yr = {h: ((df.groupby("date")[f"y{h}"].rank(pct=True).to_numpy("float64") - 0.5) * 2)
                   for h in horizons}
        D = len(self.dates)
        self.folds = {}
        for h in horizons:
            gap = int(np.ceil((h + EMBARGO) / STEP))
            b = np.linspace(int(D * start_frac), D, n_blocks + 1).astype(int)
            f = []
            for i in range(n_blocks):
                lo, hi = b[i], b[i + 1]
                tr = self.di < max(lo - gap, 0)
                te = (self.di >= lo) & (self.di < hi)
                if tr.sum() > 5000 and te.sum() > 500:
                    f.append((np.where(tr)[0], np.where(te)[0]))
            self.folds[h] = f

    def oos_scores(self, feats, h, alpha=20.0):
        cols = [self.col[f] for f in feats]
        out = np.full(len(self.df), np.nan)
        for tr, te in self.folds[h]:
            m = np.isfinite(self.Y[h][tr])
            tr = tr[m]
            A = self.X[np.ix_(tr, cols)].astype("float64")
            y = self.Yr[h][tr]
            k = A.shape[1]
            G = A.T @ A + alpha * np.eye(k)
            w = np.linalg.solve(G, A.T @ y)
            out[te] = self.X[np.ix_(te, cols)].astype("float64") @ w
        return out

    def topk_excess(self, s, h, k=10):
        y = self.Y[h]
        ex = []
        for a, b in zip(self.starts, self.ends):
            ss, yy = s[a:b], y[a:b]
            m = np.isfinite(ss) & np.isfinite(yy)
            if m.sum() < 60:
                continue
            ss, yy = ss[m], yy[m]
            idx = np.argpartition(-ss, k)[:k]
            ex.append(yy[idx].mean() - yy.mean())
        ex = np.array(ex)
        if len(ex) < 8:
            return np.nan, np.nan, 0
        return float(ex.mean()), float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))), len(ex)

    def ic(self, s, h):
        y = self.Y[h]
        out = []
        for a, b in zip(self.starts, self.ends):
            ss, yy = s[a:b], y[a:b]
            m = np.isfinite(ss) & np.isfinite(yy)
            if m.sum() < 60:
                continue
            r1 = pd.Series(ss[m]).rank().to_numpy(); r2 = pd.Series(yy[m]).rank().to_numpy()
            out.append(np.corrcoef(r1, r2)[0, 1])
        out = np.array(out)
        return float(out.mean()), float(out.mean() / (out.std(ddof=1) / np.sqrt(len(out))))


def forward(fr: Frame, h, pool, max_f=16, k=10, alpha=20.0, min_gain=0.03, verbose=True):
    chosen, best, hist = [], -9e9, []
    t0 = time.time()
    while len(chosen) < max_f:
        cand = []
        for f in pool:
            if f in chosen:
                continue
            s = fr.oos_scores(chosen + [f], h, alpha)
            ex, t, n = fr.topk_excess(s, h, k)
            if np.isfinite(t):
                cand.append((t, ex, f))
        if not cand:
            break
        cand.sort(reverse=True)
        t, ex, f = cand[0]
        if t <= best + min_gain:
            break
        chosen.append(f); best = t
        hist.append({"n": len(chosen), "added": f, "topk": ex, "t": t})
        if verbose:
            print(f"    h={h} +{f:20s} topk={ex:.4f} t={t:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return chosen, hist
