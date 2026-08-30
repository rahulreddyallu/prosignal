"""Placebo screen + redundancy analysis, per theme. TRAIN ONLY."""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, screen as SC, themes as TH, guard as G

META = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5", "mcap", "fund_age_days"}


def factor_cols(df):
    return G.factor_columns(df)


def redundancy(df, cols, min_obs=2000):
    """Within-date correlation of the sector-neutral ranks, pooled.

    Pooled ACROSS dates but computed on WITHIN-DATE ranks, so it measures what
    the composite actually double-counts rather than what moves together over
    time.
    """
    R = core.sector_neutral_ranks(df, cols)
    R.columns = cols
    C = R.corr(min_periods=min_obs)
    return C, R


def clusters(C, cols, thresh=0.80):
    """Greedy single-link grouping at |rho| >= thresh."""
    seen, out = set(), []
    for c in cols:
        if c in seen:
            continue
        grp = {c}
        frontier = [c]
        while frontier:
            x = frontier.pop()
            for y in cols:
                if y in grp or y not in C.index or x not in C.index:
                    continue
                r = C.at[x, y]
                if np.isfinite(r) and abs(r) >= thresh:
                    grp.add(y); frontier.append(y)
        seen |= grp
        out.append(sorted(grp))
    return out


if __name__ == "__main__":
    tr = core.load_train2()
    cols = factor_cols(tr)
    print(f"{len(cols)} factors, {tr['date'].nunique()} dates, {len(tr):,} rows\n",
          flush=True)
    res = SC.run(horizons=(10, 21, 42, 63), panel=tr[["date", "symbol", "sector",
                 "y10", "y21", "y42", "y63"] + cols], tag="train2")
    res["theme"] = res["factor"].map(TH.FACTOR_THEME)
    res.to_csv("/home/claude/psr/cache/screen2.csv", index=False)
    for h in sorted(res.h.unique()):
        s = res[res.h == h]
        print(f"h={h}: kept {int(s.keep.sum())}/{len(s)}")
        for t in TH.THEMES:
            st = s[s.theme == t]
            if len(st):
                k = st[st.keep].sort_values("t", key=abs, ascending=False)
                print(f"  {t:12s} {int(st.keep.sum()):2d}/{len(st):2d}  "
                      f"{', '.join(k.factor.head(6))}")
        print()
