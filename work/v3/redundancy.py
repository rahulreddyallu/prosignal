"""Critical issue 3: are these factors measuring the same thing twice?

Correlations are taken on the SECTOR-NEUTRAL WITHIN-DATE RANKS, which is what
the composite actually sums, and pooled across dates. A pair correlated over
time but not within a cross-section does not double-count; a pair correlated
within the cross-section does, whatever it looks like on a time series.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, themes as TH, guard as G

THRESH = 0.80


def build(tr, cols):
    R = core.sector_neutral_ranks(tr, cols)
    R.columns = cols
    return R, R.corr(min_periods=2000)


def link(C, cols, thresh=THRESH):
    seen, out = set(), []
    for c in cols:
        if c in seen:
            continue
        grp, frontier = {c}, [c]
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
    sc = pd.read_csv("/home/claude/psr/cache/screen2.csv")
    st = pd.read_csv("/home/claude/psr/cache/stability.csv")
    cleared = set(sc[sc.keep].factor) & set(st[st.same_sign].factor)
    cols = [c for c in G.factor_columns(tr) if c in cleared]
    print(f"{len(cols)} factors cleared the screen AND held their sign in both halves\n")
    R, C = build(tr, cols)
    C.to_csv("/home/claude/psr/cache/corr.csv")

    print("=== the pairs the brief named ===")
    for a, b in (("resid_mom_252_21", "mom_6_1"), ("mom_6_1", "rev_1m_scaled"),
                 ("resid_mom_252_21", "rev_1m_scaled"),
                 ("deliv_pct_60", "deliv_trend"), ("deliv_z_21", "deliv_trend"),
                 ("deliv_z_21", "deliv_chg_5"),
                 ("prox_52w", "prox_52w_now"),
                 ("voladj_mom_6_1", "voladj_mom_12_1")):
        Rall, Call = build(tr, [x for x in (a, b) if x in tr.columns])
        if a in Call.index and b in Call.index:
            print(f"  {a:22s} vs {b:22s} rho = {Call.at[a, b]:+.3f}")
        else:
            print(f"  {a:22s} vs {b:22s} -- at least one did not clear the screen")

    print("\n=== clusters at |rho| >= 0.80, among survivors ===")
    for g in link(C, cols):
        if len(g) > 1:
            th = {TH.FACTOR_THEME[x] for x in g}
            print(f"  [{','.join(sorted(th))}] {g}")
    print("\n=== every |rho| >= 0.60 pair among survivors ===")
    seen = set()
    for a in cols:
        for b in cols:
            if a >= b or (a, b) in seen:
                continue
            r = C.at[a, b]
            if np.isfinite(r) and abs(r) >= 0.60:
                print(f"  {a:22s} {b:22s} {r:+.3f}"
                      f"   [{TH.FACTOR_THEME[a]} / {TH.FACTOR_THEME[b]}]")
            seen.add((a, b))
