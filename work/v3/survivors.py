"""Which factors are allowed into the composite, and the rule is stated first.

A factor is admitted when ALL THREE hold:
  1. it clears its own placebo-alignment screen at some tested horizon;
  2. its information coefficient has the SAME SIGN in both halves of its own
     life -- clearing a screen on one regime is not evidence it is the same
     thing twice;
  3. it has at least MIN_DATES signal dates of evidence, which at a 42-session
     label and a weekly stride is about fourteen independent windows.

Rule (3) is what removes `gross_profitability`: it clears the screen at t -4.3,
holds its sign in both halves, and its entire life is 63 dates inside 2023-24 --
one regime, and a sign opposite to the literature it comes from. Keeping it
would put a 2023 subsample artefact in the shipped model wearing the name of a
published anomaly.

Within a theme, factors correlated at |rho| >= 0.80 on the within-date ranks are
one measurement, and only the one with the stronger both-halves evidence is
kept.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import themes as TH, guard as G

MIN_DATES = 120
DUP_RHO = 0.80


def admitted(screen_csv="/home/claude/psr/cache/screen2.csv",
             stability_csv="/home/claude/psr/cache/stability.csv",
             corr_csv="/home/claude/psr/cache/corr.csv",
             min_dates=MIN_DATES, dup_rho=DUP_RHO, verbose=True):
    sc = pd.read_csv(screen_csv)
    st = pd.read_csv(stability_csv).set_index("factor")
    cleared = set(sc[sc.keep].factor)
    keep, cut = [], {}
    for f in sorted(cleared):
        if f not in TH.FACTOR_THEME:
            cut[f] = "not a declared factor"
            continue
        if f not in st.index:
            cut[f] = "no stability record"
            continue
        r = st.loc[f]
        if not bool(r.same_sign):
            cut[f] = f"sign flips between halves ({r.ic_h1:+.4f} -> {r.ic_h2:+.4f})"
        elif int(r.n) < min_dates:
            cut[f] = f"only {int(r.n)} signal dates, under the {min_dates} floor"
        else:
            keep.append(f)

    C = pd.read_csv(corr_csv, index_col=0)
    strength = {f: abs(float(st.loc[f, "t"])) for f in keep if f in st.index}
    # ITERATIVE, NOT ONE PASS. A single sweep drops A because it duplicates B,
    # then drops B because it duplicates C -- and A is gone for a reason that no
    # longer exists. Measured on this set: `dist_200dma` and `mom_6_1` were both
    # removed in favour of `trend_slope_120`, which was itself removed two
    # comparisons later. So the strongest surviving duplicate pair is resolved
    # one at a time, and the survivors are pairwise below the threshold by
    # construction rather than by the order the loop happened to run in.
    while True:
        worst = None
        for a in keep:
            for b in keep:
                if a >= b or TH.FACTOR_THEME[a] != TH.FACTOR_THEME[b]:
                    continue
                if a not in C.index or b not in C.index:
                    continue
                r = C.at[a, b]
                if np.isfinite(r) and abs(r) >= dup_rho:
                    if worst is None or abs(r) > worst[0]:
                        worst = (abs(r), a, b)
        if worst is None:
            break
        r, a, b = worst
        weak = a if strength.get(a, 0) < strength.get(b, 0) else b
        strong = b if weak == a else a
        keep = [f for f in keep if f != weak]
        cut[weak] = (f"|rho| {r:.2f} with {strong} inside "
                     f"{TH.FACTOR_THEME[strong]}; the weaker of the pair "
                     f"(|t| {strength.get(weak, 0):.2f} vs {strength.get(strong, 0):.2f})")
    by_theme = {}
    for f in keep:
        by_theme.setdefault(TH.FACTOR_THEME[f], []).append(f)
    if verbose:
        print(f"admitted {len(keep)} of {len(cleared)} that cleared the screen")
        for t in TH.THEMES:
            fs = by_theme.get(t, [])
            print(f"  {t:12s} {len(fs):2d}  {', '.join(fs) if fs else '-- no validated factor'}")
        print("\ncut, with the reason:")
        for f, why in sorted(cut.items()):
            print(f"  {f:22s} {why}")
    return by_theme, cut


if __name__ == "__main__":
    admitted()
