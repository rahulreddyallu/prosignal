"""Does a factor that clears the screen do it in BOTH halves of its own life?

A factor whose entire evidence sits in one regime can clear a placebo screen and
still be an artefact of that regime. The screen asks "is this distinguishable
from noise"; this asks "is it the same thing twice". Only the second is what the
brief means by genuine validated signal.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, themes as TH

H = 21


def split_ic(tr, col, h=H):
    R = core.sector_neutral_ranks(tr, [col])[col + "_r"]
    d = pd.DataFrame({"date": tr["date"], "z": R, "y": tr[f"y{h}"]}).dropna()
    if d.empty:
        return None
    ics = []
    for dt_, g in d.groupby("date", sort=True):
        if len(g) < 30:
            continue
        a = g["z"].rank().to_numpy(); b = g["y"].rank().to_numpy()
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        ics.append((dt_, float(np.corrcoef(a, b)[0, 1])))
    if len(ics) < 12:
        return None
    s = pd.Series([v for _, v in ics], index=[d_ for d_, _ in ics])
    mid = len(s) // 2
    def t(x):
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 3 else np.nan
    return {"n": len(s), "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
            "ic": float(s.mean()), "t": t(s),
            "ic_h1": float(s[:mid].mean()), "t_h1": t(s[:mid]),
            "ic_h2": float(s[mid:].mean()), "t_h2": t(s[mid:]),
            "same_sign": bool(np.sign(s[:mid].mean()) == np.sign(s[mid:].mean()))}


if __name__ == "__main__":
    tr = core.load_train2()
    sc = pd.read_csv("/home/claude/psr/cache/screen2.csv")
    keep = sorted(set(sc[sc.keep].factor))
    print(f"{len(keep)} factors cleared the placebo screen at some horizon\n")
    rows = []
    for f in keep:
        if f not in tr.columns:
            continue
        r = split_ic(tr, f)
        if r is None:
            continue
        r.update({"factor": f, "theme": TH.FACTOR_THEME.get(f)})
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["theme", "t"], key=lambda s: s if s.name == "theme" else s.abs(), ascending=[True, False])
    pd.set_option("display.width", 220)
    print(df[["theme", "factor", "n", "first", "last", "ic", "t",
              "ic_h1", "t_h1", "ic_h2", "t_h2", "same_sign"]].round(4).to_string(index=False))
    df.to_csv("/home/claude/psr/cache/stability.csv", index=False)
    print("\nFAILS the both-halves test (sign flips between halves):")
    bad = df[~df.same_sign]
    print("  " + (", ".join(bad.factor) if len(bad) else "none"))
