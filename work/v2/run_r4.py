import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL

tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols)
print(f"{len(fcols)} factors | folds {{h: n}} = { {h: len(v) for h, v in fr.folds.items()} }",
      flush=True)
res = {}
for h in (10, 21, 42):
    for k in (8, 12):
        ch, hist = SEL.forward(fr, h, fcols, max_f=14, k=k, alpha=2000.0,
                               min_gain=-99.0, verbose=True)
        res[f"h{h}_k{k}"] = {"path": hist}
        json.dump(res, open("/home/claude/psr/cache/r4_select.json", "w"),
                  indent=1, default=float)
        print(f"  == h={h} k={k} path complete\n", flush=True)
print("done")
