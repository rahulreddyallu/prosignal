"""Holdout re-run on an EQUITY-ONLY universe. One evaluation per window.

WHY THIS IS NOT A RETUNE. The configuration is byte-identical to
FROZEN_V3.json -- same factors, same signs, same theme weights, same cap, floor,
gate and book. Nothing was refitted. What changed is the UNIVERSE, and it
changed because it was wrong: NSE publishes ETFs, gold and silver funds, liquid
funds and bond funds in the same cash bhavcopy under the same EQ series, the
liquidity screen admitted them, and a stock model ranked them against companies.

The defect was found on a LIVE run, not by looking at holdout results: three of
the top five names on 2026-08-25 were bond ETFs. Measured afterwards, they took
26.25% of the top-ten slots across window A and 0.53% across window B -- India's
gold and silver ETF listings expanded enormously over 2024-2026, so the recent
window is contaminated and the older one is not.

The uncorrected numbers stay on the record in HOLDOUT_V3_A.json and
HOLDOUT_V3_B.json and are reported beside these. Window A has now been used
three times -- v2, v3 on the contaminated universe, v3 on the corrected one --
and that is the multiple-testing count to charge it with.

The theme weights were fitted on a training panel that still contains these
instruments. They were deliberately NOT refitted: refitting would make this a
new configuration needing its own window, and a slightly stale weight can only
hurt the result reported here.
"""
from __future__ import annotations
import sys, json, hashlib, os, argparse, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, holdout3 as HO
from prosignal.data.instruments import non_equity_symbols

CACHE = "/home/claude/psr/cache"
D = "/mnt/user-data/uploads/Pro Stock Signal BOT/data/curated"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--window", choices=["A", "B"], required=True)
    a = ap.parse_args()
    out = f"{CACHE}/HOLDOUT_V3_{a.window}_EQUITY.json"
    if os.path.exists(out):
        print(f"WINDOW {a.window} (equity-only) ALREADY RUN -- the result stands.")
        print(open(out).read()); sys.exit(0)
    # WINDOW B USES THE PRE-B CONFIGURATION. FROZEN_V3.json was fitted on data
    # through 2024-10, which CONTAINS window B, so applying it there would be an
    # in-sample number wearing a holdout's name. The pre-B configuration was
    # produced by re-running the whole pipeline on data ending 2021-02-17.
    cfg_file = "FROZEN_V3.json" if a.window == "A" else "FROZEN_V3_PRE_B.json"
    cfg = json.load(open(f"{CACHE}/{cfg_file}"))
    print(f"configuration: {cfg_file} ({cfg.get('name')})", flush=True)
    seal = json.load(open(f"{CACHE}/SEAL2.json"))
    fn = "SEALED_A_RECENT" if a.window == "A" else "SEALED_B_ERA"
    key = "A_recent" if a.window == "A" else "B_era"
    assert hashlib.sha256(open(f"{CACHE}/{fn}.parquet", "rb").read()).hexdigest() \
        == seal[key]["sha256"], "the sealed file changed"
    panel = pd.read_parquet(f"{CACHE}/{fn}.parquet")

    import data as D_
    m = D_.build()
    close = pd.DataFrame(m["close"], index=pd.DatetimeIndex(m["dates"]),
                         columns=list(m["symbols"]))
    em = pd.read_parquet(f"{D}/equity_master.parquet")
    drop = non_equity_symbols(sorted(panel["symbol"].unique()),
                              equity_master=em, close=close)
    before = len(panel)
    panel = panel[~panel["symbol"].isin(drop)].copy()
    print(f"excluded {len(drop)} non-equity instruments, "
          f"{before - len(panel):,} of {before:,} rows ({(before-len(panel))/before:.2%})",
          flush=True)

    res = HO.evaluate(panel, cfg, 21, f"{a.window}_equity_only")
    res["seal"] = seal[key]
    res["universe_correction"] = {
        "excluded_instruments": int(len(drop)),
        "excluded_rows_share": round((before - len(panel)) / before, 5),
        "names": sorted(drop),
        "config_unchanged": True,
        "trials_on_this_window": 3 if a.window == "A" else 2}
    json.dump(res, open(out, "w"), indent=1)
    b = res["book_net_of_costs"]
    print(json.dumps({"window": res["window"], "book": b,
                      "floor": res["absolute_floor"],
                      "ranking_h21": res["ranking_h21"],
                      "ranking_h42": res["ranking_h42"],
                      "theme_ic_h21": res["theme_ic_h21"],
                      "null": res["shuffled_score_null"]}, indent=1))
