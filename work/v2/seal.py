"""Seal the holdout. Run ONCE.

The most recent 18 months of signal dates are moved to a separate file with a
recorded sha256. Nothing in the build/tune loop opens that file; `search.py`
imports `load_train()` and there is no `load_holdout()` in it.
"""
from __future__ import annotations
import hashlib, json, os, pandas as pd, numpy as np

CACHE = "/home/claude/psr/cache"
SRC = f"{CACHE}/panel_step5.parquet"
TRAIN = f"{CACHE}/TRAIN.parquet"
HOLD = f"{CACHE}/SEALED_HOLDOUT.parquet"
SEAL = f"{CACHE}/SEAL.json"

HOLDOUT_START = pd.Timestamp("2025-03-01")   # 18 months back from 2026-08-28
PURGE_EMBARGO_SESSIONS = 63 + 21             # label horizon + embargo


def main():
    p = pd.read_parquet(SRC)
    d = np.array(sorted(p["date"].unique()))
    h0 = d[d >= HOLDOUT_START][0]
    i0 = int(np.where(d == h0)[0][0])
    # signal dates are 5 sessions apart, so the gap is measured in signal steps
    gap_steps = int(np.ceil(PURGE_EMBARGO_SESSIONS / 5))
    train_end = d[i0 - gap_steps - 1]
    tr = p[p["date"] <= train_end].copy()
    ho = p[p["date"] >= h0].copy()
    tr.to_parquet(TRAIN, index=False)
    ho.to_parquet(HOLD, index=False)
    meta = {
        "holdout_start": str(pd.Timestamp(h0).date()),
        "holdout_end": str(pd.Timestamp(d[-1]).date()),
        "holdout_signal_dates": int(ho["date"].nunique()),
        "holdout_rows": int(len(ho)),
        "train_start": str(pd.Timestamp(d[0]).date()),
        "train_end": str(pd.Timestamp(train_end).date()),
        "train_signal_dates": int(tr["date"].nunique()),
        "train_rows": int(len(tr)),
        "dead_zone_signal_dates": int(((p["date"] > train_end) & (p["date"] < h0)).sum() > 0)
                                   and int(p[(p["date"] > train_end) & (p["date"] < h0)]["date"].nunique()),
        "purge_embargo_sessions": PURGE_EMBARGO_SESSIONS,
        "sha256_holdout": hashlib.sha256(open(HOLD, "rb").read()).hexdigest(),
        "sha256_train": hashlib.sha256(open(TRAIN, "rb").read()).hexdigest(),
        "sealed_at": pd.Timestamp.utcnow().isoformat(),
        "rule": "one blind evaluation per candidate configuration; no parameter "
                "may be changed after a holdout number is seen",
    }
    json.dump(meta, open(SEAL, "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import sys
    if os.path.exists(SEAL) and "--respilt" not in sys.argv and "--resplit" not in sys.argv:
        print("ALREADY SEALED -- refusing to re-seal.")
        print(open(SEAL).read())
    else:
        # --resplit re-materialises the two files from a rebuilt panel WITHOUT
        # moving the boundary. HOLDOUT_START is a constant in this file; it has
        # never been changed and no holdout row has been read.
        main()
