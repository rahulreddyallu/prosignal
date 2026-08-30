"""Seal the holdouts for the themed rebuild. Run ONCE.

TWO WINDOWS, PRE-REGISTERED BEFORE ANY MODEL EXISTS.

  A  RECENT  2025-03-06 -> 2026-08-17, the most recent 18 months, which is what
     the brief mandates. It is the SAME window the v2 configuration was
     evaluated on, because it is the only recent data there is. That makes this
     the SECOND configuration scored on window A, and the multiple-testing cost
     is charged explicitly rather than quietly enjoyed.

  B  ERA     2021-07-01 -> 2022-12-31, which NO search has touched -- not this
     one, not v2's. It covers the 2022 drawdown and the global rate shock, so it
     asks whether the architecture survives a regime, not just a stretch of
     calendar. To keep it genuinely out of sample the whole pipeline is re-run
     for it -- screen, theme weights, book -- on pre-2021-07 data only.

Neither file is opened by the build loop. `search*.py` imports `load_train()`
and there is no loader for either sealed file in it.
"""
from __future__ import annotations
import hashlib, json, os, numpy as np, pandas as pd

CACHE = "/home/claude/psr/cache"
SRC = f"{CACHE}/panel2_step5.parquet"
SEAL = f"{CACHE}/SEAL2.json"

A_START = pd.Timestamp("2025-03-01")     # most recent 18 months
B_START = pd.Timestamp("2021-07-01")     # untouched era window
B_END = pd.Timestamp("2022-12-31")
PURGE_EMBARGO_SESSIONS = 63 + 21
STEP = 5


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    p = pd.read_parquet(SRC)
    d = np.array(sorted(p["date"].unique()))
    gap = int(np.ceil(PURGE_EMBARGO_SESSIONS / STEP))

    a0 = d[d >= A_START][0]
    ia = int(np.where(d == a0)[0][0])
    train_end = d[ia - gap - 1]

    b0 = d[d >= B_START][0]
    b1 = d[d <= B_END][-1]
    ib0, ib1 = int(np.where(d == b0)[0][0]), int(np.where(d == b1)[0][0])
    b_train_end = d[ib0 - gap - 1]

    tr = p[p["date"] <= train_end]
    ha = p[p["date"] >= a0]
    hb = p[(p["date"] >= b0) & (p["date"] <= b1)]
    b_tr = p[p["date"] <= b_train_end]

    for name, frame in (("TRAIN2", tr), ("SEALED_A_RECENT", ha),
                        ("SEALED_B_ERA", hb), ("TRAIN2_PRE_B", b_tr)):
        frame.to_parquet(f"{CACHE}/{name}.parquet", index=False)

    meta = {
        "sealed_at": pd.Timestamp.now("UTC").isoformat(),
        "step_sessions": STEP,
        "purge_embargo_sessions": PURGE_EMBARGO_SESSIONS,
        "A_recent": {
            "start": str(pd.Timestamp(a0).date()), "end": str(pd.Timestamp(d[-1]).date()),
            "signal_dates": int(ha["date"].nunique()), "rows": int(len(ha)),
            "sha256": _sha(f"{CACHE}/SEALED_A_RECENT.parquet"),
            "note": "the window the v2 configuration was already scored on; this "
                    "is the SECOND configuration evaluated here and the trial "
                    "count is 2",
        },
        "B_era": {
            "start": str(pd.Timestamp(b0).date()), "end": str(pd.Timestamp(b1).date()),
            "signal_dates": int(hb["date"].nunique()), "rows": int(len(hb)),
            "sha256": _sha(f"{CACHE}/SEALED_B_ERA.parquet"),
            "note": "untouched by every search so far; the pipeline is re-run "
                    "end to end on pre-2021-07 data only before it is opened",
        },
        "train_main": {"start": str(pd.Timestamp(d[0]).date()),
                       "end": str(pd.Timestamp(train_end).date()),
                       "signal_dates": int(tr["date"].nunique()), "rows": int(len(tr)),
                       "sha256": _sha(f"{CACHE}/TRAIN2.parquet")},
        "train_pre_B": {"start": str(pd.Timestamp(d[0]).date()),
                        "end": str(pd.Timestamp(b_train_end).date()),
                        "signal_dates": int(b_tr["date"].nunique()),
                        "rows": int(len(b_tr))},
        "rule": "one blind evaluation per candidate configuration per window; no "
                "parameter may change after a holdout number is seen",
    }
    json.dump(meta, open(SEAL, "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    if os.path.exists(SEAL) and "--resplit" not in __import__("sys").argv:
        print("ALREADY SEALED -- refusing to re-seal.")
        print(open(SEAL).read())
    else:
        main()
