"""Finalist configurations, each a complete specification.

The wide search established the SHAPE of the answer: a broad liquid universe, a
wide entry band with a long rebalance stride, equal weights, and a composite
that is simple. What is left is to choose between a small number of honest
candidates on a validation window that contains the 2020 crash -- and to do it
before anything touches the sealed holdout.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core

SCREEN = pd.read_csv("/home/claude/psr/cache/screen_train.csv")


def survivors(h):
    return list(SCREEN[(SCREEN.h == h) & SCREEN.keep]
                .sort_values("t", key=abs, ascending=False).factor)


def decorrelate(fr, cands, max_r=0.70, top=12):
    keep = []
    for c in cands:
        j = fr.col[c]
        ok = True
        for k in keep:
            i = fr.col[k]
            r = abs(np.corrcoef(fr.X[:, j], fr.X[:, i])[0, 1])
            if r > max_r:
                ok = False
                break
        if ok:
            keep.append(c)
        if len(keep) >= top:
            break
    return keep


TREND_CORE = ["ma_50_200", "dist_200dma", "trend_slope_120", "prox_52w",
              "mom_consist_126", "intraday_mom_126", "voladj_mom_6_1",
              "deliv_z_21", "deliv_chg_5", "ret_kurt_126", "ulcer_120", "resid_rev_21"]

GREEDY_21 = ["mom_9_1", "trend_r2_120", "deliv_value_trend", "beta_126", "dist_200dma"]
GREEDY_10 = ["ma_50_200", "voladj_mom_6_1", "fip_6", "mom_6_1"]

INCUMBENT = ["mom_6_1", "resid_rev_21", "downside_vol_60", "beta_126", "amihud_60",
             "log_adtv_60", "max_dd_120", "prox_52w", "max5_21", "resid_mom_252_21",
             "idio_vol_126", "idio_skew_126", "deliv_pct_60", "deliv_trend"]
