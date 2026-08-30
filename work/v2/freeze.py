"""Freeze the champion configuration. Run BEFORE the holdout, once.

Writes config + factor signs estimated on TRAIN only, and a sha256 over the
whole thing. Anything that changes after the holdout number is seen changes
this hash, which is the point.
"""
import sys, json, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, selector as SEL, finalists as F

H = 42
tr = core.load_train()
meta = {"date", "symbol", "sector", "entry_px", "adtv", "adtv_rank", "close",
        "atr_pct", "mae5", "mfe5"}
fcols = [c for c in tr.columns if c not in meta and not (c[0] in "yb" and c[1:].isdigit())]
fr = SEL.Frame(tr, fcols, horizons=(H,), start_frac=0.20, n_blocks=8)
FEATS = F.decorrelate(fr, F.survivors(H), top=12)

cols = [fr.col[f] for f in FEATS]
m = np.isfinite(fr.Y[H])
A = fr.X[np.ix_(np.where(m)[0], cols)].astype("float64")
y = fr.Yr[H][m]
ic = np.array([np.corrcoef(A[:, j], y)[0, 1] for j in range(A.shape[1])])
sign = np.where(np.nan_to_num(ic) >= 0, 1.0, -1.0)

cfg = {
    "name": "prosignal-v2",
    "label_horizon_sessions": H,
    "signal_step_sessions": 5,
    "execution": {"fill": "next_session_vwap",
                  "note": "signal on the close of t, filled at the VWAP of t+1"},
    "universe": {"source": "liquidity_pit", "min_adtv_inr": 5e7,
                 "adtv_lookback_sessions": 60, "min_price_inr": 20.0,
                 "min_history_sessions": 300, "max_names": 750},
    "factors": [{"name": f, "sign": int(sign[j]), "train_ic": float(ic[j]),
                 "weight": round(1.0 / len(FEATS), 6)} for j, f in enumerate(FEATS)],
    "ranking": {"method": "sector_neutral_cross_sectional_rank",
                "range": [-1, 1], "min_sector_names": 12,
                "fallback": "single residual group per date"},
    "combination": {"method": "sign_oriented_equal_weight",
                    "note": "the only fitted parameter is each factor's sign, and it "
                            "was identical in all 8 walk-forward folds"},
    "book": {"slots": 10, "entry_rank": 15, "exit_rank": 25,
             "rebalance_every_signal_dates": 1, "signal_date_stride_sessions": 5,
             "max_per_sector": 3, "weighting": "equal", "cash_rate": 0.0},
    "entry_gate": {"mode": "none", "computed_and_reported": True,
                   "measured": "market-timing gates (ma200, vol_calm, breadth) cost "
                               "8-13pp of annual excess on 2020-2024 validation and "
                               "did not reduce max drawdown; the gate is computed and "
                               "flagged, not applied"},
    "no_trade": {"structural": "a slot with no admissible name holds cash at 0%",
                 "drawdown_circuit_breaker": {"threshold": -0.15, "action": "flag"}},
    "costs": {"model": "CostModel (delivery segment), square-root impact",
              "round_trip_bps_at_1.25L": {"adtv_20cr": 87, "adtv_5cr": 137}},
    "validation": {"window": "2020-01-31 .. 2024-10-25", "folds": 8,
                   "purge_sessions": H, "embargo_sessions": 21,
                   "excess_ann": 0.225, "ir": 1.149, "sharpe": 1.707,
                   "maxdd": -0.380, "median_hold_sessions": 20,
                   "quintile_spread_per_period": 0.0185, "quintile_t": 6.05,
                   "permuted_label_null_p_quintile": 0.0},
}
blob = json.dumps(cfg, sort_keys=True).encode()
cfg["config_sha256"] = hashlib.sha256(blob).hexdigest()
json.dump(cfg, open("/home/claude/psr/cache/FROZEN_CONFIG.json", "w"), indent=1)
print(json.dumps(cfg, indent=1))
