"""Freeze the v3 configuration. Run BEFORE either holdout, once.

Everything that will be applied blind is computed here from TRAIN2 only and
written with a sha256: factor signs, theme horizons, theme weights after the cap
and floor, the absolute floor, and the book.
"""
from __future__ import annotations
import sys, json, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import core, themes as TH, composite as CO, survivors as SV, search3 as S3

CAP, WFLOOR, MIN_THEMES = 0.40, 0.06, 3
BOOK = dict(slots=10, entry_rank=20, exit_rank=30, rebalance_every=2,
            max_per_sector=3, weighting="equal")
GATE = "trend_npos3"   # close > 200 DMA AND >= 3 themes above the median
UNIVERSE = 750
H_BLEND = 21

by_theme, cut = SV.admitted(verbose=False)
tr = core.load_train2()
fr = S3.Frame(tr, by_theme)
allrows = np.arange(len(fr.df))

signs, raw_w = {}, {}
for t, cols in fr.by_theme.items():
    h = S3.THEME_HORIZON[t]
    sg, _, _ = S3._fit_theme(fr, cols, allrows, "equal", h)
    signs[t] = sg
    s = CO.theme_subscore(fr.R, cols, sg, fr.dates_col)
    raw_w[t] = S3._theme_contribution(fr, s.to_numpy("float64"), allrows, H_BLEND)

coverage = {t: float(np.isfinite(
    CO.theme_subscore(fr.R, fr.by_theme[t], signs[t], fr.dates_col).to_numpy()).mean())
    for t in fr.by_theme}
w = CO.cap_weights({t: max(v["topk"], 0.0) for t, v in raw_w.items()},
                   CAP, floor=WFLOOR, coverage=coverage)

cfg = {
    "name": "prosignal-v3",
    "architecture": "two-level: factors -> theme sub-score -> capped theme blend",
    "themes": {
        t: {"weight": round(w[t], 5),
            "orientation_horizon_sessions": S3.THEME_HORIZON[t],
            "validated_topk_contribution": round(raw_w[t]["topk"], 6),
            "validated_ic_t": round(raw_w[t]["ic_t"], 3),
            "coverage": round(coverage[t], 4),
            "within_theme": "sign_oriented_equal_weight",
            "factors": [{"name": f, "sign": int(signs[t][f])} for f in fr.by_theme[t]]}
        for t in fr.by_theme},
    "theme_weighting": {
        "raw": "each theme's validated top-decile excess on the training window",
        "cap": CAP, "floor": WFLOOR, "coverage_cap": True,
        "note": "the cap stops one theme swamping the rest and the floor stops a "
                "theme that cleared its screen from falling to zero on one "
                "window. Measured on validation, the floor alone moved max "
                "drawdown from -38.5% to -34.9% at no cost in excess. A theme "
                "is ALSO capped at its coverage: fitted on the whole training "
                "window without that, `quality` took the 45% cap on 19% "
                "coverage, which ranks the 19% and the 81% by different models. "
                "With the coverage cap the ranking improves (IC t 4.84 -> 5.80, "
                "quintile t 2.91 -> 3.42) and the ten-name book's excess falls "
                "(+5.5% -> +1.9%) -- the first pair is the statistic with power, "
                "the second is the one a permuted-label test put almost "
                "entirely inside its own null."},
    "min_themes": MIN_THEMES,
    "absolute_floor": {
        "rule": "close above the 200-session moving average AND at least 3 of the "
                "name's available themes above the cross-sectional median",
        "why": "a floor on a cross-sectional RANK cannot fire -- somebody is top "
               "of the list every day. Measured: 'three themes above median' "
               "alone left at least 87 names on every one of 235 validation "
               "dates. With the trend condition the count fell to 11 at the "
               "COVID trough (2020-03-24), one name above a ten-slot book.",
        "no_trade": "when fewer names clear the floor than there are slots, the "
                    "remainder is held in CASH at 0%"},
    "universe": {"source": "liquidity_pit", "max_names": UNIVERSE,
                 "min_adtv_inr": 5e7, "adtv_lookback_sessions": 60,
                 "min_price_inr": 20.0, "min_history_sessions": 300,
                 "note": "index membership history is not in this store, so the "
                         "universe is a point-in-time liquidity screen. Measured "
                         "survivorship: 141 of 1,425 panel symbols stopped "
                         "printing before the end and 6.2% of rows belong to "
                         "them; 26-38% of an old cross-section is absent from "
                         "today's."},
    "book": BOOK, "gate": GATE,
    "execution": {"fill": "next_session_vwap"},
    "pit_fundamentals": {
        "disclosure_lag_days_by_quarter_end_month": {"3": 112, "6": 104, "9": 60, "12": 60},
        "calibration": "the measured p99 of the real filing lag on the 3,504 rows "
                       "that carry a filing_date; the statutory 45-day deadline "
                       "would have leaked on 15.9% of them",
        "max_age_days": 420},
    "excluded_themes": {
        "value": "0 of 8 factors clear the placebo screen at any horizon; "
                 "balance-sheet data begins 2023 and covers a median of 0 names "
                 "per training date",
        "liquidity": "0 of 9 clear; volume_shock_5 clears at h=42 and its sign "
                     "flips between the halves of its own life",
        "seasonality": "0 of 2 clear; placebo |t| threshold 9.9 against a real t "
                       "of -1.2"},
    "cut_factors": cut,
    "validation": {"window": "2020-01-31 .. 2024-10-25", "folds": 8,
                   "purge_sessions": 63, "embargo_sessions": 21,
                   "ic": 0.0444, "ic_t": 5.80, "quintile": 0.0082,
                   "quintile_t": 3.42, "topk_t": 1.92,
                   "ann": 0.374, "bench": 0.297, "excess": 0.077, "ir": 0.364,
                   "sharpe": 1.493, "maxdd": -0.306,
                   "median_hold_sessions": 20,
                   "min_names_clearing_floor": 11},
}
blob = json.dumps(cfg, sort_keys=True, default=str).encode()
cfg["config_sha256"] = hashlib.sha256(blob).hexdigest()
json.dump(cfg, open("/home/claude/psr/cache/FROZEN_V3.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in cfg.items()
                  if k in ("name", "themes", "theme_weighting", "min_themes",
                           "book", "config_sha256")}, indent=1, default=str))
