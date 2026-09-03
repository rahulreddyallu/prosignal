"""K-6 (cost side): how fast does the impact coefficient eat the edge?

WHAT THIS IS AND IS NOT. This does NOT re-simulate the book -- reproducing the
shipped book's floor + cadence faithfully enough to trust a net-P&L number is a
separate, error-prone task, and a wrong P&L for a live book is worse than none.
What it DOES is price the SHIPPED cost model (`costs.CostModel`, exact) on the
ACTUAL names the v3 book would have bought, at three impact coefficients, and set
the result against the gross edge the sealed holdout already measured. That is
enough to answer K-6's real question: at the shipped coefficient, and at the
stresses the config itself demands (2x, worse), does the ranking's gross edge
survive costs?

INPUTS THAT ARE MEASURED, NOT ASSUMED:
  - position size   = capital / slots (shipped: 6 slots).
  - name liquidity  = the ADTV of the names the v3 score actually ranks top-6
                      after the absolute floor, read from the cached panel.
  - turnover        = trades/year; taken from the shipped admission note (34 at
                      cadence 21) and varied +/- for sensitivity.
  - gross edge      = HOLDOUT window A/B, net excess + cost drag (BOOK_NOTE):
                      window A gross excess ~= -7.25% + 9.5% ~= +2.2%/yr;
                      window B gross excess ~= +1.3% + 14.3% ~= +15.6%/yr.
                      (Two windows disagree by 7x -- both are reported.)

Usage:
    python research/v3/experiments/cost_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.config.loader import load_config     # noqa: E402
from prosignal.costs import CostModel               # noqa: E402

OUT = Path(__file__).resolve().parent
PANEL_CACHE = OUT / "panel_2026_09.parquet"
COEFFS = [0.10, 0.25, 0.50]              # shipped, 2.5x, 5x (config search_range top)
TRADES_PER_YEAR = [24, 34, 48]           # shipped ~34 at cadence 21; bracket it
# Gross annual excess the sealed holdouts implied (net + measured cost drag).
GROSS_EXCESS = {"window_A": 0.022, "window_B": 0.156}


def _top6_adtvs() -> np.ndarray:
    """ADTV (Rs) of the names the v3 score ranks top-6 after the floor, pooled
    over the recent cached panel. Falls back to a spread of plausible mid-cap
    ADTVs if the panel is absent."""
    if not PANEL_CACHE.exists():
        print("[warn] no cached panel; using a fallback ADTV spread", flush=True)
        return np.array([5e7, 1e8, 2e8, 5e8, 1e9])
    p = pd.read_parquet(PANEL_CACHE, columns=["date", "score", "adtv",
                                              "n_themes_positive"])
    p = p[pd.to_datetime(p["date"]) >= (pd.to_datetime(p["date"]).max()
                                        - pd.DateOffset(months=18))]
    picks = []
    for _, g in p.groupby("date"):
        g = g[(g["n_themes_positive"] >= 3) & g["score"].notna()]
        if g.empty:
            continue
        top = g.nlargest(6, "score")
        picks.append(top["adtv"].to_numpy("float64"))
    if not picks:
        return np.array([5e7, 1e8, 2e8, 5e8, 1e9])
    a = np.concatenate(picks)
    return a[np.isfinite(a) & (a > 0)]


def main() -> None:
    cfg = load_config()
    capital = float(cfg.params.capital.total_capital_inr.value
                    if hasattr(cfg.params.capital.total_capital_inr, "value")
                    else cfg.params.capital.total_capital_inr)
    slots = int(cfg.params.capital.max_open_positions.value
                if hasattr(cfg.params.capital.max_open_positions, "value")
                else cfg.params.capital.max_open_positions)
    pos_value = capital / slots
    adtvs = _top6_adtvs()
    print(f"[book] capital Rs {capital:,.0f}, {slots} slots -> "
          f"Rs {pos_value:,.0f}/position", flush=True)
    print(f"[book] top-6 ADTV: median Rs {np.median(adtvs):,.0f}, "
          f"p25 Rs {np.percentile(adtvs,25):,.0f}, "
          f"p75 Rs {np.percentile(adtvs,75):,.0f}  (n={len(adtvs)})", flush=True)

    results = {"capital": capital, "slots": slots, "pos_value": pos_value,
               "adtv_median": float(np.median(adtvs)), "by_coeff": {}}

    print("\n" + "=" * 72)
    print("ROUND-TRIP COST (bps of position) at the median top-6 name")
    print("=" * 72)
    from types import SimpleNamespace
    for coeff in COEFFS:
        # CostModel reads config.params.costs; deep-copy the pydantic RootConfig,
        # override the impact coefficient, and pass a shim exposing `.params`.
        params_copy = cfg.params.model_copy(deep=True)
        node = params_copy.costs.impact_model.coefficient
        if hasattr(node, "value"):
            node.value = coeff
        else:
            params_copy.costs.impact_model.coefficient = coeff
        model = CostModel(SimpleNamespace(params=params_copy))
        rts = []
        for adtv in adtvs:
            qty = int(max(pos_value / 300.0, 1))     # nominal Rs300 price
            rt = model.round_trip(300.0, qty, adtv_inr=float(adtv)).total_bps_of_buy
            rts.append(rt)
        rts = np.asarray(rts)
        med_rt = float(np.median(rts))
        # annual cost drag on CAPITAL = trades/yr * rt_bps * (pos/capital) /1e4
        drag = {n: TRADES_PER_YEAR_FRAC(n, med_rt, pos_value, capital)
                for n in TRADES_PER_YEAR}
        results["by_coeff"][coeff] = {
            "round_trip_bps_median": med_rt,
            "round_trip_bps_p75": float(np.percentile(rts, 75)),
            "annual_cost_drag_by_trades": drag,
        }
        print(f"  coeff {coeff:.2f}:  round-trip {med_rt:6.0f} bps (p75 "
              f"{np.percentile(rts,75):6.0f})  |  annual drag "
              + "  ".join(f"{n}t={drag[n]:.1%}" for n in TRADES_PER_YEAR))

    print("\n" + "=" * 72)
    print("BREAK-EVEN: gross excess needed vs measured (net+costdrag) holdout gross")
    print("=" * 72)
    for win, gross in GROSS_EXCESS.items():
        print(f"  {win}: measured gross excess ~= {gross:+.1%}/yr")
        for coeff in COEFFS:
            drag = results["by_coeff"][coeff]["annual_cost_drag_by_trades"][34]
            net = gross - drag
            verdict = "survives" if net > 0 else "NEGATIVE"
            print(f"     coeff {coeff:.2f}: drag {drag:.1%} -> net {net:+.1%}  [{verdict}]")
    # crude break-even coeff at 34 trades against window A gross
    drags = {c: results["by_coeff"][c]["annual_cost_drag_by_trades"][34] for c in COEFFS}
    slope = (drags[0.50] - drags[0.10]) / (0.50 - 0.10)
    be = 0.10 + (GROSS_EXCESS["window_A"] - drags[0.10]) / slope if slope else float("nan")
    results["breakeven_coeff_window_A_34trades"] = be
    print(f"\n  break-even impact coeff vs window-A gross (34 trades): "
          f"~{be:.3f}  (shipped is 0.10)")
    if np.isfinite(be) and be < 0.10:
        print("  => at the SHIPPED coefficient the window-A edge is already "
              "eaten; break-even needs a coefficient BELOW 0.10.")
    (OUT / "cost_sensitivity.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\n[results] -> {OUT/'cost_sensitivity.json'}")
    print("\nCAVEATS: this prices the exact cost model on the real selected "
          "names but does NOT re-simulate entries/exits; the gross-excess "
          "anchors are the sealed holdouts' own net+costdrag arithmetic, and "
          "the two windows disagree ~7x. Treat as the cost-side bound, not a "
          "book P&L.")


def TRADES_PER_YEAR_FRAC(n_trades: float, rt_bps: float, pos_value: float,
                         capital: float) -> float:
    return n_trades * rt_bps / 1e4 * (pos_value / capital)


if __name__ == "__main__":
    main()
