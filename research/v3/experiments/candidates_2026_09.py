"""CANDIDATE FACTORS -- does the shipped set have a hole a new column would fill?

The 2026-09 factor audit found nine of the twenty-two shipped factors adding
nothing or worse. The obvious follow-up is whether the model is also MISSING
something, so this builds every factor that (a) has published evidence on Indian
equities and (b) can be computed point-in-time from the curated store, and asks
what each adds to the composite.

WHAT IS BUILT, and the claim each rests on:

  beta_120        Frazzini & Pedersen's betting-against-beta. The beta anomaly is
                  confirmed on the NSE over 2001-2016 (Bhattacharya & Sonaer /
                  DECISION 2021), and the v3 set has no beta column at all --
                  `families.py` carries `beta_120_r`, but the shipped 22 do not.
  idio_vol_120    Ang, Hodrick, Xing & Zhang. Residual volatility from the same
                  120-session market model, so it is the idiosyncratic half of
                  what `downside_vol_60` measures on total return.
  idio_skew_120   Skewness preference, which Bali, Cakici & Whitelaw (2011) treat
                  as a channel SEPARATE from MAX -- and `max5_21` is already in
                  the reversal theme, so this is the part not represented.
  max_dd_120      The other side of the risk axis. `families.py` split beta and
                  drawdown apart because they correlate -0.42 within date: a high
                  beta rank is riskier, a shallow drawdown is safer, and averaging
                  them cancelled the axis.
  deliv_pct_252   The theme that carries the model, slower. `deliv_pct_60` turns
                  over 0.2x a year and `deliv_chg_5` turns over 15.5x; if the
                  information is in the slow part, a 252-session mean should keep
                  it at no cost.
  deliv_val_z_60  Delivered VALUE rather than the delivered PERCENTAGE, scaled by
                  the name's own history. `deliv_pct` is scale-free and cannot
                  tell a large settling flow in a liquid name from a small one in
                  a quiet name; the institutional-accumulation reading the
                  delivery literature actually makes is about the former.

WHAT COULD NOT BE BUILT. Post-earnings-announcement drift. `statements.parquet`
joined to `results_calendar.parquet` on (symbol, period_end) yields 1,325 rows
across 664 symbols -- roughly two usable periods each -- and a standardised
unexpected earnings needs four quarters of lag plus eight for the scaling
deviation. PEAD has Indian evidence and is NOT refuted here; it is unmeasurable
from this store, which is a data finding rather than a factor finding.

Everything reads the same next-session-execution label the panel uses, and
nothing reads a session after its decision row.

Usage:
    python research/v3/experiments/candidates_2026_09.py            # build + test
    python research/v3/experiments/candidates_2026_09.py --build-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.data.store import DataStore                      # noqa: E402
from prosignal.features import engine                               # noqa: E402
from prosignal.validation.significance import overlap_lag        # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _panel_guard import provenance, require_fresh                        # noqa: E402

HERE = Path(__file__).resolve().parent
PANEL = HERE / "panel_2026_09.parquet"
CACHE = HERE / "candidates_2026_09.parquet"
OUT = HERE / "candidates_2026_09.json"
LABEL = "y21"

#: Sign each factor is entered at, from the literature and NOT from a fit on this
#: panel. A candidate whose only justification is the sign that happened to work
#: is a candidate that has been fitted to the test set.
PRIOR_SIGN = {"beta_120": -1, "idio_vol_120": -1, "idio_skew_120": -1,
              "max_dd_120": +1, "deliv_pct_252": +1, "deliv_val_z_60": +1}


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute every candidate on the panel's (date, symbol) grid."""
    syms = set(panel["symbol"].astype(str))
    start = (panel["date"].min() - pd.Timedelta(days=700)).date()
    end = panel["date"].max().date()
    store = DataStore(ROOT / "data/curated", ROOT / "data/snapshots")

    px = store.read_prices(start=start, end=end)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px[px["symbol"].astype(str).isin(syms)]
    piv = lambda c: px.pivot_table(index="date", columns="symbol", values=c,
                                   aggfunc="last", observed=True).sort_index()
    close, turn = piv("close"), piv("turnover")
    ret = close / close.shift(1) - 1.0

    # THE BENCHMARK IS THE MARKET AS IT STOOD -- equal-weight return of names that
    # actually traded that day, matching how `v3_factors` builds `bench_ret`. An
    # index would be a different market from the one the universe is drawn from.
    bench = ret.where(turn.notna() & (turn > 0)).mean(axis=1)
    bc = bench - bench.rolling(120, min_periods=80).mean()
    bvar = (bc * bc).rolling(120, min_periods=80).mean()
    beta = (ret.mul(bc, axis=0).rolling(120, min_periods=80).mean()
            .div(bvar.replace(0, np.nan), axis=0))
    resid = ret.sub(ret.rolling(120, min_periods=80).mean()).sub(beta.mul(bc, axis=0))

    F = {
        "beta_120": beta,
        "idio_vol_120": resid.rolling(120, min_periods=80).std(),
        "idio_skew_120": resid.rolling(120, min_periods=80).skew(),
        "max_dd_120": close / close.rolling(120, min_periods=80).max()
                          .replace(0, np.nan) - 1.0,
    }

    dl = store.read_delivery(start=start, end=end)
    dl["date"] = pd.to_datetime(dl["date"]).dt.normalize()
    dl = dl[dl["symbol"].astype(str).isin(syms)]
    dp = (dl.pivot_table(index="date", columns="symbol", values="deliv_pct",
                         aggfunc="last", observed=True).sort_index()
          .reindex(index=close.index, columns=close.columns))
    F["deliv_pct_252"] = dp.rolling(252, min_periods=150).mean()
    # log1p because delivered value is right-skewed across four orders of
    # magnitude; the z-score is of the name against ITSELF, so cross-sectional
    # scale never enters.
    dv = dp / 100.0 * turn.reindex(index=close.index, columns=close.columns)
    ldv = np.log1p(dv.clip(lower=0))
    F["deliv_val_z_60"] = ((ldv.rolling(60, min_periods=40).mean()
                            - ldv.rolling(252, min_periods=150).mean())
                           / ldv.rolling(252, min_periods=150).std().replace(0, np.nan))

    rows = panel[["date", "symbol"]].copy()
    rows["symbol"] = rows["symbol"].astype(str)
    out = rows.copy()
    for k, frame in F.items():
        frame = frame.replace([np.inf, -np.inf], np.nan)
        s = frame.stack(dropna=False).rename(k)
        s.index.names = ["date", "symbol"]
        out = out.merge(s.reset_index(), on=["date", "symbol"], how="left")
    return out


def nw_t(x, lags: int) -> float:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype="float64")
    n = len(x)
    if n < 6:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - L / (lags + 1.0)) * ((e[L:] @ e[:-L]) / n)
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--allow-stale", action="store_true",
                    help="report even though the store has moved under the "
                         "panel. The numbers then describe the OLD store.")
    args = ap.parse_args()

    if not PANEL.exists():
        print(f"panel not found: {PANEL}; build it with audit_2026_09.py first")
        return 2
    require_fresh(PANEL, allow_stale=args.allow_stale)
    P = pd.read_parquet(PANEL).reset_index(drop=True)
    P["date"] = pd.to_datetime(P["date"])

    if CACHE.exists() and not args.rebuild:
        C = pd.read_parquet(CACHE)
    else:
        C = build(P)
        C.to_parquet(CACHE, index=False)
        print(f"built {CACHE}")
    if args.build_only:
        return 0

    cands = [c for c in PRIOR_SIGN if c in C.columns]
    for c in cands:
        P[c] = C[c].to_numpy()

    dates = sorted(P["date"].unique())
    idx = {d: np.asarray(v) for d, v in P.groupby("date").indices.items()}
    y = P[LABEL].to_numpy("float64")
    base = P["score"].to_numpy("float64")
    pop = np.isfinite(base)
    lag = overlap_lag(21, 5)

    def ic(vals):
        v = np.asarray(vals, dtype="float64")
        out = []
        for d in dates:
            ix = idx[d]
            m = pop[ix] & np.isfinite(v[ix]) & np.isfinite(y[ix])
            if m.sum() < 30:
                continue
            a = pd.Series(v[ix][m]).rank().to_numpy()
            b = pd.Series(y[ix][m]).rank().to_numpy()
            if a.std() == 0 or b.std() == 0:
                continue
            out.append(np.corrcoef(a, b)[0, 1])
        return np.asarray(out)

    def blended(col, sign, w):
        v = np.asarray(P[col], dtype="float64") * sign
        sc = np.full(len(P), np.nan)
        for d in dates:
            ix = idx[d]
            ex = np.full(len(ix), np.nan)
            ok = np.isfinite(v[ix])
            if ok.sum() >= 30:
                ex[ok] = (pd.Series(v[ix][ok]).rank(pct=True).to_numpy() - 0.5) * 2.0
            b = base[ix]
            okb = np.isfinite(b)
            num = np.where(okb, b, 0.0) * (1 - w) + np.where(np.isfinite(ex), ex, 0.0) * w
            den = np.where(okb, 1 - w, 0.0) + np.where(np.isfinite(ex), w, 0.0)
            sc[ix] = np.where(okb & (den > 0), num / np.maximum(den, 1e-12), np.nan)
        return sc

    b_ic = ic(base)
    rows = {}
    print(f"shipped composite IC {b_ic.mean():+.5f}  NW t {nw_t(b_ic, lag):+.2f}\n")
    print(f"{'candidate':16s}{'sign':>5s}{'cov':>7s}{'own IC':>10s}{'NW t':>7s}"
          f"{'dIC@10%':>10s}{'NW t':>7s}{'dIC@15%':>10s}{'NW t':>7s}")
    for c in cands:
        sg = PRIOR_SIGN[c]
        own = ic(np.asarray(P[c], dtype="float64") * sg)
        d10 = ic(blended(c, sg, 0.10))
        d15 = ic(blended(c, sg, 0.15))
        n = min(len(d10), len(b_ic))
        dd10, dd15 = d10[:n] - b_ic[:n], d15[:n] - b_ic[:n]
        cov = float(np.isfinite(np.asarray(P[c], dtype="float64")[pop]).mean())
        rows[c] = {"sign": sg, "coverage": cov,
                   "own_ic": float(own.mean()), "own_nw_t": nw_t(own, lag),
                   "delta_10": float(dd10.mean()), "delta_10_nw_t": nw_t(dd10, lag),
                   "delta_15": float(dd15.mean()), "delta_15_nw_t": nw_t(dd15, lag)}
        r = rows[c]
        print(f"{c:16s}{sg:>5d}{cov:>7.1%}{r['own_ic']:>10.5f}{r['own_nw_t']:>7.2f}"
              f"{r['delta_10']:>10.5f}{r['delta_10_nw_t']:>7.2f}"
              f"{r['delta_15']:>10.5f}{r['delta_15_nw_t']:>7.2f}")

    # THE VERDICT IS COMPUTED, NOT TYPED. An earlier version stated the numbers
    # in prose and the panel was rebuilt underneath it the same day, leaving a
    # paragraph quoting t-statistics that no longer existed. A conclusion that
    # cannot go stale is one derived from the table above it.
    best = max(rows.items(), key=lambda kv: kv[1]["delta_10_nw_t"])
    adders = [k for k, v in rows.items() if v["delta_10_nw_t"] >= 2.0]
    strong_alone = [k for k, v in rows.items()
                    if v["own_nw_t"] >= 2.0 and v["delta_10_nw_t"] < 2.0]
    verdict = (
        (f"{len(adders)} of {len(rows)} candidates add to the composite at "
         f"|NW t| >= 2.0 ({', '.join(adders)})."
         if adders else
         f"NO candidate adds to the composite at NW t >= 2.0.")
        + f" Best is {best[0]} (dIC {best[1]['delta_10']:+.4f}, NW t "
          f"{best[1]['delta_10_nw_t']:+.2f} at 10% weight)."
        + (f" {len(strong_alone)} carry a real STANDALONE edge and add nothing "
           f"once blended ({', '.join(strong_alone)}), which is what a factor "
           f"already spanned by the shipped set looks like."
           if strong_alone else "")
        + " Read with the correlations against the incumbents: the question a "
          "candidate has to answer is not whether it predicts, but whether it "
          "predicts something the twenty-two do not already carry."
    )
    OUT.write_text(json.dumps(
        {"label": LABEL, "newey_west_lag": lag, "prior_signs": PRIOR_SIGN,
         "panel_provenance": provenance(PANEL),
         "dates": len(dates), "results": rows, "verdict": verdict,
         "pead_not_buildable": (
             "statements x results_calendar yields 1,325 rows / 664 symbols; a "
             "standardised unexpected earnings needs 4 quarters of lag and 8 for "
             "scaling. PEAD is unmeasurable from this store, not refuted.")},
        indent=2))
    print(f"\n{verdict}\n\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
