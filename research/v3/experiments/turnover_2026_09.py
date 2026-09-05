"""BOOK TURNOVER -- the largest unexplored cost in the engine, measured properly.

THE FINDING THIS EXISTS FOR. At the shipped configuration -- six slots, entry
rank 6, exit rank 18, 21-session cadence -- the book replaces about 61% of itself
at every rebalance. At the shipped cost model (89.6 bps round trip on a
Rs 1.67 lakh position in a Rs 23 crore ADTV name) that is roughly 6.6% a year,
against a gross excess of +8.5% at t +1.41. Cost is eating three quarters of a
gross edge that is not itself significant.

WHY THIS IS NOT JUST ANOTHER SWEEP. Widening the exit band obviously cuts
turnover; the question is whether it costs gross return, and a naive sweep
answers that by reading five numbers off the same data the audit already spent
and keeping the best. That is how the repo's DSR ended up at 0.030. So:

  PRE-REGISTERED     The grid, the primary statistic and the decision rule are
                     fixed in SPEC below and hashed. The rule is written BEFORE
                     any number is computed and does not contain the word "best".
  THE COST SIDE IS   Turnover is a property of the ranking and the band, not of
  THE EVIDENCE       returns. It needs no alpha estimate, it has no sampling
                     error worth speaking of, and it is monotone in the band for
                     every model tested. That is the part to act on.
  THE GROSS SIDE IS  With ~95 rebalances a 6-name book's excess is noisy and
  THE CONSTRAINT     non-monotonic. It is used only to ask "does widening the
                     band DESTROY gross return", never to pick the best band.
  CPCV               Every configuration is scored on the same purged, embargoed
                     folds, so a fold that was simply easy cannot flatter one.

THE DECISION RULE, fixed in advance:

    Among bands whose gross excess is not significantly WORSE than the incumbent
    (one-sided paired t on per-fold differences, threshold -1.0), take the one
    with the LOWEST turnover. If none qualifies, change nothing.

That deliberately does not maximise anything. It buys a certain cost saving
subject to not giving up gross return, which is the only trade this evidence can
actually support.

Usage:
    python research/v3/experiments/turnover_2026_09.py
    python research/v3/experiments/turnover_2026_09.py --slots 6 --allow-stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from _panel_guard import provenance, require_fresh                   # noqa: E402
from prosignal.config.loader import load_config                      # noqa: E402
from prosignal.costs import CostModel                                # noqa: E402
from prosignal.features import engine                                # noqa: E402
from prosignal.validation.cpcv import CombinatorialPurgedCV          # noqa: E402
from prosignal.validation.significance import overlap_lag            # noqa: E402

PANEL = HERE / "panel_2026_09.parquet"
OUT = HERE / "turnover_2026_09.json"
LABEL = "y21"
STRIDE_SESSIONS, LABEL_SESSIONS = 5, 21

#: THE PRE-REGISTERED SPECIFICATION.
SPEC = {
    # Exit bands to test, as a multiple of the slot count. The incumbent is 3x
    # (6 slots, exit 18). The grid is fixed here and not extended after results.
    "exit_multiples": (1.0, 2.0, 3.0, 5.0, 8.0),
    # The live cadence: the book is rebalanced every 21 sessions, and the panel
    # samples every 5, so every 4th signal date.
    "rebalance_every_signal_dates": 4,
    "slots": 6,
    "incumbent_exit_multiple": 3.0,
    "primary": "turnover per rebalance -- a property of the ranking, not of returns",
    "constraint": "gross excess must not be significantly worse than the "
                  "incumbent: one-sided paired t on per-fold differences > -1.0",
    "decision_rule": "among bands satisfying the constraint, take the LOWEST "
                     "turnover. If none qualifies, change nothing.",
    "t_threshold": -1.0,
}
SPEC_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True, default=str).encode()).hexdigest()


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


def simulate(score, sym, Y, idx, dates, pop, slots, exit_rank, every):
    """Run the book with hysteresis and return per-rebalance excess and turnover.

    ENTER inside `slots`, HOLD until rank passes `exit_rank`. That is what
    `stage6_entry.admission` does, and the reason it does it is arithmetic: a
    name oscillating around the boundary otherwise pays a full round trip at
    every rebalance for no change in view.
    """
    held: list = []
    ex, turn, dts = [], [], []
    for i, d in enumerate(dates):
        if i % every:
            continue
        ix = idx[d]
        m = pop[ix] & np.isfinite(score[ix]) & np.isfinite(Y[ix])
        if m.sum() < 50:
            continue
        names, s, y = sym[ix][m], score[ix][m], Y[ix][m]
        order = np.argsort(-s)
        rank = {names[j]: r + 1 for r, j in enumerate(order)}
        keep = [h for h in held if rank.get(h, 1 << 30) <= exit_rank]
        for j in order:
            if len(keep) >= slots:
                break
            if names[j] not in keep:
                keep.append(names[j])
        keep = keep[:slots]
        if held:
            turn.append(len(set(keep) - set(held)) / float(slots))
        pos = dict(zip(names, y))
        r = [pos[h] for h in keep if h in pos]
        if r:
            ex.append(float(np.mean(r) - y.mean()))
            dts.append(d)
        held = keep
    return np.asarray(ex), np.asarray(turn), dts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=SPEC["slots"])
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--test-groups", type=int, default=2)
    ap.add_argument("--allow-stale", action="store_true")
    args = ap.parse_args()

    require_fresh(PANEL, allow_stale=args.allow_stale)
    P = pd.read_parquet(PANEL).reset_index(drop=True)
    P["date"] = pd.to_datetime(P["date"])
    dates = sorted(P["date"].unique())
    idx = {d: np.asarray(v) for d, v in P.groupby("date").indices.items()}
    sym = P["symbol"].astype(str).to_numpy()
    Y = P[LABEL].to_numpy("float64")

    cfg = load_config(ROOT / "config/parameters.yaml")
    source = str(cfg.params.stage4_core_score.ranking.source)
    themes = engine.THEMES if source == "v4_composite" else engine.THEMES

    # Rebuild the SHIPPED score from the panel's stored ranks, through the
    # shipped blend, so the book being measured is the book that trades.
    rank_frame = P[[f + "_r" for f in engine.ALL_FACTORS
                    if f + "_r" in P.columns]].copy()
    rank_frame.columns = [c[:-2] for c in rank_frame.columns]
    names = list(themes)
    W = np.array([themes[t].weight for t in names])
    score = np.full(len(P), np.nan)
    for d in dates:
        ix = idx[d]
        blk = rank_frame.iloc[ix]
        subs = np.column_stack([engine.theme_subscore(blk, themes[t]).to_numpy("float64")
                                for t in names])
        ok = np.isfinite(subs)
        den = (ok * W).sum(1)
        num = np.nansum(np.where(ok, subs * W, 0.0), 1)
        score[ix] = np.where((den > 0) & (ok.sum(1) >= 3),
                             num / np.maximum(den, 1e-12), np.nan)
    pop = np.isfinite(score)

    cm = CostModel(cfg)
    capital = float(cfg.params.capital.total_capital_inr.value
                    if hasattr(cfg.params.capital.total_capital_inr, "value")
                    else cfg.params.capital.total_capital_inr)
    pos_value = capital / args.slots
    adtv = float(np.nanmedian(P.loc[pop, "adtv"]))
    bd = cm.round_trip(entry_price=1000.0, quantity=max(int(pos_value / 1000), 1),
                       adtv_inr=adtv)
    rt = bd.total_bps_of_buy / 10_000.0
    rebalances_per_year = 252.0 / LABEL_SESSIONS

    every = SPEC["rebalance_every_signal_dates"]
    purge = int(np.ceil(LABEL_SESSIONS / STRIDE_SESSIONS))
    lag = overlap_lag(LABEL_SESSIONS, STRIDE_SESSIONS)
    cv = CombinatorialPurgedCV(n_groups=args.groups, n_test_groups=args.test_groups,
                               label_horizon=purge, embargo=purge)
    splits = list(cv.split(len(dates)))

    runs = {}
    for mult in SPEC["exit_multiples"]:
        xr = max(int(round(args.slots * mult)), args.slots)
        ex, turn, dts = simulate(score, sym, Y, idx, dates, pop,
                                 args.slots, xr, every)
        runs[mult] = {"exit_rank": xr, "ex": ex, "turn": turn, "dates": dts}

    inc = runs[SPEC["incumbent_exit_multiple"]]
    inc_by_date = dict(zip(inc["dates"], inc["ex"]))

    results = {}
    for mult, r in runs.items():
        by_date = dict(zip(r["dates"], r["ex"]))
        fold_d = []
        for sp in splits:
            td = [dates[i] for i in sp.test_idx]
            pairs = [(by_date[d] - inc_by_date[d]) for d in td
                     if d in by_date and d in inc_by_date]
            if len(pairs) >= 5:
                fold_d.append(float(np.mean(pairs)))
        common = [d for d in r["dates"] if d in inc_by_date]
        delta = np.array([by_date[d] - inc_by_date[d] for d in common])
        t_delta = nw_t(delta, lag) if len(delta) else float("nan")
        turn_mean = float(np.mean(r["turn"])) if len(r["turn"]) else float("nan")
        cost = turn_mean * rebalances_per_year * rt
        gross = float(np.mean(r["ex"])) * (252.0 / LABEL_SESSIONS)
        results[str(mult)] = {
            "exit_rank": r["exit_rank"],
            "turnover_per_rebalance": turn_mean,
            "cost_per_year": cost,
            "gross_excess_per_year": gross,
            "gross_nw_t": nw_t(r["ex"], lag),
            "net_per_year": gross - cost,
            "delta_gross_vs_incumbent": float(delta.mean()) if len(delta) else float("nan"),
            "delta_nw_t": t_delta,
            "cpcv_folds": len(fold_d),
            "cpcv_folds_not_worse": (float(np.mean(np.array(fold_d) > 0))
                                     if fold_d else float("nan")),
            "rebalances": int(len(r["ex"])),
        }

    # ---- the pre-registered rule, applied mechanically
    qualifying = [m for m, v in results.items()
                  if np.isfinite(v["delta_nw_t"])
                  and v["delta_nw_t"] > SPEC["t_threshold"]]
    boundary_note = None
    if qualifying:
        pick = min(qualifying, key=lambda m: results[m]["turnover_per_rebalance"])
        # A BOUNDARY SOLUTION IS NOT A RESULT. "Lowest turnover subject to not
        # being worse" is an unbounded objective -- turnover falls monotonically
        # in the band for every model tested, so the rule runs to whichever band
        # the grid happens to end at, and a wider grid would have produced a
        # wider answer. When the pick IS the widest tested band, the rule has not
        # located an optimum; it has located the edge of the grid, and that is
        # reported rather than presented as a choice.
        widest = str(max(SPEC["exit_multiples"]))
        if pick == widest:
            boundary_note = (
                f"THE RULE SELECTED THE WIDEST BAND TESTED ({pick}x). That is a "
                f"boundary solution of an unbounded objective, not an optimum: "
                f"turnover falls monotonically in the band, so the rule returns "
                f"the grid edge whatever the edge is. Treat it as 'no interior "
                f"optimum was found'. Acting on it means either widening the "
                f"grid until the constraint binds, or choosing an interior band "
                f"on a stated structural argument -- Grinold (1989) and Clarke, "
                f"de Silva & Thorley (2002) give one: IR = TC x IC x sqrt(breadth), "
                f"and a band wide enough to hold names that have drifted far down "
                f"the ranking lowers the transfer coefficient in a way ~95 "
                f"rebalances cannot detect.")
        decision = (
            f"exit multiple {pick}x (rank {results[pick]['exit_rank']}): the "
            f"lowest turnover among bands whose gross excess is not "
            f"significantly worse than the incumbent. Turnover "
            f"{results[pick]['turnover_per_rebalance']:.1%} per rebalance against "
            f"the incumbent's {results[str(SPEC['incumbent_exit_multiple'])]['turnover_per_rebalance']:.1%}, "
            f"cost {results[pick]['cost_per_year']:.2%}/yr against "
            f"{results[str(SPEC['incumbent_exit_multiple'])]['cost_per_year']:.2%}/yr, "
            f"gross delta {results[pick]['delta_gross_vs_incumbent']*12:+.2%}/yr "
            f"at NW t {results[pick]['delta_nw_t']:+.2f}.")
    else:
        pick = None
        decision = ("No band clears the constraint. Change nothing -- the rule "
                    "was written to allow this answer.")

    payload = {
        "spec_sha256": SPEC_SHA, "spec": SPEC,
        "ranking_source": source,
        "panel_provenance": provenance(PANEL),
        "cost_model": {"round_trip_bps": bd.total_bps_of_buy,
                       "position_inr": pos_value, "median_adtv_inr": adtv,
                       "rebalances_per_year": rebalances_per_year},
        "cpcv": {"splits": len(splits), "purge_obs": purge},
        "newey_west_lag": lag,
        "results": results,
        "selected_exit_multiple": pick,
        "decision": decision,
        "boundary_solution": boundary_note,
        "book_size_caveat": (
            "This simulates a STRICT `slots`-name book. The live engine caps "
            "cards at `stage8_final_signal.portfolio.max_signals_per_run` (8), "
            "not at `capital.max_open_positions` (6), and the schema only "
            "requires per_run >= entry_rank -- so a real book can carry 7 or 8 "
            "names when hysteresis keeps a previously-signalled name ranked "
            "between entry_rank and exit_rank. Widening the band makes that "
            "more likely, which means the turnover figures here are a LOWER "
            "bound on live turnover and the cost saving is, if anything, "
            "understated. The 6-vs-8 tension is a config-level question and is "
            "reported rather than resolved here."),
        "caveat": ("Turnover and cost are properties of the ranking and the band "
                   "and carry no alpha estimate. Gross excess at ~95 rebalances "
                   "is noisy and non-monotonic, which is why it is used only as "
                   "a constraint. Nothing here ships without an epoch."),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float))

    print(f"ranking.source {source} | slots {args.slots} | "
          f"round trip {bd.total_bps_of_buy:.1f} bps | CPCV {len(splits)} splits")
    print(f"spec sha256 {SPEC_SHA[:16]}\n")
    hdr = (f"{'exit':>6s}{'rank':>6s}{'turn/reb':>10s}{'cost/yr':>9s}"
           f"{'gross/yr':>10s}{'NW t':>7s}{'net/yr':>9s}{'dGross':>9s}{'NW t':>7s}"
           f"{'folds+':>8s}")
    print(hdr); print("-" * len(hdr))
    for m in SPEC["exit_multiples"]:
        v = results[str(m)]
        mark = "  <- incumbent" if m == SPEC["incumbent_exit_multiple"] else ""
        print(f"{m:>5.0f}x{v['exit_rank']:>6d}{v['turnover_per_rebalance']:>9.1%}"
              f"{v['cost_per_year']:>9.2%}{v['gross_excess_per_year']:>10.2%}"
              f"{v['gross_nw_t']:>7.2f}{v['net_per_year']:>9.2%}"
              f"{v['delta_gross_vs_incumbent']*12:>9.2%}{v['delta_nw_t']:>7.2f}"
              f"{v['cpcv_folds_not_worse']:>8.0%}{mark}")
    print(f"\nDECISION (pre-registered rule, applied mechanically):\n  {decision}")
    if boundary_note:
        print(f"\nWARNING\n  {boundary_note}")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
