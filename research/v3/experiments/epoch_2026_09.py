"""EPOCH CANDIDATE HARNESS -- the 2026-09 factor audit's model-level findings.

This script does NOT change the shipped model. `features/v3.py` holds signs,
weights and theme membership; they are hashed into `config_sha256`, matched by
`tests/test_v3_score.py` and tied to two sealed holdouts. Anything here that
looks good has to go through a new epoch -- re-fit, re-seal, re-register the
forward test -- exactly as `docs/AUDIT_REMEDIATION_2026_09.md` requires.

WHAT IT MEASURES, and why in this order.

The audit's drop-one table was computed on the whole panel, which is the same
data the reader then chooses a prune from. That is selection on the test set and
it is why the audit reported the prune as directional rather than proven. This
harness answers the same questions with the machinery the repo already has for
exactly this problem:

  CPCV       Combinatorial Purged CV over the panel's signal dates, with the
             label window purged from training and an embargo after each test
             block. Every specification is scored on the SAME folds, so the
             comparison is paired and a fold that was simply easy cannot
             flatter one variant over another.
  paired dt  Per-fold delta against the incumbent, then the mean of the deltas
             with a Newey-West t across folds. The level of IC is not the
             question -- the incumbent's own IC is partly in-sample, since signs
             and weights were fit on 2018-11..2024-10. The DIFFERENCE between two
             specifications evaluated on the same rows does not inherit that.

THE SPECIFICATIONS ARE FIXED BEFORE THE RUN. They are listed in SPECS below,
each with the audit finding that motivates it, and none was chosen by looking at
a CPCV number. SPEC_SHA is the hash of that list; a spec added after seeing
results changes the hash and the run is no longer pre-registered.

WHAT IT CANNOT DO. It measures the RANKING. The audit's central finding is that
the ranking is not the problem -- the concentrated book is -- and a book test
needs the portfolio simulator, non-overlapping cohorts and the real cost model.
`book_sim.py` is that; SPEC-E is included here only at the ranking level and is
explicitly NOT settled by anything this script prints.

Usage:
    python research/v3/experiments/epoch_2026_09.py
    python research/v3/experiments/epoch_2026_09.py --groups 8 --test-groups 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prosignal.features import v3                              # noqa: E402
from prosignal.validation.cpcv import CombinatorialPurgedCV     # noqa: E402
from prosignal.validation.significance import overlap_lag       # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _panel_guard import provenance, require_fresh                        # noqa: E402

HERE = Path(__file__).resolve().parent
PANEL = HERE / "panel_2026_09.parquet"
CANDIDATES = HERE / "candidates_2026_09.parquet"     # optional, see --with-candidates
OUT = HERE / "epoch_2026_09.json"

LABEL = "y21"
#: The panel is sampled every 5 sessions against a 21-session label, so a row
#: shares its outcome window with the next ~4 rows. That is the purge distance
#: in PANEL OBSERVATIONS, not in sessions.
STRIDE_SESSIONS = 5
LABEL_SESSIONS = 21

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED SPECIFICATIONS
# ---------------------------------------------------------------------------
#: The seven factors that an independent split-half selection nominated for
#: removal in BOTH halves of the panel. Chosen by that agreement and not by the
#: full-sample drop-one ranking, which nominated nine.
BOTH_HALVES_SEVEN = ("mom_2_0", "mom_3_1", "mom_accel", "voladj_mom_6_1",
                     "ulcer_120", "resid_rev_21", "deliv_chg_5")

#: Uninvestable on turnover alone at the shipped 89.6 bps round trip, whatever
#: their IC: implied annual drag 24.2%, 13.9%, 9.7% and 6.8%.
COST_INFEASIBLE = ("rev_1w", "deliv_chg_5", "price_vs_vwap_20", "resid_rev_21")

SPECS = {
    "incumbent": {
        "note": "features/v3.py as shipped. The baseline every delta is against.",
        "drop_factors": (), "move": {}, "weights": None,
    },
    "A_prune7": {
        "note": "Audit item 4. Remove the seven factors both halves independently "
                "nominated. Four of them are cost-infeasible regardless of IC.",
        "drop_factors": BOTH_HALVES_SEVEN, "move": {}, "weights": None,
    },
    "B_ulcer_to_momentum": {
        "note": "Audit item 5, variant 1. `ulcer_120` correlates +0.74..+0.78 "
                "oriented with prox_52w across the momentum/risk boundary, so the "
                "40% momentum cap does not bind what it is meant to bind. Move it "
                "into momentum and let the cap see it. Nothing is deleted.",
        "drop_factors": (), "move": {"ulcer_120": "momentum"}, "weights": None,
    },
    "C_prune7_no_ulcer": {
        "note": "Audit item 5, variant 2. The prune already removes ulcer_120; "
                "kept as a named spec so 'move it' and 'drop it' are compared "
                "against the same baseline rather than against each other.",
        "drop_factors": BOTH_HALVES_SEVEN, "move": {}, "weights": None,
    },
    "D_cost_only": {
        "note": "The conservative alternative to A: remove ONLY the four factors "
                "whose own turnover exceeds any plausible gross edge. This needs "
                "no IC evidence to justify -- it is an execution constraint -- so "
                "it is the spec to prefer if A's advantage does not survive CPCV.",
        "drop_factors": COST_INFEASIBLE, "move": {}, "weights": None,
    },
    "E_equal_theme": {
        "note": "Audit item 2 follow-up. Declared weights run at 48/4/21/13/13 "
                "after coverage renormalisation, nothing like the declared "
                "40/19/19/11/11. Equal declared weight is the neutral alternative "
                "to a weight vector nobody chose. EXP-D previously found frozen "
                "beats equal on the full sample; this re-asks it out of sample.",
        "drop_factors": (), "move": {}, "weights": {t: 0.2 for t in v3.THEMES},
    },
    "F_prune7_equal_theme": {
        "note": "A and E together, because they interact: pruning changes what "
                "each theme contains, which changes what an equal weight buys.",
        "drop_factors": BOTH_HALVES_SEVEN, "move": {},
        "weights": {t: 0.2 for t in v3.THEMES},
    },
}

# ---------------------------------------------------------------------------
# ROUND 2 -- FORMED AFTER SEEING ROUND 1. Hashed separately and reported
# separately, because a specification written after the results are on the
# screen is not pre-registered and saying otherwise would be the whole problem
# this file exists to avoid. Round 1 said:
#
#   A/C  prune the seven      dIC +0.0057, NW t +1.99, 91% of folds positive
#   B    move ulcer_120       dIC +0.0016, NW t +2.39, 87% of folds positive
#   D    cost-only removal    dIC -0.0005, NW t -0.22, 44% of folds positive
#   E/F  equal theme weight   positive mean, weak t, p5 -0.009 / -0.017
#
# and the two questions it left are whether B combines with the cost removals
# (the change that needs no IC evidence to justify), and whether the one
# candidate factor that came close to adding -- `idio_vol_120`, own IC +0.0396
# at NW t +2.84 -- survives once the factors that span it are gone.
ROUND2_SPECS = {
    "G_ulcer_move_plus_cost": {
        "note": "B and D together. This is the epoch to prefer if the goal is "
                "the smallest change that is defensible on grounds other than a "
                "number selected on this panel: moving ulcer_120 is forced by the "
                "correlation structure, and the four removals are forced by "
                "turnover. Neither was chosen by an IC search.",
        "drop_factors": COST_INFEASIBLE, "move": {"ulcer_120": "momentum"},
        "weights": None,
    },
    "H_prune7_plus_ulcer_move": {
        "note": "A already deletes ulcer_120, so this asks the opposite: prune "
                "the other six and MOVE ulcer_120 rather than dropping it. If H "
                "matches A, the prune's gain is not about ulcer_120 at all.",
        "drop_factors": tuple(f for f in BOTH_HALVES_SEVEN if f != "ulcer_120"),
        "move": {"ulcer_120": "momentum"}, "weights": None,
    },
}

ROUND2_SHA = hashlib.sha256(
    json.dumps({k: {kk: (sorted(vv.items()) if isinstance(vv, dict) else vv)
                    for kk, vv in v.items() if kk != "note"}
                for k, v in ROUND2_SPECS.items()}, sort_keys=True, default=str
               ).encode()).hexdigest()

SPEC_SHA = hashlib.sha256(
    json.dumps({k: {kk: (sorted(vv.items()) if isinstance(vv, dict) else vv)
                    for kk, vv in v.items() if kk != "note"}
                for k, v in SPECS.items()}, sort_keys=True, default=str
               ).encode()).hexdigest()


# ---------------------------------------------------------------------------
def _themes_for(spec) -> dict:
    """The theme table this specification implies, built from the frozen one.

    `v3.THEMES` is never mutated. A moved factor keeps its sign and changes
    theme; a dropped factor leaves its theme, and a theme emptied by the drop is
    removed rather than left to divide by zero.
    """
    move = spec.get("move") or {}
    drop = set(spec.get("drop_factors") or ())
    members: dict = {t: [] for t in v3.THEMES}
    for tname, th in v3.THEMES.items():
        for fname, sign in th.factors:
            if fname in drop:
                continue
            members[move.get(fname, tname)].append((fname, sign))
    out = {}
    for tname, th in v3.THEMES.items():
        if not members[tname]:
            continue
        out[tname] = v3.Theme(weight=th.weight, horizon=th.horizon,
                              coverage=th.coverage, factors=tuple(members[tname]))
    return out


def build_scores(panel, spec, rank_frame, idx, dates, min_themes=3) -> np.ndarray:
    """Rebuild the composite under one specification, per date.

    THE SUB-SCORE COMES FROM `v3.theme_subscore`, not from a copy of it. A first
    version of this re-implemented the sign-oriented mean and the re-rank in
    numpy and reproduced the shipped score to 5e-3 -- two names out of 750
    breaking a tie the other way. That is small enough to pass any eyeball check
    and large enough to be a different model, and the whole point of the guard in
    `main` is that a harness measuring a near-copy is measuring nothing. The
    blend below is the only part written here, because it is the part the
    specifications change.
    """
    themes = _themes_for(spec)
    names = list(themes)
    W = np.array([(spec.get("weights") or {}).get(t, themes[t].weight)
                  for t in names], dtype="float64")
    score = np.full(len(panel), np.nan)
    for d in dates:
        ix = idx[d]
        block = rank_frame.iloc[ix]
        subs = np.column_stack([
            v3.theme_subscore(block, themes[t]).to_numpy("float64")
            for t in names])
        ok = np.isfinite(subs)
        den = (ok * W).sum(1)
        num = np.nansum(np.where(ok, subs * W, 0.0), 1)
        mt = min(min_themes, len(names))
        score[ix] = np.where((den > 0) & (ok.sum(1) >= mt),
                             num / np.maximum(den, 1e-12), np.nan)
    return score


def per_date_ic(score, y, idx, dates, pop) -> dict:
    """Rank IC per date on a FIXED population -- the names the incumbent ranks.

    Holding the population fixed is what stops a variant being credited for
    ranking a different, easier universe. A spec that scores fewer names is
    reported through `coverage`, not rewarded through IC.
    """
    out = {}
    for d in dates:
        ix = idx[d]
        m = pop[ix] & np.isfinite(score[ix]) & np.isfinite(y[ix])
        if m.sum() < 30:
            continue
        a = pd.Series(score[ix][m]).rank().to_numpy()
        b = pd.Series(y[ix][m]).rank().to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        out[d] = float(np.corrcoef(a, b)[0, 1])
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
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--test-groups", type=int, default=2)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--allow-stale", action="store_true",
                    help="report even though the store has moved under the "
                         "panel. The numbers then describe the OLD store.")
    args = ap.parse_args()

    if not PANEL.exists():
        print(f"panel not found: {PANEL}\nbuild it with audit_2026_09.py first")
        return 2
    require_fresh(PANEL, allow_stale=args.allow_stale)
    P = pd.read_parquet(PANEL).reset_index(drop=True)
    P["date"] = pd.to_datetime(P["date"])
    dates = sorted(P["date"].unique())
    idx = {d: np.asarray(v) for d, v in P.groupby("date").indices.items()}
    # `theme_subscore` reads COLUMNS NAMED AFTER THE FACTORS, which is what the
    # live path hands it; the panel stores them with an `_r` suffix.
    rank_frame = P[[f + "_r" for f in v3.ALL_FACTORS]].copy()
    rank_frame.columns = list(v3.ALL_FACTORS)
    y = P[args.label].to_numpy("float64")

    base = build_scores(P, SPECS["incumbent"], rank_frame, idx, dates)
    pop = np.isfinite(base)
    stored = P["score"].to_numpy("float64")
    agree = np.isfinite(stored) == pop
    drift = float(np.nanmax(np.abs(stored[pop] - base[pop]))) if pop.any() else float("nan")
    if not agree.all() or not (drift < 1e-9):
        print(f"REFUSING TO REPORT: the rebuild does not reproduce the stored "
              f"score (max |diff| {drift:.3g}, population agrees on "
              f"{agree.mean():.4%} of rows). Every delta below would be measured "
              f"against a model that is not the shipped one.")
        return 3

    # PURGE IN PANEL OBSERVATIONS. The panel is one row per (date, symbol) but
    # CPCV splits DATES, so the horizon is expressed in signal dates: a
    # 21-session label sampled every 5 sessions overlaps the next ceil(21/5)=5.
    purge = int(np.ceil(LABEL_SESSIONS / STRIDE_SESSIONS))
    cv = CombinatorialPurgedCV(n_groups=args.groups, n_test_groups=args.test_groups,
                               label_horizon=purge, embargo=purge)
    splits = list(cv.split(len(dates)))
    lag = overlap_lag(LABEL_SESSIONS, STRIDE_SESSIONS)

    all_specs = dict(SPECS)
    all_specs.update(ROUND2_SPECS)
    ic_by_spec, cov_by_spec = {}, {}
    for name, spec in all_specs.items():
        sc = build_scores(P, spec, rank_frame, idx, dates)
        ic_by_spec[name] = per_date_ic(sc, y, idx, dates, pop)
        cov_by_spec[name] = float(np.isfinite(sc[pop]).mean())

    results = {}
    base_ic = ic_by_spec["incumbent"]
    for name, spec in all_specs.items():
        fold_ic, fold_delta = [], []
        for sp in splits:
            test_dates = [dates[i] for i in sp.test_idx]
            v = [ic_by_spec[name][d] for d in test_dates if d in ic_by_spec[name]]
            b = [base_ic[d] for d in test_dates if d in base_ic]
            if len(v) < 8 or len(b) < 8:
                continue
            fold_ic.append(float(np.mean(v)))
            common = [d for d in test_dates
                      if d in ic_by_spec[name] and d in base_ic]
            fold_delta.append(float(np.mean([ic_by_spec[name][d] - base_ic[d]
                                             for d in common])))
        allv = np.array(list(ic_by_spec[name].values()))
        d_all = np.array([ic_by_spec[name][d] - base_ic[d]
                          for d in base_ic if d in ic_by_spec[name]])
        results[name] = {
            "note": spec["note"],
            "drop_factors": list(spec.get("drop_factors") or ()),
            "move": spec.get("move") or {},
            "equal_theme_weights": spec.get("weights") is not None,
            "full_ic": float(allv.mean()),
            "full_ic_nw_t": nw_t(allv, lag),
            "full_delta": float(d_all.mean()) if len(d_all) else float("nan"),
            "full_delta_nw_t": nw_t(d_all, lag) if len(d_all) else float("nan"),
            "cpcv_folds": len(fold_ic),
            "cpcv_ic_median": float(np.median(fold_ic)) if fold_ic else float("nan"),
            "cpcv_delta_mean": float(np.mean(fold_delta)) if fold_delta else float("nan"),
            "cpcv_delta_median": float(np.median(fold_delta)) if fold_delta else float("nan"),
            "cpcv_folds_positive": (float(np.mean(np.array(fold_delta) > 0))
                                    if fold_delta else float("nan")),
            "cpcv_delta_p5": (float(np.percentile(fold_delta, 5))
                              if fold_delta else float("nan")),
            "coverage_of_incumbent_population": cov_by_spec[name],
            "round": 2 if name in ROUND2_SPECS else 1,
        }

    payload = {
        "spec_sha256": SPEC_SHA,
        "panel_provenance": provenance(PANEL),
        "round2_spec_sha256": ROUND2_SHA,
        "round2_is_not_pre_registered": (
            "ROUND2_SPECS were written after round 1's results were visible. "
            "Their t-statistics carry that multiplicity and must not be read as "
            "pre-registered evidence."),
        "what_cpcv_does_and_does_not_buy_here": (
            "The specifications have NO FITTED PARAMETERS -- signs, weights and "
            "membership are fixed by hand -- so there is nothing to re-estimate "
            "per fold and CPCV cannot make a selection out-of-sample. What it "
            "measures is STABILITY: whether the advantage holds across 45 "
            "purged, embargoed sub-periods rather than resting on one stretch. "
            "`cpcv_folds_positive` and `cpcv_delta_p5` are the columns that "
            "carry information the full-window delta does not. Specs A, C, F and "
            "H additionally inherit the split-half selection that chose the "
            "seven factors from THIS panel; B, D, E and G do not, which makes "
            "B and G the cleaner reads."),
        "label": args.label,
        "panel": {"rows": int(len(P)), "dates": len(dates),
                  "start": str(pd.Timestamp(dates[0]).date()),
                  "end": str(pd.Timestamp(dates[-1]).date())},
        "cpcv": {"n_groups": args.groups, "n_test_groups": args.test_groups,
                 "n_splits": len(splits), "purge_obs": purge, "embargo_obs": purge,
                 "paths_per_observation": cv.paths_per_observation()},
        "newey_west_lag": lag,
        "baseline_reproduces_shipped_score": True,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"panel {len(P):,} rows, {len(dates)} dates "
          f"{payload['panel']['start']}..{payload['panel']['end']}")
    print(f"CPCV {len(splits)} splits, purge/embargo {purge} signal dates, "
          f"NW lag {lag}")
    print(f"round-1 spec sha256 {SPEC_SHA[:16]}  "
          f"round-2 (post-hoc) {ROUND2_SHA[:16]}\n")
    hdr = (f"{'spec':22s}{'full IC':>10s}{'NW t':>7s}{'dIC':>10s}{'NW t':>7s}"
           f"{'CPCV dIC':>11s}{'folds+':>8s}{'p5':>10s}{'cov':>7s}")
    print(hdr)
    print("-" * len(hdr))
    printed_round2 = False
    for name, r in results.items():
        if r["round"] == 2 and not printed_round2:
            print("-- round 2, formed after seeing round 1, NOT pre-registered --")
            printed_round2 = True
        print(f"{name:22s}{r['full_ic']:>10.5f}{r['full_ic_nw_t']:>7.2f}"
              f"{r['full_delta']:>10.5f}{r['full_delta_nw_t']:>7.2f}"
              f"{r['cpcv_delta_mean']:>11.5f}{r['cpcv_folds_positive']:>8.0%}"
              f"{r['cpcv_delta_p5']:>10.5f}{r['coverage_of_incumbent_population']:>7.1%}")
    print(f"\nwritten {OUT}")
    print("\nNOTHING HERE SHIPS. A spec that wins still needs a new epoch: "
          "re-fit, re-seal a holdout, re-register the forward test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
