"""Reconstruct the Deflated Sharpe under each of the four constructions the
dossier's section E tabulates, on the current tree's own evidence.

The claim under test is narrow and checkable: section E says the DSR was
repaired from 1.0000 PASS to 0.3463 FAIL by (a) scoring distinct dates rather
than pooled (split, date) pairs, (b) reducing to independent 63-session windows,
and (c) taking Var[SR] from the woven path Sharpes instead of the 1/(n-1)
fallback.  `prosignal research cpcv` on this tree prints 1.0000 PASS.

Either the repair is absent, or the dossier's arithmetic is wrong.  Running all
four constructions on one set of scores settles which.

Nothing here changes the engine.  It reads `run_cpcv`'s own loop, re-emitting
the per-(split, date) scores that `CpcvResult` flattens into `excess`.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.features.linear import predict
from prosignal.stages._cfg import fv, iv
from prosignal.validation.cpcv import CombinatorialPurgedCV
from prosignal.validation.harness import _fit, _rank_ic
from prosignal.validation.metrics import (deflated_sharpe_ratio,
                                          expected_max_sharpe, sharpe_ratio)

CACHE = Path(__file__).resolve().parent / "cache"


def scored_splits(panel, features, *, horizon, step, alpha, n_groups,
                  n_test_groups, purge_sessions, embargo_sessions,
                  estimator, min_train_rows=2000, top_decile=0.90):
    """Every (split, date, excess, ic) the harness would pool into `excess`.

    A copy of `run_cpcv`'s scoring loop that keeps the split and the date
    attached, because the identity of an observation is exactly what the
    pooled vector throws away.
    """
    cols = [c for c in features if c in panel.columns]
    work = panel.dropna(subset=cols + ["label_rank", "label"]).reset_index(drop=True)
    dates = sorted(work["date"].unique())
    purge_obs = int(np.ceil(purge_sessions / step))
    embargo_obs = int(np.ceil(embargo_sessions / step))
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups,
                               label_horizon=purge_obs, embargo=embargo_obs)
    by_date = {d: g for d, g in work.groupby("date")}
    rows = []
    for n, split in enumerate(cv.split(len(dates)), start=1):
        train = work[work["date"].isin([dates[i] for i in split.train_idx])]
        if len(train) < min_train_rows:
            continue
        fit = _fit(train, cols, alpha, estimator, horizon, step)
        if fit is None:
            continue
        for d in [dates[i] for i in split.test_idx]:
            te = by_date[d]
            pred = predict(fit, te[cols].to_numpy("float64"))
            ic = _rank_ic(pred, te["label_rank"].to_numpy("float64"))
            lab = te["label"].to_numpy("float64")
            ok = np.isfinite(pred) & np.isfinite(lab)
            if ok.sum() < 40:
                continue
            r = pd.Series(pred[ok]).rank(pct=True).to_numpy()
            top = r >= top_decile
            if not top.any():
                continue
            ex = float(lab[ok][top].mean() - lab[ok].mean())
            if not np.isfinite(ex):
                continue
            rows.append({"split": split.split_id, "date": pd.Timestamp(d),
                         "excess": ex, "ic": ic})
        if n % 10 == 0:
            print(f"  split {n}/{cv.n_splits}", flush=True)
    return pd.DataFrame(rows), cv


def weave(scored: pd.DataFrame):
    """Path Sharpes, the same weave `run_cpcv` performs: the k-th time a date
    is tested it belongs to path k."""
    seen: dict = {}
    paths: dict = {}
    for _, r in scored.iterrows():
        pid = seen.get(r["date"], 0)
        seen[r["date"]] = pid + 1
        paths.setdefault(pid, []).append(float(r["excess"]))
    out = []
    for pid, vals in sorted(paths.items()):
        a = np.asarray(vals, dtype="float64")
        if a.size < 4:
            continue
        out.append(float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0 else 0.0)
    return out


def report(name, series, n_trials, trial_sharpes, bar=0.95):
    d = deflated_sharpe_ratio(series, n_trials=n_trials,
                              trial_sharpes=trial_sharpes)
    arr = np.asarray(list(series), dtype="float64")
    arr = arr[np.isfinite(arr)]
    sr_var = (float(np.var(np.asarray(trial_sharpes, dtype='float64'), ddof=1))
              if trial_sharpes is not None and len(trial_sharpes) > 1
              else 1.0 / max(arr.size - 1, 1))
    print(f"{name:<46} {arr.size:>6}  {sharpe_ratio(arr):>+8.4f} "
          f"{sr_var:>10.5f} {expected_max_sharpe(n_trials, sr_var):>9.4f} "
          f"{d.deflated_sr:>8.4f}  {'PASS' if d.deflated_sr >= bar else 'FAIL'}")
    return d


def main() -> int:
    cfg = load_config()
    val, c4 = cfg.params.validation, cfg.params.stage4_core_score
    with open(CACHE / "research.pkl", "rb") as fh:
        rp = pickle.load(fh)
    panel, features, horizon = rp["panel"], rp["features"], rp["horizon"]
    step = 21
    print(f"panel {len(panel):,} rows / {panel['date'].nunique()} dates; "
          f"horizon {horizon}, step {step}")

    scored, cv = scored_splits(
        panel, features, horizon=horizon, step=step,
        alpha=fv(c4.model_ridge_alpha), n_groups=iv(val.cpcv.n_groups),
        n_test_groups=2, purge_sessions=iv(val.cpcv.purge_sessions),
        embargo_sessions=iv(val.cpcv.embargo_sessions),
        estimator=str(c4.estimator.method))
    scored.to_parquet(CACHE / "cpcv_scored.parquet")

    pooled = scored["excess"].to_numpy("float64")
    distinct = scored.groupby("date")["excess"].mean().sort_index()
    paths = weave(scored)
    n_trials = 81

    print(f"\nscored (split, date) pairs : {len(scored)}")
    print(f"distinct panel dates       : {scored['date'].nunique()}")
    print(f"woven paths                : {len(paths)}  "
          f"Var[SR] = {np.var(paths, ddof=1):.5f}")

    # Independent 63-session windows: panel dates are `step` apart, so one in
    # every horizon/step of them is non-overlapping.
    stride = max(int(np.ceil(horizon / step)), 1)
    independent = distinct.iloc[::stride]
    print(f"independent windows        : {len(independent)} "
          f"(every {stride}th distinct date)\n")

    hdr = (f"{'construction':<46} {'n':>6}  {'SR':>8} {'Var[SR]':>10} "
           f"{'E[max]':>9} {'DSR':>8}")
    print(hdr); print("-" * len(hdr))
    report("as shipped -- pooled pairs, sr_var=1/(n-1)", pooled, n_trials, None)
    report("distinct panel dates", distinct.to_numpy(), n_trials, None)
    report("independent 63-session windows", independent.to_numpy(),
           n_trials, None)
    report("windows and Var[SR] from woven paths", independent.to_numpy(),
           n_trials, paths)
    print()
    print("dossier section E, for comparison:")
    print("  as shipped                639   1.0000  pass at 100,000 trials")
    print("  distinct panel dates       71   0.4649  fail")
    print("  independent windows        23   0.1477  fail")
    print("  windows + Var[SR] paths    23   0.3130  fail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
