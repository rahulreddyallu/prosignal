"""H4 -- the R-multiple label.

`research/REPAIR_LOG.md` Job 1 turned the engine-geometry label off for a
correct reason: under that geometry a winner realises `+3 x (2.5·ATR/P)` and a
loser `-1 x (2.5·ATR/P)`, so `|label|` is proportional to the name's volatility
and any volatility-correlated feature predicts it by construction.

The fix chosen -- revert to the plain 63-session forward return -- removes the
artefact by discarding the geometry. `work/frontier.py` H3 measures what that
costs: the book's median trade lasts 26 sessions and 84% of positions exit
before the horizon, so the model is ranking names by an outcome the book does
not experience.

Dividing the same outcome by its own stop distance removes the proportionality
EXACTLY -- a winner is +3R and a loser -1R for every name, whatever its
volatility -- and keeps the label equal to the trade the engine takes. That is
the hypothesis. Three labels, one panel, same rows, same features, same
estimator, same splits.

Scoring note. The three labels are not comparable to each other on their own
units, so none of them is scored on itself. Every arm is fitted on its own
label and then scored by RUNNING THE BOOK -- the shipped simulator, at shipped
settings, proven equal to `portfolio_sim` by `work/test_parity.py`. The number
compared across arms is rupee return per period against the equal-weight
eligible universe, which is the only unit all three share.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.data.store import DataStore
from prosignal.features import crossmodel as cm
from prosignal.features.crosssec import build_panel, liquidity_mask
from prosignal.features.exits import rules_from_config
from prosignal.features.crosssec import cross_sectional_rank
from prosignal.stages._cfg import fv, iv
from prosignal.validation.significance import newey_west_t

from book import phases
from decompose import benchmark, build_rankings
from frontier import shipped_book

CACHE = Path(__file__).resolve().parent / "cache"
STEP = 21


def engine_panel(cfg, px):
    """The panel `build_panel` produces when the exit geometry is switched ON.

    This is the branch `triple_barrier: false` currently disables. Note what
    comes with it and is not optional: `resolve_exits` applies
    `tradeable_at_entry`, so this panel is drawn from the population the book
    can actually open -- which the shipped panel is not (H5).
    """
    c4, u = cfg.params.stage4_core_score, cfg.params.universe
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    end = sessions[-iv(cfg.params.validation.holdout.reserve_most_recent_sessions)]

    close, high, low, open_ = px["close"], px["high"], px["low"], px["open"]
    turnover = close * px["volume"]
    from prosignal.data.types import DATE, SYMBOL
    dl = store.read_delivery(start=sessions[0], end=end)
    delivery = None
    if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
        dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
        delivery = dl.pivot_table(index=DATE, columns=SYMBOL, values="deliv_pct",
                                  aggfunc="last", observed=True).sort_index()
    del dl
    sectors = store.read_sector_map()
    sector_map = (dict(zip(sectors["symbol"], sectors["sector"]))
                  if sectors is not None and not sectors.empty else {})
    eligible = liquidity_mask(
        close, turnover, min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names),
        min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr))
    rules = rules_from_config(c4, cfg.params.stage7_risk)
    panel = build_panel(close, turnover, horizon=iv(c4.model_horizon_sessions),
                        step=STEP, delivery=delivery, eligible=eligible,
                        sectors=sector_map, barriers=None, exit_rules=rules,
                        high=high, low=low, open_=open_)
    try:
        actions = store.read_corporate_actions()
    except Exception:
        actions = None
    panel = cm._attach_fundamentals(panel, store.read_statements(), close,
                                    iv(c4.max_fundamental_age_days),
                                    actions=actions)
    panel, features, dropped = cm.prepare_features(panel)
    return panel, features, rules


def add_r_multiple(panel, px, rules):
    """label_r = realised return / the position's own stop distance.

    The stop fraction is `stage7_risk`'s own, clipped exactly as Stage 7 clips
    it, read at the DECISION date. So a stopped position is -1R and a target
    is +3R for every name in the cross-section, and the proportionality Job 1
    identified is gone by arithmetic rather than by assumption.
    """
    close, atr = px["close"], px["atr"]
    dates = pd.to_datetime(panel["date"].unique())
    frames = []
    for d in dates:
        if d not in close.index:
            continue
        e = close.loc[d]
        a = atr.loc[d]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = rules.stop_atr_multiple * a / e * 100.0
        dist = np.clip(raw, rules.min_stop_distance_pct,
                       rules.max_stop_distance_pct) / 100.0
        frames.append(pd.DataFrame({"date": d, "symbol": dist.index,
                                    "stop_frac": dist.to_numpy(),
                                    "atr_pct": (a / e).to_numpy()}))
    ref = pd.concat(frames, ignore_index=True)
    panel = panel.merge(ref, on=["date", "symbol"], how="left")
    panel["label_r"] = panel["label"] / panel["stop_frac"]
    return panel


def artefact(panel, col) -> float:
    """Mean within-date correlation between |label| and ATR/price.

    Job 1's finding, restated as a number that can be checked on any label.
    """
    vals = []
    for _, g in panel.groupby("date"):
        a = g[col].abs().to_numpy("float64")
        v = g["atr_pct"].to_numpy("float64")
        ok = np.isfinite(a) & np.isfinite(v)
        if ok.sum() < 30 or a[ok].std() == 0 or v[ok].std() == 0:
            continue
        vals.append(float(np.corrcoef(a[ok], v[ok])[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def themes(panel, features, cfg, horizon):
    from prosignal.features.crossmodel import fit_coefficients
    c4 = cfg.params.stage4_core_score
    _fit, fm, why = fit_coefficients(
        panel, features, estimator=str(c4.estimator.method),
        alpha=fv(c4.model_ridge_alpha), horizon=horizon, step=STEP)
    if fm is None:
        return {}, why
    out = {}
    for name in features:
        lam = fm.lam.get(name)
        t = fm.t_stat.get(name)
        if t is not None:
            out[name] = (float(lam) if lam is not None else float("nan"), float(t))
    return out, why


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    horizon = iv(cfg.params.stage4_core_score.model_horizon_sessions)
    with open(CACHE / "panels.pkl", "rb") as fh:
        px = pickle.load(fh)

    cache = CACHE / "engine_panel.pkl"
    if cache.is_file():
        with open(cache, "rb") as fh:
            ep, features, rules = pickle.load(fh)
        print(f"engine panel from cache: {len(ep):,} rows / "
              f"{ep['date'].nunique()} dates")
    else:
        print("building the engine-geometry panel (this is the slow part) ...",
              flush=True)
        ep, features, rules = engine_panel(cfg, px)
        ep = add_r_multiple(ep, px, rules)
        with open(cache, "wb") as fh:
            pickle.dump((ep, features, rules), fh, protocol=4)
        print(f"  {len(ep):,} rows / {ep['date'].nunique()} dates")

    with open(CACHE / "research.pkl", "rb") as fh:
        hp = pickle.load(fh)["panel"]
    # The horizon-return panel needs atr_pct too, to answer H4a on all three.
    hp = add_r_multiple(hp, px, rules).rename(columns={"label_r": "_unused"})

    print(f"\nrows: horizon-return panel {len(hp):,}  "
          f"engine-geometry panel {len(ep):,}  "
          f"({len(ep)/len(hp) - 1:+.1%})")
    print("The engine-geometry panel is smaller because `resolve_exits` refuses "
          "names sitting below their own invalidation level -- the population "
          "the book can open. The shipped panel does not (H5).")

    print("\nH4a  |label| vs ATR/price, mean within-date correlation")
    print(f"  {'horizon return (SHIPPED)':<34}{artefact(hp, 'label'):>+8.3f}")
    print(f"  {'engine geometry, rupee return':<34}{artefact(ep, 'label'):>+8.3f}")
    print(f"  {'engine geometry, R-multiple':<34}{artefact(ep, 'label_r'):>+8.3f}")

    print("\nH4b  theme coefficients and t, fitted on the full selection panel")
    variants = {
        "horizon return (SHIPPED)": (hp, "label"),
        "engine geometry, rupee": (ep, "label"),
        "engine geometry, R-multiple": (ep, "label_r"),
    }
    names = sorted({f for f in features})
    print(f"  {'theme':<14}" + "".join(f"{k[:22]:>26}" for k in variants))
    fits = {}
    for k, (p, col) in variants.items():
        q = p.copy()
        q["label"] = q[col]
        q["label_rank"] = q.groupby("date")["label"].transform(cross_sectional_rank)
        q = q[np.isfinite(q["label"])]
        fits[k] = (q, themes(q, features, cfg, horizon)[0])
    for n in names:
        row = f"  {n:<14}"
        for k in variants:
            lam, t = fits[k][1].get(n, (float("nan"), float("nan")))
            mark = "*" if abs(t) >= 2.0 else " "
            row += f"{lam:>+16.4f}{t:>+8.2f}{mark:<2}"
        print(row)

    print("\nH4c  the book, run on each label's rankings")
    b0 = shipped_book(cfg, horizon)
    results = {}
    hdr = (f"  {'label':<32}{'dates':>6}{'net':>9}{'vs bench':>10}{'t':>7}"
           f"{'sharpe':>8}{'worst dd':>10}")
    for construction in ("weave", "forward"):
        print(f"\n  --- {construction} ---")
        print(hdr)
        for k, (q, _) in fits.items():
            rk = build_rankings(q, features, cfg)
            rankings = rk[0] if construction == "weave" else rk[1]
            work = rk[2]
            if len(rankings) < 6:
                print(f"  {k:<32}{len(rankings):>6}   too few dates")
                continue
            bench = benchmark(work, rankings, b0, px)
            bm = float(bench.mean())
            m = phases(b0, rankings, px)
            try:
                t = newey_west_t(m["returns"] - bm, horizon_sessions=horizon,
                                 step_sessions=STEP).adjusted_t
            except ValueError:
                t = float("nan")
            print(f"  {k:<32}{len(rankings):>6}{m['mean_return']:>9.2%}"
                  f"{m['mean_return'] - bm:>10.2%}{t:>7.2f}{m['sharpe']:>8.2f}"
                  f"{m['drawdown_worst_schedule']:>10.1%}")
            results.setdefault(construction, {})[k] = {
                "n_dates": len(rankings), "bench": bm,
                "mean": m["mean_return"], "vs_bench": m["mean_return"] - bm,
                "t": t, "sharpe": m["sharpe"],
                "dd": m["drawdown_worst_schedule"]}

    with open(CACHE / "h4.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
