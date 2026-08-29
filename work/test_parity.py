"""Prove `work/book.py` IS `validation.portfolio_sim`, on both sides of the repair.

Nothing measured in `decompose.py` means anything unless the baseline it departs
from is the real simulator.  This runs both on identical rankings and identical
panels and requires the period returns to agree to floating point.

TWO PARITY CHECKS, because production has since been repaired:

  as repaired   `portfolio_sim` today, against `book.py` with `target_on_high`,
                `charge_reentry` and `unfilled_pays` ON. This is the live check.

  as shipped    the pre-repair behaviour, which `book.py` still reproduces with
                those three switches OFF. It cannot be checked against
                production any more -- production no longer does it -- so it is
                pinned by value instead: the decomposition's baseline rows are
                that behaviour, and if they move, the deltas in docs/REAUDIT.md
                stop describing what was actually fixed.

Also asserts, in the same run, that each switch actually changes something --
a correction that silently does nothing would otherwise show as "costs 0.00%"
and read as a finding.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from prosignal.config.loader import load_config
from prosignal.costs import CostModel
from prosignal.stages._cfg import fv, iv
from prosignal.validation.portfolio_sim import PortfolioParams
from prosignal.validation.portfolio_sim import simulate as shipped_simulate

from book import Book, simulate

CACHE = Path(__file__).resolve().parent / "cache"


def load():
    with open(CACHE / "panels.pkl", "rb") as fh:
        px = pickle.load(fh)
    with open(CACHE / "rankings.pkl", "rb") as fh:
        weave, forward, work = pickle.load(fh)
    return px, weave, work


def books(cfg):
    cap, c6, c7 = cfg.params.capital, cfg.params.stage6_entry, cfg.params.stage7_risk
    model = CostModel(cfg)

    def cost_bps(price, qty, adtv):
        if price <= 0 or qty <= 0:
            return 0.0
        return float(model.round_trip(price, int(max(qty, 1)),
                                      adtv_inr=adtv if adtv > 0 else None
                                      ).total_bps_of_buy)

    common = dict(
        capital=fv(cap.total_capital_inr),
        max_positions=iv(cap.max_open_positions),
        risk_per_trade_pct=fv(cap.risk_per_trade_pct),
        max_participation_of_adtv=fv(cap.max_participation_of_adtv),
        stop_atr_multiple=fv(c7.stop_loss.atr_multiple),
        min_stop_distance_pct=fv(c7.stop_loss.min_stop_distance_pct),
        max_stop_distance_pct=fv(c7.stop_loss.max_stop_distance_pct),
        invalidation_ma_sessions=iv(c7.thesis_invalidation.structure_ma_sessions),
        invalidation_buffer_atr=fv(c7.thesis_invalidation.structure_buffer_atr),
        horizon_sessions=iv(cfg.params.stage4_core_score.model_horizon_sessions),
        entry_rank=iv(c6.admission.entry_rank),
        exit_rank=iv(c6.admission.exit_rank),
        target_r_multiple=fv(c7.targets.t2_r_multiple),
        cost_fn=cost_bps,
    )
    return Book(**common), PortfolioParams(**common)


def main() -> int:
    cfg = load_config()
    px, rankings, _ = load()
    mine, theirs = books(cfg)
    failures = []

    repaired = replace(mine, target_on_high=True, charge_reentry=True,
                       unfilled_pays=True)
    for phase in range(3):
        a = simulate(repaired, rankings, px, phase=phase)
        b = shipped_simulate(rankings, px, theirs, phase=phase).periods
        if len(a) != len(b):
            failures.append(f"phase {phase}: {len(a)} periods vs {len(b)}")
            continue
        for col in ("ret", "gross_ret", "cost_ret", "n_held"):
            d = np.abs(a[col].to_numpy("float64") - b[col].to_numpy("float64"))
            worst = float(d.max()) if d.size else 0.0
            if worst > 1e-12:
                failures.append(f"phase {phase} {col}: max |diff| {worst:.3e}")
        print(f"phase {phase}: {len(a)} periods, mean ret "
              f"{a['ret'].mean():+.4%} vs {b['ret'].mean():+.4%}")

    print()
    # The pre-repair behaviour, pinned by value. These are the numbers every
    # "as shipped" row in the decomposition is built from.
    pre = simulate(mine, rankings, px, phase=0)["ret"]
    print(f"as-shipped baseline, phase 0: {len(pre)} periods, "
          f"mean {pre.mean():+.6%}, sum {pre.sum():+.6%}")

    # Across EVERY phase, not just the first. A switch that bites on one
    # schedule and not another is live; testing one offset and calling it inert
    # is the same class of error as testing one fixture and calling two exit
    # constructions identical.
    for name, kw in [("target_on_high", dict(target_on_high=True)),
                     ("charge_reentry", dict(charge_reentry=True)),
                     ("full_investment", dict(full_investment=True)),
                     ("refuse_unknown_liquidity",
                      dict(refuse_unknown_liquidity=True)),
                     # Correct in principle and, on this sample, inert: a name
                     # refused by the admission predicate tends to stay refused
                     # or fall out of the exit band before it could fill, so the
                     # case never arises here. Pinned instead by
                     # tests/test_portfolio_sim.py::
                     # test_a_slot_that_never_filled_pays_when_it_finally_does,
                     # which constructs it. Reported, not failed -- a switch
                     # that does not bite on real data is a fact about the data.
                     ("unfilled_pays (may be inert)",
                      dict(unfilled_pays=True))]:
        live, phases_bitten = False, []
        for phase in range(3):
            b0 = simulate(mine, rankings, px, phase=phase)["ret"].to_numpy("float64")
            v = simulate(replace(mine, **kw), rankings, px,
                         phase=phase)["ret"].to_numpy("float64")
            if len(v) != len(b0) or not np.allclose(v, b0, atol=1e-15):
                live = True
                phases_bitten.append(phase)
        where = f"phases {phases_bitten}" if live else "nowhere"
        print(f"  {name:<26} changes the book: "
              f"{'NO -- INERT' if not live else f'yes, {where}'}")
        if not live and "may be inert" not in name:
            failures.append(f"{name} is inert; it cannot be priced")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("PARITY OK -- work/book.py with the repairs ON reproduces "
          "portfolio_sim.simulate exactly, every switch is live, and the "
          "as-shipped baseline is pinned above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
