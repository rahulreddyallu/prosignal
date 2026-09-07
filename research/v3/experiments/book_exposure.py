"""Does the rebuilt book stay invested on REAL cross-sections, and what does it turn over?

WHAT THIS IS NOT. It is not a backtest and it charges no trial. It reports two
MECHANICAL properties of `prosignal.book.build_book` -- deployed fraction and
turnover -- over the real universe, the real ADTV distribution and the real
ranking, on every signal date in the panel. Neither is a return, neither is
selected, and nothing here is tuned: the specification is the one written in
docs/REBUILD_2026_09.md 6.2 before this script existed.

WHY IT IS WORTH RUNNING ANYWAY. `tests/test_book.py` proves the construction on
synthetic inputs where every name has Rs 50 crore of ADTV. The real universe has
a long thin tail, the eligible count moves between dates, and names disappear.
Those are exactly the conditions under which "fully invested" quietly stops
being true -- which is the defect this whole exercise exists to remove, so
asserting it on synthetic data alone would repeat the original mistake at one
remove.

Turnover is the other half. The book replaces a 6-name, 21-session-cadence book
with a ~38-name one, and breadth is only free if the buffer band holds turnover
down. At the measured round trip of about 80 bps, every 100% of annual turnover
costs 0.8% a year, so this number decides whether the cost question needs
reopening.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prosignal.book import BookSpec, build_book          # noqa: E402

PANEL = HERE / "panel_2026_09.parquet"
#: Panel dates are 5 sessions apart; the shipped entry cadence is 21 sessions.
REBALANCE_EVERY = 4
#: Prices are not in the panel and the two properties measured here do not
#: depend on them except through whole-share rounding, which is immaterial at a
#: Rs 10 lakh book. A nominal price keeps the sizing arithmetic honest without
#: pretending to a precision the input cannot support.
NOMINAL_PRICE = 100.0


def main() -> int:
    panel = pd.read_parquet(PANEL, columns=["date", "symbol", "score", "adtv"])
    panel["date"] = pd.to_datetime(panel["date"])
    spec = BookSpec(capital=1_000_000.0)

    dates = sorted(panel["date"].unique())[::REBALANCE_EVERY]
    held: list[str] = []
    rows = []
    for d in dates:
        g = panel[panel["date"] == d].dropna(subset=["score"])
        if len(g) < 50:
            continue
        g = g.sort_values("score", ascending=False)
        ranking = g["symbol"].tolist()
        adtv = dict(zip(g["symbol"], g["adtv"]))
        prices = {s: NOMINAL_PRICE for s in ranking}

        b = build_book(ranking, prices, adtv, spec, held=held)
        rows.append({
            "date": d,
            "eligible": len(ranking),
            "held": len(b.positions),
            "invested": b.invested_fraction,
            "turnover": b.turnover,
            "capped": sum(1 for p in b.positions if p.binding == "liquidity cap"),
            "unsized": sum(1 for n in b.notes if "not sized" in n),
        })
        held = b.tickers

    r = pd.DataFrame(rows)
    per_year = 252 / (REBALANCE_EVERY * 5)

    print(f"rebalances: {len(r)}   {r['date'].min().date()}..{r['date'].max().date()}")
    print(f"eligible universe : {r['eligible'].min()}..{r['eligible'].max()} "
          f"(median {r['eligible'].median():.0f})")
    print(f"names held        : {r['held'].min()}..{r['held'].max()} "
          f"(median {r['held'].median():.0f})")
    print()
    print("DEPLOYED FRACTION -- the defect this replaces ran at 18.9%")
    print(f"  mean   {r['invested'].mean():.2%}")
    print(f"  min    {r['invested'].min():.2%}   on {r.loc[r['invested'].idxmin(),'date'].date()}")
    print(f"  below 95% on {int((r['invested'] < 0.95).sum())} of {len(r)} rebalances")
    print()
    print("TURNOVER (share of book replaced per rebalance)")
    print(f"  mean   {r['turnover'].mean():.1%}   median {r['turnover'].median():.1%}")
    print(f"  annualised ~{r['turnover'].mean() * per_year:.0%}"
          f"  -> ~{r['turnover'].mean() * per_year * 0.008:.2%} a year at an 80 bps round trip")
    print()
    print(f"liquidity cap bound on a name in {int((r['capped'] > 0).sum())} rebalances "
          f"(mean {r['capped'].mean():.1f} names)")
    print(f"names refused for unmeasured liquidity: mean {r['unsized'].mean():.1f}")

    out = HERE / "book_exposure.json"
    summary = {
        "rebalances": int(len(r)),
        "invested_mean": float(r["invested"].mean()),
        "invested_min": float(r["invested"].min()),
        "rebalances_below_95pct": int((r["invested"] < 0.95).sum()),
        "turnover_mean_per_rebalance": float(r["turnover"].mean()),
        "turnover_annualised": float(r["turnover"].mean() * per_year),
        "cost_at_80bps_annual": float(r["turnover"].mean() * per_year * 0.008),
        "held_median": float(r["held"].median()),
        "eligible_median": float(r["eligible"].median()),
    }
    pd.Series(summary).to_json(out, indent=1)
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
