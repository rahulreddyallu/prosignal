"""Rebuild the research panel from the current store, and stamp its provenance.

The panel is the input to every v3 experiment in this directory. It is a parquet
file with no record of when it was built or from what, and on 2026-09-05 that
cost real conclusions: the panel predated the fundamentals ingest, so its
`quality` theme covered 25% of scored names against the live engine's ~85%, and
four harnesses had been reporting on it as though it were the shipped model.

Building through `validation.v3_panel.build_v3_panel` -- the same function the
quarterly re-check uses -- so there is no second panel implementation to drift.

Usage:
    python research/v3/experiments/build_panel.py
    python research/v3/experiments/build_panel.py --out panel_2026_09.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from _panel_guard import stamp                                   # noqa: E402
from prosignal.config.loader import load_config                  # noqa: E402
from prosignal.data.store import DataStore                       # noqa: E402
from prosignal.validation.v3_panel import build_v3_panel         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="panel_2026_09.parquet")
    ap.add_argument("--keep-old", action="store_true",
                    help="save the existing panel beside the new one as "
                         "<name>.pre-rebuild.parquet, so the old numbers can "
                         "still be reproduced and compared")
    args = ap.parse_args()

    out = HERE / args.out
    cfg = load_config(ROOT / "config/parameters.yaml")
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)

    if out.exists() and args.keep_old:
        keep = out.with_suffix(".pre-rebuild.parquet")
        keep.write_bytes(out.read_bytes())
        print(f"previous panel kept at {keep.name}")

    print("building ...")
    panel = build_v3_panel(store)
    if panel is None or panel.empty:
        print("the panel came back empty; the store served nothing usable")
        return 2
    panel.to_parquet(out, index=False)

    scored = panel[panel["score"].notna()]
    cov = {t: float(scored[t + "_sub"].notna().mean())
           for t in ("momentum", "quality", "ownership", "risk", "reversal")
           if t + "_sub" in scored.columns}
    side = stamp(out, extra={
        "rows": int(len(panel)),
        "dates": int(panel["date"].nunique()),
        "start": str(pd.to_datetime(panel["date"]).min().date()),
        "end": str(pd.to_datetime(panel["date"]).max().date()),
        "theme_coverage_on_scored_rows": cov,
    })
    print(f"\n{out.name}: {len(panel):,} rows, {panel['date'].nunique()} dates, "
          f"{pd.to_datetime(panel['date']).min().date()}"
          f"..{pd.to_datetime(panel['date']).max().date()}")
    print("theme coverage on scored rows:")
    for t, c in cov.items():
        print(f"  {t:10s} {c:6.1%}")
    print(f"\nprovenance written to {side.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
