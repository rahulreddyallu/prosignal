"""Build the research panel and price panels once, cache to disk.

Everything downstream in this audit reads the cache, so every experiment is
measured on identical inputs and a difference between two runs can only come
from the thing being varied.

Selection period only. `end` is the holdout boundary the config reserves, and
--include-holdout does not exist here on purpose.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from prosignal.config.loader import load_config
from prosignal.data.store import DataStore
from prosignal.data.universe import UniverseResolver
from prosignal.stages._cfg import fv, iv, v
from prosignal.validation.research_panel import build_research_panel

OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)


def portfolio_inputs(cfg, store, sessions, end):
    from prosignal.data.types import DATE, SYMBOL

    c7 = cfg.params.stage7_risk
    px = store.read_prices(
        symbols=None, start=sessions[0], end=end,
        columns=[DATE, SYMBOL, "open", "high", "low", "close", "volume"])
    px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()
    panels = {
        col: px.pivot_table(index=DATE, columns=SYMBOL, values=col,
                            aggfunc="last", observed=True).sort_index()
        for col in ("open", "high", "low", "close", "volume")
    }
    del px
    close, high, low = panels["close"], panels["high"], panels["low"]
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).stack(), (high - prev).abs().stack(),
         (low - prev).abs().stack()], axis=1).max(axis=1).unstack()
    period = iv(c7.atr.period_sessions)
    panels["atr"] = tr.ewm(alpha=1.0 / period, adjust=False,
                           min_periods=period).mean()
    panels["ma"] = close.rolling(
        iv(c7.thesis_invalidation.structure_ma_sessions)).mean()
    panels["adtv"] = (close * panels["volume"]).rolling(21).mean()
    return panels


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    print(f"config {cfg.version}")
    store = DataStore(cfg.paths.curated, cfg.paths.snapshots)
    sessions = store.price_sessions()
    reserve = iv(cfg.params.validation.holdout.reserve_most_recent_sessions)
    end = sessions[-reserve]
    print(f"{len(sessions)} sessions, holdout boundary {end} "
          f"(reserving {reserve})")

    u = cfg.params.universe
    sectors = store.read_sector_map()
    sector_map = (dict(zip(sectors["symbol"], sectors["sector"]))
                  if sectors is not None and not sectors.empty else {})
    UniverseResolver(store, cfg).resolve_liquidity_pit(
        as_of=sessions[-1], min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names),
        min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr),
        manual_exclusions=list(v(u.manual_exclusions) or []),
        sector_map=sector_map)

    print("reading price panels ...")
    panels = portfolio_inputs(cfg, store, sessions, end)
    print(f"  close panel {panels['close'].shape}")

    print("building research panel ...")
    turnover = panels["close"] * panels["volume"]
    rp = build_research_panel(cfg, store, end, prices=panels, turnover=turnover)
    print(f"  {len(rp.panel):,} rows over {rp.n_dates} dates, "
          f"features {rp.features}")
    print(f"  dropped for coverage: {rp.dropped}")

    # Trim the price panels to the symbols and window the panel actually uses,
    # so the pickle stays small enough to reload quickly.
    syms = sorted(set(rp.panel["symbol"].unique()))
    keep = {k: df.reindex(columns=[c for c in df.columns if c in set(syms)])
            for k, df in panels.items()}

    with open(OUT / "panels.pkl", "wb") as fh:
        pickle.dump(keep, fh, protocol=4)
    with open(OUT / "research.pkl", "wb") as fh:
        pickle.dump({"panel": rp.panel, "features": rp.features,
                     "horizon": rp.horizon, "dropped": rp.dropped,
                     "end": end, "sector_map": rp.sector_map}, fh, protocol=4)
    print(f"cached in {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
