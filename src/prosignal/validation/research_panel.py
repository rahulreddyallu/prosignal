"""The training panel, built exactly once, for research and for production.

WHY THIS EXISTS. `research cpcv` and `research factors` each rebuilt the panel
by hand -- forty lines of pivots, masks and joins, duplicated and then edited
independently. They drifted, and drift here is not a cosmetic problem: what CPCV
validates stops being what stage 4 trades, and the validation number goes on
being quoted anyway.

The last time this happened, the research commands passed raw FEATURE_COLUMNS to
a harness that drops rows on all of them, deleting every row without a
fundamental and cutting a 70-date panel to 17. That was fixed by extracting
`crossmodel.prepare_features`. This module closes the same gap one level up: the
research commands did not read `high`/`low` and did not pass `barriers`, so
after the label became triple-barrier they were still measuring factors against
the horizon return -- the exact label the engine had stopped fitting on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..data.types import DATE, SYMBOL
from ..features import crossmodel as cm
from ..features.crosssec import build_panel, liquidity_mask
from ..features.exits import ExitRules, rules_from_config
from ..features.labels import BarrierSpec
from ..stages._cfg import fv, iv

__all__ = ["ResearchPanel", "build_research_panel"]

#: Distinguishes "caller said nothing" from "caller said: no exit geometry".
#: `None` is a meaningful VALUE here -- it asks for a panel with no
#: invalidation filter, which is the only way to see the population the live
#: engine scores -- so it cannot double as the default.
_USE_CONFIG = object()


@dataclass
class ResearchPanel:
    panel: pd.DataFrame
    features: List[str]
    dropped: Dict[str, float]
    close: pd.DataFrame
    horizon: int
    barriers: Optional[BarrierSpec]
    exit_rules: Optional[ExitRules]
    sector_map: Dict[str, str]

    @property
    def n_dates(self) -> int:
        return int(self.panel["date"].nunique()) if not self.panel.empty else 0


def build_research_panel(cfg, store, end, *, step: int = 21,
                         sector_neutral: bool = True,
                         prices: Optional[Dict[str, pd.DataFrame]] = None,
                         turnover: Optional[pd.DataFrame] = None,
                         exit_rules=_USE_CONFIG) -> ResearchPanel:
    """Build the panel the model is fitted on, from the same config it uses.

    ``end`` is the last session to include -- callers pass the holdout boundary
    unless they have explicitly chosen to spend it.

    ``prices`` and ``turnover`` let a caller that has ALREADY loaded the price
    frames -- the portfolio harness needs open/high/low/atr for the simulator --
    pass them in rather than paying for a second full read of the store.
    """
    p = cfg.params
    c4, u = p.stage4_core_score, p.universe
    sessions = store.price_sessions()

    # `high` and `low` are not optional extras. A stop is not a close-only
    # instrument, and a barrier label built from closes alone understates how
    # often the stop was hit.
    if prices is not None and turnover is not None:
        close, high, low = prices["close"], prices["high"], prices["low"]
        open_ = prices.get("open")
    else:
        px = store.read_prices(
            start=sessions[0], end=end,
            columns=[DATE, SYMBOL, "close", "turnover", "high", "low", "open"])
        px[DATE] = pd.to_datetime(px[DATE]).dt.normalize()

        def piv(col: str) -> pd.DataFrame:
            return px.pivot_table(index=DATE, columns=SYMBOL, values=col,
                                  aggfunc="last", observed=True).sort_index()

        close, turnover = piv("close"), piv("turnover")
        high, low, open_ = piv("high"), piv("low"), piv("open")
        del px

    delivery = None
    dl = store.read_delivery(start=sessions[0], end=end)
    if dl is not None and not dl.empty and "deliv_pct" in dl.columns:
        dl[DATE] = pd.to_datetime(dl[DATE]).dt.normalize()
        delivery = dl.pivot_table(index=DATE, columns=SYMBOL, values="deliv_pct",
                                  aggfunc="last", observed=True).sort_index()
    del dl

    sectors = store.read_sector_map()
    sector_map = (dict(zip(sectors["symbol"], sectors["sector"]))
                  if sectors is not None and not sectors.empty else {})

    # Point-in-time, per date. Building from the screen resolved for the LATEST
    # session projects today's survivors backwards: 27% of the names eligible on
    # 2024-08-12 are absent from today's set, excluded for what happened after.
    eligible = liquidity_mask(
        close, turnover, min_adtv_inr=fv(u.pit_min_adtv_inr),
        lookback_sessions=iv(u.pit_adtv_lookback_sessions),
        max_names=iv(u.pit_max_names),
        min_history_sessions=iv(u.min_history_sessions),
        min_price_inr=fv(u.min_price_inr))

    horizon = iv(c4.model_horizon_sessions)
    lab = c4.labels
    engine_geometry = (bool(lab.triple_barrier)
                       and str(lab.barrier_source) == "engine")
    # An explicit override lets a caller ask what the coefficients would be
    # under a DIFFERENT traded geometry -- which is the only way to tell a dead
    # theme from one the current label mismeasures. It is research only: the
    # shipped path passes nothing and gets the config's own rules, so the panel
    # the model trains on cannot be changed from here by accident.
    #
    # A SENTINEL, not `or`. `exit_rules or default` cannot tell "not provided"
    # from "explicitly no exit geometry", so passing None -- the way a caller
    # asks for an UNFILTERED panel, one that keeps the rows the invalidation
    # rule would drop -- silently returned the shipped panel instead. Two
    # experiments were run against it and both reported the filtered population
    # as though it were the full one.
    if exit_rules is _USE_CONFIG:
        rules = rules_from_config(c4, p.stage7_risk) if engine_geometry else None
    else:
        rules = exit_rules
    spec = (BarrierSpec(upper=fv(lab.upper_sigma), lower=fv(lab.lower_sigma),
                        horizon=horizon, vol_window=iv(lab.vol_window_sessions))
            if bool(lab.triple_barrier) and not engine_geometry else None)

    panel = build_panel(close, turnover, horizon=horizon, step=step,
                        delivery=delivery, eligible=eligible,
                        sectors=(sector_map if sector_neutral else None),
                        barriers=spec, exit_rules=rules,
                        high=high, low=low, open_=open_)
    try:
        actions = store.read_corporate_actions()
    except Exception:
        actions = None
    panel = cm._attach_fundamentals(panel, store.read_statements(), close,
                                    iv(c4.max_fundamental_age_days), actions=actions)
    # The SAME coverage tests and family construction the fit uses.
    panel, features, dropped = cm.prepare_features(panel)
    return ResearchPanel(panel=panel, features=features, dropped=dropped,
                         close=close, horizon=horizon, barriers=spec,
                         exit_rules=rules, sector_map=sector_map)
