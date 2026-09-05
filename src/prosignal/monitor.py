"""Live monitoring for the v3 thematic composite.

Three questions the engine could not previously answer about itself:

  1. Is each FACTOR still doing what the holdouts measured it doing?
  2. Is each THEME? A composite can hold up in aggregate while one theme has
     inverted, and the composite's own information coefficient will not say
     which.
  3. Is one theme quietly running the book? The weights are capped at 40% by
     construction, but a theme can dominate the REALISED ranking anyway if the
     others go flat -- the cap constrains the weight, not the influence. So each
     theme's share of the composite's cross-sectional DISPERSION is tracked, and
     a theme past `DOMINANCE_ALERT`, or `DOMINANCE_EXCESS` above its own
     declared weight, is flagged.

     Dispersion, not variance. Variance is quadratic in the weight, so at the
     shipped configuration momentum explains w^2/sum(w^2) = 62% of the variance
     while carrying 40% of the weight -- and a 55% alarm set in the units of the
     weight cap would have fired on a perfectly healthy book every day from the
     first run. A standard-deviation share equals the declared weight when the
     sub-scores are equally dispersed, which is what makes 40% vs 55% a
     comparison rather than a units error.

Plus the book's drawdown against a flag.

NONE OF THESE DISABLE ANYTHING. A monitor that switches a factor or a theme off
changes the model without a decision being taken, and the next person to read
the config sees a model that is not the one running. They all FLAG.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features.engine import THEMES, FACTOR_THEME

__all__ = ["FactorHealth", "ThemeHealth", "DrawdownFlag", "rolling_factor_ic",
           "rolling_theme_ic", "review_factors", "review_themes",
           "theme_influence_share", "cross_section_influence",
           "review_cross_section", "review_drawdown", "MIN_PERIODS",
           "IC_ALERT_T", "DOMINANCE_ALERT", "DOMINANCE_EXCESS", "DRAWDOWN_FLAG",
           "MIN_NAMES_FOR_INFLUENCE", "review_realised_drawdown",
           "MIN_CLOSED_TRADES_FOR_DRAWDOWN"]

#: Fewer than this many scored periods and a rolling IC is not worth reading.
MIN_PERIODS = 40
#: A factor or theme is flagged when its oriented IC is significantly NEGATIVE
#: at this confidence -- inverted, not merely weak. A factor at zero costs the
#: composite its weight; one that has inverted is actively subtracting.
IC_ALERT_T = 2.0
#: Share of the composite's cross-sectional dispersion above which a theme is
#: flagged as dominating. The largest shipped weight is 40%; a theme explaining
#: more than 55% of the realised spread is running the book whatever the config
#: says. Measured on the same scale as the weight -- see the note on dispersion
#: vs variance in the module docstring.
DOMINANCE_ALERT = 0.55
#: A theme is also flagged when it runs this far ABOVE its own declared weight,
#: which is how a small theme over-running is caught: quality ships at 19%, so
#: it could double its influence and never approach the absolute alarm. Fifteen
#: points, the same distance that separates the 40% cap from the 55% alarm.
DOMINANCE_EXCESS = 0.15
#: Book drawdown past which the run carries a loud flag. The deepest drawdown
#: across both sealed windows was -23.9%.
DRAWDOWN_FLAG = -0.25
#: Below this many scored names an influence share is noise, not a measurement.
MIN_NAMES_FOR_INFLUENCE = 30


@dataclass
class FactorHealth:
    name: str
    theme: str
    n_periods: int
    ic_mean: Optional[float] = None
    ic_t: Optional[float] = None
    inverted: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class ThemeHealth:
    name: str
    weight: float
    n_periods: int
    ic_mean: Optional[float] = None
    ic_t: Optional[float] = None
    influence_share: Optional[float] = None
    inverted: bool = False
    dominating: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class DrawdownFlag:
    flagged: bool
    drawdown: float
    threshold: float = DRAWDOWN_FLAG
    peak_date: Optional[dt.date] = None
    trough_date: Optional[dt.date] = None
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["peak_date"] = self.peak_date.isoformat() if self.peak_date else None
        d["trough_date"] = self.trough_date.isoformat() if self.trough_date else None
        return d


def _ic(a: np.ndarray, b: np.ndarray, min_n: int = 30) -> Optional[float]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return None
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _t(v: np.ndarray) -> Optional[float]:
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return None
    sd = v.std(ddof=1)
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else 0.0


def rolling_factor_ic(panel: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-date rank IC of each factor's SHIPPED-SIGN-ORIENTED rank column."""
    signs = {f: th.signs[f] for th in THEMES.values() for f in th.names}
    rows = []
    for d, g in panel.groupby("date", sort=True):
        y = g[label_col].to_numpy("float64")
        rec: Dict[str, object] = {"date": d, "n": int(np.isfinite(y).sum())}
        for f, sg in signs.items():
            col = f + "_r"
            rec[f] = (np.nan if col not in g.columns
                      else (_ic(g[col].to_numpy("float64") * sg, y) or np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def rolling_theme_ic(panel: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-date rank IC of each THEME's sub-score. Sub-scores are already
    oriented, so a negative value here means the theme has inverted."""
    rows = []
    for d, g in panel.groupby("date", sort=True):
        y = g[label_col].to_numpy("float64")
        rec: Dict[str, object] = {"date": d, "n": int(np.isfinite(y).sum())}
        for t in THEMES:
            col = t + "_sub"
            rec[t] = (np.nan if col not in g.columns
                      else (_ic(g[col].to_numpy("float64"), y) or np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def theme_influence_share(panel: pd.DataFrame) -> Dict[str, float]:
    """Share of the composite's cross-sectional DISPERSION each theme explains.

    Computed per date on the weighted contributions and averaged, so it measures
    INFLUENCE rather than declared weight. A theme whose sub-score has gone flat
    contributes nothing here however large its coefficient.

    The per-date statistic is the standard deviation, normalised across themes,
    NOT the variance. Variance is quadratic in the weight: at the shipped
    configuration momentum would read 62% against a declared 40%, and every
    comparison with the 40% cap would be a units error. On this scale a theme
    reads its declared weight exactly when the sub-scores are equally dispersed,
    so a departure means something has actually changed.
    """
    cols = [t + "_contrib" for t in THEMES if t + "_contrib" in panel.columns]
    if not cols:
        return {}
    acc: Dict[str, List[float]] = {c[:-8]: [] for c in cols}
    for _, g in panel.groupby("date", sort=True):
        v = g[cols].to_numpy("float64")
        if len(g) < 30 or not np.isfinite(v).any():
            continue
        sd = np.nanstd(v, axis=0)
        tot = np.nansum(sd)
        if tot <= 0:
            continue
        for i, c in enumerate(cols):
            acc[c[:-8]].append(float(sd[i] / tot))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def cross_section_influence(scored: pd.DataFrame) -> Dict[str, float]:
    """Each theme's share of TODAY's cross-sectional spread. One date, no labels.

    THIS IS THE HALF OF THE MONITOR THAT CAN RUN ON EVERY RUN. Rolling IC needs
    forward outcomes, so it cannot say anything about today until twenty-one
    sessions have passed -- which is correct, and also means that on its own it
    would leave the daily run with nothing to say about the model at all. The
    dominance question does not need outcomes: whether one theme is doing most
    of the separating between names is a property of the scores as they stand,
    visible the moment they are computed.

    So a theme that has quietly taken over the ranking is caught the same day,
    and whether it was RIGHT to is answered three weeks later by the IC.
    """
    cols = [t + "_contrib" for t in THEMES if t + "_contrib" in scored.columns]
    if not cols or len(scored) < MIN_NAMES_FOR_INFLUENCE:
        return {}
    v = scored[cols].to_numpy("float64")
    if not np.isfinite(v).any():
        return {}
    sd = np.nanstd(v, axis=0)
    tot = float(np.nansum(sd))
    if tot <= 0:
        return {}
    return {c[:-8]: float(sd[i] / tot) for i, c in enumerate(cols)}


def review_cross_section(scored: pd.DataFrame,
                         dominance: float = DOMINANCE_ALERT,
                         excess: float = DOMINANCE_EXCESS) -> List[str]:
    """Notes for one run: which themes are running more than they were given.

    Returns human-readable lines, never a decision. Nothing is disabled and no
    weight moves -- a run that silently reweighted itself because one theme
    looked large today would be a different model every morning.
    """
    share = cross_section_influence(scored)
    if not share:
        return []
    ordered = sorted(share.items(), key=lambda kv: -kv[1])
    body = ", ".join(f"{t} {v:.0%} (declared {THEMES[t].weight:.0%})"
                     for t, v in ordered)
    notes = [f"Theme influence on this cross-section: {body}. This is the share "
             f"of the spread between names that each theme actually produced "
             f"today, against the weight it was given; they agree when the "
             f"themes are equally dispersed."]
    for t, v in ordered:
        w = THEMES[t].weight
        if v > dominance or v > w + excess:
            notes.append(
                f"FLAG {t}: it is producing {v:.0%} of today's separation "
                f"between names against a declared weight of {w:.0%}. The cap "
                f"constrains the coefficient, not the influence -- if the other "
                f"themes have gone flat, this ranking is mostly {t} whatever "
                f"the config says. Nothing has been changed; check "
                f"`prosignal research v3 --monitor` for whether it is still "
                f"PAYING, which needs outcomes this run does not have yet.")
    return notes


def review_factors(ic_frame: pd.DataFrame, window: int = 52,
                   min_periods: int = MIN_PERIODS,
                   alert_t: float = IC_ALERT_T) -> List[FactorHealth]:
    """Flag any factor whose recent oriented IC is significantly negative.

    The t-statistic is naive -- overlapping labels make consecutive ICs share
    most of their window, inflating it by roughly sqrt(horizon / stride). That
    is stated rather than corrected because this is a screen for a human to
    look at, and the honest correction at these sample sizes costs more power
    than the screen has. A flag means "go and check", never "this is dead".
    """
    out: List[FactorHealth] = []
    tail = ic_frame.tail(int(window))
    for f, theme in FACTOR_THEME.items():
        if f not in tail.columns:
            out.append(FactorHealth(f, theme, 0, note="not present in the panel"))
            continue
        v = tail[f].to_numpy("float64")
        v = v[np.isfinite(v)]
        if len(v) == 0:
            # `rolling_factor_ic` writes an all-NaN column for a factor the
            # panel never carried, so an absent factor arrives here looking
            # merely short. They need different responses: one waits for more
            # data, the other means the factor is not being computed at all.
            out.append(FactorHealth(f, theme, 0,
                                    note="not present in the panel -- no IC was "
                                         "computed for it on any date"))
            continue
        if len(v) < min_periods:
            out.append(FactorHealth(f, theme, len(v),
                                    note=f"under the {min_periods}-period floor; "
                                         f"no verdict"))
            continue
        t = _t(v) or 0.0
        inv = t <= -alert_t
        out.append(FactorHealth(
            f, theme, len(v), float(v.mean()), t, inv,
            "oriented IC is significantly negative over this window; the naive t "
            "is inflated by label overlap, so check before acting"
            if inv else "consistent with its shipped sign"))
    return out


def review_themes(ic_frame: pd.DataFrame, panel: Optional[pd.DataFrame] = None,
                  window: int = 52, min_periods: int = MIN_PERIODS,
                  alert_t: float = IC_ALERT_T,
                  dominance: float = DOMINANCE_ALERT,
                  excess: float = DOMINANCE_EXCESS) -> List[ThemeHealth]:
    """Per-theme health: has it inverted, and is it running more of the ranking
    than it was given?

    Dominance fires on either of two conditions, because they catch different
    failures. The ABSOLUTE one (`dominance`) catches the composite collapsing
    onto its largest theme -- the others go flat and momentum is the ranking.
    The RELATIVE one (`excess`) catches a small theme over-running: quality
    ships at 19% and could carry twice the influence it was given without ever
    approaching an absolute 55%, and that is the same defect at a smaller size.
    """
    shares = theme_influence_share(panel) if panel is not None else {}
    out: List[ThemeHealth] = []
    tail = ic_frame.tail(int(window))
    for t, th in THEMES.items():
        share = shares.get(t)
        v = (tail[t].to_numpy("float64") if t in tail.columns
             else np.array([], dtype="float64"))
        v = v[np.isfinite(v)]
        over = bool(share is not None and share > th.weight + excess)
        dom = bool(share is not None and share > dominance) or over
        if len(v) == 0:
            out.append(ThemeHealth(t, th.weight, 0, influence_share=share,
                                   dominating=dom,
                                   note="not present in the panel -- no IC was "
                                        "computed for it on any date"))
            continue
        if len(v) < min_periods:
            out.append(ThemeHealth(t, th.weight, len(v), influence_share=share,
                                   dominating=dom,
                                   note=f"under the {min_periods}-period floor; "
                                        f"no verdict on the coefficient"))
            continue
        tt = _t(v) or 0.0
        inv = tt <= -alert_t
        notes = []
        if inv:
            notes.append("theme sub-score has inverted over this window")
        if dom:
            notes.append(f"explains {share:.0%} of the composite's cross-sectional "
                         f"spread against a declared weight of {th.weight:.0%} -- "
                         f"the cap constrains the coefficient, not the influence")
        out.append(ThemeHealth(t, th.weight, len(v), float(v.mean()), tt, share,
                               inv, dom,
                               "; ".join(notes) if notes
                               else "consistent with its shipped weight"))
    return out


#: Closed trades below which a drawdown is noise rather than a measurement. A
#: six-name book's realised curve after four exits says nothing about the book.
MIN_CLOSED_TRADES_FOR_DRAWDOWN = 20


def review_realised_drawdown(outcomes: Sequence[Dict[str, object]],
                             threshold: float = DRAWDOWN_FLAG,
                             min_trades: int = MIN_CLOSED_TRADES_FOR_DRAWDOWN
                             ) -> List[str]:
    """Run notes for the book's drawdown, from CLOSED trades only.

    WHAT THIS CANNOT SEE, said here rather than discovered later: the curve is
    built from resolved outcomes, so a position that is open and deeply
    underwater contributes NOTHING until it exits. On a six-name book held a
    median of twenty sessions that is a real lag, and it biases the measurement
    in the comfortable direction -- the realised drawdown is the shallower one.
    It is still worth having, because it is the only drawdown the engine can
    compute without marking an open book to market, and a flag that arrives
    late is better than a number nobody ever produces. Read it as a floor on
    the drawdown, not an estimate of it.

    Returns [] when there is nothing to say. Silence is correct on a book with
    no closed trades; reporting "0%, inside the flag" would be a reassurance
    about a book that has not yet been tested.
    """
    from .performance import equity_curve

    try:
        curve = equity_curve(list(outcomes or []))
    except Exception:
        return []
    if len(curve) < int(min_trades):
        return []
    # The curve is a running SUM of per-trade returns on an equal-slot book --
    # deliberately not compounded, because compounding overlapping holds would
    # imply leverage the strategy never took. So equity is 1 + that sum.
    eq = [1.0 + float(pt["cumulative"]) for pt in curve]
    dates = [pt.get("date") for pt in curve]
    flag = review_drawdown(eq, None, threshold)
    if flag.flagged:
        return [f"FLAG drawdown: {flag.note} Measured on {len(curve)} closed "
                f"trades to {dates[-1]}, realised only -- an open position "
                f"underwater is not in this number yet."]
    return [f"Book drawdown {flag.drawdown:.1%} from its peak, inside the "
            f"{threshold:.0%} flag, on {len(curve)} closed trades (realised "
            f"only; open positions are not marked to market here)."]


def review_drawdown(equity: Sequence[float],
                    dates: Optional[Sequence[dt.date]] = None,
                    threshold: float = DRAWDOWN_FLAG) -> DrawdownFlag:
    """Current drawdown from the running peak, and whether it is past the flag.

    FLAGS, NEVER DISABLES. A breaker that stops new entries in a drawdown is a
    market-timing rule, and every market-timing rule this engine measured cost
    8-13 points of annual excess without reducing the drawdown it existed to
    avoid.
    """
    e = np.asarray(list(equity), dtype="float64")
    if e.size == 0 or not np.isfinite(e).any():
        return DrawdownFlag(False, float("nan"), threshold,
                            note="no equity curve to measure")
    peaks = np.maximum.accumulate(e)
    dd = e / np.where(peaks > 0, peaks, np.nan) - 1.0
    cur = float(dd[-1])
    i_tr = int(np.nanargmin(dd))
    i_pk = int(np.nanargmax(e[: i_tr + 1])) if i_tr >= 0 else 0
    d_pk = d_tr = None
    if dates is not None and len(dates) == len(e):
        d_pk, d_tr = dates[i_pk], dates[i_tr]
    flagged = bool(np.isfinite(cur) and cur <= threshold)
    note = (f"book is {cur:.1%} below its peak, past the {threshold:.0%} flag. "
            f"The deepest drawdown across both sealed windows was -23.9%; this "
            f"is outside the range the shipped evidence describes. Nothing has "
            f"been disabled -- review position sizes, the per-theme IC and the "
            f"theme variance shares before the next entry."
            if flagged else
            f"book is {cur:.1%} below its peak, inside the {threshold:.0%} flag")
    return DrawdownFlag(flagged, cur, threshold, d_pk, d_tr, note)
