"""Calibrate market impact against what the book ACTUALLY paid.

WHAT THE COST MODEL IS TODAY. `costs.CostModel.impact_bps` is
`coefficient * participation ** exponent`, plus an assumed half-spread. Both
the coefficient and the exponent are config constants. Nothing in this engine
has ever compared them to a price it actually traded at, so the round-trip
figure every backtest subtracts -- 87 bps on a Rs 1.25 lakh position at Rs 20
crore ADTV -- is an assumption wearing the clothes of a measurement.

WHAT THIS MODULE DOES. Implementation shortfall: the gap between the price the
decision was made at (the signal date's close) and the price the position was
actually opened at, in basis points, regressed on participation. That is the
empirical impact-plus-delay cost, and it is the quantity the coefficient is
supposed to predict.

AND WHY IT CURRENTLY REFUSES. Run against `data/ledger/outcomes.jsonl` it
declines to report a calibration, because those rows are not fills. In 126 of
128 the recorded `entry_price` equals the next session's OPEN to the tick --
they were written by the simulator's own entry rule, not by a broker. A
"calibration" against them would fit the cost model to the assumption it was
built from and report the circularity as agreement. The remaining two are
VEDL rows carrying an unadjusted price against an adjusted open (415.65
against 145.90, ratio 2.85, with `price_basis_factor` recorded as 1.0), which
is the price-basis defect rather than a fill.

So `calibrate` returns a verdict, not a number, until the ledger holds
execution records. `SYNTHETIC` is a finding about the evidence; it is not a
failure of this code, and it must not be silently rendered as "no impact".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: A fill within this many basis points of a reference price, for this share of
#: rows, is that reference price rather than an execution. One tick on a Rs 100
#: name is 5 bps, so 1 bp is comfortably inside "identical".
TICK_BPS = 1.0
SYNTHETIC_SHARE = 0.75

#: Below this many genuine fills a regression on participation is not a
#: calibration. Two points define a line and say nothing about a coefficient.
MIN_FILLS = 60

#: Participation buckets the power law is fitted over. Ten is enough to see a
#: curve and few enough that each holds a real average.
IMPACT_BUCKETS = 10

#: Below this many buckets with a positive mean, there is no curve to fit and
#: saying so beats fitting one to noise.
MIN_IMPACT_BUCKETS = 4

CALIBRATED = "CALIBRATED"
SYNTHETIC = "SYNTHETIC_FILLS"
INSUFFICIENT = "INSUFFICIENT_FILLS"
NO_DATA = "NO_DATA"
#: A fit came back, and it describes something impact cannot do.
IMPLAUSIBLE = "IMPLAUSIBLE_FIT"


@dataclass
class ImpactCalibration:
    verdict: str
    reason: str
    n_rows: int = 0
    n_usable: int = 0
    #: Share of rows whose fill equals a reference price to within TICK_BPS.
    synthetic_share: float = float("nan")
    #: Which reference the fills collapse onto, when they do.
    synthetic_against: str = ""
    #: log(shortfall_bps) = log(coeff) + exponent * log(participation).
    fitted_coefficient: float = float("nan")
    fitted_exponent: float = float("nan")
    #: What the config asserts, for comparison.
    config_coefficient: float = float("nan")
    config_exponent: float = float("nan")
    median_shortfall_bps: float = float("nan")
    rows_dropped: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.verdict == CALIBRATED

    def to_dict(self) -> Dict[str, object]:
        return {"verdict": self.verdict, "reason": self.reason,
                "n_rows": self.n_rows, "n_usable": self.n_usable,
                "synthetic_share": self.synthetic_share,
                "synthetic_against": self.synthetic_against,
                "fitted_coefficient": self.fitted_coefficient,
                "fitted_exponent": self.fitted_exponent,
                "config_coefficient": self.config_coefficient,
                "config_exponent": self.config_exponent,
                "median_shortfall_bps": self.median_shortfall_bps,
                "rows_dropped": list(self.rows_dropped)}


def from_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """Recorded executions, in the shape `calibrate` reads.

    THE FEED THIS MODULE WAS WAITING FOR. `data/providers/csv_import.load_fills`
    imports what the book actually paid; this maps it onto the same columns the
    outcome ledger uses so one calibration path serves both, and so a store
    that acquires real fills starts calibrating without anything else changing.

    BUY SIDE ONLY. Implementation shortfall on a sell is a different quantity
    with a different sign, and mixing the two fits a coefficient to neither.
    Rows without a `decision_date` are dropped here rather than in `calibrate`,
    because shortfall is measured from the decision close and a fill that
    cannot name one is a record of a trade rather than a measurement of it.
    """
    if fills is None or fills.empty:
        return pd.DataFrame()
    need = {"symbol", "fill_date", "price", "side"}
    if not need <= set(fills.columns):
        return pd.DataFrame()
    buys = fills[fills["side"].astype(str).str.upper() == "BUY"]
    if "decision_date" not in buys.columns:
        return pd.DataFrame()
    buys = buys.dropna(subset=["decision_date"])
    if buys.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "ticker": buys["symbol"].astype(str),
        "signal_date": pd.to_datetime(buys["decision_date"]),
        "entry_date": pd.to_datetime(buys["fill_date"]),
        "entry_price": pd.to_numeric(buys["price"], errors="coerce"),
    }).reset_index(drop=True)


def read_outcomes(path: Path) -> pd.DataFrame:
    """The outcome ledger, as a frame. Unreadable lines are skipped, not fatal."""
    rows: List[Dict[str, object]] = []
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return pd.DataFrame(rows)


def _reference_prices(prices: pd.DataFrame, symbol: str,
                      signal_date, entry_date) -> Optional[Dict[str, float]]:
    try:
        dec = prices.loc[(symbol, pd.Timestamp(signal_date))]
        ent = prices.loc[(symbol, pd.Timestamp(entry_date))]
    except KeyError:
        return None
    out = {"decision_close": float(dec["close"]),
           "entry_open": float(ent["open"]),
           "entry_close": float(ent["close"]),
           "entry_vwap": float(ent.get("vwap", np.nan)),
           "decision_turnover": float(dec.get("turnover", np.nan))}
    return out if out["decision_close"] > 0 else None


def calibrate(outcomes: pd.DataFrame, prices: pd.DataFrame, *,
              config_coefficient: float = float("nan"),
              config_exponent: float = float("nan"),
              position_value_inr: float = float("nan")) -> ImpactCalibration:
    """Fit impact against realised shortfall, or say why it cannot be fitted.

    ``prices`` is indexed by (symbol, date) and carries open / close / vwap /
    turnover. ``outcomes`` is the ledger frame from `read_outcomes`.
    """
    if outcomes is None or outcomes.empty:
        return ImpactCalibration(NO_DATA, "the outcome ledger is empty")
    need = {"ticker", "signal_date", "entry_date", "entry_price"}
    missing = need - set(outcomes.columns)
    if missing:
        return ImpactCalibration(
            NO_DATA, f"the ledger has no {', '.join(sorted(missing))} column")

    recs: List[Dict[str, float]] = []
    dropped: List[str] = []
    for _, r in outcomes.iterrows():
        ref = _reference_prices(prices, str(r["ticker"]),
                                r["signal_date"], r["entry_date"])
        if ref is None:
            dropped.append(f"{r['ticker']} {r['signal_date']}: not in the price store")
            continue
        fill = float(r["entry_price"])
        if not np.isfinite(fill) or fill <= 0:
            dropped.append(f"{r['ticker']} {r['signal_date']}: no usable fill price")
            continue
        # A FILL THAT IS NOWHERE NEAR THE SESSION IS A PRICE-BASIS PROBLEM, not
        # an execution. Stored levels and adjusted prices diverge after any
        # corporate action, and a shortfall of +18,488 bps is a demerger, not
        # slippage. Including it would move the fitted coefficient by more than
        # every genuine row combined.
        lo, hi = ref["entry_open"] * 0.5, ref["entry_open"] * 2.0
        if not (lo <= fill <= hi):
            dropped.append(
                f"{r['ticker']} {r['signal_date']}: fill {fill:.2f} against an "
                f"open of {ref['entry_open']:.2f} -- unadjusted price basis")
            continue
        recs.append({**ref, "fill": fill, "symbol": str(r["ticker"])})

    if not recs:
        return ImpactCalibration(NO_DATA, "no ledger row could be priced",
                               n_rows=int(len(outcomes)), rows_dropped=dropped)

    f = pd.DataFrame(recs)
    f["shortfall_bps"] = (f["fill"] / f["decision_close"] - 1.0) * 1e4

    # ARE THESE FILLS AT ALL? A recorded price that equals a reference price to
    # the tick, over and over, was computed rather than executed.
    share, against = 0.0, ""
    for name in ("entry_open", "entry_vwap", "entry_close", "decision_close"):
        if name not in f:
            continue
        gap = (f["fill"] / f[name] - 1.0).abs() * 1e4
        s = float((gap <= TICK_BPS).mean())
        if s > share:
            share, against = s, name
    if share >= SYNTHETIC_SHARE:
        return ImpactCalibration(
            SYNTHETIC,
            f"{share:.1%} of ledger rows record a fill equal to the "
            f"{against.replace('_', ' ')} to within {TICK_BPS:g} bp. These are "
            f"the engine's own entry rule, not executions. Fitting the impact "
            f"model to them would fit it to the assumption it was built from "
            f"and report the circularity as agreement. Impact stays "
            f"UNCALIBRATED until the ledger holds broker fills.",
            n_rows=int(len(outcomes)), n_usable=int(len(f)),
            synthetic_share=share, synthetic_against=against,
            config_coefficient=config_coefficient,
            config_exponent=config_exponent,
            median_shortfall_bps=float(f["shortfall_bps"].median()),
            rows_dropped=dropped)

    if len(f) < MIN_FILLS:
        return ImpactCalibration(
            INSUFFICIENT,
            f"{len(f)} genuine fills, and a participation regression needs at "
            f"least {MIN_FILLS}",
            n_rows=int(len(outcomes)), n_usable=int(len(f)),
            synthetic_share=share, synthetic_against=against,
            config_coefficient=config_coefficient,
            config_exponent=config_exponent,
            median_shortfall_bps=float(f["shortfall_bps"].median()),
            rows_dropped=dropped)

    f["participation"] = (position_value_inr
                          / f["decision_turnover"].replace(0.0, np.nan))
    usable = f[(f["participation"] > 0) & np.isfinite(f["participation"])]
    if len(usable) < MIN_FILLS:
        return ImpactCalibration(
            INSUFFICIENT,
            f"only {len(usable)} fills carry a usable participation",
            n_rows=int(len(outcomes)), n_usable=int(len(f)),
            config_coefficient=config_coefficient,
            config_exponent=config_exponent,
            median_shortfall_bps=float(f["shortfall_bps"].median()),
            rows_dropped=dropped)

    # FIT ON BUCKET MEANS, NOT ON INDIVIDUAL FILLS.
    #
    # The obvious regression -- log(shortfall) on log(participation) over every
    # fill -- has to drop rows whose shortfall is non-positive, because log()
    # needs a positive. That conditions the sample ON THE DEPENDENT VARIABLE,
    # and it is not a small effect: a real fill beats the decision price about
    # as often as it misses it, so the drop keeps the half that went against
    # you and fits impact to that half alone. Every coefficient it produces is
    # biased upward, which is the direction that makes a strategy look more
    # expensive than it is and a cost model look better calibrated than it is.
    #
    # Bucketing by participation and taking the MEAN shortfall in each bucket
    # is well defined with negatives in it. Impact is a property of the
    # average fill at a participation level, not of the unlucky ones, so the
    # bucket mean is also the quantity the coefficient is supposed to predict.
    # Buckets whose mean is still non-positive are reported and excluded from
    # the log fit rather than silently dropped -- at low participation that is
    # the honest reading: no measurable impact.
    n_buckets = min(IMPACT_BUCKETS, max(len(usable) // 10, 2))
    q = pd.qcut(usable["participation"].rank(method="first"), n_buckets,
                labels=False, duplicates="drop")
    grouped = usable.groupby(q).agg(
        participation=("participation", "mean"),
        shortfall_bps=("shortfall_bps", "mean"),
        n=("shortfall_bps", "size"))
    positive = grouped[grouped["shortfall_bps"] > 0]
    if len(positive) < MIN_IMPACT_BUCKETS:
        return ImpactCalibration(
            INSUFFICIENT,
            f"only {len(positive)} of {len(grouped)} participation buckets "
            f"show positive mean shortfall, which is too few to fit a "
            f"power law. At this size the book may simply not have measurable "
            f"impact -- the median shortfall is "
            f"{float(usable['shortfall_bps'].median()):+.1f} bps -- and "
            f"reporting that is more useful than fitting a curve to noise",
            n_rows=int(len(outcomes)), n_usable=int(len(usable)),
            synthetic_share=share, synthetic_against=against,
            config_coefficient=config_coefficient,
            config_exponent=config_exponent,
            median_shortfall_bps=float(usable["shortfall_bps"].median()),
            rows_dropped=dropped)

    x = np.log(positive["participation"].to_numpy(dtype="float64"))
    y = np.log(positive["shortfall_bps"].to_numpy(dtype="float64") / 1e4)
    expo, log_c = np.polyfit(x, y, 1)

    # A NEGATIVE EXPONENT IS NOT AN IMPACT CURVE. It says a larger trade moves
    # the price LESS, and fed back into `CostModel.impact_bps` it would make
    # size cheaper than a small order -- the one direction a cost model must
    # never be wrong in. It is what a fit returns when the shortfall is noise
    # around a level, which at this book's participation (0.014% of ADTV) is
    # the expected result rather than a surprise. Report the level and refuse
    # the curve.
    if not np.isfinite(expo) or expo <= 0.0:
        return ImpactCalibration(
            IMPLAUSIBLE,
            f"the fitted exponent is {expo:+.3f}, which says a larger trade "
            f"moves the price less. That is not an impact curve; it is what "
            f"comes back when shortfall is noise around a level. Median "
            f"shortfall is "
            f"{float(usable['shortfall_bps'].median()):+.1f} bps across "
            f"{len(usable)} fills, and quoting THAT as a flat cost is honest "
            f"where quoting a downward-sloping power law is not",
            n_rows=int(len(outcomes)), n_usable=int(len(usable)),
            synthetic_share=share, synthetic_against=against,
            config_coefficient=config_coefficient,
            config_exponent=config_exponent,
            median_shortfall_bps=float(usable["shortfall_bps"].median()),
            rows_dropped=dropped)

    return ImpactCalibration(
        CALIBRATED,
        f"fitted on {len(positive)} participation buckets covering "
        f"{int(positive['n'].sum())} of {len(usable)} fills",
        n_rows=int(len(outcomes)), n_usable=int(len(usable)),
        synthetic_share=share, synthetic_against=against,
        fitted_coefficient=float(np.exp(log_c)), fitted_exponent=float(expo),
        config_coefficient=config_coefficient, config_exponent=config_exponent,
        median_shortfall_bps=float(usable["shortfall_bps"].median()),
        rows_dropped=dropped)
