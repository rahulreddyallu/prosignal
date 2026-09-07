"""Reference data the store already holds and nothing was reading.

Four tables were being ingested, written, given readers, and then consumed by
nobody:

    shareholding      22,787 rows -- promoter %, public %, FREE FLOAT
    security_list      3,534 rows -- series, price band, GSM stage, T2T, restricted
    fo_lots              216 rows -- F&O eligibility and lot size
    results_calendar  91,585 rows -- filing and broadcast timestamps

This module is the consumer. It exists because the information is real and the
absence of a reader was an oversight, not a decision.

THE TWO TABLES HAVE DIFFERENT POINT-IN-TIME STANDING, and conflating them would
be the worst kind of leak -- the kind that improves a backtest.

`shareholding` IS point-in-time. Every row carries an `availability_date`
derived from the exchange's own broadcast timestamp, and there are 1,234
distinct ones spanning 2022-01-07 onward. It can be as-of joined honestly.

`security_list` and `fo_lots` ACCUMULATE FORWARD. NSE publishes neither at a
dated URL, so the table cannot be reconstructed backwards: every ingest stores
one snapshot, and the table is point-in-time only from its first one onward.
This reader therefore takes the latest snapshot AT OR BEFORE the run date --
never the newest overall, which would hand a 2024 replay the 2026 surveillance
state. A date before the first snapshot gets NOT_TESTABLE rather than a
silently wrong answer, and so does a snapshot gone stale.

WHAT `restricted` MEANS. The provider defines it as trade-for-trade settlement,
OR an explicit GSM stage, OR a price band cut below the ordinary 20%. Its own
docstring: a name under any of them "cannot be filled at a simulated price".
That is an investability fact, and it is exactly the kind of hard gate Stage 3
exists to apply.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .types import SYMBOL

__all__ = [
    "SurveillanceSnapshot",
    "surveillance",
    "FreeFloat",
    "free_float",
    "ORDINARY_BAND_PCT",
]

#: The ordinary NSE band. Anything tighter is a surveillance measure.
ORDINARY_BAND_PCT = 20.0

#: How stale the newest snapshot at or before the run date may be. Seven days:
#: the ingest refreshes on the reference cadence, so a week covers a normal gap,
#: and beyond it the exchange has likely revised measures the file predates.
#: Lookahead is handled separately and absolutely -- a snapshot after the run
#: date is never used at any tolerance.
DEFAULT_TOLERANCE_DAYS = 7


@dataclass(frozen=True)
class SurveillanceSnapshot:
    """Exchange surveillance state, and whether it may be used for this date."""

    snapshot_date: Optional[dt.date]
    as_of: dt.date
    tolerance_days: int
    #: symbol -> True when the exchange has it under any surveillance measure.
    restricted: Dict[str, bool] = field(default_factory=dict)
    #: symbol -> price band %, where one is imposed. None means no band.
    band_pct: Dict[str, float] = field(default_factory=dict)
    #: symbol -> GSM stage string, where one applies.
    gsm_stage: Dict[str, str] = field(default_factory=dict)
    #: symbol -> settles trade-for-trade.
    t2t: Dict[str, bool] = field(default_factory=dict)
    #: symbol -> F&O eligible. Separate table, same snapshot standing.
    fo_eligible: Dict[str, bool] = field(default_factory=dict)
    unavailable: Optional[str] = None

    def testable(self) -> bool:
        return self.unavailable is None

    def is_restricted(self, symbol: str) -> Optional[bool]:
        """True, False, or None for "the list does not cover this name".

        A symbol absent from the security list is NOT unrestricted. It is
        unknown, and the caller must treat it as untestable rather than clear.
        """
        if not self.testable():
            return None
        return self.restricted.get(symbol)

    def reason(self, symbol: str) -> str:
        """Why a restricted name is restricted, in the exchange's own terms."""
        bits: List[str] = []
        if self.t2t.get(symbol):
            bits.append("settles trade-for-trade")
        stage = self.gsm_stage.get(symbol)
        if stage:
            bits.append(f"under {stage}")
        band = self.band_pct.get(symbol)
        if band is not None and band < ORDINARY_BAND_PCT:
            bits.append(f"price band cut to {band:.0f}% from the ordinary "
                        f"{ORDINARY_BAND_PCT:.0f}%")
        return "; ".join(bits) or "under an exchange surveillance measure"


def surveillance(store, as_of: dt.date,
                 tolerance_days: int = DEFAULT_TOLERANCE_DAYS
                 ) -> SurveillanceSnapshot:
    """Surveillance state for ``as_of``, or a stated refusal.

    The snapshot has ONE date. It describes that date. A run more than
    ``tolerance_days`` from it gets NOT_TESTABLE -- never a guess, and never
    today's list applied to a historical session.
    """
    try:
        sec = store.read_security_list()
    except Exception as exc:                                # noqa: BLE001
        return SurveillanceSnapshot(
            None, as_of, tolerance_days,
            unavailable=f"the security list could not be read: {exc}")
    if sec is None or sec.empty:
        return SurveillanceSnapshot(
            None, as_of, tolerance_days,
            unavailable="no security list in the store; run `prosignal data "
                        "ingest` to fetch NSE's surveillance file")

    # THE LATEST SNAPSHOT AT OR BEFORE THE RUN DATE, never the newest overall.
    #
    # NSE publishes neither the security list nor the F&O list at a dated URL,
    # so the table cannot be reconstructed backwards -- it accumulates FORWARD,
    # one snapshot per ingest. Once it holds more than one, taking `max()`
    # would hand a 2024 replay the 2026 surveillance state: it would reject
    # names for measures the exchange had not yet imposed and clear names it
    # later restricted. Lookahead in both directions, and it runs in the
    # flattering one.
    dates = pd.to_datetime(sec["snapshot_date"]).dt.date
    prior = sorted({d for d in dates.dropna() if d <= as_of})
    if not prior:
        first = min(dates.dropna()) if dates.notna().any() else None
        return SurveillanceSnapshot(
            None, as_of, tolerance_days,
            unavailable=(
                f"no surveillance snapshot on or before {as_of.isoformat()}"
                + (f"; the table starts at {first.isoformat()}" if first else "")
                + ". Projecting a later list backwards would be lookahead."))
    snap_date = prior[-1]

    # And it must not be STALE either: a snapshot months old describes a
    # surveillance state the exchange has since revised.
    gap = (as_of - snap_date).days
    if gap > tolerance_days:
        return SurveillanceSnapshot(
            snap_date, as_of, tolerance_days,
            unavailable=(
                f"the newest surveillance snapshot at or before this run is "
                f"dated {snap_date.isoformat()}, {gap} days stale against "
                f"{as_of.isoformat()} -- beyond the {tolerance_days}-day "
                f"tolerance. The exchange revises these measures continuously."))

    latest = sec[pd.to_datetime(sec["snapshot_date"]).dt.date == snap_date]
    idx = latest[SYMBOL].astype(str)

    def col(name, cast):
        if name not in latest.columns:
            return {}
        out = {}
        for s, v in zip(idx, latest[name]):
            if pd.isna(v):
                continue
            out[s] = cast(v)
        return out

    fo: Dict[str, bool] = {}
    try:
        lots = store.read_fo_lots()
        if lots is not None and not lots.empty and "fo_eligible" in lots.columns:
            fo = {str(s): bool(v) for s, v in
                  zip(lots[SYMBOL].astype(str), lots["fo_eligible"])
                  if not pd.isna(v)}
    except Exception:                                       # noqa: BLE001
        fo = {}

    return SurveillanceSnapshot(
        snapshot_date=snap_date, as_of=as_of, tolerance_days=tolerance_days,
        restricted=col("restricted", bool),
        band_pct=col("band_pct", float),
        gsm_stage=col("gsm_stage", str),
        t2t=col("is_t2t", bool),
        fo_eligible=fo,
    )


@dataclass(frozen=True)
class FreeFloat:
    """Point-in-time free float, as-of joined on the exchange broadcast."""

    as_of: dt.date
    #: symbol -> free float % of shares outstanding.
    pct: Dict[str, float] = field(default_factory=dict)
    #: symbol -> the disclosure date the figure came from.
    dated: Dict[str, dt.date] = field(default_factory=dict)
    unavailable: Optional[str] = None

    def testable(self) -> bool:
        return self.unavailable is None and bool(self.pct)

    def get(self, symbol: str) -> Optional[float]:
        return self.pct.get(symbol)


def free_float(store, as_of: dt.date,
               symbols: Optional[Iterable[str]] = None) -> FreeFloat:
    """Free float known on ``as_of``, from the quarterly shareholding pattern.

    AS-OF JOINED ON `availability_date`, which the provider derives from the
    exchange's broadcast timestamp -- not on `period_end`. The measured NSE
    disclosure lag runs to 45 days, so keying on the quarter end would use a
    filing weeks before it existed. This is the same discipline
    `pit_fundamentals` applies, for the same reason.
    """
    try:
        sh = store.read_shareholding()
    except Exception as exc:                                # noqa: BLE001
        return FreeFloat(as_of, unavailable=f"shareholding unreadable: {exc}")
    if sh is None or sh.empty:
        return FreeFloat(as_of, unavailable="no shareholding data in the store")
    if "free_float_pct" not in sh.columns or "availability_date" not in sh.columns:
        return FreeFloat(as_of, unavailable="shareholding lacks free float or "
                                            "an availability date")

    sh = sh.copy()
    sh["availability_date"] = pd.to_datetime(sh["availability_date"]).dt.date
    known = sh[sh["availability_date"] <= as_of]
    if known.empty:
        first = sh["availability_date"].min()
        return FreeFloat(
            as_of,
            unavailable=(f"no shareholding disclosed on or before "
                         f"{as_of.isoformat()}; the table starts at "
                         f"{first.isoformat()}"))
    if symbols is not None:
        want = {str(s) for s in symbols}
        known = known[known[SYMBOL].astype(str).isin(want)]
        if known.empty:
            return FreeFloat(as_of, unavailable="no shareholding for these names")

    # Latest disclosure per symbol at or before the run date.
    known = known.sort_values("availability_date")
    last = known.groupby(known[SYMBOL].astype(str), sort=False).tail(1)
    pct, dated = {}, {}
    for s, v, d in zip(last[SYMBOL].astype(str), last["free_float_pct"],
                       last["availability_date"]):
        if pd.isna(v):
            continue
        pct[s] = float(v)
        dated[s] = d
    if not pct:
        return FreeFloat(as_of, unavailable="every free-float figure was null")
    return FreeFloat(as_of, pct=pct, dated=dated)
