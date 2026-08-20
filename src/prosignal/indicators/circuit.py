"""Price-band (circuit) state, derived from the daily bar.

NSE's UDiFF bhavcopy carries 34 columns and none of them is the price band, so
band membership cannot be read from the feed. It has to be inferred, and the
inference is deliberately narrow: a session where the stock traded but the high
and the low are the same price is a session where only one price was ever
available. Whether that is a band lock or a single print in a thin name does not
matter for our purpose -- in both cases an execution assumption that names some
other price is fiction.

The inference checks out against the band grid. Across 1.36M sessions from 2024,
3,507 are frozen, and their returns cluster where NSE's bands sit: 52% at about
5%, 7% at 2%, 5% at 10%, 1% at 20%. A detector firing on noise would not
reproduce the exchange's own ladder.

What this is not: a claim to know the band. A name frozen 1% from its previous
close is more likely thin than locked. Both are reported, because both break the
same execution assumption, but they are reported as different states so a reader
can tell which is which.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["BandState", "band_state", "is_untradeable", "annotate_band_state"]

#: Returns at which NSE's standard bands sit, and the tolerance for matching.
_BAND_LEVELS = (0.02, 0.05, 0.10, 0.20)
_BAND_TOLERANCE = 0.006

#: Below this move a frozen bar is more plausibly a thin single print than a
#: band lock, so it is reported as frozen rather than as a circuit.
_MIN_BAND_MOVE = 0.015


class BandState(str, Enum):
    """What the bar says about whether a price was reachable."""

    OPEN = "open"                    #: traded across a range; normal execution
    UPPER_CIRCUIT = "upper_circuit"  #: frozen at a standard band, above prev close
    LOWER_CIRCUIT = "lower_circuit"  #: frozen at a standard band, below prev close
    FROZEN = "frozen"                #: one price all session, not at a known band
    NO_TRADE = "no_trade"            #: volume was zero; nothing was executable
    UNKNOWN = "unknown"              #: the bar is incomplete; state not determinable


def _at_band(move: float) -> bool:
    return any(abs(abs(move) - level) <= _BAND_TOLERANCE for level in _BAND_LEVELS)


def band_state(
    high: float, low: float, close: float, prev_close: float, volume: float
) -> BandState:
    """Classify one bar.

    An incomplete bar returns UNKNOWN, which is not the same as NO_TRADE.
    NO_TRADE is a fact about the market -- nothing changed hands. UNKNOWN is a
    fact about us -- a column was missing. Collapsing the two would let a gap in
    our own inputs reject the entire universe while reading like a market
    event, which is precisely the failure this module exists to prevent.
    """
    values = (high, low, close, volume)
    if any(v is None or not np.isfinite(float(v)) for v in values):
        return BandState.UNKNOWN
    if float(volume) <= 0:
        return BandState.NO_TRADE
    if float(high) != float(low):
        return BandState.OPEN
    if prev_close is None or not np.isfinite(float(prev_close)) or float(prev_close) <= 0:
        # Frozen is still established by high == low; only the band label needs
        # the previous close, so its absence downgrades the label, not the fact.
        return BandState.FROZEN

    move = float(close) / float(prev_close) - 1.0
    if abs(move) < _MIN_BAND_MOVE or not _at_band(move):
        return BandState.FROZEN
    return BandState.UPPER_CIRCUIT if move > 0 else BandState.LOWER_CIRCUIT


#: States in which the bar demonstrably offered no reachable price.
_UNTRADEABLE = frozenset({
    BandState.UPPER_CIRCUIT, BandState.LOWER_CIRCUIT,
    BandState.FROZEN, BandState.NO_TRADE,
})


def is_untradeable(state: BandState) -> bool:
    """True when the bar demonstrably offered no price to execute against.

    UNKNOWN is excluded. Not knowing is not evidence, and a gate that rejects on
    a missing column would empty the universe on a feed change while reporting a
    market condition. Callers surface UNKNOWN as NOT_TESTABLE instead.
    """
    return state in _UNTRADEABLE


def annotate_band_state(frame: pd.DataFrame) -> pd.Series:
    """Vectorised ``band_state`` over an OHLCV frame, as a string series."""
    required = {"high", "low", "close", "prev_close", "volume"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return pd.Series(dtype="object")

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev = pd.to_numeric(frame["prev_close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    out = pd.Series(BandState.OPEN.value, index=frame.index, dtype="object")
    finite = high.notna() & low.notna() & close.notna() & volume.notna()
    out[~finite] = BandState.UNKNOWN.value
    out[finite & (volume <= 0)] = BandState.NO_TRADE.value

    frozen = finite & (volume > 0) & (high == low)
    move = (close / prev.where(prev > 0)) - 1.0
    at_band = move.abs().ge(_MIN_BAND_MOVE)
    grid = pd.Series(False, index=frame.index)
    for level in _BAND_LEVELS:
        grid |= (move.abs() - level).abs().le(_BAND_TOLERANCE)

    out[frozen] = BandState.FROZEN.value
    out[frozen & at_band & grid & (move > 0)] = BandState.UPPER_CIRCUIT.value
    out[frozen & at_band & grid & (move < 0)] = BandState.LOWER_CIRCUIT.value
    return out
