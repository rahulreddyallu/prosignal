"""Which listed instruments are EQUITIES, and which only trade like them.

NSE publishes ETFs, index funds, gold and silver funds, liquid funds and bond
funds in the same cash bhavcopy, under the same `EQ` series, as ordinary shares.
A liquidity screen admits them -- several are among the most traded lines on the
exchange -- and a cross-sectional stock model then ranks them against companies.

They do not rank neutrally. A bond or liquid fund has almost no drawdown, almost
no downside volatility, near-zero return kurtosis, a steady upward drift and a
high delivered fraction, so it scores near the top of the risk, ownership and
momentum-consistency themes simultaneously. Measured on a live run before this
filter existed, three of the top five names were Bharat Bond ETFs.

THE RULE. A symbol is excluded when it is ABSENT FROM THE NSE EQUITY MASTER and
either matches a collective-investment-scheme name pattern or shows a volatility
no listed Indian equity sustains. Requiring absence from the master first is
what protects real companies: GOLDIAM, SKYGOLD, SILVERTUC and PNBGILTS all match
the pattern and are all kept, because they are in the master.

This would be unnecessary with a point-in-time index-membership file. There
isn't one in this store, which is also why the universe is a liquidity screen
rather than NIFTY 200.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

__all__ = ["SCHEME_PATTERN", "MIN_EQUITY_VOL", "MIN_VOL_OBS",
           "non_equity_symbols", "equity_only"]

#: Collective-investment-scheme naming on NSE. Matched only against symbols that
#: are ALSO absent from the equity master, so a real company sharing a substring
#: is never removed by it.
SCHEME_PATTERN = re.compile(
    r"(BEES$|ETF|^LIQUID|GOLD|SILVER|GILT|BOND|^SETF|^MON\d|^MAFANG$|"
    r"^MAHKTECH$|IETF$|^BBETF|^EBBETF|^ICICIB\d|^ALPHA$|^NIFTY|^SMALLCAP$|"
    r"SML\d|^MID\d|^NV20|^HNGSNG|^CPSE|^PSUBNK|^JUNIOR|^HDFCNIFTY|^UTINIFT)")

#: Annualised realised volatility below which a line is not behaving like a
#: listed Indian equity. Debt and liquid funds sit near 1-3%; the thinnest real
#: equity in this store is above 15%.
MIN_EQUITY_VOL = 0.10

#: Observations the volatility backstop needs before it may exclude anything.
MIN_VOL_OBS = 250


def non_equity_symbols(symbols: Iterable[str],
                       equity_master: Optional[pd.DataFrame] = None,
                       close: Optional[pd.DataFrame] = None,
                       window: int = 504) -> Set[str]:
    """The symbols to drop, and nothing else."""
    syms = list(symbols)
    known: Set[str] = set()
    if equity_master is not None and not equity_master.empty \
            and "symbol" in equity_master.columns:
        known = set(equity_master["symbol"].astype(str))
    unknown = [s for s in syms if s not in known]
    out = {s for s in unknown if SCHEME_PATTERN.search(str(s))}
    # The volatility backstop needs a real sample. Run on a sixty-session slice
    # it flagged 199 names on this store against the pattern's 61, because a
    # quiet quarter in a thin SME line looks like a liquid fund. It is a
    # backstop for instruments the naming rule misses, not a screen in its own
    # right, so it requires MIN_VOL_OBS observations and stays silent below that.
    if close is not None and not close.empty and len(close) >= MIN_VOL_OBS:
        cols = [s for s in unknown if s in close.columns]
        if cols:
            tail = close[cols].tail(int(window))
            ret = tail / tail.shift(1) - 1.0
            enough = ret.notna().sum() >= MIN_VOL_OBS
            vol = ret.std() * np.sqrt(252)
            out |= {s for s in cols
                    if bool(enough.get(s, False))
                    and np.isfinite(vol.get(s, np.nan))
                    and vol[s] < MIN_EQUITY_VOL}
    return out


def equity_only(symbols: Iterable[str], **kw) -> list:
    drop = non_equity_symbols(symbols, **kw)
    return [s for s in symbols if s not in drop]
