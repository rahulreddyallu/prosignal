"""Yahoo Finance provider -- the secondary source.

Three jobs the NSE archives cannot do alone:

1. Cross-source agreement. Stage 1 wants two independent price sources where
   possible, so disagreement is flagged rather than resolved by preference.

2. Corporate-action ratios. Yahoo maps Indian bonus issues into its
   ``Stock Splits`` series -- Reliance's Oct-2024 1:1 bonus appears as ratio
   2.0 -- giving the adjustment engine real ratios without depending on NSE's
   frequently-403 JSON API.

3. Scheduled earnings dates for the Stage 3 earnings-proximity gate.

Two cautions handled in code rather than left to the caller:

* Yahoo prices are its own reconstruction, not the exchange record. NSE stays
  primary for anything feeding a return calculation.
* Yahoo returns timezone-aware timestamps for ``.NS`` tickers and US/Eastern
  earnings dates. Both are converted to IST dates here, once.
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...core.errors import ProviderError
from ...core.logging import get_logger
from ..types import (
    CORPORATE_ACTION_COLUMNS,
    DATE,
    SYMBOL,
    coerce_ohlcv,
    empty_index_frame,
    normalise_symbol,
)

__all__ = ["YFinanceProvider"]

log = get_logger(__name__)

_IST = "Asia/Kolkata"


def _to_naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Convert any tz-aware/naive index to tz-naive IST midnight timestamps."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(_IST).tz_localize(None)
    return idx.normalize()


class YFinanceProvider:
    """Thin, defensive wrapper over ``yfinance``."""

    name = "yfinance"

    def __init__(self, cfg: "object") -> None:
        self.cfg = cfg
        self._yf = None
        self.last_error: Optional[str] = None

    # -- lazy import ---------------------------------------------------------
    @property
    def yf(self):
        if self._yf is None:
            try:
                import yfinance  # noqa: PLC0415 - deliberately lazy
            except ImportError as exc:  # pragma: no cover
                raise ProviderError(
                    self.name,
                    "yfinance is not installed; run `pip install -r requirements.txt`",
                ) from exc
            self._yf = yfinance
        return self._yf

    # -- symbol mapping ------------------------------------------------------
    def to_yahoo(self, symbol: str) -> str:
        sym = normalise_symbol(symbol)
        overrides: Dict[str, str] = dict(getattr(self.cfg, "symbol_overrides", {}) or {})
        if sym in overrides:
            return overrides[sym]
        return f"{sym}{self.cfg.equity_suffix}"

    def from_yahoo(self, yahoo_symbol: str) -> str:
        overrides: Dict[str, str] = dict(getattr(self.cfg, "symbol_overrides", {}) or {})
        for nse, yahoo in overrides.items():
            if yahoo == yahoo_symbol:
                return nse
        return normalise_symbol(yahoo_symbol)

    # =========================================================================
    # Prices
    # =========================================================================
    def fetch_ohlcv(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        """Unadjusted OHLCV plus Yahoo's adjusted close, in tidy form.

        ``auto_adjust=False`` is deliberate: the engine performs its own
        corporate-action adjustment from explicit ratios so that the adjustment
        is auditable, and keeps Yahoo's ``Adj Close`` alongside purely as a
        cross-check on that work.
        """
        symbols = [normalise_symbol(s) for s in symbols if str(s).strip()]
        if not symbols:
            return coerce_ohlcv(pd.DataFrame(), source=self.name)

        batch_size = int(getattr(self.cfg, "batch_size", 40))
        pause = float(getattr(self.cfg, "pause_between_batches_seconds", 1.0))
        frames: List[pd.DataFrame] = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            tickers = [self.to_yahoo(s) for s in batch]
            raw = self._download(tickers, start, end)
            if raw is None or raw.empty:
                continue
            frames.extend(self._split_download(raw, batch, tickers))
            if pause > 0 and i + batch_size < len(symbols):
                import time

                time.sleep(pause)

        if not frames:
            return coerce_ohlcv(pd.DataFrame(), source=self.name)
        return coerce_ohlcv(pd.concat(frames, ignore_index=True), source=self.name)

    def _download(
        self, tickers: List[str], start: dt.date, end: dt.date
    ) -> Optional[pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return self.yf.download(
                    tickers=tickers,
                    start=start.isoformat(),
                    # yfinance treats `end` as exclusive.
                    end=(end + dt.timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception as exc:  # yfinance raises a wide variety
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "yfinance download failed",
                    extra={"tickers": len(tickers), "error": self.last_error},
                )
                return None

    def _split_download(
        self, raw: pd.DataFrame, nse_symbols: List[str], yahoo_tickers: List[str]
    ) -> List[pd.DataFrame]:
        """Normalise yfinance's single-vs-multi-ticker column shapes."""
        out: List[pd.DataFrame] = []
        multi = isinstance(raw.columns, pd.MultiIndex)

        for nse_sym, yahoo_sym in zip(nse_symbols, yahoo_tickers):
            if multi:
                level0 = set(raw.columns.get_level_values(0))
                if yahoo_sym in level0:
                    sub = raw[yahoo_sym]
                else:
                    # Some yfinance versions order the levels (field, ticker).
                    level1 = set(raw.columns.get_level_values(1))
                    if yahoo_sym in level1:
                        sub = raw.xs(yahoo_sym, axis=1, level=1)
                    else:
                        continue
            else:
                if len(yahoo_tickers) > 1:
                    continue
                sub = raw

            sub = sub.dropna(how="all")
            if sub.empty:
                continue

            frame = pd.DataFrame(
                {
                    DATE: _to_naive_dates(sub.index),
                    SYMBOL: nse_sym,
                    "open": pd.to_numeric(sub.get("Open"), errors="coerce").to_numpy(),
                    "high": pd.to_numeric(sub.get("High"), errors="coerce").to_numpy(),
                    "low": pd.to_numeric(sub.get("Low"), errors="coerce").to_numpy(),
                    "close": pd.to_numeric(sub.get("Close"), errors="coerce").to_numpy(),
                    "volume": pd.to_numeric(sub.get("Volume"), errors="coerce").to_numpy(),
                }
            )
            if "Adj Close" in sub.columns:
                adj = pd.to_numeric(sub["Adj Close"], errors="coerce").to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    frame["adj_factor"] = np.where(
                        frame["close"].to_numpy() > 0, adj / frame["close"].to_numpy(), 1.0
                    )
            frame = frame.dropna(subset=["close"])
            if not frame.empty:
                out.append(frame)
        return out

    # =========================================================================
    # Indices
    # =========================================================================
    def fetch_index(self, index_name: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Index OHLC via Yahoo, as a cross-check on ``ind_close_all``."""
        symbols: Dict[str, str] = dict(getattr(self.cfg, "index_symbols", {}) or {})
        yahoo_sym = symbols.get(index_name)
        if not yahoo_sym:
            return empty_index_frame()

        raw = self._download([yahoo_sym], start, end)
        if raw is None or raw.empty:
            return empty_index_frame()
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(raw.columns.get_level_values(0))
            raw = raw[yahoo_sym] if yahoo_sym in level0 else raw.droplevel(1, axis=1)

        out = pd.DataFrame(
            {
                DATE: _to_naive_dates(raw.index),
                "index_name": index_name,
                "open": pd.to_numeric(raw.get("Open"), errors="coerce").to_numpy(),
                "high": pd.to_numeric(raw.get("High"), errors="coerce").to_numpy(),
                "low": pd.to_numeric(raw.get("Low"), errors="coerce").to_numpy(),
                "close": pd.to_numeric(raw.get("Close"), errors="coerce").to_numpy(),
                "volume": pd.to_numeric(raw.get("Volume"), errors="coerce").to_numpy(),
            }
        )
        out["source"] = self.name
        out = out.dropna(subset=["close"])
        return out.reindex(columns=empty_index_frame().columns).reset_index(drop=True)

    # =========================================================================
    # Corporate actions
    # =========================================================================
    def fetch_corporate_actions(self, symbols: Iterable[str]) -> pd.DataFrame:
        """Splits and bonuses as ex-date-stamped multiplicative price factors.

        Yahoo's ``Stock Splits`` value is the share-multiplication ratio (2.0
        for a 1:1 bonus or a 2-for-1 split). The engine stores the *price*
        factor, i.e. ``1 / ratio``: multiply pre-ex-date prices by it to put
        them on the same footing as post-ex-date prices.
        """
        rows: List[Dict[str, object]] = []
        for sym in symbols:
            nse_sym = normalise_symbol(sym)
            try:
                ticker = self.yf.Ticker(self.to_yahoo(nse_sym))
                splits = ticker.splits
                divs = ticker.dividends
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.debug("corporate actions unavailable", extra={"symbol": nse_sym})
                continue

            if splits is not None and len(splits):
                for ts, ratio in splits.items():
                    ratio = float(ratio)
                    if ratio <= 0:
                        continue
                    rows.append(
                        {
                            SYMBOL: nse_sym,
                            "ex_date": _to_naive_dates(pd.DatetimeIndex([ts]))[0],
                            "action_type": "split_or_bonus",
                            "ratio": 1.0 / ratio,
                            "raw_details": f"share ratio {ratio:g}",
                            "source": self.name,
                        }
                    )

            if divs is not None and len(divs):
                for ts, amount in divs.items():
                    rows.append(
                        {
                            SYMBOL: nse_sym,
                            "ex_date": _to_naive_dates(pd.DatetimeIndex([ts]))[0],
                            "action_type": "dividend",
                            # Dividends do not rescale the price series the way
                            # a split does; kept for the anomaly cross-check,
                            # not applied as an adjustment factor.
                            "ratio": 1.0,
                            "raw_details": f"dividend {float(amount):g}",
                            "source": self.name,
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
        return pd.DataFrame(rows)[CORPORATE_ACTION_COLUMNS]

    # =========================================================================
    # Earnings calendar
    # =========================================================================
    def fetch_earnings_dates(self, symbols: Iterable[str], limit: int = 12) -> pd.DataFrame:
        """Scheduled and historical results dates, converted to IST dates."""
        rows: List[Dict[str, object]] = []
        for sym in symbols:
            nse_sym = normalise_symbol(sym)
            try:
                ticker = self.yf.Ticker(self.to_yahoo(nse_sym))
                table = ticker.get_earnings_dates(limit=limit)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.debug("earnings dates unavailable", extra={"symbol": nse_sym})
                continue
            if table is None or len(table) == 0:
                continue
            for ts in _to_naive_dates(table.index):
                rows.append(
                    {
                        SYMBOL: nse_sym,
                        "earnings_date": ts,
                        # Yahoo does not distinguish confirmed from estimated
                        # dates for Indian names; saying so is more useful than
                        # implying a certainty that is not there.
                        "confirmed": False,
                        "source": self.name,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=[SYMBOL, "earnings_date", "confirmed", "source"])
        return (
            pd.DataFrame(rows)
            .drop_duplicates(subset=[SYMBOL, "earnings_date"])
            .sort_values([SYMBOL, "earnings_date"])
            .reset_index(drop=True)
        )
