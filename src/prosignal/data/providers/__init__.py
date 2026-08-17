"""Data providers.

Priority is deliberate and reflects what was verified working against the live
hosts during the build:

* :class:`NseArchivesProvider` -- PRIMARY. Official, free, unauthenticated, and
  reachable (``nsearchives.nseindia.com`` and ``archives.nseindia.com`` both
  return 200). Supplies OHLCV, delivery, all index series including India VIX,
  index constituents with sector labels, listing dates, and F&O open interest.
* :class:`YFinanceProvider` -- SECONDARY. The independent second opinion Stage 1
  needs for cross-source agreement, plus corporate-action ratios and earnings
  dates.
* :class:`CsvImportProvider` -- MANUAL. Anything no free source supplies
  honestly (promoter pledging, point-in-time fundamentals). Absent file means
  NOT_TESTABLE, never a silent pass.
* :class:`NseJsonSession` -- BEST-EFFORT. ``www.nseindia.com``'s JSON API sits
  behind a bot shield that returned 403 from the build machine, so no required
  feed depends on it.
"""

from __future__ import annotations

from .csv_import import REFERENCE_TEMPLATES, CsvImportProvider
from .http import FetchResult, HttpClient, NseJsonSession
from .nse_archives import INDIA_VIX_NAME, NseArchivesProvider
from .yfinance_provider import YFinanceProvider

__all__ = [
    "CsvImportProvider",
    "REFERENCE_TEMPLATES",
    "FetchResult",
    "HttpClient",
    "NseJsonSession",
    "NseArchivesProvider",
    "INDIA_VIX_NAME",
    "YFinanceProvider",
]
