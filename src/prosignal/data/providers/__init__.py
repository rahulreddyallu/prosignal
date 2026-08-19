"""Data providers, in priority order.

* :class:`NseArchivesProvider` -- primary. Official, free, unauthenticated;
  ``nsearchives.nseindia.com`` and ``archives.nseindia.com`` both return 200.
  Supplies OHLCV, delivery, all index series including India VIX, constituents
  with sector labels, listing dates and F&O open interest.
* :class:`YFinanceProvider` -- secondary. The independent second source Stage 1
  needs for cross-source agreement, plus corporate-action ratios and earnings
  dates.
* :class:`CsvImportProvider` -- manual. Feeds no free source supplies honestly
  (promoter pledging, point-in-time fundamentals). An absent file means
  NOT_TESTABLE, never a silent pass.
* :class:`NseJsonSession` -- best effort. ``www.nseindia.com``'s JSON API sits
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
