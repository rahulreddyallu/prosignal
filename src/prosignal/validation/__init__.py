"""Out-of-sample validation: CPCV splits, PBO, and the Deflated Sharpe Ratio.

* CPCV gives a distribution of out-of-sample results across many plausible
  historical paths, rather than the single path walk-forward tests.
* PBO estimates how often the in-sample winner is noise.
* DSR discounts a Sharpe ratio for the number of configurations tried and for
  non-normal returns.

None is computed anywhere else in this codebase. A number claiming to be a PBO
or DSR came from here, on real returns, with a trial count from the ledger.
"""

from __future__ import annotations

from .cpcv import CombinatorialPurgedCV, CpcvSplit, make_groups
from .metrics import (
    DsrResult,
    PboResult,
    compute_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

__all__ = [
    "CombinatorialPurgedCV",
    "CpcvSplit",
    "make_groups",
    "DsrResult",
    "PboResult",
    "compute_pbo",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "norm_cdf",
    "norm_ppf",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]
