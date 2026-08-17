"""Out-of-sample validation: CPCV splits, PBO, and the Deflated Sharpe Ratio.

The engine's discipline rests on three numbers, and this package computes all
three honestly:

* **CPCV** gives a *distribution* of out-of-sample results across many
  plausible historical paths, rather than the single path walk-forward tests.
* **PBO** says how often the in-sample winner turns out to be noise.
* **DSR** says how much of a Sharpe ratio survives being charged for the number
  of configurations tried and for non-normal returns.

None of them can be computed on made-up data, and none of them is computed
anywhere else in this codebase. If a number claims to be a PBO or a DSR, it
came from here, on real returns, with a trial count taken from the research
ledger.
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
