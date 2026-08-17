"""Combinatorial Purged Cross-Validation.

Walk-forward validation tests a single historical path. That makes the result
contingent on the one sequence of train/test windows you happened to draw, and
it offers no defence at all against a researcher quietly trying many
configurations along that path. Arian, Norouzi & Seco (2024) find walk-forward
has "notable shortcomings in false discovery prevention" relative to CPCV on
both PBO and DSR criteria.

CPCV instead produces a *distribution* of out-of-sample estimates across many
plausible historical paths. You then evaluate the stability of that
distribution rather than a single headline number.

The construction (López de Prado):

1. Split the history into ``N`` contiguous, chronological groups.
2. Take every combination of ``k`` groups as the test set -- ``C(N, k)`` splits.
3. **Purge** from training any observation whose label window overlaps the test
   window. This matters enormously here: a 21-session forward-return label
   means an observation 20 sessions before a test block still "knows" part of
   that block's outcome.
4. **Embargo** a further span after each test block before training resumes, to
   kill residual serial-correlation leakage.

Purging and embargoing are not optional refinements. Without them a CPCV run
reports optimistic numbers for a mechanical reason and you will believe them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from ..core.errors import ConfigError

__all__ = ["CpcvSplit", "CombinatorialPurgedCV", "make_groups"]


@dataclass
class CpcvSplit:
    """One train/test partition, already purged and embargoed."""

    split_id: int
    test_groups: Tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    purged_count: int = 0
    embargoed_count: int = 0

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)

    def summary(self) -> dict:
        return {
            "split_id": self.split_id,
            "test_groups": list(self.test_groups),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "purged": self.purged_count,
            "embargoed": self.embargoed_count,
        }


def make_groups(n_obs: int, n_groups: int) -> List[np.ndarray]:
    """Split ``n_obs`` chronological observations into contiguous groups."""
    if n_groups < 2:
        raise ConfigError("cpcv requires at least 2 groups")
    if n_obs < n_groups:
        raise ConfigError(
            f"cannot split {n_obs} observations into {n_groups} groups"
        )
    return [g for g in np.array_split(np.arange(n_obs), n_groups)]


class CombinatorialPurgedCV:
    """Generate purged, embargoed train/test splits over a time series.

    Parameters
    ----------
    n_groups, n_test_groups:
        ``C(n_groups, n_test_groups)`` splits are produced.
    label_horizon:
        Length of the forward-return label, in observations. An observation at
        ``t`` carries information about ``[t, t + label_horizon]``, so training
        rows whose label window reaches into a test block are purged.
    embargo:
        Observations dropped from training immediately after each test block.
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        label_horizon: int = 21,
        embargo: int = 21,
    ) -> None:
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ConfigError(
                f"n_test_groups must satisfy 1 <= k < n_groups; got "
                f"k={n_test_groups}, n_groups={n_groups}"
            )
        if label_horizon < 0 or embargo < 0:
            raise ConfigError("label_horizon and embargo must be non-negative")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.label_horizon = label_horizon
        self.embargo = embargo

    # -- geometry -----------------------------------------------------------
    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def paths_per_observation(self) -> int:
        """How many distinct backtest paths the splits can be woven into.

        Each observation appears in the test set of ``C(N-1, k-1)`` splits, so
        that is the number of complete out-of-sample paths CPCV yields.
        """
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    # -- splitting ----------------------------------------------------------
    def split(self, n_obs: int) -> Iterator[CpcvSplit]:
        groups = make_groups(n_obs, self.n_groups)
        all_idx = np.arange(n_obs)

        for split_id, test_combo in enumerate(
            combinations(range(self.n_groups), self.n_test_groups)
        ):
            test_idx = np.sort(np.concatenate([groups[g] for g in test_combo]))
            candidate_train = np.setdiff1d(all_idx, test_idx, assume_unique=True)

            # Contiguous test blocks (adjacent chosen groups merge into one).
            blocks = _contiguous_blocks(test_idx)

            purged = np.zeros(candidate_train.shape, dtype=bool)
            embargoed = np.zeros(candidate_train.shape, dtype=bool)

            for start, end in blocks:
                # PURGE: a training observation at t has a label spanning
                # [t, t + horizon]. If that reaches the test block, it overlaps.
                overlap = (candidate_train + self.label_horizon >= start) & (
                    candidate_train <= end
                )
                purged |= overlap

                # EMBARGO: drop training observations just after the block.
                if self.embargo > 0:
                    after = (candidate_train > end) & (
                        candidate_train <= end + self.embargo
                    )
                    embargoed |= after

            drop = purged | embargoed
            train_idx = candidate_train[~drop]

            yield CpcvSplit(
                split_id=split_id,
                test_groups=test_combo,
                train_idx=train_idx,
                test_idx=test_idx,
                purged_count=int(np.count_nonzero(purged & ~embargoed)),
                embargoed_count=int(np.count_nonzero(embargoed)),
            )

    def split_dates(
        self, dates: Sequence[dt.date]
    ) -> Iterator[Tuple[CpcvSplit, List[dt.date], List[dt.date]]]:
        """Convenience wrapper yielding actual dates alongside indices."""
        dates = list(dates)
        for sp in self.split(len(dates)):
            yield (
                sp,
                [dates[i] for i in sp.train_idx],
                [dates[i] for i in sp.test_idx],
            )

    # -- reporting ----------------------------------------------------------
    def describe(self, n_obs: int) -> dict:
        splits = list(self.split(n_obs))
        train_sizes = [s.n_train for s in splits]
        return {
            "n_observations": n_obs,
            "n_groups": self.n_groups,
            "n_test_groups": self.n_test_groups,
            "n_splits": len(splits),
            "backtest_paths": self.paths_per_observation(),
            "label_horizon": self.label_horizon,
            "embargo": self.embargo,
            "mean_train_size": float(np.mean(train_sizes)) if train_sizes else 0.0,
            "min_train_size": int(np.min(train_sizes)) if train_sizes else 0,
            "total_purged": int(sum(s.purged_count for s in splits)),
            "total_embargoed": int(sum(s.embargoed_count for s in splits)),
        }


def _contiguous_blocks(idx: np.ndarray) -> List[Tuple[int, int]]:
    """Collapse a sorted index array into (start, end) inclusive runs."""
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]
