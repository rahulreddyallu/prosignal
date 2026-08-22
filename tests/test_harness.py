"""The CPCV harness: leakage guards, path weaving, and the shape of the result.

validation/cpcv.py was written, tested and never called by anything. The whole
repository's evidence came from one walk-forward path, and its stated limitation
was always that a single sequence of windows cannot show dispersion. These tests
cover the module that finally uses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.harness import configuration_matrix, run_cpcv


def _panel(n_dates: int = 60, n_symbols: int = 60, seed: int = 0) -> pd.DataFrame:
    """A panel where one feature genuinely predicts and one is pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2019-01-01", periods=n_dates, freq="21B")
    for d in dates:
        signal = rng.normal(size=n_symbols)
        noise = rng.normal(size=n_symbols)
        label = 0.6 * signal + rng.normal(scale=1.2, size=n_symbols)
        rows.append(pd.DataFrame({
            "date": d,
            "symbol": [f"S{i:03d}" for i in range(n_symbols)],
            "signal_r": signal, "noise_r": noise,
            "label": label * 0.02,
            "label_rank": pd.Series(label).rank(pct=True).to_numpy() * 2 - 1,
        }))
    return pd.concat(rows, ignore_index=True)


FEATURES = ["signal_r", "noise_r"]
KW = dict(horizon_sessions=63, step_sessions=21, alpha=100.0,
          n_groups=6, n_test_groups=2, purge_sessions=63, embargo_sessions=21,
          min_train_rows=200)


def test_it_weaves_the_number_of_paths_the_combinatorics_promise():
    r = run_cpcv(_panel(), FEATURES, **KW)
    from math import comb
    assert r.n_splits == comb(6, 2) == 15
    assert r.n_paths == comb(5, 1) == 5
    assert len(r.path_sharpes) == r.n_paths, (
        "every observation is tested C(N-1,k-1) times, so that many complete "
        "out-of-sample paths must come back -- fewer means dates were dropped"
    )


def test_it_finds_a_signal_that_is_there():
    r = run_cpcv(_panel(), FEATURES, **KW)
    assert r.mean_ic > 0.10, f"planted signal not recovered: IC {r.mean_ic}"
    assert np.mean(r.excess) > 0


def test_it_does_not_find_a_signal_that_is_not_there():
    """Guards the guard. A harness that always reports an edge is worthless."""
    panel = _panel()
    panel["label_rank"] = np.random.default_rng(7).permutation(panel["label_rank"].to_numpy())
    panel["label"] = np.random.default_rng(8).permutation(panel["label"].to_numpy())
    r = run_cpcv(panel, FEATURES, **KW)
    assert abs(r.mean_ic) < 0.05, f"found IC {r.mean_ic} in shuffled labels"


def test_purging_actually_removes_observations():
    r = run_cpcv(_panel(), FEATURES, **KW)
    assert r.purged_total > 0, (
        "a 63-session label at 21-session steps overlaps three observations; "
        "purging nothing means the leak the harness exists to prevent is open"
    )
    assert r.embargoed_total > 0


def test_a_purge_shorter_than_the_label_leaks_and_shows_it():
    """The measurable consequence of under-purging, on planted data."""
    honest = run_cpcv(_panel(), FEATURES, **KW)
    leaky = run_cpcv(_panel(), FEATURES, **{**KW, "purge_sessions": 0, "embargo_sessions": 0})
    assert leaky.purged_total == 0
    assert honest.purged_total > 0


def test_too_few_dates_for_the_group_count_is_refused():
    with pytest.raises(ValueError, match="cannot support"):
        run_cpcv(_panel(n_dates=8), FEATURES, **{**KW, "n_groups": 10})


def test_the_configuration_matrix_scores_every_column_on_one_index():
    """PBO compares configurations, so the columns must be commensurable."""
    m = configuration_matrix(
        _panel(), {"both": FEATURES, "signal_only": ["signal_r"],
                   "noise_only": ["noise_r"]},
        step_sessions=21, alpha=100.0, purge_sessions=63,
        min_train_dates=10, min_train_rows=200,
    )
    assert list(m.columns) == ["both", "signal_only", "noise_only"]
    assert not m.isna().any().any(), "a ragged matrix would bias the PBO ranking"
    assert m["signal_only"].mean() > m["noise_only"].mean()
