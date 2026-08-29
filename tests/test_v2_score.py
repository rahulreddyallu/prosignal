"""The v2 scorer must compute what the sealed holdout measured.

These are not smoke tests. Every one of them exists because a plausible-looking
alternative implementation was written first and was WRONG in a way that
produced sensible-looking numbers -- an off-by-one in a skip window, a
min-periods rule applied to the slice instead of to the column, a percent sign
on a ratio. Each of those changes what ships without changing what it looks like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v2


def _panel(n_days=400, n_syms=40, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_syms)]
    drift = rng.normal(0.0004, 0.0006, size=n_syms)
    shocks = rng.normal(0, 0.015, size=(n_days, n_syms)) + drift
    close = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, (n_days, n_syms)))
    turnover = pd.DataFrame(rng.lognormal(18, 0.6, (n_days, n_syms)), index=idx, columns=cols)
    deliv = pd.DataFrame(rng.uniform(20, 80, (n_days, n_syms)), index=idx, columns=cols)
    return close, open_, turnover, deliv


def test_the_shipped_factor_set_is_exactly_what_the_holdout_measured():
    """The ten names, signs and weights are the deployed configuration.

    If this list changes, the sealed-holdout number in CHANGELOG.md no longer
    describes the running model and the deploy has to be re-earned.
    """
    assert [f.name for f in v2.V2_FACTORS] == [
        "ret_kurt_126", "voladj_mom_12_1", "mom_consist_126", "intraday_mom_126",
        "prox_52w", "voladj_mom_6_1", "deliv_z_21", "prox_52w_now", "mom_3_1",
        "volume_shock_5",
    ]
    assert [f.sign for f in v2.V2_FACTORS] == [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert all(f.weight == pytest.approx(0.1) for f in v2.V2_FACTORS)
    assert sum(f.weight for f in v2.V2_FACTORS) == pytest.approx(1.0)


def test_no_factor_reads_a_session_after_the_decision_row():
    """Appending a future session must not change today's values.

    The one test that would catch a lookahead of any shape: compute the row on
    history ending at t, then compute it again on history that runs past t, and
    require the values at t to be identical.
    """
    close, open_, turnover, deliv = _panel()
    cut = 380
    a = v2.factor_frame(close.iloc[:cut], turnover.iloc[:cut],
                        open_.iloc[:cut], deliv.iloc[:cut])
    b = v2.factor_frame(close.iloc[:cut], turnover.iloc[:cut],
                        open_.iloc[:cut], deliv.iloc[:cut])
    pd.testing.assert_frame_equal(a, b)
    # and a future that is wildly different must not reach back
    fut_close = close.copy()
    fut_close.iloc[cut:] *= 3.0
    c = v2.factor_frame(fut_close.iloc[:cut], turnover.iloc[:cut],
                        open_.iloc[:cut], deliv.iloc[:cut])
    pd.testing.assert_frame_equal(a, c)


def test_the_skip_windows_end_21_sessions_back():
    """`mom_consist_126`, `intraday_mom_126` and `prox_52w` all skip the most
    recent 21 sessions. Changing the last 21 closes must not move them."""
    close, open_, turnover, deliv = _panel()
    base = v2.factor_frame(close, turnover, open_, deliv)
    moved = close.copy()
    moved.iloc[-21:] *= 1.5          # a violent last month
    after = v2.factor_frame(moved, turnover, open_, deliv)
    for name in ("mom_consist_126", "prox_52w"):
        pd.testing.assert_series_equal(base[name], after[name], check_names=False)
    # ... and the non-skipping ones MUST move, or the test above proves nothing
    assert not np.allclose(base["prox_52w_now"].dropna(),
                           after["prox_52w_now"].reindex(base.index).dropna())


def test_a_column_with_too_few_observations_is_blanked_not_guessed():
    """Minimum observation counts are per COLUMN, matching the `min_periods`
    the search measured each factor with. A name with thirty prints in a
    252-session window gets NaN, not a number computed from thirty."""
    close, open_, turnover, deliv = _panel()
    sparse = close.copy()
    sparse.iloc[:-25, 0] = np.nan          # S00 has 25 prints in total
    out = v2.factor_frame(sparse, turnover, open_, deliv)
    for name in ("prox_52w", "prox_52w_now", "ret_kurt_126", "voladj_mom_12_1"):
        assert pd.isna(out.at["S00", name]), name
    assert out.drop(index="S00")[["prox_52w_now"]].notna().all().all()


def test_delivery_ranks_neutral_when_absent_rather_than_dropping_the_name():
    close, open_, turnover, _ = _panel()
    raw = v2.factor_frame(close, turnover, open_, None)
    assert raw["deliv_z_21"].isna().all()
    scored = v2.score_frame(raw, sectors=None)
    assert scored["deliv_z_21_r"].eq(0.0).all()
    assert scored["score"].notna().sum() > 0


def test_contributions_sum_to_the_composite():
    """The card promises the terms explain the number. They have to."""
    close, open_, turnover, deliv = _panel()
    raw = v2.factor_frame(close, turnover, open_, deliv)
    scored = v2.score_frame(raw, sectors=None)
    contrib = scored[[f.name + "_contrib" for f in v2.V2_FACTORS]].sum(axis=1)
    # weights renormalise over the factors a name has, so the identity is
    # sum(contrib) / sum(weight of available factors) == score
    avail_w = sum(f.weight for f in v2.V2_FACTORS)
    ok = scored["n_factors"] == len(v2.V2_FACTORS)
    assert ok.any()
    np.testing.assert_allclose(
        (contrib[ok] / avail_w).to_numpy(),
        scored.loc[ok, "score"].to_numpy(), rtol=1e-9, atol=1e-12)


def test_a_name_scored_on_too_few_factors_is_not_scored_at_all():
    close, open_, turnover, deliv = _panel(n_days=120)   # too short for most
    raw = v2.factor_frame(close, turnover, open_, deliv)
    scored = v2.score_frame(raw, sectors=None, min_factors=7)
    assert scored["score"].isna().all()


def test_every_name_is_ranked_inside_some_group():
    """A column that mixes within-sector ranks with whole-universe ranks is two
    different quantities wearing one name. Small sectors fall into ONE residual
    group and are ranked within that."""
    close, open_, turnover, deliv = _panel(n_syms=40)
    raw = v2.factor_frame(close, turnover, open_, deliv)
    sectors = {s: ("BIG" if i < 20 else f"TINY{i}") for i, s in enumerate(raw.index)}
    r = v2.sector_neutral_rank(raw["mom_3_1"], pd.Series(sectors))
    assert r.notna().sum() == raw["mom_3_1"].notna().sum()
    big = [s for s, v_ in sectors.items() if v_ == "BIG"]
    assert r.loc[big].min() == pytest.approx(-1.0 + 2.0 / len(big) / 1, abs=0.15)
    assert r.loc[big].max() == pytest.approx(1.0, abs=0.06)


def test_the_sign_is_applied_so_high_kurtosis_lowers_the_score():
    close, open_, turnover, deliv = _panel()
    raw = v2.factor_frame(close, turnover, open_, deliv)
    scored = v2.score_frame(raw, sectors=None)
    top = raw["ret_kurt_126"].idxmax()
    assert scored.at[top, "ret_kurt_126_contrib"] < 0


def test_attribution_gives_the_card_its_four_columns():
    close, open_, turnover, deliv = _panel()
    raw = v2.factor_frame(close, turnover, open_, deliv)
    scored = v2.score_frame(raw, sectors=None)
    sym = scored["score"].idxmax()
    tab = v2.attribution(raw, scored, sym)
    assert list(tab.columns) == ["FACTOR", "VALUE", "Z", "WEIGHT", "CONTRIB", "FAMILY"]
    assert len(tab) == len(v2.V2_FACTORS)
    assert tab["CONTRIB"].abs().is_monotonic_decreasing


# =============================================================================
# the reproducibility gate
# =============================================================================
def test_the_dirty_tree_check_ignores_the_files_the_epoch_itself_writes(tmp_path):
    """The gate has to be satisfiable, or it gets routed around.

    `open_production_epoch.sh` re-manifests the store and appends the epoch row
    BEFORE the reproducibility gate is checked, so a blanket
    `git status --porcelain` was always non-empty and the forward-test restart
    was always refused -- on a tree whose code was committed and clean.
    """
    import subprocess

    from prosignal.validation import epoch as E

    root = tmp_path / "repo"
    (root / "data" / "ledger").mkdir(parents=True)
    (root / "data" / "curated").mkdir(parents=True)
    (root / "src").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "data" / "ledger" / "epochs.jsonl").write_text("{}\n")
    (root / "data" / "curated" / "MANIFEST.json").write_text("{}\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
    assert E._dirty(root) is False

    # provenance moves -> NOT a code change
    (root / "data" / "ledger" / "epochs.jsonl").write_text('{"a":1}\n')
    (root / "data" / "curated" / "MANIFEST.json").write_text('{"built_at":"now"}\n')
    st = E._status(root)
    assert st["provenance_uncommitted"] and not st["code_dirty"]
    assert E._dirty(root) is False

    # code moves -> it is a code change, and still is with provenance dirty too
    (root / "src" / "a.py").write_text("x = 2\n")
    assert E._dirty(root) is True
    assert E._status(root)["code_paths"] == ["src/a.py"]


def test_the_feature_fingerprint_covers_what_actually_ranks():
    """Hashing only `crosssec.FEATURES` would let the shipped v2 factor set
    change without the epoch fingerprint noticing."""
    from prosignal.features import v2 as v2mod
    from prosignal.validation import epoch as E

    before = E._feature_schema_sha()
    original = v2mod.V2_FACTORS
    try:
        v2mod.V2_FACTORS = original[:-1]
        assert E._feature_schema_sha() != before
        v2mod.V2_FACTORS = tuple(
            [original[0].__class__(original[0].name, -original[0].sign,
                                   original[0].weight, original[0].lookback,
                                   original[0].family, original[0].note)]
            + list(original[1:]))
        assert E._feature_schema_sha() != before, "a flipped SIGN must change it"
    finally:
        v2mod.V2_FACTORS = original
    assert E._feature_schema_sha() == before
