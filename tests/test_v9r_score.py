"""The v9R scorer must reproduce the research implementation to machine precision.

A model that earned an out-of-sample number has to compute what it was measured
computing. `features/v9r.py` is a reimplementation of `work/v9r/engine.py` for the
production path, and a reimplementation that is merely close is a different model
with the first model's credibility attached to it.

The research code ranks with a 0-based ordinal rank scaled to [-1, 1] and stable
tie-breaking; pandas' `rank(pct=True)` averages ties and normalises differently.
Both are defensible. Only one of them was measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v9r


def _reference_rank(a: np.ndarray) -> np.ndarray:
    """The research implementation, transcribed from the v9R engine.

    The research tree was deleted in the 2026-09-03 cleanup, so this transcription
    is now the reference the production scorer is held to. Changing it changes what
    the sealed +9.50% / t +1.87 result describes.
    """
    ok = np.isfinite(a)
    out = np.full(a.shape, np.nan)
    n = int(ok.sum())
    if n == 0:
        return out
    order = np.argsort(np.where(ok, a, np.inf), kind="stable")
    rk = np.empty(len(a)); rk[order] = np.arange(len(a))
    out[ok] = (rk[ok] / max(n - 1, 1)) * 2.0 - 1.0
    return out


def test_rank_matches_the_research_convention():
    rng = np.random.default_rng(11)
    for _ in range(30):
        a = rng.normal(size=200)
        a[rng.random(200) < 0.15] = np.nan
        got = v9r.rank_pct(pd.Series(a)).to_numpy()
        want = _reference_rank(a)
        assert np.allclose(got, want, equal_nan=True)


def test_rank_spans_minus_one_to_one_and_preserves_order():
    s = pd.Series([5.0, 1.0, 3.0, np.nan, 2.0])
    r = v9r.rank_pct(s)
    assert np.isnan(r.iloc[3])
    assert r.min() == pytest.approx(-1.0)
    assert r.max() == pytest.approx(1.0)
    # ordering preserved
    fin = r.dropna()
    assert list(fin.sort_values().index) == [1, 4, 2, 0]


def test_missing_factor_contributes_zero_and_does_not_reweight():
    """The defect this model exists to remove: a thin factor must NOT donate its
    weight to the factors that happen to be present."""
    idx = [f"S{i}" for i in range(60)]
    rng = np.random.default_rng(3)
    raw = pd.DataFrame({f: rng.normal(size=60) for f in v9r.FACTORS}, index=idx)
    full = v9r.score_frame(raw)

    holed = raw.copy()
    holed.loc[idx[:30], "ret_kurt_126"] = np.nan     # 18.16% of weight, half the names
    part = v9r.score_frame(holed)

    w = v9r.WEIGHTS["ret_kurt_126"] / sum(v9r.WEIGHTS.values())
    lost, kept = idx[:30], idx[30:]

    # coverage falls by exactly the missing factor's weight, and by nothing else
    assert np.allclose(part.loc[lost, "coverage"], 1.0 - w, atol=1e-12)
    assert np.allclose(part.loc[kept, "coverage"], 1.0, atol=1e-12)

    # THE POINT OF THE TEST. For a name that lost the factor, the score must equal
    # the weighted sum over the factors it still has -- with those weights UNCHANGED.
    # Under coverage renormalisation the surviving weights would each be scaled by
    # 1/(1-w), which is the defect v3 carries and this model removes.
    others = [f for f in v9r.FACTORS if f != "ret_kurt_126"]
    ow = np.array([v9r.WEIGHTS[f] for f in others]) / sum(v9r.WEIGHTS.values())
    ranks = part.loc[lost, [f + "_r" for f in others]].to_numpy()
    expected = (ranks * ow).sum(axis=1)
    assert np.allclose(part.loc[lost, "score"].to_numpy(), expected, atol=1e-12)

    renormalised = (ranks * (ow / ow.sum())).sum(axis=1)
    assert not np.allclose(part.loc[lost, "score"].to_numpy(), renormalised, atol=1e-9), \
        "score matches the RENORMALISED blend -- coverage renormalisation is back"


def test_coverage_floor_nans_the_score_rather_than_guessing():
    idx = [f"S{i}" for i in range(40)]
    rng = np.random.default_rng(5)
    raw = pd.DataFrame({f: rng.normal(size=40) for f in v9r.FACTORS}, index=idx)
    # strip enough weight from one name to drop it under the floor
    heavy = ["ret_kurt_126", "mom_2_0", "mom_accel"]      # 0.1816+0.1380+0.1126 = 43%
    raw.loc[idx[0], heavy] = np.nan
    scored = v9r.score_frame(raw)
    assert scored.loc[idx[0], "coverage"] < v9r.COVERAGE_FLOOR
    assert np.isnan(scored.loc[idx[0], "score"])
    assert scored["score"].notna().sum() == 39


def test_weights_are_the_preregistered_ones():
    """These were frozen before the sealed window was opened. Changing them makes
    the +9.50% / t +1.87 result describe a different model."""
    assert set(v9r.FACTORS) == set(v9r.WEIGHTS)
    assert len(v9r.FACTORS) == 9
    assert sum(v9r.WEIGHTS.values()) == pytest.approx(1.0, abs=5e-4)
    assert v9r.WEIGHTS["ret_kurt_126"] == pytest.approx(0.1816)
    assert v9r.SPEC_SHA256 == (
        "d2dfba4f9a1e4ee1ed24d5cb6429307fdcf53bddb531979b25c2dbb78100f877")


def test_missing_factor_column_raises_rather_than_scoring_a_subset():
    idx = ["A", "B", "C"]
    raw = pd.DataFrame({f: [1.0, 2.0, 3.0] for f in v9r.FACTORS[:-1]}, index=idx)
    with pytest.raises(KeyError, match="mom_consist_126"):
        v9r.score_frame(raw)


def test_attribution_sums_to_the_score():
    idx = [f"S{i}" for i in range(50)]
    rng = np.random.default_rng(9)
    raw = pd.DataFrame({f: rng.normal(size=50) for f in v9r.FACTORS}, index=idx)
    scored = v9r.score_frame(raw)
    att = v9r.attribution(raw, "S7")
    assert att["CONTRIB"].sum() == pytest.approx(scored.loc["S7", "score"], abs=1e-12)


def test_reproduces_the_research_scorer_on_a_real_cross_section(live_cfg):
    """End-to-end fidelity against the research panel the sealed number came from."""
    import pathlib
    panel = pathlib.Path("work/v9r/panel_v9r.parquet")
    if not panel.exists():
        pytest.skip("research panel not present")
    P = pd.read_parquet(panel, columns=["date", "symbol"] + list(v9r.FACTORS))
    dates = np.sort(P["date"].unique())
    for d in dates[[100, 900, 1800, 2500, 3400]]:
        g = P[P["date"] == d]
        raw = g.set_index("symbol")[list(v9r.FACTORS)]
        got = v9r.score_frame(raw)["score"].to_numpy()

        # reference: the research path, computed inline
        R = np.column_stack([_reference_rank(raw[f].to_numpy("float64"))
                             for f in v9r.FACTORS])
        w = np.array([v9r.WEIGHTS[f] for f in v9r.FACTORS]); w = w / w.sum()
        pres = np.isfinite(R)
        cov = (pres * w).sum(1)
        want = np.where(cov >= v9r.COVERAGE_FLOOR,
                        (np.where(pres, R, 0.0) * w).sum(1), np.nan)
        assert np.allclose(got, want, equal_nan=True), f"drift on {pd.Timestamp(d).date()}"
