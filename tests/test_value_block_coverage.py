"""Five factors that are a constant for three quarters of the universe.

`_attach_fundamentals` neutral-filled a missing filing, so every name without a
statement landed on exactly the same rank for all five value factors. On a card
they read as five independent z-scores all sitting at -0.01 — identical to two
decimals across five ratios built from five different line items, which cannot
happen by chance.

Measured on the live universe:

    statements feed covers          192 of 750 names
    value factors at exactly 0.0    74-78% of names
    coverage on the training panel  10-12%

And the gap is not random. Names WITH statements have **7.5x** the median
turnover of names without (Rs 176 cr against Rs 23 cr), so the value block was
substantially a disguised size bet: the model differentiated on which names the
feed happened to cover.

Stage 4 already states the rule — "a factor scored on a minority of names ranks
the rest by median fill, which is not a ranking" — and enforced it on the
hand-weighted composite. The fitted model, which is the one that actually ranks,
imputed instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import crossmodel as cm


def test_a_missing_filing_leaves_the_rank_absent_not_neutral():
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"] * 3),
                          "symbol": ["A", "B", "C"]})
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = pd.DataFrame({c: 100.0 for c in ["A", "B", "C"]}, index=idx)
    out = cm._attach_fundamentals(panel, None, close, None)
    for f in cm.FUNDAMENTAL_FEATURES:
        assert out[f + "_r"].isna().all(), (
            "a neutral fill here is what made the gap invisible downstream"
        )


def test_the_coverage_floor_is_the_rule_stage_4_already_states():
    assert cm.MIN_FACTOR_COVERAGE == pytest.approx(0.60)


def test_a_model_reports_the_factors_it_actually_used():
    m = cm.CrossSectionalModel(
        coef={"mom_6_1_r": 0.01, "prox_52w_r": 0.02},
        n_train=1000, train_end=pd.Timestamp("2026-01-01").date(),
    )
    assert m.features == ["mom_6_1_r", "prox_52w_r"]
    assert "ridge on 2 cross-sectional features" in m.summary()


def test_scoring_follows_the_models_feature_list_not_the_module_constant():
    """A model fitted on 12 factors must not be applied as though it had 17.
    `mu` and `sd` are positional, so a mismatch mis-standardises silently."""
    m = cm.CrossSectionalModel(
        coef={"mom_6_1_r": 1.0, "prox_52w_r": -1.0},
        n_train=1000, train_end=pd.Timestamp("2026-01-01").date(),
        features=["mom_6_1_r", "prox_52w_r"],
    )
    m.mu = np.array([0.0, 0.0])
    m.sd = np.array([1.0, 1.0])
    m.intercept = 0.0
    feats = pd.DataFrame({
        "symbol": ["A", "B"],
        "mom_6_1_r": [0.5, -0.5],
        "prox_52w_r": [-0.5, 0.5],
        # a column the model was NOT fitted on; it must be ignored, not
        # positionally consumed
        "earnings_yield_r": [9.9, -9.9],
    })
    out = cm.score_with(m, feats)
    assert out.loc["A"] > out.loc["B"]
    contrib = cm.contributions(m, feats)
    assert list(contrib.columns) == ["mom_6_1", "prox_52w"]
    assert "earnings_yield" not in contrib.columns


def test_a_cached_model_with_fewer_families_is_accepted(tmp_path):
    """A stored model may legitimately carry fewer families than the code
    declares -- one whose members the feed could not serve was never built."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["mom_f", "lottery_f"],
        "coef": {"mom_f": 0.01, "lottery_f": 0.02},
        "mu": [0.0, 0.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    m = cm.load_cached(path, dt.date(2026, 1, 6), 21)
    assert m is not None
    assert m.features == ["mom_f", "lottery_f"]


def test_a_cache_from_before_the_family_fit_forces_a_refit(tmp_path):
    """Individual-factor coefficients describe a different model. Reusing them
    against a family frame would standardise against the wrong columns."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["mom_6_1_r", "prox_52w_r"],
        "coef": {"mom_6_1_r": 0.01, "prox_52w_r": 0.02},
        "mu": [0.0, 0.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    assert cm.load_cached(path, dt.date(2026, 1, 6), 21) is None


def test_a_cached_model_naming_an_unknown_factor_forces_a_refit(tmp_path):
    """Fewer is fine. A factor the code no longer knows means the definitions
    moved, and the stored coefficients describe a different model."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["mom_f", "a_family_we_deleted_f"],
        "coef": {"mom_f": 0.01, "a_family_we_deleted_f": 0.02},
        "mu": [0.0, 0.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    assert cm.load_cached(path, dt.date(2026, 1, 6), 21) is None


def test_the_stored_feature_order_is_preserved(tmp_path):
    """`mu` and `sd` are positional. A dict whose iteration order changed would
    standardise every factor against the wrong column."""
    import datetime as dt
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "fitted_for": "2026-01-05", "train_end": "2025-10-01", "n_train": 5000,
        "features": ["lottery_f", "mom_f"],            # deliberately not sorted
        "coef": {"mom_f": 0.01, "lottery_f": 0.02},
        "mu": [1.0, 2.0], "sd": [1.0, 1.0], "intercept": 0.0,
    }))
    m = cm.load_cached(path, dt.date(2026, 1, 6), 21)
    assert m.features == ["lottery_f", "mom_f"]


# --------------------------------------------------------- redundancy
def test_the_redundancy_check_measures_what_carries_the_coefficients():
    """It ran on `frame` -- the hand-weighted composite's factor block -- so the
    columns that actually rank the universe were never checked against each
    other. It then ran on the individual `_r` factors, which is the opposite
    error: those are collinear BY DESIGN, and averaging them is what the
    families are for. Measured live on 2026-08-25 they breach five times --
    mom_6_1/resid_mom +0.72, downside_vol/idio_vol +0.69, mom_6_1/prox_52w
    +0.64 -- and every pair sits inside one family.

    The question is whether the things given INDEPENDENT coefficients are
    independent, so the check measures the fitted columns and reports the
    within-family structure as context.
    """
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4.run)
    assert "model_block" in src and "member_block" in src
    assert 'getattr(model, "features", None)' in src, (
        "the fitted block must come from the model's own feature list, not "
        "from a filter over column suffixes that cannot tell a fitted column "
        "from a diagnostic one"
    )
    assert "members=member_block" in src, (
        "the within-family overlap must still reach the report -- it is what "
        "justifies the aggregation"
    )


def test_an_all_empty_column_cannot_silence_the_whole_redundancy_report():
    """`today_features` attaches every fundamental column whether or not the
    fit could build it. A listwise dropna then emptied the frame on one absent
    column, and the report said 'correlation not measurable' with twelve
    fully-populated factors sitting in it, on every run."""
    import numpy as np
    import pandas as pd

    from prosignal.indicators import spearman_pairs

    frame = pd.DataFrame({
        "a": np.linspace(0, 1, 50),
        "b": np.linspace(1, 0, 50),
        "all_empty": [np.nan] * 50,
    })
    pairs = spearman_pairs(frame)
    assert pairs, "one empty column silenced the whole report"
    assert abs(pairs["a|b"] + 1.0) < 1e-9


def test_a_breach_names_the_pair_rather_than_only_counting_them():
    import inspect

    from prosignal.stages import stage4_core_score as s4

    src = inspect.getsource(s4.run)
    block = src[src.index("if redundancy.breaches:"):]
    assert "pairs" in block and "{a}/{b}" in block, (
        "a count alone does not say which factors are the same bet"
    )


def test_the_panel_says_score_and_contributions_are_different_units():
    """A reader adding up contributions of ~0.12 and seeing a score of 0.898 has
    no way to connect them. The chain is: the model's raw prediction (which the
    contributions DO sum to), ranked across the day's eligible universe, mapped
    onto [0,1]. So 0.898 is the 89.8th percentile of today's names."""
    import pathlib

    page = pathlib.Path("src/prosignal/static/index.html").read_text()
    # The caption moved and shortened with the rest of the panel's prose; what
    # it has to keep saying is that Score is a POSITION, not a probability.
    assert "not a probability" in page
    assert "eligible universe" in page
    # The old caption spelled out that contributions sum to the prediction and
    # not to Score. The contributions ARE the score now -- v3's theme
    # contributions sum to the composite by construction -- so the sentence
    # that reconciled two units describes a mismatch that no longer exists.
    assert "sums to the score" in page.lower()


def test_the_score_is_a_percentile_of_the_days_universe():
    """The property itself, not the label: the transform is order-preserving and
    lands the best name at 1.0 and the worst at 0.0."""
    import pandas as pd

    from prosignal.indicators.crosssection import rank_to_unit_interval

    raw = pd.Series({"worst": -0.05, "mid": 0.01, "best": 0.12})
    out = rank_to_unit_interval(raw)
    assert out["best"] == pytest.approx(1.0)
    assert out["worst"] == pytest.approx(0.0)
    assert out["mid"] == pytest.approx(0.5)
    # and it does NOT preserve the magnitudes the contributions carried
    assert out["best"] != pytest.approx(0.12)


# ----------------------------------------------- two failures, two tests
def test_coverage_is_two_different_questions():
    """A single number over the whole panel conflated them and reported the
    wrong one. After the fundamentals ingest took symbol coverage from 26% to
    100%, the value factors read 40% panel coverage and were dropped for
    "ranking too few names" -- when they rank **73%** of the universe on every
    date they exist. What is actually wrong is that they exist on only 35 of 88
    dates."""
    assert cm.MIN_FACTOR_COVERAGE == pytest.approx(0.60)
    assert cm.MIN_FACTOR_DATE_SPAN == pytest.approx(0.60)


def test_a_factor_absent_early_is_dropped_for_span_not_for_thinness():
    """The distinction matters because the remedies differ: thinness needs more
    names, absence needs more history, and the log has to say which."""
    import inspect

    src = inspect.getsource(cm.prepare_features)
    assert "absent for too much of the panel" in src
    assert "ranks too few names on the dates it exists" in src
    # Span is tested FIRST, so the more fundamental failure is the one reported.
    assert (src.index("span < MIN_FACTOR_DATE_SPAN")
            < src.index("within < MIN_FACTOR_COVERAGE"))


def test_within_date_coverage_ignores_dates_the_factor_never_existed_on():
    """Averaging zeros from an era the feed does not reach drags a perfectly
    rankable factor under the floor."""
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 4 + ["2024-01-01"] * 4),
        "f_r": [np.nan] * 4 + [1.0, 2.0, 3.0, np.nan],
    })
    per_date = panel.groupby("date")["f_r"].apply(lambda x: float(x.notna().mean()))
    live = per_date[per_date > 0]
    assert len(live) == 1, "one date has data"
    assert float(live.median()) == pytest.approx(0.75), "and covers 75% of it"
    assert float(panel["f_r"].notna().mean()) == pytest.approx(0.375), (
        "while the flat average reads 37.5% and would drop it"
    )


def test_the_fit_and_the_research_paths_share_one_definition():
    """CPCV passed every raw FEATURE_COLUMN straight to `dropna`, which deleted
    every row without a fundamental and cut a 70-date panel to 17 -- too few to
    build ten CPCV groups, so the run did not merely validate the wrong model,
    it could not complete at all. One implementation, used by both."""
    import inspect

    from prosignal import cli

    from prosignal.validation import research_panel

    # The shared definition may be reached DIRECTLY or through the shared panel
    # builder. What must never happen is a research command assembling its own.
    shared = inspect.getsource(research_panel.build_research_panel)
    assert "prepare_features" in shared

    assert "prepare_features" in inspect.getsource(cm.fit_predict)
    for fn in (cli.cmd_research_cpcv, cli.cmd_research_portfolio,
               cli.cmd_research_factors, cli.cmd_research_estimator):
        src = inspect.getsource(fn)
        assert ("prepare_features" in src
                or "build_research_panel(" in src), (
            f"{fn.__name__} builds its own feature set and would validate a "
            f"different model than the one that runs"
        )
        assert "build_panel(" not in src, (
            f"{fn.__name__} calls build_panel directly, which bypasses the "
            f"shared builder and can silently drop the barrier label"
        )
