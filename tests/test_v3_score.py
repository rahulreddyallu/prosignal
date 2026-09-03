"""The v3 thematic composite.

Every test here exists because a plausible alternative was written first and was
wrong in a way that produced sensible-looking numbers: a theme oriented at the
wrong horizon, a weight cap that ignored coverage, a floor that could never
fire, a factor list picked by exclusion that swept in forward-looking columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features import v3, v3_factors


def _panel(n_days=400, n_syms=60, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_syms)]
    shocks = rng.normal(0.0004, 0.015, size=(n_days, n_syms))
    close = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, (n_days, n_syms)))
    vwap = close * (1 + rng.normal(0, 0.002, (n_days, n_syms)))
    turnover = pd.DataFrame(rng.lognormal(18, 0.6, (n_days, n_syms)), index=idx, columns=cols)
    deliv = pd.DataFrame(rng.uniform(20, 80, (n_days, n_syms)), index=idx, columns=cols)
    bench = close.mean(axis=1).pct_change(fill_method=None)
    return close, open_, vwap, turnover, deliv, bench


# ---------------------------------------------------------------- the shape
def test_the_shipped_themes_and_weights_are_the_deployed_configuration():
    """If any of this changes, the sealed-holdout numbers in CHANGELOG.md stop
    describing the running model and the deploy has to be re-earned."""
    assert set(v3.THEMES) == {"momentum", "quality", "ownership", "risk", "reversal"}
    assert sum(t.weight for t in v3.THEMES.values()) == pytest.approx(1.0)
    assert max(t.weight for t in v3.THEMES.values()) <= 0.40 + 1e-9
    assert min(t.weight for t in v3.THEMES.values()) >= 0.06 - 1e-9
    assert len(v3.ALL_FACTORS) == 22
    # Both were frozen rounded (weight to 5dp, coverage to 4dp), so the
    # comparison is only meaningful to the coarser of the two.
    assert v3.THEMES["quality"].weight <= v3.THEMES["quality"].coverage + 5e-5
    for name, th in v3.THEMES.items():
        assert th.weight <= th.coverage + 5e-5, f"{name} outruns its coverage"


def test_every_theme_is_oriented_at_its_own_horizon():
    """One label for all five made the reversal sub-score anti-predictive."""
    h = {t: th.horizon for t, th in v3.THEMES.items()}
    assert h["momentum"] == 42 and h["reversal"] == 10 and h["ownership"] == 10
    assert h["risk"] == 21 and h["quality"] == 21
    assert len(set(h.values())) > 1, "a single horizon is the bug this prevents"


def test_no_factor_belongs_to_two_themes():
    seen = [f for t in v3.THEMES.values() for f in t.names]
    assert len(seen) == len(set(seen))
    assert set(seen) == set(v3.ALL_FACTORS)


# ---------------------------------------------------------------- the weights
def test_the_cap_stops_one_theme_swamping_the_rest():
    w = v3.cap_weights({"a": 0.9, "b": 0.05, "c": 0.05}, cap=0.40, floor=0.0)
    assert w["a"] == pytest.approx(0.40)
    assert sum(w.values()) == pytest.approx(1.0)


def test_a_theme_is_also_capped_at_its_coverage():
    """Weights renormalise over the themes a NAME has, so a theme carried at 40%
    while a fifth of names have it ranks two populations by different models."""
    w = v3.cap_weights({"a": 0.25, "b": 0.50, "c": 0.15, "d": 0.10},
                       cap=0.40, floor=0.0,
                       coverage={"a": 1.0, "b": 0.19, "c": 1.0, "d": 1.0})
    assert w["b"] == pytest.approx(0.19)
    assert max(w.values()) <= 0.40 + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


def test_caps_that_cannot_all_be_met_degrade_to_their_own_proportions():
    """Two themes, a 40% cap and a 19% coverage cap: the caps sum to 0.59 and
    the weights must still sum to 1, so every cap CANNOT hold. The documented
    fallback is to allocate in proportion to the caps -- the low-coverage theme
    still ends up the smaller of the two, which is the property that matters.
    Pinned here so the infeasible branch is a decision and not an accident."""
    w = v3.cap_weights({"a": 0.5, "b": 0.5}, cap=0.40, floor=0.0,
                       coverage={"a": 1.0, "b": 0.19})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["a"] / w["b"] == pytest.approx(0.40 / 0.19)
    assert w["b"] < w["a"]


def test_the_floor_keeps_a_theme_that_one_window_dislikes():
    w = v3.cap_weights({"a": 1.0, "b": 0.0, "c": 0.0}, cap=0.40, floor=0.06)
    assert w["b"] >= 0.06 - 1e-9 and w["c"] >= 0.06 - 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- lookahead
def test_no_factor_reads_a_session_after_the_decision_row():
    close, open_, vwap, turnover, deliv, bench = _panel()
    cut = 380
    sl = slice(0, cut)
    a = v3_factors.factor_frame(close.iloc[sl], open_.iloc[sl], vwap.iloc[sl],
                                turnover.iloc[sl], deliv.iloc[sl], bench.iloc[sl])
    future = close.copy()
    future.iloc[cut:] *= 4.0
    b = v3_factors.factor_frame(future.iloc[sl], open_.iloc[sl], vwap.iloc[sl],
                                turnover.iloc[sl], deliv.iloc[sl], bench.iloc[sl])
    pd.testing.assert_frame_equal(a, b)


def test_the_momentum_skip_windows_end_21_sessions_back():
    close, open_, vwap, turnover, deliv, bench = _panel()
    base = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    moved = close.copy()
    moved.iloc[-21:] *= 1.6
    after = v3_factors.factor_frame(moved, open_, vwap, turnover, deliv, bench)
    for name in ("mom_consist_126", "prox_52w", "mom_12_6"):
        pd.testing.assert_series_equal(base[name], after[name], check_names=False)
    # and the non-skipping ones must move, or the test above proves nothing
    assert not np.allclose(base["prox_52w_now"].dropna(),
                           after["prox_52w_now"].reindex(base.index).dropna())


# ---------------------------------------------------------------- the composite
def test_theme_contributions_sum_to_the_score():
    close, open_, vwap, turnover, deliv, bench = _panel()
    raw = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    scored = v3.score_frame(raw, sectors=None)
    contrib = scored[[t + "_contrib" for t in v3.THEMES]].sum(axis=1, min_count=1)
    ok = scored["score"].notna()
    np.testing.assert_allclose(contrib[ok].to_numpy(),
                               scored.loc[ok, "score"].to_numpy(),
                               rtol=1e-9, atol=1e-12)


def test_a_name_missing_a_theme_is_scored_on_the_rest_not_pushed_to_zero():
    close, open_, vwap, turnover, deliv, bench = _panel()
    raw = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    assert raw["net_margin"].isna().all(), "no fundamentals in this fixture"
    scored = v3.score_frame(raw, sectors=None)
    assert (scored["n_themes"] == 4).all()
    assert scored["score"].notna().any()
    assert scored["score"].abs().max() > 0.5, "a missing theme must not flatten the score"


def test_a_name_on_too_few_themes_is_not_scored_at_all():
    close, open_, vwap, turnover, deliv, bench = _panel(n_days=120)
    raw = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    scored = v3.score_frame(raw, sectors=None, min_themes=5)
    assert scored["score"].isna().all()


def test_the_signs_are_applied():
    """`ulcer_120` carries sign -1: deeper drawdown must SCORE WORSE.

    The sign lives in the theme, not in the rank -- `ulcer_120_r` is the raw
    sector-neutral rank and is SUPPOSED to be highest for the deepest drawdown.
    So the sign is checked where it is applied: hold the other two risk factors
    flat, vary ulcer alone, and the risk sub-score must fall as ulcer rises."""
    th = v3.THEMES["risk"]
    assert th.signs["ulcer_120"] == -1

    idx = [f"S{i:02d}" for i in range(20)]
    ranks = pd.DataFrame(
        {"ulcer_120": np.linspace(-1.0, 1.0, len(idx)),
         "downside_vol_60": 0.0,
         "ret_kurt_126": 0.0},
        index=idx)
    sub = v3.theme_subscore(ranks, th)
    assert sub.iloc[0] > sub.iloc[-1], "deepest drawdown scored best"
    assert (np.diff(sub.to_numpy()) < 0).all(), "not monotone in ulcer"


def test_the_sign_reaches_the_per_stock_card():
    """A reader checking the theme against its parts needs the sign on the row,
    otherwise a high rank on a bad-is-high factor reads as a positive."""
    close, open_, vwap, turnover, deliv, bench = _panel()
    raw = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    scored = v3.score_frame(raw, sectors=None)
    card = v3.attribution(raw, scored, scored.index[0])
    rows = card[card.LEVEL == "factor"].set_index("FACTOR")
    for tname, th in v3.THEMES.items():
        n = len(th.names)
        for fname, sign in th.signs.items():
            w = rows.at[fname, "WEIGHT"]
            assert np.sign(w) == sign, f"{tname}/{fname} sign lost on the card"
            assert abs(w) == pytest.approx(1.0 / n)
    # Not every member of a theme points the same way, and this one surprises
    # people: acceleration screened NEGATIVE inside momentum. Pinned so it is
    # not "corrected" to +1 by someone reading the theme name alone.
    assert v3.THEMES["momentum"].signs["mom_accel"] == -1


# ---------------------------------------------------------------- the floor
def test_the_absolute_floor_can_actually_empty_the_list():
    """A floor on a cross-sectional RANK cannot fire -- somebody is top of the
    list every day. This one is measured against the stock."""
    close, open_, vwap, turnover, deliv, bench = _panel()
    raw = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    scored = v3.score_frame(raw, sectors=None)
    everyone_below = pd.Series(-0.2, index=scored.index)
    assert not v3.absolute_floor(scored, everyone_below).any()
    everyone_above = pd.Series(0.2, index=scored.index)
    passed = v3.absolute_floor(scored, everyone_above)
    assert passed.any() and not passed.all(), "the theme half must bind too"


def test_the_floor_ships_disabled_and_its_scope_is_still_entries():
    """The floor was DISABLED on measurement on 2026-09-02.

    Treatment effect on the v3 composite across 66 purged and embargoed CPCV
    folds, measured the way this config applies it (`applies_to: entries`, so it
    blocks new opens and leaves the name ranked and holdable): ATE -2.2%, 95% CI
    [-3.2%, -1.4%]. The interval sits clear of zero, so it is a removal rather
    than a re-tune.

    The SCOPE is still asserted because it is the part that was measured wrong
    twice: an earlier draft claimed -14.3% by filtering the whole population
    before ranking, which is not what this engine does. If the floor is ever
    re-enabled, it must come back as `entries`.
    """
    from prosignal.config.loader import load_config
    f = load_config().params.stage4_core_score.absolute_floor
    assert bool(f.enabled.value) is False
    assert str(f.applies_to.value) == "entries"
    assert int(f.min_positive_themes.value) == 3


# ---------------------------------------------------------------- attribution
def test_attribution_gives_the_card_theme_and_factor_levels():
    close, open_, vwap, turnover, deliv, bench = _panel()
    raw = v3_factors.factor_frame(close, open_, vwap, turnover, deliv, bench)
    scored = v3.score_frame(raw, sectors=None)
    sym = scored["score"].idxmax()
    tab = v3.attribution(raw, scored, sym)
    assert list(tab.columns) == ["FACTOR", "THEME", "VALUE", "Z", "WEIGHT",
                                "CONTRIB", "LEVEL"]
    assert set(tab["LEVEL"]) == {"theme", "factor"}
    assert (tab["LEVEL"] == "theme").sum() == len(v3.THEMES)
    assert (tab["LEVEL"] == "factor").sum() == len(v3.ALL_FACTORS)


# ---------------------------------------------------------------- instruments
def test_etfs_and_funds_are_not_ranked_as_stocks():
    """Three of the top five names on a live run were bond ETFs, and they took
    26% of the ten-name book across the 2025-26 sealed window."""
    from prosignal.data.instruments import non_equity_symbols

    master = pd.DataFrame({"symbol": ["RELIANCE", "GOLDIAM", "SKYGOLD", "PNBGILTS"]})
    syms = ["RELIANCE", "GOLDIAM", "SKYGOLD", "PNBGILTS", "GOLDBEES", "LIQUIDBEES",
            "NIFTYBEES", "SILVERETF", "BBETF0432", "SETFGOLD", "MON100"]
    drop = non_equity_symbols(syms, equity_master=master)
    assert {"GOLDBEES", "LIQUIDBEES", "NIFTYBEES", "SILVERETF", "BBETF0432",
            "SETFGOLD", "MON100"} <= drop
    assert not ({"RELIANCE", "GOLDIAM", "SKYGOLD", "PNBGILTS"} & drop), \
        "a real company in the equity master must never be removed by a name pattern"


def test_the_volatility_backstop_will_not_fire_on_a_short_sample():
    """Run on sixty sessions it flagged 199 names against the pattern's 61: a
    quiet quarter in a thin line looks exactly like a liquid fund."""
    from prosignal.data.instruments import non_equity_symbols

    idx = pd.bdate_range("2025-01-01", periods=60)
    close = pd.DataFrame({"THINCO": np.linspace(100, 101, 60)}, index=idx)
    assert non_equity_symbols(["THINCO"], equity_master=pd.DataFrame({"symbol": []}),
                              close=close) == set()


# ---------------------------------------------------------------- the card
def test_the_percentile_on_the_card_is_a_real_ordinal():
    """The theme line printed `f"{pct:.0f}th"`: right at 98, wrong at 92, 41
    and 21. It is the one line a reader uses to place a name against its
    sector, so it should not read like a typo."""
    from prosignal.stages.stage8_final_signal import _ordinal
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 41, 92, 100)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "41st",
         "92nd", "100th"]


# ---------------------------------------------------------------- the books
def test_the_live_book_mirror_matches_the_config_that_actually_trades():
    """THE MISREAD THIS PREVENTS. `v3.BOOK` describes a 12-slot book on a
    10-session rebalance. Production trades SIX positions on a 21-session
    cadence, from `config/parameters.yaml`. Nothing reads `v3.BOOK` -- so a
    reader of the shipped scorer would have taken it for the live book, and
    reasoned about turnover, concentration and cost for a book that does not
    exist. A stale mirror is worse than no mirror, so this fails when it drifts.
    """
    from prosignal.config.loader import load_config

    p = load_config().params
    val = lambda x: getattr(x, "value", x)
    live = {"slots": int(val(p.capital.max_open_positions)),
            "entry_rank": int(val(p.stage6_entry.admission.entry_rank)),
            "exit_rank": int(val(p.stage6_entry.admission.exit_rank)),
            "entry_cadence_sessions":
                int(val(p.stage6_entry.admission.entry_cadence_sessions))}
    assert v3.LIVE_BOOK == live, (
        "features/v3.py::LIVE_BOOK has drifted from parameters.yaml, which is "
        "the only thing that changes what trades. Update the mirror.")


def test_the_three_books_are_distinct_and_the_note_says_which_one_trades():
    """A number measured on one book and quoted about another is how a backtest
    becomes a claim it never made."""
    assert v3.LIVE_BOOK != v3.RESEARCH_BOOK != v3.HOLDOUT_BOOK
    assert v3.BOOK is v3.RESEARCH_BOOK, "BOOK must stay the research book"
    assert v3.LIVE_BOOK["slots"] < v3.HOLDOUT_BOOK["slots"], \
        "the live book is the more concentrated one -- that is the point"
    note = v3.BOOK_NOTE
    assert "NO BOOK DOES" in note, "the note must not imply a book was validated"
    assert "SIX positions" in note and "21-session" in note
    assert "t 0.81" in note, "the weakest holdout statistic must be named"
