"""Live monitoring for the v3 thematic composite.

The monitor's whole job is to notice a model that has stopped being the model
that was tested. So every test here builds a panel where something IS wrong and
checks the monitor says so -- a monitor that never fires is indistinguishable
from one that is broken, and both look healthy.

The last test is the one that matters most: the monitor must not DISABLE
anything. A breaker that quietly drops a factor changes the running model
without a decision being taken.
"""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal import v3_monitor as mon
from prosignal.features import v3


def _panel(n_dates=80, n_syms=90, seed=7, invert=None, flatten=None):
    """A scored panel: per-date factor ranks, theme sub-scores, contributions
    and a forward label the composite genuinely predicts.

    `invert` names factors whose oriented rank is flipped against the label.
    `flatten` names themes whose sub-score carries no information at all.
    """
    rng = np.random.default_rng(seed)
    invert, flatten = set(invert or ()), set(flatten or ())
    dates = pd.bdate_range("2025-01-01", periods=n_dates)
    syms = [f"S{i:03d}" for i in range(n_syms)]
    rows = []
    for d in dates:
        truth = rng.normal(size=n_syms)                 # the thing that pays
        y = truth + rng.normal(scale=1.5, size=n_syms)  # the realised label
        rec = {"date": d, "symbol": syms, "fwd": y}
        contribs = {}
        for tname, th in v3.THEMES.items():
            if tname in flatten:
                sub = rng.normal(size=n_syms)
            else:
                sub = truth + rng.normal(scale=1.0, size=n_syms)
            sub = (pd.Series(sub).rank(pct=True).to_numpy() - 0.5) * 2.0
            rec[tname + "_sub"] = sub
            contribs[tname] = sub * th.weight
            for f, sg in th.signs.items():
                base = sub + rng.normal(scale=0.8, size=n_syms)
                if f in invert:
                    base = -base
                rec[f + "_r"] = ((pd.Series(base).rank(pct=True).to_numpy() - 0.5)
                                 * 2.0) * sg   # stored RAW: the sign is re-applied
                                               # by the monitor, as in production
        for tname, c in contribs.items():
            rec[tname + "_contrib"] = c
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------ per-factor IC
def test_a_healthy_factor_reads_positive_and_is_not_flagged():
    p = _panel()
    ic = mon.rolling_factor_ic(p, "fwd")
    assert len(ic) == p["date"].nunique()
    assert ic["mom_12_6"].mean() > 0
    health = {h.name: h for h in mon.review_factors(ic)}
    assert set(health) == set(v3.FACTOR_THEME)
    assert not any(h.inverted for h in health.values())
    assert health["mom_12_6"].theme == "momentum"


def test_an_inverted_factor_is_caught():
    """The failure this exists for: a factor still computing, still weighted,
    now pointing the wrong way. Aggregate composite IC will not say which."""
    p = _panel(invert={"ulcer_120"})
    ic = mon.rolling_factor_ic(p, "fwd")
    health = {h.name: h for h in mon.review_factors(ic)}
    assert health["ulcer_120"].inverted
    assert health["ulcer_120"].ic_t < -mon.IC_ALERT_T
    assert "check before acting" in health["ulcer_120"].note
    others = [h for n, h in health.items() if n != "ulcer_120"]
    assert not any(h.inverted for h in others), "flagged a healthy factor too"


def test_a_short_history_gets_no_verdict_rather_than_a_wrong_one():
    p = _panel(n_dates=mon.MIN_PERIODS - 5, invert={"ulcer_120"})
    ic = mon.rolling_factor_ic(p, "fwd")
    health = {h.name: h for h in mon.review_factors(ic)}
    assert not health["ulcer_120"].inverted, "verdict on too little data"
    assert "no verdict" in health["ulcer_120"].note
    assert health["ulcer_120"].ic_t is None


def test_a_factor_missing_from_the_panel_is_reported_not_skipped():
    p = _panel().drop(columns=["prox_52w_r"])
    ic = mon.rolling_factor_ic(p, "fwd")
    health = {h.name: h for h in mon.review_factors(ic)}
    assert health["prox_52w"].n_periods == 0
    assert "not present in the panel" in health["prox_52w"].note


# ------------------------------------------------------------- per-theme IC
def test_theme_ic_is_reported_for_every_shipped_theme():
    p = _panel()
    ic = mon.rolling_theme_ic(p, "fwd")
    assert set(v3.THEMES).issubset(ic.columns)
    health = {h.name: h for h in mon.review_themes(ic, p)}
    assert set(health) == set(v3.THEMES)
    assert all(h.ic_mean > 0 for h in health.values())
    assert not any(h.inverted for h in health.values())
    for name, h in health.items():
        assert h.weight == v3.THEMES[name].weight


def test_an_inverted_theme_is_caught_even_while_the_composite_holds_up():
    """Four themes right and one inverted still leaves the composite positive.
    The per-theme IC is the only place the inversion is visible."""
    p = _panel(seed=3)
    p["reversal_sub"] = -p["reversal_sub"]
    ic = mon.rolling_theme_ic(p, "fwd")
    health = {h.name: h for h in mon.review_themes(ic, p)}
    assert health["reversal"].inverted
    assert "inverted" in health["reversal"].note
    assert not health["momentum"].inverted


# ------------------------------------------------------------- dominance
def test_a_theme_running_the_book_is_flagged_despite_its_capped_weight():
    """The cap constrains the coefficient, not the influence. If four themes go
    flat the fifth runs the ranking at its declared 40%, and nothing in the
    config would say so."""
    p = _panel(flatten={"quality", "ownership", "risk", "reversal"})
    for t in ("quality", "ownership", "risk", "reversal"):
        p[t + "_contrib"] = 0.0                     # gone flat, not gone away
    share = mon.theme_influence_share(p)
    assert share["momentum"] > mon.DOMINANCE_ALERT
    assert sum(share.values()) == pytest.approx(1.0)
    health = {h.name: h for h in mon.review_themes(
        mon.rolling_theme_ic(p, "fwd"), p)}
    assert health["momentum"].dominating
    assert "the cap constrains the coefficient" in health["momentum"].note
    assert health["momentum"].influence_share > mon.DOMINANCE_ALERT
    assert not any(h.dominating for n, h in health.items() if n != "momentum")


def test_a_balanced_book_is_not_flagged_for_dominance():
    p = _panel()
    share = mon.theme_influence_share(p)
    assert max(share.values()) <= mon.DOMINANCE_ALERT
    health = mon.review_themes(mon.rolling_theme_ic(p, "fwd"), p)
    assert not any(h.dominating for h in health)


def test_the_healthy_design_point_reads_back_its_own_declared_weights():
    """The regression that produced this test: influence was measured as a
    VARIANCE share, which is quadratic in the weight, so momentum read 62%
    against a declared 40% and tripped a 55% alarm on a completely healthy
    book -- every day, from the first run. A monitor that always fires is a
    monitor that gets turned off. On the dispersion scale each theme reads its
    own weight when the sub-scores are equally dispersed, which is the only
    thing that makes 40% and 55% comparable numbers."""
    p = _panel()
    share = mon.theme_influence_share(p)
    for t, th in v3.THEMES.items():
        assert share[t] == pytest.approx(th.weight, abs=0.02)
    assert share["momentum"] == pytest.approx(0.40, abs=0.02)


def test_a_small_theme_over_running_is_flagged_below_the_absolute_alarm():
    """quality ships at 19%. At 38% it carries twice the ranking it was given
    and is nowhere near an absolute 55%, so only the relative rule catches it."""
    p = _panel()
    # three times the cross-sectional spread it was given, not a level shift
    p["quality_contrib"] = p["quality_sub"] * v3.THEMES["quality"].weight * 3.1
    share = mon.theme_influence_share(p)
    assert share["quality"] < mon.DOMINANCE_ALERT, "not testing the relative rule"
    assert share["quality"] > v3.THEMES["quality"].weight + mon.DOMINANCE_EXCESS
    health = {h.name: h for h in mon.review_themes(
        mon.rolling_theme_ic(p, "fwd"), p)}
    assert health["quality"].dominating
    assert "declared weight of 19%" in health["quality"].note


def test_variance_share_measures_influence_not_declared_weight():
    """momentum ships at 40% and quality at 19%; if quality's contribution is
    the only one moving, quality is what the ranking is."""
    p = _panel()
    for t in v3.THEMES:
        p[t + "_contrib"] = 0.0
    p["quality_contrib"] = p["quality_sub"]
    share = mon.theme_influence_share(p)
    assert share["quality"] == pytest.approx(1.0)
    assert v3.THEMES["quality"].weight < v3.THEMES["momentum"].weight


def test_variance_share_is_empty_rather_than_wrong_without_contributions():
    p = _panel().drop(columns=[t + "_contrib" for t in v3.THEMES])
    assert mon.theme_influence_share(p) == {}
    health = mon.review_themes(mon.rolling_theme_ic(p, "fwd"), p)
    assert all(h.influence_share is None for h in health)
    assert not any(h.dominating for h in health)


# ------------------------------------------------------------- drawdown
def test_the_drawdown_flag_fires_past_the_threshold():
    eq = list(np.linspace(1.0, 1.5, 50)) + list(np.linspace(1.5, 1.05, 30))
    dates = list(pd.bdate_range("2025-01-01", periods=len(eq)).date)
    f = mon.review_drawdown(eq, dates)
    assert f.flagged and f.drawdown == pytest.approx(1.05 / 1.5 - 1.0)
    assert f.peak_date < f.trough_date
    assert "Nothing has been disabled" in f.note


def test_a_shallow_drawdown_does_not_fire():
    eq = list(np.linspace(1.0, 1.5, 50)) + list(np.linspace(1.5, 1.40, 10))
    f = mon.review_drawdown(eq)
    assert not f.flagged and f.drawdown > mon.DRAWDOWN_FLAG
    assert "inside the" in f.note


def test_recovery_to_a_new_peak_reads_zero_not_the_worst_point():
    """The flag is on the CURRENT drawdown. A book that fell 30% and recovered
    is not in a drawdown, and reporting the trough would flag it forever."""
    eq = [1.0, 1.5, 1.0, 1.6]
    f = mon.review_drawdown(eq)
    assert f.drawdown == pytest.approx(0.0)
    assert not f.flagged


def test_an_empty_curve_says_so_instead_of_dividing_by_zero():
    f = mon.review_drawdown([])
    assert not f.flagged and not np.isfinite(f.drawdown)
    assert "no equity curve" in f.note
    assert f.to_dict()["peak_date"] is None


# ------------------------------------------------------------- the contract
def test_the_monitor_changes_no_state():
    """It flags. It never disables. If reviewing a panel could mutate THEMES,
    the deployed model would silently stop being the model the sealed holdouts
    describe -- which is the exact failure the flag-don't-disable rule exists
    to prevent."""
    before_themes = copy.deepcopy(v3.THEMES)
    before_factors = copy.deepcopy(v3.FACTOR_THEME)
    p = _panel(invert={"ulcer_120", "mom_12_6"}, flatten={"quality", "risk"})
    snapshot = p.copy(deep=True)

    mon.review_factors(mon.rolling_factor_ic(p, "fwd"))
    mon.review_themes(mon.rolling_theme_ic(p, "fwd"), p)
    mon.theme_influence_share(p)
    mon.review_drawdown([1.0, 1.4, 0.9])

    assert v3.THEMES == before_themes, "the monitor rewrote the shipped config"
    assert v3.FACTOR_THEME == before_factors
    assert {t: th.weight for t, th in v3.THEMES.items()} == \
           {t: th.weight for t, th in before_themes.items()}
    pd.testing.assert_frame_equal(p, snapshot)


def test_every_flag_is_serialisable_for_the_run_record():
    """A flag nobody can persist is a flag nobody sees the next morning."""
    p = _panel(invert={"ulcer_120"})
    f = mon.review_factors(mon.rolling_factor_ic(p, "fwd"))[0].to_dict()
    t = mon.review_themes(mon.rolling_theme_ic(p, "fwd"), p)[0].to_dict()
    d = mon.review_drawdown([1.0, 0.7], [dt.date(2026, 1, 1),
                                         dt.date(2026, 2, 1)]).to_dict()
    import json
    for payload in (f, t, d):
        json.loads(json.dumps(payload, default=float))
    assert d["peak_date"] == "2026-01-01" and d["trough_date"] == "2026-02-01"


# ------------------------------------------- the half that runs every day
def _one_date(seed=5, flat=(), scale=None):
    """One scored cross-section, as stage 4 produces it: no forward labels."""
    rng = np.random.default_rng(seed)
    n = 200
    idx = [f"S{i:03d}" for i in range(n)]
    out = pd.DataFrame(index=idx)
    for t, th in v3.THEMES.items():
        sub = (pd.Series(rng.normal(size=n)).rank(pct=True).to_numpy() - 0.5) * 2
        out[t + "_sub"] = sub
        w = 0.0 if t in flat else th.weight * (scale or {}).get(t, 1.0)
        out[t + "_contrib"] = sub * w
    return out


def test_the_daily_check_needs_no_forward_outcome():
    """Rolling IC cannot say anything about today -- the outcome is 21 sessions
    away. Dominance can: it is a property of the scores as they stand. That is
    the whole reason this half runs on every run."""
    scored = _one_date()
    assert not [c for c in scored.columns if c.startswith("y")]
    share = mon.cross_section_influence(scored)
    assert set(share) == set(v3.THEMES)
    assert sum(share.values()) == pytest.approx(1.0)
    for t, th in v3.THEMES.items():
        assert share[t] == pytest.approx(th.weight, abs=0.03)


def test_a_healthy_cross_section_gets_a_line_and_no_flag():
    notes = mon.review_cross_section(_one_date())
    assert len(notes) == 1 and notes[0].startswith("Theme influence")
    assert "declared 40%" in notes[0]
    assert not [n for n in notes if n.startswith("FLAG")]


def test_the_daily_check_catches_the_takeover_the_brief_asked_about():
    """Four themes go flat, momentum keeps its 40% weight, and the ranking is
    now momentum alone. Nothing in the config would say so."""
    scored = _one_date(flat={"quality", "ownership", "risk", "reversal"})
    share = mon.cross_section_influence(scored)
    assert share["momentum"] == pytest.approx(1.0)
    flags = [n for n in mon.review_cross_section(scored) if n.startswith("FLAG")]
    assert len(flags) == 1 and "momentum" in flags[0]
    assert "Nothing has been changed" in flags[0]
    assert "still" in flags[0] and "PAYING" in flags[0], \
        "the flag must say what it cannot know from one date"


def test_a_small_theme_over_running_is_caught_on_one_date_too():
    scored = _one_date(scale={"quality": 3.1})
    flags = [n for n in mon.review_cross_section(scored) if n.startswith("FLAG")]
    assert len(flags) == 1 and "quality" in flags[0]
    assert "declared weight of 19%" in flags[0]


def test_too_few_names_is_no_measurement_rather_than_a_wrong_one():
    scored = _one_date().head(mon.MIN_NAMES_FOR_INFLUENCE - 1)
    assert mon.cross_section_influence(scored) == {}
    assert mon.review_cross_section(scored) == []


def test_a_frame_without_contributions_says_nothing_instead_of_guessing():
    scored = _one_date().drop(columns=[t + "_contrib" for t in v3.THEMES])
    assert mon.cross_section_influence(scored) == {}
    assert mon.review_cross_section(scored) == []


def test_the_daily_check_changes_no_state_either():
    before = copy.deepcopy(v3.THEMES)
    scored = _one_date(flat={"quality", "risk"})
    snap = scored.copy(deep=True)
    mon.review_cross_section(scored)
    assert v3.THEMES == before
    pd.testing.assert_frame_equal(scored, snap)


# ------------------------------------- the channel the flags travel through
def test_the_scoring_notes_reach_the_run_and_the_persisted_payload(runnable_cfg):
    """THE BUG THIS PINS: stage 4 wrote `CoreScores.notes` on every run since
    the v2 deploy -- which scorer ordered the book, the sealed-holdout numbers
    behind it, why the regime multipliers were withheld, how many names cleared
    the absolute floor -- and nothing read the field. Not the card, not the
    ledger, not the run-detail payload the screen renders.

    So the daily theme-dominance flag would have been raised into a field
    nobody reads, which is worse than not raising it: the engine would look
    monitored. A flag needs a channel, and this test is the channel."""
    from prosignal.pipeline import run_analysis
    from prosignal import rundetail

    run = run_analysis(runnable_cfg)
    assert run.scoring_notes, "stage 4 said nothing about how it ranked"

    joined = " ".join(run.scoring_notes)
    assert "v3 composite" in joined, "the run does not record which model ranked it"
    assert "Theme influence on this cross-section" in joined, \
        "the dominance check did not run on this run"

    payload = rundetail.shape(run)
    assert payload["scoring_notes"] == run.scoring_notes
    import json
    json.loads(json.dumps(payload, default=str))


def test_the_daily_influence_line_reports_every_shipped_theme(runnable_cfg):
    from prosignal.pipeline import run_analysis

    line = next(n for n in run_analysis(runnable_cfg).scoring_notes
                if n.startswith("Theme influence"))
    for t, th in v3.THEMES.items():
        assert t in line
        assert f"declared {th.weight:.0%}" in line
