"""The training population and the admission population must be the same set.

`resolve_exits` gives a name already below its thesis-invalidation level a NaN
label, and `build_panel` drops non-finite labels. So that name is absent from
the training panel -- AND from every ranking any validation derives from that
panel: CPCV, purged walk-forward, portfolio_sim, research spread.

The live engine builds no label. `today_features` applies no such filter, and
once Stage 6 stopped gating on price structure nothing else did either. The
result was two different populations wearing one set of performance numbers.

Measured on the eligible universe: 21.8% of the selection period and 26.9% of
the holdout sits below the level. The harness could not see it -- restricting
an already-restricted panel finds 1.8% left to remove, which is what E1
reported before the cause was found.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.exits import (
    ExitRules, atr_panel, ma_panel, invalidation_level, resolve_exits,
    tradeable_at_entry,
)

RULES = ExitRules(horizon=20, invalidation_ma_sessions=50,
                  invalidation_buffer_atr=1.5)


def _series(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    up = pd.Series(np.linspace(100, 160, n), index=idx)
    down = pd.Series(np.r_[np.linspace(100, 160, n // 2),
                           np.linspace(160, 90, n - n // 2)], index=idx)
    close = pd.DataFrame({"UP": up, "DOWN": down})
    return close, close * 1.01, close * 0.99


# ------------------------------------------------- one predicate, both sides
def test_the_label_excludes_exactly_what_the_predicate_refuses():
    """THE guard. If these two ever disagree, the engine is trading a
    population its coefficients were not estimated on."""
    close, high, low = _series()
    atr = atr_panel(high, low, close, RULES.atr_period_sessions, RULES.atr_method)
    ma = ma_panel(close, RULES.invalidation_ma_sessions)
    i = 150

    predicate = tradeable_at_entry(close.iloc[i], ma.iloc[i], atr.iloc[i], RULES)
    labelled = resolve_exits(close, i, RULES, high=high, low=low,
                             open_=close, atr=atr, ma=ma)["ret"].notna()

    for sym in close.columns:
        assert bool(predicate[sym]) == bool(labelled[sym]), (
            f"{sym}: the predicate says tradeable={predicate[sym]} but the "
            f"label says usable={labelled[sym]}"
        )


def test_a_name_below_its_invalidation_level_gets_no_label():
    close, high, low = _series()
    atr = atr_panel(high, low, close, RULES.atr_period_sessions, RULES.atr_method)
    ma = ma_panel(close, RULES.invalidation_ma_sessions)
    i = 150
    lvl = invalidation_level(ma.iloc[i], atr.iloc[i], RULES)
    assert close["DOWN"].iloc[i] < lvl["DOWN"], "fixture is not below its level"
    out = resolve_exits(close, i, RULES, high=high, low=low, open_=close,
                        atr=atr, ma=ma)
    assert np.isnan(out["ret"]["DOWN"])
    assert np.isfinite(out["ret"]["UP"])


def test_an_unknown_level_is_not_a_cleared_one():
    """NaN in, refusal out. Treating 'cannot check' as 'passed' is the failure
    the eligibility contract exists to prevent."""
    assert not bool(tradeable_at_entry(100.0, np.nan, 2.0, RULES))
    assert not bool(tradeable_at_entry(100.0, 95.0, np.nan, RULES))
    assert not bool(tradeable_at_entry(np.nan, 95.0, 2.0, RULES))


def test_the_predicate_is_a_level_comparison_not_a_trend_test():
    """Just above the level passes; just below fails. Nothing else enters."""
    ma, atr = 100.0, 4.0                      # level = 100 - 1.5*4 = 94
    assert bool(tradeable_at_entry(94.01, ma, atr, RULES))
    assert not bool(tradeable_at_entry(93.99, ma, atr, RULES))


# ------------------------------------------------------ the live admission
def test_stage_6_refuses_an_entry_below_the_invalidation_level():
    import inspect

    from prosignal.stages import stage6_entry as s6

    src = inspect.getsource(s6._evaluate)
    assert "require_above_invalidation" in src
    assert "_below_invalidation" in src
    helper = inspect.getsource(s6._below_invalidation)
    assert "tradeable_at_entry" in helper, (
        "the live admission must read the SAME predicate the label uses, not "
        "a second implementation of the same arithmetic"
    )


def test_a_held_position_is_exempt_from_the_entry_rule():
    """An entry constraint must never close an open position -- the exit band
    and Stage 7's hierarchy own that decision."""
    import inspect

    from prosignal.stages import stage6_entry as s6

    src = inspect.getsource(s6._evaluate)
    block = src[src.index("if not is_held and bv("):]
    block = block[:block.index("# ---- admission: rank")]
    assert "not is_held" in block


def test_the_rule_can_be_switched_off_but_ships_on():
    from prosignal.config.loader import load_config

    a = load_config().params.stage6_entry.admission
    assert a.require_above_invalidation.value is True, (
        "shipping this off means trading a population no validation in this "
        "repository has measured"
    )


# ------------------------------------- the research panel's own population
def test_asking_for_an_unfiltered_panel_is_not_the_same_as_asking_for_nothing():
    """`exit_rules or default` cannot tell "not provided" from "explicitly no
    exit geometry". Passing None -- which is how a caller asks for a panel that
    KEEPS the rows the invalidation rule drops -- silently returned the shipped
    filtered panel, and two experiments reported the filtered population as the
    full one before the cause was found."""
    import inspect

    from prosignal.validation import research_panel as rp

    sig = inspect.signature(rp.build_research_panel)
    default = sig.parameters["exit_rules"].default
    assert default is rp._USE_CONFIG, (
        "the default must be a sentinel; None is a meaningful value here"
    )
    assert default is not None

    src = inspect.getsource(rp.build_research_panel)
    assert "exit_rules is _USE_CONFIG" in src
    assert "exit_rules or (" not in src, (
        "`or` collapses an explicit None into the default and silently returns "
        "the filtered panel"
    )


# ------------------------------------------- one definition of a trade's end
def test_no_module_books_profit_at_t1():
    """`features/exits.py` exists to collapse the codebase's competing answers
    to "how did this trade end". It reached the label, portfolio_sim and
    research_panel -- and missed two: `outcomes._resolve_one` and
    `backtest._simulate` both took profit at T1 (1.5R) while the model is
    fitted against T2 (3.0R) via `rules_from_config`.

    Measured out-of-sample over 49 purged dates, crossing label target against
    book target:

        label   book     NET     Sharpe
        3.0R    3.0R   +0.36%    +0.21
        1.5R    1.5R   -0.12%    -0.06
        3.0R    1.5R   -0.22%    -0.18   <- what both modules were doing
        1.5R    3.0R   +1.15%    +0.43

    Booking at 1.5R is worse under BOTH labels, and the combination in use was
    the worst of the four.
    """
    import inspect

    from prosignal import backtest, outcomes

    for fn in (outcomes._resolve_one, backtest._simulate):
        src = inspect.getsource(fn)
        assert 'reason = t1, "target_1"' not in src, (
            f"{fn.__qualname__} books profit at T1; the model is fitted at T2"
        )
        assert "touched_t1" in src, (
            f"{fn.__qualname__} must still RECORD reaching T1 -- it is real "
            f"information about the trade, just not the end of it"
        )


def test_the_traded_target_is_the_one_the_label_is_fitted_on():
    """`rules_from_config` reads t2_r_multiple. If the label ever drifts off
    the target the book actually takes, the two are measuring different
    strategies again."""
    from prosignal.config.loader import load_config
    from prosignal.features.exits import rules_from_config

    cfg = load_config()
    p = cfg.params
    rules = rules_from_config(p.stage4_core_score, p.stage7_risk)
    assert rules.target_r_multiple == float(p.stage7_risk.targets.t2_r_multiple.value)


# ------------------------------------- an entry rule must not trap a position
def test_a_held_name_below_invalidation_still_reaches_the_exit_band():
    """THE second-order regression the domain filter introduced.

    Stage 3 removes below-invalidation names from the universe. For an ENTRY
    that is right. For a name already held it is catastrophic: the position
    reaches neither Stage 6's exit band nor Stage 7's exit hierarchy, falls
    through to the orphan review, and is reported "in universe and trading
    normally" at the exact moment it has met its own first exit condition.

    The original defect BOUGHT names below invalidation. This one refused to
    SELL them, which is worse.
    """
    import inspect

    from prosignal.stages import stage3_eligibility as s3

    src = inspect.getsource(s3.run)
    assert "open_book" in src, "Stage 3 must know what is held"
    block = src[src.index("# 5b. THE MODEL'S OWN DOMAIN."):]
    block = block[:block.index("# 6. Manual exclusion")]
    assert "sym not in open_book" in block, (
        "the model-domain filter must exempt held names -- every OTHER gate in "
        "this stage is about whether the name may be traded at all, which "
        "applies to a sale as much as to a purchase"
    )


def test_the_open_book_is_read_before_the_stage_that_can_exclude_a_name():
    """Ordering, not just presence. Reading the book after Stage 3 leaves the
    filter with no way to know what is held."""
    import inspect

    from prosignal import pipeline

    src = inspect.getsource(pipeline._run_analysis_locked)
    assert src.index("open_book = list(") < src.index("stage3_eligibility.run"), (
        "the open book must be read before Stage 3 runs"
    )
    # And still reaches the two stages that already relied on it.
    assert src.count("held=open_book") >= 2


def test_every_other_stage_3_gate_still_applies_to_a_held_name():
    """The exemption is scoped to ONE gate. Liquidity, data quality, price
    floor and the circuit test are about whether the name can be traded at all,
    and a position you cannot sell is not one to stop checking."""
    import inspect

    from prosignal.stages import stage3_eligibility as s3

    src = inspect.getsource(s3.run)
    # The per-symbol gate loop, up to the model-domain filter. The book is
    # ASSIGNED above the loop; what matters is that no gate inside it reads.
    loop = src[src.index("for sym in symbols:"):]
    before = loop[:loop.index("# 5b. THE MODEL\'S OWN DOMAIN.")]
    assert "open_book" not in before, (
        "no gate before the model-domain filter may consult the book -- "
        "liquidity, data quality, the price floor and the circuit test are "
        "about whether the name can be traded at all, and a position you "
        "cannot sell is not one to stop checking"
    )


# ------------------------------------- the orphan review must not reassure
def test_a_held_name_the_run_rejected_is_not_reported_as_trading_normally():
    """`in_universe` is tested against the RAW universe, so a held name that an
    eligibility gate explicitly rejected still reads as "in universe" and the
    directive said "in universe and trading normally". The operator was told
    nothing was wrong about a position the engine had refused to evaluate.

    The ACTION stays HOLD -- entry gates do not govern open positions, and
    exiting into one pays the worst price available for a reason the thesis
    never priced. Only the reason is corrected.
    """
    import datetime as dt

    import numpy as np
    import pandas as pd

    from prosignal.positions import PositionAction, review_open_position

    idx = pd.bdate_range("2026-01-01", periods=80)
    px = np.linspace(100, 120, 80)
    frame = pd.DataFrame({"date": idx, "open": px, "high": px * 1.01,
                          "low": px * 0.99, "close": px, "volume": 1e6})
    sessions = [d.date() for d in idx]

    quiet = review_open_position("X", frame, sessions[-1], in_universe=True,
                                 sessions=sessions)
    assert quiet.reason == "in universe and trading normally"

    flagged = review_open_position(
        "X", frame, sessions[-1], in_universe=True, sessions=sessions,
        excluded_because="results due in ~39 sessions, inside the 45-session "
                         "holding window")
    assert flagged.action is PositionAction.HOLD, "the action must not change"
    assert "trading normally" not in flagged.reason
    assert "set it aside" in flagged.reason
    assert "45-session" in flagged.reason, "the actual gate must reach the reader"


def test_the_pipeline_hands_the_rejection_reason_to_the_review():
    import inspect

    from prosignal import pipeline

    src = inspect.getsource(pipeline._review_open_positions)
    assert "eligibility.rejected" in src
    assert "excluded_because=why" in src
    call = inspect.getsource(pipeline._run_analysis_locked)
    assert "eligibility=eligibility" in call, (
        "the review cannot explain a rejection it was never given"
    )


# =============================================================================
# The panel and the book must draw from the SAME population
# =============================================================================


def test_the_panel_can_be_restricted_to_what_the_book_can_open():
    """`admission_rules` applies the live predicate whatever the LABEL is.

    The mask lived inside `resolve_exits`, which `build_panel` reaches only
    when `exit_rules` is not None -- and the shipped `triple_barrier: false`
    makes it None. So on the shipped config the model is fitted and ranked on
    a population the engine refuses part of, and the simulator discovers the
    difference at fill time by leaving slots empty: 7.29 of 8 filled.

    Admission and labelling are independent questions. Tying them together is
    how they came apart, so this checks they can now be asked separately.
    """
    import pandas as pd

    from prosignal.features.crosssec import build_panel

    n, m = 500, 60
    rng = np.random.default_rng(17)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"S{i:02d}" for i in range(m)]
    close = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))) for c in cols},
        index=idx)
    high, low = close * 1.01, close * 0.99
    open_ = close.shift(1).bfill()
    turnover = pd.DataFrame(5e8, index=idx, columns=cols)

    common = dict(horizon=63, step=21, high=high, low=low, open_=open_,
                  min_names=5)
    wide = build_panel(close, turnover, **common)
    narrow = build_panel(close, turnover, admission_rules=RULES, **common)

    assert not wide.empty and not narrow.empty
    assert len(narrow) < len(wide), (
        "the admission mask removed nothing, so either the fixture has no name "
        "below its invalidation level or the mask is not being applied"
    )
    # Every row kept must satisfy the predicate the live path applies.
    atr = atr_panel(high, low, close, RULES.atr_period_sessions, RULES.atr_method)
    ma = ma_panel(close, RULES.invalidation_ma_sessions)
    for _, row in narrow.iterrows():
        d, s = pd.Timestamp(row["date"]), row["symbol"]
        lvl_ma, lvl_atr = ma.loc[d, s], atr.loc[d, s]
        if not (np.isfinite(lvl_ma) and np.isfinite(lvl_atr)):
            continue                      # unknown level: admitted, documented
        assert bool(tradeable_at_entry(close.loc[d, s], lvl_ma, lvl_atr, RULES)), (
            f"{s} on {d.date()} is in the panel and below its own invalidation "
            f"level, which is a name the engine would refuse to open"
        )


def test_the_admission_policy_is_a_config_value_the_panel_reads():
    """It must be reachable from `parameters.yaml`, not buried in a module.

    It briefly lived as a code constant to protect
    `baseline-v1@127d8a314ec49aa2` from a schema change. That reasoning expired
    when v1 was closed VOID in the epoch ledger. A correctness control the
    config cannot reach is the mirror image of `holdout.sacred`, which was a
    config value nothing read -- both leave the operator unable to see what the
    engine is doing from the file that is supposed to say.
    """
    from prosignal.config.loader import load_config
    from prosignal.validation.research_panel import ADMIT_ONLY_TRADEABLE

    cfg = load_config()
    declared = cfg.params.universe.train_on_admissible_only
    assert bool(declared.value) is True, (
        "the shipped engine must fit on the population it can trade; turning "
        "this off is a research question and opens a new epoch"
    )
    assert ADMIT_ONLY_TRADEABLE is True, (
        "the module fallback must agree with the shipped config, or a caller "
        "without a config silently trains on a different population"
    )
