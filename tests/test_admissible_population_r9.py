"""R9 -- the population the model is fitted on, and where the fit came from.

The audit asked for five specific tests and these are them: point-in-time
membership, a future-membership mutation, the leave-the-panel case, a population
audit as a time series rather than a single number, and coefficient provenance.

`test_pit_universe.py` already covers the universe SCREEN and is untouched. The
question here is narrower and was the one nothing asked: the screen decides what
was LISTED, and this file is about the difference between that and what the book
could have OPENED -- the gap R9 is.

WHY A TIME SERIES. The engine reported "7.29 of 8 slots fill" -- one number,
averaged over nine years. An average hides the shape, and the shape is the
finding. Measured on the real panel the admissible fraction runs from 38% to
100% with a standard deviation of 16 points, and its tightest dates are
2020-03-02, 2022-06-15 and 2019-02-18. It is not a level shift that biases a fit
predictably; it is a drift that is worst exactly when a stop matters.

WHY A MUTATION TEST. Every test here would pass against a universe resolver that
leaked the future, because leakage makes the answer BETTER-looking rather than
malformed. So the guard is inverted: the leak is introduced deliberately and the
test must go red. A survivorship check that has never been shown to fail is not
a check.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prosignal.features.crosssec import build_panel
from prosignal.features.exits import (ExitRules, atr_panel, ma_panel,
                                      tradeable_at_entry)

RULES = ExitRules(horizon=20, invalidation_ma_sessions=50,
                  invalidation_buffer_atr=1.5)


# =============================================================================
# 1. Point-in-time membership
# =============================================================================


class _Store:
    """A price store that RECORDS what was asked of it.

    The interesting property of a point-in-time resolver is not what it
    returns; it is what it read. A resolver that reads the whole history and
    then filters looks identical from the outside until the day someone
    changes the filter.
    """

    def __init__(self, frame, sessions):
        self._f, self._s = frame, sessions
        self.reads = []

    def price_sessions(self):
        return list(self._s)

    def read_prices(self, symbols=None, start=None, end=None, columns=None):
        self.reads.append((start, end))
        f = self._f
        if start is not None:
            f = f[f["date"] >= pd.Timestamp(start)]
        if end is not None:
            f = f[f["date"] <= pd.Timestamp(end)]
        return f.copy()

    def read_sector_map(self):
        return None

    def read_equity_master(self):
        """Every symbol listed long before the fixture starts.

        The listing-date screen is a different question from liquidity and
        must not be what decides these tests; if it were, a name could pass or
        fail for a reason unrelated to the property being checked.
        """
        return pd.DataFrame({
            "symbol": sorted(self._f["symbol"].unique()),
            "listing_date": pd.Timestamp("2015-01-01"),
        })


def _prices(symbols, start="2023-01-02", n=400, turnover=5e8):
    idx = pd.bdate_range(start, periods=n)
    rows = []
    for s in symbols:
        rows.append(pd.DataFrame({
            "date": idx, "symbol": s, "close": 500.0,
            "turnover": turnover, "series": "EQ", "volume": 1e6,
            "open": 500.0, "high": 505.0, "low": 495.0, "deliv_pct": 50.0}))
    return pd.concat(rows, ignore_index=True)


def _resolver(store, cfg):
    from prosignal.data.universe import UniverseResolver

    return UniverseResolver(store, cfg)


def test_membership_reads_nothing_after_the_decision_date(live_cfg):
    """The property, checked at the boundary rather than asserted.

    A universe resolved for a past date must be built from sessions at or
    before that date. Anything else is survivorship: the names that were
    liquid in 2023 are being chosen using how they traded in 2025.
    """
    px = _prices(["AAA", "BBB", "CCC"])
    sessions = sorted({d.date() for d in px["date"]})
    store = _Store(px, sessions)
    as_of = sessions[200]

    snap = _resolver(store, live_cfg).resolve_liquidity_pit(
        as_of=as_of, min_adtv_inr=1e7, lookback_sessions=60, max_names=50,
        min_history_sessions=20, min_price_inr=10.0)

    assert snap.symbols
    assert snap.as_of == as_of
    for start, end in store.reads:
        assert end is not None and pd.Timestamp(end).date() <= as_of, (
            f"the resolver read prices up to {end} while resolving {as_of}"
        )
    assert snap.survivorship_risk is False


def test_a_name_that_dies_later_is_still_in_an_earlier_universe(live_cfg):
    """Survivorship, stated as the case that matters.

    A name that was liquid in 2023 and collapsed in 2024 MUST appear in the
    2023 universe. A backtest built on today's survivors would drop it, and
    would report the returns of a portfolio nobody could have held.
    """
    px = _prices(["ALIVE", "DOOMED"])
    dead = (px["symbol"] == "DOOMED") & (px["date"] > pd.Timestamp("2024-01-01"))
    px.loc[dead, "turnover"] = 1e4          # collapses, stops trading
    px.loc[dead, "close"] = 3.0
    sessions = sorted({d.date() for d in px["date"]})
    store = _Store(px, sessions)

    early = _resolver(store, live_cfg).resolve_liquidity_pit(
        as_of=dt.date(2023, 9, 1), min_adtv_inr=1e7, lookback_sessions=60,
        max_names=50, min_history_sessions=20, min_price_inr=10.0)
    late = _resolver(store, live_cfg).resolve_liquidity_pit(
        as_of=sessions[-1], min_adtv_inr=1e7, lookback_sessions=60,
        max_names=50, min_history_sessions=20, min_price_inr=10.0)

    assert "DOOMED" in early.symbols, (
        "a name that was liquid on the date is missing from that date's "
        "universe because of what happened to it afterwards"
    )
    assert "DOOMED" not in late.symbols, (
        "and it must leave once it stops qualifying, or the screen is not a "
        "screen"
    )


# =============================================================================
# 2. The future-membership mutation
# =============================================================================


def test_a_resolver_that_saw_the_future_would_be_caught(live_cfg):
    """THE MUTATION. Introduce the leak; the check must go red.

    A survivorship guard that has never been observed to fail is not evidence.
    Here the store is rigged to ignore the `end` bound -- the single most
    likely way this breaks in practice, since every other argument still looks
    correct -- and the point-in-time assertion must catch it.
    """
    px = _prices(["AAA", "LATEBLOOMER"])
    as_of = dt.date(2023, 6, 1)
    # Illiquid on every session the honest resolver may look at, and liquid on
    # every session after it. The median over the lookback window is therefore
    # below the floor, and the median over the whole file is above it -- so the
    # leak, and only the leak, changes the answer.
    early = ((px["symbol"] == "LATEBLOOMER")
             & (px["date"] <= pd.Timestamp(as_of)))
    px.loc[early, "turnover"] = 1e4

    sessions = sorted({d.date() for d in px["date"]})

    class _Leaky(_Store):
        def read_prices(self, symbols=None, start=None, end=None, columns=None):
            # The bug: `end` is dropped. Everything else is unchanged.
            return super().read_prices(symbols, start, None, columns)

    honest = _resolver(_Store(px, sessions), live_cfg).resolve_liquidity_pit(
        as_of=as_of, min_adtv_inr=1e7, lookback_sessions=60, max_names=50,
        min_history_sessions=20, min_price_inr=10.0)
    leaked = _resolver(_Leaky(px, sessions), live_cfg).resolve_liquidity_pit(
        as_of=as_of, min_adtv_inr=1e7, lookback_sessions=60, max_names=50,
        min_history_sessions=20, min_price_inr=10.0)

    assert "LATEBLOOMER" not in honest.symbols
    assert "LATEBLOOMER" in leaked.symbols, (
        "the leak did not change the answer, so this mutation proves nothing "
        "and the fixture needs a name whose eligibility actually changes"
    )
    assert set(honest.symbols) != set(leaked.symbols), (
        "a resolver reading past its decision date produced the same universe "
        "as one that did not -- the point-in-time property is untested"
    )


# =============================================================================
# 3. Leave the panel
# =============================================================================


#: `crosssec.MIN_LOOKBACK` is 274 sessions -- the longest feature window -- and
#: `build_panel` skips any date carrying fewer than `min_names` = 40 usable
#: rows. A fixture that misses either produces an EMPTY panel, and a test whose
#: subject is an empty frame passes for the wrong reason. Both are asserted
#: below rather than assumed.
PANEL_SESSIONS = 420
PANEL_NAMES = 60

#: `build_panel` skips any DATE with fewer than `min_names` usable rows. At the
#: shipped default of 40 the excluded cohort takes the whole date with it, and
#: the comparison below would then be measuring the date threshold rather than
#: the admission predicate. Lowered here so the only thing that differs between
#: the two panels is the predicate.
MIN_NAMES = 20

#: Every fifth session. The panel loop is the slow part and admission is a
#: per-date predicate, so sampling dates costs nothing the test needs.
STEP = 5


def _panel_frames(n=PANEL_SESSIONS, k=PANEL_NAMES):
    """Half the universe holds above its invalidation level; half falls through.

    Paths are perturbed per name so the cross-sectional ranks are not
    degenerate: with identical series every feature ties, `cross_sectional_rank`
    returns one value for the whole date, and the panel would be uninformative
    in a way that has nothing to do with what is being tested.
    """
    from prosignal.features.crosssec import MIN_LOOKBACK

    assert n > MIN_LOOKBACK + 60, "the fixture is too short to produce a panel"
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(20260829)
    turn = MIN_LOOKBACK + 40           # the break, well after the windows fill
    cols = {}
    for j in range(k // 2):
        drift = 1.0 + 0.15 * rng.standard_normal()
        wig = np.cumsum(rng.normal(0.0, 0.6, n))
        cols[f"HOLDS{j:02d}"] = np.linspace(100.0, 100.0 + 160.0 * drift, n) + wig
        cols[f"FALLS{j:02d}"] = np.r_[
            np.linspace(100.0, 100.0 + 140.0 * drift, turn),
            np.linspace(100.0 + 140.0 * drift, 55.0, n - turn)] + wig
    close = pd.DataFrame(cols, index=idx).clip(lower=5.0)
    return close, close * 1.01, close * 0.99


def _falls(frame):
    return frame[frame["symbol"].str.startswith("FALLS")]


def _holds(frame):
    return frame[frame["symbol"].str.startswith("HOLDS")]


def test_a_name_leaves_the_panel_and_its_earlier_rows_survive():
    """The half of R9 that is easy to get wrong in the other direction.

    Admission is a per-DATE predicate. A name that becomes untradeable must
    disappear from that date forward and must keep every row from before it --
    dropping the symbol entirely would delete real evidence, and keeping it
    throughout is the defect.
    """
    close, high, low = _panel_frames()
    turnover = pd.DataFrame(5e8, index=close.index, columns=close.columns)

    wide = build_panel(close, turnover, horizon=20, step=STEP, high=high,
                       low=low, open_=close, min_names=MIN_NAMES)
    narrow = build_panel(close, turnover, horizon=20, step=STEP, high=high,
                         low=low, open_=close, admission_rules=RULES,
                         min_names=MIN_NAMES)

    assert not wide.empty and not narrow.empty, (
        "the fixture produced no panel at all, so nothing below is a test"
    )
    w, n = _falls(wide), _falls(narrow)
    assert len(n) < len(w), "the falling names were never excluded"
    assert len(n) > 0, (
        "the falling names lost every row, including the ones where they were "
        "perfectly tradeable -- admission is per date, not per symbol"
    )
    # Per name, not in aggregate: an aggregate count can hide one symbol
    # vanishing entirely behind another keeping extra rows.
    per_wide = w.groupby("symbol").size()
    per_narrow = n.groupby("symbol").size()
    dropped_entirely = [s for s in per_wide.index if per_narrow.get(s, 0) == 0]
    assert not dropped_entirely, (
        f"{dropped_entirely[:3]} lost every row; their pre-break sessions were "
        f"tradeable and are real training evidence"
    )
    assert n["date"].max() < w["date"].max(), (
        "exclusion is happening somewhere other than the end of these names' "
        "lives"
    )
    # And the names that never fall through must be untouched.
    assert len(_holds(narrow)) == len(_holds(wide)), (
        "admission removed rows from names that were always tradeable"
    )


def test_the_panel_predicate_is_the_predicate_the_book_uses():
    """One rule, checked against `tradeable_at_entry` directly.

    Two implementations of "can this be opened" is how the populations came
    apart in the first place, so the panel's exclusions are compared against
    the live predicate row by row rather than by shape.
    """
    close, high, low = _panel_frames()
    turnover = pd.DataFrame(5e8, index=close.index, columns=close.columns)
    atr = atr_panel(high, low, close, RULES.atr_period_sessions, RULES.atr_method)
    ma = ma_panel(close, RULES.invalidation_ma_sessions)

    narrow = build_panel(close, turnover, horizon=20, step=STEP, high=high,
                         low=low, open_=close, admission_rules=RULES,
                         min_names=MIN_NAMES)
    kept = {(pd.Timestamp(d), s) for d, s in zip(narrow["date"], narrow["symbol"])}

    checked = 0
    for i in range(60, len(close) - 21):
        d = close.index[i]
        if not np.isfinite(ma.iloc[i]).any():
            continue
        ok = tradeable_at_entry(close.iloc[i], ma.iloc[i], atr.iloc[i], RULES)
        for sym in close.columns:
            if not np.isfinite(ma.iloc[i][sym]) or not np.isfinite(atr.iloc[i][sym]):
                continue
            checked += 1
            if not bool(ok[sym]):
                assert (d, sym) not in kept, (
                    f"{sym} on {d.date()} is below its invalidation level and "
                    f"is in the training panel"
                )
    assert checked > 500, "the comparison covered too few rows to mean anything"


# =============================================================================
# 4. The population audit, as a series
# =============================================================================


def test_the_admissible_fraction_is_reported_over_time_not_as_one_number():
    """One average hid the finding. The shape is the finding.

    A flat exclusion rate would mean the wide-panel fit is biased but stable.
    A rate that moves means the training population drifts against the tradable
    one, and it drifts most when the market falls -- which is when the
    difference between "eligible" and "can be bought" matters.
    """
    close, high, low = _panel_frames()
    turnover = pd.DataFrame(5e8, index=close.index, columns=close.columns)
    atr = atr_panel(high, low, close, RULES.atr_period_sessions, RULES.atr_method)
    ma = ma_panel(close, RULES.invalidation_ma_sessions)

    series = []
    for i in range(60, len(close) - 21):
        ok = tradeable_at_entry(close.iloc[i], ma.iloc[i], atr.iloc[i], RULES)
        known = np.isfinite(ma.iloc[i]) & np.isfinite(atr.iloc[i])
        if not known.any():
            continue
        series.append((close.index[i], float(ok[known].mean())))

    frac = pd.Series([v for _, v in series],
                     index=[d for d, _ in series]).sort_index()
    assert len(frac) > 100

    assert frac.min() < frac.max() - 0.10, (
        "the admissible fraction is nearly constant on this fixture, so the "
        "series proves nothing the average did not; if this ever becomes true "
        "of the real panel, R9 is a level shift and not a drift"
    )
    assert frac.iloc[-1] < frac.iloc[0], (
        "on a fixture built to fall through its own level, admission must "
        "tighten -- if it loosens the predicate has the wrong sign"
    )
    assert 0.0 <= frac.min() and frac.max() <= 1.0


# =============================================================================
# 5. Coefficient provenance
# =============================================================================


def test_the_fit_records_which_population_it_was_estimated_on():
    """R9's cache trap, closed.

    `label_fingerprint` recorded what the model PREDICTS. Flipping
    `train_on_admissible_only` changes every coefficient and left the
    fingerprint identical, so a cached wide-population fit would have looked
    valid in every respect `load_cached` checks -- and the engine would have
    scored with it for up to `refit_every * 2` sessions after the correction
    shipped, with every run looking normal.
    """
    from prosignal.features import crossmodel as cm

    wide = cm.label_fingerprint(63)
    narrow = cm.label_fingerprint(63, admission_rules=RULES)

    assert wide["population"] == "all_eligible"
    assert narrow["population"] == "admissible"
    assert wide != narrow, (
        "the two populations fingerprint identically, so a fit from one is "
        "served for the other"
    )
    # And the geometry matters, not only the on/off switch.
    other = cm.label_fingerprint(
        63, admission_rules=ExitRules(horizon=20, invalidation_ma_sessions=200,
                                      invalidation_buffer_atr=1.5))
    assert other != narrow, (
        "two different invalidation geometries admit two different "
        "populations and must not share a fingerprint"
    )


def test_a_wide_population_cache_is_refused_after_the_switch_flips(tmp_path):
    """End to end through `load_cached`, which is what actually decides."""
    import json

    from prosignal.features import crossmodel as cm

    path = tmp_path / "crosssec_model.json"
    wide = cm.label_fingerprint(63)
    path.write_text(json.dumps({
        "fitted_for": dt.date(2026, 8, 1).isoformat(),
        "train_end": dt.date(2026, 5, 1).isoformat(),
        "n_train": 5000, "intercept": 0.0,
        "coef": {"mom_f": 0.4, "value_f": 0.1},
        "features": ["mom_f", "value_f"],
        "mu": [0.0, 0.0], "sd": [1.0, 1.0],
        "estimator": "fama_macbeth", "label": wide,
    }))
    as_of = dt.date(2026, 8, 5)

    assert cm.load_cached(path, as_of, 21, estimator="fama_macbeth",
                          label=wide) is not None, "the fixture itself is stale"
    assert cm.load_cached(path, as_of, 21, estimator="fama_macbeth",
                          label=cm.label_fingerprint(
                              63, admission_rules=RULES)) is None, (
        "a fit estimated on the wide population was served to an engine "
        "trading the admissible one"
    )




