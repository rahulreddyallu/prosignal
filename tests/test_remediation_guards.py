"""Regression protection for the audit remediation.

One test per defect the audit found, written so that reverting the fix turns it
red. Each names the finding it guards and what the original failure looked like,
because a guard whose reason is lost is a guard that gets deleted in the next
refactor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.crosssec import (LIVE_HISTORY_SESSIONS, MIN_LOOKBACK,
                                         NO_ADJUSTMENT, build_panel,
                                         features_for_date, liquidity_mask)
from prosignal.features.famamacbeth import fama_macbeth, newey_west_se
from prosignal.validation.metrics import deflated_sharpe_ratio
from prosignal.validation.significance import analytic_vif

SESSIONS = 900
SYMS = [f"S{i:02d}" for i in range(60)]


def _frames(n=SESSIONS, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    steps = rng.normal(0.0005, 0.018, size=(n, len(SYMS)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=SYMS)
    turnover = pd.DataFrame(rng.uniform(2e8, 9e8, size=(n, len(SYMS))), index=idx, columns=SYMS)
    high = close * 1.01
    low = close * 0.99
    return close, turnover, high, low


# =============================================================================
# F2 -- the live feature row is dated the DECISION DATE
# =============================================================================
class TestLiveFeatureDate:
    """`build_panel(hist.tail(MIN_LOOKBACK + 5), horizon=1, step=21)` could never
    reach the last row: the loop bound is `len(dates) - horizon`, so at any
    horizon >= 1 the final row is unreachable, and with the +5 buffer and a
    21-session stride it landed exactly FOUR sessions early. Observed live: a run
    as_of 2026-08-25 scored features dated 2026-08-19. Only 64% of the top eight
    names agreed with the top eight the same model gives on the decision date."""

    def test_the_feature_row_is_dated_the_last_session(self):
        close, turnover, _h, _l = _frames()
        row = features_for_date(close, turnover)
        assert not row.empty
        assert row["date"].max() == close.index[-1], (
            "the live feature row must be dated the decision session; anything "
            "earlier prices a stale thesis at today's close")

    def test_it_is_exact_at_every_history_length(self):
        """The old idiom failed as a function of the window, so the guard has to
        vary the window rather than test one length."""
        close, turnover, _h, _l = _frames()
        for extra in (1, 2, 5, 21, 40, 100):
            n = MIN_LOOKBACK + extra
            sub_c, sub_t = close.tail(n), turnover.tail(n)
            row = features_for_date(sub_c, sub_t)
            if row.empty:
                continue
            assert row["date"].max() == sub_c.index[-1], f"stale at +{extra}"

    def test_it_refuses_rather_than_returning_an_earlier_date(self):
        close, turnover, _h, _l = _frames()
        short = close.tail(MIN_LOOKBACK - 5)
        assert features_for_date(short, turnover.tail(MIN_LOOKBACK - 5)).empty

    def test_the_eligible_benchmark_path_does_not_crash(self):
        """Found by the adversarial pass over the fix itself. `close.where(series,
        axis=1)` raises on a column-indexed condition, so passing a screen to the
        new live row was a latent crash on a path production reaches as soon as a
        caller supplies one."""
        close, turnover, _h, _l = _frames()
        e = pd.Series(True, index=close.columns)
        e.iloc[:10] = False
        row = features_for_date(close, turnover, eligible=e, min_names=10)
        assert not row.empty
        assert not set(row["symbol"]) & set(close.columns[:10])

    def test_narrowing_rows_does_not_move_the_benchmark(self):
        """`admissible` narrows the ROWS; `eligible` defines the MARKET. Folding
        them into one argument measured beta against a different market live than
        in training."""
        close, turnover, _h, _l = _frames()
        e = pd.Series(True, index=close.columns)
        adm = pd.Series(True, index=close.columns)
        adm.iloc[:20] = False
        wide = features_for_date(close, turnover, eligible=e, min_names=10)
        narrow = features_for_date(close, turnover, eligible=e, admissible=adm,
                                   min_names=10)
        common = sorted(set(wide["symbol"]) & set(narrow["symbol"]))
        assert len(common) > 20
        a = wide.set_index("symbol").loc[common, "beta_120"].to_numpy()
        b = narrow.set_index("symbol").loc[common, "beta_120"].to_numpy()
        assert np.allclose(a, b, atol=1e-12, equal_nan=True)

    def test_the_live_window_covers_the_reversal_standardisation(self):
        """`resid_reversal` standardises over 756 sessions where they exist. A
        live row built off MIN_LOOKBACK computed a different statistic from the
        training rows it was scored against."""
        from prosignal.features.crosssec import REVERSAL_STD_WINDOW
        assert LIVE_HISTORY_SESSIONS > REVERSAL_STD_WINDOW


# =============================================================================
# F1 -- the Deflated Sharpe must be able to fail
# =============================================================================
class TestDeflatedSharpeCanFail:
    """It ran on the POOLED (split, date) vector -- 639 entries over 71 distinct
    dates -- with `sr_var = 1/(n-1)` while the docstring claimed a conservative
    unit variance. It returned 1.000 and still passed at 100,000 trials."""

    @staticmethod
    def _series(n=71, seed=5):
        """A DETERMINISTICALLY STRONG series.

        The direction tests below need the observed Sharpe to sit ABOVE the
        benchmark: the PSR is Phi((SR - SR*) * sqrt(n_eff - 1) / den), so below
        the bar a smaller n_eff moves the answer UP toward 0.5 and the
        assertion reverses for arithmetic reasons rather than wiring ones. The
        earlier fixture drew SR ~ 0.26 against a bar that could exceed it.
        """
        x = np.random.default_rng(seed).normal(0.0, 0.04, n)
        x = x - x.mean() + 0.030          # SR ~ 0.75 per period, deterministic
        return list(x)

    def test_effective_n_moves_the_answer(self):
        r = self._series()
        naive = deflated_sharpe_ratio(r, n_trials=81, sr_variance=0.005)
        # `sr_variance` is PINNED so the only thing that differs between the
        # two calls is the independent-observation count. Without it the
        # comparison is confounded: an argument that changed BOTH the PSR's
        # sqrt(n-1) term and the benchmark could cancel itself and still look
        # wired. See metrics.deflated_sharpe_ratio's variance ladder.
        honest = deflated_sharpe_ratio(r, n_trials=81, effective_n=len(r) / 2.96,
                                       sr_variance=0.005)
        assert honest.deflated_sr < naive.deflated_sr, (
            "declaring fewer independent observations must lower the DSR; if it "
            "does not, the overlap correction is not reaching the statistic")
        assert honest.effective_n < honest.n_observations

    def test_effective_n_cannot_exceed_the_sample(self):
        r = self._series()
        d = deflated_sharpe_ratio(r, n_trials=10, effective_n=10_000)
        assert d.effective_n == len(r), (
            "a caller must not be able to declare independence it does not have")

    def test_it_responds_to_the_trial_count_on_the_PRODUCTION_branch(self):
        """The two existing DSR tests both pass `trial_sharpes` explicitly. No
        production caller ever did, so they exercised a branch the engine never
        took -- and on the branch it DID take the statistic was flat in the
        trial count. This one deliberately omits trial_sharpes."""
        r = self._series()
        few = deflated_sharpe_ratio(r, n_trials=1, effective_n=24)
        many = deflated_sharpe_ratio(r, n_trials=5_000, effective_n=24)
        assert many.benchmark_sr > few.benchmark_sr
        assert many.deflated_sr < few.deflated_sr - 0.05, (
            "the DSR must move materially with the trial count even when no "
            "trial distribution is supplied")

    def test_an_absurd_trial_count_fails(self):
        r = self._series()
        d = deflated_sharpe_ratio(r, n_trials=1_000_000, effective_n=24)
        assert not d.passes, "a million trials must not pass"

    def test_it_reports_where_its_variance_came_from(self):
        r = self._series()
        a = deflated_sharpe_ratio(r, n_trials=10, effective_n=24)
        # SIX scores against ten charged trials: above
        # MIN_TRIAL_SCORE_COVERAGE, so the measured dispersion is allowed to
        # set the bar. Four of ten is under-covered and falls back to the
        # conservative unit, which is a different branch and a different test.
        b = deflated_sharpe_ratio(r, n_trials=10, effective_n=24,
                                  trial_sharpes=[0.1, 0.4, -0.2, 0.3, 0.25, -0.05])
        # The SOURCE is a stable machine token; the human phrase lives in the
        # interpretation. Asserting the phrase against the token coupled a
        # report string to an enum and broke when the enum was named.
        from prosignal.validation.metrics import (SR_VAR_UNIT,
                                                   SR_VAR_FROM_TRIALS)
        assert a.sr_variance_source == SR_VAR_UNIT
        assert "no trial Sharpes" in a.interpretation
        assert b.sr_variance_source == SR_VAR_FROM_TRIALS
        assert a.sr_variance != b.sr_variance


# =============================================================================
# F3 -- the significance gate carries the analytic overlap inflation
# =============================================================================
class TestGateUsesAnalyticInflation:
    """`significance.py` derives the inflation an h/s sampling scheme induces and
    documents that estimating it recovers 1.74 where the arithmetic gives 3.00.
    Every reported figure used it; the gate that decides which themes are traded
    did not, and the difference decided `lottery` (t -2.26 vs -1.72)."""

    def test_passing_the_scheme_widens_the_error(self):
        rng = np.random.default_rng(3)
        s = rng.normal(0.05, 0.10, 83)
        plain = newey_west_se(s, 2)
        scheme = newey_west_se(s, 2, horizon_sessions=63, step_sessions=21)
        assert scheme >= plain
        assert scheme > plain * 1.05, (
            "on a 63/21 scheme the analytic inflation is ~2.97; a standard error "
            "that barely moves means the floor is not being applied")

    def test_it_never_narrows_the_error(self):
        """The larger of the two is taken, so real serial dependence on top of
        the scheme's own overlap still gets the bigger penalty."""
        rng = np.random.default_rng(9)
        for lag_corr in (0.0, 0.5, 0.9):
            base = rng.normal(0, 1, 200)
            s = base + lag_corr * np.roll(base, 1)
            assert (newey_west_se(s, 2, horizon_sessions=63, step_sessions=21)
                    >= newey_west_se(s, 2) - 1e-12)

    def test_fama_macbeth_t_stats_carry_it(self):
        rng = np.random.default_rng(4)
        rows = []
        for d in range(90):
            n = 60
            x = rng.normal(0, 1, n)
            rows.append(pd.DataFrame({"date": d, "f_f": x,
                                      "label_rank": 0.08 * x + rng.normal(0, 1, n)}))
        panel = pd.concat(rows, ignore_index=True)
        fm = fama_macbeth(panel, ["f_f"], horizon=63, step=21)
        naive_se = float(np.std(fm.slopes["f_f"].to_numpy(), ddof=1) / np.sqrt(fm.n_dates))
        assert fm.se["f_f"] > naive_se, "the reported error must exceed the naive one"
        implied = (fm.se["f_f"] / naive_se) ** 2
        assert implied >= analytic_vif(63, 21, fm.n_dates) * 0.95


# =============================================================================
# F4 -- the price floor reads the QUOTED price
# =============================================================================
class TestPriceFloorIsNotLookAhead:
    """The floor was applied to the back-adjusted close, so a name that traded at
    Rs 200 before a 1:20 split read Rs 10 on that historical date and failed a
    floor it had cleared. Membership in a past universe depended on a corporate
    action nobody had announced. Measured: 58,411 cells, 165 symbols, none in the
    other direction."""

    def test_a_later_split_does_not_retroactively_delist(self):
        close, turnover, _h, _l = _frames(n=400)
        close = close.copy()
        sym = SYMS[0]
        # traded at ~100 throughout; a 1:20 split lands at the end, so every
        # earlier row is back-adjusted to ~5 and falls under a Rs 20 floor.
        fac = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        fac.loc[fac.index[:-10], sym] = 0.05
        adjusted = close * fac

        # max_names is set wide so the top-N cap does not bind. The cap is a
        # RANKING, so with it binding the screen is not monotone in the floor:
        # recovering one name displaces another, and the one-way property below
        # would not hold for a reason that has nothing to do with the fix.
        blind = liquidity_mask(adjusted, turnover, min_adtv_inr=1e8,
                               lookback_sessions=60, max_names=len(SYMS),
                               min_history_sessions=100, min_price_inr=20.0)
        aware = liquidity_mask(adjusted, turnover, min_adtv_inr=1e8,
                               lookback_sessions=60, max_names=len(SYMS),
                               min_history_sessions=100, min_price_inr=20.0,
                               adj_factor=fac)
        early = adjusted.index[150:-20]
        assert not blind.loc[early, sym].any(), "fixture must reproduce the bug"
        recovered = int(aware.loc[early, sym].sum())
        assert recovered > 0.9 * len(early), (
            f"only {recovered} of {len(early)} sessions recovered; the name "
            f"traded above the floor on those dates and only a split that had "
            f"not happened yet pushed it under")
        # and the correction is one-way: nothing the blind screen admitted is
        # lost. Measured on the real store, 4,905 cells were recovered and none
        # lost -- the bias runs in a single direction, against names that later
        # split.
        assert not (blind & ~aware).to_numpy().any()

    def test_the_sentinel_distinguishes_absent_from_forgotten(self):
        assert NO_ADJUSTMENT is not None
        close, turnover, _h, _l = _frames(n=350)
        a = liquidity_mask(close, turnover, min_adtv_inr=1e8, lookback_sessions=60,
                           max_names=50, min_history_sessions=100, min_price_inr=20.0)
        b = liquidity_mask(close, turnover, min_adtv_inr=1e8, lookback_sessions=60,
                           max_names=50, min_history_sessions=100, min_price_inr=20.0,
                           adj_factor=NO_ADJUSTMENT)
        assert a.equals(b)


# =============================================================================
# F5 -- the panel is masked to the population the engine can open from
# =============================================================================
class TestAdmissiblePopulation:
    """Turning the barrier label off removed `tradeable_at_entry` from the panel
    while Stage 6 went on enforcing it, so the model ranked a population 23.3% of
    which it could never buy and 1.55 of its top eight were refused on 72% of
    dates."""

    def test_the_mask_removes_rows_from_the_panel(self):
        """The predicate is DERIVED, not injected.

        The earlier version handed `build_panel` a boolean frame. That could
        not catch the failure it existed for -- the live gate computes the
        predicate from price against the invalidation level, so a test that
        supplies its own answer never exercises the construction the engine
        uses. Here the first twenty symbols are driven below their own
        invalidation level and the panel must drop them on its own.
        """
        from prosignal.features.exits import ExitRules
        close, turnover, high, low = _frames(n=SESSIONS)
        close = close.copy()
        # A sustained collapse: the last 120 sessions at a third of the level,
        # which puts price far under the 50-session average less 1.5 ATR.
        close.loc[close.index[-120:], SYMS[:20]] = (
            close.loc[close.index[-120:], SYMS[:20]] * 0.33)
        high = close * 1.005
        low = close * 0.995
        rules = ExitRules()
        full = build_panel(close, turnover, horizon=63, step=21)
        masked = build_panel(close, turnover, horizon=63, step=21,
                             high=high, low=low, admission_rules=rules)
        assert not full.empty and not masked.empty
        assert len(masked) < len(full), (
            "the admission predicate removed nothing; the panel is still the "
            "population the book cannot buy from")
        tail = masked[masked["date"] >= close.index[-90]]
        assert not set(tail["symbol"]) & set(SYMS[:20]), (
            "a name below its own invalidation level is still in the panel")

    def test_ranks_are_taken_after_the_mask(self):
        """The point of the mask: a rank must mean the same thing in training and
        at the decision. If ranks were taken first, the surviving rows would
        carry ranks computed over a population that includes names Stage 6
        refuses."""
        from prosignal.features.exits import ExitRules
        close, turnover, high, low = _frames(n=SESSIONS)
        close = close.copy()
        # 45 of 60 survive, above build_panel's min_names floor.
        close.loc[close.index[-120:], SYMS[:15]] = (
            close.loc[close.index[-120:], SYMS[:15]] * 0.33)
        high = close * 1.005
        low = close * 0.995
        masked = build_panel(close, turnover, horizon=63, step=21,
                             high=high, low=low, admission_rules=ExitRules())
        assert not masked.empty
        for _d, g in masked.groupby("date"):
            if len(g) < 5:
                continue
            r = g["mom_6_1_r"].dropna()
            # A rank over the surviving population spans [-1, 1]; a rank over a
            # wider one, subset afterwards, would not reach both ends.
            assert r.max() > 0.85 and r.min() < -0.85, (
                "ranks must be computed over the masked population")
            break


# =============================================================================
# M1 -- book metrics carry a benchmark, or say they do not
# =============================================================================
class TestBookIsBenchmarked:
    """Nothing in the repository ever compared the book to buying the market.
    Every book-level conclusion was stated against zero."""

    @staticmethod
    def _result(with_bench):
        from prosignal.validation.portfolio_sim import PortfolioResult
        rng = np.random.default_rng(2)
        n = 24
        df = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=n),
            "ret": rng.normal(0.01, 0.05, n), "equity": 1.0,
            "gross_ret": rng.normal(0.015, 0.05, n),
            "cost_ret": np.full(n, 0.004),
            "n_held": 8, "n_new": 6, "deployed_frac": 0.8,
            "vol_scale": 1.0, "realised_vol": 0.2})
        if with_bench:
            df["bench_ret"] = rng.normal(0.02, 0.045, n)
        return PortfolioResult(periods=df)

    def test_it_reports_whether_a_benchmark_was_available(self):
        assert self._result(False).metrics()["benchmarked"] is False
        assert self._result(True).metrics()["benchmarked"] is True

    def test_the_benchmark_fields_appear_only_with_one(self):
        bare = self._result(False).metrics()
        full = self._result(True).metrics()
        for key in ("mean_excess", "information_ratio", "bench_sharpe",
                    "beta_to_benchmark", "alpha_per_period"):
            assert key not in bare, f"{key} without a benchmark is not a number"
            assert key in full

    def test_excess_is_return_less_benchmark(self):
        r = self._result(True)
        m = r.metrics()
        expected = float((r.periods["ret"] - r.periods["bench_ret"]).mean())
        assert m["mean_excess"] == pytest.approx(expected, abs=1e-12)


# =============================================================================
# N1 -- adj_factor is never served as a placeholder
# =============================================================================
def test_adj_factor_without_a_price_column_is_refused():
    """The stored column is 1.0 for every row; the meaningful factor is computed
    only when prices are adjusted. A caller that asked for adj_factor alone got
    all-ones and, using it to recover the quoted price, recovered the ADJUSTED
    price -- the exact look-ahead the quoted-price floor exists to remove. Found
    by walking into it while verifying F4."""
    import datetime as dt
    from pathlib import Path
    from prosignal.core.errors import IntegrityError
    from prosignal.data.store import DataStore

    store = DataStore(Path("/nonexistent-curated"), Path("/nonexistent-snapshots"))
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "symbol": ["A", "A"], "adj_factor": [1.0, 1.0], "turnover": [1e8, 1e8],
    })
    with pytest.raises(IntegrityError, match="without a price column"):
        store._apply_corporate_actions(frame, ["date", "symbol", "adj_factor"])


# =============================================================================
# Mutation-probe gaps: controls that survived a deliberate break
# =============================================================================
class TestSurvivingMutations:
    """Six injected defects passed the whole suite unnoticed. Two turned out to
    be correct behaviour; these guard the four that were real gaps."""

    def test_the_stop_distance_clip_is_enforced(self):
        """A live risk control that also drives position size, untested.
        Removing the 2%-15% clip changed no test."""
        from prosignal.features.exits import ExitRules
        r = ExitRules()
        atr = np.array([0.001, 5.0, 50.0])      # tiny, normal, enormous
        entry = np.array([100.0, 100.0, 100.0])
        frac = r.stop_fraction(atr, entry)
        assert frac.min() >= r.min_stop_distance_pct / 100.0 - 1e-12, (
            "an unclipped tiny ATR puts the stop on top of the entry and buys a "
            "position sized by division by almost zero")
        assert frac.max() <= r.max_stop_distance_pct / 100.0 + 1e-12
        assert frac[0] == pytest.approx(r.min_stop_distance_pct / 100.0)
        assert frac[2] == pytest.approx(r.max_stop_distance_pct / 100.0)

    def test_the_liquidity_screen_uses_a_trailing_window(self):
        """Changing the ADTV window from trailing-median to expanding changed no
        test. The screen could silently become a different screen."""
        close, turnover, _h, _l = _frames(n=500)
        turnover = turnover.copy()
        # a name that was liquid early and is illiquid now must FAIL today
        turnover[SYMS[0]] = 1e9
        turnover.iloc[-120:, turnover.columns.get_loc(SYMS[0])] = 1e5
        mask = liquidity_mask(close, turnover, min_adtv_inr=5e8,
                              lookback_sessions=60, max_names=len(SYMS),
                              min_history_sessions=100, min_price_inr=1.0)
        assert not bool(mask[SYMS[0]].iloc[-1]), (
            "a trailing window must forget liquidity from a year ago; an "
            "expanding one would still admit this name")
        assert bool(mask[SYMS[0]].iloc[200]), "and must admit it while it was liquid"

    def test_the_horizon_truncation_is_redundant_not_load_bearing(self):
        """`train_close = hist.iloc[:len(hist)-H]` is commented as 'the leak this
        model exists to avoid'. It is not: `build_panel`'s loop already stops a
        horizon short, so every label ends at or before as_of regardless. The
        REAL protection is the loop bound, and this pins it -- removing the
        comment's claim without pinning the thing that actually protects would
        leave the next reader falsely reassured."""
        close, turnover, _h, _l = _frames(n=SESSIONS)
        panel = build_panel(close, turnover, horizon=63, step=21)
        assert not panel.empty
        last = panel["date"].max()
        pos = list(close.index).index(last)
        assert pos + 63 <= len(close) - 1, (
            "the panel's last label must resolve inside the frame; the loop "
            "bound is what guarantees it")

    def test_uniqueness_weights_are_refused_not_dropped(self):
        """The shipped Fama-MacBeth branch deliberately does not weight, and
        that decision is now stated where the code makes it.

        WHAT CHANGED. This used to assert only that the weights never reached
        `fama_macbeth`, citing a config note whose premise turned out to be
        false: it claimed within-date uniqueness had sd exactly 0.000, which
        would make a per-date WLS arithmetically identical. Measured, the sd
        runs 0.082-0.207 and is zero on none of 87 dates. The conclusion
        survives on a different argument -- uniqueness measures redundancy
        across dates, and a per-date WLS instead up-weights names at the edge
        of their eligibility span by up to 3x -- so the assertion now pins the
        REASON as well as the behaviour."""
        import inspect
        from prosignal.features import crossmodel
        src = inspect.getsource(crossmodel.fit_coefficients)
        ridge_branch = src.split('if estimator == "ridge"')[1].split(
            'if estimator != "fama_macbeth"')[0]
        assert "weights=weights" in ridge_branch, "the ridge branch honours them"
        fm_call = src.split("fm = fama_macbeth(")[1].split(")")[0]
        assert "weight" not in fm_call, (
            "the shipped Fama-MacBeth branch must not weight by uniqueness")
        assert "0.082" in src and "across dates" in src, (
            "the refusal must carry its measured reason at the point of "
            "decision; a bare omission is what let a false justification "
            "stand in the config for months")

    def test_the_weighting_it_refuses_is_implemented_correctly(self):
        """The capability exists and is right -- the decision not to use it is
        a research judgement, not a gap. Uniform weights must reproduce OLS
        bit-for-bit and a zero weight must remove a row exactly."""
        from prosignal.features.famamacbeth import _ols_slopes
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(size=(n, 2))
        y = x @ np.array([0.3, -0.2]) + rng.normal(size=n)
        base = _ols_slopes(x, y)
        assert np.allclose(base, _ols_slopes(x, y, w=np.ones(n)), atol=1e-12)
        assert np.allclose(base, _ols_slopes(x, y, w=np.full(n, 7.3)), atol=1e-12)
        w = np.ones(n)
        w[:20] = 0.0
        assert np.allclose(_ols_slopes(x, y, w=w), _ols_slopes(x[20:], y[20:]),
                           atol=1e-12)


# =============================================================================
# WIRING. The original audit's sharpest finding was that the two DSR tests
# exercised a branch production never took. Testing the function and not the
# call site reproduces exactly that mistake, and a mutation probe over the
# remediation caught it doing so four times. These test the call sites.
# =============================================================================
class TestTheCallSitesAreWired:

    @staticmethod
    def _result(n_dates=60, copies=9, seed=3):
        """A CpcvResult shaped like the real one: every date scored by several
        splits, plus woven path Sharpes."""
        from prosignal.validation.harness import CpcvResult
        rng = np.random.default_rng(seed)
        by_date, pooled = {}, []
        for d in range(n_dates):
            vals = list(rng.normal(0.012, 0.04, copies))
            by_date[d] = vals
            pooled.extend(vals)
        return CpcvResult(
            n_splits=45, n_paths=copies, excess=pooled, excess_by_date=by_date,
            path_sharpes=list(rng.normal(0.2, 0.16, copies)),
            horizon_sessions=63, step_sessions=21)

    def test_deflated_collapses_the_duplication(self):
        r = self._result()
        assert len(r.excess) == 540 and len(r.excess_per_date()) == 60
        d = r.deflated(n_trials=81, horizon_sessions=63, step_sessions=21)
        assert d.n_observations == 60, (
            "the DSR must run on distinct dates, not on (split, date) pairs")
        assert d.effective_n == pytest.approx(
            r.effective_observations(), rel=1e-9), (
            "the overlap deduction must be the harness's own arithmetic")

    def test_deflated_declares_the_overlap_corrected_sample(self):
        r = self._result()
        d = r.deflated(n_trials=81, horizon_sessions=63, step_sessions=21)
        assert d.effective_n < d.n_observations / 2.5, (
            f"effective_n {d.effective_n} does not carry the 63/21 overlap "
            f"inflation; the harness is not passing it")

    def test_deflated_does_NOT_use_the_woven_path_sharpes(self):
        """REVERSED, on measurement. This guard used to require `deflated` to
        default `trial_sharpes` to `path_sharpes`.

        `trial_sharpes` is Bailey & Lopez de Prado's Var[SR]: the dispersion of
        Sharpe ratios ACROSS THE CONFIGURATIONS THAT WERE SEARCHED. The woven
        paths are nine resamples of the ONE configuration that was selected, so
        their spread measures sampling noise inside this strategy rather than
        the spread of the alternatives it beat. It is the smaller number and
        using it SHRINKS the multiple-testing bar -- measured on this engine,
        0.0083 against 0.0455, which moves the answer from FAIL 0.38 to PASS
        0.91. A defence that can be relaxed by handing it the wrong quantity is
        not a defence.

        The variance now comes from the trial registry, or the conservative
        unit says so. `cli.py` passes `reg.recorded_scores()`.
        """
        r = self._result()
        d = r.deflated(n_trials=81, horizon_sessions=63, step_sessions=21)
        assert d.sr_variance == pytest.approx(1.0), (
            "with no registry scores the bar must be the conservative one")
        assert "no trial" in d.interpretation
        path_var = float(np.var(np.asarray(r.path_sharpes), ddof=1))
        assert d.sr_variance != pytest.approx(path_var, rel=1e-6), (
            "the woven path spread is back in the benchmark")
        # And it is still REACHABLE, for a caller that has the real thing.
        d2 = r.deflated(n_trials=81, horizon_sessions=63, step_sessions=21,
                        trial_sharpes=list(np.linspace(-0.3, 0.5, 81)))
        from prosignal.validation.metrics import SR_VAR_FROM_TRIALS
        assert d2.sr_variance_source == SR_VAR_FROM_TRIALS

    def test_dropping_either_wire_raises_the_dsr(self):
        """Both corrections must push the same way, so a silent revert of either
        shows up as a more permissive number."""
        from prosignal.validation.metrics import deflated_sharpe_ratio
        r = self._result()
        # Var[SR] PINNED on both sides. The conservative unit fallback puts
        # the benchmark far above any realistic per-period Sharpe, which
        # saturates both answers at 0.0000 and makes the comparison vacuous --
        # the guard would then pass whatever the sample-size wiring did.
        VAR = 0.005
        honest = deflated_sharpe_ratio(
            r.excess_per_date(), n_trials=81, sr_variance=VAR,
            effective_n=r.effective_observations()).deflated_sr
        pooled_n = deflated_sharpe_ratio(r.excess, n_trials=81,
                                         sr_variance=VAR).deflated_sr
        assert pooled_n > honest, "reverting to the pooled vector must inflate"

    def test_the_simulator_hold_reads_the_intraday_high(self):
        """`_hold` passed high=None, so the TARGET was tested on the close while
        the stop was tested on the low. The discriminating case is a bar whose
        HIGH clears the target and whose close does not."""
        from prosignal.validation.portfolio_sim import _hold, PortfolioParams
        from prosignal.features.exits import atr_panel, ma_panel

        n = 120
        idx = pd.bdate_range("2021-01-01", periods=n)
        sym = "T"
        close = pd.DataFrame({sym: np.full(n, 100.0)}, index=idx)
        high = close.copy(); low = close.copy(); open_ = close.copy()
        # A single spike that touches far above any 3R target and closes flat,
        # placed INSIDE the holding window (entry i=60, horizon 63). It sat at
        # bar 40 -- four bars before the position exists -- so the assertion
        # below was unreachable and the guard could not have failed for the
        # right reason. Caught by re-running it against a deliberately
        # high=None call and finding both arms return 0.0.
        high.iloc[70, 0] = 400.0
        atr = atr_panel(high, low, close, 14, "wilder")
        ma = ma_panel(close, 50)
        p = PortfolioParams(
            capital=1e6, max_positions=8, risk_per_trade_pct=1.0,
            max_participation_of_adtv=0.02, stop_atr_multiple=2.5,
            min_stop_distance_pct=2.0, max_stop_distance_pct=15.0,
            invalidation_ma_sessions=50, invalidation_buffer_atr=1.5,
            horizon_sessions=63, entry_rank=8, exit_rank=16, target_r_multiple=3.0)
        ret, _side = _hold(sym, 60, close, low, open_, ma, atr, p, high=high)
        assert ret is not None
        # entry 100, stop distance clipped to 2%, target = 3 x 2% = +6%
        assert ret == pytest.approx(0.06, abs=1e-9), (
            f"got {ret}; a target cleared by the intraday high must register. "
            f"Reading the close instead returns 0.0 and understates every win.")
        # THE NEGATIVE ARM. A guard that cannot fail is not a guard: this is the
        # exact call the code used to make, and it must give a different answer.
        blind_out = _hold(sym, 60, close, low, open_, ma, atr, p, high=None)
        blind = None if blind_out is None else blind_out[0]
        assert blind == pytest.approx(0.0, abs=1e-12), (
            "without `high` the resolver cannot see the touch; if this now "
            "agrees with the sighted call the test has stopped discriminating")

    def test_fit_predict_passes_the_mask_to_the_training_panel(self):
        """SURVIVED A MUTATION. Replacing `admissible=admissible` with
        `admissible=None` in `fit_predict`'s call to `build_panel` broke no
        test: the sibling guard below watches the LIVE mask, which still
        filters the crashed names out of the output, so the training population
        can silently revert while the ranking still looks right. That is the
        whole defect -- the model fitted on one population and ranking another.

        Captured at the call site rather than grepped for, so it observes what
        `build_panel` actually receives."""
        import prosignal.features.crossmodel as cm
        from prosignal.features.exits import ExitRules

        seen = {}
        real = cm.build_panel

        def spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        n = 800
        rng = np.random.default_rng(11)
        idx = pd.bdate_range("2018-01-01", periods=n)
        syms = [f"N{i:02d}" for i in range(60)]
        close = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, size=(n, len(syms))), axis=0)),
            index=idx, columns=syms)
        turnover = pd.DataFrame(rng.uniform(3e8, 9e8, size=(n, len(syms))),
                                index=idx, columns=syms)
        high, low, open_ = close * 1.005, close * 0.995, close

        cm.build_panel = spy
        try:
            cm.fit_predict(close, turnover, idx[-1].date(), horizon=63,
                           min_train_rows=300, high=high, low=low, open_=open_,
                           admission_rules=ExitRules(), score_symbols=syms)
        finally:
            cm.build_panel = real

        assert "admission_rules" in seen, "build_panel was called without the argument"
        adm = seen["admission_rules"]
        assert adm is not None, (
            "the training panel was built over the whole eligible universe "
            "while the live path masks to the admissible one; the model then "
            "ranks a population it was not fitted on")
        # THE PREDICATE, not a pre-built frame. `build_panel` derives the mask
        # itself so training and the live gate read one construction; what this
        # guard has to pin is that the geometry reaches it.
        from prosignal.features.exits import ExitRules as _ER
        assert isinstance(adm, _ER)
        assert adm.invalidation_ma_sessions > 0 and adm.atr_period_sessions > 0

    def test_fit_predict_masks_the_panel_to_the_admissible_population(self):
        """The end-to-end wire: a name below its own invalidation level on the
        decision date must not appear in the scored output."""
        from prosignal.features import crossmodel as cm
        from prosignal.features.exits import ExitRules

        n = 1000
        rng = np.random.default_rng(21)
        idx = pd.bdate_range("2018-01-01", periods=n)
        syms = [f"N{i:02d}" for i in range(70)]
        steps = rng.normal(0.0004, 0.015, size=(n, len(syms)))
        close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)),
                             index=idx, columns=syms)
        # drive five names far below their 50-session average at the very end
        crashed = syms[:5]
        for s in crashed:
            close.loc[close.index[-3:], s] = close[s].iloc[-4] * 0.35
        turnover = pd.DataFrame(rng.uniform(3e8, 9e8, size=(n, len(syms))),
                                index=idx, columns=syms)
        high, low, open_ = close * 1.005, close * 0.995, close
        geom = ExitRules()
        scores, model, why = cm.fit_predict(
            close, turnover, idx[-1].date(), horizon=63, min_train_rows=300,
            high=high, low=low, open_=open_, admission_rules=geom,
            score_symbols=syms)
        if scores is None:
            pytest.skip(f"the fixture did not produce a model: {why}")
        assert not set(scores.index) & set(crashed), (
            "a name below its own invalidation level cannot be bought, so it "
            "must not be ranked -- otherwise the model ranks a population "
            "Stage 6 refuses and its top eight is not the book")


# =============================================================================
# W-drawdown -- the path figure is the worst SCHEDULE, not a pooled artifact
# =============================================================================
class TestPathDrawdownIsOneSchedule:
    """Caught by re-running the evidence against the restored source tree and
    finding this one number had moved: -35.4% where the record said -21.7%.

    Two wrong answers are available and this pins the boundary between them.
    """

    @staticmethod
    def _phase(rets):
        from prosignal.validation.portfolio_sim import PortfolioResult
        n = len(rets)
        return PortfolioResult(periods=pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=n, freq="63D"[:1] + "B"),
            "ret": np.asarray(rets, dtype="float64"),
            "n_held": 8, "n_new": 6}))

    def test_it_reports_the_worst_phase_not_their_mean(self):
        """An investor runs ONE offset. Averaging three describes a book nobody
        holds, and always a milder one than the schedule that got unlucky."""
        from prosignal.validation.portfolio_sim import _path_drawdown
        deep = self._phase([0.05, -0.30, 0.04, 0.03])
        mild = self._phase([0.02, -0.05, 0.03, 0.02])
        got = _path_drawdown([deep, mild])
        assert got == pytest.approx(-0.30, abs=1e-9), (
            f"got {got}; the reported path drawdown must be the worst single "
            f"schedule, not the mean of the phases")

    def test_it_does_not_compound_the_phases_end_to_end(self):
        """The phases PARTITION the rebalance dates -- they do not run
        alongside one another -- so pooling and compounding lays about three
        times the elapsed period over the sample. On the real book that turned
        -21.7% into -35.4%, and applied to the BENCHMARK it manufactured a
        -62.4% drawdown for a universe whose worst trough was near -38%."""
        from prosignal.validation.portfolio_sim import _path_drawdown
        # Two phases that each recover fully. Concatenated they compound into a
        # far deeper hole than either schedule ever saw.
        # DELIBERATELY DIFFERENT. Two identical phases would let any per-phase
        # reduction pass -- min, max, mean, first -- so the test would not
        # distinguish the fix from several wrong answers.
        a = self._phase([-0.25, 0.40, -0.25, 0.40])   # worst dd -25%
        b = self._phase([-0.10, 0.20, -0.10, 0.20])   # worst dd -10%
        assert _path_drawdown([a]) == pytest.approx(-0.25, abs=1e-9)
        assert _path_drawdown([b]) == pytest.approx(-0.10, abs=1e-9)
        both = _path_drawdown([a, b])
        assert both == pytest.approx(-0.25, abs=1e-9), (
            f"got {both}: two phases that each recover fully must report the "
            f"WORSE schedule (-25%), not their mean (-17.5%) and not the "
            f"end-to-end compounding of both")

    def test_the_mean_of_phases_is_still_reported_under_its_own_name(self):
        """Never delete a number that earlier reports quote. The mean-of-phases
        figure stays, named for what it is, so an old report can be
        reconciled against a new one."""
        import inspect
        from prosignal.validation import portfolio_sim
        src = inspect.getsource(portfolio_sim.phase_summary)
        assert '"max_drawdown_period"' in src
        assert '"max_drawdown_path"' in src
        assert '"max_drawdown"' in src


# =============================================================================
# W5 -- a dead cross-section is not a measurement of zero
# =============================================================================
class TestDeadCrossSectionsAreNotMeasurements:
    """`lstsq` on a rank-deficient design returns the minimum-norm solution,
    which sets a constant column's coefficient to exactly 0.0 -- silently, and
    indistinguishably from a theme measured and found flat. Five of eighty-three
    cross-sections entered `delivery_f`'s mean and standard error that way, and
    removing them moved `lottery_f` across the |t| >= 2 gate."""

    @staticmethod
    def _panel(dead_dates=5, n_dates=40, n=200, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for di, d in enumerate(pd.bdate_range("2020-01-01", periods=n_dates, freq="21B")):
            a = rng.normal(size=n)
            b = rng.normal(size=n)
            dead = di < dead_dates
            rows.append(pd.DataFrame({
                "date": d, "a_f": a, "b_f": (np.zeros(n) if dead else b),
                "label_rank": 0.3 * a + (0.0 if dead else 0.5) * b
                              + rng.normal(size=n)}))
        return pd.concat(rows, ignore_index=True)

    def test_a_constant_column_is_not_identified(self):
        from prosignal.features.famamacbeth import _ols_slopes
        rng = np.random.default_rng(1)
        n = 200
        a = rng.normal(size=n)
        y = 0.3 * a + rng.normal(size=n)
        for const in (0.0, 7.5, -3.0):
            beta = _ols_slopes(np.column_stack([a, np.full(n, const)]), y)
            assert np.isnan(beta[1]), (
                f"a column constant at {const} carries no cross-sectional "
                f"information; its slope is not identified and must not be "
                f"reported as {beta[1]}")

    def test_the_dead_column_does_not_disturb_its_neighbours(self):
        """The orthogonality claim the fix rests on, asserted rather than
        assumed: blanking a constant column must leave every other slope
        bit-identical to the regression that never had it."""
        from prosignal.features.famamacbeth import _ols_slopes
        rng = np.random.default_rng(1)
        n = 200
        a = rng.normal(size=n)
        y = 0.3 * a + rng.normal(size=n)
        joint = _ols_slopes(np.column_stack([a, np.zeros(n)]), y)
        alone = _ols_slopes(a.reshape(-1, 1), y)
        assert joint[0] == pytest.approx(alone[0], abs=1e-12)

    def test_the_theme_is_averaged_over_the_dates_that_measured_it(self):
        from prosignal.features.famamacbeth import fama_macbeth
        p = self._panel(dead_dates=5)
        full = fama_macbeth(p, ["a_f", "b_f"])
        live = sorted(p["date"].unique())[5:]
        trimmed = fama_macbeth(p[p["date"].isin(live)], ["a_f", "b_f"])
        assert full.n_dates_by_feature["b_f"] == full.n_dates - 5
        assert full.n_dates_by_feature["a_f"] == full.n_dates
        assert full.lam["b_f"] == pytest.approx(trimmed.lam["b_f"], abs=1e-12), (
            "the dead dates are still entering the mean as zeros")
        assert full.se["b_f"] == pytest.approx(trimmed.se["b_f"], abs=1e-12), (
            "the dead dates are still suppressing the dispersion the standard "
            "error is built from")

    def test_the_per_feature_count_is_reported(self):
        """A theme estimated on fewer cross-sections than its neighbours must
        not read as though it were not."""
        from prosignal.features.famamacbeth import fama_macbeth
        r = fama_macbeth(self._panel(dead_dates=7), ["a_f", "b_f"])
        assert r.n_dates_by_feature, "the per-feature counts are not reported"
        assert r.n_dates_by_feature["b_f"] < r.n_dates_by_feature["a_f"]


# =============================================================================
# W3 -- one column, one normalisation
# =============================================================================
class TestSectorRankIsOneQuantity:
    """The column carried a within-sector rank for 58% of rows and a UNIVERSE
    rank for the rest, and both were averaged into the same family aggregate
    and handed to the same regression. A within-sector +0.9 in a fourteen-name
    sector is not a universe +0.9."""

    @staticmethod
    def _case(seed=0, n=200, big=(60, 40), tiny=5):
        rng = np.random.default_rng(seed)
        v = pd.Series(rng.normal(size=n), index=[f"s{i}" for i in range(n)])
        labels = (["A"] * big[0] + ["B"] * big[1] + ["Tiny"] * tiny
                  + [None] * (n - sum(big) - tiny))
        return v, pd.Series(labels, index=v.index)

    def test_every_group_is_normalised_the_same_way(self):
        from prosignal.features.crosssec import sector_neutral_rank
        v, sec = self._case()
        r = sector_neutral_rank(v, sec)
        groups = {
            "A": r[sec == "A"], "B": r[sec == "B"],
            "residual": r[(sec.isna()) | (sec == "Tiny")],
        }
        spreads = {k: float(x.std()) for k, x in groups.items()}
        means = {k: float(x.mean()) for k, x in groups.items()}
        for k, m in means.items():
            assert abs(m) < 0.10, f"{k} mean rank {m:+.3f}; groups must centre alike"
        lo, hi = min(spreads.values()), max(spreads.values())
        assert hi - lo < 0.05, (
            f"group spreads {spreads} differ; a rank from one group is not "
            f"comparable with a rank from another, which is the defect")

        # THE DISCRIMINATING PART. The two assertions above pass on the OLD
        # code too -- an adversarial re-audit reverted `sector_neutral_rank` to
        # the universe-rank fallback and this test still went green, spreads
        # differing by only 0.013. Ranks are near-uniform either way; what
        # distinguishes the constructions is WHICH VALUES the residual pool
        # gets. Under the fallback its ranks are the universe's, computed over
        # every name; under the fix they are computed within the pool.
        from prosignal.features.crosssec import cross_sectional_rank
        resid_mask = (sec.isna()) | (sec == "Tiny")
        within_pool = cross_sectional_rank(v[resid_mask])
        universe = cross_sectional_rank(v)[resid_mask]
        assert not np.allclose(within_pool.to_numpy(), universe.to_numpy(), atol=1e-6), (
            "fixture is not discriminating: the two constructions agree")
        assert np.allclose(groups["residual"].to_numpy(),
                           within_pool.to_numpy(), atol=1e-12), (
            "the residual pool is ranked against the whole universe, so the "
            "column still carries two different normalisations")

    def test_the_unclassified_pool_is_ranked_within_itself(self):
        """Not against the universe. The discriminating case: an unsectored
        name that is mid-pack market-wide but top of the unclassified pool."""
        from prosignal.features.crosssec import sector_neutral_rank, cross_sectional_rank
        v, sec = self._case()
        resid = (sec.isna()) | (sec == "Tiny")
        got = sector_neutral_rank(v, sec)[resid]
        want = cross_sectional_rank(v[resid])
        assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-12), (
            "the residual pool is still carrying universe ranks")
        universe = cross_sectional_rank(v)[resid]
        assert not np.allclose(got.to_numpy(), universe.to_numpy(), atol=1e-6), (
            "the test cannot discriminate: within-pool and universe ranks agree")

    def test_coverage_is_reportable(self):
        """The 58% figure was only discoverable by instrumenting the code from
        outside. A property that decides what a whole column means must be
        readable from within."""
        from prosignal.features.crosssec import sector_rank_coverage
        _, sec = self._case()
        cov = sector_rank_coverage(sec)
        assert cov["n_sectors"] == 2, "Tiny is below the floor and is not a sector"
        assert cov["within_sector"] == pytest.approx(0.5, abs=1e-9)
        assert cov["unclassified"] == pytest.approx(0.5, abs=1e-9)

    def test_a_residual_pool_too_small_to_rank_is_the_only_mixed_case(self):
        from prosignal.features.crosssec import (MIN_SECTOR_NAMES,
                                                 cross_sectional_rank,
                                                 sector_neutral_rank)
        rng = np.random.default_rng(3)
        n = 80
        v = pd.Series(rng.normal(size=n), index=[f"s{i}" for i in range(n)])
        # one real sector plus a handful of orphans, fewer than the floor
        orphans = MIN_SECTOR_NAMES - 1
        sec = pd.Series(["A"] * (n - orphans) + [None] * orphans, index=v.index)
        r = sector_neutral_rank(v, sec)
        tail = r[sec.isna()]
        assert np.allclose(tail.to_numpy(), cross_sectional_rank(v)[sec.isna()].to_numpy(),
                           atol=1e-12), (
            "below the floor the residual pool keeps the universe rank; this is "
            "the one surviving mixed case and it is bounded by MIN_SECTOR_NAMES-1")


# =============================================================================
# W4 -- a single-theme arm under a gated estimator is not a comparable arm
# =============================================================================
class TestSingleThemeArmsAreNotSelectedOnThemselves:
    """Under the gated Fama-MacBeth a one-theme arm trades only on the splits
    where that theme cleared |t| >= 2 and holds cash on the rest, so its series
    is conditioned on its own in-sample significance while a multi-theme arm's
    is not. Any comparison between them compares selection regimes.

    NOTE THE DIRECTION. This correction flatters the production model, because
    the artifact made the single-theme CONTROLS look good. It is guarded, and
    every conclusion resting on such a matrix has to be re-derived."""

    def test_a_one_feature_arm_is_run_ungated(self):
        import inspect
        from prosignal.validation import harness
        src = inspect.getsource(harness.configuration_matrix)
        assert "arm_floor = 0.0 if len(cols) == 1 else None" in src
        assert "significance_floor=arm_floor" in src, (
            "the per-arm floor is computed and not passed to the fit")

    def test_the_treatment_is_declared_on_the_result(self):
        """Silently changing an arm's estimator would be a worse defect than
        the one it fixes."""
        import inspect
        from prosignal.validation import harness
        src = inspect.getsource(harness.configuration_matrix)
        assert 'frame.attrs["single_theme_ungated"]' in src, (
            "a reader of the matrix must be able to see which arms were "
            "treated differently and why")

    def test_the_flattering_direction_is_recorded_at_the_change(self):
        import inspect
        from prosignal.validation import harness
        src = inspect.getsource(harness.configuration_matrix)
        assert "FLATTERING DIRECTION" in src.upper(), (
            "a correction that can only help the system under audit must say "
            "so where it is made")


# =============================================================================
# W2 -- the selection-bias diagnostic (REPORTED, not traded: see
# work/audit/W2_failure_model.md; criterion 6 failed at m >= 3.0)
# =============================================================================
class TestSelectionBiasDiagnostic:
    """The gate keeps |t| >= floor and reports the survivor's lambda, so
    selection and estimation share a sample and the survivor is biased away
    from zero. This pins the correction's PROPERTIES; the decision not to trade
    it is recorded in the failure model, because it failed its own
    pre-committed simulation criterion where selection is barely binding."""

    def test_it_decays_to_nothing_far_from_the_boundary(self):
        from prosignal.features.famamacbeth import selection_corrected_t
        assert selection_corrected_t(20.0, 2.0) == pytest.approx(20.0, abs=1e-6)
        assert selection_corrected_t(10.0, 2.0) == pytest.approx(10.0, abs=1e-4)
        near = abs(selection_corrected_t(2.1, 2.0) - 2.1)
        far = abs(selection_corrected_t(6.0, 2.0) - 6.0)
        assert near > far, "the bias must be largest at the boundary"

    def test_it_only_ever_shrinks_toward_zero(self):
        from prosignal.features.famamacbeth import selection_corrected_t
        for t in (2.0, 2.3, 2.87, 3.5, 5.0, -2.1, -2.63, -4.0):
            c = selection_corrected_t(t, 2.0)
            assert abs(c) <= abs(t) + 1e-12, f"t {t} inflated to {c}"
            assert c == 0.0 or np.sign(c) == np.sign(t), f"t {t} changed sign to {c}"

    def test_an_unselected_theme_is_untouched(self):
        """Nothing was conditioned, so nothing is corrected."""
        from prosignal.features.famamacbeth import selection_corrected_t
        for t in (0.0, 0.9, 1.5, -1.8):
            assert selection_corrected_t(t, 2.0) == pytest.approx(t, abs=1e-12)

    def test_it_recovers_the_truth_where_selection_binds(self):
        """The criterion that decides whether the maths is right.

        TWO THINGS THIS TEST GOT WRONG, both found by an adversarial re-audit
        and both fixed here, because a criterion that cannot see the error it
        was written to catch is worse than none:

        * It conditioned on ``abs(draws) >= 2``. A surviving theme survived
          with a KNOWN SIGN and that sign is what gets traded, so the
          conditioning is one-sided. Averaging over both tails let the negative
          survivors' corrections cancel the positive ones and hid the defect.
        * Its grid omitted m = 0 -- the only value that matters for a
          false-discovery correction, and the one where the old two-sided
          estimator was worst: it reported a theme with NO true effect at all,
          scraping the gate on noise, as a genuine +1.0 sigma effect.

        Still fails at m >= 3.0, which is why the correction is reported and
        not traded. See work/audit/W2_failure_model.md.
        """
        from prosignal.features.famamacbeth import selection_corrected_t as corr
        rng = np.random.default_rng(7)
        for m in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            draws = rng.normal(m, 1.0, size=200_000)
            sel = draws[draws >= 2.0]                      # ONE-SIDED
            raw_err = abs(sel.mean() - m)
            fixed_err = abs(np.mean([corr(t, 2.0) for t in sel[:8000]]) - m)
            assert fixed_err < raw_err, (
                f"at true m {m} the correction ({fixed_err:.3f}) is no better "
                f"than the raw selected mean ({raw_err:.3f})")

    def test_a_theme_at_the_pure_selection_expectation_implies_no_effect(self):
        """The case the old two-sided version got exactly backwards.

        With floor 2.0, a theme with a true effect of ZERO that survives has
        E[t | t >= +2] = 2.373. Anything at or below that is fully explained by
        the selection and must correct to zero. The two-sided estimator
        returned +0.99 here."""
        from prosignal.features.famamacbeth import selection_corrected_t as corr
        for t in (2.0, 2.2, 2.373):
            assert corr(t, 2.0) == pytest.approx(0.0, abs=1e-9), (
                f"t {t} is at or below the pure-selection expectation of 2.373 "
                f"and implies no effect; got {corr(t, 2.0):+.3f}")
        assert corr(2.5, 2.0) > 0.0, "the test must still admit a real effect"

    def test_the_shipped_coefficients_are_not_corrected(self):
        """W2 is OPEN. If someone wires the diagnostic into the traded path,
        the failure model's ship rule has to be revisited first."""
        import inspect
        from prosignal.features import famamacbeth
        src = inspect.getsource(famamacbeth.gated_shrink)
        assert "selection_corrected" not in src, (
            "the selection correction failed criterion 6 of its own "
            "pre-committed ship rule; trading it requires re-deciding that, "
            "not quietly wiring it in")


# =============================================================================
# W1 -- an IC is a statement about the LABEL, and must say so
# =============================================================================
class TestTheTargetTravelsWithTheFigure:
    """The ranker is fitted against the h-session forward return; the book
    earns whatever resolve_exits produces. Within-date rank correlation between
    them is 0.531 -- about 28% of the variance -- and the book's positions exit
    by invalidation 39.3% of the time, by stop 32.1%, by target 17.9%, by
    timeout 10.7%. Every IC and excess in the repository is about the first
    quantity and was routinely read as though it were about the second."""

    def test_the_result_carries_its_target(self):
        from prosignal.validation.harness import CpcvResult
        r = CpcvResult(n_splits=1, n_paths=1)
        assert r.target == "label_rank"
        assert hasattr(r, "label_book_rank_corr")

    def test_the_gap_is_measured_when_the_panel_carries_the_outcome(self):
        """The field existed and NOTHING EVER ASSIGNED IT, so the CLI took its
        else-branch on every run and printed a hardcoded 0.531 as though it
        were that run's number. Behavioural, because the source-grep version of
        this guard broke on a wording change while the behaviour was fine --
        which is the wrong way round."""
        from prosignal.validation.harness import _label_book_rank_corr
        rng = np.random.default_rng(4)
        rows = []
        for d in pd.bdate_range("2021-01-01", periods=12, freq="21B"):
            lab = rng.normal(size=200)
            # a deliberately imperfect relative: rank corr well below 1
            book = 0.5 * lab + rng.normal(size=200)
            rows.append(pd.DataFrame({"date": d, "label": lab, "book_ret": book}))
        work = pd.concat(rows, ignore_index=True)
        got = _label_book_rank_corr(work)
        assert got is not None
        assert 0.2 < got < 0.8, f"expected a partial correlation, got {got}"

    def test_an_unmeasured_gap_returns_none_rather_than_a_constant(self):
        """'We did not check' and 'they agree' are different statements, and a
        constant presented as a measurement is worse than an omission because
        it cannot go stale visibly."""
        from prosignal.validation.harness import _label_book_rank_corr
        work = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=50),
                             "label": np.arange(50.0)})
        assert _label_book_rank_corr(work) is None


# =============================================================================
# C6 -- the README must not contradict the code or itself
# =============================================================================
class TestTheReadmeAgreesWithHead:
    """The README carried three mutually inconsistent CPCV results, described
    an estimator the engine does not use, and listed a holdout figure under
    'Reasonably supported' four hundred lines after declaring the same holdout
    withdrawn. Prose drifts silently; these are the claims worth pinning."""

    @staticmethod
    def _readme():
        from pathlib import Path
        import prosignal
        return (Path(prosignal.__file__).parent.parent.parent / "README.md").read_text()

    def test_the_readme_does_not_claim_a_table_describes_what_ships(self):
        """This asserted that RESULTS OF RECORD "supersede every other number
        in this file". That claim was withdrawn on 2026-09-05, and the test is
        strengthened rather than deleted.

        The two headline tables measure two different models with two different
        exit geometries: the executive summary is `mom_6_1_r` traded alone, and
        RESULTS OF RECORD is the fitted Fama-MacBeth composite with a 2.5xATR
        stop, a 3R target and risk-budget sizing. The engine ranks on
        `v3_composite` and uses none of those exits. So neither table is the
        live one, and a README that nominates either is more misleading than one
        that nominates neither.

        What must remain true: both are still present (a bad result is never
        deleted here) and the file says plainly that neither describes what
        ships."""
        r = self._readme()
        assert "## RESULTS OF RECORD" in r, "the table itself must not be deleted"
        assert "NEITHER TABLE IN THIS FILE DESCRIBES WHAT SHIPS" in r, (
            "the README must say which model each table measured, or a reader "
            "takes one of them for the shipped configuration"
        )
        assert "supersede every other number in this file" not in r, (
            "the withdrawn claim is back"
        )

    def test_superseded_numbers_are_marked_not_deleted(self):
        """Never delete a bad result. Mark it."""
        r = self._readme()
        assert r.count("SUPERSEDED") >= 2
        # the old passing DSR is still visible, and labelled
        assert "Deflated Sharpe, charging 44 trials | 1.000, pass" in r
        assert "0.346 — FAIL" in r

    def test_it_names_the_estimator_that_actually_ships(self):
        r = self._readme()
        from prosignal.config.loader import load_config
        method = str(load_config().params.stage4_core_score.estimator.method)
        assert method == "fama_macbeth"
        head = r[:r.index("## RESULTS OF RECORD")]
        assert "Fama–MacBeth" in head or "Fama-MacBeth" in head, (
            "the executive summary described ridge regression while the engine "
            "runs Fama-MacBeth")
        assert "Ridge regression on **5 factor families**" not in r

    def test_the_within_sector_claim_matches_the_code(self):
        r = self._readme()
        assert "All ranks are taken within sector**" not in r, (
            "that sentence was false for roughly half of every cross-section")
        assert "UNCLASSIFIED" in r

    def test_the_withdrawn_holdout_is_not_also_supported(self):
        r = self._readme()
        assert "**Withdrawn**" in r
        supported = r[r.index("**Reasonably supported**"):r.index("**Preliminary**")]
        assert "+4.35%" not in supported, (
            "the holdout excess is listed as supported while the executive "
            "summary declares the same holdout withdrawn")

    def test_the_book_result_is_stated_not_buried(self):
        r = self._readme()
        for claim in ("−4.23%", "−0.83", "information ratio"):
            assert claim in r, f"the README does not state {claim}"


# =============================================================================
# C5 -- the forward test must ask the question the audit turned on
# =============================================================================
class TestForwardTestIsBenchmarkRelative:
    """The first registration had no benchmark-relative hypothesis, and
    neither did any other code path in the repository: every economic
    conclusion was stated against zero. Adding one is legitimate only because
    the window has not opened."""

    def test_a_tertiary_benchmark_hypothesis_exists(self):
        import datetime as dt
        import tempfile
        from pathlib import Path
        from prosignal.validation.forward import register
        with tempfile.TemporaryDirectory() as d:
            reg = register(Path(d), config_version="v", engine_version="e",
                           git_commit="c", started_on=dt.date(2026, 1, 1),
                           unchecked_reason="unit test: no config in scope")
        assert reg.tertiary, "there is no benchmark-relative hypothesis"
        t = reg.tertiary.upper()
        assert "EQUAL-WEIGHT" in t and "EXCESS" in t
        assert "expected to FAIL" in reg.tertiary, (
            "the registration must state the prior honestly; a forward test "
            "whose outcome is not in doubt is not a test")

    def test_the_hypotheses_are_inside_the_fingerprint(self):
        """Otherwise a criterion could be edited after the result lands."""
        import datetime as dt
        import tempfile
        from pathlib import Path
        from dataclasses import replace
        from prosignal.validation.forward import register
        with tempfile.TemporaryDirectory() as d:
            reg = register(Path(d), config_version="v", engine_version="e",
                           git_commit="c", started_on=dt.date(2026, 1, 1),
                           unchecked_reason="unit test: no config in scope")
        assert replace(reg, tertiary="anything else").fingerprint() != reg.fingerprint()

    def test_an_unavailable_benchmark_voids_the_window(self):
        import datetime as dt
        import tempfile
        from pathlib import Path
        from prosignal.validation.forward import register
        with tempfile.TemporaryDirectory() as d:
            reg = register(Path(d), config_version="v", engine_version="e",
                           git_commit="c", started_on=dt.date(2026, 1, 1),
                           unchecked_reason="unit test: no config in scope")
        assert any("benchmark" in i.lower() for i in reg.invalidation)


# =============================================================================
# Findings from the ADVERSARIAL RE-AUDIT of this remediation.
# Every one of these is a defect the remediation itself introduced or left.
# =============================================================================
class TestAdversarialFindings:

    def test_the_cached_live_path_masks_the_admissible_population(self):
        """`model_refit_every_sessions` is 21, so `fit_predict` -- where the
        admissible mask was wired -- runs on 1 session in 21. Every other day
        came through `today_features`, which called `features_for_date` with no
        `admissible` argument and ranked over the wider population the model
        was not fitted on. Fixing the rare path and leaving the common one is
        worse than not fixing it: the two then disagree."""
        import inspect
        from prosignal.features import crossmodel
        src = inspect.getsource(crossmodel.today_features)
        assert "admissible=live_adm" in src, (
            "the cached live path, which runs 20 days in 21, does not apply "
            "the admissible mask")
        assert "_admissible_frame" in src

    def test_a_result_without_its_date_keys_refuses_rather_than_reverts(self):
        """`excess_by_date` defaulted to {} and `horizon_sessions` to 0, so a
        result built without them silently returned the POOLED vector and the
        raw count -- restoring the exact defect the methods exist to remove.
        Measured on a 540-entry/60-date fixture: effective_n 20.3 -> 540 and
        DSR 0.878 FAIL -> 1.0000 PASS, with no error and no note."""
        from prosignal.validation.harness import CpcvResult
        bare = CpcvResult(n_splits=9, n_paths=3, excess=[0.01] * 540)
        with pytest.raises(ValueError, match="excess_by_date"):
            bare.excess_per_date()
        keyed = CpcvResult(n_splits=9, n_paths=3,
                           excess_by_date={i: [0.01] for i in range(60)})
        with pytest.raises(ValueError, match="step_sessions|horizon"):
            keyed.effective_observations()

    def test_a_missing_adjustment_factor_does_not_delist_a_name(self):
        """Dividing by NaN excluded the name outright -- a factor frame missing
        one column dropped that symbol on EVERY date. Worse than the bug being
        fixed, and running the same direction: names vanish from the past for a
        data reason. `universe.resolve_liquidity_pit` already did the right
        thing; the two paths now agree."""
        from prosignal.features.crosssec import liquidity_mask
        close, turnover, _h, _l = _frames(n=400)
        kw = dict(min_adtv_inr=1e6, lookback_sessions=21, max_names=len(SYMS),
                  min_history_sessions=50, min_price_inr=10.0)
        base = liquidity_mask(close, turnover, **kw)
        fac = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        fac = fac.drop(columns=[close.columns[0]])          # one column absent
        got = liquidity_mask(close, turnover, adj_factor=fac, **kw)
        assert got[close.columns[0]].sum() == base[close.columns[0]].sum(), (
            "a symbol with no adjustment factor was delisted entirely")
        sparse = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        sparse.iloc[10:20, 1] = np.nan
        got2 = liquidity_mask(close, turnover, adj_factor=sparse, **kw)
        assert got2.equals(base), "scattered NaN factors changed the screen"

    def test_a_weighted_fit_respects_the_effective_row_count(self):
        """The n <= p + 1 guard read the RAW row count while weights were
        applied afterwards, so 60 rows with non-zero weight on 3 of them passed
        a check the same 3 rows fail unweighted, and lstsq returned four
        fabricated slopes from an underdetermined system."""
        from prosignal.features.famamacbeth import _ols_slopes
        rng = np.random.default_rng(0)
        n = 60
        x = rng.normal(size=(n, 4))
        y = rng.normal(size=n)
        w = np.zeros(n)
        w[:3] = 1.0
        assert _ols_slopes(x, y, w=w) is None, (
            "an underdetermined weighted fit returned slopes")
        assert _ols_slopes(x[:3], y[:3]) is None, "control: unweighted refuses"

    def test_liveness_is_judged_on_the_weights_the_fit_used(self):
        """`live` tested the RAW weights while the fit used the sanitised ones,
        so a column varying only at an inf-weight row -- sanitised to zero, and
        contributing nothing -- was judged live and its slope of -3e-17 entered
        the average. That is the dead-cross-section failure reintroduced inside
        the fix for it."""
        from prosignal.features.famamacbeth import _ols_slopes
        rng = np.random.default_rng(0)
        n = 60
        x = rng.normal(size=(n, 4))
        y = rng.normal(size=n)
        x[:, 1] = 0.0
        x[0, 1] = 5.0
        w = np.ones(n)
        w[0] = np.inf
        beta = _ols_slopes(x, y, w=w)
        assert beta is not None
        assert np.isnan(beta[1]), (
            f"a column live only at a zero-weight row was measured as "
            f"{beta[1]}")

    def test_the_benchmark_is_the_population_the_book_selects_from(self):
        """Built over every symbol in the store it is a wider, less liquid
        population than the book's -- and it carries the one conclusion that
        reverses the verdict, so a mislabelled benchmark would be the worst
        possible place for this error."""
        import inspect
        from prosignal import cli
        src = inspect.getsource(cli._portfolio_inputs)
        assert "liquidity_mask" in src, (
            "the benchmark is computed over the whole store while every caller "
            "describes it as the eligible universe")
        assert ".shift(1)" in src, (
            "the screen must be lagged: membership on date t decided by data "
            "through t-1")

    def test_training_and_live_ranks_are_computed_at_the_same_precision(self):
        """build_panel cast to float32 and THEN ranked; features_for_date ranks
        float64. Near-ties therefore ordered differently in training and at the
        decision -- inside the function whose docstring promises the two cannot
        drift apart in definition."""
        import inspect
        from prosignal.features import crosssec
        src = inspect.getsource(crosssec.build_panel)
        cast = src.index('astype("float32")')
        rank = src.index("sector_neutral_rank(g[f]")
        assert cast > rank, (
            "the float32 cast still runs before the ranking; training ranks "
            "are taken at a different precision from live ranks")


def test_effective_n_actually_changes_the_deflated_sharpe():
    """SURVIVED A MUTATION. Replacing

        n_eff = float(n) if effective_n is None else float(min(effective_n, n))

    with `n_eff = float(n)` -- i.e. ignoring the caller's effective count
    entirely -- broke no test. Every guard reached the parameter through
    `CpcvResult.deflated()`, which supplies it, so they exercised the wiring
    and never the arithmetic: the DSR could quietly go back to reading
    duplicated, overlapping observations as independent ones.

    The discriminating pair is the same return series scored with and without a
    declared effective count."""
    from prosignal.validation.metrics import deflated_sharpe_ratio
    rng = np.random.default_rng(5)
    r = list(rng.normal(0.012, 0.05, 600))

    naive = deflated_sharpe_ratio(r, n_trials=50, sr_variance=0.005)
    honest = deflated_sharpe_ratio(r, n_trials=50, effective_n=24.0,
                                   sr_variance=0.005)

    assert honest.effective_n == pytest.approx(24.0)
    assert naive.effective_n == pytest.approx(600.0)
    assert honest.deflated_sr < naive.deflated_sr, (
        f"declaring 24 independent observations out of 600 gave the SAME "
        f"deflated Sharpe ({honest.deflated_sr:.4f} vs {naive.deflated_sr:.4f}); "
        f"the effective count is being ignored")
    assert honest.n_observations == 600, "the raw count is still reported"
    # and a caller cannot manufacture independence it does not have
    assert deflated_sharpe_ratio(r, n_trials=50, effective_n=10_000).effective_n \
        == pytest.approx(600.0)


def test_the_psr_itself_honours_the_effective_count():
    """The half of the previous test that was still uncovered.

    `deflated_sharpe_ratio` uses its effective count TWICE -- once for the
    null variance and once by passing it down to `probabilistic_sharpe_ratio`
    for the sqrt(n-1) term. Breaking only the PSR's copy left every guard
    green, because the DSR's own use of the count still moved the answer
    enough to satisfy them. So the PSR is asserted directly: its whole job here
    is that a series of 600 values carrying 24 independent observations must
    not be read as 600 independent ones."""
    from prosignal.validation.metrics import probabilistic_sharpe_ratio
    rng = np.random.default_rng(5)
    r = list(rng.normal(0.012, 0.05, 600))

    naive = probabilistic_sharpe_ratio(r, benchmark_sr=0.10)
    honest = probabilistic_sharpe_ratio(r, benchmark_sr=0.10, effective_n=24.0)
    assert honest < naive - 1e-6, (
        f"declaring 24 independent observations out of 600 gave PSR "
        f"{honest:.6f} against {naive:.6f}; the sqrt(n-1) term is still "
        f"reading the raw length")
    # monotone in the declared count, and never inflated past the raw length
    mid = probabilistic_sharpe_ratio(r, benchmark_sr=0.10, effective_n=100.0)
    assert honest < mid < naive + 1e-12
    assert probabilistic_sharpe_ratio(r, benchmark_sr=0.10, effective_n=10_000) \
        == pytest.approx(naive, abs=1e-12), (
        "a caller cannot declare more independence than the series has")
