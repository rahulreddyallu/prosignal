"""Guards for the measurement defects found in the August/September re-audit.

Each test names the defect it prevents and the scenario that would fail without
the fix. Every one of them was written by reproducing the defect first and
watching the test go red, then applying the fix and watching it go green -- a
guard that has never failed is a guard nobody has checked.

The theme is the same throughout, and it is the theme of the whole engine: a
number that reads like a measurement and is not one. A Deflated Sharpe computed
on nine copies of each observation. A trial count with no trial scores. A flag
called `sacred` that nothing read. A pre-registration with no hypothesis about
whether running the engine beats not running it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prosignal.validation.forward import Registration, load_registration, progress
from prosignal.validation.harness import CpcvResult
from prosignal.validation.metrics import (SR_VAR_FROM_TRIALS, SR_VAR_UNIT,
                                          deflated_sharpe_ratio)
from prosignal.validation.registry import TrialRegistry


# =============================================================================
# The Deflated Sharpe
# =============================================================================


def test_the_dsr_fallback_variance_is_a_unit_not_one_over_n():
    """The defect that made the multiple-testing defence unable to fail.

    `deflated_sharpe_ratio` documented "a conservative unit variance" and used
    `1/(n-1)`. At the 612 pooled observations a CPCV run produces that is
    0.0016, an expected-maximum benchmark of 0.099, and a DSR of 1.0000 for any
    positive result at any trial count -- including 100,000.

    Fails if the fallback goes back to scaling with n: the two calls below
    differ only in length and must charge the same benchmark.
    """
    rng = np.random.default_rng(11)
    short = rng.normal(0.01, 0.05, 30)
    long = np.repeat(short, 20)              # same distribution, 20x the count

    a = deflated_sharpe_ratio(short, n_trials=81)
    b = deflated_sharpe_ratio(long, n_trials=81)
    assert a.sr_variance_source == SR_VAR_UNIT
    assert a.sr_variance == pytest.approx(1.0)
    assert a.benchmark_sr == pytest.approx(b.benchmark_sr, rel=1e-9), (
        "the multiple-testing benchmark moved because the OBSERVATION count "
        "changed; it must depend on the TRIAL count and the dispersion of "
        "trial scores, and on nothing else"
    )


def test_the_dsr_says_where_its_variance_came_from():
    """The single most load-bearing input, previously invisible."""
    x = np.random.default_rng(3).normal(0.01, 0.05, 40)
    # SAME trial count on both sides: the only thing that differs is whether
    # the trials' own scores were available. Varying n_trials too would make
    # the comparison below true whatever the variance did.
    unit = deflated_sharpe_ratio(x, n_trials=5)
    trials = deflated_sharpe_ratio(x, n_trials=5, trial_sharpes=[0.1, 0.2, 0.15,
                                                                 0.05, 0.3])
    assert unit.sr_variance_source == SR_VAR_UNIT
    assert trials.sr_variance_source == SR_VAR_FROM_TRIALS
    assert trials.benchmark_sr < unit.benchmark_sr, (
        "a measured trial dispersion smaller than 1.0 must lower the bar; if "
        "it does not, the variance is not reaching the benchmark"
    )
    for r in (unit, trials):
        assert r.sr_variance_source in r.interpretation or str(
            round(r.sr_variance, 4)) in r.interpretation


def test_a_degenerate_variance_falls_back_rather_than_passing_everything():
    x = np.random.default_rng(5).normal(0.01, 0.05, 40)
    r = deflated_sharpe_ratio(x, n_trials=81, sr_variance=float("nan"))
    assert r.sr_variance_source == SR_VAR_UNIT
    assert r.sr_variance == pytest.approx(1.0)


# =============================================================================
# The observations the DSR is fed
# =============================================================================


def _result(dates, per_date=9):
    """A CpcvResult shaped the way `run_cpcv` shapes one: every date scored
    once per split that held it out."""
    r = CpcvResult(n_splits=45, n_paths=9)
    rng = np.random.default_rng(7)
    for d in dates:
        base = float(rng.normal(0.01, 0.04))
        for _ in range(per_date):
            r.excess.append(base + float(rng.normal(0, 0.002)))
            r.excess_dates.append(d)
    return r


def test_the_dsr_scores_independent_windows_not_duplicated_pairs():
    """612 entries describing 70 dates describing 23 independent windows.

    `CpcvResult.deflated` used to hand `excess` straight to the DSR. With
    N=10, k=2 each date is tested in nine splits, so the count was inflated
    ninefold before the overlap between neighbouring dates was even considered.

    Fails if `deflated` stops reducing.
    """
    dates = list(range(70))
    res = _result(dates)
    assert len(res.excess) == 630

    indep = res.independent_excess(horizon_sessions=63, step_sessions=21)
    assert len(indep) == 24, (
        "70 dates, one score each, then every third for the 63/21 overlap"
    )
    d = res.deflated(n_trials=81, horizon_sessions=63, step_sessions=21)
    assert d.n_observations == len(indep)
    assert d.n_observations < len(res.excess) / 20


def test_deflated_refuses_to_guess_the_sampling_scheme():
    """Both arguments are keyword-only and required.

    A default would let a caller get the old pooled answer by omission, which
    is exactly how the defect survived: the call site read
    `result.deflated(n_trials=trials)` and looked complete.
    """
    res = _result(list(range(20)))
    with pytest.raises(TypeError):
        res.deflated(n_trials=81)          # type: ignore[call-arg]


def test_excess_and_its_dates_cannot_drift_apart_silently():
    res = _result(list(range(10)))
    res.excess.append(0.05)               # a score with no date
    with pytest.raises(ValueError, match="drifted apart"):
        res.independent_excess(63, 21)


# =============================================================================
# The trial registry
# =============================================================================


def test_the_registry_records_what_each_trial_scored(tmp_path: Path):
    """Counting the denominator and discarding the numerator.

    The DSR needs the trial COUNT and the trial DISPERSION. The registry
    supplied the first and nothing supplied the second, so every DSR had to
    assume Var[SR]. On this engine's evidence, moving between two defensible
    assumptions moved the answer from 0.38 FAIL to 0.91.
    """
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record("research estimator", ["ridge", "fama_macbeth", "ols"],
               scores=[0.11, 0.24, 0.07])
    assert reg.count() == 3
    assert sorted(reg.recorded_scores()) == pytest.approx([0.07, 0.11, 0.24])

    # And it round-trips through the file, not just through memory.
    assert sorted(TrialRegistry(tmp_path / "trials.jsonl").recorded_scores()
                  ) == pytest.approx([0.07, 0.11, 0.24])


def test_scores_that_do_not_line_up_with_labels_are_refused(tmp_path: Path):
    """A registry whose scores belong to the wrong arms is worse than one with
    no scores, because the variance computed from it looks like evidence."""
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="correspond one to one"):
        reg.record("research spread", ["a", "b", "c"], scores=[0.1, 0.2])


def test_a_registry_without_scores_still_counts(tmp_path: Path):
    """Backwards compatible: the trials already on disk carry no score and
    must not vanish from the count."""
    p = tmp_path / "trials.jsonl"
    p.write_text(json.dumps({"key": "abc", "command": "old", "label": "x",
                             "recorded_at": "2026-01-01T00:00:00+00:00"}) + "\n")
    reg = TrialRegistry(p)
    assert reg.count() == 1
    assert reg.recorded_scores() == []


# =============================================================================
# The holdout
# =============================================================================


def test_sacred_holdout_refuses_include_holdout():
    """`validation.holdout.sacred` shipped true and was read by NOTHING.

    Eight research commands each carried their own `sessions[-1] if
    args.include_holdout` with a printed warning and no check, so the flag that
    names the holdout untouchable was decoration.
    """
    import argparse

    from prosignal.cli import _selection_end
    from prosignal.config.loader import load_config
    from prosignal.core.errors import ConfigError

    cfg = load_config()
    sessions = list(range(1000))
    args = argparse.Namespace(include_holdout=False)
    reserve = int(cfg.params.validation.holdout.reserve_most_recent_sessions.value)
    assert _selection_end(cfg, args, sessions) == sessions[-reserve]

    if not bool(cfg.params.validation.holdout.sacred):
        pytest.skip("the shipped config no longer marks the holdout sacred")
    args = argparse.Namespace(include_holdout=True)
    with pytest.raises(ConfigError, match="sacred"):
        _selection_end(cfg, args, sessions)


def test_no_research_command_reads_the_holdout_by_its_own_arithmetic():
    """One reader, so the rule cannot be enforced in seven places and missed in
    the eighth."""
    src = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "cli.py"
           ).read_text(encoding="utf-8")
    body = src.split("def _selection_end", 1)[1].split("\ndef ", 1)[1]
    assert "args.include_holdout" not in body, (
        "a command is deciding the holdout boundary for itself again; every "
        "one of them must go through _selection_end or the sacred flag is "
        "enforced everywhere except there"
    )


# =============================================================================
# The forward test
# =============================================================================


def _reg(**over):
    base = dict(started_on="2026-08-27", config_version="baseline-v1@aaaa",
                engine_version="0.1.0", git_commit="deadbeef",
                target_sessions=375, target_months=18,
                primary="P", secondary="S", tertiary="T", invalidation=[])
    base.update(over)
    return Registration(**base)


def test_the_tertiary_hypothesis_is_inside_the_fingerprint():
    """Otherwise it can be added, softened or deleted after a result lands."""
    a = _reg(tertiary="beats the equal-weight eligible universe at t >= 2.0")
    b = _reg(tertiary="beats the equal-weight eligible universe at t >= 0.5")
    assert a.fingerprint() != b.fingerprint()


def test_a_registration_with_no_benchmark_hypothesis_is_not_graded(tmp_path: Path):
    """`primary` regresses on factors and `secondary` scores the ranking.

    Neither asks whether the book beats holding the universe it selects from,
    and on the selection period it does not -- by about 3.9 points per period.
    A window carrying only those two could be passed by an engine that loses to
    buying everything, so it is marked broken rather than graded.
    """
    reg = _reg(tertiary="")
    payload = {k: v for k, v in reg.__dict__.items()}
    payload["fingerprint"] = reg.fingerprint()
    (tmp_path / "forward_test.json").write_text(json.dumps(payload))

    prog = progress(tmp_path, [], live_config_version=reg.config_version)
    assert prog is not None
    assert any("benchmark-relative" in b for b in prog.broken), prog.broken
    assert not prog.complete


def test_the_shipped_registration_text_names_the_benchmark():
    """The hypothesis `register` writes must actually be the one that matters."""
    import inspect

    from prosignal.validation import forward

    src = inspect.getsource(forward.register)
    assert "tertiary=" in src
    low = src.lower()
    assert "equal-weight" in low and "eligible universe" in low
    assert "expected to fail" in low, (
        "a forward test whose outcome is not in doubt is not a test; the "
        "registration should say which way it is expected to go"
    )


def test_run_cpcv_records_the_date_of_every_score():
    """`independent_excess` cannot reduce what it cannot identify.

    Fails if `run_cpcv` stops appending to `excess_dates`: the two lists then
    have different lengths and every DSR downstream is either wrong or an
    exception, depending on which the reader notices first.
    """
    import numpy as np
    import pandas as pd

    from prosignal.validation.harness import run_cpcv

    rng = np.random.default_rng(4)
    rows = []
    dates = pd.bdate_range("2021-01-04", periods=40, freq="21B")
    for d in dates:
        n = 120
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        lab = 0.02 * f1 + rng.normal(0, 0.05, n)
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"S{i:03d}" for i in range(n)],
            "f1": f1, "f2": f2, "label": lab,
            "label_rank": pd.Series(lab).rank(pct=True).to_numpy()}))
    panel = pd.concat(rows, ignore_index=True)

    res = run_cpcv(panel, ["f1", "f2"], horizon_sessions=63, step_sessions=21,
                   alpha=1.0, n_groups=6, n_test_groups=2, purge_sessions=63,
                   embargo_sessions=21, min_train_rows=200, estimator="ridge")
    assert res.excess, "the fixture scored no test date"
    assert len(res.excess_dates) == len(res.excess), (
        "every score must carry the date it belongs to, or the duplication "
        "CPCV introduces cannot be undone downstream"
    )
    assert len(set(res.excess_dates)) < len(res.excess), (
        "the fixture is meant to test each date in several splits; if it does "
        "not, it cannot demonstrate the reduction"
    )
    d = res.deflated(n_trials=20, horizon_sessions=63, step_sessions=21)
    assert d.n_observations < len(res.excess)


def test_an_existing_trial_can_gain_a_score_but_not_a_second_count(tmp_path: Path):
    """The trials already on disk were recorded before scores existed.

    `record` is idempotent by (command, label), so without a supplementary path
    they could never acquire a score and Var[SR] would stay unmeasurable
    forever on the very campaign the DSR is charging for. Filling in a MISSING
    score must not change the count, and an existing score must never be
    overwritten -- a re-run that scored differently is a second measurement,
    not a correction, and replacing the first with it lets a disappointing arm
    be re-rolled.
    """
    p = tmp_path / "trials.jsonl"
    reg = TrialRegistry(p)
    assert reg.record("research spread", ["a", "b"]) == 2
    assert reg.count() == 2 and reg.recorded_scores() == []

    assert reg.record("research spread", ["a", "b"], scores=[0.4, 0.7]) == 0, (
        "filling in a score is not a new configuration and must not be counted"
    )
    assert reg.count() == 2
    assert sorted(reg.recorded_scores()) == pytest.approx([0.4, 0.7])

    # A later, different score is ignored: the first stands. Written STRAIGHT
    # TO THE FILE, because `record` has its own guard and testing through it
    # cannot tell whether `load`'s merge rule is also correct -- the file is
    # append-only and a hand-edited or concurrently-written line has to be
    # handled by the reader, not only by the writer.
    with p.open("a", encoding="utf-8") as fh:
        for label, bogus in (("a", 9.9), ("b", -9.9)):
            fh.write(json.dumps({
                "key": TrialRegistry.key_for("research spread", label),
                "command": "research spread", "label": label,
                "recorded_at": "2026-12-31T00:00:00+00:00",
                "score": bogus}) + "\n")
    assert sorted(TrialRegistry(p).recorded_scores()) == pytest.approx([0.4, 0.7]), (
        "a later line replaced a score that was already recorded; a re-run "
        "that scored differently is a second measurement, not a correction, "
        "and letting it overwrite lets a disappointing arm be re-rolled"
    )
    assert TrialRegistry(p).count() == 2

    # And a genuinely new arm still counts.
    assert reg.record("research spread", ["c"], scores=[0.1]) == 1
    assert reg.count() == 3


def test_trial_scores_from_a_fraction_of_the_search_do_not_set_the_bar():
    """Measured Var[SR] is authoritative only when it describes the search.

    A command that sweeps eighteen buy/hold bands records eighteen
    near-identical Sharpes, while the genuinely different ideas were compared
    once and moved on from. Estimating the dispersion of 87 configurations from
    the 18 most similar of them gave Var[SR] 0.00178 against a conservative
    1.0 -- an expected-maximum bar 24x lower, and a comfortable PASS.

    Fails if the coverage rule is removed: the same scores then set the bar.
    """
    from prosignal.validation.metrics import (MIN_TRIAL_SCORE_COVERAGE,
                                              SR_VAR_UNDERCOVERED)

    x = np.random.default_rng(21).normal(0.02, 0.05, 40)
    tight = [0.30, 0.31, 0.29, 0.30, 0.32, 0.31]     # one sweep, barely varying

    thin = deflated_sharpe_ratio(x, n_trials=87, trial_sharpes=tight)
    assert thin.sr_variance_source == SR_VAR_UNDERCOVERED
    assert thin.sr_variance == pytest.approx(1.0)
    assert thin.sr_variance_measured < 0.01, "the fixture is not tight enough"
    assert thin.trials_scored == len(tight)
    assert str(len(tight)) in thin.interpretation

    # Enough coverage and the measurement is used.
    n = 10
    covered = deflated_sharpe_ratio(x, n_trials=n, trial_sharpes=tight)
    assert len(tight) / n >= MIN_TRIAL_SCORE_COVERAGE
    assert covered.sr_variance == pytest.approx(thin.sr_variance_measured)
    assert covered.benchmark_sr < thin.benchmark_sr
