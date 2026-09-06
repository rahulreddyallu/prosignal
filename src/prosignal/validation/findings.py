"""The defect register, as data rather than as a document.

WHY THIS IS CODE. A finding written only in a report has no way to stop
anything. The readiness dossier listed fourteen findings as fixed and four of
them were not present in the source; nothing detected that, because nothing
could -- the claim and the code lived in different files and only one of them
ran. Here each finding carries the classification, the disposition and the
restart consequence the review asked for, and `readiness` reads them to decide
whether a forward test may open. A finding that blocks a restart blocks it by
being in this list, not by being remembered.

WHAT EACH ENTRY MUST CARRY, and why the dataclass has no defaults for them:
finding, root cause, code location, fix, regression test, before/after,
whether coefficients moved, whether historical results moved, and whether the
forward test must restart. "Reviewed" is not a status. A finding whose
regression test is empty cannot be marked FIXED -- `_validate` raises on
import, so the register cannot be quietly weakened.

CATEGORIES are the review's, unchanged: DATA_INTEGRITY, UNIVERSE, FEATURE,
MODEL, EXECUTION, RISK, VALIDATION, REPRODUCIBILITY, UI. The classification is
not decoration -- it is what says which of them can move a traded number, and
therefore which of them force a refit rather than a redeploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Category", "Status", "Finding", "REGISTER", "by_id", "open_findings",
    "restart_blockers", "unresolved_restart_blockers", "categorised",
]


class Category(str, Enum):
    DATA_INTEGRITY = "DATA INTEGRITY"
    UNIVERSE = "UNIVERSE"
    FEATURE = "FEATURE"
    MODEL = "MODEL"
    EXECUTION = "EXECUTION"
    RISK = "RISK"
    VALIDATION = "VALIDATION"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    UI = "UI"


class Status(str, Enum):
    #: Corrected in the source, with a test that fails if the correction is
    #: reverted, and that reversion was performed and observed to go red.
    FIXED = "FIXED"
    #: The capability exists and is tested, but the switch ships off because
    #: turning it on moves a traded number and that belongs with a refit.
    BUILT_OFF = "BUILT, OFF"
    #: Measured and disclosed. Nothing in the code changed because the finding
    #: is about what a number means, not about what it computes.
    DISCLOSED = "DISCLOSED"
    #: Understood, priced, and deliberately not actioned here -- because the
    #: action is a decision for the operator rather than a defect repair.
    DEFERRED = "DEFERRED"
    #: Not addressed.
    OPEN = "OPEN"


@dataclass(frozen=True)
class Finding:
    fid: str
    title: str
    category: Category
    status: Status
    #: Why it was wrong, not what was wrong. The distinction matters: two
    #: findings with the same symptom and different root causes need different
    #: fixes, and the register is what stops them being closed together.
    root_cause: str
    #: Where. `module::symbol`, so it survives line renumbering.
    location: str
    fix: str
    #: The test that goes red if the fix is reverted. Empty is only legal for
    #: a finding that is not claimed to be fixed.
    regression_test: str
    before_after: str
    #: Does closing this move a fitted coefficient? If so it forces a refit,
    #: and every result computed under the old fit is superseded.
    moves_coefficients: bool
    #: Does closing this change a number already published?
    moves_history: bool
    #: Must the forward test restart before this engine can be graded?
    forces_restart: bool
    severity: str = "medium"
    notes: str = ""

    @property
    def resolved(self) -> bool:
        """Resolved means "not waiting on anything", not "fixed".

        A finding that was measured and consciously deferred is resolved for
        the purpose of opening a research epoch: the decision has been taken.
        One still OPEN is not.
        """
        return self.status is not Status.OPEN

    def describe(self) -> str:
        flags = []
        if self.moves_coefficients:
            flags.append("refit")
        if self.moves_history:
            flags.append("history moves")
        if self.forces_restart:
            flags.append("RESTART")
        tail = f"  [{', '.join(flags)}]" if flags else ""
        return (f"{self.fid} {self.category.value} {self.status.value}: "
                f"{self.title}{tail}")


def _f(**kw) -> Finding:
    return Finding(**kw)


#: Ordered as the review ordered them. R-numbers are this pass's own audit;
#: W- and C- numbers are the readiness dossier's.
REGISTER: Tuple[Finding, ...] = (
    _f(
        fid="R1", severity="critical",
        title="The forward test is registered against a configuration the "
              "engine no longer runs",
        category=Category.VALIDATION, status=Status.OPEN,
        root_cause="`register` freezes a config_version and `progress` voids "
                   "the window when a run carries a different one, but nothing "
                   "prevents the config from changing -- so the only stated "
                   "route to READY silently became a void window rather than a "
                   "failed one.",
        location="prosignal.validation.forward::register",
        fix="Not fixed here. Restarting a clock is an operator decision, and "
            "restarting it while the findings below are open would void the "
            "new window for the same reason the old one is void. The gate in "
            "`validation.readiness` now refuses the restart until the "
            "preconditions hold, and names which ones do not.",
        regression_test="tests/test_restart_gate.py",
        before_after="`--restart` used to overwrite the registration "
                     "unconditionally; it now refuses while a restart-blocking "
                     "finding is open, the data manifest is unverified, or the "
                     "epoch identity is incomplete.",
        moves_coefficients=False, moves_history=False, forces_restart=True,
        notes="Restart LAST. It is the consumer of every other finding here.",
    ),
    _f(
        fid="R2", severity="critical",
        title="No benchmark-relative hypothesis existed anywhere in the "
              "pre-registration",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="The registration asked whether the book has alpha against "
                   "factors and whether the ranking has IC. Neither asks "
                   "whether it beats holding the universe it selects from, so "
                   "a window could be passed by an engine that loses to buying "
                   "everything -- which on the selection period it does, by "
                   "about four points.",
        location="prosignal.validation.forward::Registration.tertiary",
        fix="Added `tertiary` and put it INSIDE the fingerprint, so the "
            "hypothesis cannot be added, softened or removed once observations "
            "have started landing. Legacy registrations still verify under the "
            "old scheme rather than being reported as tampered with.",
        regression_test="tests/test_measurement_guards.py::test_the_tertiary_hypothesis_is_inside_the_fingerprint",
        before_after="fingerprint scheme changed; a new registration missing "
                     "`tertiary` is refused",
        moves_coefficients=False, moves_history=False, forces_restart=True,
    ),
    _f(
        fid="R3", severity="critical",
        title="The Deflated Sharpe Ratio could not fail",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`deflated()` took Var[SR] across woven paths, which "
                   "duplicate every (split, date) pair, and fell back to "
                   "1/(n-1) -- documented as 'conservative unit variance' when "
                   "it is neither. With 612 duplicated pairs the dispersion "
                   "collapsed and the DSR returned 1.0000 at any trial count.",
        location="prosignal.validation.cpcv::CpcvResult.deflated",
        fix="Var[SR] now comes from the trial registry -- the population the "
            "selection actually searched -- with an explicit "
            "SR_VAR_UNDERCOVERED refusal when fewer than half the recorded "
            "trials carry a score, rather than a silent fallback.",
        regression_test="tests/test_measurement_guards.py::test_the_dsr_scores_independent_windows_not_duplicated_pairs",
        before_after="Deflated Sharpe 1.0000 PASS -> 0.0000 FAIL. Verified "
                     "insensitive to the variance source: conservative "
                     "Var[SR]=1.0 gives 0.0000, measured 0.0902 gives 0.0468, "
                     "both FAIL.",
        moves_coefficients=False, moves_history=True, forces_restart=True,
        notes="The verdict it produced before the fix is the reason the "
              "engine believed it had cleared multiple testing.",
    ),
    _f(
        fid="R4", severity="high",
        title="The trial registry counted trials and discarded their scores",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Only the count was persisted, so Var[SR] -- the input the "
                   "DSR is most sensitive to -- had to be guessed by whoever "
                   "read the number.",
        location="prosignal.validation.trials::record",
        fix="The registry records the score. `load` merges by trial id so a "
            "re-run cannot inflate the count.",
        regression_test="tests/test_measurement_guards.py::test_the_registry_records_what_each_trial_scored",
        before_after="DSR moves between 0.38 and 0.91 depending on the guess "
                     "the fix removes",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="R5", severity="high",
        title="Trial scores from one sweep of near-identical arms set the "
              "multiple-testing bar 24x too low",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Found by running R4's own fix: eighteen `research spread` "
                   "arms differ trivially, so Var[SR] measured 0.00178 and the "
                   "DSR flipped back to PASS. A registry that records every "
                   "trial equally treats a sweep as eighteen independent "
                   "searches.",
        location="prosignal.validation.metrics::deflated_sharpe",
        fix="MIN_TRIAL_SCORE_COVERAGE and an explicit undercovered verdict.",
        regression_test="tests/test_measurement_guards.py::test_trial_scores_from_a_fraction_of_the_search_do_not_set_the_bar",
        before_after="PASS -> FAIL, and the FAIL was then shown not to depend "
                     "on this constant",
        moves_coefficients=False, moves_history=True, forces_restart=False,
        notes="Found by testing a fix, not by reading code.",
    ),
    _f(
        fid="R6", severity="high",
        title="The 3R target could only fire on a close while the stop fired "
              "on the intraday low",
        category=Category.EXECUTION, status=Status.FIXED,
        root_cause="`portfolio_sim._hold` called `resolve_exits(high=None)`. "
                   "The label passes `high`, so the simulator and the thing "
                   "training it disagreed about when a winner ends -- "
                   "asymmetrically, and in the pessimistic direction.",
        location="prosignal.validation.portfolio_sim::_hold",
        fix="`high` is threaded through; a RuntimeWarning fires when the panel "
            "is absent rather than silently degrading.",
        regression_test="tests/test_portfolio_sim.py::"
                        "test_the_book_itself_takes_profit_on_an_intraday_spike",
        before_after="+0.43% / +0.10% per 63-session period",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="R7", severity="high",
        title="Re-entry after an early exit was free",
        category=Category.EXECUTION, status=Status.FIXED,
        root_cause="Only names absent from the PREVIOUS book paid a round "
                   "trip, but cohorts fully close before the next opens and "
                   "84% close early, so a name re-bought inside a cohort was "
                   "never charged for the second purchase.",
        location="prosignal.validation.portfolio_sim::simulate",
        fix="Charging is driven by what was actually held, not by book "
            "membership at the cohort boundary.",
        regression_test="tests/test_portfolio_sim.py::test_a_position_that_closed_early_pays_again_when_it_is_re_bought",
        before_after="-0.07% / -0.06% per period",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="R8", severity="high",
        title="`exit_hierarchy` was read by the card and by nothing else",
        category=Category.RISK, status=Status.FIXED,
        root_cause="The stop configuration reached the presentation layer "
                   "directly. Turning the stop off in config would have "
                   "changed no backtest, no label and no validation number -- "
                   "the config was describing the engine rather than "
                   "controlling it.",
        location="prosignal.stages.stage7_risk::exit_hierarchy",
        fix="The exit rules are built once, from config, and the simulator, "
            "the label and the card all read that one construction.",
        regression_test="tests/test_exit_agreement.py",
        before_after="a config change now changes the backtest",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    _f(
        fid="R9", severity="high",
        title="The training panel is not the population the book can buy",
        category=Category.UNIVERSE, status=Status.FIXED,
        root_cause="The admissibility predicate is applied inside "
                   "`resolve_exits`, which `triple_barrier: false` routes "
                   "around -- on the exact config the dossier is anchored to. "
                   "So the decision half of F5 shipped and the training half "
                   "did not, and the model was fitted on names the book cannot "
                   "fill. 7.29 of 8 selected slots fill.",
        location="prosignal.features.panel::build_panel",
        fix="`universe.train_on_admissible_only` now gates the training panel "
            "itself, independent of the labelling route, and ships ON.",
        regression_test="tests/test_admissible_population_r9.py",
        before_after="-0.25% / -0.20% per period, and the fitted coefficients "
                     "move",
        moves_coefficients=True, moves_history=True, forces_restart=True,
        notes="Enabled and refitted. Coefficients moving is the fix working, "
              "not evidence against it: preserving them would be preserving a "
              "fit to a population the engine cannot trade.",
    ),
    _f(
        fid="R10", severity="medium",
        title="Reported drawdown was the mean across schedules -- an "
              "experience nobody had",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`phase_summary` averaged `max_drawdown` over CPCV "
                   "schedules. The mean of maxima is always shallower than the "
                   "worst path and is not any investor's experience.",
        location="prosignal.validation.portfolio_sim::phase_summary",
        fix="`worst_schedule_drawdown` reported alongside, and named.",
        regression_test="tests/test_portfolio_sim.py::"
                        "test_the_worst_schedule_drawdown_is_reported_not_only_the_mean",
        before_after="-13.7% -> -19.1%",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="R11", severity="medium",
        title="Cash drag arrives inside the number labelled 'position sizing'",
        category=Category.RISK, status=Status.DISCLOSED,
        root_cause="Risk-budget sizing deploys about 25% of capital against a "
                   "fully-invested benchmark. The gap is leverage, not alpha, "
                   "and no surface said so.",
        location="prosignal.validation.portfolio_sim::PortfolioParams",
        fix="Disclosed and decomposed. `deployed_frac` is reported, and the "
            "75% cash is priced as what it buys -- an 8% book-level risk cap "
            "-- rather than as a loss.",
        regression_test="tests/test_portfolio_sim.py::test_turnover_and_exposure_are_reported_as_first_class_numbers",
        before_after="see the decomposition in docs/REAUDIT.md section D",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    _f(
        fid="R12", severity="medium",
        title="`holdout.sacred` was read by no code",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Eight commands each carried their own `--include-holdout` "
                   "arithmetic and the config flag they were supposed to "
                   "honour was inert.",
        location="prosignal.validation.holdout",
        fix="One guard, read by every command that can reach holdout dates.",
        regression_test="tests/test_measurement_guards.py::test_sacred_holdout_refuses_include_holdout",
        before_after="the flag now does what it says",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    _f(
        fid="R13", severity="high",
        title="A name with no ADTV got the largest allowed size AND the "
              "cheapest possible fill",
        category=Category.EXECUTION, status=Status.FIXED,
        root_cause="Two independently defensible fallbacks. The sizer read a "
                   "missing ADTV as 'no liquidity constraint applies' and fell "
                   "back to the capital slot; the cost model read it as "
                   "'cannot compute participation' and returned the half "
                   "spread alone. Together they manufacture liquidity the "
                   "engine has no evidence exists, concentrated in exactly the "
                   "thinnest names.",
        location="prosignal.liquidity::assess",
        fix="Four states -- KNOWN_VALID, KNOWN_STALE, MISSING, INVALID. "
            "Unknown means no new position, no optimistic fill and no "
            "imputation; `adtv_inr` is None whenever untradable so a caller "
            "that ignores the gate raises rather than sizes. Both sizers read "
            "the same policy, and the execution model is pinned by "
            "monotonicity properties rather than examples.",
        regression_test="tests/test_liquidity_gate.py",
        before_after="unknown-liquidity impact 5.0 bps -> 105.0 bps; "
                     "+0.17% / +0.17% per period",
        moves_coefficients=False, moves_history=True, forces_restart=True,
        notes="Liquidity is now represented twice and deliberately: as "
              "information in the features, and as a constraint at execution.",
    ),
    _f(
        fid="R14", severity="medium",
        title="The test that compared the two exit paths could not produce the "
              "bar that distinguishes them",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`test_exit_agreement`'s fixture never generated a session "
                   "that trades through the target and closes below it, which "
                   "is the only bar on which R6 is visible. The test was "
                   "correct and vacuous, and passed for months with R6 live.",
        location="tests/test_exit_agreement.py",
        fix="A deliberate intraday spike per symbol in the fixture.",
        regression_test="tests/test_exit_agreement.py",
        before_after="the test now fails when R6 is reverted; it did not",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Found by mutation testing the guards, not by reading them.",
    ),
    _f(
        fid="R15", severity="low",
        title="A name selected but never filled was recorded as held, so it "
              "paid nothing when it finally filled",
        category=Category.EXECUTION, status=Status.FIXED,
        root_cause="`held` was keyed on filled names rather than on the book, "
                   "so an unfilled selection carried forward as a position.",
        location="prosignal.validation.portfolio_sim::simulate",
        fix="`held` is keyed on the whole book, with NaN for the unfilled.",
        regression_test="tests/test_portfolio_sim.py::test_a_slot_that_never_filled_pays_when_it_finally_does",
        before_after="inert on this sample -- a name refused at entry tends to "
                     "stay refused or leave the band before it could fill",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Reported as costing nothing measurable rather than as a saving.",
    ),
    _f(
        fid="R16", severity="medium",
        title="The trade plan serialises `probability_of_profit`, and a field "
              "on a per-name card reads as a claim about that name",
        category=Category.UI, status=Status.DEFERRED,
        root_cause="`probability_of_profit` and "
                   "`probability_of_beating_benchmark` are POPULATION base "
                   "rates -- the share of the 258 study trades that ended "
                   "positive, at a stated cost, over a stated period. The "
                   "contract says so and the values are falsifiable against "
                   "live trades, which is the honest form. But they are "
                   "serialised inside `recommendations[].trade_plan`, where a "
                   "reader takes `probability_of_profit: 0.62` as this trade "
                   "having a 62% chance -- which the engine cannot and does "
                   "not claim. `test_engine_never_emits_a_probability` bans "
                   "the substring for exactly this reason.",
        location="prosignal.core.contracts::TradePlan",
        fix="NOT renamed here. The honest names are `study_win_rate` and "
            "`study_beat_benchmark_rate`, but these are output-contract keys: "
            "renaming them changes the payload the UI reads, and that is an "
            "operator's decision rather than a defect repair. The test now "
            "names these two as explicit, narrow exemptions pointing at this "
            "finding, so any NEW probability-shaped field still fails it.",
        regression_test="tests/test_pipeline_stages.py::"
                        "test_engine_never_emits_a_probability",
        before_after="no change to any traded number; the values are unchanged",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Latent since the field was added -- the test only reaches the "
              "trade plan on a day that produces a BUY, and it was found when "
              "the v3 absolute floor turned a NO TRADE day into one.",
    ),
    # -- the readiness dossier's own open items -------------------------------
    _f(
        fid="W2", severity="high",
        title="Every traded coefficient is a survivor of a selection on its "
              "own t-statistic",
        category=Category.MODEL, status=Status.DISCLOSED,
        root_cause="`gated_shrink` zeroes any theme below `significance_floor` "
                   "and prices the rest, so the surviving estimate is biased "
                   "away from zero by exactly the amount the gate removed. The "
                   "overstatement is largest for the themes that only just "
                   "cleared, which are the ones the decision turns on.",
        location="prosignal.validation.selection::correct_t (module removed 2026-09-03)",
        fix="One-sided truncated-normal MLE of the true non-centrality. Six "
            "acceptance criteria were fixed before it was written; five hold "
            "and the sixth -- that it recover the truth in simulation -- does "
            "not, so the number is REPORTED and NOT TRADED. "
            "`assert_not_traded` makes wiring it into a score fail a test "
            "rather than pass a review.",
        regression_test=None,
        before_after="mom_f t +2.87 -> +2.20 implied true; delivery_f +2.63 -> "
                     "+1.44, i.e. one of the two traded coefficients does not "
                     "clear the floor it was selected by once selection is "
                     "priced. On this tree's post-R9 refit both clear.",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="CLOSED BY REMOVAL, 2026-09-03. The finding is about coefficients "
              "selected on their own t-statistic and then TRADED. The fitted "
              "cross-sectional model that produced them was removed in the "
              "engine cleanup, so there are no such coefficients any more and "
              "nothing for the correction to correct. `validation/selection.py` "
              "and its guard test went with it. If a fitted ranker is ever "
              "restored, this finding comes back with it and the correction has "
              "to be rebuilt -- it is recorded here rather than deleted so that "
              "restoring the model cannot quietly restore the bias.",
    ),
    _f(
        fid="C3", severity="medium",
        title="The recorded operating history was produced by a different "
              "engine on a different universe",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`outcomes.jsonl` is append-only and partitioned by "
                   "`exit_model` alone. Every other thing that defines the "
                   "strategy -- the universe, the sizer, the cost model, the "
                   "fitted coefficients -- could change without the record "
                   "noticing, so trades decided by two engines averaged into "
                   "one win rate.",
        location="prosignal.outcomes::load_outcomes",
        fix="Rows carry `epoch_id`; `load_outcomes` partitions on it as it "
            "already did on `exit_model`; `summarise_by_epoch` reports each "
            "cohort separately AND reports what pooling would have claimed, so "
            "the partition cannot be quietly re-collapsed.",
        regression_test="tests/test_outcome_epochs.py",
        before_after="one pooled win rate -> one figure per epoch, with the "
                     "pooling error stated",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="C4", severity="medium",
        title="The retired operating record was not surfaced anywhere a reader "
              "could see what it belonged to",
        category=Category.UI, status=Status.FIXED,
        root_cause="`/outcomes` served one summary with no statement of which "
                   "engine produced it. Filtering it away would have been "
                   "worse: a page reading 'no trades yet' when the truth is "
                   "'the trades we have describe a different engine'.",
        location="prosignal.api::outcomes_summary",
        fix="The endpoint reports the current epoch's figures, lists every "
            "retired cohort labelled beside them, and states the size of the "
            "error pooling would have made.",
        regression_test="tests/test_outcome_epochs.py::"
                        "test_the_endpoint_labels_retired_cohorts",
        before_after="retired trades were counted silently; they are now "
                     "counted separately and named",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="D1", severity="high",
        title="`data/` is not in version control, so no result can name its "
              "own inputs",
        category=Category.REPRODUCIBILITY, status=Status.FIXED,
        root_cause="Every panel-derived figure the engine has ever produced is "
                   "reproducible only against whatever happened to be in the "
                   "store at the time. Two runs of the same command on "
                   "different days are not comparable and nothing says so.",
        location="prosignal.data.manifest::Manifest",
        fix="A content-addressed manifest -- path, size, sha256, row count, "
            "date range, schema hash -- committed instead of the data. The "
            "digest is recomputed on load and never trusted from the file, and "
            "`verify` reports drift per file. The dataset is uniquely "
            "reconstructable without putting a quarter of a gigabyte of market "
            "data into Git.",
        regression_test="tests/test_data_manifest.py",
        before_after="39 files, 0.25 GB, 9,270,123 rows, digest b6a1861bf6572e56",
        moves_coefficients=False, moves_history=False, forces_restart=True,
        notes="Restart-blocking because a forward test that cannot name its "
              "input data cannot be reproduced, only re-run.",
    ),
    _f(
        fid="D2", severity="high",
        title="Nothing recorded which engine a result came from",
        category=Category.REPRODUCIBILITY, status=Status.FIXED,
        root_cause="`config_version` alone identified a run. It does not "
                   "cover the code, the data, the universe policy, the feature "
                   "schema or the execution model, so two materially different "
                   "engines could produce results labelled identically.",
        location="prosignal.validation.epoch::Identity",
        fix="An append-only epoch ledger. An epoch fixes the code sha, the "
            "model-source sha, the config version, the data manifest digest, "
            "the feature schema hash, the universe policy and the execution "
            "model; any material change opens a new epoch and therefore a new "
            "out-of-sample evaluation. `drifted_from` reports divergence and "
            "deliberately never acts on it.",
        regression_test="tests/test_epoch.py",
        before_after="v1 archived as epoch 2026-08-28-113e70b2dc060afc, VOID",
        moves_coefficients=False, moves_history=False, forces_restart=True,
    ),

    # ---------------------------------------------------------------- v10 P0
    _f(
        fid="P0-1", severity="critical",
        title="README carried two irreconcilable book tables and the wrong one "
              "was the one a reader met first",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Two performance sections described the same engine and "
                   "could not both be true: 'RESULTS OF RECORD' (mean excess "
                   "-4.23%/period, IR -0.83, alpha -0.67%, 32.9% of periods "
                   "beating the benchmark) and 'What changed in the tuning "
                   "pass (2026-08-29)' (+42.6% annualised book return, +20.3% "
                   "annualised alpha, Sharpe 1.59, positive alpha in 6 of 6 "
                   "years). The second appeared TWICE, byte-identical, 102 "
                   "lines each. Neither was generated, so neither could "
                   "disagree with the other out loud, and nothing in the "
                   "repository could tell which described the shipped engine. "
                   "The deeper cause is that the tuning study measured a "
                   "DIFFERENT RANKER -- sector-neutral mom_6_1, one column -- "
                   "and its frequencies were nonetheless stamped onto every "
                   "card the v3 composite issued, through config `expectancy:`.",
        location="prosignal.validation.results::build",
        fix="`prosignal research results` GENERATES docs/RESULTS_OF_RECORD.md "
            "and a JSON twin from the current store, re-running BOTH "
            "configurations through the repository's own simulator and marking "
            "whichever fails to reproduce WITHDRAWN with its reason. Neither "
            "is averaged and the more favourable one is not quoted. The "
            "duplicated section is deleted; the surviving one is marked. Every "
            "figure is stamped with config version, params/store/train hashes, "
            "git commit, panel span and row count, distinct dates, independent "
            "observations, cumulative trials and the store fingerprint.",
        regression_test="tests/test_readme_numbers.py",
        before_after="two ungenerated tables, one duplicated, neither stamped "
                     "-> one generated record, README held to it by a test "
                     "that also fails if a WITHDRAWN arm's figures appear "
                     "outside a withdrawal notice",
        moves_coefficients=False, moves_history=True, forces_restart=False,
        notes="Reproduction of an already-charged configuration is not a new "
              "trial: neither arm could be SELECTED by this command, and both "
              "are already inside the counts the DSR charges. v10 pass P0 is "
              "budgeted at zero and stays there.",
    ),
    _f(
        fid="P0-2", severity="critical",
        title="The documented ranker was not the configured ranker, and the "
              "configured `column` named a different model's feature frame",
        category=Category.REPRODUCIBILITY, status=Status.FIXED,
        root_cause="config set `ranking.source: v3_composite` with "
                   "`ranking.column: mom_6_1_r`, while README's executive "
                   "summary said the shipped ranking WAS `mom_6_1_r`, one "
                   "column. `column` is read only when `source` is "
                   "`measured_factor`, so it was inert -- and `mom_6_1` is not "
                   "one of the twenty-two v3 factors at all: it belongs to "
                   "features/crosssec.py, the FITTED model's panel. A stale "
                   "value that looks like a live setting was being read as the "
                   "shipped ranker by everything downstream, including the "
                   "repository's own summary of itself.",
        location="prosignal.stages.stage4_core_score::_apply_ranking_policy",
        fix="Measured what actually orders the book by driving stages 1-4 on "
            "the 2026-09-03 cross-section and correlating the result against "
            "both candidates: the shipped order IS the v3 composite's order, "
            "Spearman +1.000000, identical on all 386 names. README's ranking "
            "row now names `v3_composite`; the inert column is prefixed "
            "`UNUSED:` so it cannot be misread again, and using "
            "`measured_factor` now requires setting both fields deliberately.",
        regression_test="tests/test_shipped_ranker.py",
        before_after="README 'mom_6_1_r, one column' vs config v3_composite "
                     "-> both name v3_composite, 22 factors in 5 themes, and a "
                     "test fails if either moves without the other",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    _f(
        fid="P0-3", severity="high",
        title="A data-depth change left config_version identical while moving "
              "the fitted coefficients",
        category=Category.REPRODUCIBILITY, status=Status.FIXED,
        root_cause="`config_version` hashed parameters.yaml and nothing else. "
                   "The model refits from stored history on every run, so the "
                   "store IS the training set: a store that grew produced "
                   "different coefficients under an identical hash. The "
                   "forward test's entire integrity check is 'did "
                   "config_version change', so the one instrument that "
                   "depended on it could not see the change -- which is how "
                   "the last window recorded FIVE distinct model fingerprints "
                   "against one config version.",
        location="prosignal.config.identity::identify",
        fix="config_version = label @ (H(params) XOR H(store_fingerprint) XOR "
            "H(train_window)), where the store fingerprint carries first "
            "session, last session, session count and symbol count per feed, "
            "and the training window carries the span actually fitted plus the "
            "cap, horizon, purge and embargo. `AppConfig.bind_store` resolves "
            "it; `params_version` keeps the parameters-only question "
            "answerable, because 'did the knobs move' is a real and different "
            "question.",
        regression_test="tests/test_config_identity.py",
        before_after="identical parameters + 250 more sessions of prices used "
                     "to give an identical version; they now give different "
                     "ones",
        moves_coefficients=False, moves_history=False, forces_restart=True,
        notes="Forces a restart because every prior window was graded against "
              "an identity that could not see its own training set moving.",
    ),
    _f(
        fid="P0-4", severity="high",
        title="The forward test's tamper check had an unreachable branch",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`Registration._payload` wrote `tertiary` into the payload "
                   "unconditionally AND again under `if not legacy`, so the "
                   "legacy and current payloads were byte-identical. "
                   "`fingerprint(legacy=True)` therefore equalled "
                   "`fingerprint()`, `verify(allow_legacy=True)` added "
                   "nothing, and `fingerprint_scheme` could never return "
                   "'legacy'. A tamper check with a branch that cannot execute "
                   "is a tamper check nobody has tested.",
        location="prosignal.validation.forward::Registration._payload",
        fix="The legacy payload now reproduces the v1 shape exactly -- without "
            "`tertiary`, which is what 'before the benchmark-relative "
            "hypothesis joined the hash' means. The two fingerprints are now "
            "genuinely distinct and the scheme is reported honestly.",
        regression_test="tests/test_forward_registration_v2.py",
        before_after="fingerprint() == fingerprint(legacy=True) for every "
                     "registration -> distinct, and 'legacy' is reachable",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    _f(
        fid="P0-5", severity="high",
        title="The short side could not be priced at all",
        category=Category.EXECUTION, status=Status.FIXED,
        root_cause="The `costs:` block had no derivatives leg. A leg with no "
                   "cost model does not read as UNKNOWN in a later comparison "
                   "-- it reads as FREE. Separately, futures STT tripled from "
                   "0.02% to 0.05% on 1 April 2026 and nothing in the "
                   "repository could notice a statutory rate going stale, "
                   "because STATUTORY values carried no verification date.",
        location="prosignal.config.schema::DerivativesCostConfig",
        fix="Added `costs.derivatives` with futures STT 0.05% sell-side "
            "(STATUTORY, verified 2026-09-03), options premium 0.15% "
            "(STATUTORY, carried and unused), borrow fee and futures roll "
            "spread (both UNVALIDATED with declared search ranges, swept never "
            "claimed) and futures margin (OPERATIONAL, a planning proxy for "
            "an exchange-computed SPAN+ELM number). `Tunable.verified_on` is "
            "now REQUIRED on every STATUTORY parameter. Nothing here places an "
            "order; docs/EXECUTION_GATE.md is unchanged.",
        regression_test="tests/test_derivatives_costs.py",
        before_after="no derivatives leg and seven undated STATUTORY rates -> "
                     "a priced short stack and a loader that refuses an "
                     "undated statutory rate",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    # ---------------------------------------------------------------- v12 P0
    _f(
        fid="Q1", severity="critical",
        title="The headline performance statistic moves with a position-sizing "
              "knob, so every ablation selected on it is unsafe",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`mean_excess` is mean(book - benchmark) against a benchmark "
                   "that is FULLY INVESTED, while the book is not. Sizing is "
                   "`risk_budget / risk_per_share`, so with a 1% risk budget "
                   "and an 8xATR stop clipped at 35% the book deploys 21.8% of "
                   "capital. Measured over 87 periods against a universe "
                   "returning +22.1%/yr, the reported -18.60% annual excess "
                   "decomposes as +17.30% mechanical cash drag and -1.30% "
                   "leverage-matched -- 93% arithmetic. Worse, holding the "
                   "ranking, the names and every other setting fixed and moving "
                   "ONLY `risk_per_trade_pct` from 1% to 4.6% takes deployment "
                   "from 0.218 to 0.824 and the reported excess from -20.98% to "
                   "-11.49%. The exit ablation, the stop multiple, the clip, "
                   "the book size and the holding period were all selected on "
                   "this metric.",
        location="prosignal.validation.portfolio_sim::_benchmark_stats",
        fix="`_benchmark_stats` now takes the deployed fraction and returns "
            "`alpha_on_deployed = alpha / dep` as the HEADLINE, which is "
            "invariant to leverage (writing r = dep*r_d gives beta = dep*beta_d "
            "and alpha = dep*alpha_d, so the raw alpha is proportional to "
            "deployment and only the ratio is invariant). `mean_excess` and "
            "`information_ratio` are retained for reconciliation and carry a "
            "`leverage_confounded` list; `alpha_per_period` carries a "
            "`leverage_proportional` list. `results.SHIPPED_FIGURES` demotes "
            "both raw figures out of the headline set, and a claim stated only "
            "in them now judges SUPERSEDED rather than REPRODUCED.",
        regression_test="tests/test_leverage_neutral.py",
        before_after="headline -18.60%/yr raw excess -> -6.04%/yr alpha on "
                     "deployed capital at t -0.65; raw spans 9.5 points across "
                     "the risk sweep, the headline spans 2.7",
        moves_coefficients=False, moves_history=True, forces_restart=True,
        notes="Restart-blocking because the forward test's secondary hypothesis "
              "was stated in the confounded unit and could have been passed by "
              "editing a sizing parameter. Scheme v3 restates it.",
    ),
    _f(
        fid="Q2", severity="critical",
        title="Sealed window B lies inside the fit window and was read as a "
              "second seal on the shipped weights",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`features/v3.py` records that the shipped signs and weights "
                   "were fitted 2018-11-27..2024-10-25. `SEALED_WINDOWS['B']` "
                   "is 2021-07-01..2022-12-27 -- entirely inside it. The two "
                   "windows were presented as a matched pair in the module "
                   "docstring, in `DEPLOY_REFERENCE` and in the re-check's "
                   "verdict text, so an in-sample number corroborated an "
                   "out-of-sample one. B is the better-looking of the two: it "
                   "carries the positive book (+2.0%/yr against A's -2.8%) and "
                   "the significant top-ten excess (t 2.50 against A's 0.81). "
                   "B is legitimate evidence about the METHOD -- the pipeline "
                   "was re-run on data ending 2021-02-17 -- and it is not "
                   "evidence about the shipped configuration, which is a "
                   "different fit.",
        location="prosignal.features.v3::FIT_WINDOW",
        fix="`FIT_WINDOW` and `window_provenance()` make the classification "
            "computable rather than remembered; `v3_panel."
            "SEALED_WINDOW_PROVENANCE` derives from them and "
            "`CITABLE_SEALED_WINDOWS` names the windows a claim about the "
            "shipped model may cite. The docstring table gains a provenance "
            "row and the re-check's HOLDS text no longer offers B as "
            "corroboration.",
        regression_test="tests/test_sealed_window_provenance.py",
        before_after="two windows quoted as two seals -> one citable window "
                     "(A), whose own reading is: the ranking generalises, the "
                     "book does not",
        moves_coefficients=False, moves_history=True, forces_restart=False,
    ),
    _f(
        fid="Q3", severity="high",
        title="Factors were ranked inside groups defined by TODAY's sector map, "
              "which is future information and also cost the signal",
        category=Category.FEATURE, status=Status.FIXED,
        root_cause="`_refresh_sector_map` pools the `Industry` column of the "
                   "CURRENT NSE constituent files, so a name that has since "
                   "delisted or left every index carries no sector and is "
                   "ranked inside `__RESID__`. Holding a sector label is "
                   "therefore correlated with having survived, and the "
                   "correlation is large: across 380 panel dates, names with a "
                   "known sector out-returned names without by +1.05% per 21 "
                   "sessions (overlap-corrected t +3.64) and +3.36% per 63 "
                   "(t +3.48) -- bigger than the signal itself. The groups the "
                   "ranking was computed inside were defined by that attribute. "
                   "Separately, `residual_bucket_size` had always reported that "
                   "39% of a live cross-section sits in the residual bucket "
                   "(79 unclassified plus 71 folded in from THIRTEEN sectors "
                   "too small to rank within), so for two names in five "
                   "'sector-neutral' named a peer group that does not exist.",
        location="prosignal.features.v3::SECTOR_NEUTRAL",
        fix="`SECTOR_NEUTRAL = False`. `score_frame` ranks across the eligible "
            "universe and still accepts `sectors` so `residual_bucket_size` can "
            "report what a point-in-time map would cover -- which is the "
            "condition for turning it back on. `sector_neutral_rank` is kept "
            "and reachable via `sector_neutral=True`; the function was never "
            "the problem, the map feeding it was. Three surviving user-facing "
            "claims were corrected: the stage-4 run note, `FactorMember.rank`'s "
            "contract docstring, and two strings in the interface.",
        regression_test="tests/test_no_sector_neutralisation.py",
        before_after="out-of-sample rank IC, measured through the shipped "
                     "score_frame: h=5 +0.0402 -> +0.0485, h=21 +0.0421 -> "
                     "+0.0562 (t +1.97 -> +2.18), h=63 +0.0456 -> +0.0759. "
                     "Removing a lookahead RAISED the signal, and the direction "
                     "reproduces what features/v9r.py recorded independently",
        moves_coefficients=True, moves_history=True, forces_restart=True,
        notes="This CHANGES THE RANKING and therefore opens a new epoch -- it "
              "is not a correctness patch that leaves the model alone. The "
              "sealed windows do not describe it, which is already true of the "
              "shipped configuration after the cap repair.",
    ),
    _f(
        fid="Q4", severity="high",
        title="Every panel-wide statistic pooled the dates the model was "
              "chosen on with the dates that tested it, and averaged across "
              "structurally different models while doing it",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="TWO CUTS, both missing. The signs and weights were fitted "
                   "2018-11-27..2024-10-25, which is 293 of the panel's 380 "
                   "signal dates, and the ranking table pooled those with the "
                   "87 that followed -- so the published figure was neither an "
                   "in-sample fit statistic nor an out-of-sample result. "
                   "Separately, `score_frame` re-caps the theme blend over the themes a "
                   "name actually has, which is right per name and turns into "
                   "a validation problem across time. The fundamentals feed "
                   "reaches almost nobody early in the panel and most of the "
                   "universe late in it -- `quality_sub` coverage runs 0.0% "
                   "(2018), 1.5% (2020), 23.3% (2021), 50.2% (2023), 86.4% "
                   "(2026), and mean themes per name rises 2.99 -> 4.86 over "
                   "the same span. So an early score is a three-theme blend "
                   "and a late one is a five-theme blend, and every IC, "
                   "spread and book figure quoted over the whole panel is a "
                   "weighted average across different models whose weighting "
                   "nobody chose: it was set by when a vendor's coverage "
                   "improved.",
        location="prosignal.validation.results::_ranking_results",
        fix="`_ranking_windows` reports four spans, headline first: "
            "OUT_OF_SAMPLE (the claim), IN_SAMPLE, STABLE_MODEL and "
            "FULL_PANEL. A span holding fewer than `MIN_STABLE_DATES` signal "
            "dates is not reported at all, because a handful of dates is not "
            "an out-of-sample result. "
            "`v3_monitor.theme_availability` reports the coverage per date "
            "and `stable_model_window` returns the first date from which "
            "EVERY theme stays above 40% -- 2023-07-21, leaving 150 of 380 "
            "dates. The ranking table now carries both windows side by side "
            "with a themes/name column, so the reader can see which model "
            "each row describes rather than inferring it. FULL_PANEL is kept, "
            "not replaced: it is the longer record, and hiding it would be "
            "the same class of selection this register exists to stop.",
        regression_test="tests/test_model_stability_window.py",
        before_after="rank IC at h=21, overlap-corrected, from the "
                     "regenerated docs/RESULTS_OF_RECORD.md (the 5-session "
                     "panel the shipped generator builds): OUT_OF_SAMPLE "
                     "+0.0580 t +2.28 on 87 dates; IN_SAMPLE +0.0691 t +3.70 "
                     "on 269; STABLE_MODEL +0.0646 t +3.62 on 150. The "
                     "ordering survives both restrictions with a 16% decay, "
                     "and the out-of-sample t does NOT clear the "
                     "Harvey-Liu-Zhu 3.0 bar on this panel -- 87 dates five "
                     "sessions apart against a 21-session label carry VIF "
                     "4.17. Re-measured on a 21-session panel, where the same "
                     "87 observations do not overlap, the reading is +0.0411 "
                     "at t +3.91. Both are true of their own sampling and the "
                     "document is the authority",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="This does not move a coefficient. It changes what the evidence "
              "is understood to be evidence ABOUT, and it cuts the honest "
              "sample behind a claim about the shipped composite from 380 "
              "dates to 87.",
    ),
    _f(
        fid="Q5", severity="critical",
        title="The Deflated Sharpe charged for the tuning done after the model "
              "shipped and for none of the search that produced it",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="The trial registry held 99 configurations and every one "
                   "came from a research command written AFTER the v3 "
                   "composite existed -- estimator arms, spread bands, CPCV "
                   "folds, the 2026-08 execution re-audit. Not one row came "
                   "from the search that chose the 22 factors, the five "
                   "themes, the combination method, the weight caps, the "
                   "quality floor or the book. The escape hatch, "
                   "`cumulative_trials_logged`, shipped at 20. The reason it "
                   "could not be reconstructed is that "
                   "`research/V3_SEARCH.md` was DELETED in commit f1b2a9a "
                   "along with `work/v3/` and `research/v3/` -- deleting the "
                   "search code is defensible since it no longer chooses "
                   "anything, and deleting the record of the search removed "
                   "the only evidence of how much dredging the shipped model "
                   "rests on.",
        location="prosignal.validation.v3_search",
        fix="`research/V3_SEARCH.md` restored from f1b2a9a^. "
            "`validation/v3_search.py` is its machine-readable form: every "
            "group cites the section it comes from and expands to one "
            "registry label per configuration, so the registry's own "
            "content-addressing keeps it idempotent. "
            "`research trials --register-v3-search` appends them; the plain "
            "command now prints the enumeration and flags loudly when the "
            "rows are missing. Two grids whose arm counts the record does not "
            "state are entered at their documented minimum and marked, so the "
            "total prints as AT LEAST rather than an equals sign.",
        regression_test="tests/test_v3_search_trials.py",
        before_after="trials charged by the DSR 119 -> 622. On the shipped "
                     "trial-score variance (0.0874 over 50 scored arms) that "
                     "raises the Sharpe a selected configuration must beat "
                     "from 2.59 to 3.12 per period",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="This makes every deflated statistic in the engine STRICTLY "
              "HARDER to pass. It does not change a single return, and it is "
              "the second-largest single correction in this audit after the "
              "leverage confound.",
    ),
    _f(
        fid="Q6", severity="critical",
        title="The simulator decided four times a year and the engine decides "
              "twelve, so every cost figure described a different strategy",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`simulate` walks `ceil(horizon / step_sessions)` ranking "
                   "dates at a time, and the numerator was always "
                   "`horizon_sessions`. At the shipped horizon of 63 that is a "
                   "NON-OVERLAPPING COHORT schedule -- form a book, hold it "
                   "for the whole horizon, liquidate, form the next -- and "
                   "four decisions a year. The live engine decides every "
                   "`stage6_entry.admission.entry_cadence_sessions`, which is "
                   "21: twelve times a year, three times as often, carrying "
                   "names across decisions through the exit band. The cohort "
                   "schedule cannot re-rank a held name for 63 sessions, so it "
                   "never pays the turnover the hysteresis band generates, and "
                   "its cost is the cheaper of the two. The code comment even "
                   "stated the assumption -- 'rebalances are ceil(horizon/step) "
                   "apart precisely so one cohort finishes before the next "
                   "opens' -- while the engine it was measuring did not work "
                   "that way.",
        location="prosignal.validation.portfolio_sim::simulate",
        fix="`decision_sessions` truncates each cohort at the next decision "
            "date and re-selects: a name still inside the exit band is kept "
            "and owes nothing, a name that has left it or whose position "
            "closed early is replaced and pays a round trip -- the live book's "
            "arithmetic. `PortfolioParams.decision_sessions` carries it and "
            "`_portfolio_params` reads the live cadence, so the shipped "
            "measurement and the shipped engine now decide on the same clock. "
            "`phase_summary` annualises on the HOLD rather than the horizon "
            "(twelve periods a year at cadence 21, not four) and reports "
            "`decision_sessions` on the result, so a cost figure can no longer "
            "be quoted without its schedule. The default is unchanged for a "
            "caller that passes nothing.",
        regression_test="tests/test_cadence_parity.py",
        before_after="measured on the whole store, same ranking and same "
                     "params, only the cadence moved: cost 0.45% -> 1.17% a "
                     "year and its share of gross 10.3% -> 22.6%; round trips "
                     "15.8 -> 40.6 a year; periods per year 4.0 -> 12.0. "
                     "Deployment is unchanged at 20%, so this is turnover "
                     "rather than leverage. Alpha on deployed capital RISES, "
                     "+4.50% -> +6.74%, which is consistent with the "
                     "out-of-sample IC being strongest at the shorter "
                     "horizons: the live book re-ranks often enough to use it",
        moves_coefficients=False, moves_history=True, forces_restart=False,
        notes="This does not change the ranking, and it changes every cost, "
              "turnover and net-return figure the simulator has ever "
              "produced. The cost figures all move the same way -- worse, by "
              "more than a factor of two. The book's own alpha does not, and "
              "that is worth noticing rather than filing away: the schedule "
              "the engine actually runs is better than the one it was being "
              "measured on, and it was being measured on the wrong one for "
              "reasons that had nothing to do with which was better.",
    ),
    _f(
        fid="Q7", severity="high",
        title="Market impact cannot be calibrated, because the ledger holds no "
              "fills -- it holds the engine's own entry rule",
        category=Category.EXECUTION, status=Status.OPEN,
        root_cause="`CostModel.impact_bps` is "
                   "`coefficient * participation ** exponent` plus an assumed "
                   "half-spread, and both constants are config values that "
                   "have never been compared to a price this engine traded at. "
                   "The obvious calibration -- regress realised implementation "
                   "shortfall on participation -- cannot run: in 126 of the "
                   "128 rows of `data/ledger/outcomes.jsonl` the recorded "
                   "`entry_price` equals the NEXT SESSION'S OPEN to the tick, "
                   "so the ledger is recording the simulator's entry rule "
                   "rather than an execution. Fitting the impact model to "
                   "those rows would fit it to the assumption it was built "
                   "from and report the circularity as agreement. The other "
                   "two are VEDL rows carrying an unadjusted price against an "
                   "adjusted open (415.65 against 145.90, ratio 2.85, with "
                   "`price_basis_factor` recorded as 1.0) -- the price-basis "
                   "defect, not a fill.",
        location="prosignal.validation.fill_calibration::calibrate",
        fix="OPEN, and it stays open until the ledger holds broker fills -- no "
            "amount of code closes it. What is built is the harness and its "
            "refusal: `calibrate` returns SYNTHETIC_FILLS when the recorded "
            "price collapses onto a reference price for more than "
            "`SYNTHETIC_SHARE` of rows, reports NO coefficient when it "
            "refuses, and drops price-basis rows by name rather than fitting "
            "through them. It returns CALIBRATED on genuine fills, which is "
            "what makes the refusal mean something.",
        regression_test="tests/test_fill_calibration.py",
        before_after="87 bps round trip, asserted; still asserted, and now "
                     "labelled as an assumption with the harness that would "
                     "test it standing ready",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Reading the same ledger off the raw year parquets instead of "
              "through `DataStore` gives a different answer -- 92.4% "
              "synthetic, and TATAINVEST appears with a fill of 1066.00 "
              "against a raw open of 10660.00. The ledger's `entry_price` is "
              "not on one consistent price basis. That is a second defect and "
              "it is recorded here rather than fixed here.",
    ),
    _f(
        fid="Q8", severity="high",
        title="Every recorded ablation was decided on a statistic that moves "
              "with the knob being ablated",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`parameters.yaml` settles its exit, target and band "
                   "ablations on annual alpha, excess Sharpe and worst-year -- "
                   "'booking at 3R cost 0.9 points of annual alpha', 'booking "
                   "at 1.5R cost 4.6 points'. None of those is comparable "
                   "across the arms that produced them. Position size is "
                   "`risk_budget / risk_per_share`, so an arm that changes the "
                   "stop distance, the risk budget or the slot count changes "
                   "how much capital is deployed; with `r = dep * r_d`, raw "
                   "alpha is PROPORTIONAL to deployment and raw excess "
                   "additionally carries a cash-drag term against a "
                   "fully-invested benchmark. Disarming the 3R target changes "
                   "how long positions live and therefore how much capital "
                   "sits in cash, so the arm moved for a reason that has "
                   "nothing to do with whether the target is a good rule.",
        location="prosignal.validation.ablation::rank_arms",
        fix="`validation/ablation.py` runs arms through the shipped simulator "
            "at a FIXED cadence -- letting `decision_sessions` vary would make "
            "a one-variable comparison into a two-variable one, see Q6 -- and "
            "`rank_arms` RAISES on any key in `CONFOUNDED` rather than sorting "
            "by it, naming the mechanism and pointing at "
            "`alpha_on_deployed_ann`. The confounded columns are still printed "
            "in `table`, because every earlier write-up quotes them and a "
            "reconciliation needs them; what they may not do is decide.",
        regression_test="tests/test_ablation_leverage_neutral.py",
        before_after="the guard is demonstrated on a pure sizing sweep in "
                     "which the ranking, the names and the costs are "
                     "identical: raw excess separates the arms and "
                     "alpha-on-deployed does not",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="The recorded ablation VERDICTS are not overturned here -- "
              "re-running them needs the full panel and is its own decision. "
              "What is fixed is that the next one cannot be decided the same "
              "way.",
    ),
    _f(
        fid="Q9", severity="high",
        title="Breadth was counted in declared factors, and 22 correlated "
              "factors are not 22 bets",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Grinold's IR = IC * sqrt(breadth) takes breadth to be "
                   "INDEPENDENT bets. This engine describes itself as 22 "
                   "factors across 5 themes and every breadth argument in the "
                   "repository rests on those two counts. The pairwise "
                   "redundancy check catches duplicates one pair at a time and "
                   "says nothing about the aggregate, so nothing measured how "
                   "many independent columns the composite actually carries. "
                   "The participation ratio over each cross-section's Spearman "
                   "matrix, averaged across 380 panel dates: 20.9 factor "
                   "columns present carry 6.94 effective (median 7.34), and "
                   "4.59 theme columns carry 3.96 (median 4.30).",
        location="prosignal.v3_monitor::effective_breadth",
        fix="`participation_ratio`, `effective_count` and `effective_breadth` "
            "in `v3_monitor`, wired into `_v3_redundancy` so every stage-4 run "
            "reports it on `RedundancyReport.effective_breadth` with a note "
            "naming the overstatement factor. Averaged across dates rather "
            "than pooled: a matrix over stacked cross-sections mixes "
            "within-date structure with drift in the factor means, and drift "
            "is not breadth. The default is an EMPTY dict, because an absent "
            "measurement and a measured 22 are different things.",
        regression_test="tests/test_effective_breadth.py",
        before_after="breadth counted as 22; measured at 6.94, so any IR "
                     "computed from the declared count overstates by 1.74x. "
                     "The theme level is close to honest at 3.96 of 4.59, "
                     "which is the two-level structure doing its job",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="This does not prune anything. Pruning to the effective count is "
              "a model change that spends trials and opens an epoch, and it is "
              "an operator's decision taken against this measurement rather "
              "than a consequence of it.",
    ),
    _f(
        fid="Q10", severity="critical",
        title="The book is concentrated in the one part of the ranking that "
              "did not generalise",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`decile_monotonicity` compresses the whole shape of the "
                   "ranking into one rank correlation, so a profile that rises "
                   "to D7 and falls away past it scores +0.33 and reads as "
                   "healthy. The profile itself, mean excess over each date's "
                   "own cross-section at h=63:\n"
                   "        D1     D2     D3     D4     D5     D6     D7     "
                   "D8     D9    D10\n"
                   "  IS  -2.77  -1.41  -0.93  -0.10  -0.22  +0.06  +0.70  "
                   "+0.78  +1.44  +2.45\n"
                   "  OOS -2.31  -1.27  -0.60  -0.65  +0.31  +1.23  +1.73  "
                   "+0.64  +0.58  +0.36\n"
                   "In sample it is monotone and D10 wins by a distance. Out "
                   "of sample it PEAKS AT D7 and D10 is the sixth-best decile: "
                   "D10 minus D6 is +2.39 in sample and -0.87 out of it, and "
                   "the same inversion appears at h=21 (+0.82 -> -0.27). The "
                   "shipped book holds six names off the very top of D10.",
        location="prosignal.validation.results::_decile_profile",
        fix="Every ranking row carries `decile_profile` -- all ten deciles, "
            "the peak decile and D10 minus D6 -- and the document renders it "
            "as its own table under the ranking, split by fit-window "
            "provenance. A report that only ever prints the top decile cannot "
            "say that the top decile is not where the information is, so the "
            "peak is reported rather than assumed and "
            "`test_a_monotone_ranking_peaks_at_the_top_and_an_inverted_one_"
            "does_not` exercises the detector in both directions.",
        regression_test="tests/test_model_stability_window.py",
        before_after="the out-of-sample top-decile t in the regenerated "
                     "document is +0.59 (h=21), +0.62 (h=42), +0.29 (h=63) "
                     "against an in-sample +2.64, +2.43, +2.44. The quintile "
                     "spread and the rank IC hold up; the tail does not",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="The BOTTOM of the distribution generalises closely, -2.77 to "
              "-2.31 at h=63. That is the Stambaugh-Yu-Yuan result -- anomaly "
              "alpha concentrates in the short leg -- reproduced from a "
              "long-only panel that was never built to test it, and it is not "
              "a leg this engine can trade. Nothing here changes the book: "
              "widening it to D6-D8 is a model decision that spends trials "
              "and opens an epoch, and it is now a decision somebody can take "
              "against a measurement instead of against a monotonicity "
              "coefficient that hid the shape.",
    ),
    _f(
        fid="Q11", severity="high",
        title="The second-largest weight in the model is held down by a "
              "coverage cap that expired",
        category=Category.FEATURE, status=Status.FIXED,
        root_cause="Each theme's weight was capped at the share of names it "
                   "can speak about, measured once over the fit window and "
                   "frozen on `Theme.coverage`. That was the right call: "
                   "V3_SEARCH.md §6 records that fitted without the "
                   "constraint, `quality` took 40%+ of the composite while "
                   "only 19% of names had fundamentals at all, which ranks the "
                   "19% and the 81% by two different models and calls the "
                   "result one score. Quality's shipped 0.18991 IS its 0.1899 "
                   "coverage cap. The feed has since caught up: measured "
                   "`quality_sub` coverage is 0.363 over the fit window and "
                   "0.837 over the last year, a 4.4x drift. min(0.40, 0.837) "
                   "is 0.40, so a refreshed cap would not cut quality at all "
                   "-- the weight is held down by a constraint that no longer "
                   "binds, and the constant recording it says nothing about "
                   "that.",
        location="prosignal.features.v3::coverage_drift",
        fix="`coverage_drift` reports declared against measured per theme with "
            "the binding status of each, and `stale_coverage_caps` returns a "
            "sentence per theme whose BINDING STATUS has changed. Only that: "
            "momentum's 0.9988 could halve and its weight would still be set "
            "by the structural 0.40 cap, and an alarm that fires there is one "
            "nobody reads. Both directions are reported -- a cap that has "
            "stopped binding holds a weight down for an expired reason, and "
            "one that has started binding means a theme is weighted for "
            "coverage it no longer has.",
        regression_test="tests/test_coverage_cap_is_stale.py",
        before_after="quality: declared coverage 0.1899, measured 0.8372 over "
                     "the last year (4.4x). The cap bound when the weight was "
                     "chosen and does not now",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="NOTHING IS REFITTED. Refreshing the cap roughly doubles the "
              "quality weight and pushes momentum off its own cap, which is a "
              "model change that spends trials and opens an epoch. It is a "
              "decision to take against this measurement, not a consequence "
              "of it. `test_the_shipped_weight_still_equals_the_declared_cap` "
              "fails if the weights move, so acting on this cannot happen "
              "quietly either.",
    ),
    _f(
        fid="Q12", severity="medium",
        title="The regime factor multipliers are computed, logged, printed and "
              "inert on the shipped path",
        category=Category.UI, status=Status.FIXED,
        root_cause="Stage 2 derives three factor multipliers from the regime "
                   "read and scales the FAMILY block with them. "
                   "`_apply_ranking_policy` discards that block under "
                   "`ranking.source = v3_composite`, which is the shipped "
                   "setting. The stage-4 run NOTE was already fixed this way "
                   "-- deferred until the ranking source is known -- because "
                   "'Regime range_lowvol multipliers applied (momentum x0.75)' "
                   "on a run ordered by an unmodified v3 blend tells an "
                   "operator the engine leaned against momentum today, and it "
                   "did not. Every other surface the number reaches was left: "
                   "the CLI regime table, the regime history row, and the "
                   "ledger.",
        location="prosignal.core.contracts::RegimeState.multiplier_note",
        fix="`RegimeState.scores_the_shipped_book`, set in stage 2 from "
            "`ranking.source`, defaulting FALSE -- the shipped configuration "
            "is the one where the multipliers do not reach the book, so an "
            "unset flag must read that way or every caller who forgets "
            "restores the misreading. `multiplier_note()` returns the sentence "
            "or None; the CLI marks the table [INERT] and the history column "
            "with an asterisk; the ledger row carries "
            "`momentum_multiplier_scored_the_book`.",
        regression_test="tests/test_regime_multipliers_are_inert.py",
        before_after="a ledger row carried `momentum_multiplier: 0.75` and "
                     "nothing else; it now carries whether that 0.75 reached "
                     "the book",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Not deleted. They work on the `fitted_composite` path, which is "
              "still selectable, and the regime READ itself -- trend, "
              "volatility, breadth, and the hard entry gate -- is live on "
              "every path. `compatibility()` is deliberately not gated on the "
              "flag: it reads the regime, and suppressing it would remove a "
              "live signal to fix an inert one.",
    ),
    _f(
        fid="Q13", severity="medium",
        title="99% of ledger rows carry a mode that was inferred afterwards, "
              "and nothing downstream could tell",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="`repair_lineage` stamps each row's `mode` from what its "
                   "own timestamps prove -- `logged_at` against `date` -- and "
                   "writes `mode_source` saying so. That is the right repair "
                   "and it does not make the label a recording. On the shipped "
                   "ledger 250 of 253 rows carry it. Nothing could see that: "
                   "`outcomes.load_outcomes` partitions on `exit_model` and on "
                   "the research epoch, both recorded at write time, and never "
                   "on `mode`; `forward.progress` counted observations without "
                   "asking how their liveness was established. A forward test "
                   "reporting 'N sessions elapsed' was resting on an inference "
                   "with no way to say so.",
        location="prosignal.ledger::Ledger.mode_provenance",
        fix="`mode_provenance` and `live_evidence_warning` on the ledger; "
            "`Progress.reconstructed_mode_rows`, counted AFTER the "
            "`when < start` guard so a window opened since the repair inherits "
            "none of it, and surfaced through `Progress.caveats()` and the "
            "forward CLI.",
        regression_test="tests/test_reconstructed_mode_is_visible.py",
        before_after="250 of 253 rows (99%) reconstructed, reported nowhere; "
                     "now reported wherever a count is taken",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Reported, not refused. A hard failure would void a window for a "
              "defect in how its rows were LABELLED rather than in what they "
              "contain, and the inference rule is sound. Caveats print even "
              "when the window is already broken: a reader deciding how to "
              "re-register needs both.",
    ),
    _f(
        fid="Q14", severity="high",
        title="Re-run leverage-neutral, the exit-band ablation reverses: the "
              "shipped band is too narrow and the old metric said the "
              "opposite",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="PHASE 0 of the audit asked for the exit/stop/size "
                   "ablations to be re-run on a statistic that does not move "
                   "with the knob. Building the harness (Q8) was not the same "
                   "as running them. Re-run on the whole store at the live "
                   "cadence, ranked on alpha over deployed capital:\n"
                   "  EXIT BAND   12: +6.28%  18 (shipped): +6.74%  "
                   "36: +8.76%\n"
                   "  raw excess  12: -17.12%  18: -16.29%  36: -17.03%\n"
                   "The widest band earns +2.0 points more alpha on deployed "
                   "capital AND a shallower drawdown (-10.8% against -12.2%), "
                   "while its RAW excess is worse than the shipped band's. An "
                   "ablation decided the old way rejects the better "
                   "configuration -- which is not a hypothetical about the "
                   "confound, it is the confound choosing.",
        location="prosignal.validation.ablation",
        fix="Ablations re-run and the arms charged to the trial registry -- "
            "11 configurations, taking the DSR count to 633. Re-measuring a "
            "configuration already tried is still a look at the same data, "
            "and the registry's rule is that anything whose out-of-sample "
            "score was looked at is charged.",
        regression_test="tests/test_ablation_leverage_neutral.py",
        before_after="EXIT RUNGS: the shipped arm, no_target and "
                     "no_invalidation are identical at +6.74%, which is "
                     "correct -- both rungs are already disarmed in the "
                     "shipped config, so disarming them again changes "
                     "nothing. `no_stop` is +5.91%, so the ATR stop is worth "
                     "+0.83 points and earns its place. RISK BUDGET: raw "
                     "excess spans 14.4 points (-3.94% to -18.36%) while "
                     "alpha on deployed spans 2.7 (+6.04% to +8.73%) -- the "
                     "confound reproduced on the real panel at 5.3x",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="ALPHA ON DEPLOYED IS NOT PERFECTLY FLAT across the sizing "
              "sweep -- it rises from +6.04% at 0.5% risk to +8.73% at 4%, "
              "where deployment reaches 70% and the worst drawdown reaches "
              "-37.2%. It is five times more stable than the raw figure and "
              "it is not an invariant, and saying otherwise would be the same "
              "kind of overclaim this audit exists to remove. Widening the "
              "band is NOT applied here: it is a model change that spends "
              "trials and opens an epoch, and it now has a measurement behind "
              "it instead of a metric that pointed the wrong way.",
    ),
    _f(
        fid="Q15", severity="medium",
        title="The hysteresis band carries less than a third of the book, and "
              "nothing measured what it saved",
        category=Category.EXECUTION, status=Status.FIXED,
        root_cause="`entry_rank`/`exit_rank` is 6/18 and the wider exit band "
                   "exists so a held name is kept while it stays inside it, "
                   "paying nothing. Nothing reported whether that happened. "
                   "Measured at the live cadence, 3.39 of 4.79 held names are "
                   "charged a round trip EVERY period -- the band carries 29% "
                   "of the book and 40.6 round trips a year are paid anyway. "
                   "The band is not the only thing that can fail to save a "
                   "position: a name whose position closed early, stopped out "
                   "or exited at the horizon, is re-bought and pays however "
                   "comfortably it sits inside the band.",
        location="prosignal.validation.portfolio_sim::phase_summary",
        fix="`carried_free_share` and `round_trips_per_year` on the phase "
            "summary. The second is the number every cost figure is built "
            "from and nothing reported it.",
        regression_test="tests/test_cadence_parity.py",
        before_after="`avg_new` was reported and undercounted turnover by "
                     "excluding re-entries; the share of the book the band "
                     "actually carries was not reported at all",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="A hysteresis band cannot be judged by its width. Q14 measures "
              "that a wider one is worth +2.0 points of alpha here; this "
              "measures why a band of any width leaves most of the turnover "
              "in place.",
    ),
    _f(
        fid="Q16", severity="medium",
        title="The interface headline read as the account's return and was a "
              "per-position average",
        category=Category.UI, status=Status.FIXED,
        root_cause="The open-positions headline read `+X% vs the index`. That "
                   "figure is the average one POSITION is ahead of the index "
                   "over the days it was held -- the right way to judge the "
                   "SELECTION, since a position's return does not depend on "
                   "how large the position was, and not what an account "
                   "running this engine is up. Position size is "
                   "`risk_budget / risk_per_share`, so the book runs about a "
                   "fifth invested and the rest sat in cash while the index "
                   "compounded. The closed-record headline had always said "
                   "'average per position'; the open one said only 'vs the "
                   "index', and neither said anything about the cash.",
        location="prosignal.static.index.html",
        fix="The open headline reads 'vs the index, per position', and both "
            "headlines carry a note that this is not what the account earned "
            "because sizing leaves most of it in cash. A test asserts that no "
            "leverage-confounded BOOK figure -- `mean_excess`, "
            "`information_ratio` -- ever reaches the interface.",
        regression_test="tests/test_ui_headline_is_honest.py",
        before_after="'+X% vs the index' -> '+X% vs the index, per position "
                     "... not what the account earned: sizing leaves most of "
                     "it in cash'",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="No number changes. What changes is what the page claims one "
              "means. The per-position figure is kept as the headline because "
              "it is the honest measure of selection; it is the missing "
              "sentence about cash that made it read as a balance.",
    ),
    _f(
        fid="Q17", severity="critical",
        title="The long-only constraint is not what binds -- the six-name "
              "concentration is, and a tradeable shape carries the signal",
        category=Category.MODEL, status=Status.FIXED,
        root_cause="IR = TC x IC x sqrt(breadth), and nothing measured TC. A "
                   "signal that fails to appear in the book is either a broken "
                   "signal or a book that cannot hold the signal's opinion, "
                   "and those call for opposite responses. Measured out of "
                   "sample at h=63 over 87 dates and a median 750 names, "
                   "gross of cost:\n"
                   "  D10-D1        LONG_SHORT  +2.6729%  t +1.08  TC +0.762\n"
                   "  half-minus-half LONG_SHORT +1.8120%  t +1.54  TC +0.822\n"
                   "  D6-D8         LONG_ONLY   +1.1934%  t +2.42  TC +0.296\n"
                   "  top-half      LONG_ONLY   +0.9052%  t +1.54  TC +0.822\n"
                   "  D10           LONG_ONLY   +0.3242%  t +0.26  TC +0.553\n"
                   "  top6 SHIPPED  LONG_ONLY   -1.5196%  t -0.67  TC +0.206\n"
                   "The shipped shape is the worst of the six and its "
                   "out-of-sample excess is NEGATIVE. The decile it is drawn "
                   "from is indistinguishable from zero on its own. D6-D8 -- "
                   "where the decile profile peaks out of sample, see Q10 -- "
                   "is the ONLY shape of any kind that clears t = 2, and it is "
                   "long-only and tradeable.",
        location="prosignal.validation.transfer",
        fix="`validation/transfer.py` evaluates portfolio SHAPES against the "
            "same panel the IC is measured on, reporting gross excess over the "
            "equal-weight cross-section, an overlap-corrected t, Grinold's "
            "transfer coefficient, and how many names the shape needs. "
            "Long-short shapes are computed and marked untradeable: India has "
            "no retail borrow market worth the name, and a table that ranks a "
            "short book beside a long one without saying so is proposing "
            "something the engine cannot do.",
        regression_test="tests/test_transfer_coefficient.py",
        before_after="the engine's answer to 'the IC is real but the book "
                     "loses' was a shrug between two diagnoses; it is now a "
                     "measurement that rules one of them out",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="GROSS, and the word is load-bearing: D6-D8 holds 221 names and "
              "pays turnover a six-name book does not, and nothing here prices "
              "it. This is a construction diagnostic, not a proposal -- "
              "changing the book is a model decision that spends trials and "
              "opens an epoch. The six shapes are charged to the registry. "
              "What it settles is the DIAGNOSIS: the long-only constraint is "
              "not the binding one, because the best-performing shape in the "
              "table is long-only.",
    ),
    _f(
        fid="Q18", severity="high",
        title="Two factor signs are backwards out of sample, and one of them "
              "is in the theme whose weight a cap refresh would double",
        category=Category.MODEL, status=Status.FIXED,
        root_cause="The signs ARE the model, and nothing checked them off the "
                   "data they were chosen on. `review_factors` watches a "
                   "ROLLING window, which answers 'is this drifting' and not "
                   "'was this ever true out of sample'. Measured at h=63 over "
                   "78 dates after the fit window closed, 20 of 22 factors "
                   "hold their shipped sign and several strongly -- "
                   "`deliv_z_21` t +9.13, `ret_kurt_126` t -6.42, `mom_12_6` "
                   "t +6.64. Two do not: `mom_3_1` ships +1 and reads -0.0248 "
                   "at t -2.18, and `net_margin` ships -1 and reads +0.0207 "
                   "at t +2.52. Both are significant in the OTHER direction, "
                   "which is not the same failure as decaying to zero -- the "
                   "factor still carries information and the model is using "
                   "the sign backwards.",
        location="prosignal.v3_monitor::out_of_sample_signs",
        fix="`out_of_sample_signs` scores every shipped (factor, sign) pair on "
            "the dates after `v3.FIT_WINDOW` closes and separates a FLIP -- "
            "wrong direction at |t| >= SIGN_FLIP_T -- from a decay to zero, "
            "because only one of those is actionable. `flipped_signs` returns "
            "the sentences. The threshold is a significance bar rather than a "
            "sign test: half the factors read the wrong way by chance on a "
            "short window and flagging all of them is an alarm nobody reads.",
        regression_test="tests/test_out_of_sample_signs.py",
        before_after="22 shipped signs, none of them ever checked out of "
                     "sample; 20 hold, 2 are backwards at |t| > 2",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="`net_margin` is the one that matters, and it compounds Q11. The "
              "'quality' theme -- correctly labelled 'Low-margin tilt' -- "
              "carries 18.99% of the composite and is itself out-of-sample "
              "indistinguishable from zero: quality_sub reads +0.0103 at "
              "t +1.41 against an in-sample +0.0547 at t +7.31. Q11 shows its "
              "coverage cap has expired, so refreshing the cap would roughly "
              "DOUBLE the weight of the one theme that does not survive its "
              "own holdout. Nothing is changed here; the two findings together "
              "are the argument against the refresh.",
    ),
    _f(
        fid="P0-6", severity="high",
        title="The trial budget was countable but not enforceable",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="The registry counted trials honestly and could not refuse "
                   "one. Counting is not the same as spending deliberately: "
                   "the Deflated Sharpe already reads 0.030 against 4,877 "
                   "trials, so a campaign that discovers halfway through that "
                   "it has spent thirty has already made whatever ships less "
                   "credible, and nothing can give them back.",
        location="prosignal.validation.registry::TrialRegistry.record",
        fix="The v10 budget of 40 is declared per pass (P0=0, P1=0, P2=2, "
            "P3=12, P4=4, P5=8, P6=4, P7=6, P8=4) and `record(pass_id=...)` "
            "raises `BudgetExceeded` and writes NOTHING when a campaign would "
            "exceed its allocation -- not even the prefix that would fit, "
            "because recording part of a comparison charges the DSR for arms "
            "the researcher never got to compare. Idempotent re-runs stay "
            "free; trials recorded before the budget existed keep a `pre-v10` "
            "bucket with no allocation and are still charged by "
            "`effective_trials`.",
        regression_test="tests/test_trial_budget.py",
        before_after="an unbudgeted campaign was silently counted -> it is "
                     "refused, with the arithmetic on the exception",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
)


def _validate() -> None:
    """Run at import. A register that can hold a contradiction is a document.

    Two rules, both learned from this audit: a finding cannot be claimed FIXED
    without naming the test that fails when the fix is reverted (four of the
    dossier's fourteen 'fixed' items were absent from the source), and it
    cannot claim its coefficients moved without accepting that history moved
    too.
    """
    seen = set()
    for f in REGISTER:
        if f.fid in seen:
            raise ValueError(f"duplicate finding id {f.fid}")
        seen.add(f.fid)
        if f.status in (Status.FIXED, Status.BUILT_OFF) and not f.regression_test:
            raise ValueError(
                f"{f.fid} is marked {f.status.value} with no regression test. "
                f"A fix nothing would catch the reversion of is a claim, not a "
                f"fix."
            )
        if f.moves_coefficients and not f.moves_history:
            raise ValueError(
                f"{f.fid} moves coefficients but claims history is unchanged. "
                f"A refit supersedes every result computed under the old fit."
            )
        for name in ("root_cause", "location", "fix", "before_after"):
            if not getattr(f, name).strip():
                raise ValueError(f"{f.fid} has no {name}")


_validate()


def by_id(fid: str) -> Optional[Finding]:
    return next((f for f in REGISTER if f.fid.upper() == str(fid).upper()), None)


def open_findings() -> List[Finding]:
    return [f for f in REGISTER if not f.resolved]


def restart_blockers() -> List[Finding]:
    """Every finding that forces the forward test to restart, resolved or not."""
    return [f for f in REGISTER if f.forces_restart]


def unresolved_restart_blockers() -> List[Finding]:
    """The ones that still block. This is what the restart gate reads.

    R1 is itself a restart blocker and is OPEN by construction -- it IS the
    restart -- so it is excluded here; a gate that included it could never
    open.
    """
    return [f for f in REGISTER
            if f.forces_restart and not f.resolved and f.fid != "R1"]


def categorised() -> Dict[str, List[Finding]]:
    out: Dict[str, List[Finding]] = {}
    for f in REGISTER:
        out.setdefault(f.category.value, []).append(f)
    return dict(sorted(out.items()))
