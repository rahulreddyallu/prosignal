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
    #: THE ONLY THING THAT CLOSES THIS IS THE FORWARD TEST ITSELF.
    #:
    #: A gate that refuses to open a window while such a finding is open can
    #: never open, because the finding is the window. R1 was hardcoded into two
    #: gates by fid for exactly this reason, which worked until a second
    #: finding of the same kind arrived: T5 says the shipped configuration does
    #: not clear the Deflated Sharpe, and the evidence that would settle it is
    #: out-of-sample observations that do not exist yet.
    #:
    #: Declared here rather than listed in the gate so the exemption is visible
    #: where a reader meets the finding, and so claiming it is a deliberate act
    #: recorded in the register rather than an id buried in a conditional. It
    #: is NOT a way to park an inconvenient finding: a finding may claim this
    #: only when no amount of work on this tree could resolve it.
    resolved_by_forward_test: bool = False

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
        resolved_by_forward_test=True,
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
    # -- the readiness dossier's own open items -------------------------------
    _f(
        fid="W2", severity="high",
        title="Every traded coefficient is a survivor of a selection on its "
              "own t-statistic",
        category=Category.MODEL, status=Status.BUILT_OFF,
        root_cause="`gated_shrink` zeroes any theme below `significance_floor` "
                   "and prices the rest, so the surviving estimate is biased "
                   "away from zero by exactly the amount the gate removed. The "
                   "overstatement is largest for the themes that only just "
                   "cleared, which are the ones the decision turns on.",
        location="prosignal.validation.selection::correct_t",
        fix="One-sided truncated-normal MLE of the true non-centrality. Six "
            "acceptance criteria were fixed before it was written; five hold "
            "and the sixth -- that it recover the truth in simulation -- does "
            "not, so the number is REPORTED and NOT TRADED. "
            "`assert_not_traded` makes wiring it into a score fail a test "
            "rather than pass a review.",
        regression_test="tests/test_selection_correction.py",
        before_after="mom_f t +2.87 -> +2.20 implied true; delivery_f +2.63 -> "
                     "+1.44, i.e. one of the two traded coefficients does not "
                     "clear the floor it was selected by once selection is "
                     "priced. On this tree's post-R9 refit both clear.",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="The criterion it fails is asserted AS a failure in the test "
              "suite, so improving the correction forces the disposition to be "
              "re-decided in the open rather than drifting.",
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
    # -------------------------------------------------------------------------
    # T-numbers: the tuning pass. These are not defects found by reading the
    # code; they are findings from MEASURING what the code does, at trade level,
    # against an investable benchmark. Each one changes what the engine trades.
    # -------------------------------------------------------------------------
    _f(
        fid="T1", severity="critical",
        title="The fitted composite lost to an equal-weight benchmark in every "
              "one of its configurations",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="The composite is fitted against the CROSS-SECTIONAL RANK "
                   "of the forward outcome. Rank rewards ordering the middle of "
                   "the distribution; the cross-section is strongly "
                   "right-skewed (+1.87 at 63 sessions) and a book of six names "
                   "out of seven hundred lives entirely in the tail a rank "
                   "target is indifferent to. Measured at H=63 its rank IC is "
                   "+0.0338 while its TOP-DECILE excess is -0.35% (t -0.28): it "
                   "orders the universe adequately on average and mis-orders "
                   "the only part of it that is ever bought.",
        location="prosignal.stages.stage4_core_score::run",
        fix="`stage4_core_score.ranking` separates WHAT THE THEMES ARE WORTH "
            "from WHAT ORDERS THE BOOK. The book is ordered by mom_6_1_r, a "
            "sector-neutral rank the model already computes. The composite is "
            "still fitted, recorded, attributed on the card and monitored by "
            "`research decay` -- it just does not choose. Three repairs were "
            "tried first and each failed on measurement: refitting on the "
            "RETURN (positive alpha in 6.2% of 960 configurations), using the "
            "composite as an exclusion filter (every level cost return), and "
            "trading the engine's own three-column momentum FAMILY (positive "
            "in 33% against the single column's 86%).",
        regression_test="tests/test_production_build.py::TestTheRankingPolicy",
        before_after="best of 144 composite configurations -5.2% alpha; "
                     "mom_6_1_r +20.3% at excess Sharpe 1.12",
        moves_coefficients=True, moves_history=True, forces_restart=True,
        notes="4,877 trade-level configurations on a rebuilt point-in-time "
              "panel: 2,212 sessions, 1,517 ever-eligible symbols, entries on "
              "cron dates only, exits checked every session, scored against an "
              "investable equal-weight benchmark of the same eligible universe.",
    ),
    _f(
        fid="T2", severity="high",
        title="Three of the four price exits cost more than they saved",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Every rule that closes a position before the time backstop "
                   "removes part of the population the strategy exists to hold: "
                   "39% of positions reach the limit, they win 69% of the time "
                   "and average +16.1% net, against +3.3% for those that leave "
                   "on the rank band. The shipped geometry -- 2.5 ATR stop, 3R "
                   "target, MA50 - 1.5 ATR invalidation -- sold a large share "
                   "of them early.",
        location="prosignal.stages.stage7_risk::_exit_hierarchy",
        fix="Each exit measured ALONE rather than removed as a bundle. Target "
            "and invalidation disarmed; the stop walked out to a disaster floor "
            "at 8 ATR clipped to 35% of entry, which the paired test across 54 "
            "configurations shows ADDING 2.0 points of annual alpha (better in "
            "52 of 54) while cutting the worst single trade by 21.5 points. "
            "The invalidation LEVEL is retained as the admission predicate, so "
            "the population the model is fitted on did not widen.",
        regression_test="tests/test_production_build.py::TestTheShippedExitGeometry",
        before_after="p_win 0.384 -> 0.578, annual alpha +3.1% -> +20.3%",
        moves_coefficients=True, moves_history=True, forces_restart=True,
        notes="Ablation, 258-385 trades each: floor only 0.578/+20.3%; +3R "
              "target 0.578/+19.4%; +1.5R target 0.571/+15.7%; +MA50-3.0 ATR "
              "0.526/+14.7%; +MA50-1.5 ATR 0.422/+6.0%; shipped v4 geometry "
              "0.384/+3.1%.",
    ),
    _f(
        fid="T3", severity="high",
        title="Running daily and buying daily were the same event, and only one "
              "of them had been measured",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="The engine had no way to express 'check every session, open "
                   "on a cadence'. A daily entry clock and a 21-session entry "
                   "clock are different strategies, not the same strategy "
                   "sampled differently.",
        location="prosignal.cadence::resolve",
        fix="An entry clock counted in SESSIONS from a fixed anchor against the "
            "exchange calendar, so a holiday cannot re-phase it and the live "
            "schedule reproduces the backtested one. Exits, the disaster floor "
            "and outcome resolution still run every session; the cron is "
            "unchanged and deliberately so. An unresolvable clock fails OPEN, "
            "because a clock stuck closed stops the book silently and looks "
            "exactly like a market with no candidates.",
        regression_test="tests/test_production_build.py::TestTheEntryClock",
        before_after="entries every session -> every 21 sessions; the only stem "
                     "on the surface positive in all six calendar years",
        moves_coefficients=False, moves_history=True, forces_restart=True,
    ),
    _f(
        fid="T4", severity="medium",
        title="The engine recorded what it liked and what happened, never what "
              "it expected",
        category=Category.VALIDATION, status=Status.FIXED,
        root_cause="Nothing on a run stated a cadence, a planned hold or an "
                   "expected outcome, so a resolved trade could be compared "
                   "with the market and never with the engine's own claim. "
                   "Calibration was impossible by construction.",
        location="prosignal.tradeplan::build_trade_plan",
        fix="Every issued trade carries a `TradePlan`: the cadence, the planned "
            "hold, the risk at the floor, and the frozen frequencies of the "
            "study named on it. Recorded IN THE LEDGER ROW rather than read "
            "from the config later, so a trade issued in March is never "
            "explained by August's settings. The frequencies are a population, "
            "never a per-name forecast, and the caveat saying so is a required "
            "field.",
        regression_test="tests/test_production_build.py::TestTheTradePlan",
        before_after="no expectation recorded -> p_win, p_beat, mean and median "
                     "return, cost sensitivity and basis on every trade",
        moves_coefficients=False, moves_history=False, forces_restart=False,
    ),
    _f(
        fid="T5", severity="high",
        title="The shipped configuration does not clear the Deflated Sharpe, "
              "and is shipped anyway",
        category=Category.VALIDATION, status=Status.OPEN,
        root_cause="It was chosen by looking at 4,877 configurations. Against "
                   "that trial count with the trial variance measured across "
                   "all of them, the DSR threshold is an annual excess Sharpe "
                   "of 1.80 and this configuration has 1.12, giving DSR 0.030.",
        location="prosignal.validation.metrics::deflated_sharpe_ratio",
        fix="NOT FIXED, and recorded as open rather than argued away. Three "
            "readings are published together because choosing one silently is "
            "how a search gets laundered: 0.030 against all 4,877 trials, 0.50 "
            "with the trial variance measured within the family the winner came "
            "from, 0.97 within the final surface. The spread is driven by the "
            "variance, and the DSR's null -- every trial has zero true Sharpe "
            "and they differ only by noise -- is false when the trials include "
            "signals that differ for real reasons. What supports shipping is "
            "different evidence: every one of the 378 surface cells has "
            "positive mean excess over 2021-2026, the shipped cell is positive "
            "in all six years, PBO is 0.388, and the block bootstrap puts "
            "annual alpha at [+7.7%, +30.9%] with P(alpha<=0) = 0.000. That is "
            "a hypothesis worth forward-testing on paper, not an established "
            "result, and the forward test is the fix.",
        resolved_by_forward_test=True,
        regression_test="tests/test_production_build.py::TestTheShippedExitGeometry",
        before_after="DSR 0.000 on the previous configuration; 0.030 on this "
                     "one, still short of the bar",
        moves_coefficients=False, moves_history=False, forces_restart=False,
        notes="Deliberately OPEN. Closing it would claim a promotion that has "
              "not happened.",
    ),
    _f(
        fid="T6", severity="medium",
        title="A rejected name inside the entry band leaves the slot empty "
              "rather than passing it down",
        category=Category.VALIDATION, status=Status.DISCLOSED,
        root_cause="The band is `model_rank <= entry_rank`, a THRESHOLD rather "
                   "than a queue. When Stage 5 hard-rejects a name inside it -- "
                   "which it did on the verification run, at rank 5 of 6 -- the "
                   "name at rank 7 is outside the band and cannot take the "
                   "slot, so the book runs light and the capital sits in cash.",
        location="prosignal.stages.stage6_entry::_admit",
        fix="DISCLOSED, not changed. Backfilling is a strategy change: the "
            "trade-level study filled K slots from the top of the ranking "
            "because it had no false-signal defense to reject anybody, so its "
            "96.5% fill is not a measurement of the engine that ships. Every "
            "card now names the ranks that went missing and says the capital "
            "stays in cash for the cycle, which puts the drag in the record "
            "instead of leaving it to surface later as an unexplained gap. "
            "Whether to backfill is an operator decision that needs its own "
            "measurement against the defense that actually runs.",
        regression_test="tests/test_production_build.py::TestTheShippedExitGeometry",
        before_after="the shortfall was visible only as a funnel arithmetic "
                     "step; it is now stated on the card",
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

    A finding whose only resolution is the forward test is excluded, because a
    gate that included one could never open -- the finding IS the window. That
    exemption is declared on the finding (`resolved_by_forward_test`) rather
    than matched by id here, so it is visible in the register and a second such
    finding does not need this function edited.
    """
    return [f for f in REGISTER
            if f.forces_restart and not f.resolved
            and not f.resolved_by_forward_test]


def categorised() -> Dict[str, List[Finding]]:
    out: Dict[str, List[Finding]] = {}
    for f in REGISTER:
        out.setdefault(f.category.value, []).append(f)
    return dict(sorted(out.items()))
