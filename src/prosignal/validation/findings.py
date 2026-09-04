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
