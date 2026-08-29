"""Revert each fix in the source and require its guard to go red.

A guard that has never been seen to fail is not a guard, it is a comment that
runs. Each mutation below is a minimal, faithful reversion of one repair -- the
line as it actually stood before -- applied to the file on disk, the named tests
run, and the file restored.

PASS means the test went RED under the mutation, i.e. the guard works.
FAIL means the test stayed green with the defect reinstated, i.e. it does not.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"

# (name, file, before -> after, tests that must fail)
MUTATIONS = [
    (
        "DSR fallback variance back to 1/(n-1)",
        "src/prosignal/validation/metrics.py",
        "        # A true unit variance, not 1/(n-1). See the docstring.\n"
        "        sr_var, source = 1.0, SR_VAR_UNIT",
        "        sr_var, source = 1.0 / max(n - 1, 1), SR_VAR_UNIT",
        ["tests/test_measurement_guards.py::"
         "test_the_dsr_fallback_variance_is_a_unit_not_one_over_n"],
    ),
    (
        "DSR variance provenance not carried",
        "src/prosignal/validation/metrics.py",
        "        sr_var, source = measured_var, SR_VAR_FROM_TRIALS",
        "        sr_var, source = 1.0, SR_VAR_UNIT",
        ["tests/test_measurement_guards.py::"
         "test_the_dsr_says_where_its_variance_came_from"],
    ),
    (
        "deflated() scores the pooled (split, date) vector again",
        "src/prosignal/validation/harness.py",
        "            self.independent_excess(horizon_sessions, step_sessions),",
        "            self.excess,",
        ["tests/test_measurement_guards.py::"
         "test_the_dsr_scores_independent_windows_not_duplicated_pairs"],
    ),
    (
        "excess and its dates allowed to drift",
        "src/prosignal/validation/harness.py",
        "        if len(self.excess_dates) != len(self.excess):",
        "        if False:",
        ["tests/test_measurement_guards.py::"
         "test_excess_and_its_dates_cannot_drift_apart_silently"],
    ),
    (
        "run_cpcv stops recording the date of each score",
        "src/prosignal/validation/harness.py",
        "                result.excess_dates.append(d)",
        "                pass",
        ["tests/test_measurement_guards.py::"
         "test_run_cpcv_records_the_date_of_every_score"],
    ),
    (
        "simulator reads the target on the close again (high=None)",
        "src/prosignal/validation/portfolio_sim.py",
        "            outcome = _hold(sym, i, close, low, open_, ma, atr, params, high=high)",
        "            outcome = _hold(sym, i, close, low, open_, ma, atr, params, high=None)",
        ["tests/test_portfolio_sim.py::"
         "test_the_book_itself_takes_profit_on_an_intraday_spike"],
    ),
    (
        "_hold stops forwarding high to the resolver",
        "src/prosignal/validation/portfolio_sim.py",
        "                        high=(high[one] if high is not None else None),",
        "                        high=None,",
        ["tests/test_portfolio_sim.py::test_the_target_is_read_on_the_high_not_the_close"],
    ),
    (
        "re-entry after an early exit is free again",
        "src/prosignal/validation/portfolio_sim.py",
        "            if reopened is None or reopened != EXIT_TIMEOUT:",
        "            if reopened is None:",
        ["tests/test_portfolio_sim.py::"
         "test_a_position_that_closed_early_pays_again_when_it_is_re_bought"],
    ),
    (
        "held loses the exit side and carries a bare 1 again",
        "src/prosignal/validation/portfolio_sim.py",
        '        held = {s: outcomes.get(s, float("nan")) for s in book}',
        "        held = {s: 1 for s in book}",
        ["tests/test_portfolio_sim.py::"
         "test_a_position_carried_through_the_horizon_pays_nothing_to_keep"],
    ),
    (
        "an unfilled slot becomes a free entry",
        "src/prosignal/validation/portfolio_sim.py",
        '        held = {s: outcomes.get(s, float("nan")) for s in book}',
        "        held = {s: outcomes.get(s, EXIT_TIMEOUT) for s in book}",
        ["tests/test_portfolio_sim.py::"
         "test_a_slot_that_never_filled_pays_when_it_finally_does"],
    ),
    (
        "worst single schedule replaced by the mean of phases",
        "src/prosignal/validation/portfolio_sim.py",
        '        "worst_schedule_drawdown": float(np.min(drawdowns)),',
        '        "worst_schedule_drawdown": float(np.mean(drawdowns)),',
        ["tests/test_portfolio_sim.py::"
         "test_the_worst_schedule_drawdown_is_reported_not_only_the_mean"],
    ),
    (
        "a missing high panel absorbed silently again",
        "src/prosignal/validation/portfolio_sim.py",
        "        warnings.warn(",
        "        _ = lambda *a, **k: None; _(",
        ["tests/test_portfolio_sim.py::"
         "test_a_missing_high_panel_is_warned_about_rather_than_absorbed"],
    ),
    (
        "sacred holdout stops being enforced",
        "src/prosignal/cli.py",
        '    if bool(getattr(cfg.params.validation.holdout, "sacred", False)):',
        "    if False:",
        ["tests/test_measurement_guards.py::"
         "test_sacred_holdout_refuses_include_holdout"],
    ),
    (
        "trial scores dropped on the way to disk",
        "src/prosignal/validation/registry.py",
        '            row["score"] = float(self.score)',
        "            pass",
        ["tests/test_measurement_guards.py::"
         "test_the_registry_records_what_each_trial_scored"],
    ),
    (
        "mismatched scores zipped short instead of refused",
        "src/prosignal/validation/registry.py",
        "        if scores is not None and len(scores) != len(labels):",
        "        if False:",
        ["tests/test_measurement_guards.py::"
         "test_scores_that_do_not_line_up_with_labels_are_refused"],
    ),
    (
        "tertiary hypothesis dropped out of the fingerprint",
        "src/prosignal/validation/forward.py",
        '            out["tertiary"] = self.tertiary',
        '            out["_tertiary_not_hashed"] = ""',
        ["tests/test_measurement_guards.py::"
         "test_the_tertiary_hypothesis_is_inside_the_fingerprint"],
    ),
    (
        "a registration with no benchmark hypothesis graded anyway",
        "src/prosignal/validation/forward.py",
        '    if not str(reg.tertiary or "").strip():',
        "    if False:",
        ["tests/test_measurement_guards.py::"
         "test_a_registration_with_no_benchmark_hypothesis_is_not_graded"],
    ),
    (
        "register() stops writing the benchmark hypothesis",
        "src/prosignal/validation/forward.py",
        "        tertiary=(",
        "        notes=[] and (",
        ["tests/test_measurement_guards.py::"
         "test_the_shipped_registration_text_names_the_benchmark"],
    ),
    (
        "exit_hierarchy switches stop reaching the measurement",
        "src/prosignal/features/exits.py",
        "        use_stop=_armed(h.stop_loss_breach),",
        "        use_stop=True,",
        ["tests/test_exit_agreement.py::TestTheHierarchySwitchesReachTheMeasurement::"
         "test_rules_from_config_carries_the_switches"],
    ),
    (
        "a disarmed stop fires anyway",
        "src/prosignal/features/exits.py",
        "    f_stop = _first(hit_stop) if rules.use_stop else never",
        "    f_stop = _first(hit_stop)",
        ["tests/test_exit_agreement.py::TestTheHierarchySwitchesReachTheMeasurement::"
         "test_disarming_the_stop_removes_stop_exits"],
    ),
    (
        "the panel stops applying the admission mask",
        "src/prosignal/features/crosssec.py",
        "        if admission_rules is not None and adm_atr is not None:",
        "        if False:",
        ["tests/test_admission_population.py::"
         "test_the_panel_can_be_restricted_to_what_the_book_can_open"],
    ),
    (
        "under-covered trial scores set the multiple-testing bar",
        "src/prosignal/validation/metrics.py",
        "    elif len(scored) > 1 and covered >= MIN_TRIAL_SCORE_COVERAGE:",
        "    elif len(scored) > 1:",
        ["tests/test_measurement_guards.py::"
         "test_trial_scores_from_a_fraction_of_the_search_do_not_set_the_bar"],
    ),
    (
        "an existing trial's score can be overwritten by a re-run",
        "src/prosignal/validation/registry.py",
        "                if prior.score is None and score is not None:",
        "                if score is not None:",
        ["tests/test_measurement_guards.py::"
         "test_an_existing_trial_can_gain_a_score_but_not_a_second_count"],
    ),
    (
        "filling in a score counts as a new trial",
        "src/prosignal/validation/registry.py",
        "        return added",
        "        return len(fresh)",
        ["tests/test_measurement_guards.py::"
         "test_an_existing_trial_can_gain_a_score_but_not_a_second_count"],
    ),
    # -- second pass: the provenance, gate and partition repairs -------------
    (
        "R9: the live refit stops applying the admission predicate",
        "src/prosignal/features/crossmodel.py",
        "                        admission_rules=admission_rules,\n",
        "",
        ["tests/test_admissible_population_r9.py::test_the_live_refit_applies_the_admission_predicate"],
    ),
    (
        "R9: the fit stops recording which population it was estimated on",
        "src/prosignal/features/crossmodel.py",
        '        "population": ("admissible" if admission_rules is not None\n'
        '                       else "all_eligible"),\n',
        "",
        ["tests/test_admissible_population_r9.py::test_the_fit_records_which_population_it_was_estimated_on",
         "tests/test_admissible_population_r9.py::test_a_wide_population_cache_is_refused_after_the_switch_flips"],
    ),
    (
        "R9: stage 4 stops reading the admission policy",
        "src/prosignal/stages/stage4_core_score.py",
        "        admission_rules = (rules_from_config(cfg, risk_cfg)\n"
        "                           if admit_only and risk_cfg is not None else None)",
        "        admission_rules = None",
        ["tests/test_admissible_population_r9.py::"
         "test_the_live_refit_asks_the_cache_for_the_admissible_population"],
    ),
    (
        "R13: unknown liquidity goes back to the half spread alone",
        "src/prosignal/costs.py",
        "            participation = float(_fv(m.unknown_liquidity_participation))",
        "            return float(_fv(m.assumed_half_spread_bps))",
        ["tests/test_liquidity_gate.py::test_unknown_liquidity_is_never_the_cheapest_answer"],
    ),
    (
        "R13: an unmeasured reading becomes a zero one",
        "src/prosignal/liquidity.py",
        "        return LiquidityView(LiquidityState.MISSING, None, None, age_sessions,\n"
        '                             "no ADTV was computed for this name on this date")',
        "        return LiquidityView(LiquidityState.INVALID, None, None, age_sessions,\n"
        '                             "no ADTV was computed for this name on this date")',
        ["tests/test_liquidity_gate.py::test_missing_is_not_zero_and_neither_is_tradable"],
    ),
    (
        "R13: the simulator sizes an unmeasured name again",
        "src/prosignal/validation/portfolio_sim.py",
        "    refuse_unknown_liquidity: bool = True",
        "    refuse_unknown_liquidity: bool = False",
        ["tests/test_liquidity_gate.py::"
         "test_the_simulator_refuses_the_same_names_the_live_sizer_does"],
    ),
    (
        "W2: the correction conditions two-sided again",
        "src/prosignal/validation/selection.py",
        "    def g(m: float) -> float:\n"
        "        return m + _lambda(gate - m) - t",
        "    def g(m: float) -> float:\n"
        "        num = _phi(gate - m) - _phi(-gate - m)\n"
        "        den = 1.0 - norm_cdf(gate - m) + norm_cdf(-gate - m)\n"
        "        return m + (num / den if den > 1e-300 else 0.0) - t",
        ["tests/test_selection_correction.py::test_the_conditioning_is_one_sided",
         "tests/test_selection_correction.py::"
         "test_even_the_correct_estimator_over_reports_a_true_zero"],
    ),
    (
        "W2: the scoring path is allowed to reach the correction",
        "src/prosignal/validation/selection.py",
        '    "prosignal.validation",\n)',
        '    "prosignal",\n)',
        ["tests/test_selection_correction.py::"
         "test_the_scoring_path_cannot_reach_the_correction"],
    ),
    (
        "C3: outcomes stop carrying the epoch that produced them",
        "src/prosignal/outcomes.py",
        '        "epoch_id": item.get("epoch_id") or _active_epoch_id(),',
        '        "epoch_id": None,',
        ["tests/test_outcome_epochs.py::test_a_row_written_now_carries_the_open_epoch"],
    ),
    (
        "C3: load_outcomes stops partitioning on the epoch",
        "src/prosignal/outcomes.py",
        '    return [r for r in rows if (r.get("epoch_id") or PRE_EPOCH) == want]',
        "    return rows",
        ["tests/test_outcome_epochs.py::test_loading_serves_one_epoch"],
    ),
    (
        "C4: the endpoint serves one pooled summary again",
        "src/prosignal/api.py",
        '        every = _out.load_outcomes(path, epoch="*")',
        "        every = _out.load_outcomes(path)",
        ["tests/test_outcome_epochs.py::test_the_endpoint_labels_retired_cohorts"],
    ),
    (
        "R1: register stops gating on readiness",
        "src/prosignal/validation/forward.py",
        "        check_may_restart(cfg)",
        "        pass",
        ["tests/test_restart_gate.py::test_register_is_blocked_by_the_gate_it_calls"],
    ),
    (
        "R1: the gate reports only the first blocker",
        "src/prosignal/validation/readiness.py",
        "    r = assess(cfg)\n    for name in RESTART_GATES:",
        "    if out:\n        return out\n    r = assess(cfg)\n    for name in RESTART_GATES:",
        ["tests/test_restart_gate.py::test_both_halves_of_the_gate_are_reported_together"],
    ),
    (
        "R1: an undecidable gate counts as a pass",
        "src/prosignal/validation/readiness.py",
        "    @property\n    def ready(self) -> bool:\n        return all(g.passed for g in self.gates)",
        "    @property\n    def ready(self) -> bool:\n        return all(g.passed or g.unknown for g in self.gates)",
        ["tests/test_restart_gate.py::test_an_undecidable_gate_is_not_a_pass"],
    ),
    (
        "D1: the manifest digest is read from the file instead of recomputed",
        "src/prosignal/data/manifest.py",
        "    # RECOMPUTED, never read from the file. A digest a manifest asserts about\n"
        "    # itself is not evidence; a digest derived from its contents is.\n"
        "    m.digest = m.compute_digest()",
        '    m.digest = str(blob.get("digest", ""))',
        ["tests/test_data_manifest.py::"
         "test_a_hand_edited_manifest_cannot_claim_a_digest_it_does_not_have"],
    ),
    (
        "D1: mtime enters the digest",
        "src/prosignal/data/manifest.py",
        '        d = {k: v for k, v in asdict(self).items() if k != "mtime_ms"}',
        "        d = dict(asdict(self))",
        ["tests/test_data_manifest.py::"
         "test_the_digest_is_over_content_and_not_over_timestamps"],
    ),
    (
        "D2: an epoch can be closed with no reason",
        "src/prosignal/validation/epoch.py",
        "    if not str(reason).strip():",
        "    if False:",
        ["tests/test_epoch.py::test_an_epoch_cannot_be_closed_without_a_reason"],
    ),
    (
        "D2: closing an epoch rewrites its line instead of appending",
        "src/prosignal/validation/epoch.py",
        "        if e.epoch_id not in latest:\n            order.append(e.epoch_id)",
        "        if e.epoch_id not in latest:\n            order.append(e.epoch_id)\n        "
        "        blob = blob  # noqa",
        ["tests/test_epoch.py::test_closing_appends_rather_than_rewrites"],
    ),
    (
        "D2: two epochs may be open at once",
        "src/prosignal/validation/epoch.py",
        "    if not allow_while_open:\n        live = active(ledger_root)",
        "    if False:\n        live = active(ledger_root)",
        ["tests/test_epoch.py::test_two_epochs_cannot_be_open_at_once"],
    ),
]


def run(tests) -> bool:
    """True when at least one named test FAILED."""
    proc = subprocess.run(
        [str(PY), "-m", "pytest", "-q", "-p", "no:randomly", "-x", *tests],
        cwd=ROOT, capture_output=True, text=True)
    return proc.returncode != 0


def main() -> int:
    caught = missed = 0
    for name, rel, before, after, tests in MUTATIONS:
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if before not in original:
            print(f"  SKIP  {name}\n        anchor not found in {rel}")
            missed += 1
            continue
        if original.count(before) != 1:
            print(f"  SKIP  {name}\n        anchor appears "
                  f"{original.count(before)} times in {rel}")
            missed += 1
            continue
        path.write_text(original.replace(before, after), encoding="utf-8")
        try:
            went_red = run(tests)
        finally:
            path.write_text(original, encoding="utf-8")
        if went_red:
            caught += 1
            print(f"  caught  {name}")
        else:
            missed += 1
            print(f"  MISSED  {name}\n          {tests} stayed green with the "
                  f"defect reinstated")
    print(f"\n{caught} of {caught + missed} mutations caught")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
