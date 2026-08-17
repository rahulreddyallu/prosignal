"""The config layer is the user's entire interface, so it is tested hardest.

The property under test throughout: **a bad edit to parameters.yaml fails
loudly at load time.** Silent fallback to a hidden default is the failure mode
that would let a fat-fingered threshold reach a live decision unnoticed.
"""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from prosignal.config.loader import config_hash, load_config
from prosignal.config.schema import ParamStatus, RootConfig, Tunable
from prosignal.core.errors import ConfigError

from .conftest import write_config


# =============================================================================
# Tunable
# =============================================================================


def test_tunable_accepts_bare_scalar():
    t = Tunable[float].model_validate(2.5)
    assert t.value == 2.5
    assert t.status is ParamStatus.UNVALIDATED
    assert t.v == 2.5


def test_tunable_accepts_long_form():
    t = Tunable[float].model_validate(
        {"value": 2.5, "status": "UNVALIDATED", "search_range": [1.5, 3.5], "note": "hi"}
    )
    assert t.value == 2.5
    assert t.search_range == [1.5, 3.5]


def test_tunable_rejects_value_outside_search_range():
    with pytest.raises(Exception) as exc:
        Tunable[float].model_validate({"value": 9.0, "search_range": [1.5, 3.5]})
    assert "search_range" in str(exc.value)


def test_tunable_rejects_inverted_search_range():
    with pytest.raises(Exception):
        Tunable[float].model_validate({"value": 2.0, "search_range": [3.5, 1.5]})


def test_validated_status_requires_provenance():
    """A parameter cannot claim validation without naming the trial."""
    with pytest.raises(Exception) as exc:
        Tunable[float].model_validate({"value": 2.5, "status": "VALIDATED"})
    assert "validated_by" in str(exc.value)

    ok = Tunable[float].model_validate(
        {
            "value": 2.5,
            "status": "VALIDATED",
            "validated_by": "T-014",
            "validated_on": "2026-09-02",
        }
    )
    assert ok.status is ParamStatus.VALIDATED
    assert ok.validated_on == dt.date(2026, 9, 2)


def test_tunable_rejects_unknown_metadata_key():
    with pytest.raises(Exception):
        Tunable[float].model_validate({"value": 2.5, "serach_range": [1, 3]})


# =============================================================================
# whole-file loading
# =============================================================================


def test_baseline_config_loads(cfg):
    assert cfg.params.universe.index_name.value == "NIFTY 200"
    assert cfg.params.stage4_core_score.weighting_mode.value == "equal_weight"
    assert len(cfg.params.iter_tunables()) > 100


def test_config_hash_is_stable_and_ignores_prose(tmp_project, baseline_yaml):
    reset = copy.deepcopy(baseline_yaml)
    write_config(tmp_project, reset)
    a = load_config(project_root=tmp_project, use_cache=False)

    changed = copy.deepcopy(baseline_yaml)
    changed["meta"]["description"] = "totally different prose here"
    changed["stage7_risk"]["stop_loss"]["atr_multiple"]["note"] = "different note"
    write_config(tmp_project, changed)
    b = load_config(project_root=tmp_project, use_cache=False)

    assert a.hash == b.hash, "editing prose must not change the config hash"


def test_config_hash_changes_when_a_value_changes(tmp_project, baseline_yaml):
    changed = copy.deepcopy(baseline_yaml)
    write_config(tmp_project, changed)
    a = load_config(project_root=tmp_project, use_cache=False)

    changed["stage7_risk"]["stop_loss"]["atr_multiple"]["value"] = 2.0
    write_config(tmp_project, changed)
    b = load_config(project_root=tmp_project, use_cache=False)

    assert a.hash != b.hash


def test_unknown_key_is_rejected(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["stage7_risk"]["stop_loss"]["atr_multiplier_typo"] = 2.5
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "atr_multiplier_typo" in str(exc.value)


def test_missing_feed_policy_is_rejected(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    del bad["feeds"]["india_vix"]
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "india_vix" in str(exc.value)


def test_unknown_feed_name_is_rejected(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["feeds"]["moon_phase"] = {"max_age_sessions": 1, "required": False}
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError):
        load_config(project_root=tmp_project, use_cache=False)


# =============================================================================
# cross-section invariants
# =============================================================================


def test_history_shorter_than_longest_lookback_is_rejected(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["universe"]["min_history_sessions"]["value"] = 100
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "longest lookback" in str(exc.value)


def test_purge_shorter_than_label_horizon_is_rejected(tmp_project, baseline_yaml):
    """Purge < label horizon means training labels overlap the test window."""
    bad = copy.deepcopy(baseline_yaml)
    bad["validation"]["label"]["forward_return_sessions"]["value"] = 42
    bad["validation"]["cpcv"]["purge_sessions"]["value"] = 21
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "purge_sessions" in str(exc.value)


def test_order_placement_interlock_cannot_be_enabled(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["api"]["allow_order_placement"] = True
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "decision-support" in str(exc.value)


def test_estimate_revision_factor_cannot_carry_weight(tmp_project, baseline_yaml):
    """No timestamped India analyst feed exists, so this must stay at zero."""
    bad = copy.deepcopy(baseline_yaml)
    bad["stage4_core_score"]["factors"]["estimate_revision_momentum"]["enabled"] = True
    bad["stage4_core_score"]["factors"]["estimate_revision_momentum"]["weight_band"]["value"] = [
        0.1,
        0.2,
    ]
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "point-in-time India analyst data" in str(exc.value)


def test_earnings_hard_reject_cannot_be_relaxed_without_pead_flag(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["stage5_false_signal"]["earnings_distortion"]["action"] = "penalty"
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "pead_conditional_signal_enabled" in str(exc.value)


def test_momentum_skip_must_be_shorter_than_lookback(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["stage4_core_score"]["factors"]["momentum_12_1"]["skip_sessions"]["value"] = 300
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError):
        load_config(project_root=tmp_project, use_cache=False)


def test_regime_bucket_reference_must_exist(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["stage2_regime"]["no_new_entry_buckets"]["value"] = ["not_a_real_bucket"]
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError) as exc:
        load_config(project_root=tmp_project, use_cache=False)
    assert "not_a_real_bucket" in str(exc.value)


def test_regime_multiplier_rows_must_be_triples(tmp_project, baseline_yaml):
    bad = copy.deepcopy(baseline_yaml)
    bad["stage2_regime"]["multipliers"]["table"]["uptrend_lowvol"] = [1.0, 1.0]
    write_config(tmp_project, bad)
    with pytest.raises(ConfigError):
        load_config(project_root=tmp_project, use_cache=False)


# =============================================================================
# derived values & transparency
# =============================================================================


def test_position_value_splits_capital_when_unset(cfg):
    expected = (
        cfg.params.capital.total_capital_inr.value
        / cfg.params.capital.max_open_positions.value
    )
    assert cfg.params.capital.position_value_inr() == pytest.approx(expected)


def test_transparency_report_counts_match(cfg):
    report = cfg.transparency_report()
    assert report["total_parameters"] == len(report["parameters"])
    assert report["unvalidated_count"] == cfg.params.unvalidated_count()
    assert report["unvalidated_count"] > 0, (
        "a config with zero UNVALIDATED parameters means something was promoted "
        "without a CPCV run behind it"
    )


def test_tunable_lookup_by_path(cfg):
    entry = cfg.tunable("stage7_risk.stop_loss.atr_multiple")
    assert entry["status"] == "UNVALIDATED"
    assert entry["search_range"] == [1.5, 3.5]
    with pytest.raises(ConfigError):
        cfg.tunable("no.such.parameter")


def test_local_overlay_is_merged(tmp_project, baseline_yaml):
    write_config(tmp_project, baseline_yaml)
    overlay = tmp_project / "config" / "parameters.local.yaml"
    overlay.write_text(
        "capital:\n  total_capital_inr:\n    value: 5000000\n    status: OPERATIONAL\n",
        encoding="utf-8",
    )
    cfg = load_config(project_root=tmp_project, use_cache=False)
    assert cfg.params.capital.total_capital_inr.value == 5_000_000
    # Untouched sections survive the merge.
    assert cfg.params.universe.index_name.value == "NIFTY 200"
