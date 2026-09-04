"""The v2 forward registration, and the hash that freezes it.

WHY IT WAS RE-REGISTERED. `prosignal research forward` reported the previous
window INVALID on five counts, not the three the build plan quoted: the config
had changed after registration, the pre-registration no longer matched its hash,
the registration carried no benchmark-relative hypothesis, five distinct model
fingerprints had been recorded, and session coverage was 40% against the 60%
its own invalidation clause requires.

WHAT v2 CHANGES, and each is a fix for something measured.

  PRIMARY moves to the long-short SPREAD against an EXTERNAL factor model. The
  v1 primary regressed on six long-short factors built from this engine's own
  ranked columns, which is a circular and easier question.

  SECONDARY becomes the benchmark-relative one: long-leg excess over an
  equal-weight hold of the eligible universe. In v1 that question was a
  `tertiary` bolted on afterwards.

  A FALSIFICATION SET, because at this engine's information ratio an
  eighteen-month window cannot decide either hypothesis, and a registration
  with nothing falsifiable in it is registered to produce no evidence.

  A POWER STATEMENT, written before the first observation, so the eventual
  failure to reach t >= 2.0 reads as a fact about the horizon rather than
  about the strategy.

  INSTRUMENTS_REQUIRED, because the primary names a regressor set that does
  not exist yet. NOT_TESTABLE is a distinct outcome from a failure and it is
  never upgraded to a pass.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from prosignal.validation.forward import (
    REGISTRATION_NAME, SCHEME, Registration, fingerprint_scheme,
    load_registration, progress, register, verify,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "ledger"

WHY = ("fixture: this file tests the registration contract itself, not the "
       "readiness gate that guards it")


def open_v2(tmp_path, started=dt.date(2026, 9, 4), version="cfg@aaa"):
    return register(tmp_path, config_version=version, engine_version="0.1.0",
                    git_commit="deadbeef" * 5, started_on=started,
                    unchecked_reason=WHY)


def row(date, version="cfg@aaa", error=None):
    return {"date": date.isoformat(), "config_version": version, "error": error}


# ------------------------------------------------------------- the v2 contract
def test_a_new_registration_is_written_under_the_v2_scheme(tmp_path):
    assert open_v2(tmp_path).scheme == SCHEME == "v2"


def test_the_primary_is_the_spread_against_an_external_model(tmp_path):
    p = open_v2(tmp_path).primary
    assert "LONG-SHORT SPREAD" in p or "long-short spread" in p.lower()
    assert "EXTERNAL" in p
    assert "2.0" in p and "overlap-corrected" in p
    assert "NOT_TESTABLE" in p, (
        "the primary regresses on a model that does not exist yet and must say "
        "so. Grading it against the self-built regressors would reinstate the "
        "circularity this registration was rewritten to remove.")


def test_the_secondary_is_benchmark_relative(tmp_path):
    s = open_v2(tmp_path).secondary
    assert "equal-weight" in s
    assert "long leg" in s.lower()
    assert "2.0" in s


def test_the_engine_is_registered_to_fail_the_secondary(tmp_path):
    """A forward test whose outcome is not in doubt is not a test."""
    assert "expected to FAIL" in open_v2(tmp_path).secondary


def test_the_falsification_set_is_the_brief_s(tmp_path):
    """What an eighteen-month window can actually decide."""
    text = " ".join(open_v2(tmp_path).falsification).lower()
    for needed in ("rank ic", "monotonic", "decay", "cost", "fill", "breadth"):
        assert needed in text, f"the falsification set does not cover {needed!r}"


def test_a_power_statement_is_registered_before_the_first_observation(tmp_path):
    p = open_v2(tmp_path).power
    assert "sqrt" in p.lower() and "IR" in p
    assert "6.5 years" in p, (
        "the power statement does not state how long t=2.0 actually takes at "
        "the only out-of-sample information ratio this engine has measured. "
        "Without it the registered failure reads as a verdict on the strategy.")
    assert "UNDERPOWERED" in p.upper()


def test_the_instruments_that_do_not_exist_yet_are_named(tmp_path):
    req = open_v2(tmp_path).instruments_required
    assert req, ("the primary needs an external factor model and a long-short "
                 "leg series, neither of which exists. Naming them is what "
                 "keeps the hypothesis NOT_TESTABLE instead of silently graded "
                 "against a substitute.")
    joined = " ".join(req).lower()
    assert "external" in joined and "legs.py" in joined


def test_the_invalidation_conditions_are_all_four_plus_the_benchmark(tmp_path):
    inv = open_v2(tmp_path).invalidation
    joined = " ".join(inv).lower()
    assert "config_version changes" in joined
    assert "retuned" in joined
    assert "real capital" in joined
    assert "60%" in joined
    assert "benchmark panel" in joined


def test_the_config_version_clause_mentions_the_data(tmp_path):
    """P0-3 changed what config_version covers; the clause has to say so."""
    inv = " ".join(open_v2(tmp_path).invalidation)
    assert "DATA" in inv and "TRAINING WINDOW" in inv


# ---------------------------------------------------------------- the hashing
def test_the_criteria_are_hashed_at_registration(tmp_path):
    reg = open_v2(tmp_path)
    blob = json.loads((tmp_path / REGISTRATION_NAME).read_text())
    assert blob["fingerprint"] == reg.fingerprint()
    assert verify(tmp_path)
    assert fingerprint_scheme(tmp_path) == "current"


def test_the_new_fields_are_inside_the_hash(tmp_path):
    """A hypothesis, a falsification criterion or a power claim cannot be
    added, softened or deleted once observations have started landing."""
    reg = open_v2(tmp_path)
    base = reg.fingerprint()
    import dataclasses as dc
    assert dc.replace(reg, power="").fingerprint() != base
    assert dc.replace(reg, falsification=[]).fingerprint() != base
    assert dc.replace(reg, instruments_required=[]).fingerprint() != base
    assert dc.replace(reg, scheme="v1").fingerprint() != base


def test_editing_the_criteria_afterwards_is_detected(tmp_path):
    open_v2(tmp_path)
    path = tmp_path / REGISTRATION_NAME
    blob = json.loads(path.read_text())
    blob["secondary"] = blob["secondary"].replace("2.0", "1.0")
    path.write_text(json.dumps(blob))
    assert not verify(tmp_path)
    assert fingerprint_scheme(tmp_path) == "mismatch"


def test_the_legacy_fingerprint_is_genuinely_different(tmp_path):
    """P0-4. The legacy branch used to be unreachable.

    `_payload` wrote `tertiary` unconditionally AND again under `if not
    legacy`, so both payloads were byte-identical, `fingerprint(legacy=True)`
    equalled `fingerprint()` and `fingerprint_scheme` could never return
    'legacy'. A tamper check with a branch that cannot execute is one nobody
    has tested.
    """
    reg = open_v2(tmp_path)
    assert reg.fingerprint() != reg.fingerprint(legacy=True)


def test_a_v1_registration_still_verifies_under_the_legacy_scheme(tmp_path):
    """Predating the contract is not the same statement as tampering."""
    v1 = Registration(
        started_on="2026-08-27", config_version="cfg@old",
        engine_version="0.1.0", git_commit="a" * 40,
        target_sessions=375, target_months=18,
        primary="P", secondary="S", invalidation=["x"], scheme="v1")
    path = tmp_path / REGISTRATION_NAME
    import dataclasses as dc
    blob = dc.asdict(v1)
    blob["fingerprint"] = v1.fingerprint(legacy=True)
    path.write_text(json.dumps(blob))
    assert verify(tmp_path), (
        "a registration written under the v1 payload no longer verifies. It "
        "has not been edited; it predates the contract, and saying it was "
        "tampered with would be exactly the untrue message this engine is "
        "being audited for.")
    assert fingerprint_scheme(tmp_path) == "legacy"


# --------------------------------------------------------------- the grading
def test_a_fresh_v2_window_is_not_broken(tmp_path):
    open_v2(tmp_path)
    p = progress(tmp_path, [], live_config_version="cfg@aaa")
    assert p.broken == [], p.broken


def test_a_v2_registration_without_a_falsification_set_is_refused(tmp_path):
    reg = open_v2(tmp_path)
    import dataclasses as dc
    stripped = dc.replace(reg, falsification=[])
    blob = dc.asdict(stripped)
    blob["fingerprint"] = stripped.fingerprint()
    (tmp_path / REGISTRATION_NAME).write_text(json.dumps(blob))
    p = progress(tmp_path, [], live_config_version="cfg@aaa")
    assert any("falsification" in b for b in p.broken)


def test_a_v2_registration_without_a_power_statement_is_refused(tmp_path):
    reg = open_v2(tmp_path)
    import dataclasses as dc
    stripped = dc.replace(reg, power="")
    blob = dc.asdict(stripped)
    blob["fingerprint"] = stripped.fingerprint()
    (tmp_path / REGISTRATION_NAME).write_text(json.dumps(blob))
    p = progress(tmp_path, [], live_config_version="cfg@aaa")
    assert any("power statement" in b for b in p.broken)


def test_a_v2_secondary_that_is_not_benchmark_relative_is_refused(tmp_path):
    reg = open_v2(tmp_path)
    import dataclasses as dc
    bad = dc.replace(reg, secondary="SECONDARY. Pooled rank IC, t >= 2.0.")
    blob = dc.asdict(bad)
    blob["fingerprint"] = bad.fingerprint()
    (tmp_path / REGISTRATION_NAME).write_text(json.dumps(blob))
    p = progress(tmp_path, [], live_config_version="cfg@aaa")
    assert any("benchmark-relative" in b for b in p.broken)


def test_a_config_change_still_breaks_the_window(tmp_path):
    open_v2(tmp_path)
    p = progress(tmp_path, [row(dt.date(2026, 9, 5))],
                 live_config_version="cfg@MOVED")
    assert any("configuration changed after registration" in b for b in p.broken)


# ------------------------------------------------------ the shipped ledger, in CI
@pytest.mark.skipif(not (LEDGER / REGISTRATION_NAME).is_file(),
                    reason="no forward test registered in this checkout")
def test_the_shipped_registration_is_v2_and_intact():
    """The assertion the brief asks CI to carry."""
    assert verify(LEDGER), (
        "data/ledger/forward_test.json does not match its own fingerprint. "
        "The criteria are supposed to be frozen; if they were edited the "
        "window is void and must be restarted, not repaired.")
    assert fingerprint_scheme(LEDGER) == "current"
    reg = load_registration(LEDGER)
    assert reg is not None and reg.scheme == SCHEME
    assert reg.falsification and reg.power
