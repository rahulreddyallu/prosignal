"""The V11 audit's findings, each pinned by the thing that would have caught it.

Every defect here shipped through a green suite, which is the finding behind the
findings: the tests asserted what the code returned and never what an operator
would read, or which store a process would write to.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "prosignal"
INDEX = SRC / "static" / "index.html"


# =============================================================================
# D-101 -- a sandbox switch that did not sandbox
# =============================================================================

def test_the_entrypoints_read_the_config_environment_variable():
    """`PROSIGNAL_CONFIG` was honoured by ONE call site out of three.

    `api.create_app` and the CLI both called `load_config()`, which never looks
    at the environment; `outcomes.py` called `get_config()`, which does. So
    setting the variable did not have no effect -- it split the process in half,
    and outcomes resolved against the store you asked for while the pipeline
    read and WROTE the one you did not. That is how this audit put a `live` row
    in the production ledger from a session it believed was sandboxed.
    """
    api = (SRC / "api.py").read_text(encoding="utf-8")
    cli = (SRC / "cli.py").read_text(encoding="utf-8")

    assert re.search(r"cfg = config or get_config\(\)", api), (
        "create_app must build its config through get_config, or the served "
        "app ignores $PROSIGNAL_CONFIG and writes to the production store"
    )
    assert re.search(r"cfg = get_config\(config_path=args\.config\)", cli), (
        "the CLI must resolve through get_config so --config and the "
        "environment agree"
    )


def test_an_explicit_none_path_still_lets_the_environment_win(tmp_path, monkeypatch):
    """The subtle half. `get_config` tested `"config_path" not in kwargs`, and
    the CLI forwards its optional flag straight through -- so on every run
    WITHOUT `--config` the key was present-but-None and the environment was
    skipped exactly when it was the only thing left to consult."""
    from prosignal.config import loader

    cfg_file = tmp_path / "params.yaml"
    src = Path(loader.__file__).resolve().parents[2].parent / "config" / "parameters.yaml"
    if not src.exists():                                   # pragma: no cover
        pytest.skip("shipped config not found")
    store = tmp_path / "store"
    text = src.read_text(encoding="utf-8")
    text = re.sub(r'(\n    data_dir: )"[^"]*"', rf'\g<1>"{store}"', text, count=1)
    cfg_file.write_text(text, encoding="utf-8")

    monkeypatch.setenv("PROSIGNAL_CONFIG", str(cfg_file))
    loader.reset_config_cache()

    assert loader.get_config().paths.data == store
    assert loader.get_config(config_path=None).paths.data == store, (
        "config_path=None is the CLI's default and must mean 'not specified', "
        "not 'specified as nothing'"
    )
    loader.reset_config_cache()


# =============================================================================
# D-103 -- the screen certified a model that had been deleted
# =============================================================================

def test_an_untiered_run_is_not_certified_as_the_deleted_fitted_model():
    """`_scorer_used` was keyed on `evidence_tier`, which fixed the run in front
    of you and left the old branch reachable for every payload written before
    tiers existed. `MODEL_KEYS - COMPOSITE_KEYS` is {beta, delivery, drawdown,
    lottery, mom, reversal, skew}, so ANY stored pre-v3 run matched it and came
    back `cross-sectional / validated: True / note: None`.

    Rendered, that is: today's date, five BUY badges, "Lottery-like payoff shape
    leads at -1.94 sd" as the reason, and NO caveat -- the interface shows its
    caveat only when `validated` is false.
    """
    from prosignal.presentation.viewmodel import _scorer_used

    legacy = [{"factors": {k: {"tier": None} for k in
               ("lottery", "reversal", "delivery", "mom", "risk", "beta",
                "skew", "drawdown")}}]
    out = _scorer_used(legacy)

    # NAMING it is fine and useful -- "the model we retired" tells a reader
    # more than "unknown", and `test_coverage` pins that direction. CERTIFYING
    # it is the defect.
    assert out["validated"] is False, (
        "validated=True suppresses the caveat in index.html, which is what made "
        "this invisible for the life of the product"
    )
    assert out.get("severity") == "alarm", (
        "a run from a retired model is a failure, not a standing disclosure"
    )
    assert out.get("note"), "and there has to be something to render"
    assert "retired" in out["note"].lower()


def test_the_source_carries_no_branch_that_returns_validated_true():
    """The narrower guard: no scorer path may hand the interface a run it calls
    validated, because nothing in this engine is."""
    vm = (SRC / "presentation" / "viewmodel.py").read_text(encoding="utf-8")
    body = vm[vm.index("def _scorer_used"):vm.index("def build_view")]
    live = [ln for ln in body.splitlines()
            if not ln.lstrip().startswith("#")]
    assert not [ln for ln in live if '"validated": True' in ln], (
        "a scorer branch still returns validated=True"
    )


# =============================================================================
# D-113 -- the screen served a run the record does not name
# =============================================================================

def test_the_ledger_can_name_the_newest_run_for_a_date(tmp_path):
    """`rundetail.save` never raises -- correctly, the run IS in the ledger --
    but nothing compared the two. Forced by making the run-detail directory
    unwritable: the ledger took the new row and the screen went on serving the
    PREVIOUS run for the same date. Same `as_of`, so the staleness check passed
    and nothing looked wrong."""
    import datetime as dt
    import json

    from prosignal.ledger import Ledger

    (tmp_path / "runs-2026.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"date": "2026-09-03", "run_id": "aaaaaaaaaaaa", "mode": "live",
         "logged_at": "2026-09-05T07:39:14"},
        {"date": "2026-09-03", "run_id": "bbbbbbbbbbbb", "mode": "live",
         "logged_at": "2026-09-05T07:44:39"},
        {"date": "2026-09-02", "run_id": "cccccccccccc", "mode": "live",
         "logged_at": "2026-09-03T07:54:06"},
    ]) + "\n", encoding="utf-8")

    led = Ledger(tmp_path)
    newest = led.newest_on(dt.date(2026, 9, 3), mode="live")
    assert newest and newest["run_id"] == "bbbbbbbbbbbb", (
        "newest_on must order by logged_at, not by position in the file"
    )
    assert led.newest_on(dt.date(2026, 8, 1), mode="live") is None


def test_today_reports_when_it_is_not_showing_the_recorded_run():
    """The comparison itself, without standing an API up."""
    from prosignal import api

    class _Cfg:
        class paths:
            ledger = "/nonexistent"

    # A ledger that cannot be read must not break the page -- the note is a
    # disclosure, not a dependency.
    assert api._served_run_is_the_recorded_run(
        _Cfg, {"as_of_date": "2026-09-03", "run_id": "aaaaaaaaaaaa"}) == ""


# =============================================================================
# The screen: pointers, not paragraphs
# =============================================================================

def test_the_standing_disclosure_is_not_a_paragraph_on_the_shortlist():
    """Ninety words above the shortlist, on every run of every day. It is a
    property of the shipped configuration and not of today's cross-section, so
    it belongs in Settings beside the configuration -- and as lines, not prose.
    """
    from prosignal.presentation.viewmodel import _scorer_used

    out = _scorer_used([{"factors": {"momentum": {"tier": "v3_theme"}}}])
    assert out["points"], "the disclosure must be pointers"
    for line in out["points"]:
        assert len(line) <= 90, f"still a sentence: {line!r}"
    assert len(out["note"]) <= 140

    # every number that was in the paragraph survives
    joined = " ".join(out["points"])
    for fact in ("22 factors", "5 themes", "40%", "+0.049", "+0.036",
                 "+0.38%", "0.81"):
        assert fact in joined, f"{fact} was dropped, not shortened"


def test_the_card_counts_the_factors_it_actually_measured():
    """Every card printed "22 factors behind them". On HFCL four of the
    twenty-two had no value for that name, the theme carried its full weight
    anyway, and nothing said so."""
    html = INDEX.read_text(encoding="utf-8")
    assert "m.rank != null" in html and "nSeen" in html, (
        "the factor count must exclude members with no value for this name"
    )
    assert "' of ' + nFac + ' factors measured'" in html


def test_the_card_prints_the_total_it_claims_to_sum_to():
    """The heading read "sums to the score" and the score was on no surface of
    the card -- the one claim a reader might have checked."""
    html = INDEX.read_text(encoding="utf-8")
    live = [ln for ln in html.splitlines() if "sums to the score" in ln
            and not ln.lstrip().startswith(("/*", "*", "//"))]
    assert not live, "the unverifiable claim is still rendered"
    assert 'ph-s">sums to ' in html, "the heading must print the total"


@pytest.mark.parametrize("gone", [
    "No name qualified, and none came close enough to be worth monitoring",
    "That is a result, not a failure to find one",
    "Every call lands here the day it is issued",
    "until then it is above, marked and uncounted",
    "Pct is its percentile in its sector",
    "The scan runs daily; entries open every",
    "The names the engine ranks highest, and the case for each",
])
def test_the_prose_is_gone(gone):
    """Each of these rendered on a screen an operator reads daily."""
    assert gone not in INDEX.read_text(encoding="utf-8"), f"still rendered: {gone!r}"


# =============================================================================
# The daily loop
# =============================================================================

def test_the_nightly_remanifests_after_it_ingests():
    """The ingest changes the store, so the manifest stops describing it, so the
    restart gate fails -- every morning, for a reason that is not the reason the
    gate exists. `data manifest --write` was reachable only by hand."""
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "forward_run.sh").read_text()
    i_ingest = sh.index("data ingest")
    i_manifest = sh.index("data manifest --write")
    i_analyse = sh.index("analyse run")
    assert i_ingest < i_manifest < i_analyse, (
        "the manifest must be rewritten between the ingest and the analysis"
    )


def test_the_nightly_does_not_alarm_when_there_is_simply_no_api():
    """On a host that runs the job but not the service every warm curl fails and
    the screen check alarms on a night when nothing is wrong. An alarm that
    fires on every healthy run trains the reader to ignore the real one."""
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "forward_run.sh").read_text()
    assert "/health" in sh and "skipping cache warm" in sh


def test_provisioning_configures_somewhere_for_a_failure_to_go():
    """`forward_run.sh` alerts on failed ingest, failed analysis and an invalid
    forward test -- through a variable `cloud-init.sh` never set. Every failure
    it is careful to detect went to a log file on a box nobody logs into."""
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "cloud-init.sh").read_text()
    assert "PROSIGNAL_ALERT_CMD=${ALERT_CMD}" in sh, (
        "the alert command must be written into /etc/prosignal.env"
    )
    assert "NEVER STARTED" in sh, (
        "the file must say that a failure alarm cannot report a job that never "
        "ran -- that is the failure this deployment actually had"
    )
