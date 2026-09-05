"""README.md may not quote a headline figure that disagrees with the record.

THE DEFECT THIS CLOSES. README.md carried two book tables that cannot both
describe the same engine -- "RESULTS OF RECORD" (mean excess -4.23% per period,
IR -0.83) and "What changed in the tuning pass (2026-08-29)" (+42.6% book
return, +20.3% alpha, Sharpe 1.59), the second appearing TWICE, byte-identical,
102 lines each. Nothing detected that, because prose has no way to disagree
with itself out loud.

So the arbitration moved into code. `prosignal research results` GENERATES
docs/RESULTS_OF_RECORD.md and its JSON twin from the current store; this file
holds README to it.

WHAT IS CHECKED, and why each one.

  1. The generated document exists and its JSON twin parses. Without it there
     is no record to check against and the whole mechanism is decorative.

  2. Every figure inside README's generated-figures block matches the JSON to
     the precision it is quoted at. The block is delimited by HTML comments so
     the comparison is exact rather than a fuzzy scan of prose.

  3. A WITHDRAWN arm's numbers do not appear in README outside a section that
     is explicitly marked WITHDRAWN. This is the one that actually stops the
     original defect recurring: it is not enough to add the right table if the
     wrong one is still sitting six screens further down being quoted.

  4. No section heading is duplicated. The tuning-pass section appeared twice
     and neither copy knew about the other.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RECORD_MD = ROOT / "docs" / "RESULTS_OF_RECORD.md"
RECORD_JSON = ROOT / "docs" / "RESULTS_OF_RECORD.json"

BEGIN = "<!-- RESULTS_OF_RECORD:BEGIN -->"
END = "<!-- RESULTS_OF_RECORD:END -->"

#: Figures README is allowed to quote in its generated block, and where each
#: one lives in the JSON. `fmt` must render exactly as README writes it.
#:
#: Adding a headline number to README means adding it here. That is deliberate
#: friction: an unlisted number is one nothing checks, which is how the two
#: tables drifted apart in the first place.
FIGURES = {
    "shipped_mean_excess": ("arm:results_of_record", "mean_excess_per_period",
                            lambda v: f"{v:+.2%}"),
    "shipped_ir": ("arm:results_of_record", "ir", lambda v: f"{v:+.2f}"),
    "shipped_alpha": ("arm:results_of_record", "alpha_per_period",
                      lambda v: f"{v:+.2%}"),
    "shipped_beat_rate": ("arm:results_of_record", "periods_beating_benchmark",
                          lambda v: f"{v:.1%}"),
    "shipped_gross_excess_ann": ("arm:results_of_record", "gross_excess_ann",
                                 lambda v: f"{v:+.1%}"),
    "shipped_cost_drag_ann": ("arm:results_of_record", "cost_drag_ann",
                              lambda v: f"{v:.1%}"),
    "shipped_net_excess_ann": ("arm:results_of_record", "excess_ann",
                               lambda v: f"{v:+.1%}"),
}


@pytest.fixture(scope="module")
def record():
    if not RECORD_JSON.is_file():
        pytest.fail(
            f"{RECORD_JSON.relative_to(ROOT)} does not exist. Run "
            f"`prosignal research results` to generate it. README's headline "
            f"figures cannot be checked against a record that has not been "
            f"produced, and an unchecked README is how this repository came to "
            f"carry two irreconcilable performance tables."
        )
    return json.loads(RECORD_JSON.read_text(encoding="utf-8"))


def _lookup(record: dict, where: str, key: str):
    if where.startswith("arm:"):
        want = where.split(":", 1)[1]
        for a in record["arms"]:
            if a["key"] == want:
                return a["measured"].get(key)
        raise AssertionError(f"no arm {want!r} in the record")
    return record[where].get(key)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


# ------------------------------------------------------------------ the record
def test_the_generated_record_exists(record):
    assert RECORD_MD.is_file(), (
        f"{RECORD_MD.relative_to(ROOT)} is missing. Run "
        f"`prosignal research results`.")
    body = RECORD_MD.read_text(encoding="utf-8")
    assert "This file is GENERATED" in body, (
        "the record does not announce that it is generated. A reader who "
        "edits it by hand will have their edit silently overwritten, and a "
        "reader who trusts a hand-edit will be trusting prose again.")


def test_the_record_stamps_what_produced_it(record):
    """A number nobody can tie to a dataset is an anecdote, not a result."""
    s = record["stamp"]
    for field in ("config_version", "params_hash", "store_hash", "train_hash",
                  "git_commit", "panel_first_date", "panel_last_date",
                  "panel_rows", "panel_distinct_dates",
                  "independent_observations", "cumulative_trials",
                  "store_fingerprint", "ranking_source"):
        assert s.get(field) not in (None, "", "unbound"), (
            f"the record's stamp is missing {field!r}. Every one of these is "
            f"needed to reconstruct which engine and which data produced the "
            f"table.")


def test_the_record_reports_gross_and_cost_separately(record):
    """Netting them and keeping the last number is a defect already fixed once."""
    for arm in record["arms"]:
        if not arm["measured"]:
            continue
        for key in ("gross_excess_ann", "cost_drag_ann", "excess_ann"):
            assert key in arm["measured"], (
                f"arm {arm['key']!r} reports no {key!r}. Gross, cost and net "
                f"are three numbers and reporting only the third makes 'the "
                f"book loses to its universe' indistinguishable from 'the book "
                f"pays too much to trade'.")


# ------------------------------------------------------------------- the README
def test_readme_carries_a_generated_figures_block():
    text = _readme()
    assert BEGIN in text and END in text, (
        f"README.md has no {BEGIN} ... {END} block. That block is what the "
        f"generated record is compared against; without it nothing checks "
        f"README's headline numbers.")
    assert text.count(BEGIN) == 1 and text.count(END) == 1, (
        "README has more than one generated-figures block. One record, one "
        "block.")


def test_every_headline_figure_matches_the_record(record):
    text = _readme()
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    missing, wrong = [], []
    for name, (where, key, fmt) in FIGURES.items():
        value = _lookup(record, where, key)
        if value is None:
            continue
        expected = fmt(value)
        marker = f"<!--{name}-->"
        if marker not in block:
            missing.append(f"{name} (expected {expected}, marker {marker})")
            continue
        # The figure is the text immediately following its marker.
        seg = block.split(marker, 1)[1]
        actual = seg.strip().split()[0].rstrip("|,.;)")
        if actual != expected:
            wrong.append(f"{name}: README says {actual}, the record says "
                         f"{expected}")
    assert not missing, ("README's generated block is missing figures:\n  "
                         + "\n  ".join(missing))
    assert not wrong, (
        "README disagrees with docs/RESULTS_OF_RECORD.json:\n  "
        + "\n  ".join(wrong)
        + "\n\nThe record is generated from the store; README is not. Fix "
          "README, or regenerate the record if the store has moved.")


def test_a_withdrawn_arms_numbers_do_not_appear_outside_a_withdrawn_section(record):
    """The check that actually stops the original defect recurring.

    Adding the correct table is not enough if the withdrawn one is still six
    screens further down, unlabelled, being quoted.
    """
    text = _readme()
    withdrawn = [a for a in record["arms"] if a["status"] == "WITHDRAWN"]
    if not withdrawn:
        pytest.skip("no arm is currently WITHDRAWN")

    # Split README into sections. A `###` subsection INHERITS its parent `##`
    # section's withdrawal: the tuning-pass section carries the banner on its
    # H2 and has four H3 subsections under it, and demanding a banner on each
    # would be noise rather than discipline.
    sections = re.split(r"^(#{2,6} .+)$", text, flags=re.M)
    offenders = []
    for arm in withdrawn:
        claims = {k: v for k, v in arm["claimed"].items()
                  if isinstance(v, (int, float))}
        needles = set()
        for v in claims.values():
            # Only the distinctive percentage forms. A bare "1.59" or "258"
            # collides with unrelated numbers elsewhere in a 2,000-line README
            # and would make this test fail for reasons that are not drift.
            if 0 < abs(v) < 1:
                needles.add(f"{v:.1%}".lstrip("+"))
    withdrawn_h2 = False
    head = ""
    for i, chunk in enumerate(sections):
        if i % 2 == 1:
            head = chunk
            level = len(head) - len(head.lstrip("#"))
            if level <= 2:                      # a new top-level section
                withdrawn_h2 = ("WITHDRAWN" in head.upper()
                                or "SUPERSEDED" in head.upper())
            continue
        marked = (withdrawn_h2
                  or "WITHDRAWN" in chunk[:800].upper()
                  or "SUPERSEDED" in chunk[:800].upper())
        if marked:
            continue
        for needle in sorted(needles):
            if needle and needle in chunk:
                offenders.append(
                    f"{needle!r} (a WITHDRAWN figure) appears under "
                    f"{head.strip() or '<preamble>'!r}, which is not marked "
                    f"WITHDRAWN or SUPERSEDED")
    assert not offenders, (
        "README quotes figures from a WITHDRAWN arm outside any withdrawal "
        "notice:\n  " + "\n  ".join(sorted(set(offenders)))
        + "\n\nMark the section `WITHDRAWN <date> <config_hash>` with the "
          "reason, or delete the figure. Do not average it with the "
          "reproduced one and do not quote the more favourable one.")


def test_no_section_heading_is_duplicated():
    """The tuning-pass section appeared twice, byte-identical, 102 lines each."""
    heads = re.findall(r"^#{2,3} (.+)$", _readme(), flags=re.M)
    dupes = {h: n for h, n in Counter(heads).items() if n > 1}
    assert not dupes, (
        f"README repeats section headings: {dupes}. A duplicated section is "
        f"two documents pretending to be one, and neither copy knows about "
        f"the other.")
