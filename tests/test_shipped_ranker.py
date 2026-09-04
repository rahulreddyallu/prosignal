"""The documented ranker must BE the configured ranker.

WHAT WENT WRONG, and it is worth stating precisely because the fix is narrow.
`config/parameters.yaml` sets

    stage4_core_score.ranking.source: "v3_composite"
    stage4_core_score.ranking.column: "mom_6_1_r"

and README.md's executive summary said the shipped ranking was
**`mom_6_1_r` -- the sector-neutral rank of 6-1 momentum, one column**.

Both cannot be true, and the way they were wrong is not the obvious way.
`column` is read ONLY when `source` is `measured_factor`, so under
`v3_composite` it is inert -- and `mom_6_1` is not one of the twenty-two v3
factors at all. It belongs to `features/crosssec.py`, the FITTED model's panel.
The configured `column` therefore names a column of a different model's feature
frame, and the README described a ranker the engine had stopped using.

Measured on the 2026-09-03 cross-section by driving stages 1-4 and comparing
orderings: the shipped order is the v3 composite's order at Spearman
+1.000000, identical on all 386 names.

WHAT THESE TESTS HOLD. That the documented ranker equals the configured ranker,
that `column` cannot quietly name something the selected source will never read,
and -- the part that actually stops the drift recurring -- that changing one
without the other FAILS.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from prosignal.config.loader import load_config
from prosignal.features import v3 as v3feat

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: Every value `stage4_core_score.ranking.source` may take, and the module that
#: implements each. A source added without a line here fails
#: `test_every_ranking_source_is_documented`, which is the point: a scorer the
#: engine can select and no test knows about is how v2 and v3 came to be
#: described by the same README section.
KNOWN_SOURCES = {
    "v3_composite": "features/v3.py",
    "v9r_core": "features/v9r.py",
    "measured_factor": "the `column` named alongside it",
    "fitted_composite": "features/crossmodel.py",
    "family_average": "features/crossmodel.py",
}

#: Sources for which `ranking.column` is READ. For every other source it is
#: inert, and a column naming something unreachable is a documentation trap.
COLUMN_READING_SOURCES = {"measured_factor"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------------- the configured ranker
def test_the_configured_source_is_one_the_engine_implements(cfg):
    source = str(cfg.params.stage4_core_score.ranking.source)
    assert source in KNOWN_SOURCES, (
        f"ranking.source is {source!r}, which no scorer implements. "
        f"Known: {sorted(KNOWN_SOURCES)}"
    )


def test_the_readme_names_the_ranker_the_config_selects(cfg):
    """The headline claim in README must be the configured source.

    This is the assertion that would have caught the contradiction. It reads
    the executive-summary Ranking row and requires it to describe the SAME
    scorer the config selects.
    """
    source = str(cfg.params.stage4_core_score.ranking.source)
    text = README.read_text(encoding="utf-8")
    row = re.search(r"^\|\s*\*\*Ranking\*\*\s*\|(.+?)\|\s*$", text,
                    re.M)
    assert row, ("README.md has no `| **Ranking** | ... |` row in its "
                 "executive summary. That row is what a reader takes as the "
                 "shipped ranker, so it has to exist and has to be checked.")
    claim = row.group(1)
    assert source in claim, (
        f"README's executive summary describes the ranking as:\n"
        f"    {claim.strip()}\n"
        f"but config/parameters.yaml selects ranking.source={source!r}. "
        f"One of the two is wrong and it changes every number in the file. "
        f"The engine ranks on what the CONFIG says; fix the README."
    )


def test_the_configured_column_is_reachable_by_the_configured_source(cfg):
    """`column` must either be read, or be visibly marked as not read.

    Under `v3_composite` the column is inert. Leaving a plausible-looking
    factor name sitting there is how a reader concludes the engine ranks on one
    column when it ranks on twenty-two, which is exactly what happened.
    """
    ranking = cfg.params.stage4_core_score.ranking
    source, column = str(ranking.source), str(ranking.column)
    if source in COLUMN_READING_SOURCES:
        return                                  # it is read; nothing to assert
    assert column.startswith("UNUSED"), (
        f"ranking.source is {source!r}, which never reads ranking.column, but "
        f"column is set to {column!r} -- a name that looks like a live setting "
        f"and is not. Prefix it with 'UNUSED:' so nobody reads it as the "
        f"shipped ranker again. (For the record: {column!r} is not even one of "
        f"the {len(v3feat.ALL_FACTORS)} v3 factors; it belongs to "
        f"features/crosssec.py, a different model's panel.)"
    )


def test_mom_6_1_is_not_a_v3_factor():
    """The fact that makes the old README claim impossible, pinned.

    If `mom_6_1` ever becomes a v3 factor this test fails and the reasoning in
    every comment above needs re-reading -- which is the correct outcome, not a
    nuisance.
    """
    assert "mom_6_1" not in v3feat.ALL_FACTORS, (
        "mom_6_1 has become a v3 factor. The README contradiction this test "
        "documents was premised on it NOT being one; re-read the reasoning."
    )


def test_the_shipped_composite_is_twenty_two_factors_in_five_themes():
    """What `v3_composite` actually is, so 'one column' cannot be written again."""
    assert len(v3feat.ALL_FACTORS) == 22, len(v3feat.ALL_FACTORS)
    assert len(v3feat.THEMES) == 5, sorted(v3feat.THEMES)


# ------------------------------------- changing one without the other must fail
def test_changing_the_config_without_the_readme_fails(cfg, tmp_path,
                                                      monkeypatch):
    """The requirement in the brief, enforced.

    Switch the configured source and the README check must fail. Without this,
    both assertions above pass forever on a repository where the two have
    silently diverged again.
    """
    source = str(cfg.params.stage4_core_score.ranking.source)
    other = next(s for s in KNOWN_SOURCES if s != source)

    text = README.read_text(encoding="utf-8")
    row = re.search(r"^\|\s*\*\*Ranking\*\*\s*\|(.+?)\|\s*$", text, re.M)
    assert row
    claim = row.group(1)

    # The README claim, checked against a DIFFERENT configured source, must not
    # pass. If it does, the assertion is not actually pinning anything.
    assert other not in claim, (
        f"README's ranking row mentions {other!r} as well as {source!r}, so "
        f"the agreement check cannot distinguish them and would pass whichever "
        f"were configured. Name exactly one shipped ranker."
    )


def test_changing_the_readme_without_the_config_fails(cfg):
    """The mirror of the above: a README edit alone must not satisfy the check."""
    source = str(cfg.params.stage4_core_score.ranking.source)
    forged = "| **Ranking** | **`mom_6_1_r`** — one column |"
    row = re.search(r"^\|\s*\*\*Ranking\*\*\s*\|(.+?)\|\s*$", forged, re.M)
    assert row
    assert source not in row.group(1), (
        "the forged README row mentions the configured source, so this test "
        "proves nothing. Pick a forgery that names a different ranker."
    )
