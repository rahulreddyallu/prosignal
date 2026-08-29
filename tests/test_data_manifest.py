"""D1 -- a result that cannot name its inputs is not reproducible.

`data/` is not in version control, so every panel-derived figure this engine
has ever produced was reproducible only against whatever happened to be in the
store on the day. Two runs of the same command on different days are not
comparable and nothing said so.

The fix is a manifest rather than the data: the store is described precisely
enough to be reconstructed and checked, and the description is what gets
committed. These tests pin the properties that make it worth committing --
that the digest is over content and not over timestamps, that it is never
trusted from the file it was read out of, and that every way a store can
diverge is reported rather than one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from prosignal.data.manifest import (EXCLUDED_NAMES, EXCLUDED_SUFFIXES,
                                     MANIFEST_NAME, build, digest_of, load,
                                     verify, write)


@pytest.fixture
def store(tmp_path):
    (tmp_path / "prices").mkdir()
    (tmp_path / "prices" / "a.csv").write_text("date,close\n2024-01-01,100\n")
    (tmp_path / "prices" / "b.csv").write_text("date,close\n2024-01-01,200\n")
    (tmp_path / "meta.json").write_text('{"universe": "NIFTY_200"}')
    return tmp_path


def _manifest(root: Path):
    m = build(root)
    write(m, root)
    return m


# =============================================================================
# What the digest is over
# =============================================================================


def test_the_digest_is_over_content_and_not_over_timestamps(store):
    """Re-downloading an identical file must not invent a new dataset.

    If mtime entered the digest, every refresh of an unchanged feed would
    produce a new data identity, every research epoch would be forced open,
    and the mechanism would be abandoned within a week for crying wolf.
    """
    before = build(store).compute_digest()
    time.sleep(0.01)
    (store / "prices" / "a.csv").touch()
    assert build(store).compute_digest() == before


def test_any_change_of_content_changes_the_digest(store):
    before = build(store).compute_digest()
    (store / "prices" / "a.csv").write_text("date,close\n2024-01-01,101\n")
    assert build(store).compute_digest() != before


def test_a_new_file_changes_the_digest(store):
    before = build(store).compute_digest()
    (store / "prices" / "c.csv").write_text("date,close\n2024-01-01,300\n")
    assert build(store).compute_digest() != before


def test_renaming_a_file_changes_the_digest(store):
    """Path is part of the content. The same bytes read from a different name
    are a different store, because the code that reads it reads by name."""
    before = build(store).compute_digest()
    (store / "prices" / "a.csv").rename(store / "prices" / "z.csv")
    assert build(store).compute_digest() != before


def test_housekeeping_files_are_not_the_dataset(store):
    """A lock file appearing must not read as the data having changed."""
    before = build(store).compute_digest()
    (store / ".store.lock").write_text("pid 1234")
    (store / "ingest.log").write_text("...")
    (store / "partial.tmp").write_text("...")
    assert build(store).compute_digest() == before
    assert ".store.lock" in EXCLUDED_NAMES
    assert ".log" in EXCLUDED_SUFFIXES


def test_the_manifest_does_not_describe_itself(store):
    """Writing the manifest into the store it describes must not change the
    store's digest, or the digest could never settle."""
    before = build(store).compute_digest()
    write(build(store), store)
    assert build(store).compute_digest() == before
    assert MANIFEST_NAME in EXCLUDED_NAMES


# =============================================================================
# The digest is never trusted
# =============================================================================


def test_a_hand_edited_manifest_cannot_claim_a_digest_it_does_not_have(store):
    """The one attack this has to survive.

    A manifest asserting its own digest is not evidence -- anyone can type a
    hex string. `load` recomputes from the file records, so editing the stored
    digest changes nothing and editing a record changes the digest.
    """
    _manifest(store)
    path = store / MANIFEST_NAME
    blob = json.loads(path.read_text())
    honest = blob["digest"]

    blob["digest"] = "0" * 16
    path.write_text(json.dumps(blob))
    assert load(store).digest == honest, "the claimed digest was believed"

    blob["files"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(blob))
    assert load(store).digest != honest, (
        "a record was altered and the digest did not move"
    )


def test_an_unmanifested_store_says_so_rather_than_building_one(store):
    """`digest_of` must not describe whatever is on disk right now.

    A research run has to record the digest of a manifest somebody committed.
    Building one on the fly would make every run self-certifying: it would
    always match, and would prove nothing.
    """
    assert digest_of(store) == "unmanifested"
    assert not (store / MANIFEST_NAME).exists(), "digest_of wrote a manifest"
    _manifest(store)
    assert digest_of(store) != "unmanifested"


def test_a_corrupt_manifest_is_absent_rather_than_believed(store):
    (store / MANIFEST_NAME).write_text("{not json")
    assert load(store) is None
    assert digest_of(store) == "unmanifested"


# =============================================================================
# Verify reports every way a store can diverge
# =============================================================================


def test_a_changed_file_is_caught_even_at_the_same_length(store):
    """The case `quick` misses, which is why it is not the default."""
    _manifest(store)
    (store / "prices" / "a.csv").write_text("date,close\n2024-01-01,900\n")

    ok, drift = verify(store)
    assert not ok and [d.kind for d in drift] == ["changed"]
    assert "sha256" in drift[0].detail

    quick_ok, _ = verify(store, quick=True)
    assert quick_ok, (
        "this fixture is meant to be same-length so the full check is the one "
        "that catches it; if quick now catches it the test proves nothing"
    )


def test_a_missing_file_and_an_extra_file_are_different_findings(store):
    """They license different follow-up: one is data loss, the other is an
    un-manifested ingest. Reporting both as "drift" loses that."""
    _manifest(store)
    (store / "prices" / "a.csv").unlink()
    (store / "prices" / "new.csv").write_text("date,close\n2024-02-01,50\n")

    ok, drift = verify(store)
    kinds = {d.kind: d.path for d in drift}
    assert not ok
    assert kinds["missing"] == "prices/a.csv"
    assert kinds["untracked"] == "prices/new.csv"


def test_verify_without_a_manifest_fails_rather_than_passing_vacuously(store):
    """The dangerous default. "Nothing to check" must never read as "checked"."""
    ok, drift = verify(store)
    assert not ok
    assert drift and drift[0].kind == "missing"


def test_a_clean_store_verifies(store):
    _manifest(store)
    ok, drift = verify(store)
    assert ok and not drift


# =============================================================================
# The shipped manifest
# =============================================================================


def test_the_shipped_store_is_manifested_and_verifies():
    """The claim the whole audit rests on, checked rather than asserted."""
    from prosignal.config.loader import load_config

    root = Path(load_config().paths.curated)
    m = load(root)
    if m is None:
        pytest.skip("no curated store in this checkout")
    ok, drift = verify(root, quick=True)
    assert ok, [f"{d.path}: {d.detail}" for d in drift[:5]]
    assert m.files, "an empty manifest verifies against an empty store"
    assert any(f.rows for f in m.files), (
        "no file records a row count, so the manifest describes bytes and not "
        "data -- a truncated table would verify"
    )
