"""An analysis must not read a store being rewritten underneath it.

Individual files were already safe -- every write lands via os.replace, and a
reader holding a descriptor keeps the old inode. Corruption was never the
exposure.

The exposure is ACROSS files. The store is roughly twenty-two of them and an
ingest rewrites them one at a time over several minutes, so an analysis
starting halfway sees prices through Friday and delivery through Tuesday. Each
file is internally valid; the set describes no day that ever existed, and every
feature spanning the boundary is computed across two.

The API could not hit this because bootstrap and analysis share a job slot. Two
terminals could: `prosignal data ingest` and `prosignal analyse run` had no
relationship at all.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from prosignal.data.storelock import LOCK_NAME, StoreBusy, store_lock


def test_two_readers_do_not_block_each_other(tmp_path):
    """Analyses are shared. Serialising them would be a pointless cost."""
    with store_lock(tmp_path, exclusive=False, what="analysis A"):
        with store_lock(tmp_path, exclusive=False, what="analysis B"):
            pass


def _hold_exclusive(root, seconds, ready):
    with store_lock(Path(root), exclusive=True, what="ingest test"):
        ready.set()
        time.sleep(seconds)


def test_a_reader_is_refused_while_an_ingest_holds_the_store(tmp_path):
    """The case that mattered: analysis during ingest."""
    ready = mp.Event()
    proc = mp.Process(target=_hold_exclusive, args=(str(tmp_path), 3, ready))
    proc.start()
    try:
        assert ready.wait(timeout=10), "writer never acquired the lock"
        with pytest.raises(StoreBusy, match="rewritten underneath"):
            with store_lock(tmp_path, exclusive=False, what="analysis"):
                pass
    finally:
        proc.join(timeout=10)


def test_the_refusal_names_what_holds_the_lock(tmp_path):
    ready = mp.Event()
    proc = mp.Process(target=_hold_exclusive, args=(str(tmp_path), 3, ready))
    proc.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(StoreBusy) as err:
            with store_lock(tmp_path, exclusive=False, what="analysis"):
                pass
        assert "ingest test" in str(err.value), (
            "a refusal that does not say what holds the lock leaves the "
            "operator guessing whether to wait or to investigate"
        )
    finally:
        proc.join(timeout=10)


def test_the_lock_is_released_when_the_block_exits(tmp_path):
    with store_lock(tmp_path, exclusive=True, what="first"):
        pass
    with store_lock(tmp_path, exclusive=True, what="second"):
        pass


def test_the_lock_is_released_even_when_the_body_raises(tmp_path):
    """An ingest that dies mid-write must not wedge the store forever."""
    with pytest.raises(ValueError):
        with store_lock(tmp_path, exclusive=True, what="doomed"):
            raise ValueError("boom")
    with store_lock(tmp_path, exclusive=True, what="after"):
        pass


def test_the_lock_file_lives_beside_the_store(tmp_path):
    with store_lock(tmp_path, exclusive=True, what="ingest"):
        pass
    assert (tmp_path / LOCK_NAME).is_file()


def test_the_pipeline_takes_a_shared_lock_and_the_ingest_an_exclusive_one():
    """Structural: the direction matters. A writer taking a shared lock, or a
    reader taking an exclusive one, would both compile and neither would
    protect anything."""
    root = Path(__file__).resolve().parents[1] / "src" / "prosignal"
    pipeline = (root / "pipeline.py").read_text(encoding="utf-8")
    ingest = (root / "data" / "ingest.py").read_text(encoding="utf-8")
    assert "store_lock(config.paths.curated, exclusive=False" in pipeline
    assert "exclusive=True" in ingest and "blocking=True" in ingest
