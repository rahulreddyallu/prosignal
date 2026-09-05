"""Provenance for the research panel, and the guard that reads it.

WHY THIS EXISTS. `panel_2026_09.parquet` was built on 2026-09-03 at 20:47. The
fundamentals ingest landed `results_calendar.parquet` on 2026-09-04 and rewrote
`statements.parquet` on 2026-09-05, taking it from 253 KB to 917 KB. Those files
feed `pit_fundamentals.build_records`, which feeds the v3 `quality` theme -- so
the panel carries a quality theme covering 25% of scored names while the live
engine, reading the same code against the current store, covers about 85%.

Every panel-derived conclusion about that theme was therefore computed on a
different model from the one that ships, and nothing said so. The panel is a
parquet file: it has no idea when it was built or from what, and neither did any
harness reading it.

`prosignal data manifest --verify` already detects the store drift. Nobody ran
it, because nothing connected "the store moved" to "the numbers in this JSON are
about a model that no longer exists". This module is that connection.

THE GUARD REFUSES BY DEFAULT. A harness that warns and carries on produces a
number that will be quoted later without the warning attached.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CURATED = ROOT / "data/curated"

#: Curated tables the panel's factor block actually reads. A change anywhere
#: else in the store does not invalidate it, and pretending otherwise would make
#: the guard fire so often it gets switched off.
PANEL_INPUTS = (
    "prices", "delivery", "statements.parquet", "fundamentals.parquet",
    "results_calendar.parquet", "sector_map.parquet", "equity_master.parquet",
    "corporate_actions.parquet",
)


def _fingerprint() -> dict:
    """Size and mtime of every input the panel depends on.

    Not sha256: the guard runs on every harness start and re-hashing a quarter
    of a gigabyte to answer "has anything moved" is the kind of cost that gets a
    check deleted. Size plus mtime catches every real ingest, and
    `data manifest --verify` is the authoritative check when it matters.
    """
    out = {}
    for name in PANEL_INPUTS:
        p = CURATED / name
        if p.is_dir():
            for f in sorted(p.glob("*.parquet")):
                st = f.stat()
                out[f"{name}/{f.name}"] = [st.st_size, int(st.st_mtime)]
        elif p.exists():
            st = p.stat()
            out[name] = [st.st_size, int(st.st_mtime)]
    return out


def stamp(panel_path: Path, extra: Optional[dict] = None) -> Path:
    """Write the sidecar recording what the panel was built from."""
    panel_path = Path(panel_path)
    side = panel_path.with_suffix(".provenance.json")
    payload = {
        "panel": panel_path.name,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": _fingerprint(),
    }
    try:
        from prosignal.data.manifest import digest_of
        payload["store_manifest_digest"] = digest_of(CURATED)
    except Exception:
        payload["store_manifest_digest"] = "unavailable"
    if extra:
        payload.update(extra)
    side.write_text(json.dumps(payload, indent=2))
    return side


def check(panel_path: Path) -> tuple:
    """``(fresh, reasons)`` -- has the store moved under this panel?"""
    side = Path(panel_path).with_suffix(".provenance.json")
    if not side.exists():
        return False, [f"no provenance sidecar beside {Path(panel_path).name}; "
                       f"the panel cannot say what store it was built from"]
    try:
        rec = json.loads(side.read_text()).get("inputs", {})
    except Exception as exc:
        return False, [f"provenance sidecar unreadable: {exc}"]
    now = _fingerprint()
    reasons = []
    for k, v in sorted(now.items()):
        old = rec.get(k)
        if old is None:
            reasons.append(f"{k}: present now, absent when the panel was built")
        elif list(old) != list(v):
            reasons.append(f"{k}: {old[0]:,} bytes at build time, {v[0]:,} now")
    for k in sorted(rec):
        if k not in now:
            reasons.append(f"{k}: was read when the panel was built, gone now")
    return (not reasons), reasons


def provenance(panel_path: Path) -> dict:
    """The sidecar, trimmed for embedding in a result JSON.

    A result file that does not say which panel produced it is a number with no
    provenance, which is how the stale-panel conclusions survived two audits.
    The per-file fingerprint is dropped -- it is long and the digest identifies
    the build -- but `built_at` and the theme coverages are kept, because a
    coverage collapse is the specific failure this whole module exists to catch.
    """
    side = Path(panel_path).with_suffix(".provenance.json")
    if not side.exists():
        return {"panel": Path(panel_path).name, "provenance": "absent"}
    try:
        rec = json.loads(side.read_text())
    except Exception as exc:
        return {"panel": Path(panel_path).name, "provenance": f"unreadable: {exc}"}
    return {k: v for k, v in rec.items() if k != "inputs"}


def require_fresh(panel_path: Path, allow_stale: bool = False) -> None:
    """Stop the harness unless the panel still describes the current store."""
    fresh, reasons = check(panel_path)
    if fresh:
        return
    head = (f"PANEL IS STALE -- {Path(panel_path).name} was built from a "
            f"different store than the one on disk:")
    body = "\n".join(f"    {r}" for r in reasons[:12])
    if len(reasons) > 12:
        body += f"\n    ... and {len(reasons) - 12} more"
    if allow_stale:
        print(f"{head}\n{body}\n  --allow-stale given; numbers below describe "
              f"the OLD store and must be labelled that way.\n")
        return
    raise SystemExit(
        f"{head}\n{body}\n\n  Rebuild it:  python research/v3/experiments/"
        f"build_panel.py\n  Or pass --allow-stale to report on the old store "
        f"anyway, knowing the numbers may describe a model that no longer runs."
    )
