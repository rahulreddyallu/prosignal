"""Which declared parameters are actually consumed.

A parameter that exists, validates on every startup and is read by nothing is
worse than a missing one: it states a behaviour the engine does not have.
Three shipped that way -- ma_reclaim.reference promised an anchored VWAP the
code never computed, require_above_average_volume promised a volume bar the
function was not passed the data for, and reject_if_overextended promised a
distance check no stage performed.

This finds them mechanically so the next one cannot ship quietly. A parameter
counts as consumed when its name is read anywhere in the package other than
the schema line that declares it -- including inside the config package, since
`capital.max_open_positions` is read there to derive `position_value_inr` and
is live by that route.

Parameters with no consumer must carry ``status: RESERVED``, which is a claim
about intent rather than behaviour and reads as one on the card.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set

__all__ = ["declared_parameters", "consumed_names", "inert_parameters",
           "RESERVED"]

#: Keys that describe a parameter rather than being one.
_METADATA = frozenset({
    "value", "status", "note", "search_range", "search_points",
    "optimization_tier", "source", "unit",
})


def declared_parameters(config_path: Path) -> Dict[str, int]:
    """Every leaf PARAMETER in parameters.yaml, mapped to its line number.

    A key that only groups other keys is a section, not a parameter: ``costs``
    and ``validation`` are not settings and nothing reads them by name. A key
    counts when it either carries a scalar on the same line or opens a block
    whose first meaningful child is ``value:``.
    """
    text = Path(config_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    out: Dict[str, int] = {}
    for i, raw in enumerate(lines):
        m = re.match(r"^(\s*)([a-z_][a-z0-9_]*):(.*)$", raw)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3).strip()
        if key in _METADATA:
            continue
        if rest and not rest.startswith("#"):
            out.setdefault(key, i + 1)              # scalar on the same line
            continue
        for nxt in lines[i + 1:]:                   # a block: is it a Tunable?
            if not nxt.strip() or nxt.strip().startswith("#"):
                continue
            child = re.match(r"^(\s*)([a-z_][a-z0-9_]*):", nxt)
            if not child or len(child.group(1)) <= indent:
                break
            if child.group(2) == "value":
                out.setdefault(key, i + 1)
            break
    return out


def _names_in(tree: ast.AST) -> Set[str]:
    """Every identifier the module READS.

    A pydantic field line (``foo: TF``) declares; it does not consume. Only that
    exact Name node is skipped -- the same identifier read elsewhere in the file
    still counts, which is how ``capital.max_open_positions`` is recognised as
    live: it is declared as a field and then read by ``position_value_inr``.
    """
    declarations = {
        id(node.target)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name) and id(node) not in declarations:
            found.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # getattr(cfg, "foo") and cfg["foo"] both hide the name in a string
            if node.value.isidentifier():
                found.add(node.value)
    return found


def consumed_names(package_root: Path) -> Set[str]:
    """Identifiers read anywhere in the package, excluding schema declarations."""
    consumed: Set[str] = set()
    for path in sorted(Path(package_root).rglob("*.py")):
        if path.name == "liveness.py":
            continue
        consumed |= _names_in(ast.parse(path.read_text(encoding="utf-8")))
    return consumed


def inert_parameters(config_path: Path, package_root: Path) -> List[str]:
    """Declared in parameters.yaml, read by nothing. Sorted for stable output."""
    declared = declared_parameters(config_path)
    consumed = consumed_names(package_root)
    return sorted(k for k in declared if k not in consumed)


#: A limitation worth stating: this check is by NAME, so a parameter read in one
#: place and ignored in another reads as live. stage5_false_signal.news_spike.
#: volume_multiple was exactly that -- live in Stage 6's confirmation block and
#: silently unread inside _news_spike, which made that check fire 2.7x its own
#: specification. Name-level liveness catches parameters nothing reads; it does
#: not catch a parameter the wrong module reads.
#:
#: Parameters declared in parameters.yaml that nothing reads, each with the
#: reason it is allowed to stay. The test suite asserts this set matches what
#: :func:`inert_parameters` finds, in both directions:
#:
#:   * a new inert parameter fails until it is wired up or listed here
#:   * wiring one up fails until its entry is removed
#:
#: Neither direction can be satisfied by editing the config alone, which is the
#: point. Three parameters shipped stating behaviour the engine did not have --
#: ma_reclaim.reference promised an anchored VWAP nothing computed,
#: require_above_average_volume promised a volume bar the function had no data
#: for, reject_if_overextended promised a distance check no stage ran.
RESERVED: Dict[str, str] = {
    # -- deployment: read from the environment, not from this file ------------
    "port": "the process binds $PORT (see render.yaml startCommand); this value "
            "is never read and setting it does not move the port",
    "cors_origins": "no CORS middleware is installed; setting this grants nothing",
    "owner": "provenance only; appears in no output",

    # -- research protocol: describes how research is conducted, not what the
    # -- engine does. There is no harness in the package that consumes them.
    "sacred": "states that the holdout is used once. Enforced by discipline, "
              "not by the loader -- nothing can technically prevent a second look",

    # -- housekeeping that was specified and never built ----------------------
    "retain_runs": "ledger pruning is not implemented; runs are kept forever",
    "retain_sessions": "store pruning by session count is not implemented",
    "run_detail_subdir": "per-run detail files are not written",
    "write_run_detail": "per-run detail files are not written",
    "fail_run_if_unwritable": "a ledger write failure is already fatal "
                              "unconditionally; this cannot soften it",

    # -- reference CSV paths the ingest does not consult ----------------------
    "corporate_actions_path": "corporate actions come from NSE and yfinance; "
                              "the CSV override is not read",
    "event_calendar_path": "the earnings calendar is assembled from feeds; "
                           "this override is not read",
    "shareholding_path": "promoter pledging is not ingested at all, which the "
                         "run already reports as NOT_TESTABLE",

    # -- behaviour specified for stages that do not implement it --------------
    "data_quality_gate_penalty": "Stage 4 applies no data-quality multiplier to "
                                 "the composite",
    "max_forward_fill_sessions": "nothing forward-fills; every pct_change passes "
                                 "fill_method=None and the store writes only "
                                 "sessions that published",
    "outlier_action": "Stage 5 has no outlier branch to act on",
    "stale_data_action": "Stage 5 re-affirms nothing; staleness is decided in "
                         "Stage 1 against the manifest",
    "execution_realism_hard_reject_participation": "Stage 5 does not compare "
                                                   "implied fill size to ADTV; "
                                                   "Stage 7 caps participation "
                                                   "at entry instead",
    "execution_realism_penalty_per_pct_over": "same; no participation penalty exists",
    "max_candidates_per_sector_soft": "Stage 3 applies no soft sector cap; the "
                                      "hard cap lives in Stage 8",
    "momentum_rank_exit_percentile": "exits are the stop, the invalidation level "
                                     "and the horizon; rank decay is not one",
    "run_technical_collapse_diagnostic": "the diagnostic was never written",
    "require_edge_survives_stress": "the cost model computes a stressed figure "
                                    "and reports it; nothing gates on it",
    "expect_frequent_no_trade": "documentation of intent; NO TRADE is a "
                                "consequence of the gates, not a setting",
}
