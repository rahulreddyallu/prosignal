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


def declared_paths(config_path: Path) -> Dict[str, List[str]]:
    """Every leaf parameter mapped to the FULL PATHS it is declared at.

    The liveness check above matches by NAME, which is its documented blind
    spot: a parameter read in one place and ignored in another reads as live.
    That is not hypothetical -- `stage4_core_score.metalabel.min_win_probability`
    was read by nothing and passed on every startup because
    `stage8_final_signal.scarcity.min_win_probability` is read. Setting it
    looked exactly like arming the NO TRADE veto and did nothing at all.

    This exposes the ambiguity so it can be judged rather than assumed.
    """
    text = Path(config_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    leaves = declared_parameters(config_path)
    stack: List[str] = []
    out: Dict[str, List[str]] = {}
    for raw in lines:
        m = re.match(r"^(\s*)([a-z_][a-z0-9_]*):(.*)$", raw)
        if not m:
            continue
        depth, key = len(m.group(1)) // 2, m.group(2)
        if key in _METADATA:
            continue
        stack = stack[:depth] + [key]
        if key in leaves:
            out.setdefault(key, [])
            path = ".".join(stack)
            if path not in out[key]:
                out[key].append(path)
    return out


def ambiguous_parameters(config_path: Path) -> Dict[str, List[str]]:
    """Leaf names declared at more than one path, minus the allowed ones."""
    return {
        name: paths
        for name, paths in declared_paths(config_path).items()
        if len(paths) > 1 and name not in SHARED_LEAF_NAMES
    }


#: Config section -> the module that owns it. A shared leaf name declared under
#: one of these must be read by that module, or it is inert THERE however live
#: it is elsewhere.
SECTION_OWNER: Dict[str, str] = {
    "stage1_data_quality": "stages/stage1_data_quality.py",
    "stage2_regime": "stages/stage2_regime.py",
    "stage3_eligibility": "stages/stage3_eligibility.py",
    "stage4_core_score": "stages/stage4_core_score.py",
    "stage5_false_signal": "stages/stage5_false_signal.py",
    "stage6_entry": "stages/stage6_entry.py",
    "stage7_risk": "stages/stage7_risk.py",
    "stage8_final_signal": "stages/stage8_final_signal.py",
}


#: Shared leaf names that are inert IN A PARTICULAR SECTION, with the reason.
#: Keyed by full path, because the same name is genuinely live elsewhere -- that
#: is what made these invisible to the name-level check.
#:
#: Found by `unread_in_owning_section` the first time it ran, alongside Stage
#: 5's twelve unread `enabled` flags. Those were wired up; these state
#: behaviours no stage implements, so they are recorded rather than built.
RESERVED_IN_SECTION: Dict[str, str] = {
    "stage4_core_score.factors.momentum_12_1.explicit_weight":
        "Stage 4 weights from the midpoint of `weight_band`; there is no "
        "explicit-weight override path. Setting this changes nothing.",
    "stage4_core_score.factors.quality.explicit_weight":
        "same -- `_weights` reads weight_band only",
    "stage4_core_score.factors.sector_relative_strength.explicit_weight":
        "same -- `_weights` reads weight_band only",
    "stage3_eligibility.pledging.on_missing_data":
        "Stage 3 records an untestable gate unconditionally; the policy is "
        "hardcoded to NOT_TESTABLE and there is no branch on this value.",
    "stage3_eligibility.earnings_proximity.on_missing_data":
        "same -- absence is always recorded NOT_TESTABLE, never configured",
    "stage5_false_signal.data_integrity.source_disagreement_action":
        "Stage 5 has no source-disagreement branch to act on, the same reason "
        "`outlier_action` and `stale_data_action` are in RESERVED.",
}


def unread_in_owning_section(config_path: Path,
                             package_root: Path) -> Dict[str, List[str]]:
    """Shared leaf names whose OWNING module never reads them.

    The blind spot the allowlist opened. `SHARED_LEAF_NAMES` says a repeated
    name is legitimate because the copies mean different things -- but it does
    not check that each copy is actually read. Stage 5 declared twelve
    `enabled` flags and read none of them: every check ran unconditionally, so
    setting one to false changed nothing, and `enabled` passed the liveness
    check because Stage 4, Stage 6 and the providers read the same word.

    A parameter that states a behaviour its own stage does not have is exactly
    what this module exists to catch, and twelve of them hid behind one shared
    name.
    """
    root = Path(package_root)
    out: Dict[str, List[str]] = {}
    for name, paths in declared_paths(config_path).items():
        if name not in SHARED_LEAF_NAMES:
            continue                       # the ambiguity check already covers it
        for path in paths:
            section = path.split(".", 1)[0]
            owner = SECTION_OWNER.get(section)
            if owner is None:
                continue                   # no single owning module
            try:
                source = (root / owner).read_text(encoding="utf-8")
            except OSError:
                continue
            if name not in source:
                out.setdefault(name, []).append(f"{path} (owner {owner})")
    return out


#: Leaf names that legitimately repeat across sections, because they are
#: STRUCTURAL -- every check has an `enabled`, every penalty has a `penalty` --
#: rather than two copies of one setting. Anything not here that appears twice
#: is a name-level ambiguity the liveness check cannot see through, and the
#: test suite fails until it is either removed or declared.
#:
#: The distinction is whether the two declarations mean DIFFERENT THINGS.
#: `stage5_false_signal.gap_signal.penalty` and `...news_spike.penalty` do.
#: Two `min_win_probability` fields did not: they were one threshold written
#: twice, and only one of them was wired.
SHARED_LEAF_NAMES: Set[str] = {
    # structural: one per check, one per trigger, one per component
    "enabled", "penalty", "action", "weight", "weight_band", "explicit_weight",
    "higher_is_better", "lookback_sessions", "ma_sessions", "method",
    "min_volume_multiple", "volume_multiple", "atr_multiple",
    "on_missing_data", "source_disagreement_action", "window_dates",
    # a section name that is also a leaf elsewhere
    "pledging", "thesis_invalidation", "trailing_stop",
}


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
    # -- housekeeping that was specified and never built ----------------------
    "retain_runs": "ledger pruning is not implemented; runs are kept forever",
    "retain_sessions": "store pruning by session count is not implemented",
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
    "momentum_rank_exit_percentile": "Stage 7 performs no rank-decay exit. Rank "
                                     "decay IS an exit, but it lives in Stage 6 "
                                     "as admission.exit_rank; this parameter is "
                                     "a second, unwired copy of that idea",
    "run_technical_collapse_diagnostic": "the diagnostic was never written",
    "require_edge_survives_stress": "the cost model computes a stressed figure "
                                    "and reports it; nothing gates on it",
    "expect_frequent_no_trade": "documentation of intent; NO TRADE is a "
                                "consequence of the gates, not a setting",
}
