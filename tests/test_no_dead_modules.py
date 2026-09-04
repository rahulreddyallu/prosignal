"""No module may sit in `src/` with nothing importing it.

WHY THIS EXISTS. Before the 2026-09-03 cleanup the package carried eight modules
that nothing imported -- backtest, portfolio, monitor, sla, config.liveness and
three under validation/ -- for about 1,700 lines. Every one of them had passing
tests, which is exactly why nobody noticed: a test suite proves code runs, not
that anything calls it.

Resolution is AST-based and full-path. Two cheaper approaches were tried first and
both produced wrong answers that would have deleted live code:

  * matching on the leaf name conflated `presentation.selection`, which
    `pipeline.py` imports, with `validation.selection`, which nothing did;
  * resolving relative imports without special-casing `__init__.py` reported the
    live `presentation.viewmodel` as unreachable, because `from .viewmodel import
    build_view` inside `presentation/__init__.py` resolves against the PACKAGE and
    not against a phantom `presentation.__init__`.

Entry points are exempt: `cli`, `api` and `jobs` are invoked, not imported.
"""
from __future__ import annotations

import ast
import collections
from pathlib import Path

import pytest

# Invoked from the command line or by the web server; nothing imports them.
ENTRY_POINTS = {"prosignal.cli", "prosignal.api", "prosignal.jobs"}


def _src_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "src" / "prosignal"
        if cand.is_dir():
            return cand
    pytest.skip("source tree not found")


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    return ".".join(rel.parts).replace(".__init__", "")


def _import_graph(root: Path):
    modules = {_module_name(f, root): f for f in root.rglob("*.py")}
    importers = collections.defaultdict(set)
    for name, path in modules.items():
        # A relative import inside `__init__.py` is relative to the PACKAGE.
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                       # a broken file is another test's job
            continue
        for node in ast.walk(tree):
            targets = set()
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package
                    for _ in range(node.level - 1):
                        base = base.rsplit(".", 1)[0]
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                if target.startswith("prosignal"):
                    targets.add(target)
                    # `from pkg import submodule` binds the SUBMODULE
                    for alias in node.names:
                        targets.add(f"{target}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("prosignal"):
                        targets.add(alias.name)
            for target in targets:
                if target in modules and target != name:
                    importers[target].add(name)
    return modules, importers


def test_every_module_is_imported_by_something():
    root = _src_root()
    modules, importers = _import_graph(root)
    assert len(modules) > 50, "the import graph found almost nothing; the walk is wrong"

    orphans = sorted(
        name for name, path in modules.items()
        if path.name != "__init__.py"           # packages execute on import
        and name not in ENTRY_POINTS
        and not importers[name]
    )
    lines = {
        name: sum(1 for _ in modules[name].open(encoding="utf-8"))
        for name in orphans
    }
    assert not orphans, (
        "these modules are in src/ and nothing imports them "
        f"({sum(lines.values())} lines):\n  "
        + "\n  ".join(f"{n}  ({lines[n]} lines)" for n in orphans)
        + "\n\nDelete them, or wire them up. A module with tests but no caller is "
          "still dead: the tests prove it runs, not that anything runs it."
    )


def test_the_check_can_actually_fail():
    """A guard that cannot fire is not a guard.

    Builds the same graph over a temporary package containing one unreferenced
    module, and asserts the detector finds it. Without this the test above would
    keep passing if the resolution logic silently broke.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "prosignal"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("from .used import thing\n")
        (pkg / "used.py").write_text("thing = 1\n")
        (pkg / "stranded.py").write_text("nobody = True\n")
        modules, importers = _import_graph(pkg)
        orphans = {n for n, p in modules.items()
                   if p.name != "__init__.py" and not importers[n]}
        assert "prosignal.stranded" in orphans
        assert "prosignal.used" not in orphans
