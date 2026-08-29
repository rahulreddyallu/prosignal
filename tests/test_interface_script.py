"""The interface's own script, checked the way the browser checks it.

WHY THIS EXISTS. The screen is one 3,000-line HTML file with the whole
application inside a single <script>. Nothing in the suite parsed it, so any
edit to it was verified by opening a browser and looking -- and a reference
that no longer resolves does not look like anything until the moment a person
presses the control that needs it.

That is not hypothetical. Rewriting the settings drawer removed `openSettings`
along with the block it sat in. Every test passed, the page rendered, the
shortlist was correct, and the gear icon did nothing at all: the handler threw
`ReferenceError: openSettings is not defined` into the console where no test
was listening. The same edit left `upside` called from the analysis panel after
its definition had gone, which broke every "View analysis" press.

Both are the same defect -- a call with no declaration behind it -- and both
are caught by parsing the script and resolving its identifiers, which is what
these tests do. Node is used as the parser because it is the same language
engine the browser runs; the tests skip rather than fail where it is absent, so
this cannot become a reason a checkout will not test.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
INDEX = STATIC / "index.html"

node = pytest.mark.skipif(shutil.which("node") is None,
                          reason="node is not installed; the browser-parity "
                                 "checks need a JavaScript engine")


def _script() -> str:
    """The contents of the single inline <script>."""
    html = INDEX.read_text()
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1, f"expected one inline script, found {len(blocks)}"
    return blocks[0]


def _run_node(source: str, argument: str) -> subprocess.CompletedProcess:
    return subprocess.run(["node", "-e", source, argument],
                          capture_output=True, text=True, timeout=60)


@node
def test_the_interface_script_parses():
    """A syntax error ships as a blank page: the browser abandons the whole
    script, so every handler in the file dies together."""
    proc = subprocess.run(["node", "--check", "-"], input=_script(),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        "the interface script does not parse:\n" + proc.stderr)


@node
def test_every_function_called_is_declared():
    """No call without a declaration behind it.

    The check runs in Node rather than over a regular expression because the
    file is full of strings that contain code-shaped text -- the exit-rung
    labels, the CSS in the template literals -- and a regular expression cannot
    tell those from real calls. Node tokenises it properly by compiling the
    script inside a Function, then the declared names are compared against the
    names the source calls at statement level.
    """
    script = _script()
    checker = r"""
const src = process.argv[1];
// Compile only -- never execute. `new Function` throws on a syntax error and
// otherwise gives us a body we know the engine accepted.
try { new Function(src); }
catch (e) { console.log(JSON.stringify({error: String(e)})); process.exit(0); }

// ONE LEFT-TO-RIGHT PASS, not a sequence of replaces. Stripping comments
// before strings makes "http://x" open a line comment that eats the closing
// quote; stripping strings first makes a comment containing an apostrophe open
// a string. Only a scanner that walks the source once gets both right, and
// getting it wrong here produces phantom findings that would train a reader to
// ignore this test.
function strip(src) {
  let out = "", i = 0, prev = "";
  const isRegexPos = () => !/[\w$)\]]$/.test(prev.trimEnd());
  while (i < src.length) {
    const c = src[i], d = src[i + 1];
    if (c === "/" && d === "/") { while (i < src.length && src[i] !== "\n") i++; continue; }
    if (c === "/" && d === "*") { i += 2; while (i < src.length && !(src[i] === "*" && src[i+1] === "/")) i++; i += 2; continue; }
    if (c === "'" || c === '"' || c === "`") {
      const q = c; i++;
      while (i < src.length && src[i] !== q) { if (src[i] === "\\") i++; i++; }
      i++; out += '""'; prev = '""'; continue;
    }
    if (c === "/" && isRegexPos()) {
      let j = i + 1, cls = false, ok = false;
      while (j < src.length) {
        const e = src[j];
        if (e === "\\") { j += 2; continue; }
        if (e === "[") cls = true;
        else if (e === "]") cls = false;
        else if (e === "/" && !cls) { ok = true; break; }
        else if (e === "\n") break;
        j++;
      }
      if (ok) { i = j + 1; while (i < src.length && /[gimsuy]/.test(src[i])) i++; out += "R"; prev = "R"; continue; }
    }
    out += c; prev = out.slice(-40); i++;
  }
  return out;
}

const s = strip(src);
const declared = new Set();
for (const m of s.matchAll(/function\s*\*?\s*([A-Za-z_$][\w$]*)/g)) declared.add(m[1]);
for (const m of s.matchAll(/(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g)) declared.add(m[1]);
for (const m of s.matchAll(/([A-Za-z_$][\w$]*)\s*:\s*(?:function|\()/g)) declared.add(m[1]);
// Parameters, including the ones arrow functions introduce -- `new
// Promise((resolve) => ...)` declares `resolve` and it is then called.
for (const m of s.matchAll(/\(([^()]*)\)\s*=>/g)) {
  for (const part of m[1].split(",")) {
    const name = part.trim().split(/[\s=]/)[0];
    if (/^[A-Za-z_$][\w$]*$/.test(name)) declared.add(name);
  }
}
for (const m of s.matchAll(/function\s*\*?\s*[A-Za-z_$\w$]*\s*\(([^()]*)\)/g)) {
  for (const part of m[1].split(",")) {
    const name = part.trim().split(/[\s=]/)[0];
    if (/^[A-Za-z_$][\w$]*$/.test(name)) declared.add(name);
  }
}
for (const m of s.matchAll(/([A-Za-z_$][\w$]*)\s*=>/g)) declared.add(m[1]);
for (const m of s.matchAll(/catch\s*\(\s*([A-Za-z_$][\w$]*)/g)) declared.add(m[1]);

const KEYWORDS = new Set(["if","for","while","switch","catch","return","typeof",
  "function","new","await","delete","void","in","of","do","else","case","yield",
  "instanceof","throw","super","import","export","async"]);
// Browser globals Node does not have. Listed rather than inferred: the point
// of the test is that an unknown name is a defect, so the set of names that
// are allowed to be unknown has to be written down and short.
const BROWSER = new Set(["confirm","alert","prompt","matchMedia",
  "requestAnimationFrame","cancelAnimationFrame","getComputedStyle",
  "requestIdleCallback","scrollTo","open","close","btoa","atob"]);
const missing = new Set();
for (const m of s.matchAll(/(^|[^.\w$])([A-Za-z_$][\w$]*)\s*\(/gm)) {
  const name = m[2];
  if (KEYWORDS.has(name) || declared.has(name) || BROWSER.has(name)) continue;
  if (name in globalThis) continue;          // fetch, Math, Promise, ...
  missing.add(name);
}

// REFERENCES, NOT ONLY CALLS. `openSettings` reached the gear icon as
// `addEventListener("click", openSettings)` -- a bare identifier, never
// followed by "(" -- so a check that looked for calls found nothing wrong
// with a handler whose function had been deleted. Callback positions are
// where a handler is wired, and they are exactly where a dangling reference
// is invisible until the control is pressed.
const dangling = new Set();
const CALLBACK = [
  /addEventListener\s*\(\s*""\s*,\s*([A-Za-z_$][\w$]*)\s*[,)]/g,
  /\.then\s*\(\s*([A-Za-z_$][\w$]*)\s*[,)]/g,
  /\.catch\s*\(\s*([A-Za-z_$][\w$]*)\s*[,)]/g,
  /\.forEach\s*\(\s*([A-Za-z_$][\w$]*)\s*[,)]/g,
  /\.map\s*\(\s*([A-Za-z_$][\w$]*)\s*[,)]/g,
  /setTimeout\s*\(\s*([A-Za-z_$][\w$]*)\s*[,)]/g,
];
for (const re of CALLBACK) {
  for (const m of s.matchAll(re)) {
    const name = m[1];
    if (declared.has(name) || BROWSER.has(name) || name in globalThis) continue;
    dangling.add(name);
  }
}
console.log(JSON.stringify({missing: [...missing].sort(),
                            dangling: [...dangling].sort()}));
"""
    proc = _run_node(checker, script)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "error" not in result, result["error"]
    assert not result["missing"], (
        "the interface calls functions that are not declared anywhere: "
        + ", ".join(result["missing"])
        + ". A call with no declaration behind it is silent until someone "
          "presses the control that needs it."
    )
    assert not result["dangling"], (
        "the interface wires handlers to functions that do not exist: "
        + ", ".join(result["dangling"])
        + ". The control renders, looks enabled, and throws into the console "
          "when pressed."
    )


def test_every_css_variable_used_is_defined():
    """An undefined custom property is `unset`, not a fallback.

    Four rules in the analysis panel carried a `--dim` colour that nothing ever
    declared, so each inherited its container's colour and the muted hierarchy
    the panel is built around rendered at full contrast. Nothing errors; it
    just quietly looks wrong.
    """
    html = INDEX.read_text()
    style = html[html.index("<style>"):html.index("</style>")]
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", style))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", html))
    missing = sorted(used - defined)
    assert not missing, f"CSS variables used but never defined: {missing}"


def test_no_function_is_declared_twice():
    """A second declaration silently replaces the first.

    `curveSVG` was defined twice. The second drew the shortlist alone; the
    first -- the one written for the page -- drew it against the benchmark and
    never ran once, so the comparison the whole result rests on was absent from
    a chart that looked complete.
    """
    names = re.findall(r"^function\s+([A-Za-z_$][\w$]*)", _script(), re.M)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"declared more than once, so only the last one runs: {dupes}"


def test_the_document_shell_is_intact():
    """The parts of the page that are not CSS and not script.

    Added after an automated pass over the stylesheet rebuilt the file as
    `style + body` and silently dropped everything before `<style>` -- the
    doctype, the charset, the viewport meta and the title. The page still
    rendered, the layout still looked right, and the only visible symptom was
    the browser tab reading "127.0.0.1:8099" instead of "ProSignal". Losing the
    viewport meta would have been worse and even less visible from a desktop
    screenshot: the whole mobile layout depends on it.
    """
    html = INDEX.read_text()
    required = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        'name="viewport"',
        "width=device-width",
        "<title>",
        "</title>",
        "<body>",
        "</body>",
        "</html>",
    ]
    missing = [r for r in required if r not in html]
    assert not missing, f"the document shell is missing: {missing}"
    # Order matters as much as presence.
    assert html.index("<title>") < html.index("<style>"), \
        "the title must be in the head, before the stylesheet"
    assert html.index("</style>") < html.index("<body>"), \
        "the stylesheet must close before the body opens"


def test_the_shell_is_revalidated_rather_than_cached_blind():
    """A deploy must not be one refresh behind.

    The whole application is this one HTML file. `FileResponse` sets an ETag
    and no `Cache-Control`, and with no directive a browser may apply heuristic
    freshness and reuse its copy WITHOUT asking the server -- so after an
    update the operator can be looking at the old markup, the old styles and
    the old script against a new API, with a reload that is itself served from
    cache.

    `no-cache` means "revalidate", not "do not store", so the ETag still saves
    the transfer: a matching one gets a 304 with no body.
    """
    from fastapi.testclient import TestClient

    from prosignal.api import create_app

    client = TestClient(create_app())
    first = client.get("/")
    assert first.status_code == 200
    directive = first.headers.get("cache-control", "")
    assert "no-cache" in directive, (
        f"the shell is served with cache-control {directive!r}; without "
        f"no-cache a browser may serve a stale copy of the entire application "
        f"without contacting the server"
    )
    tag = first.headers.get("etag")
    assert tag, "the shell needs an ETag or every load re-sends the whole file"

    again = client.get("/", headers={"If-None-Match": tag})
    assert again.status_code == 304, (
        "an unchanged shell must revalidate to 304; returning 200 re-sends "
        f"{len(first.content)} bytes on every single page load"
    )

    stale = client.get("/", headers={"If-None-Match": '"not-the-etag"'})
    assert stale.status_code == 200 and stale.content, \
        "a non-matching ETag must serve the current file"
