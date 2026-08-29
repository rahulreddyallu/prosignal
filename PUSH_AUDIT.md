# Pushing the re-audit to `rahulreddyallu/prosignal`

Both passes are written into your working tree. They carry no git history of
their own, so this is one commit made from your Mac.

Run these in order. Steps 0 and 3 print things worth reading before you
continue; the rest are safe to paste straight through.

---

## 0 · Where you are

```bash
cd "/Users/rahulreddyallu/Desktop/Pro Stock Signal BOT"
git remote -v
git branch --show-current
git log --oneline -3
```

Expect `origin` pointing at `https://github.com/rahulreddyallu/prosignal.git`.
Note which branch you are on — step 4 branches from wherever you are now.

If `git remote -v` prints nothing:

```bash
git remote add origin https://github.com/rahulreddyallu/prosignal.git
```

---

## 1 · Remove the file that was renamed

`config/parameters-v2.yaml` became `config/parameters-candidate-stop35.yaml`.
The old copy is still on disk and would be committed as a duplicate.

```bash
rm -f config/parameters-v2.yaml
```

---

## 2 · Drop derived caches from the index

`work/cache/` holds regenerable pickles — `panels.pkl` alone is 134 MB, which
is above GitHub's file limit. The new `.gitignore` excludes them; this removes
them if a previous commit tracked them. It is a no-op otherwise.

```bash
git rm -r --cached --ignore-unmatch -q work/cache work/cache_v1
```

---

## 3 · Stage, then look before you leap

```bash
git add -A
git status --short
```

Then two checks. The first prints the ten largest staged files:

```bash
git diff --cached --name-only | while IFS= read -r f; do
  [ -f "$f" ] && printf '%10d  %s\n' "$(wc -c < "$f")" "$f"
done | sort -rn | head -10
```

Nothing should exceed a few hundred kilobytes. The largest legitimate entries
are `config/parameters.yaml` and `config/parameters-candidate-stop35.yaml` at
about 100 KB each, and `README.md`.

The second checks nothing sensitive slipped through:

```bash
git diff --cached --name-only | grep -E '\.parquet$|\.env$|secret|\.pem$|\.key$' \
  || echo "clean — no parquet, no .env, no secrets"
```

**Three files under `data/` are staged on purpose** and are the point of the
reproducibility work:

| file | what it is |
|---|---|
| `data/curated/MANIFEST.json` | sha256, row count and date span of all 37 store files — 11 KB |
| `data/curated/trial_registry.jsonl` | every configuration compared, with its score — the Deflated Sharpe's input |
| `data/ledger/epochs.jsonl` | which engine produced which result |

No market data is staged. `.gitignore` now excludes `data/curated/` and
`data/ledger/` by *contents* rather than as directories, because git cannot
re-include a file whose parent directory is excluded — which is why the
manifest was uncommittable before.

If you would rather not carry the audit artefacts in the repo:

```bash
git restore --staged ProSignal-Audit.pdf prosignal-audit-pack.tar.gz \
  prosignal-remediation.bundle 2>/dev/null
printf 'ProSignal-Audit.pdf\nprosignal-audit-pack.tar.gz\nprosignal-remediation.bundle\n' >> .gitignore
git add .gitignore
```

---

## 4 · Branch and commit

```bash
git checkout -b audit-pass-2
```

```bash
git commit -F- <<'MSG'
Re-audit: defects priced, R9 refit, provenance, and the restart gate

Two independent passes over the engine after the September readiness dossier.
Nothing here was taken on trust: every figure was regenerated on this tree, and
where the dossier and the measurement disagree the measurement is reported.

PASS 1 -- fifteen defects found and priced, no traded value changed.

  R3  The Deflated Sharpe could not fail. Var[SR] came from woven CPCV paths,
      which duplicate every (split, date) pair, with a fallback of 1/(n-1)
      documented as "conservative unit variance". It returned 1.0000 PASS at
      any trial count. It now returns 0.0000 FAIL, and the verdict is
      insensitive to which variance is used.
  R6  The 3R target could fire only on a close while the stop fired on the
      intraday low -- an asymmetry worth +0.43% per 63-session period.
  R7  Re-entry after an early exit was free. 84% of positions close early.
  R9  The training panel is not the population the book can buy.
  R10 Reported drawdown was the mean across CPCV schedules -- an experience
      nobody had, and always shallower. -13.7% became -19.1%.
  R13 A name with no ADTV received the largest size the slot allows AND the
      cheapest fill in the model.
  Plus R2, R4, R5, R8, R11, R12, R14, R15. Three were found by testing the
  fixes rather than by reading the code.

PASS 2 -- everything the first pass left open, in dependency order: data
provenance, then the point-in-time population, then the coefficients, then the
execution assumptions, and the forward test last.

  R9  The correction existed only in the research panel. `fit_predict` -- what
      the engine actually refits with -- never received it, so the audited
      numbers had moved and the shipped coefficients had not. Threaded through
      the live refit, with the population inside the cache fingerprint so a
      wide-population fit cannot be served to an engine trading the admissible
      one.

      REFITTED. Slots filled 7.29 -> 8.00 of 8. All seven coefficients moved
      and reversal_f crossed the significance floor (t +1.59 -> +2.71), so the
      engine trades three themes where it traded two. The book gets WORSE
      against its benchmark: -3.88% -> -4.19% per period on the CPCV weave,
      -4.35% -> -5.20% on the purged walk-forward. The ranking's own edge over
      equal-weight falls from +1.16% to +0.10%. Part of the edge this engine
      was credited with was earned on names it cannot buy.

  R13 Four liquidity states. Unknown means no position, no optimistic fill and
      no imputation; `adtv_inr` is None whenever untradable so a caller that
      ignores the gate raises rather than sizes. Unknown-liquidity impact
      5.0 -> 105.0 bps. The execution model is pinned by monotonicity
      properties, not examples.

  W2  Winner's-curse correction, one-sided truncated-normal MLE. Five of six
      acceptance criteria hold; the sixth was fixed in advance and is not met,
      so the number is REPORTED and NOT TRADED and `assert_not_traded` makes
      wiring it into a score fail a test. On the positive tail a true effect of
      zero returns +0.59 under the correct estimator and +1.00 under the
      two-sided one.

  C3/C4 Outcomes carry `epoch_id`. Statistics partition on it as they already
      did on `exit_model`; the retired record is served LABELLED beside the
      current epoch rather than dropped, with the size of the pooling error
      printed next to it.

  D1  A content-addressed manifest of the store -- 37 files, 9,270,123 rows,
      digest 6b6737fc418864aa. The digest is over content, never mtime, and is
      recomputed on load rather than trusted. `.gitignore` excluded
      `data/curated/` as a directory, which made the manifest itself
      uncommittable; it now excludes by contents.

  D2  An append-only research-epoch ledger binding code, config, data, feature
      schema, universe policy and execution model into one identity. v1 is
      archived VOID; v2 is open as 2026-08-29-6451d9181041cdb4.

  R1  NOT RESTARTED. `research forward --start/--restart` now refuses while a
      restart-blocking finding is open, the manifest is unverified, or no epoch
      describes the engine -- naming every reason at once, before the overwrite
      check. Every precondition now holds and the gate permits a restart.
      Starting it discards the observations collected so far and begins an
      eighteen-month clock, which is an operator's decision.

READY now means eight gates -- data, universe, features, model, execution,
validation, reproducibility, forward test -- rather than a green suite. Seven
pass. The eighth is the forward window.

New commands: `data manifest [--verify]`, `research epoch {status,list,open,
close}`, `research findings [--id]`, `research readiness`, `research record`.

VERIFICATION. 1,551 tests pass, 0 fail, 9 skipped -- from 1,376 on a clean
checkout. The mutation probe reverts 43 repairs one at a time and every one is
caught; four only after the test meant to catch them was found incapable of it,
including a W2 guard whose statistic cancelled the very defect it watched for.

THE VERDICT IS UNCHANGED: NOT READY. The shipped book returns about four points
per 63-session period LESS than an equal-weight hold of the universe it selects
from, the Deflated Sharpe fails on all four constructions, and the deficit sits
entirely in the exit layer -- `no exits at all` lands within half a point of the
benchmark. What changed is that the engine now says so itself.
MSG
```

---

## 5 · Push

```bash
git push -u origin audit-pass-2
```

---

## 6 · Merge

Either open a pull request on GitHub from `audit-pass-2`, or merge locally:

```bash
git checkout main
git merge --no-ff audit-pass-2 -m "Merge the re-audit: R9 refit, provenance, restart gate"
git push origin main
```

`--no-ff` keeps the work identifiable as one unit in history.

---

## 7 · Confirm the engine agrees, from the merged tree

```bash
prosignal data manifest --verify
prosignal research epoch status
prosignal research readiness
prosignal research findings
```

`readiness` should print seven PASS and one FAIL (FORWARD), and end with *"The
forward test MAY be restarted: every precondition holds."* Restarting is yours
to decide; nothing in this commit does it.

---

## If the push is rejected

`! [rejected] ... (fetch first)` means the remote moved since your last fetch.
Do **not** force-push. Rebase onto it and look at what changed:

```bash
git fetch origin
git log --oneline HEAD..origin/main
git rebase origin/main
```

If a rebase conflict lands in a file this audit rewrote — anything under
`src/prosignal/validation/`, `src/prosignal/liquidity.py`,
`src/prosignal/data/manifest.py` — stop and check which side is which before
resolving. Those files carry the fixes and the guards together, and taking the
wrong side leaves a repair whose test still passes.
