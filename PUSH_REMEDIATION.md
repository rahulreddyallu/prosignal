# Pushing the remediation to GitHub

The 18 commits exist and are verified, but this session is not authorised to
write to `rahulreddyallu/prosignal` — the git proxy refuses to inject a
credential for a repo outside the session's source list. Read access works;
`origin/main` is still at `a359622`, so nothing of mine has landed.

Two ways forward. The first is faster if you want me to do it.

---

## Option 1 — authorise this session, and I push

Add `rahulreddyallu/prosignal` to this session's sources, then tell me. I will
push `remediation`, open the merge, and confirm the result.

## Option 2 — apply the bundle yourself, from your Mac

`prosignal-remediation.bundle` is beside this file, in your
`Pro Stock Signal BOT` folder. It carries all 18 commits with their messages,
authorship and hashes intact — it is not a squashed patch.

```bash
cd "/Users/rahulreddyallu/Desktop/Pro Stock Signal BOT"

# 1. sanity: you should be at the base commit these apply to
git fetch origin
git log --oneline -1 origin/main          # expect a359622

# 2. bring the branch in. This does NOT touch your working tree.
git bundle verify prosignal-remediation.bundle
git fetch prosignal-remediation.bundle remediation:remediation

# 3. look before you leap
git log --oneline main..remediation       # expect 18 commits
git diff --stat main..remediation         # expect 22 files, +3153 / -164

# 4. push the branch
git push -u origin remediation
```

Then either open a pull request on GitHub, or merge locally:

```bash
git checkout main
git merge --no-ff remediation -m "Remediate the August audit findings

18 commits. Every defect worked to a root cause, each fix carrying a
regression guard shown to fail without it. Mutation probe 14/14 caught.
Suite 1,420 passed, 0 failed.

Config unchanged at baseline-v1@127d8a314ec49aa2 - the diff to config/ is
comments only. Verdict remains NOT READY: the book underperforms the
equal-weight eligible universe by 4.23% per period."
git push origin main
```

`--no-ff` is deliberate: it keeps the remediation identifiable as one unit in
history rather than dissolving 18 commits into the mainline.

---

## What you are merging

| | |
|---|---|
| Commits | 18, base `a359622` → head `21aa26c` |
| Files | 22 — source, tests, config comments, docs |
| Lines | +3,153 / −164 |
| Suite | 1,420 passed, 0 failed, 51 skipped |
| Mutation probe | 14 of 14 deliberate reversions caught |
| Config hash | unchanged, `127d8a314ec49aa2` |
| Touches `data/`, `.env`, tokens | no — verified |

The merge does not make the system ready. It makes it honest: the measurement
apparatus now reports the engine's performance accurately, and what it reports
is negative. That distinction is the whole point, and it is why merging is
still worth doing.
