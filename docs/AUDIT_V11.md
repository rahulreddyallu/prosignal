# ProSignal — Audit V11: does the screen tell the truth, and does the machine run itself

Companion to `BUILD_PLAN_V10.md`. V10 audited the engine and read the interface
from source. This pass renders the interface and interrogates the host.

---

## 1. Header

| field | value |
|---|---|
| audit date | 2026-09-05 |
| code sha | `f7595d7568d41e3343e0259aa62888849738aeda` (branch `build-plan-v10-phase0-2`) |
| config fingerprint | `baseline-v2@d86f0b3d983ba00f`, 201 parameters (127 UNVALIDATED) |
| store as-of | prices to 2026-09-03; delivery 2026-09-03; corporate actions to 2026-09-04; statements 2026-06-30; **fundamentals 2025-03-11 (543 days)** |
| tree state | measured against the Phase 0–5 working tree, committed as `f7595d7` immediately after so the sha above names exactly what was audited. `data/` verified byte-identical before and after (`runs-2026.jsonl` sha `efd278660516f2e1`, `outcomes.jsonl` sha `f78fb2c3efdddb1c`, 8 run-detail files) — see D-101 for the one write that had to be reverted |
| suite at audit time | **1,702 passed · 2 failed · 4 skipped** (958 s). The two failures are the deliberately-deferred `test_data_manifest` and `test_restart_gate` |
| **host every measurement was taken on** | `Rahuls-MacBook-Air.local` — macOS 15.0.1, arm64 **T8103 (M1)**, TZ `Asia/Kolkata`, Python **3.9.6** |
| UI measurements | Chromium against `uvicorn` on `127.0.0.1:8932`, served from a 923 MB copy of `data/` in a scratch directory |

**There is no second host.** Everything below was measured on the development
machine because that is the only machine that exists. See D-102.

---

## 2. Verdict

The screen now tells the truth about the run in front of it, and that is a real
change: the theme table renders five themes summing to **exactly 100%** with the
momentum cap landing on **exactly 40.00%**, the label reads **`Low-margin tilt`**
where the code says `quality`, the cost is priced per name (**54–84 bps**, never
`40 bps`), and the "what is evidenced here, and what is not" disclosure — which
had been unreachable for the entire life of the product — **renders, and reads as
a disclosure rather than an alarm**. The single thing most wrong is not on the
screen at all: **nothing schedules this system on any host**
(`scripts/cloud-init.sh:170` is an unexecuted template; `crontab -l`, `launchctl
list`, `/etc/cron.d`, `systemctl` and `.github/workflows` are all empty or
absent), the nightly script has run **exactly four times in its life**, all by
hand, on 2026-08-23 and 2026-08-24, **none of them at its own 20:30 IST slot**
(`data/ledger/forward.log:1,707,1413,2118`), and it has not run in the twelve
days since. Everything Part B was written to test — timezone drift, holiday
duplicates, missed-window catch-up, manifest churn under a daily cron — is
downstream of a cron that does not exist, so the honest finding is that this
system is not automated, not unattended, and not self-reporting, and the second
most serious finding is that a documented sandbox switch (`PROSIGNAL_CONFIG`) is
honoured by exactly one call site in the codebase, which is how this audit wrote
a `live` row into the production ledger while explicitly trying not to.

---

## 3. Defect register

### P0

---

**D-102 · P0 · VERIFIED · no scheduler exists on any host**

**Address:** `scripts/cloud-init.sh:170-176` (the only cron definition in the
repo), `render.yaml:11-20` (the only deployed service definition), and the
absence of any scheduler on the measured host.

**How I found it.** Enumerated from the machine rather than from `scripts/`:

```
crontab -l                        -> "no crontab for rahulreddyallu"
/etc/cron.d, /etc/crontab         -> do not exist
~/Library/LaunchAgents            -> Google, Microsoft, MathWorks. Nothing else.
launchctl list | grep -i prosignal-> no match (rc=1)
.github/workflows                 -> directory does not exist
render.yaml                       -> one `type: web` service, no cron service
```

`scripts/cloud-init.sh` *does* define the job — `30 20 * * 1-5` under
`timedatectl set-timezone Asia/Kolkata`, which is correct: 20:30 IST leaves five
hours after the 15:30 close for the bhavcopy. But it is AWS "user data", a
first-boot template still carrying its `__DOMAIN__` and `__TOKEN__`
placeholders, and no instance has run it. `render.yaml`, the other deployment
path, defines a web service only — **Render has no cron here at all**, so the
two deployment paths in this repo are not merely different, they disagree about
whether the system has a daily job.

**Blast radius.** Everything the daily job is for. The forward test accumulates
no observations. Outcomes are resolved only when somebody opens the page. Prices,
delivery and corporate actions age until a human runs `data ingest`. The store
is one session behind as of this audit and would be behind by however long
nobody looks.

**Evidence it has never run on a schedule.** The script's own log is the record:

```
$ grep -n "observation start" data/ledger/forward.log
1:[2026-08-23T05:43:38Z]     -> 11:13 IST
707:[2026-08-23T05:59:16Z]   -> 11:29 IST
1413:[2026-08-24T02:57:16Z]  ->  8:27 IST
2118:[2026-08-24T03:03:27Z]  ->  8:33 IST
total: 4        last write: 2026-08-24 08:34 IST (12 days ago)
```

Four runs, in a two-day window, at four times of day, **none within nine hours
of 20:30 IST**. These are hand-runs. And this log lives in the development
checkout — if a deploy host were running the job, the file would be there, not
here.

**Which host it would run on if it ran.** The M1 laptop. That is the P0 the
prompt anticipated, and it is worse than anticipated: it does not run there
either. A laptop cron would stop when the lid closed; this one does not start.

**Fix.** Two decisions, in order: (1) choose one of the two deployment paths and
delete the other, because keeping both guarantees that whichever one is read is
the wrong one; (2) stand the chosen one up. If Render: add a `type: cron`
service, and note that Render cron services do not share the web service's disk,
so the job needs its own access to the store — this is a design change, not a
config line. If EC2/cloud-init: run it, fill the placeholders, and set
`PROSIGNAL_ALERT_CMD` in `/etc/prosignal.env` (D-105).
**Estimate: 3 agent-hours to make the repo state coherent; standing up a host is
an operator action with a card and a domain, not an agent-hour.**

---

### P1

---

**D-101 · P1 · VERIFIED · `PROSIGNAL_CONFIG` is honoured by one call site out of three**

**Address:** `src/prosignal/api.py:63`, `src/prosignal/cli.py:2683`,
`src/prosignal/config/loader.py:256-263`, `src/prosignal/outcomes.py:178`.

```python
# loader.py:256 -- the accessor that reads the environment
def get_config(**kwargs):
    """Process-wide accessor. Honours ``$PROSIGNAL_CONFIG`` if set."""

# api.py:63   -- does NOT use it
cfg = config or load_config()
# cli.py:2683 -- does NOT use it
cfg = load_config(config_path=args.config)      # args.config defaults to None
# outcomes.py:178 -- the ONLY caller that does
ledger_root = Path(get_config().paths.ledger)
```

**How I found it.** By being bitten. I started the API with
`PROSIGNAL_CONFIG` pointed at a scratch config whose `runtime.paths` were
absolute paths into a scratch copy of `data/`, drove one scan from the rendered
page, and then found the run in the **production** ledger:

```
data/ledger/runs-2026.jsonl        1686 -> 1687 lines, the new row mode="live"
data/ledger/runs/2026-09-03_20260905-072821_ad2e3f253bd5.json   (created)
data/ledger/outcomes.jsonl         128 -> 158 lines (resolved at startup)
scratchpad/data/ledger/...         unchanged
```

**Blast radius.** This is the failure mode where a defence is worse than no
defence. `outcomes.py` *does* honour the variable, so setting it does not simply
have no effect — it splits the process in half: outcomes resolve against the
store you asked for while the pipeline reads and **writes** the one you did not.
Anyone pointing a staging instance, a candidate config, or an audit at a copy of
the store gets production writes and a silent split-brain. It is also the exact
mechanism by which a `live` ledger row — the rows that feed the forward test and
the open-book memory — can be created by someone who believed they were
sandboxed.

**Verification that it is the cause, not a coincidence:** `core/paths.py:83-86`
resolves absolute config paths correctly (`p.resolve() if p.is_absolute()`), and
a harness that passes the config *explicitly* to `create_app` writes to the
scratch store and leaves production byte-identical — confirmed by hash across
every subsequent scan in this audit (`efd278660516f2e1`, unchanged).

**Repair performed.** The production store was restored to its pre-audit state
and verified byte-exact: `runs-2026.jsonl` truncated to its first 1,686 lines
(sha `efd278660516f2e1`, matching the pre-scan snapshot), `outcomes.jsonl` to its
first 128 (sha `f78fb2c3efdddb1c`), the stray rundetail deleted, `jobs.sqlite3`
restored. The 30 outcome rows were deterministic resolutions that any page load
would have produced; they were reverted anyway so the store is exactly as found.

**Fix.** `api.py:63` → `cfg = config or get_config()`; `cli.py:2683` →
`get_config(config_path=args.config)` (the accessor already prefers an explicit
path over the environment). Then a test that sets the variable, builds the app,
and asserts `cfg.paths.data` is the override.
**Estimate: 1 agent-hour, including the test.**

---

**D-103 · P1 · VERIFIED · a pre-fix run renders as the deleted fitted model, marked validated, with the caveat suppressed**

**Address:** `src/prosignal/presentation/viewmodel.py:167`,
`src/prosignal/rundetail.py:73-77` (the cache key), `index.html:2196-2205`.

D-004's fix keyed `_scorer_used` on `evidence_tier`, which is correct for any run
the current engine produced. The branch it replaced was **left in place** for
runs that carry no tier:

```python
if "v3_theme" in tiers:            # the new, correct path
    ...
if "model" in tiers or (seen & (MODEL_KEYS - COMPOSITE_KEYS)):
    return {"model": "cross-sectional", "validated": True, "note": None}
```

`MODEL_KEYS - COMPOSITE_KEYS` is `{beta, delivery, drawdown, lottery, mom,
reversal, skew}` — every stored pre-v3 payload matches it.

**Why the display cache lets this reach the screen.** `rundetail.save` names
files `{as_of_date}_{generated_at}_{run_id}.json` and `load_latest` returns the
newest by market date. **The key contains neither `code_sha` nor
`config_version`** — the prompt asked whether that is a defect on its own, and it
is: the payload carries `config_version` (`rundetail.py:132`) but nothing
compares it to the running engine, so a payload written by any past engine is
served by the current one with no marker.

**VERIFIED render.** I served the stored `baseline-v1@9776e5d6b3a3dd3e` payload
(2026-08-25, pre-fix) dated to the store's latest session — the deploy-day
condition, where the code is upgraded and the last run predates it:

```
/today  ->  "scorer": {"model": "cross-sectional", "validated": true, "note": null}
```

and the rendered page:

| assertion | rendered |
|---|---|
| heading | `Ranked from the close of 2026-09-03.` — today's session |
| evidence disclosure box | **absent** (`hasEvidenceBox: false`) |
| decisions shown | **5 BUY** |
| why-line | `Lottery-like payoff shape leads at -1.94 sd` |
| market trend | `Uptrend, slope +19.0%` — the real 2026-09-03 reading is `Range-bound, +14.0%` |

An eight-day-old ranking from a model deleted on 2026-09-03, under today's date,
with BUY badges, and **the caveat suppressed precisely because `validated` is
true** — `index.html` renders the disclosure only when it is false. This is
D-004 intact on the path the fix did not cover.

**Mitigation that already exists, and its limit.** `isCurrent()`
(`index.html:2113-2127`) compares the payload's `as_of` with `/ready`'s
`latest_session` and, when they differ, replaces the slate with a "New market
data / Scan" prompt — I saw that intercept before forcing the dates to match. So
the exposure is not "any old payload"; it is the specific and likely case where
**the store is current and the newest run predates a deploy**. That is deploy day.

**Fix.** Delete the `cross-sectional` branch — the model it names does not exist.
An untiered payload should fall through to the existing `unknown / severity:
alarm` return, which is the safe direction and already written. Separately, stamp
`code_sha` into the rundetail payload and have `load_latest` refuse — or visibly
mark — a payload whose `config_version` or `code_sha` is not the running one.
**Estimate: 2 agent-hours.**

---

**D-104 · P1 · VERIFIED · `/ready` returns green with 543-day-old fundamentals**

**Address:** `src/prosignal/api.py` `/ready` handler.

```json
{"ready": true,
 "fundamentals": "newest filing 2025-03-11 (543 days old), 186 symbols",
 "fundamentals_stale": true}
```

The dimension is computed, named, and reported with its own number — which is
what V10's D-022 asked for, and that half is done. But `fundamentals_stale:
true` does not participate in `ready`, so the endpoint the deploy health check
watches, and the badge the operator reads, are green while a feed is eighteen
months dead.

This matters more than it did before Phase 5. The `Low-margin tilt` theme now
runs at **18.99%** of every name's score instead of 1.67%, because the
`write_statements` repair took fundamental coverage from 8.8% to 100%. An
eleven-fold exposure increase to a tilt whose inputs are stale is exactly the
condition `/ready` exists to refuse.

**Note the two feeds are different and only one is dead.** The model reads
`statements.parquet` (newest period 2026-06-30, 758 symbols — current).
`fundamentals.parquet` is the frozen NSE endpoint. `/ready` reports both. The
defect is that neither staleness gates the verdict.

**Fix.** `ready = ... and not fundamentals_stale`, with
`max_fundamental_age_days` as the threshold (already present in
`config/parameters.yaml`), and a `/ready` 503 body that names the feed and its
age.
**Estimate: 1 agent-hour.**

---

**D-105 · P1 · READ · the alert path is inert on the only host that was ever meant to run it**

**Address:** `scripts/forward_run.sh:32-45` vs `scripts/cloud-init.sh:97-104`.

`forward_run.sh` alerts on failed ingest, failed analysis, and an invalid forward
test, through `PROSIGNAL_ALERT_CMD` — "inert unless set", by design.
`cloud-init.sh` writes `/etc/prosignal.env` with `PROSIGNAL_AUTH_TOKEN`,
`PROSIGNAL_PUBLIC` and the three allocator variables, and **does not set
`PROSIGNAL_ALERT_CMD`**. The cron sources that file and nothing else.

So on the deployment the repo describes, every failure the script is careful to
detect goes to `data/ledger/forward.log` and stops there. The script's own
comment states the consequence: "a job that stops in week three is discovered in
week eleven."

**Measured:** zero alerts have ever fired — `grep -cE "FAILED|INVALID|could
not|NO names" data/ledger/forward.log` returns **0** across all four runs.
That is consistent with four clean runs; it is not evidence the path works.

**Fix.** Add `PROSIGNAL_ALERT_CMD` to the `cat > /etc/prosignal.env` heredoc with
a healthcheck.io ping or equivalent as the default, and make the setup script
fail loudly if it is left empty. A dead-man's switch (alert on *absence* of a
run) is the version that catches D-102; an alert on failure cannot fire when the
job never starts.
**Estimate: 1 agent-hour for the wiring; the dead-man's switch is an external service.**

---

### P2

---

**D-106 · P2 · VERIFIED · nothing re-manifests after ingest, and the manifest under-describes the store**

**Address:** `scripts/forward_run.sh` (zero occurrences of "manifest"),
`src/prosignal/data/manifest.py:215` (`write` is called only from
`cli.py:2717 cmd_data_manifest`).

The prompt asked which of three is true. **Option 2**, with an extra:

```
$ .venv/bin/python -m prosignal.cli data manifest --verify
DRIFTED -- 10 discrepancies:
  delivery/year=2026.parquet   changed   5,555,905 -> 5,585,439
  indices/year=2026.parquet    changed   1,234,600 -> 1,241,919
  prices/year=2026.parquet     changed  22,451,944 -> 22,580,636
  prices_secondary.parquet     changed     124,890 ->   131,432
  sector_map.parquet           changed       7,120 ->     9,838
  statements.parquet           changed     253,563 ->   916,771
  fo_lots.parquet              untracked
  results_calendar.parquet     untracked
  security_list.parquet        untracked
  shareholding.parquet         untracked
```

The gate has not been "relaxed to make the cron work" — `git log` on
`tests/test_restart_gate.py` and `data/manifest.py` shows one commit each
(`5ef4196`), so option 3 is refuted. The gate is failing, honestly, and the
suite reports it.

**The extra finding is the four `untracked` rows.** `fo_lots`,
`results_calendar`, `security_list` and `shareholding` are on disk and absent
from the manifest — so even a freshly-written manifest does not describe the
whole store, and `digest_of(root)` names a subset. `manifest.py:62-72` excludes
lock files, `_state.json`, `trial_registry.jsonl` and `crosssec_model.json` on
purpose; these four are not in that list, they are simply missing.

**Fix.** Add `data manifest --write` to `forward_run.sh` after a successful
ingest and before `analyse run`, and open the epoch there — the manifest and the
epoch are one decision, as the CLI's own message says ("Re-ingest, or
re-manifest and open a new epoch"). Then establish why four tables are untracked:
either `_walk` misses them or they postdate the manifest's last write.
**Estimate: 2 agent-hours.**

---

**D-107 · P2 · VERIFIED · the card claims 22 factors on a name where 18 were measurable**

**Address:** `index.html` (`"22 factors behind them"`), theme `members[].rank`.

Every card prints `22 factors behind them`. Measured across the six names on the
2026-09-03 cross-section:

| ticker | members with `rank: null` |
|---|---|
| SKYGOLD | 1 (`resid_rev_21`) |
| HFCL | **4** (`mom_2_0`, `prox_52w_now`, `price_vs_vwap_20`, `rev_1w`) |
| MBAPL, ASTERDM, GRWRHITECH, SJS | 0 |

For HFCL, **18% of the factors the card names did not exist for that name**, and
the theme carried its full weight regardless. Theme-level coverage *is*
disclosed — `whyLine` appends `· N of 5 themes` when a theme is missing
(`index.html:2440-2447`), and it correctly printed nothing here because all six
names carry all five themes since the statements repair. Member-level coverage is
not disclosed anywhere.

This is the honest remainder of D-007/D-024: the theme population is now uniform,
the factor population inside a theme is not.

**Fix.** Make the count per name — `N of 22 factors` — and render the shortfall
in the panel where the members already are.
**Estimate: 2 agent-hours.**

---

**D-108 · P2 · VERIFIED · the store is 923 MB against a 3 GB ceiling, and the two largest consumers are unbounded**

**Address:** `config/parameters.yaml:152-171` (`storage.max_total_mb: 3072`,
`raw_cache.max_mb: 384`), `src/prosignal/jobs.py` (no retention of any kind).

The prompt's premise — "the store was 251 MB" — describes `data/curated` (246
MB). The whole tree:

| path | size | bounded? |
|---|---|---|
| `data/cache` | **505 MB** | capped at **384 MB** and 31% over it |
| `data/curated` | 246 MB | grows ~19 MB/year in prices |
| `data/jobs.sqlite3` | **121 MB** | **no retention, no prune, no vacuum** |
| `data/ledger` | 40 MB | of which **19.7 MB is `.pre-lineage-repair` backups** |
| `data/raw` | 10 MB | rolling, off by default |
| **total** | **923 MB** | against `max_total_mb: 3072` |

`jobs.sqlite3` holds **594 rows at ~204 KB each** — every job stores its full
result payload. `grep` for `KEEP|retention|prune|vacuum|DELETE FROM|max_jobs` in
`jobs.py` returns nothing; `reap_stale` marks stale *running* jobs and deletes no
rows. At one nightly run per session that is ~50 MB/year before a human presses
Scan.

`data/cache` is the NSE HTTP cache (`nsearchives` 312 MB, `www.nseindia.com` 119
MB, `archives` 74 MB) and is **over its own configured cap** despite an ingest
having run on 2026-09-04 — eviction is documented as running "after every
ingest" and evidently is not bringing it under 384 MB.

**And the backups do accumulate**, as the prompt suspected: four
`.pre-lineage-repair` files totalling 19,709,542 bytes, written by
`data lineage --repair`, with nothing to prune them.

When `max_total_mb` is reached the documented behaviour is "evict, then
**refuse**" — the ingest stops, which means the daily observation stops. The
things that would fill it are all non-durable.

**Fix.** Retention on `jobs` (keep 60, matching `rundetail.KEEP_RUNS`) plus a
`VACUUM`; find out why cache eviction is not honouring 384 MB; prune
`.pre-lineage-repair` backups older than the newest two.
**Estimate: 3 agent-hours.**

---

**D-113 · P2 · VERIFIED · when the display cache cannot be written, the screen silently serves the previous run**

**Address:** `src/prosignal/rundetail.py:54-87` (`save` never raises, returns
`None`), `pipeline.py:453` (return value discarded), `rundetail.py:108-121`
(`load_latest`).

V10 predicted this state renders "an empty evidence panel". It does not. It
renders the **previous run's slate, under the current date, with no marker.**

Forced by making the run-detail directory read-only in the scratch store and
running the engine:

```
rundetail files   8 -> 8       (the write failed, silently, by design)
ledger rows    1687 -> 1688    (the run WAS recorded)

ledger newest : run_id e03df4e43e14   logged 2026-09-05T07:44:39
screen serves : run_id c37d69837621   logged 2026-09-05T07:39:14
screen as_of  : 2026-09-03            picks: SKYGOLD MBAPL HFCL ASTERDM GRWRHITECH SJS
```

The ledger and the screen disagree about which run is the record. Because the
stale payload carries the *same* `as_of`, `isCurrent()` passes and the staleness
prompt that saved D-103 in the ordinary case does not fire here. The comment on
`save` is right that a display-cache failure must not fail the run — the run is
in the ledger, which is the record. What is missing is that nobody is told the
screen is not showing it.

**Also verified here, and it belongs to D-112:** the CLI prints both
data-quality flags in full — `promoter-pledging data is absent … NOT_TESTABLE`
and `35 of 750 names failed Stage 1` — on the same run where the web card shows
neither. That is the cross-surface disagreement in one command.

**Fix.** Have the pipeline record that `save` returned `None`, and have `/today`
compare the served payload's `run_id` against the ledger's newest row for that
date, rendering a one-line notice when they differ.
**Estimate: 1.5 agent-hours.**

---

**D-109 · P2 · READ · the tested Python and the deployed Python are three minor versions apart**

**Address:** `.venv` is **Python 3.9.6**; `render.yaml:23` pins
`PYTHON_VERSION: "3.12"`; `cloud-init.sh:47` installs the distribution
`python3-venv` (Ubuntu 24.04 → 3.12).

Every number in this audit and every one of the 1,702 passing tests was produced
on 3.9.6. `requirements.txt` resolved under 3.9 and under 3.12 does not give the
same pandas/pyarrow/numpy wheels, and this engine's results depend on pandas
groupby and sort semantics in places the audit trail has already been bitten by
(`write_statements`, the dedup key). `requirements.lock.txt` exists but is
untracked, so it pins nothing for the deploy.

**Fix.** Build the venv on the version the deploy uses, re-run the suite there,
and commit the lock file. This is cheap and it is the only way the manifest and
epoch story means anything across hosts.
**Estimate: 2 agent-hours, plus one suite run.**

---

### P3

---

**D-110 · P3 · VERIFIED · the card says its rows "sum to the score" and the score is not on the screen**

The panel heading reads `WHAT ORDERED IT · sums to the score`. The rows sum to
0.70732 for SKYGOLD, which is `composite_raw` — and `composite_raw` appears
nowhere on the card or panel. The displayed `RANK #1` is not it, and
`composite_score` (the rank-normalised 0–1 value) is not it either. The claim is
true and unverifiable by the reader.

Related, and the reason the prompt's arithmetic assertion cannot be run from the
DOM as written: the table's columns are **THEME · PCT · ADDS**, where PCT is the
percentile `(z+1)/2*100`, not `z`. Multiplying the two visible numbers does not
give the third. The relation `z × w == contribution` **does** hold — verified
below — but only against the payload.

**Fix.** Print the composite next to the heading, or change the heading.
**Estimate: 0.5 agent-hours.**

---

**D-111 · P3 · VERIFIED · `/ready` and `data lineage` describe the same rows in opposite words**

`/ready` reports `"ledger_conflicts": 0`. `data lineage` reports:

```
DATES RECORDED ON THE DAY THAT DISAGREE WITH THEMSELVES:
  2026-08-17  130 runs, 2 different books, 3 config versions
  2026-08-25  105 runs, 8 different books, 8 config versions
```

Both are correct. 130 + 105 = **235**, exactly the quarantine count, so there are
zero *live* conflicts and 235 rows already set aside. An operator reading the two
surfaces in sequence has no way to know that.

**Fix.** Have `data lineage` say "already quarantined" against dates that are.
**Estimate: 0.5 agent-hours.**

---

**D-112 · P3 · VERIFIED · data-quality flags are on the payload and off the screen, deliberately**

`/today` carries:

```
"flags": ["promoter-pledging data is absent, so disclosure-date alignment cannot
          be enforced. The Stage 3 pledging gate reports NOT_TESTABLE -- it does
          not pass.",
          "35 of 750 names failed Stage 1 data-quality checks and were excluded."],
"complete": false
```

Neither string appears in the rendered DOM. `index.html:2376-2378` says why:
"The data notes lived here and are gone at the owner's request; the flags remain
on the run payload and in the ledger for anyone reading the record rather than
the screen."

Recorded as a finding rather than a defect: it is a documented owner decision. It
is listed because a gate that reports NOT_TESTABLE and is invisible is the shape
of thing this audit exists to surface, and because the decision predates the
disclosure box that now sits at the top of the page and could carry it cheaply.

---

## 4. Refutations

Things this prompt or V10 asserted that did not survive contact with the code.

| # | claim | what is true | address |
|---|---|---|---|
| 1 | "fundamentals now ingest on a daily cron" | no cron exists anywhere; fundamentals are **543 days old** | D-102, D-104 |
| 2 | `run_kind` labels ledger rows | **there is no `run_kind` field.** All 1,942 rows carry `mode` ∈ {live, replay, quarantine}. The Part C gate as written filters to zero rows and `max()` raises on an empty Counter | `ledger.py`, `pipeline.py:176` |
| 3 | "a cron that writes `run_kind: live` on a catch-up re-creates D-008 one row at a time" | the caller does not label the row. `pipeline.py:176` derives `mode = "live" if as_of is None else "replay"` from a single append site (`pipeline.py:451`). A catch-up with `--as-of` is `replay`; the nightly without it is `live`. The derivation is correct | `pipeline.py:176,451` |
| 4 | "the honest trial count feeds the Deflated Sharpe directly" from ledger rows | `n_trials` comes from `trial_registry.jsonl` via `reg.count()` (`cli.py:1384`), **not** from the ledger. The 200 duplicate live rows on 2026-08-18 do not inflate the DSR | `cli.py:1362-1384`, `validation/registry.py:154` |
| 5 | "a 409 from the single-flight lock" | a second `POST /analysis/run` of the **same kind** returns **HTTP 200** with the running job and `already_running: true` — it joins rather than erroring. 409 is for a job of a *different* kind (`OPERATING.md` item [3]) | measured |
| 6 | "`validation/decay.py` was CLI-only with no threshold defined anywhere… If there is no number, the monitor is decoration" | **a number exists and is pre-registered**: `decay_monitor: window_dates: 24, kill_t_stat: 0.0, required_breaches: 24, post_publication_haircut: 0.58`, with the comment "Declared before the numbers were looked at." The half that stands: **nothing runs it on a schedule** — `forward_run.sh` calls `data ingest`, `analyse run`, `research forward`, and no decay command | `config/parameters.yaml:1570-1581` |
| 7 | "the store was 251 MB" | 251 MB is `data/curated`. The tree is **923 MB** | D-108 |
| 8 | grep `index.html` for `750`, `mom_6_1_r` as live lies | both survive only in **comments and dead fallbacks**: `750` is inside a CSS comment (`index.html:538`), and `whyLine`'s `mom_6_1_r` branch (`:2450`) is unreachable whenever `p.themes` is non-empty, which is every v3 run. Neither renders. `750` is also a *live* number — `data.flags` says "35 of 750 names failed Stage 1" | measured on the DOM |
| 9 | "the restart gate may have been relaxed or removed to make the cron work" (option 3, "the one to look hardest for") | refuted. `git log` shows one commit each on `tests/test_restart_gate.py` and `data/manifest.py` (`5ef4196`). The gate is intact and failing honestly | measured |
| 10 | V10 D-007 "CLOSED — every one of 386 names is scored on all five themes" | true at the theme level, and it is why the D-003 mutant survived. Not true at the factor level: HFCL carries 4 null members of 22 | D-107 |

---

## 5. Exit gates

| gate | result |
|---|---|
| theme weights sum to 100% on every card | **PASS** — all six names `sum = 1.00000` exactly |
| no theme prints above 40.0% | **PASS** — `max = 0.4000` on all six, exactly |
| `z × w == contribution` | **PASS** — max error `1.8e-4`, entirely 3-dp display rounding on `z` |
| the number shown is that name's effective weight, not `Theme.weight` | **PASS at the payload.** Not separable on today's data: coverage is 100%, so the two are numerically identical. Pinned by test, not by this cross-section (see V10 Round 8) |
| `quality` appears nowhere an operator can see | **PASS** — absent from Today and History DOM; `Low-margin tilt` present |
| scorer block: `v3_composite`, `validated: false`, warning branch renders | **PASS** — and the `index.html:2182` branch renders **for the first time in the product's life**, as a disclosure |
| no visible `40 bps`, `mom_6_1_r`, `cross-sectional`, `NIFTY 200` | **PASS** on both tabs |
| no `NaN` / `undefined` / `null` / `0.00%` | **PASS** — zero occurrences on both tabs |
| cost is per name and per size | **PASS** — 54, 60, 65, 65, 68, 84 bps with impact broken out |
| expectancy absent | **PASS** — every field null, nothing rendered |
| mobile 375 px: theme table survives | **PASS** — table 327 px, `scrollWidth == clientWidth`, no horizontal page overflow |
| `max(live runs per market date) == 1` | **FAIL** — 200 on 2026-08-18, 135 on 2026-08-21, 6 on 2026-09-02. `Ledger.append` has no uniqueness invariant; the protection is the opt-in `--skip-if-recorded` |
| the cron's invocation is idempotent | **PASS** — `analyse run --watch 0 --skip-if-recorded` on an already-recorded session exits 0, refuses loudly, writes no row (1,687 → 1,687) |
| `model_fingerprint` populated on new rows | **PASS** — new scratch row carries `14f75e5621a8/8`. Historical: still null on 1,733 of 1,942 |
| `pytest tests/test_restart_gate.py tests/test_data_manifest.py` | **FAIL**, deliberately deferred; `data manifest --verify` reports DRIFTED with 10 discrepancies |
| `research ready` — every dimension its own number | **PASS on reporting, FAIL on the verdict** — `ready: true` with `fundamentals_stale: true` (D-104) |
| egress probe from the deploy host | **BLOCK** — there is no deploy host (D-102) |
| full suite | **1,702 passed · 2 failed · 4 skipped** — the two above |

---

## 6. Checklist delta

V10's 29 items are unchanged except where a measurement moved them:

| item | was | now |
|---|---|---|
| 12 — the client renders correctly | UNVERIFIED | **VERIFIED PASS** — desktop and 375 px, both tabs, scan driven end to end |
| 15 — D-012 egress probe on the deploy host | UNVERIFIED | **BLOCK** — no deploy host exists |
| 20 — IC decay monitored with a threshold | "no number, decoration" | **threshold VERIFIED to exist and be pre-registered; scheduling BLOCK** |
| 22 — `/ready` gates on what the live path uses | CLOSED | **REOPENED** — reports every dimension, gates on none of the stale ones (D-104) |

**Eight operational items to add, for a system that is meant to run unattended:**

| # | item | binary test |
|---|---|---|
| 30 | one deployment path exists in the repo, and it is the one that is deployed | `render.yaml` and `cloud-init.sh` do not both describe a daily job |
| 31 | a scheduler exists on the host that runs it | `crontab -l` / `systemctl list-timers` names the job on the deploy host |
| 32 | the schedule fires in IST after the bhavcopy | the last 5 `observation start` lines in `forward.log` fall in 20:30–21:00 IST |
| 33 | absence is alerted, not just failure | a dead-man's switch fires when no observation is recorded for 2 consecutive sessions |
| 34 | the manifest describes the store every morning | `data manifest --verify` exits 0 after the nightly |
| 35 | the store cannot fill the disk | `du -sm data` < 0.5 × `storage.max_total_mb`, with retention on `jobs.sqlite3` and the cache under its cap |
| 36 | the deployed Python is the tested Python | `python -V` on the deploy host equals the venv the suite ran in |
| 37 | a sandbox override actually sandboxes | `PROSIGNAL_CONFIG=<copy> uvicorn …` writes only inside the copy |

---

## 7. Unverified

| what | why | the one command that would confirm it |
|---|---|---|
| behaviour on an NSE holiday (26 January) | no scheduler to observe; `--skip-if-recorded` is the designed answer and is VERIFIED idempotent, but never observed on a real holiday | `prosignal analyse run --skip-if-recorded` on the next NSE holiday, then `data lineage` |
| reboot survival and missed-window catch-up | no scheduler | reboot the deploy host, then `grep "observation start" forward.log` |
| the 409 on a job of a *different* kind | would require starting a real NSE ingest against the network mid-scan | `POST /analysis/run` then `POST /admin/run-now`; expect 409 |
| ingest vs read-path lock contention (B5) | scoped out after D-102 made it moot — a lock cannot be contended by a job that never runs | hold `store_lock(curated, exclusive=True)` and `curl /today`; expect block or 503, not a partial read |
| D-012 egress from the deploy host | no deploy host | `ssh <host> 'python probe_d012.py'` |
| whether cache eviction is broken or merely lagging | measured the overage (505 MB vs 384 MB cap), not the mechanism | `du -sm data/cache` before and after `prosignal data ingest` |
| NO TRADE / cash rule rendered | the rule is behind `blocked_reason is None` and the live cross-section does not trip it (309 of 386 above the bar against a floor of 18). Forcing it needs a fixture date, and the plan sequences that after Phase 6 | replay 2020-03-23 through the pipeline with the cash rule enabled, then render |

---

## 8. Order of work

Dependencies first.

1. **D-101** (1 h) — the sandbox override. Everything else is safer to test once
   a copy of the store is genuinely a copy.
2. **D-102** (3 h, plus an operator action) — decide the deployment path and
   delete the other. Items 3–5 are shaped by which one is chosen.
3. **D-105** (1 h) — wire the alert, including the dead-man's switch, *before*
   the schedule starts, so the first failure is seen.
4. **D-106** (2 h) — re-manifest and open the epoch inside the nightly.
5. **D-104** (1 h) — `/ready` gates on staleness. Cheap, and it is what the
   health check reads.
6. **D-103** (2 h) — delete the `cross-sectional` branch; key the display cache
   on `code_sha`.
7. **D-113** (1.5 h) — say when the screen is not showing the recorded run.
8. **D-108** (3 h) — retention, before the disk decides the schedule.
9. **D-109** (2 h + a suite run) — rebuild on the deployed Python.
10. **D-107** (2 h), **D-110**/**D-111** (1 h together) — screen honesty, no
   dependencies.

**Total: 18.5 agent-hours**, plus one operator action that no agent can perform
(standing up a host, and re-registering the forward test on it).
