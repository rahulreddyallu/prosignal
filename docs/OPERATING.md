# How to operate ProSignal

Operational trust audit — 27 August 2026
engine 0.1.0 · config `baseline-v1@5ffc70e01ee1ba1d` (changed by the fixes) · audited at HEAD `f65801a`

> **Verdict: leave it running, but do not yet believe the evidence.**
> The machine side is sound. The pre-registered forward test — the apparatus that turns a log into
> an experiment — is not currently in a valid state.

> ## ⚠ Do this first, once
>
> The audit's defects have been fixed, and fixing them changed `config/parameters.yaml`
> (three constants that were hardcoded are now real parameters). **The configuration hash moved to
> `baseline-v1@5ffc70e01ee1ba1d`, so the forward test must be re-registered — after deploying, not
> before.**
>
> 1. `git pull` on the instance, then `sudo systemctl daemon-reload && sudo systemctl restart prosignal`
> 2. Open **Settings → Forward test**. It will say "Not registered" or show a drift warning.
> 3. Press **Open**.
> 4. Do not edit `config/parameters.yaml` again.
>
> From that moment the clock is running and every night counts. Before it, nothing does.

---

## 1. What ProSignal does

Every weekday after the close it downloads NSE market data, rebuilds a point-in-time universe of
liquid names, fits a cross-sectional ranking model to its own stored history, ranks the universe,
applies eligibility gates, a false-signal defence, a rank-hysteresis admission band, stops and
targets, and portfolio caps — then writes a permanent record of what it said and why.

It places no orders. There is no order-routing code in the project.

---

## 2. The one-minute daily check

1. Open the app.
2. The subtitle should read **"Ranked from the close of &lt;date&gt;"** — that date must be the last
   trading session. If it says **"New market data — Last ranked …"**, last night did not produce a
   run.
3. Shortlist present, or an explicit "No recommendation was produced" with a reason.
4. Glance at open-position alerts (they render on every path, including empty days).

If those agree: **close the app and press nothing.** Do not read the performance number daily.

Ground truth from a terminal:

```bash
tail -20 data/ledger/forward.log
```

A good night ends with `observation recorded`.

---

## 3. Weekly — 10 to 20 minutes

- **Count the week's runs in History.** One per trading session. A missing day is the single most
  important thing this review catches, because nothing else will tell you.
- **Settings → Forward test.** Must read "N of 375 sessions since &lt;date&gt;". If it shows
  **"⚠ The configuration has changed since it was registered"**, the window is void.
- **Read the funnel on one day.** Currently ~750 considered → ~450 eligible → 8 admitted. A big move
  in the middle number is a data or universe event, not a market one.
- **Settings → Model.** Should read "3 of 5 themes priced, fitted &lt;latest session&gt;".
- **`tail -50 data/ledger/forward.log`** — `ingest FAILED` / `analysis FAILED` are the only failure
  notices that exist.

---

## 4. Monthly — observe, do not intervene

Observe: closed-trade count, exit mix, holding distribution, calibration, config hash.
Do not act on: a negative month, a sub-50% hit rate, individual losing names, or the headline return.

**Quarterly:** nothing. The pre-registration forbids interim reading.

---

## 5. What cron does, and does not

`/etc/cron.d/prosignal` → `30 20 * * 1-5` → `scripts/forward_run.sh`:

```
1. cron.paused?            → log the decline, exit 0
2. cli data ingest         → FAIL: log "ingest FAILED", exit 1, nothing recorded
3. cli analyse run         → FAIL: log "analysis FAILED", exit 1, nothing recorded
4. "observation recorded"
5. cli research forward    → progress into the log
6. warm /performance, /ready over loopback
```

**Does not:** notify you of anything; back anything up; skip NSE holidays; re-register the forward
test; schedule outcome resolution (that happens when History opens); notice config drift.

**Holidays.** One-day holiday: the ingest fetches nothing, the staleness gate counts one weekday and
passes, and the analysis re-runs the previous session — a duplicate ledger row for a date that
already has one. Two-day holiday: the gate counts two, Stage 1 halts, and the log says
`analysis FAILED`, which looks exactly like a real failure. Check the NSE calendar before
investigating a failure after a long weekend.

---

## 6. Buttons

| Control | Endpoint | What it actually does | Safe? |
|---|---|---|---|
| Scan Market | `POST /analysis/run` | Runs the pipeline. Refuses with 409 if another kind of job holds the slot; the button is disabled while one does. *(fixed)* | safe |
| Cancel | `POST /admin/…/cancel` | Marks the row. Cannot stop the worker thread — Python has no safe way — but its result is now discarded and a new job waits for it to finish. *(fixed)* | safe |
| Refresh | `POST /admin/ingest` | Pulls the newest session, advances backfill one 90-session chunk. | safe |
| Refit | `POST /admin/refit` | Archives and retires the coefficient cache; next scan refits. | not during a window |
| Resolve | `POST /admin/resolve-outcomes` | Scores signals whose window elapsed. Idempotent. | safe |
| Open / Re-open | `POST /admin/forward/register` | **Re-open discards the window and every observation in it.** | destructive |
| Clear | `DELETE /history` | Watermark only. Hides, never deletes. Reversible. | safe |
| **Rebuild** | `POST /admin/reset/market-data` | Wipes curated/snapshots/cache/raw, now preserving `trial_registry.jsonl` and the model version archive. *(fixed)* Still hours of re-downloading, and it resets the training depth, which changes the model. | avoid |
| Daily signals | `POST /operations/pause` · `resume` | Writes `cron.paused` and closes/opens a measurement period. Cron still wakes and declines; the decline is recorded. | safe |
| Build data store | `POST /admin/bootstrap` | One 90-session chunk toward 2,200. Resumable. | safe |

### The sequence that used to lie to you — fixed

Press **Refresh**, then **Scan Market** while it ran, and the interface reported
*"Scan complete. 0 qualifying, 0 monitored."* for a scan that never happened. The scan button is now
disabled while a refresh runs, the endpoint answers 409, and the refresh message no longer promises
the button already works. If you want both in one press, `POST /admin/run-now` now genuinely does
ingest-then-rank.

---

## 7. Operator don'ts

**Never, once a window is open**

- Edit `config/parameters.yaml` — any change voids the window.
- Press **Re-open** on the forward test.
- Press **Rebuild** — it changes the training depth (the model refits from the store, so a shorter
  store *is* a different model) and deletes the trial registry.
- Run `prosignal research cpcv|estimator|spread|metalabel|volscale` — each records a trial and
  permanently raises the Deflated-Sharpe bar.
- Trade the shortlist with real capital — named in the pre-registration as an invalidation condition.
- `analyse run --date …` — a run against a past close is a re-score of data the model has seen.

**Avoid, but recoverable:** scanning during a refresh; cancelling a scan; repeated reruns in one
evening; reading the performance number daily; refitting mid-window.

**Feels destructive, is not:** **Clear** (watermark, reversible) and **Pause** (the decline is
written down on purpose).

---

## 8. What was fixed

Every defect below is closed, with a test that fails without the fix. Numbers in brackets match the
findings list that follows.

| Fix | What changed |
|---|---|
| **[1] Drift is detectable before the first observation** | `progress()` now takes the live config version and flags a mismatch with the registration. `research forward` exits non-zero, and the nightly script now *alerts* on that rather than logging it. Verified against the real registration: the CLI reports `The forward test is INVALID`. |
| **[2] The coverage criterion is evaluated** | `MIN_SESSION_COVERAGE` moved to `validation/forward.py` and is applied against the sessions the market actually printed, once the window is at least 20 sessions long. Duplicate runs on one date cannot buy back a missing day. |
| **[3] A job of the wrong kind is refused** | `JobBusy` is raised instead of handing back a foreign job; the API answers 409 with a readable message; the Scan button is disabled while a refresh runs; the refresh handler no longer claims the button works. |
| **[3b] Cancel actually cancels** | The final state write is `WHERE state = RUNNING`, so a cancelled job cannot come back COMPLETED, and a new job is refused while the orphaned worker is still alive. |
| **[4] Failures reach something other than the log** | `PROSIGNAL_ALERT_CMD` in `forward_run.sh` — inert unless set, ignores its own failures. Fires on failed ingest, failed analysis, and an invalid forward test. |
| **[5] `/ready` reports freshness** | Same arithmetic and tolerance Stage 1 halts on; returns `sessions_behind`, `staleness_limit`, `data_stale`, and 503 with a remedy. |
| **[6] `latest_session` is always returned** | So the interface's staleness check can never silently disable itself. |
| **[6b] The UI splits "too short" from "too old"** | `checkReady` and `pollReady` now agree; a stale store offers **Refresh market data**, not a rebuild. |
| **[7] Rebuild preserves research evidence** | `trial_registry.jsonl` and `crosssec_model_versions/` are carried across the wipe. `erase_everything` still erases everything. |
| **[8] The holiday duplicate is gone** | `analyse run --skip-if-recorded`, used by cron only. A person pressing Scan still gets the rerun. |
| **[9] The three hidden constants are real parameters** | `api.job_timeout_seconds` (now **1800s**, was an invisible 900), `api.bootstrap_chunk_sessions`, `storage.validated_training_sessions`. All three appear in `config show`. |
| **[10] `runtime.timezone` is read** | New `core/clock.py`. "Today" for the staleness gate comes from the configured zone, not the host's. |
| **[11] Auth fails closed behind a proxy** | `PROSIGNAL_PUBLIC=1`, written beside the token by cloud-init. Removing the token now stops the service instead of opening `/admin/reset/everything`. |
| **[12] `/admin/run-now` does what it says** | Ingest *then* analyse, in one job. |
| **[13] Model staleness and trade concurrency are shown** | `/admin/model` returns `stale`; `/performance` returns `concurrency` and the list of configurations it pooled. Measured on the current record: **60 of 100 trades overlapped another**. |
| **[14] `rundetail` has tests** | 11 of them, including the run-ordering bug that served the oldest of three. |
| **[15] The suite no longer goes red on a stale store** | The nine tests that drive the pipeline use a `runnable_cfg` fixture that skips with an actionable message. The engine's refusal is still asserted directly. |

Suite after the fixes: **1,362 passed, 10 skipped, 0 failed** (was 1,311 passed / 9 failed).

Still **open, and deliberately** — these are operator actions, not code:

- **Re-register the forward test on the instance.** Nothing can do this for you; doing it automatically is exactly what a pre-registration must not allow.
- **Back up `data/ledger/`.** 19 MB, monthly, off the instance. There is still no backup code and adding one that writes somewhere you have not chosen would be worse.
- **The History headline still pools configurations by default.** The scoping switch exists; defaulting it on emptied a history of 136 closed trades the first time and the isolation was real while the screen was wrong. The payload now tells you which configurations it pooled and how many trades overlapped.

---

## 8b. Findings, ranked (as found)

### P0 — research integrity

1. **The window is registered against a config the engine no longer runs, and its status reads
   healthy.** Verified: `config_matches: False`, `broken: []`, summary *"Registered 2026-08-27; no
   forward session yet"*. `progress()` compares config versions only on ledger rows *inside* the
   window; before the first observation there are none. The Settings row does catch it; the CLI and
   the nightly log do not. — `validation/forward.py:276`
2. **One of the four pre-registered invalidation criteria is never evaluated.**
   `MIN_SESSION_COVERAGE = 0.60` (`operations.py:43`) is referenced nowhere. A job that silently
   stops for weeks still reports steady progress.

### P1 — major operational

3. **"Scan complete. 0 qualifying" for a scan that never ran** (verified by execution).
   `jobs.py:223`, `api.py:250`, `index.html:1176`.
4. **`/ready` says ready on a store the pipeline would refuse to analyse.** Verified:
   `ready: true, latest_session: 2026-08-25` on 2026-08-27, while `_sessions_behind = 2 > limit 1`
   halts Stage 1. The full suite demonstrates it — 9 of 1,320 tests fail today, all on the identical
   halt.
5. **The freshness indicator disables itself on a partly-built store.** `isCurrent()` returns `true`
   when `latestSession` is null, and `/ready` omits `latest_session` below full validated depth.
6. **"Rebuild" destroys research evidence its own label promises to keep** —
   `trial_registry.jsonl` lives in `curated/`.
7. **No alerting of any kind.** A failed night writes one log line. Silence and success are the same
   signal.
8. **The evidence exists in one un-backed-up place.** `data/ledger/` is gitignored; there is no
   backup code. 19 MB. Copy it monthly.
9. **The History headline pools six configurations** and sums per-trade returns on a book whose
   concurrency is computed (`overlaps()`) and never shown.

### P2

- `cancel` does not stop the worker; a second job can start alongside it.
- A run slower than 900 s is declared FAILED while still running, freeing the slot.
- `job_timeout_seconds` (900), `bootstrap_chunk_sessions` (90) and `validated_training_sessions`
  (2200) are `getattr` defaults present in neither the YAML nor the schema.
- `runtime.timezone` is never read; "today" is the OS timezone, set once by cloud-init.
- `assert_safe_to_serve` is passed `api.host` = `127.0.0.1`, so the fail-closed check never fires on
  the instance where Caddy is the public face. Auth rests entirely on `PROSIGNAL_AUTH_TOKEN`.
- `POST /admin/run-now` claims to "refresh the data, then rank" and only ingests. No control reaches
  it; the FastAPI docs page does.
- `rundetail.py` — the module the entire Today screen depends on — has zero tests.
- The ingest's exit code is not a freshness signal (its staleness is measured circularly).

### P3 / P4

21 inert parameters (including `fail_run_if_unwritable` and `stale_data_action`, which read as safety
controls); a dead `mdl.stale` indicator; `GET /measurement` writing on read; `forward.log` never
rotated; the README nine commits behind the behaviour it describes.

---

## 9. Top ten fixes

1. Re-register the forward test on the instance, then freeze the config.
2. Preserve `trial_registry.jsonl` and the model version archive in `reset_market_data`.
3. Make `/analysis/run` refuse a job of the wrong kind.
4. Add any failure notification to `forward_run.sh` (three lines).
5. Evaluate the 60% session-coverage criterion in `progress()`.
6. Compare the registered config against the live one in `broken`.
7. Add a freshness check to `/ready`.
8. Always return `latest_session` from `/ready`.
9. Skip the analysis when the ingest fetched zero sessions (ends the holiday duplicate).
10. Promote the three hidden constants into the schema.

---

## 10. Measuring whether it works

**Not** the number on the History page. The measurement is the pre-registered test: regress the paper
portfolio's monthly excess return on six long-short factors over eighteen forward months; pass only
if the intercept is positive at `|t| ≥ 2`.

Derived from this system's own numbers — per-trade excess `sd = 4.87%`, overlap `VIF = 3.49`
(97 trades → 32.6 effective):

| True edge per trade | Independent trades for \|t\|=2 | Actual trades at this overlap | Roughly |
|---|---:|---:|---|
| 2.00% | 24 | 83 | ~2 months |
| 1.50% | 42 | 147 | ~3 months |
| 1.00% | 95 | 331 | ~6 months |
| 0.75% | 169 | 589 | ~11 months |
| 0.50% | 379 | 1,325 | ~2 years |

Current record (local ledger, mixed configs — **not** evidence): 97 closed, avg −0.43%, excess
−0.66%, win rate 44.3%, beat rate 41.2%, corrected `t = −0.77`. Exit mix `book_exit 86 / stop 10 /
stop_gap 1`; median hold **3 sessions**.

### Do not panic

- Median hold 3 sessions — the book *is* the exit rule; stops and targets are the early exits.
- 89% book exits — by design.
- Two of five themes priced at zero — the significance floor working.
- A day with no shortlist — allowed, and it names the gate.
- A negative cumulative line at n=97 with t=−0.77 — indistinguishable from zero either way.

### Stop the experiment if

The Settings row shows the drift warning · `hash_intact` is false · more than one `config_version` or
`model_fingerprint` appears inside the window · recorded runs fall below 60% of sessions (**you must
check this; the code does not**) · anything is retuned on in-window data · the shortlist is traded
with real money · a leakage defect is found in a live factor · the ledger is partially lost.

---

## 11. Incident playbook

**Cron did not run** — `tail -30 data/ledger/forward.log` for tonight's start line;
`systemctl status cron`; `cat /etc/cron.d/prosignal`; check Settings (Daily signals may be off);
confirm `scripts/forward_run.sh` exists and is executable. Recover: Refresh, wait, Scan.

**Ran but no signals** — normal outcome; check the regime strip and the funnel; if the log says
`analysis FAILED`, the reasons name the feed and its age.

**Provider failed** — the script stops before the analysis and records nothing. Retry with Refresh.
Two consecutive misses put the store two sessions behind and Stage 1 halts until it catches up. Do
not lower the staleness tolerance.

**UI says success but records are missing** — almost certainly the scan-during-refresh path. No
History row means no run.

**Signals look very different** — check Model (fitted date, priced themes), the funnel, the regime
transition flag, and whether the config hash moved.

**After a deployment** — `git pull`; `sudo systemctl daemon-reload && sudo systemctl restart
prosignal`; confirm `/health` version; confirm `forward_run.sh` is present; **check whether
`config/parameters.yaml` changed** — that is the most common way the window breaks.

**Duplicate or partial run** — duplicates are safe (dedupe by date everywhere it matters); a partial
run leaves no row, so a rerun is safe; a stale RUNNING job is reaped after 900 s.

**Corruption** — the store rebuilds from NSE; the ledger rebuilds from nothing. A truncated ledger
line is skipped. A corrupt model cache is refused loudly with the previous version in
`crosssec_model_versions/`.

---

## 12. Method and limits

- **Verified dynamically** — executed in a sandbox (`config/` and `data/ledger/` copied; `curated`,
  `snapshots`, `cache` symlinked read-only; real ledger mtimes confirmed unchanged afterwards).
- **Verified by code** — read from the implementation, cited by file and line.
- **Inferred** — follows from code paths, not executed (the holiday duplicate; concurrent runs after
  a cancel; the 900 s reap racing a slow run).
- **Unknown** — everything about the deployed instance.

> **This audit ran against the local development checkout, not EC2.** The local ledger holds 1,935
> rows across 251 dates with 676 on a single day — development re-runs, not production history. The
> last direct evidence showed the instance reporting *"No forward test is registered"*. Whether that
> is still true is **UNKNOWN**. Check **Settings → Forward test** first.

Test suite today: **1,311 passed · 9 failed · 1 skipped**. All nine share one cause —
`MarketWideHalt: required feed 'equity_ohlcv' is STALE: last data 2026-08-25, 2 sessions old,
limit 1`. The suite is coupled to live data freshness and goes red on its own whenever the store
falls behind.

---

## 13. The decision

**Leave it alone — after one intervention.** Open Settings, check the forward-test row, and press
**Open** if it says "Not registered" or shows the drift warning. Then stop touching the
configuration, take a copy of `data/ledger/`, and give it sixty seconds a day and ten minutes a week.

The work left is not in the model. It is in making sure that eighteen months from now, the thing it
has been recording is admissible.
