"""Analysis job manager -- SQLite-backed, single-process, single-flight.

The requirement is narrow: one user, one button, one expensive market-wide
analysis at a time. SQLite plus a thread meets it exactly. Celery or Redis would
add an operational surface (a broker to run, monitor, and recover) to solve a
concurrency problem this system does not have.

Two properties matter and both are enforced here rather than by convention:

**Single flight.** Clicking the button twice must not launch two full-universe
analyses. `start()` returns the ALREADY-RUNNING job instead of queueing a second
one, so a double click is idempotent rather than expensive.

**No permanently stuck jobs.** A process that dies mid-run leaves a row marked
RUNNING that nothing will ever finish. On startup, and before each new job, any
RUNNING row older than the timeout is reaped and marked FAILED with a reason.
Without that, one crash blocks the button forever.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .core.logging import get_logger
from .core.memory import release_memory

__all__ = ["JobManager", "JobState", "Job"]

log = get_logger(__name__)


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)


@dataclass
class Job:
    id: str
    state: JobState
    created_at: dt.datetime
    kind: str = "analysis"
    started_at: Optional[dt.datetime] = None
    finished_at: Optional[dt.datetime] = None
    progress_step: int = 0
    progress_total: int = 9
    progress_label: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": {
                "step": self.progress_step,
                "total": self.progress_total,
                "label": self.progress_label,
                "pct": round(100.0 * self.progress_step / max(self.progress_total, 1)),
            },
            "duration_seconds": (
                round((self.finished_at - self.started_at).total_seconds(), 1)
                if self.started_at and self.finished_at
                else None
            ),
            "result": self.result,
            "error": self.error,
            "error_detail": self.error_detail,
        }


_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL DEFAULT 'analysis',
    state          TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    progress_step  INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 9,
    progress_label TEXT NOT NULL DEFAULT 'queued',
    result_json    TEXT,
    error          TEXT,
    error_detail   TEXT
);
"""

#: Indexes are created AFTER column migrations. An index on a column that a
#: migration is about to add will fail against an older database, which is how
#: the first version of this broke: the schema script referenced `kind` before
#: the ALTER that creates it had run.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_jobs_state   ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_kind    ON jobs(kind);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


#: Additive-only migrations, applied when a column is absent. Append here;
#: never rewrite history, and never drop a column -- an older process may still
#: be reading the same file during a rolling restart.
_MIGRATIONS = [
    ("kind", "kind TEXT NOT NULL DEFAULT 'analysis'"),
]


class JobManager:
    """Runs one analysis at a time and records its lifecycle."""

    def __init__(
        self,
        db_path: Path,
        runner: Callable[[Callable[[int, str], None]], Dict[str, Any]],
        timeout_seconds: float = 900.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._init_db()
        self.reap_stale()

    # -- storage ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the schema, then apply additive migrations.

        `CREATE TABLE IF NOT EXISTS` silently does nothing when the table
        already exists with an older shape, so a database created by a previous
        version keeps its old columns and every query against a new one fails.
        That matters most in deployment, where a persistent disk carries the old
        file across releases -- exactly the case a fresh local run would never
        reproduce.
        """
        with self._connect() as conn:
            conn.executescript(_TABLE)
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
            for column, ddl in _MIGRATIONS:
                if column not in existing:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
                    log.info("jobs schema migrated", extra={"added": column})
            conn.executescript(_INDEXES)

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            kind=(row["kind"] if "kind" in row.keys() else "analysis"),
            state=JobState(row["state"]),
            created_at=dt.datetime.fromisoformat(row["created_at"]),
            started_at=dt.datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=dt.datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            progress_step=row["progress_step"],
            progress_total=row["progress_total"],
            progress_label=row["progress_label"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            error_detail=row["error_detail"],
        )

    # -- lifecycle ----------------------------------------------------------
    def reap_stale(self) -> int:
        """Fail any RUNNING job older than the timeout.

        Called on construction and before every new job. A crashed process
        otherwise leaves a RUNNING row that blocks the button permanently.
        """
        cutoff = (dt.datetime.now() - dt.timedelta(seconds=self.timeout_seconds)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id FROM jobs WHERE state IN (?, ?) AND created_at < ?",
                (JobState.RUNNING.value, JobState.QUEUED.value, cutoff),
            )
            ids = [r["id"] for r in cur.fetchall()]
            if ids:
                conn.execute(
                    f"UPDATE jobs SET state=?, finished_at=?, error=? "
                    f"WHERE id IN ({','.join('?' * len(ids))})",
                    [
                        JobState.FAILED.value,
                        dt.datetime.now().isoformat(),
                        "job exceeded the timeout or the process restarted while it "
                        "was running; marked failed so a new analysis can start",
                        *ids,
                    ],
                )
        if ids:
            log.warning("reaped stale jobs", extra={"count": len(ids)})
        return len(ids)

    def active_job(self) -> Optional[Job]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE state IN (?, ?) ORDER BY created_at DESC LIMIT 1",
                (JobState.QUEUED.value, JobState.RUNNING.value),
            )
            row = cur.fetchone()
        return self._row_to_job(row) if row else None

    def start(self, kind: str = "analysis", runner=None) -> Job:
        """Start a job, or return whichever one is already running.

        Single-flight is GLOBAL rather than per-kind on purpose: both job types
        are expensive and both touch the same data store, so running an
        analysis while the store is being rebuilt underneath it would produce a
        result from a half-written store.
        """
        with self._lock:
            self.reap_stale()
            existing = self.active_job()
            if existing is not None:
                log.info("analysis already running", extra={"job_id": existing.id})
                return existing

            job_id = uuid.uuid4().hex[:12]
            now = dt.datetime.now()
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO jobs (id, kind, state, created_at, progress_label) "
                    "VALUES (?,?,?,?,?)",
                    (job_id, kind, JobState.QUEUED.value, now.isoformat(), "queued"),
                )
            self._thread = threading.Thread(
                target=self._execute, args=(job_id, runner or self.runner), daemon=True
            )
            self._thread.start()
            log.info("analysis job started", extra={"job_id": job_id})
            return self.get(job_id)  # type: ignore[return-value]

    def _progress(self, job_id: str, step: int, label: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress_step=?, progress_label=? WHERE id=?",
                (step + 1, label, job_id),
            )

    def _execute(self, job_id: str, runner=None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET state=?, started_at=? WHERE id=?",
                (JobState.RUNNING.value, dt.datetime.now().isoformat(), job_id),
            )
        try:
            runner = runner or self.runner
            result = runner(lambda step, label: self._progress(job_id, step, label))
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET state=?, finished_at=?, result_json=?, "
                    "progress_step=progress_total, progress_label=? WHERE id=?",
                    (
                        JobState.COMPLETED.value,
                        dt.datetime.now().isoformat(),
                        json.dumps(result, default=str),
                        "complete",
                        job_id,
                    ),
                )
            log.info("analysis job completed", extra={"job_id": job_id})
            # An idle web process must not sit on the run's peak.
            release_memory()
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            detail = traceback.format_exc()
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET state=?, finished_at=?, error=?, error_detail=? WHERE id=?",
                    (
                        JobState.FAILED.value,
                        dt.datetime.now().isoformat(),
                        f"{type(exc).__name__}: {exc}",
                        detail,
                        job_id,
                    ),
                )
            log.error("analysis job failed", extra={"job_id": job_id, "error": str(exc)})
            release_memory()

    # -- queries ------------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
            row = cur.fetchone()
        return self._row_to_job(row) if row else None

    def recent(self, limit: int = 20) -> List[Job]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [self._row_to_job(r) for r in cur.fetchall()]

    def last_completed(self) -> Optional[Job]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE state=? ORDER BY finished_at DESC LIMIT 1",
                (JobState.COMPLETED.value,),
            )
            row = cur.fetchone()
        return self._row_to_job(row) if row else None

    def cancel(self, job_id: str) -> bool:
        """Mark a job cancelled.

        The worker thread is not forcibly killed -- Python cannot do that
        safely. The row is marked so the UI stops polling and a new job may
        start; the orphaned thread finishes and its result is discarded.
        """
        job = self.get(job_id)
        if job is None or job.state.is_terminal:
            return False
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET state=?, finished_at=?, error=? WHERE id=?",
                (
                    JobState.CANCELLED.value,
                    dt.datetime.now().isoformat(),
                    "cancelled by user",
                    job_id,
                ),
            )
        return True
