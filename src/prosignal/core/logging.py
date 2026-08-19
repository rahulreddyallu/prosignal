"""Structured logging.

Two consumers, two formats:

* Console -- short human-readable lines for the CLI.
* File -- one JSON object per line under ``logs/``, so a run can be replayed
  later. The ledger records decisions; the log records how the engine reached
  them, including provider fallbacks.

A ``run_id`` is threaded through a contextvar so concurrent API requests do not
interleave unattributably.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["setup_logging", "get_logger", "bind_run_id", "current_run_id"]

_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("prosignal_run_id", default="-")

_CONFIGURED = False

_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


def bind_run_id(run_id: str) -> None:
    """Attach a run id to every subsequent log record on this context."""
    _RUN_ID.set(run_id)


def current_run_id() -> str:
    return _RUN_ID.get()


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and key != "run_id" and not key.startswith("_"):
                try:
                    json.dumps(val)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = repr(val)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;196m",
        "CRITICAL": "\033[48;5;196;38;5;231m",
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        name = record.name.replace("prosignal.", "")
        if self.use_color:
            color = self._COLORS.get(level, "")
            level_s = f"{color}{level:<8}{self._RESET}"
        else:
            level_s = f"{level:<8}"
        line = f"{ts} {level_s} {name:<28} {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    to_console: bool = True,
    to_file: bool = True,
    backup_count: int = 14,
    force: bool = False,
) -> None:
    """Configure the ``prosignal`` logger tree. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger("prosignal")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    run_filter = _RunIdFilter()

    if to_console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_ConsoleFormatter(use_color=sys.stderr.isatty()))
        stream.addFilter(run_filter)
        root.addHandler(stream)

    if to_file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_dir / "prosignal.jsonl"),
            when="midnight",
            backupCount=backup_count,
            encoding="utf-8",
            utc=False,
        )
        file_handler.setFormatter(_JsonFormatter())
        file_handler.addFilter(run_filter)
        root.addHandler(file_handler)

    # Third-party chatter is not our signal.
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. ``get_logger(__name__)``."""
    if not name.startswith("prosignal"):
        name = f"prosignal.{name}"
    return logging.getLogger(name)
