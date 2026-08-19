"""Return freed memory to the operating system.

`gc.collect()` frees Python objects, but glibc keeps the underlying arenas and
RSS does not fall. RSS is what a container platform measures and kills on: a
512 MB instance was restarted while live data was a fraction of that, because
each stage's freed frames were still held in the arena.

Measured here: RSS climbed 206 -> 546 MB across six stages although no stage
needs the previous stage's frames.

`malloc_trim(0)` returns free arenas to the OS. It exists only on glibc, so
this is a no-op on macOS and musl; the guard keeps the function safe to call
unconditionally from the pipeline.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import platform
from typing import Optional

from .logging import get_logger

__all__ = ["release_memory", "trim_available"]

log = get_logger(__name__)

_TRIM: Optional[ctypes.CDLL] = None
_CHECKED = False


def _load_trim() -> Optional[ctypes.CDLL]:
    """Resolve glibc's malloc_trim once, or return None where it does not exist."""
    global _TRIM, _CHECKED
    if _CHECKED:
        return _TRIM
    _CHECKED = True
    if platform.system() != "Linux":
        return None
    try:
        path = ctypes.util.find_library("c")
        lib = ctypes.CDLL(path) if path else ctypes.CDLL("libc.so.6")
        if hasattr(lib, "malloc_trim"):
            lib.malloc_trim.argtypes = [ctypes.c_size_t]
            lib.malloc_trim.restype = ctypes.c_int
            _TRIM = lib
    except (OSError, AttributeError):
        _TRIM = None
    return _TRIM


def trim_available() -> bool:
    """True when malloc_trim can actually be called on this platform."""
    return _load_trim() is not None


def release_memory() -> bool:
    """Collect garbage and hand free arenas back to the OS.

    Returns True when the arena was actually trimmed. Cheap enough to call
    between pipeline stages; not cheap enough to call inside a loop.
    """
    gc.collect()
    lib = _load_trim()
    if lib is None:
        return False
    try:
        lib.malloc_trim(0)
        return True
    except OSError:  # pragma: no cover - platform specific
        return False
