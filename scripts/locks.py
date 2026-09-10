"""One run at a time on a catalog.

The lock belongs to the catalog rather than to a command: sync and delete touch
the same rows and the same index, so they have to exclude each other, not only
themselves.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Not POSIX: no lock is available, two runs would overlap.
    fcntl = None  # type: ignore[assignment]


@contextmanager
def single_run(db_path: Path, *, enabled: bool = True) -> Iterator[None]:
    """Refuse to start while another run is already working on this catalog.

    A dry run reads only, so it neither takes the lock nor creates the lock file.
    """
    if fcntl is None or not enabled:
        yield
        return

    lock_path = db_path.with_name(f"{db_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            message = f"Another run is already using this catalog: {lock_path}"
            raise SystemExit(message) from None
        yield
