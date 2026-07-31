"""Make the ``app`` package importable from the scripts/ directory.

Every script in this folder starts with ``import _bootstrap``. This avoids
requiring the reviewer to set PYTHONPATH or install the backend as a package
before running anything - `python scripts/run_etl.py` simply works from the
repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

for path in (BACKEND_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _force_utf8_output() -> None:
    """Let these scripts print £ on a Windows console.

    The dataset is from a UK retailer, so every revenue figure carries a pound
    sign. Windows defaults stdout to cp1252, which renders it as ``?`` - the
    scoring summary printed "Revenue at risk: ?181,604.50". Cosmetic, but it is
    in the first output the seed script shows a reviewer.

    Doing it here covers every script at once, since they all import this
    module. ``errors="replace"`` keeps the old degraded-but-alive behaviour if
    the stream cannot be reconfigured (already detached, or captured by a test
    harness) rather than failing a long ETL run over a currency symbol.

    Caveat, so nobody re-debugs this: piping a script through Windows
    PowerShell 5.1 (``... | tail``) still shows ``??``. Python emits correct
    UTF-8 there; PowerShell re-decodes the pipe with the console codepage and
    turns the two bytes of £ into two replacement chars. Fixing that means
    ``[Console]::OutputEncoding = [Text.Encoding]::UTF8`` in the *shell*, which
    is the user's environment and not this repo's business. Direct-to-terminal
    output - how the seed script is actually run - is correct.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - not a TextIOWrapper
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - stream already closed
            pass


_force_utf8_output()

__all__ = ["REPO_ROOT", "BACKEND_DIR"]
