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

__all__ = ["REPO_ROOT", "BACKEND_DIR"]
