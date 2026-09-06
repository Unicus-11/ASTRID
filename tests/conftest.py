"""
conftest.py
============
Ensures sensors/, dataset/, and models/ are importable as plain modules
when pytest is run from the repo root -- these directories aren't
packages (no __init__.py, no pip install -e), so pytest's own rootdir
insertion doesn't cover them.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for sub in ("sensors", "dataset", "models", "models/results"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)