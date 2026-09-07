""""
persistence.py
================
Simple, model-agnostic save/load utilities for trained model objects.

No model-specific logic lives here: any picklable object with a
scikit-learn-like interface can be saved and loaded through these
functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> Path:
    """Create `path` (and any missing parents) if it doesn't exist yet.
    Returns path for convenient chaining."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, path: Path) -> Path:
    """Serialize any trained model object to disk with joblib.

    Creates the parent directory if necessary. Overwrites an existing
    file at `path` without warning.
    """
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    """Load a model object previously saved with save_model()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at: {path}")
    return joblib.load(path)