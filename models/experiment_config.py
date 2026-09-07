"""
experiment_config.py
=====================
Central, shared definitions for every model experiment: which layer,
which splits exist, where the target column lives, and the common shape
of an experiment configuration.

This module defines NO model-specific hyperparameters. Those belong to
each model implementation, added later, which should accept its own
hyperparameters separately alongside an ExperimentConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Layer(str, Enum):
    """The two assembled dataset layers available for modeling."""

    LAYER1 = "layer1"
    LAYER2_P11 = "layer2_p11"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

class Split(str, Enum):
    """The four data splits produced by the assembly pipeline.

    TRAIN and VALIDATION are for fitting and model selection.
    TEST is the held-out set for a final, single reported result.
    OOD is reserved strictly for out-of-distribution robustness
    evaluation -- it must never be used to select or tune a model.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

# The single regression target for this project. Defined once, here, so
# every loader/trainer/evaluator agrees on it.
TARGET_COLUMN: str = "true_queue_length_m"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# models/ is assumed to be a sibling of dataset/, per the project layout:
#   <repo_root>/dataset/assembled/{layer1,layer2_p11}/...
#   <repo_root>/models/...
_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_ROOT: Path = _REPO_ROOT / "dataset" / "assembled"
DEFAULT_OUTPUT_ROOT: Path = _REPO_ROOT / "models" / "artifacts"


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Common configuration shared by every model experiment.

    Intentionally holds NO model-specific hyperparameters (no
    n_estimators, no learning_rate, no layer sizes, etc.). Those are the
    responsibility of each model implementation, which should accept its
    own hyperparameters separately and receive an ExperimentConfig for
    everything that must stay identical across models.
    """

    layer: Layer
    target_column: str = TARGET_COLUMN
    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    output_root: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT)

    # Identifies this experiment/model run for artifact naming, e.g.
    # "random_forest_layer1_baseline". Left blank by default -- callers
    # should set something meaningful before training.
    experiment_name: str = ""

    # Shared random seed convention for anything stochastic in a future
    # model's training procedure (this is NOT itself a hyperparameter
    # value -- just the common seed every model should use).
    random_state: int = 42

    # Free-text notes about the run, for humans, not consumed by code.
    notes: Optional[str] = None

    def output_dir(self) -> Path:
        """Where this experiment's artifacts (trained model, metrics,
        etc.) should be written."""
        name = self.experiment_name or "unnamed_experiment"
        return Path(self.output_root) / self.layer.value / name