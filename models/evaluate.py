"""
evaluate.py
============
Common evaluation interface.


Metrics used for model comparison.

Validation is the primary split for model selection.
Test is retained for final held-out reporting.
OOD is reserved for the final robustness/generalization analysis
and must not influence model selection.

OOD is always evaluated and reported as a clearly separate field, and
this module provides no mechanism for OOD results to feed back into
model selection -- callers that need to pick between models should use
EvaluationReport.selection_metrics(), which excludes OOD entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from data_loader import SplitData
from experiment_config import Split
from metrics import compute_all_metrics
from train import RegressionModel


@dataclass
class SplitEvaluation:
    """Evaluation result for a single split."""

    split: Split
    n_rows: int
    metrics: Dict[str, float]


@dataclass
class EvaluationReport:
    """Full evaluation report for one trained model.

      `validation` is used for model selection.
      `test` is a final held-out evaluation.
      `ood` is a separate robustness/generalization evaluation and must
      not influence model selection.
   """

    validation: SplitEvaluation
    test: SplitEvaluation
    ood: SplitEvaluation

    def selection_metrics(self) -> Dict[str, Dict[str, float]]:
        """Metrics safe to use for model selection/comparison (validation
        and test only). Deliberately excludes OOD."""
        return {
            "validation": self.validation.metrics,
            "test": self.test.metrics,
        }


def evaluate_split(model: RegressionModel, split_data: SplitData) -> SplitEvaluation:
    """Evaluate a trained model on one already-loaded split."""
    predictions = model.predict(split_data.X)
    return SplitEvaluation(
        split=split_data.split,
        n_rows=len(split_data),
        metrics=compute_all_metrics(split_data.y, predictions),
    )


def evaluate_model(
    model: RegressionModel,
    validation_split: SplitData,
    test_split: SplitData,
    ood_split: SplitData,
) -> EvaluationReport:
    """Evaluate a trained model consistently across validation, test, and
    OOD splits. Callers must pass SplitData objects loaded (via
    data_loader.load_split() / DatasetLoader) for the SAME layer the
    model was trained on.
    """
    if validation_split.split != Split.VALIDATION:
        raise ValueError("validation_split must be the VALIDATION split")
    if test_split.split != Split.TEST:
        raise ValueError("test_split must be the TEST split")
    if ood_split.split != Split.OOD:
        raise ValueError("ood_split must be the OOD split")

    return EvaluationReport(
        validation=evaluate_split(model, validation_split),
        test=evaluate_split(model, test_split),
        ood=evaluate_split(model, ood_split),
    )