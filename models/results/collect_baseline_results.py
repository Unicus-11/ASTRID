"""
collect_baseline_results.py
==============================
Small aggregation script that runs the six existing baseline model
implementations across both dataset layers, using ONLY the existing
shared infrastructure (experiment_config.py, experiment_runner.py, and
the existing model_implementations/*.py wrappers), and writes a single
combined comparison table to disk.

This is NOT a second evaluation framework: it performs no data loading,
no metric computation, no splitting, and no persistence logic of its
own. It only:

    1. builds one ExperimentConfig per (model, layer) pair, using the
       exact same naming convention as the individual run_*.py scripts
    2. calls the existing run_experiment() for each pair
    3. calls the existing results_to_dataframe() once, on all results
    4. writes that DataFrame to models/results/baseline_results.csv

No metric values are hard-coded anywhere in this file -- every number in
the output CSV comes from an actual run_experiment() call executed when
this script runs.

Run from anywhere; paths are resolved relative to this file, not to the
current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

# models/results/collect_baseline_results.py -> models/ is the parent dir.
# All shared infrastructure (experiment_config.py, experiment_runner.py,
# model_implementations/) lives directly under models/, so it must be on
# sys.path regardless of the caller's current working directory.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from experiment_config import ExperimentConfig, Layer  # noqa: E402
from experiment_runner import (  # noqa: E402
    ExperimentResult,
    run_experiment,
    results_to_dataframe,
)

from model_implementations.random_forest import RandomForestModel  # noqa: E402
from model_implementations.extra_trees import ExtraTreesModel  # noqa: E402
from model_implementations.xgboost_model import XGBoostModel  # noqa: E402
from model_implementations.lightbgm_model import LightGBMModel  # noqa: E402
from model_implementations.catboost_model import CatBoostModel  # noqa: E402
from model_implementations.hist_gradient_boosting import (  # noqa: E402
    HistGradientBoostingModel,
)

RANDOM_STATE = 42

# Output location: models/results/baseline_results.csv
_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "baseline_results.csv"

# One entry per model:
#   - key: matches the naming convention already used by the individual
#     run_*.py scripts (e.g. "random_forest_layer1_baseline")
#   - model_name: passed to run_experiment() as model_name (also used
#     for the saved-model filename, matching the individual runners)
#   - factory: builds a FRESH model instance per (model, layer) pair,
#     since a fitted estimator should not be reused across independent
#     training runs
_MODEL_REGISTRY: List[Tuple[str, str, Callable[[], object]]] = [
    ("random_forest", "random_forest", lambda: RandomForestModel(random_state=RANDOM_STATE)),
    ("extra_trees", "extra_trees", lambda: ExtraTreesModel(random_state=RANDOM_STATE)),
    ("xgboost", "xgboost", lambda: XGBoostModel(random_state=RANDOM_STATE)),
    ("lightbgm", "lightbgm", lambda: LightGBMModel(random_state=RANDOM_STATE)),
    ("catboost", "catboost", lambda: CatBoostModel(random_seed=RANDOM_STATE)),
    (
        "hist_gradient_boosting",
        "hist_gradient_boosting",
        lambda: HistGradientBoostingModel(random_state=RANDOM_STATE),
    ),
]

_LAYERS: List[Layer] = [Layer.LAYER1, Layer.LAYER2_P11]


def main() -> None:
    results: List[ExperimentResult] = []

    for layer in _LAYERS:
        for key, model_name, factory in _MODEL_REGISTRY:
            experiment_name = f"{key}_{layer.value}_baseline"
            config = ExperimentConfig(
                layer=layer,
                random_state=RANDOM_STATE,
                experiment_name=experiment_name,
            )
            model = factory()
            result = run_experiment(
                config=config,
                model=model,
                model_name=model_name,
                save_model_flag=False,
            )
            results.append(result)
            print(f"Completed: {experiment_name}")

    comparison_df = results_to_dataframe(results)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(_OUTPUT_CSV, index=False)

    print()
    print(comparison_df.to_string(index=False))
    print(f"\nWritten to: {_OUTPUT_CSV}")


if __name__ == "__main__":
    main()