"""
tune_hist_gradient_boosting.py
=================================
Hyperparameter tuning for HistGradientBoosting on Layer 2 (layer2_p11),
validation-only.

This script:
    * loads TRAIN and VALIDATION via the existing DatasetLoader --
      no new split logic, no dataset/feature modification
    * constructs sklearn.ensemble.HistGradientBoostingRegressor directly
      for each candidate (bypassing model_implementations/
      hist_gradient_boosting.py, which does not expose early_stopping) --
      that wrapper file is left completely unmodified
    * fits every candidate on TRAIN only, scores on VALIDATION only
    * NEVER loads or evaluates TEST or OOD
    * reuses metrics.compute_all_metrics() for MAE/RMSE/R2 -- no metric
      math is reimplemented
    * does NOT call experiment_runner.run_experiment() (which always
      touches test/ood and persists a model artifact per call -- not
      appropriate for a 40-candidate validation-only search)
    * saves all 40 trial results to
      models/results/hist_gradient_boosting_tuning_results.csv
    * selects a winner using a documented tie-break rule, using
      VALIDATION metrics only
    * does NOT train or save a final test/ood model -- that is a
      separate, later step

Baseline-control trial
-----------------------
The existing baseline HistGradientBoosting artifact was trained via
model_implementations/hist_gradient_boosting.py, which does NOT
explicitly set early_stopping (sklearn's default, "auto", may enable
internal early stopping for this dataset size). Every tuning candidate
here uses early_stopping=False. To avoid conflating "the effect of
changing early_stopping" with "the effect of tuning the six
hyperparameters," Trial 1 is a baseline-control: the exact six baseline
hyperparameter values, evaluated under the same early_stopping=False
setting as every other candidate. Trials 2-40 are randomly sampled.
This still totals exactly 40 trials.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/results/tune_hist_gradient_boosting.py -> models/ is the parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader  # noqa: E402
from experiment_config import Layer, Split  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

LAYER = Layer.LAYER2_P11
MODEL_RANDOM_STATE = 42          # fixed for every candidate model
SEARCH_SAMPLING_SEED = 42        # controls which 39 random configs are drawn
N_CANDIDATES = 40                # 1 baseline-control + 39 randomly sampled
TIE_THRESHOLD_FRACTION = 0.01    # 1% of best validation MAE
FLOAT_TOLERANCE = 1e-9           # tolerance for np.isclose() tie comparisons

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "hist_gradient_boosting_tuning_results.csv"

# Exact baseline hyperparameters (from
# model_implementations/hist_gradient_boosting.py's defaults). Used both
# as Trial 1 (the baseline-control) and for reference/printing.
_BASELINE = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_depth": None,
}

_MAX_DEPTH_CHOICES = [None, 3, 4, 5, 6, 8, 10]


# ---------------------------------------------------------------------------
# Search space sampling
# ---------------------------------------------------------------------------

def sample_candidates(n_random: int, seed: int) -> List[Dict[str, Any]]:
    """Draw n_random hyperparameter configurations (the search-region
    ranges are a pragmatic first-pass search region, not a claim of
    scientifically established "typical" GBM ranges) using a fixed
    RandomState, so the search itself is reproducible.

    learning_rate: log-uniform [0.01, 0.2]
    max_iter: integer uniform [100, 500]
    max_leaf_nodes: integer uniform [15, 63]
    min_samples_leaf: integer uniform [5, 50]
    l2_regularization: uniform [0.0, 1.0]
    max_depth: categorical {None, 3, 4, 5, 6, 8, 10}
    """
    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(n_random):
        learning_rate = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))
        max_iter = int(rng.randint(100, 501))
        max_leaf_nodes = int(rng.randint(15, 64))
        min_samples_leaf = int(rng.randint(5, 51))
        l2_regularization = float(rng.uniform(0.0, 1.0))
        max_depth = _MAX_DEPTH_CHOICES[rng.randint(0, len(_MAX_DEPTH_CHOICES))]

        candidates.append(
            {
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf,
                "l2_regularization": l2_regularization,
                "max_depth": max_depth,
            }
        )
    return candidates


def build_all_trials() -> List[Dict[str, Any]]:
    """Trial 1 = exact baseline hyperparameters (baseline-control).
    Trials 2-40 = 39 randomly sampled configurations."""
    baseline_control = dict(_BASELINE)
    random_candidates = sample_candidates(N_CANDIDATES - 1, SEARCH_SAMPLING_SEED)
    return [baseline_control] + random_candidates


# ---------------------------------------------------------------------------
# Complexity rule (tie-break level 4) -- explicit and documented
# ---------------------------------------------------------------------------

def complexity_key(params: Dict[str, Any]) -> tuple:
    """Sort key for "simplest configuration" (lower = simpler, sorts first).

    Rule, in order:
    1. lower max_iter * max_leaf_nodes (total leaf-splitting capacity)
    2. lower finite max_depth
    3. max_depth=None is treated as the LEAST simple depth value (sorts
       last among depth options), since an unbounded depth places no cap
       on tree complexity.
    """
    capacity = params["max_iter"] * params["max_leaf_nodes"]
    max_depth = params["max_depth"]
    if max_depth is None:
        depth_rank = (1, 0)  # group 1: None, sorts after all finite depths
    else:
        depth_rank = (0, max_depth)  # group 0: finite depths, ascending
    return (capacity, depth_rank)


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(trial_number: int, params: Dict[str, Any], is_baseline_control: bool, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    model = HistGradientBoostingRegressor(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        max_depth=params["max_depth"],
        early_stopping=False,
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    m = compute_all_metrics(y_val, predictions)

    return {
        "trial": trial_number,
        "is_baseline_control": is_baseline_control,
        "learning_rate": params["learning_rate"],
        "max_iter": params["max_iter"],
        "max_leaf_nodes": params["max_leaf_nodes"],
        "min_samples_leaf": params["min_samples_leaf"],
        "l2_regularization": params["l2_regularization"],
        "max_depth": params["max_depth"],
        "validation_mae": m["mae"],
        "validation_rmse": m["rmse"],
        "validation_r2": m["r2"],
        "validation_n": m["n"],
    }


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------

def select_best(trials_df: pd.DataFrame) -> "tuple[pd.Series, pd.DataFrame]":
    """Apply the documented 4-level tie-break rule using VALIDATION
    metrics only. Returns (winning_row, tied_group_df)."""
    best_mae = trials_df["validation_mae"].min()
    threshold = best_mae * (1.0 + TIE_THRESHOLD_FRACTION)

    # Level 1: within 1% of best MAE.
    tied = trials_df[trials_df["validation_mae"] <= threshold].copy()

    # Level 2: lowest validation RMSE (np.isclose tolerance instead of
    # exact float equality).
    min_rmse = tied["validation_rmse"].min()
    tied = tied[np.isclose(tied["validation_rmse"], min_rmse, atol=FLOAT_TOLERANCE, rtol=0.0)]

    # Level 3: highest validation R2 (np.isclose tolerance).
    max_r2 = tied["validation_r2"].max()
    tied = tied[np.isclose(tied["validation_r2"], max_r2, atol=FLOAT_TOLERANCE, rtol=0.0)]

    # Level 4: simplest configuration (lowest complexity_key).
    if len(tied) > 1:
        tied = tied.copy()
        tied["_complexity_key"] = tied.apply(
            lambda row: complexity_key(
                {
                    "max_iter": row["max_iter"],
                    "max_leaf_nodes": row["max_leaf_nodes"],
                    "max_depth": row["max_depth"],
                }
            ),
            axis=1,
        )
        tied = tied.sort_values("_complexity_key")
        tied = tied.drop(columns=["_complexity_key"])

    winner = tied.iloc[0]

    # Full within-1% group (before RMSE/R2/complexity narrowing), for
    # reporting purposes.
    full_tied_group = trials_df[trials_df["validation_mae"] <= threshold].sort_values("validation_mae")

    return winner, full_tied_group


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"{value:.{decimals}f}"


def print_config(row: pd.Series) -> None:
    control_tag = "  [BASELINE-CONTROL]" if bool(row.get("is_baseline_control", False)) else ""
    print(f"  learning_rate      = {_fmt(row['learning_rate'], 5)}{control_tag}")
    print(f"  max_iter           = {int(row['max_iter'])}")
    print(f"  max_leaf_nodes     = {int(row['max_leaf_nodes'])}")
    print(f"  min_samples_leaf   = {int(row['min_samples_leaf'])}")
    print(f"  l2_regularization  = {_fmt(row['l2_regularization'], 5)}")
    max_depth_val = row["max_depth"]
    print(f"  max_depth          = {'None' if pd.isna(max_depth_val) else int(max_depth_val)}")
    print(f"  early_stopping     = False")
    print(f"  validation_mae     = {_fmt(row['validation_mae'])}")
    print(f"  validation_rmse    = {_fmt(row['validation_rmse'])}")
    print(f"  validation_r2      = {_fmt(row['validation_r2'])}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    print("HISTGRADIENTBOOSTING HYPERPARAMETER TUNING (Layer 2, validation-only)")
    print("=" * 74)

    print(f"\nLayer: {LAYER.value}")
    print(f"Model random_state: {MODEL_RANDOM_STATE}")
    print(f"Search-sampling seed: {SEARCH_SAMPLING_SEED}")
    print(f"Candidates: {N_CANDIDATES} (Trial 1 = baseline-control, Trials 2-{N_CANDIDATES} = random)")
    print("early_stopping=False for every candidate (including the baseline-control)")
    print("TEST and OOD are not loaded or evaluated by this script.")

    # ---- Load TRAIN and VALIDATION only ----
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    val_split = loader.load(Split.VALIDATION)

    print(f"\nTRAIN rows: {len(train_split)}   VALIDATION rows: {len(val_split)}")

    # ---- Build all 40 trials: 1 baseline-control + 39 random ----
    all_params = build_all_trials()

    # ---- Run trials ----
    trial_rows = []
    for i, params in enumerate(all_params, start=1):
        is_baseline_control = (i == 1)
        result = run_trial(
            trial_number=i,
            params=params,
            is_baseline_control=is_baseline_control,
            X_train=train_split.X,
            y_train=train_split.y,
            X_val=val_split.X,
            y_val=val_split.y,
        )
        trial_rows.append(result)
        tag = " [baseline-control]" if is_baseline_control else ""
        print(
            f"Trial {i:02d}/{N_CANDIDATES}{tag}: "
            f"MAE={_fmt(result['validation_mae'])}  "
            f"RMSE={_fmt(result['validation_rmse'])}  "
            f"R2={_fmt(result['validation_r2'])}"
        )

    trials_df = pd.DataFrame(trial_rows)

    # ---- Save all 40 trials ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(_OUTPUT_CSV, index=False)

    # ---- Selection ----
    winner, tied_group = select_best(trials_df)
    runner_up_candidates = trials_df[trials_df["trial"] != winner["trial"]].sort_values("validation_mae")
    runner_up = runner_up_candidates.iloc[0] if not runner_up_candidates.empty else None

    baseline_control_row = trials_df[trials_df["is_baseline_control"]].iloc[0]

    print("\nBASELINE-CONTROL (Trial 1: baseline hyperparameters, early_stopping=False)")
    print("-" * 74)
    print_config(baseline_control_row)

    print("\nBEST CONFIGURATION")
    print("-" * 19)
    print_config(winner)
    print(f"  (trial #{int(winner['trial'])})")

    if int(winner["trial"]) == 1:
        print("  Note: the baseline-control configuration itself won the search.")
    else:
        mae_delta = baseline_control_row["validation_mae"] - winner["validation_mae"]
        print(
            f"  Validation MAE improvement over baseline-control (same "
            f"early_stopping=False setting): {_fmt(mae_delta)} "
            f"({_fmt(mae_delta / baseline_control_row['validation_mae'] * 100, 2)}%)"
        )

    if runner_up is not None:
        print("\nRUNNER-UP (next-lowest validation MAE, outside or within the tie group)")
        print("-" * 74)
        print_config(runner_up)
        print(f"  (trial #{int(runner_up['trial'])})")

    print(f"\nCandidates within {TIE_THRESHOLD_FRACTION * 100:.0f}% of best validation MAE: {len(tied_group)}")
    if len(tied_group) > 1:
        print("(tie-break applied: lowest RMSE -> highest R2 -> simplest configuration)")
        print("(simplicity rule: lower max_iter * max_leaf_nodes, then lower finite max_depth, "
              "with max_depth=None treated as least simple)")

    print("\nTOP 10 TRIALS BY VALIDATION MAE")
    print("-" * 32)
    top10 = trials_df.sort_values("validation_mae").head(10)
    display_cols = [
        "trial", "is_baseline_control", "learning_rate", "max_iter", "max_leaf_nodes",
        "min_samples_leaf", "l2_regularization", "max_depth",
        "validation_mae", "validation_rmse", "validation_r2",
    ]
    print(top10[display_cols].to_string(index=False))

    print(f"\nAll {N_CANDIDATES} trial results written to: {_OUTPUT_CSV}")
    print(
        "\nNo test/OOD evaluation was performed. No model artifact was "
        "trained or saved for the final test/OOD run -- that remains a "
        "separate, later step using the winning configuration above."
    )


if __name__ == "__main__":
    main()