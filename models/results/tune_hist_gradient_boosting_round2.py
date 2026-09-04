"""
tune_hist_gradient_boosting_round2.py
========================================
Round 2 hyperparameter tuning for HistGradientBoosting on Layer 2
(layer2_p11), validation-only. Narrows the search region around Round 1's
winning trial (Trial 26).

This script does NOT modify tune_hist_gradient_boosting.py or its output
(models/results/hist_gradient_boosting_tuning_results.csv) in any way --
both are left completely untouched.

This script:
    * loads TRAIN and VALIDATION via the existing DatasetLoader --
      no new split logic, no dataset/feature modification
    * constructs sklearn.ensemble.HistGradientBoostingRegressor directly
      for each candidate (same approach as Round 1, bypassing
      model_implementations/hist_gradient_boosting.py, which does not
      expose early_stopping) -- that wrapper file is left unmodified
    * fits every candidate on TRAIN only, scores on VALIDATION only
    * NEVER loads or evaluates TEST or OOD
    * reuses metrics.compute_all_metrics() for MAE/RMSE/R2 -- no metric
      math is reimplemented
    * does NOT call experiment_runner.run_experiment()
    * saves all 40 trial results to
      models/results/hist_gradient_boosting_tuning_round2_results.csv
    * selects a winner using the SAME documented tie-break rule as
      Round 1, using VALIDATION metrics only
    * does NOT train or save a final test/ood model -- that is a
      separate, later step

Round-1-control trial
-----------------------
Round 1's winner (Trial 26) is used here as Trial 1 -- the Round-1
control -- so every Round 2 candidate can be compared against it under
identical conditions (same TRAIN/VALIDATION split, same early_stopping,
same model random_state). Trials 2-40 are 39 newly sampled candidates
drawn from the narrower Round 2 search region below. This still totals
exactly 40 trials.

Round-1 winner (Trial 26), used verbatim as Trial 1 here:
    learning_rate      = 0.03419168411765895
    max_iter           = 270
    max_leaf_nodes     = 43
    min_samples_leaf   = 40
    l2_regularization  = 0.942853570557981
    max_depth          = 10
    early_stopping     = False
    random_state       = 42
Round 1 validation result (for reference only, not re-verified here
except by re-running Trial 1 under this script's own pipeline):
    MAE  = 5.075883918524597
    RMSE = 20.365644...
    R2   = 0.972108...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/results/tune_hist_gradient_boosting_round2.py -> models/ is the
# parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader  # noqa: E402
from experiment_config import Layer, Split  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration (identical to Round 1)
# ---------------------------------------------------------------------------

LAYER = Layer.LAYER2_P11
MODEL_RANDOM_STATE = 42          # fixed for every candidate model
SEARCH_SAMPLING_SEED = 42        # controls which 39 random configs are drawn
N_CANDIDATES = 40                # 1 Round-1 control + 39 randomly sampled
TIE_THRESHOLD_FRACTION = 0.01    # 1% of best validation MAE
FLOAT_TOLERANCE = 1e-9           # tolerance for np.isclose() tie comparisons

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "hist_gradient_boosting_tuning_round2_results.csv"

# Round 1's winning configuration (Trial 26), used verbatim as Trial 1
# here -- the Round-1 control -- so Round 2 candidates are directly
# comparable against it.
_ROUND1_CONTROL = {
    "learning_rate": 0.03419168411765895,
    "max_iter": 270,
    "max_leaf_nodes": 43,
    "min_samples_leaf": 40,
    "l2_regularization": 0.942853570557981,
    "max_depth": 10,
}

_MAX_DEPTH_CHOICES = [5, 6, 7, 8, 9, 10, 12, None]


# ---------------------------------------------------------------------------
# Search space sampling -- narrowed Round 2 region around Trial 26
# ---------------------------------------------------------------------------

def sample_candidates(n_random: int, seed: int) -> List[Dict[str, Any]]:
    """Draw n_random hyperparameter configurations from the narrower
    Round 2 search region (centered on the promising Round 1 region),
    using a fixed RandomState so the search itself is reproducible.

    learning_rate: log-uniform [0.02, 0.10]
    max_iter: integer uniform [150, 400]
    max_leaf_nodes: integer uniform [25, 55]
    min_samples_leaf: integer uniform [20, 50]
    l2_regularization: uniform [0.2, 1.5]
    max_depth: categorical {5, 6, 7, 8, 9, 10, 12, None}
    """
    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(n_random):
        learning_rate = float(np.exp(rng.uniform(np.log(0.02), np.log(0.10))))
        max_iter = int(rng.randint(150, 401))
        max_leaf_nodes = int(rng.randint(25, 56))
        min_samples_leaf = int(rng.randint(20, 51))
        l2_regularization = float(rng.uniform(0.2, 1.5))
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
    """Trial 1 = exact Round-1 winner (Trial 26), used as the Round-1
    control. Trials 2-40 = 39 newly sampled configurations from the
    narrower Round 2 search region."""
    round1_control = dict(_ROUND1_CONTROL)
    random_candidates = sample_candidates(N_CANDIDATES - 1, SEARCH_SAMPLING_SEED)
    return [round1_control] + random_candidates


# ---------------------------------------------------------------------------
# Complexity rule (tie-break level 4) -- identical to Round 1, explicit
# and documented
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

def run_trial(trial_number: int, params: Dict[str, Any], is_round1_control: bool, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
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
        "is_round1_control": is_round1_control,
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
# Selection rule -- identical 4-level rule to Round 1
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
    control_tag = "  [ROUND-1 CONTROL]" if bool(row.get("is_round1_control", False)) else ""
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
    print("HISTGRADIENTBOOSTING HYPERPARAMETER TUNING -- ROUND 2 (Layer 2, validation-only)")
    print("=" * 82)

    print(f"\nLayer: {LAYER.value}")
    print(f"Model random_state: {MODEL_RANDOM_STATE}")
    print(f"Search-sampling seed: {SEARCH_SAMPLING_SEED}")
    print(f"Candidates: {N_CANDIDATES} (Trial 1 = Round-1 control [Trial 26], Trials 2-{N_CANDIDATES} = random)")
    print("early_stopping=False for every candidate (including the Round-1 control)")
    print("TEST and OOD are not loaded or evaluated by this script.")
    print("\nRound 2 search region (narrowed around Round 1's winning trial):")
    print("  learning_rate     : log-uniform [0.02, 0.10]")
    print("  max_iter          : integer uniform [150, 400]")
    print("  max_leaf_nodes    : integer uniform [25, 55]")
    print("  min_samples_leaf  : integer uniform [20, 50]")
    print("  l2_regularization : uniform [0.2, 1.5]")
    print("  max_depth         : categorical {5, 6, 7, 8, 9, 10, 12, None}")

    # ---- Load TRAIN and VALIDATION only ----
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    val_split = loader.load(Split.VALIDATION)

    print(f"\nTRAIN rows: {len(train_split)}   VALIDATION rows: {len(val_split)}")

    # ---- Build all 40 trials: 1 Round-1 control + 39 random ----
    all_params = build_all_trials()

    # ---- Run trials ----
    trial_rows = []
    for i, params in enumerate(all_params, start=1):
        is_round1_control = (i == 1)
        result = run_trial(
            trial_number=i,
            params=params,
            is_round1_control=is_round1_control,
            X_train=train_split.X,
            y_train=train_split.y,
            X_val=val_split.X,
            y_val=val_split.y,
        )
        trial_rows.append(result)
        tag = " [round1-control]" if is_round1_control else ""
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

    round1_control_row = trials_df[trials_df["is_round1_control"]].iloc[0]

    print("\nROUND-1 CONTROL (Trial 1: Round 1's winning configuration, Trial 26)")
    print("-" * 82)
    print_config(round1_control_row)

    print("\nBEST ROUND 2 CONFIGURATION")
    print("-" * 27)
    print_config(winner)
    print(f"  (trial #{int(winner['trial'])})")

    if int(winner["trial"]) == 1:
        print("  Note: the Round-1 control configuration itself won this search.")
    else:
        mae_delta = round1_control_row["validation_mae"] - winner["validation_mae"]
        direction = "improvement" if mae_delta > 0 else "degradation"
        print(
            f"  Validation MAE {direction} over Round-1 control (same "
            f"early_stopping=False setting): {_fmt(abs(mae_delta))} "
            f"({_fmt(abs(mae_delta) / round1_control_row['validation_mae'] * 100, 2)}%)"
        )

    if runner_up is not None:
        print("\nRUNNER-UP (next-lowest validation MAE, outside or within the tie group)")
        print("-" * 82)
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
        "trial", "is_round1_control", "learning_rate", "max_iter", "max_leaf_nodes",
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