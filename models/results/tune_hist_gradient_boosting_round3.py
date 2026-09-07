"""
tune_hist_gradient_boosting_round3.py
========================================
Round 3 hyperparameter tuning for HistGradientBoosting on Layer 2
(layer2_p11), validation-only. A SMALL, TARGETED LOCAL REFINEMENT around
Round 2's strongest configurations -- not another broad random search.

This script does NOT modify tune_hist_gradient_boosting.py,
tune_hist_gradient_boosting_round2.py, or either of their CSV outputs in
any way -- all four are left completely untouched.

This script:
    * loads TRAIN and VALIDATION via the existing DatasetLoader --
      no new split logic, no dataset/feature modification
    * constructs sklearn.ensemble.HistGradientBoostingRegressor directly
      for each candidate (same approach as Rounds 1/2, bypassing
      model_implementations/hist_gradient_boosting.py, which does not
      expose early_stopping) -- that wrapper file is left unmodified
    * fits every candidate on TRAIN only, scores on VALIDATION only
    * NEVER loads or evaluates TEST or OOD
    * reuses metrics.compute_all_metrics() for MAE/RMSE/R2 -- no metric
      math is reimplemented
    * does NOT call experiment_runner.run_experiment()
    * saves all 24 trial results to
      models/results/hist_gradient_boosting_tuning_round3_results.csv
    * selects a winner using the SAME documented tie-break rule as
      Rounds 1/2, using VALIDATION metrics only
    * does NOT train or save a final test/ood model -- that is a
      separate, later step

Search-space derivation (from inspecting the Round 2 CSV directly, not
from a fixed assumption -- see the module-level comment block below for
the observed top-10 table this was derived from)
------------------------------------------------------------------------
Round 2's top 10 trials by validation MAE cluster roughly as:
    learning_rate     : mostly 0.025-0.039, two competitive outliers at
                         0.058 (best trial) and 0.084
    max_iter          : spread 151-398, no strong concentration; the
                         single best trial sits at the LOW edge (151)
    max_leaf_nodes    : mostly 45-55 (range across top 10: 36-55)
    min_samples_leaf  : mostly 25-35 (range across top 10: 21-35)
    l2_regularization : mostly 0.47-1.01, two higher-l2 outliers at
                         1.30 and 1.43 still placed top-6
    max_depth         : mostly 9-10, with 7, 8, and 12 each appearing
                         once in the top 10; 5/6/None never competitive
                         in Round 1 or Round 2

Round 3 search region used below (narrower than Round 2's, derived from
the above rather than assumed):
    learning_rate     : log-uniform [0.02, 0.07]
    max_iter          : integer uniform [150, 380]
    max_leaf_nodes    : integer uniform [42, 55]
    min_samples_leaf  : integer uniform [20, 38]
    l2_regularization : uniform [0.4, 1.2]
    max_depth         : categorical {7, 8, 9, 10, 12}

Control / reference trials
-----------------------------
Four fixed control/reference trials are included first (Trials 1-4),
before any randomly sampled local candidate, so every Round 3 candidate
can be compared against Round 2's strongest known points under identical
conditions (same TRAIN/VALIDATION split, same early_stopping, same model
random_state):

    Trial 1 = Round 2's Trial 13 (best Round 2 MAE = 4.984759)
    Trial 2 = Round 2's Trial 7  (2nd-best Round 2 MAE = 4.986771)
    Trial 3 = Round 2's Trial 39 (3rd-best Round 2 MAE = 4.992469)
    Trial 4 = Round 2's own control point (= Round 1's winning Trial 26;
              validation MAE = 5.075884), kept as the fixed anchor/
              baseline reference across all three rounds

Trials 5-24 are 20 newly sampled candidates drawn from the narrower
Round 3 search region above. Total budget: 24 trials (small, targeted,
per the local-refinement goal of this round).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/results/tune_hist_gradient_boosting_round3.py -> models/ is the
# parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader  # noqa: E402
from experiment_config import Layer, Split  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration (identical philosophy to Rounds 1/2, smaller budget)
# ---------------------------------------------------------------------------

LAYER = Layer.LAYER2_P11
MODEL_RANDOM_STATE = 42          # fixed for every candidate model
SEARCH_SAMPLING_SEED = 42        # controls which local candidates are drawn
N_CONTROL_TRIALS = 4             # Round 2's Trial 13, Trial 7, Trial 39 + anchor
N_RANDOM_TRIALS = 20             # local candidates around the good region
N_CANDIDATES = N_CONTROL_TRIALS + N_RANDOM_TRIALS  # 24 total
TIE_THRESHOLD_FRACTION = 0.01    # 1% of best validation MAE
FLOAT_TOLERANCE = 1e-9           # tolerance for np.isclose() tie comparisons

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "hist_gradient_boosting_tuning_round3_results.csv"

# Four control/reference configurations, taken verbatim from the Round 2
# CSV -- used as Trials 1-4 here, ahead of any randomly sampled local
# candidate, so Round 3 results are directly comparable against them.
_CONTROLS: List[Dict[str, Any]] = [
    {  # Round 2 Trial 13 -- best Round 2 MAE (4.984759)
        "label": "round2_trial13_best",
        "learning_rate": 0.058091581453305466,
        "max_iter": 151,
        "max_leaf_nodes": 54,
        "min_samples_leaf": 25,
        "l2_regularization": 0.4703241617286455,
        "max_depth": 8,
    },
    {  # Round 2 Trial 7 -- 2nd-best Round 2 MAE (4.986771)
        "label": "round2_trial7_2nd",
        "learning_rate": 0.025034102152280427,
        "max_iter": 369,
        "max_leaf_nodes": 52,
        "min_samples_leaf": 35,
        "l2_regularization": 0.5026027425593955,
        "max_depth": 10,
    },
    {  # Round 2 Trial 39 -- 3rd-best Round 2 MAE (4.992469)
        "label": "round2_trial39_3rd",
        "learning_rate": 0.029408750401585824,
        "max_iter": 279,
        "max_leaf_nodes": 45,
        "min_samples_leaf": 31,
        "l2_regularization": 0.7821181139450191,
        "max_depth": 10,
    },
    {  # Round 2's own control (= Round 1's winning Trial 26) -- fixed
        # anchor/baseline reference carried across all three rounds
        "label": "round2_control_anchor",
        "learning_rate": 0.03419168411765895,
        "max_iter": 270,
        "max_leaf_nodes": 43,
        "min_samples_leaf": 40,
        "l2_regularization": 0.942853570557981,
        "max_depth": 10,
    },
]

_MAX_DEPTH_CHOICES = [7, 8, 9, 10, 12]


# ---------------------------------------------------------------------------
# Search space sampling -- narrow Round 3 local-refinement region,
# derived from inspecting Round 2's top-10 trials (see module docstring)
# ---------------------------------------------------------------------------

def sample_candidates(n_random: int, seed: int) -> List[Dict[str, Any]]:
    """Draw n_random hyperparameter configurations from the narrow Round 3
    local-refinement region, using a fixed RandomState so the search
    itself is reproducible.

    learning_rate: log-uniform [0.02, 0.07]
    max_iter: integer uniform [150, 380]
    max_leaf_nodes: integer uniform [42, 55]
    min_samples_leaf: integer uniform [20, 38]
    l2_regularization: uniform [0.4, 1.2]
    max_depth: categorical {7, 8, 9, 10, 12}
    """
    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(n_random):
        learning_rate = float(np.exp(rng.uniform(np.log(0.02), np.log(0.07))))
        max_iter = int(rng.randint(150, 381))
        max_leaf_nodes = int(rng.randint(42, 56))
        min_samples_leaf = int(rng.randint(20, 39))
        l2_regularization = float(rng.uniform(0.4, 1.2))
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
    """Trials 1-4 = the four control/reference configurations (Round 2's
    Trial 13, Trial 7, Trial 39, and the Round-2 control anchor), in that
    order. Trials 5-24 = 20 newly sampled local candidates."""
    control_trials = [dict(c) for c in _CONTROLS]
    random_candidates = sample_candidates(N_RANDOM_TRIALS, SEARCH_SAMPLING_SEED)
    return control_trials + random_candidates


# ---------------------------------------------------------------------------
# Complexity rule (tie-break level 4) -- identical to Rounds 1/2,
# explicit and documented
# ---------------------------------------------------------------------------

def complexity_key(params: Dict[str, Any]) -> tuple:
    """Sort key for "simplest configuration" (lower = simpler, sorts first).

    Rule, in order:
    1. lower max_iter * max_leaf_nodes (total leaf-splitting capacity)
    2. lower finite max_depth
    3. max_depth=None is treated as the LEAST simple depth value (sorts
       last among depth options), since an unbounded depth places no cap
       on tree complexity. (Not reachable in Round 3's search space,
       since None is not a candidate max_depth here -- kept only for
       consistency with the identical rule used in Rounds 1/2.)
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

def run_trial(
    trial_number: int,
    params: Dict[str, Any],
    is_control: bool,
    control_label: str,
    X_train, y_train, X_val, y_val,
) -> Dict[str, Any]:
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
        "is_control": is_control,
        "control_label": control_label,
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
# Selection rule -- identical 4-level rule to Rounds 1/2
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
    control_tag = f"  [CONTROL: {row['control_label']}]" if bool(row.get("is_control", False)) else ""
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
    print("HISTGRADIENTBOOSTING HYPERPARAMETER TUNING -- ROUND 3 (Layer 2, validation-only)")
    print("Small, targeted local refinement around Round 2's best configurations.")
    print("=" * 82)

    print(f"\nLayer: {LAYER.value}")
    print(f"Model random_state: {MODEL_RANDOM_STATE}")
    print(f"Search-sampling seed: {SEARCH_SAMPLING_SEED}")
    print(
        f"Candidates: {N_CANDIDATES} total "
        f"({N_CONTROL_TRIALS} control/reference trials + {N_RANDOM_TRIALS} local candidates)"
    )
    print("early_stopping=False for every candidate (including all controls)")
    print("TEST and OOD are not loaded or evaluated by this script.")
    print("\nRound 3 search region (narrow local refinement, derived from Round 2's top-10 trials):")
    print("  learning_rate     : log-uniform [0.02, 0.07]")
    print("  max_iter          : integer uniform [150, 380]")
    print("  max_leaf_nodes    : integer uniform [42, 55]")
    print("  min_samples_leaf  : integer uniform [20, 38]")
    print("  l2_regularization : uniform [0.4, 1.2]")
    print("  max_depth         : categorical {7, 8, 9, 10, 12}")

    # ---- Load TRAIN and VALIDATION only ----
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    val_split = loader.load(Split.VALIDATION)

    print(f"\nTRAIN rows: {len(train_split)}   VALIDATION rows: {len(val_split)}")

    # ---- Build all 24 trials: 4 controls + 20 random local candidates ----
    all_params = build_all_trials()

    # ---- Run trials ----
    trial_rows = []
    for i, params in enumerate(all_params, start=1):
        is_control = i <= N_CONTROL_TRIALS
        control_label = params.get("label", "") if is_control else ""
        result = run_trial(
            trial_number=i,
            params=params,
            is_control=is_control,
            control_label=control_label,
            X_train=train_split.X,
            y_train=train_split.y,
            X_val=val_split.X,
            y_val=val_split.y,
        )
        trial_rows.append(result)
        tag = f" [control: {control_label}]" if is_control else ""
        print(
            f"Trial {i:02d}/{N_CANDIDATES}{tag}: "
            f"MAE={_fmt(result['validation_mae'])}  "
            f"RMSE={_fmt(result['validation_rmse'])}  "
            f"R2={_fmt(result['validation_r2'])}"
        )

    trials_df = pd.DataFrame(trial_rows)

    # ---- Save all 24 trials ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(_OUTPUT_CSV, index=False)

    # ---- Selection ----
    winner, tied_group = select_best(trials_df)
    runner_up_candidates = trials_df[trials_df["trial"] != winner["trial"]].sort_values("validation_mae")
    runner_up = runner_up_candidates.iloc[0] if not runner_up_candidates.empty else None

    control_rows = trials_df[trials_df["is_control"]].sort_values("validation_mae")
    best_round2_row = control_rows.iloc[0]  # best of the four Round-2-derived controls

    print("\nCONTROL / REFERENCE CONFIGURATIONS (Round 2's best-known points)")
    print("-" * 82)
    for _, row in control_rows.iterrows():
        print_config(row)
        print()

    print("\nBEST ROUND 3 CONFIGURATION")
    print("-" * 27)
    print_config(winner)
    print(f"  (trial #{int(winner['trial'])})")

    if bool(winner.get("is_control", False)):
        print("  Note: a control/reference configuration itself won this search "
              "(no local candidate beat Round 2's best-known points).")
    else:
        mae_delta = best_round2_row["validation_mae"] - winner["validation_mae"]
        direction = "improvement" if mae_delta > 0 else "degradation"
        print(
            f"  Validation MAE {direction} over the best Round 2 configuration "
            f"({best_round2_row['control_label']}, MAE={_fmt(best_round2_row['validation_mae'])}): "
            f"{_fmt(abs(mae_delta))} "
            f"({_fmt(abs(mae_delta) / best_round2_row['validation_mae'] * 100, 2)}%)"
        )

    if runner_up is not None:
        print("\nRUNNER-UP (next-lowest validation MAE, outside or within the tie group)")
        print("-" * 82)
        print_config(runner_up)
        print(f"  (trial #{int(runner_up['trial'])})")

    print(f"\nCandidates within {TIE_THRESHOLD_FRACTION * 100:.0f}% of best validation MAE: {len(tied_group)}")
    if len(tied_group) > 1:
        print("(tie-break applied: lowest RMSE -> highest R2 -> simplest configuration)")
        print("(simplicity rule: lower max_iter * max_leaf_nodes, then lower finite max_depth)")

    print("\nTOP 10 TRIALS BY VALIDATION MAE")
    print("-" * 32)
    top10 = trials_df.sort_values("validation_mae").head(10)
    display_cols = [
        "trial", "is_control", "control_label", "learning_rate", "max_iter", "max_leaf_nodes",
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