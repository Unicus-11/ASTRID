"""
tune_hist_gradient_boosting_round4.py
========================================
Round 4 hyperparameter tuning for HistGradientBoosting on Layer 2
(layer2_p11), validation-only. FINAL, SMALLEST fine-tuning round: a
structured local search targeted specifically at resolving whether
there is a better configuration between Round 2/3's two strongest,
meaningfully-different local points (Trial 13 and Trial 7), rather than
another broad or even narrowed random search.

This script does NOT modify tune_hist_gradient_boosting.py,
tune_hist_gradient_boosting_round2.py, tune_hist_gradient_boosting_round3.py,
or any of their CSV outputs -- all are left completely untouched.

This script:
    * loads TRAIN and VALIDATION via the existing DatasetLoader --
      no new split logic, no dataset/feature modification
    * constructs sklearn.ensemble.HistGradientBoostingRegressor directly
      for each candidate (same approach as Rounds 1-3, bypassing
      model_implementations/hist_gradient_boosting.py, which does not
      expose early_stopping) -- that wrapper file is left unmodified
    * fits every candidate on TRAIN only, scores on VALIDATION only
    * NEVER loads or evaluates TEST or OOD
    * reuses metrics.compute_all_metrics() for MAE/RMSE/R2 -- no metric
      math is reimplemented
    * does NOT call experiment_runner.run_experiment()
    * saves all 18 trial results to
      models/results/hist_gradient_boosting_tuning_round4_results.csv
    * selects a winner using the SAME documented tie-break rule as
      Rounds 1-3, using VALIDATION metrics only
    * does NOT train or save a final test/ood model

------------------------------------------------------------------------
WHY THIS DESIGN (derived from inspecting the actual Round 2 + Round 3
trial results -- see the accompanying chat explanation for the full
analysis; summarized here for anyone reading this file later)
------------------------------------------------------------------------
Across 60 trials total (40 in Round 2, 24 in Round 3, including 20 freshly
random Round 3 candidates spanning a range that covers both regions), NO
candidate has ever beaten Round 2's Trial 13 (MAE=4.984759) or Trial 7
(MAE=4.986771). Trial 13 and Trial 7 differ by roughly 2.3x in
learning_rate, 2.4x in max_iter, and use different max_depth (8 vs 10),
yet land within 0.04% of each other on validation MAE. Their
learning_rate * max_iter products are nearly equal (8.77 vs 9.24) -- the
classic boosting step-size/step-count trade-off -- suggesting a possible learning-rate / iteration trade-off, which Round 4 explicitly test. along that product, rather than two
disconnected optima. Independent random sampling of learning_rate and
max_iter (as Rounds 2 and 3 did) rarely lands ON a ridge; it mostly lands
off it, which is consistent with Round 3 finding nothing better despite
covering the region.

Round 4 therefore replaces independent random sampling with:
  (a) explicit interpolation between Trial 13 and Trial 7 in parameter
      space, to probe the ridge directly, and
  (b) small local jitter around EACH anchor separately, holding the
      learning_rate * max_iter product close to that anchor's own
      product (preserving its "regime") while refining the secondary
      parameters (max_leaf_nodes, min_samples_leaf, l2_regularization,
      max_depth), which the Round 2/3 data show are comparatively STABLE
      (top trials cluster tightly: leaves 43-55, l2 0.44-0.94, depth
      7-10) and therefore need only small refinement, not broad
      re-exploration.
  (c) two explicit mid-product points with fixed (not interpolated)
      secondary parameters, as a third, independent probe of the middle
      of the ridge.

Total: 18 trials (3 controls/references + 3 interpolation + 5 Trial-13
jitter + 5 Trial-7 jitter + 2 mid-product points).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# models/results/tune_hist_gradient_boosting_round4.py -> models/ is the
# parent's parent.
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
JITTER_SAMPLING_SEED = 42        # controls the local-jitter candidates
TIE_THRESHOLD_FRACTION = 0.01    # 1% of best validation MAE
FLOAT_TOLERANCE = 1e-9           # tolerance for np.isclose() tie comparisons

_RESULTS_DIR = Path(__file__).resolve().parent
_OUTPUT_CSV = _RESULTS_DIR / "hist_gradient_boosting_tuning_round4_results.csv"

# ---------------------------------------------------------------------------
# The two anchor points (Round 2's Trial 13 and Trial 7) and the Trial 39
# reference, all taken verbatim from the Round 2 CSV.
# ---------------------------------------------------------------------------

TRIAL_13 = {
    "label": "trial13_anchor",
    "learning_rate": 0.058091581453305466,
    "max_iter": 151,
    "max_leaf_nodes": 54,
    "min_samples_leaf": 25,
    "l2_regularization": 0.4703241617286455,
    "max_depth": 8,
}
TRIAL_7 = {
    "label": "trial7_anchor",
    "learning_rate": 0.025034102152280427,
    "max_iter": 369,
    "max_leaf_nodes": 52,
    "min_samples_leaf": 35,
    "l2_regularization": 0.5026027425593955,
    "max_depth": 10,
}
TRIAL_39 = {
    "label": "trial39_reference",
    "learning_rate": 0.029408750401585824,
    "max_iter": 279,
    "max_leaf_nodes": 45,
    "min_samples_leaf": 31,
    "l2_regularization": 0.7821181139450191,
    "max_depth": 10,
}

_PRODUCT_13 = TRIAL_13["learning_rate"] * TRIAL_13["max_iter"]   # ~8.7718
_PRODUCT_7 = TRIAL_7["learning_rate"] * TRIAL_7["max_iter"]      # ~9.2376


# ---------------------------------------------------------------------------
# (a) Ridge interpolation between Trial 13 and Trial 7
# ---------------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    """Standard round-half-up (not Python's banker's rounding), used only
    for the small number of interpolated integer/categorical values below
    so the mapping from a fractional interpolation to an int is explicit
    and unambiguous."""
    import math
    return int(math.floor(x + 0.5))


def build_interpolation_trials() -> List[Dict[str, Any]]:
    """Three points linearly interpolated between Trial 13 and Trial 7 in
    full parameter space, at t=0.25, 0.5, 0.75. This directly probes the
    hypothesized ridge connecting the two anchors, rather than resampling
    the region around them independently."""
    trials = []
    for t in (0.25, 0.5, 0.75):
        learning_rate = TRIAL_13["learning_rate"] + t * (TRIAL_7["learning_rate"] - TRIAL_13["learning_rate"])
        max_iter = _round_half_up(TRIAL_13["max_iter"] + t * (TRIAL_7["max_iter"] - TRIAL_13["max_iter"]))
        max_leaf_nodes = _round_half_up(TRIAL_13["max_leaf_nodes"] + t * (TRIAL_7["max_leaf_nodes"] - TRIAL_13["max_leaf_nodes"]))
        min_samples_leaf = _round_half_up(TRIAL_13["min_samples_leaf"] + t * (TRIAL_7["min_samples_leaf"] - TRIAL_13["min_samples_leaf"]))
        l2_regularization = TRIAL_13["l2_regularization"] + t * (TRIAL_7["l2_regularization"] - TRIAL_13["l2_regularization"])
        max_depth = _round_half_up(TRIAL_13["max_depth"] + t * (TRIAL_7["max_depth"] - TRIAL_13["max_depth"]))

        trials.append(
            {
                "label": f"ridge_interp_t{t}",
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf,
                "l2_regularization": l2_regularization,
                "max_depth": max_depth,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# (b) Local jitter around each anchor, holding lr*max_iter close to that
# anchor's own product (preserving its regime), refining only the
# comparatively stable secondary parameters.
# ---------------------------------------------------------------------------

def build_local_jitter_trials(
    anchor: Dict[str, Any],
    product: float,
    lr_low: float,
    lr_high: float,
    max_iter_clip: "tuple[int, int]",
    leaf_nodes_low: int,
    leaf_nodes_high: int,
    min_leaf_low: int,
    min_leaf_high: int,
    l2_low: float,
    l2_high: float,
    depth_cycle: List[int],
    n_candidates: int,
    seed: int,
    label_prefix: str,
) -> List[Dict[str, Any]]:
    """n_candidates small local perturbations around one anchor.
    learning_rate is drawn uniformly in [lr_low, lr_high]; max_iter is
    back-solved from a product close to the anchor's own product
    (+/-10%, via a fresh RandomState draw) rather than sampled
    independently, so each candidate stays in the anchor's regime rather
    than drifting toward the other anchor's. max_leaf_nodes,
    min_samples_leaf, and l2_regularization are jittered within narrow,
    explicit bounds around the anchor. max_depth cycles deterministically
    through depth_cycle so every depth in the small candidate set is
    covered exactly once (round-robin, not randomly sampled)."""
    rng = np.random.RandomState(seed)
    trials = []
    for i in range(n_candidates):
        learning_rate = float(rng.uniform(lr_low, lr_high))
        product_i = product * float(rng.uniform(0.9, 1.1))
        max_iter = int(round(product_i / learning_rate))
        max_iter = int(np.clip(max_iter, max_iter_clip[0], max_iter_clip[1]))

        max_leaf_nodes = int(rng.randint(leaf_nodes_low, leaf_nodes_high + 1))
        min_samples_leaf = int(rng.randint(min_leaf_low, min_leaf_high + 1))
        l2_regularization = float(rng.uniform(l2_low, l2_high))
        max_depth = depth_cycle[i % len(depth_cycle)]

        trials.append(
            {
                "label": f"{label_prefix}_{i + 1}",
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf,
                "l2_regularization": l2_regularization,
                "max_depth": max_depth,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# (c) Explicit mid-product points with fixed (not interpolated) secondary
# parameters -- a third, independent probe of the ridge's middle.
# ---------------------------------------------------------------------------

def build_mid_product_trials() -> List[Dict[str, Any]]:
    mean_product = (_PRODUCT_13 + _PRODUCT_7) / 2.0  # ~9.0047
    mean_leaves = _round_half_up((TRIAL_13["max_leaf_nodes"] + TRIAL_7["max_leaf_nodes"]) / 2.0)  # 53
    mean_min_leaf = _round_half_up((TRIAL_13["min_samples_leaf"] + TRIAL_7["min_samples_leaf"]) / 2.0)  # 30
    mean_l2 = (TRIAL_13["l2_regularization"] + TRIAL_7["l2_regularization"]) / 2.0  # ~0.4865

    lr_a = 0.035  # leaning toward Trial 7's (lower-lr, higher-iter) side
    lr_b = 0.045  # leaning toward Trial 13's (higher-lr, lower-iter) side

    return [
        {
            "label": "mid_product_lean_trial7",
            "learning_rate": lr_a,
            "max_iter": int(round(mean_product / lr_a)),
            "max_leaf_nodes": mean_leaves,
            "min_samples_leaf": mean_min_leaf,
            "l2_regularization": mean_l2,
            "max_depth": 10,  # Trial 7's depth
        },
        {
            "label": "mid_product_lean_trial13",
            "learning_rate": lr_b,
            "max_iter": int(round(mean_product / lr_b)),
            "max_leaf_nodes": mean_leaves,
            "min_samples_leaf": mean_min_leaf,
            "l2_regularization": mean_l2,
            "max_depth": 8,  # Trial 13's depth
        },
    ]


def build_all_trials() -> List[Dict[str, Any]]:
    """Trials 1-3: controls/references (Trial 13, Trial 7, Trial 39).
    Trials 4-6: ridge interpolation.
    Trials 7-11: local jitter around Trial 13 (depth cycling 7,8,9).
    Trials 12-16: local jitter around Trial 7 (depth cycling 9,10,11).
    Trials 17-18: explicit mid-product points.
    Total: 18."""
    controls = [dict(TRIAL_13), dict(TRIAL_7), dict(TRIAL_39)]
    interpolation = build_interpolation_trials()

    jitter_13 = build_local_jitter_trials(
        anchor=TRIAL_13,
        product=_PRODUCT_13,
        lr_low=0.045, lr_high=0.075,
        max_iter_clip=(120, 200),
        leaf_nodes_low=50, leaf_nodes_high=56,
        min_leaf_low=20, min_leaf_high=30,
        l2_low=0.35, l2_high=0.65,
        depth_cycle=[7, 8, 9],
        n_candidates=5,
        seed=JITTER_SAMPLING_SEED,
        label_prefix="near_trial13",
    )
    jitter_7 = build_local_jitter_trials(
        anchor=TRIAL_7,
        product=_PRODUCT_7,
        lr_low=0.018, lr_high=0.032,
        max_iter_clip=(320, 420),
        leaf_nodes_low=48, leaf_nodes_high=55,
        min_leaf_low=28, min_leaf_high=40,
        l2_low=0.35, l2_high=0.75,
        depth_cycle=[9, 10, 12, 10, 9],
        n_candidates=5,
        # Different seed offset from the Trial-13 jitter draw so the two
        # local-jitter batches are independent, while each individually
        # remains reproducible from JITTER_SAMPLING_SEED.
        seed=JITTER_SAMPLING_SEED + 1,
        label_prefix="near_trial7",
    )

    mid_product = build_mid_product_trials()

    all_trials = controls + interpolation + jitter_13 + jitter_7 + mid_product
    return all_trials


CONTROL_LABELS = {"trial13_anchor", "trial7_anchor", "trial39_reference"}


# ---------------------------------------------------------------------------
# Complexity rule (tie-break level 4) -- identical to Rounds 1-3
# ---------------------------------------------------------------------------

def complexity_key(params: Dict[str, Any]) -> tuple:
    """Sort key for "simplest configuration" (lower = simpler, sorts first).

    Rule, in order:
    1. lower max_iter * max_leaf_nodes (total leaf-splitting capacity)
    2. lower finite max_depth
    3. max_depth=None is treated as the LEAST simple depth value. (Not
       reachable in Round 4's design -- every candidate has a finite
       max_depth -- kept only for consistency with the identical rule
       used in Rounds 1-3.)
    """
    capacity = params["max_iter"] * params["max_leaf_nodes"]
    max_depth = params["max_depth"]
    if max_depth is None:
        depth_rank = (1, 0)
    else:
        depth_rank = (0, max_depth)
    return (capacity, depth_rank)


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(
    trial_number: int,
    params: Dict[str, Any],
    X_train, y_train, X_val, y_val,
) -> Dict[str, Any]:
    is_control = params["label"] in CONTROL_LABELS

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
        "label": params["label"],
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
# Selection rule -- identical 4-level rule to Rounds 1-3
# ---------------------------------------------------------------------------

def select_best(trials_df: pd.DataFrame) -> "tuple[pd.Series, pd.DataFrame]":
    best_mae = trials_df["validation_mae"].min()
    threshold = best_mae * (1.0 + TIE_THRESHOLD_FRACTION)

    tied = trials_df[trials_df["validation_mae"] <= threshold].copy()

    min_rmse = tied["validation_rmse"].min()
    tied = tied[np.isclose(tied["validation_rmse"], min_rmse, atol=FLOAT_TOLERANCE, rtol=0.0)]

    max_r2 = tied["validation_r2"].max()
    tied = tied[np.isclose(tied["validation_r2"], max_r2, atol=FLOAT_TOLERANCE, rtol=0.0)]

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
    tag = f"  [{row['label']}]" if row.get("label") else ""
    print(f"  learning_rate      = {_fmt(row['learning_rate'], 5)}{tag}")
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
    all_params = build_all_trials()
    n_candidates = len(all_params)

    print("HISTGRADIENTBOOSTING HYPERPARAMETER TUNING -- ROUND 4 (Layer 2, validation-only)")
    print("Final, smallest round: structured ridge-interpolation + local jitter around Trial 13 / Trial 7.")
    print("=" * 90)

    print(f"\nLayer: {LAYER.value}")
    print(f"Model random_state: {MODEL_RANDOM_STATE}")
    print(f"Jitter-sampling seed: {JITTER_SAMPLING_SEED} (and seed+1 for the second jitter batch)")
    print(f"Candidates: {n_candidates} total")
    print("  Trials 1-3   : controls/references (Trial 13, Trial 7, Trial 39, verbatim from Round 2)")
    print("  Trials 4-6   : ridge interpolation between Trial 13 and Trial 7 (t=0.25, 0.5, 0.75)")
    print("  Trials 7-11  : local jitter around Trial 13's regime (depth cycling 7,8,9)")
    print("  Trials 12-16 : local jitter around Trial 7's regime (depth cycling 9,10,11)")
    print("  Trials 17-18 : explicit mid-product points (fixed averaged secondary params)")
    print("early_stopping=False for every candidate (including all controls)")
    print("TEST and OOD are not loaded or evaluated by this script.")

    # ---- Load TRAIN and VALIDATION only ----
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    val_split = loader.load(Split.VALIDATION)

    print(f"\nTRAIN rows: {len(train_split)}   VALIDATION rows: {len(val_split)}")

    # ---- Run trials ----
    trial_rows = []
    for i, params in enumerate(all_params, start=1):
        result = run_trial(
            trial_number=i,
            params=params,
            X_train=train_split.X,
            y_train=train_split.y,
            X_val=val_split.X,
            y_val=val_split.y,
        )
        trial_rows.append(result)
        tag = f" [{params['label']}]"
        print(
            f"Trial {i:02d}/{n_candidates}{tag}: "
            f"MAE={_fmt(result['validation_mae'])}  "
            f"RMSE={_fmt(result['validation_rmse'])}  "
            f"R2={_fmt(result['validation_r2'])}"
        )

    trials_df = pd.DataFrame(trial_rows)

    # ---- Save all trials ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(_OUTPUT_CSV, index=False)

    # ---- Selection ----
    winner, tied_group = select_best(trials_df)
    runner_up_candidates = trials_df[trials_df["trial"] != winner["trial"]].sort_values("validation_mae")
    runner_up = runner_up_candidates.iloc[0] if not runner_up_candidates.empty else None

    control_rows = trials_df[trials_df["is_control"]].sort_values("validation_mae")
    best_prior_row = control_rows.iloc[0]  # best of Trial 13 / Trial 7 / Trial 39

    print("\nCONTROL / REFERENCE CONFIGURATIONS (Trial 13, Trial 7, Trial 39)")
    print("-" * 90)
    for _, row in control_rows.iterrows():
        print_config(row)
        print()

    print("\nBEST ROUND 4 CONFIGURATION")
    print("-" * 27)
    print_config(winner)
    print(f"  (trial #{int(winner['trial'])})")

    if bool(winner.get("is_control", False)):
        print("  Note: a control/reference configuration itself won this search "
              "(no interpolated or jittered candidate beat Trial 13 / Trial 7 / Trial 39).")
    else:
        mae_delta = best_prior_row["validation_mae"] - winner["validation_mae"]
        direction = "improvement" if mae_delta > 0 else "degradation"
        print(
            f"  Validation MAE {direction} over the best known prior configuration "
            f"({best_prior_row['label']}, MAE={_fmt(best_prior_row['validation_mae'])}): "
            f"{_fmt(abs(mae_delta))} "
            f"({_fmt(abs(mae_delta) / best_prior_row['validation_mae'] * 100, 2)}%)"
        )

    if runner_up is not None:
        print("\nRUNNER-UP (next-lowest validation MAE, outside or within the tie group)")
        print("-" * 90)
        print_config(runner_up)
        print(f"  (trial #{int(runner_up['trial'])})")

    print(f"\nCandidates within {TIE_THRESHOLD_FRACTION * 100:.0f}% of best validation MAE: {len(tied_group)}")
    if len(tied_group) > 1:
        print("(tie-break applied: lowest RMSE -> highest R2 -> simplest configuration)")
        print("(simplicity rule: lower max_iter * max_leaf_nodes, then lower finite max_depth)")

    print("\nALL TRIALS BY VALIDATION MAE")
    print("-" * 28)
    all_sorted = trials_df.sort_values("validation_mae")
    display_cols = [
        "trial", "is_control", "label", "learning_rate", "max_iter", "max_leaf_nodes",
        "min_samples_leaf", "l2_regularization", "max_depth",
        "validation_mae", "validation_rmse", "validation_r2",
    ]
    print(all_sorted[display_cols].to_string(index=False))

    print(f"\nAll {n_candidates} trial results written to: {_OUTPUT_CSV}")
    print(
        "\nNo test/OOD evaluation was performed. No model artifact was "
        "trained or saved -- that remains a separate, later step using "
        "the winning configuration above."
    )


if __name__ == "__main__":
    main()