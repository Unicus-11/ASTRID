"""
tune_hist_gradient_boosting_round5.py
======================================
HISTGRADIENTBOOSTING HYPERPARAMETER TUNING -- ROUND 5 (Layer 2, validation-only)

Final, smallest-scope round. This is NOT a random search. Every candidate is a
deliberate, single-factor (or depth-crossed / hybrid) perturbation of the two
Pareto-relevant anchors identified across Rounds 2-4:

    Trial 7  (round2_trial7_2nd / trial7_anchor)  -> best validation RMSE / R^2
    Trial 12 (round4 near_trial7_1)                -> best validation MAE

Round 4's ridge-interpolation experiment showed that averaging the two basins'
hyperparameters produces WORSE results than either anchor -- so Round 5 does
NOT interpolate between them. Instead it probes each anchor's neighborhood
along one axis at a time (depth, lr/iter jointly, leaves, min_samples_leaf,
l2) plus one explicit depth-swap pair and one hybrid cross-pairing, to
determine which specific parameter is responsible for the MAE-vs-RMSE split
between the two anchors.

CORRECTION (re-run, not a new round): the first run of this script
constructed each candidate via the project's model_implementations
wrapper (HistGradientBoostingModel), which does not expose/forward
early_stopping. With no early_stopping argument passed, sklearn's
HistGradientBoostingRegressor falls back to its own default
(early_stopping="auto"), which silently enables early stopping whenever
n_samples > 10,000 -- true for this project's 11,536-row TRAIN split.
That made the first run's fitted models (and therefore its validation
metrics) not comparable to Rounds 2-4, which all construct
sklearn.ensemble.HistGradientBoostingRegressor directly with an explicit
early_stopping=False. This corrected version does the same: every
candidate below constructs HistGradientBoostingRegressor directly,
bypassing model_implementations/hist_gradient_boosting.py entirely (that
wrapper file is left unmodified and is not imported here). The candidate
list, parameter values, controls, and selection rule are byte-for-byte
unchanged from the original Round 5 design -- only the model-construction
path is fixed.

Protocol (unchanged from Rounds 2-4):
    TRAIN      -> fit
    VALIDATION -> hyperparameter selection
    TEST/OOD   -> NOT loaded, NOT evaluated
    early_stopping = False for every candidate (including controls),
        passed explicitly to HistGradientBoostingRegressor
    model random_state = 42
    Layer = Layer.LAYER2_P11 (GPS penetration 11%), no dataset/feature/split changes

No model artifact is trained or saved. This script only prints/logs results
and writes the results CSV. Previous round scripts and CSVs are untouched.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader
from experiment_config import Layer, Split
from metrics import compute_all_metrics
# NOTE: model_implementations.hist_gradient_boosting.HistGradientBoostingModel
# is intentionally NOT used here -- that wrapper does not expose/forward
# early_stopping (see correction note above). Every candidate below
# constructs sklearn.ensemble.HistGradientBoostingRegressor directly,
# matching Rounds 2-4 exactly.

LAYER = Layer.LAYER2_P11
MODEL_RANDOM_STATE = 42
RESULTS_CSV_PATH = Path("models/results/hist_gradient_boosting_tuning_round5_results.csv")


@dataclass
class Candidate:
    label: str
    is_control: bool
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_depth: Optional[int]
    note: str = field(default="")


# --------------------------------------------------------------------------
# ROUND 5 CANDIDATES -- deterministic, targeted local design (no sampling)
# --------------------------------------------------------------------------
CANDIDATES: list[Candidate] = [
    # ---- Controls / references (4) ----
    Candidate(
        label="trial7_anchor", is_control=True,
        learning_rate=0.025034102152280427, max_iter=369, max_leaf_nodes=52,
        min_samples_leaf=35, l2_regularization=0.5026027425593955, max_depth=10,
        note="Best RMSE/R2 across Rounds 2-4; current rule-selected winner",
    ),
    Candidate(
        label="trial12_anchor", is_control=True,
        learning_rate=0.019610763929456904, max_iter=420, max_leaf_nodes=53,
        min_samples_leaf=38, l2_regularization=0.44623584798613947, max_depth=9,
        note="Best MAE across Rounds 2-4; new formal control",
    ),
    Candidate(
        label="trial13_anchor", is_control=True,
        learning_rate=0.058091581453305466, max_iter=151, max_leaf_nodes=54,
        min_samples_leaf=25, l2_regularization=0.4703241617286455, max_depth=8,
        note="Anchor of the high-lr/low-iter basin (Basin B)",
    ),
    Candidate(
        label="trial39_reference", is_control=True,
        learning_rate=0.029408750401585824, max_iter=279, max_leaf_nodes=45,
        min_samples_leaf=31, l2_regularization=0.7821181139450191, max_depth=10,
        note="Third-best historical config; drift sanity check",
    ),
    # ---- Local candidates (10): single-factor perturbations ----
    Candidate(
        label="depth_swap_trial7", is_control=False,
        learning_rate=0.025034102152280427, max_iter=369, max_leaf_nodes=52,
        min_samples_leaf=35, l2_regularization=0.5026027425593955, max_depth=9,
        note="Trial 7 with depth->9: isolates depth's effect on RMSE",
    ),
    Candidate(
        label="depth_swap_trial12", is_control=False,
        learning_rate=0.019610763929456904, max_iter=420, max_leaf_nodes=53,
        min_samples_leaf=38, l2_regularization=0.44623584798613947, max_depth=10,
        note="Trial 12 with depth->10: does this close the RMSE gap?",
    ),
    Candidate(
        label="iter_boost_trial12", is_control=False,
        learning_rate=0.019610763929456904, max_iter=460, max_leaf_nodes=53,
        min_samples_leaf=38, l2_regularization=0.44623584798613947, max_depth=9,
        note="Tests whether Trial 12 was iteration-starved at its low lr",
    ),
    Candidate(
        label="lr_step_to_A_at_depth9", is_control=False,
        learning_rate=0.0225, max_iter=400, max_leaf_nodes=52,
        min_samples_leaf=36, l2_regularization=0.48, max_depth=9,
        note="Small step within Basin A/B boundary at depth 9 (not a midpoint interpolation)",
    ),
    Candidate(
        label="lr_step_to_B_at_depth10", is_control=False,
        learning_rate=0.0225, max_iter=410, max_leaf_nodes=52,
        min_samples_leaf=36, l2_regularization=0.48, max_depth=10,
        note="Same small lr/iter step at depth 10, paired with the row above",
    ),
    Candidate(
        label="leaf_up_trial7", is_control=False,
        learning_rate=0.025034102152280427, max_iter=369, max_leaf_nodes=55,
        min_samples_leaf=35, l2_regularization=0.5026027425593955, max_depth=10,
        note="Probes upper edge (55) of converged max_leaf_nodes range around Trial 7",
    ),
    Candidate(
        label="minleaf_down_trial12", is_control=False,
        learning_rate=0.019610763929456904, max_iter=420, max_leaf_nodes=53,
        min_samples_leaf=30, l2_regularization=0.44623584798613947, max_depth=9,
        note="Probes lower edge (30) of converged min_samples_leaf range around Trial 12",
    ),
    Candidate(
        label="l2_down_trial7", is_control=False,
        learning_rate=0.025034102152280427, max_iter=369, max_leaf_nodes=52,
        min_samples_leaf=35, l2_regularization=0.40, max_depth=10,
        note="Probes lower edge (0.40) of converged l2 range around Trial 7",
    ),
    Candidate(
        label="l2_up_trial12", is_control=False,
        learning_rate=0.019610763929456904, max_iter=420, max_leaf_nodes=53,
        min_samples_leaf=38, l2_regularization=0.55, max_depth=9,
        note="Probes upper edge (0.55) of converged l2 range around Trial 12",
    ),
    Candidate(
        label="hybrid_A_lr_iter_depth_B_leaves", is_control=False,
        learning_rate=0.025034102152280427, max_iter=369, max_leaf_nodes=53,
        min_samples_leaf=38, l2_regularization=0.44623584798613947, max_depth=10,
        note="Trial 7's lr/iter/depth crossed with Trial 12's leaves/min_leaf/l2",
    ),
]


def simplicity_key(c: Candidate) -> tuple:
    """Existing project tie-break rule: lower max_iter * max_leaf_nodes first,
    then lower finite max_depth, with max_depth=None treated as least simple."""
    depth_for_sort = c.max_depth if c.max_depth is not None else float("inf")
    return (c.max_iter * c.max_leaf_nodes, depth_for_sort)


def select_best(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Existing selection rule:
    1. lowest validation MAE
    2. candidates within 1% of best MAE are tied
    3. among tied candidates, lowest validation RMSE
    4. then highest validation R2
    5. then simplest configuration
    """
    best_mae = df["validation_mae"].min()
    tie_mask = df["validation_mae"] <= best_mae * 1.01
    tied = df[tie_mask].copy()

    tied["_simplicity"] = tied.apply(
        lambda r: simplicity_key(
            Candidate(
                label=r["label"], is_control=bool(r["is_control"]),
                learning_rate=r["learning_rate"], max_iter=int(r["max_iter"]),
                max_leaf_nodes=int(r["max_leaf_nodes"]),
                min_samples_leaf=int(r["min_samples_leaf"]),
                l2_regularization=r["l2_regularization"],
                max_depth=None if pd.isna(r["max_depth"]) else int(r["max_depth"]),
            )
        ),
        axis=1,
    )
    tied_sorted = tied.sort_values(
        by=["validation_rmse", "validation_r2", "_simplicity"],
        ascending=[True, False, True],
    )
    return tied_sorted.iloc[0], len(tied)


def main() -> None:
    print("HISTGRADIENTBOOSTING HYPERPARAMETER TUNING -- ROUND 5 (Layer 2, validation-only)")
    print("Smallest, most targeted round: single-factor perturbations around Trial 7 / Trial 12.")
    print("=" * 90)
    print()
    print(f"Layer: {LAYER.value}")
    print(f"Model random_state: {MODEL_RANDOM_STATE}")
    print(f"Candidates: {len(CANDIDATES)} total")
    n_controls = sum(c.is_control for c in CANDIDATES)
    print(f"  Controls/references : {n_controls} (Trial 7, Trial 12, Trial 13, Trial 39, verbatim from prior rounds)")
    print(f"  Local perturbations  : {len(CANDIDATES) - n_controls} (deterministic, one-factor-at-a-time / depth-swap / hybrid)")
    print("early_stopping=False for every candidate (including all controls)")
    print("TEST and OOD are not loaded or evaluated by this script.")
    print()

    # --- Data loading: existing project interface, TRAIN + VALIDATION only ---
    loader = DatasetLoader(layer=LAYER)
    train_split = loader.load(Split.TRAIN)
    val_split = loader.load(Split.VALIDATION)

    X_train, y_train = train_split.X, train_split.y
    X_val, y_val = val_split.X, val_split.y

    print(f"TRAIN rows: {len(X_train)}   VALIDATION rows: {len(X_val)}")

    rows = []
    for i, c in enumerate(CANDIDATES, start=1):
        # Direct sklearn construction (matches Rounds 2-4) -- explicit
        # early_stopping=False, not routed through the project wrapper.
        model = HistGradientBoostingRegressor(
            learning_rate=c.learning_rate,
            max_iter=c.max_iter,
            max_leaf_nodes=c.max_leaf_nodes,
            max_depth=c.max_depth,
            min_samples_leaf=c.min_samples_leaf,
            l2_regularization=c.l2_regularization,
            early_stopping=False,
            random_state=MODEL_RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Same metrics function used by Rounds 2-4 -- no metric
        # reimplementation here.
        metrics = compute_all_metrics(y_val, preds)
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        r2 = metrics["r2"]

        tag = f" [{'control: ' if c.is_control else ''}{c.label}]"
        print(f"Trial {i:02d}/{len(CANDIDATES)}{tag}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

        rows.append({
            "trial": i,
            "is_control": c.is_control,
            "label": c.label,
            "learning_rate": c.learning_rate,
            "max_iter": c.max_iter,
            "max_leaf_nodes": c.max_leaf_nodes,
            "min_samples_leaf": c.min_samples_leaf,
            "l2_regularization": c.l2_regularization,
            "max_depth": c.max_depth,
            "validation_mae": mae,
            "validation_rmse": rmse,
            "validation_r2": r2,
            "validation_n": len(y_val),
        })

    df = pd.DataFrame(rows)

    print()
    print("CONTROL / REFERENCE CONFIGURATIONS")
    print("-" * 90)
    for _, r in df[df["is_control"]].iterrows():
        print(f"  learning_rate      = {r['learning_rate']:.5f}  [{r['label']}]")
        print(f"  max_iter           = {r['max_iter']}")
        print(f"  max_leaf_nodes     = {r['max_leaf_nodes']}")
        print(f"  min_samples_leaf   = {r['min_samples_leaf']}")
        print(f"  l2_regularization  = {r['l2_regularization']:.5f}")
        print(f"  max_depth          = {r['max_depth']}")
        print(f"  early_stopping     = False")
        print(f"  validation_mae     = {r['validation_mae']:.4f}")
        print(f"  validation_rmse    = {r['validation_rmse']:.4f}")
        print(f"  validation_r2      = {r['validation_r2']:.4f}")
        print()

    best_row, n_tied = select_best(df)
    print("BEST ROUND 5 CONFIGURATION")
    print("-" * 40)
    print(f"  learning_rate      = {best_row['learning_rate']:.5f}  [{best_row['label']}]")
    print(f"  max_iter           = {best_row['max_iter']}")
    print(f"  max_leaf_nodes     = {best_row['max_leaf_nodes']}")
    print(f"  min_samples_leaf   = {best_row['min_samples_leaf']}")
    print(f"  l2_regularization  = {best_row['l2_regularization']:.5f}")
    print(f"  max_depth          = {best_row['max_depth']}")
    print(f"  early_stopping     = False")
    print(f"  validation_mae     = {best_row['validation_mae']:.4f}")
    print(f"  validation_rmse    = {best_row['validation_rmse']:.4f}")
    print(f"  validation_r2      = {best_row['validation_r2']:.4f}")
    print(f"  (trial #{int(best_row['trial'])})")
    print()
    print(f"Candidates within 1% of best validation MAE: {n_tied}")
    print("(tie-break applied: lowest RMSE -> highest R2 -> simplest configuration)")
    print("(simplicity rule: lower max_iter * max_leaf_nodes, then lower finite max_depth,")
    print(" with max_depth=None treated as least simple)")
    print()

    top10 = df.sort_values("validation_mae").head(10)
    print("TOP 10 TRIALS BY VALIDATION MAE")
    print("-" * 40)
    print(top10.to_string(index=False))
    print()

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"All {len(CANDIDATES)} trial results written to: {RESULTS_CSV_PATH.resolve()}")
    print()
    print("No test/OOD evaluation was performed. No model artifact was trained or saved --")
    print("that remains a separate, later step using the winning configuration above.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nElapsed: {time.time() - start:.1f}s")