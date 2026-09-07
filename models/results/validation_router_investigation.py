"""
validation_router_investigation.py
====================
Read-only, analysis-only investigation into whether some observable
(camera-derived) feature or simple rule over such features can decide,
row by row, whether Random Forest or HistGradientBoosting should make
the Layer 2 prediction.

SCOPE / GUARANTEE -- VALIDATION ONLY:
    Every candidate rule in this script is developed AND scored
    exclusively on the Layer 2 VALIDATION split. TEST and OOD are never
    loaded, read, or referenced anywhere in this file. This is
    deliberate: whichever rule (if any) looks best here is meant to be
    frozen and handed to a *separate* script for a single, final
    TEST/OOD evaluation -- this script does not perform that evaluation
    and must not be extended to do so without violating the point of a
    held-out check.

ROUTING-INPUT GUARANTEE:
    true_queue_length_m (the target) is used ONLY to (a) score how well
    RF/Hist/each candidate rule performs, via metrics.compute_all_metrics(),
    and (b) as an aggregate signal when *fitting* a rule's parameters
    (e.g. "which model has lower mean validation error on approach X" or
    "which side of this visible_queue_length_m cut point should go to
    which model") -- exactly the kind of validation-set model selection
    the validation split exists for. It is NEVER read by the functions
    that apply a rule to decide a routing outcome at the row level
    (every `apply_*` function below takes only `X`, never `y`), so no
    candidate rule can leak the target into its per-row routing
    decision, only into its aggregate calibration on this split.

This script:
    * loads BOTH already-trained models via persistence.load_model() --
      no retraining, no fitting of RF or Hist
    * loads ONLY the Layer 2 VALIDATION split via the existing
      DatasetLoader -- no new splitting logic, no dataset modification
    * calls model.predict(X) once per model on validation
    * defines several candidate routing rules, each built solely from
      observable Layer 2 feature columns (never true_queue_length_m as
      a per-row input)
    * reuses metrics.compute_all_metrics() for every MAE/RMSE/R2 number
      -- no metric formula is reimplemented
    * prints a ranked comparison of RF-alone, Hist-alone, and every
      candidate rule, and calls out the simplest rule (if any) that
      beats both standalone models on validation

Nothing here writes to disk. No CSV/JSON is created. No infrastructure
file (data_loader.py, evaluate.py, experiment_runner.py, metrics.py,
persistence.py, model implementations, error_analysis*.py,
evaluate_hybrid_router.py, or model_disagreement_analysis.py) is
modified. No model is retrained, re-fit, or tuned; this is evaluation of
two existing artifacts plus a handful of fixed, validation-scored rules.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

# models/results/validation_router_investigation.py -> models/ is the
# parent's parent.
_MODELS_DIR = Path(__file__).resolve().parent.parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from data_loader import DatasetLoader, SplitData  # noqa: E402
from experiment_config import ExperimentConfig, Layer, Split  # noqa: E402
from persistence import load_model  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402

RANDOM_STATE = 42
LAYER = Layer.LAYER2_P11

# ---------------------------------------------------------------------------
# Model identity, mirrors error_analysis.py / evaluate_hybrid_router.py /
# model_disagreement_analysis.py exactly -- same artifact paths, nothing
# re-derived or guessed.
# ---------------------------------------------------------------------------
RF_EXPERIMENT_NAME = "random_forest_layer2_p11_baseline"
RF_MODEL_FILE_NAME = "random_forest.joblib"

HIST_EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p11_baseline"
HIST_MODEL_FILE_NAME = "hist_gradient_boosting.joblib"

# Observable candidate feature names. Every candidate is checked for
# actual presence in the validation split's X before use -- nothing is
# assumed or fabricated.
CONGESTION_COLUMN = "queue_reaches_camera_edge"
VISIBLE_QUEUE_COLUMN = "visible_queue_length_m"
APPROACH_COLUMN = "approach_edge"  # a key column, carried via SplitData.keys
SIGNAL_STATE_CANDIDATES = ["is_green_for_approach", "signal_state", "signal_phase"]

# Candidate cut points for visible_queue_length_m are taken from the
# OBSERVED validation distribution of that feature itself (an
# unsupervised choice of where to consider splitting) -- never from
# true_queue_length_m. Only which SIDE of a cut point routes to which
# model is chosen using validation performance.
VISIBLE_QUEUE_QUANTILES_FOR_CUTS = [0.10, 0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------------------------------
# Loading (no retraining, no dataset modification, validation split only)
# ---------------------------------------------------------------------------

def load_trained_model(experiment_name: str, model_file_name: str):
    config = ExperimentConfig(
        layer=LAYER,
        random_state=RANDOM_STATE,
        experiment_name=experiment_name,
    )
    model_path = config.output_dir() / model_file_name
    model = load_model(model_path)
    return model, model_path, config


def load_validation_split() -> SplitData:
    """Loads ONLY Split.VALIDATION. TEST and OOD are intentionally never
    referenced in this module."""
    loader = DatasetLoader(layer=LAYER)
    return loader.load(Split.VALIDATION)


def _find_signal_state_column(X: pd.DataFrame) -> Optional[str]:
    for candidate in SIGNAL_STATE_CANDIDATES:
        if candidate in X.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Candidate rule representation
# ---------------------------------------------------------------------------
#
# A "rule" is just: given X (observable features only), return a boolean
# array that is True where the row should be routed to Hist and False
# where it should go to RF. Every apply_fn below takes ONLY X.

@dataclass
class CandidateRule:
    name: str
    description: str
    apply_fn: Callable[[pd.DataFrame], np.ndarray]  # X -> route_to_hist bool array
    n_free_parameters: int  # for ranking "simplicity" among rules that tie/improve


def hybrid_predictions_for_rule(rule: CandidateRule, X: pd.DataFrame, rf_preds: np.ndarray, hist_preds: np.ndarray) -> np.ndarray:
    route_to_hist = rule.apply_fn(X)
    return np.where(route_to_hist, hist_preds, rf_preds)


# ---------------------------------------------------------------------------
# Candidate rule construction (fit on validation only; y used only in
# aggregate, never inside any apply_fn)
# ---------------------------------------------------------------------------

def build_congestion_rule(X: pd.DataFrame, y: pd.Series, rf_preds: np.ndarray, hist_preds: np.ndarray) -> Optional[CandidateRule]:
    """Route by queue_reaches_camera_edge. Both possible directions are
    scored on validation and the better one is kept, so this rule is not
    assumed a priori to favor Hist-when-congested."""
    if CONGESTION_COLUMN not in X.columns:
        return None

    col = X[CONGESTION_COLUMN]
    is_true = col.isin([True, 1, 1.0, "True", "true"]).to_numpy()

    rf_err = np.abs(rf_preds - y.to_numpy())
    hist_err = np.abs(hist_preds - y.to_numpy())

    # Direction 1: True -> Hist, False -> RF
    mae_dir1 = np.mean(np.where(is_true, hist_err, rf_err))
    # Direction 2 (reversed): True -> RF, False -> Hist
    mae_dir2 = np.mean(np.where(is_true, rf_err, hist_err))

    if mae_dir1 <= mae_dir2:
        route_to_hist_values = is_true
        direction_desc = f"{CONGESTION_COLUMN}==True -> Hist, ==False -> RF"
    else:
        route_to_hist_values = ~is_true
        direction_desc = f"{CONGESTION_COLUMN}==True -> RF, ==False -> Hist"

    def apply_fn(X_apply: pd.DataFrame, _values=route_to_hist_values, _fit_index=X.index) -> np.ndarray:
        # Recomputed generically from X_apply's own values, not captured
        # values, so this generalizes beyond the exact fit frame -- but
        # for the fit frame itself we just recompute identically.
        c = X_apply[CONGESTION_COLUMN]
        t = c.isin([True, 1, 1.0, "True", "true"]).to_numpy()
        return t if direction_desc.startswith(f"{CONGESTION_COLUMN}==True -> Hist") else ~t

    return CandidateRule(
        name="congestion_flag",
        description=f"Route on {CONGESTION_COLUMN} ({direction_desc}).",
        apply_fn=apply_fn,
        n_free_parameters=1,  # one binary direction choice
    )


def build_signal_state_rule(X: pd.DataFrame, y: pd.Series, rf_preds: np.ndarray, hist_preds: np.ndarray) -> Optional[CandidateRule]:
    """Route by a green/red (or similar binary/categorical) signal-state
    column, if the manifest declares one for this layer. Handles a
    boolean-like column the same way as the congestion rule; for a
    genuinely categorical signal-state column, falls back to a per-value
    lookup (same idea as the approach_edge rule below)."""
    signal_col = _find_signal_state_column(X)
    if signal_col is None:
        return None

    col = X[signal_col]
    unique_vals = col.dropna().unique()

    rf_err = np.abs(rf_preds - y.to_numpy())
    hist_err = np.abs(hist_preds - y.to_numpy())

    # Per-value lookup: for each observed value of the signal-state
    # column, whichever model has lower mean absolute error on
    # validation rows with that value is assigned to it. Missing
    # values default to RF (mirrors evaluate_hybrid_router.py's
    # "missing routing input -> RF" convention).
    lookup = {}
    for value in unique_vals:
        mask = (col == value).to_numpy()
        if mask.sum() == 0:
            continue
        lookup[value] = "Hist" if hist_err[mask].mean() < rf_err[mask].mean() else "RF"

    def apply_fn(X_apply: pd.DataFrame, _lookup=lookup, _col=signal_col) -> np.ndarray:
        c = X_apply[_col]
        route_to_hist = c.map(lambda v: _lookup.get(v, "RF") == "Hist")
        return route_to_hist.fillna(False).to_numpy()

    return CandidateRule(
        name=f"signal_state_{signal_col}",
        description=f"Per-value lookup on {signal_col} ({len(lookup)} distinct value(s) seen on validation).",
        apply_fn=apply_fn,
        n_free_parameters=len(lookup),
    )


def build_approach_edge_rule(keys: pd.DataFrame, y: pd.Series, rf_preds: np.ndarray, hist_preds: np.ndarray) -> Optional[CandidateRule]:
    """Per-approach lookup: for each approach_edge value observed on
    validation, route to whichever model has the lower mean absolute
    error for that approach. approach_edge is a key column (identifies
    which physical camera/approach the row came from), not derived from
    the target, so this is a legitimate observable-feature rule -- but
    note it has as many free parameters as there are approaches, so it
    is the least "simple" candidate here."""
    if APPROACH_COLUMN not in keys.columns:
        return None

    approach = keys[APPROACH_COLUMN]
    rf_err = np.abs(rf_preds - y.to_numpy())
    hist_err = np.abs(hist_preds - y.to_numpy())

    lookup = {}
    for value in approach.unique():
        mask = (approach == value).to_numpy()
        if mask.sum() == 0:
            continue
        lookup[value] = "Hist" if hist_err[mask].mean() < rf_err[mask].mean() else "RF"

    def apply_fn(X_apply: pd.DataFrame, _lookup=lookup) -> np.ndarray:
        # approach_edge is a key column, not part of X in this project's
        # convention (see DatasetLoader / SplitData). This rule is scored
        # against the validation split's own keys inside main(); a real
        # deployment would need approach_edge threaded alongside X.
        raise NotImplementedError(
            "approach_edge is a key column, not a feature column, in this "
            "project's schema -- see the dedicated scoring path in main() "
            "for how this rule is actually evaluated on validation."
        )

    rule = CandidateRule(
        name="approach_edge_lookup",
        description=f"Per-approach lookup on {APPROACH_COLUMN} ({len(lookup)} distinct approach(es) seen on validation).",
        apply_fn=apply_fn,
        n_free_parameters=len(lookup),
    )
    # Stash the lookup for the dedicated scoring path (approach_edge is a
    # key column, so it can't be generically applied via X the way the
    # other rules are -- see score_approach_edge_rule below).
    rule.lookup = lookup  # type: ignore[attr-defined]
    return rule


def score_approach_edge_rule(rule: CandidateRule, keys: pd.DataFrame, rf_preds: np.ndarray, hist_preds: np.ndarray) -> np.ndarray:
    lookup = rule.lookup  # type: ignore[attr-defined]
    approach = keys[APPROACH_COLUMN]
    route_to_hist = approach.map(lambda v: lookup.get(v, "RF") == "Hist").to_numpy()
    return np.where(route_to_hist, hist_preds, rf_preds)


def build_visible_queue_threshold_rules(X: pd.DataFrame, y: pd.Series, rf_preds: np.ndarray, hist_preds: np.ndarray) -> List[CandidateRule]:
    """For each candidate cut point (taken from the OBSERVED quantiles of
    visible_queue_length_m on validation -- an unsupervised choice of
    where to look), try both directions (below-cut -> Hist / above-cut
    -> RF, and vice versa) and keep whichever direction has the lower
    validation MAE for that cut point. Returns one CandidateRule per cut
    point actually tried, ranked later alongside everything else."""
    if VISIBLE_QUEUE_COLUMN not in X.columns:
        return []

    values = X[VISIBLE_QUEUE_COLUMN]
    rf_err = np.abs(rf_preds - y.to_numpy())
    hist_err = np.abs(hist_preds - y.to_numpy())

    rules: List[CandidateRule] = []
    cut_points = sorted(set(values.quantile(VISIBLE_QUEUE_QUANTILES_FOR_CUTS).round(3).tolist()))

    for cut in cut_points:
        below = (values < cut).to_numpy()
        # NaNs in `values` produce False in `below` (not-below), which is
        # then treated the same as "above" -- i.e. defaults toward
        # whichever model the "above" branch assigns, an explicit,
        # reportable default rather than a silent one.
        mae_dir1 = np.mean(np.where(below, hist_err, rf_err))   # below -> Hist, above -> RF
        mae_dir2 = np.mean(np.where(below, rf_err, hist_err))   # below -> RF, above -> Hist

        if mae_dir1 <= mae_dir2:
            direction_desc = f"< {cut} -> Hist, >= {cut} -> RF"
            def apply_fn(X_apply: pd.DataFrame, _cut=cut) -> np.ndarray:
                v = X_apply[VISIBLE_QUEUE_COLUMN]
                return (v < _cut).fillna(False).to_numpy()
        else:
            direction_desc = f"< {cut} -> RF, >= {cut} -> Hist"
            def apply_fn(X_apply: pd.DataFrame, _cut=cut) -> np.ndarray:
                v = X_apply[VISIBLE_QUEUE_COLUMN]
                return (~(v < _cut).fillna(True)).to_numpy()

        rules.append(
            CandidateRule(
                name=f"visible_queue_cut_{cut}",
                description=f"Route on {VISIBLE_QUEUE_COLUMN} ({direction_desc}).",
                apply_fn=apply_fn,
                n_free_parameters=2,  # one cut-point location + one direction choice
            )
        )

    return rules


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return f"{value:.{decimals}f}"


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_ranked_table(rows: List[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["mae"])
    header = (
        f"{'rule':<28} {'mae':>10} {'rmse':>10} {'r2':>10} "
        f"{'n_params':>9} {'beats_both_baselines':>21}"
    )
    print(header)
    print("-" * len(header))
    for r in rows_sorted:
        print(
            f"{r['name']:<28} {_fmt(r['mae']):>10} {_fmt(r['rmse']):>10} "
            f"{_fmt(r['r2']):>10} {r['n_free_parameters']:>9} "
            f"{str(r['beats_both_baselines']):>21}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    title = "LAYER 2 VALIDATION-ONLY ROUTING RULE INVESTIGATION"
    print(title)
    print("=" * len(title))
    print(
        "\nScope: every number below is computed on the VALIDATION split only.\n"
        "TEST and OOD are not loaded anywhere in this script. Whichever rule\n"
        "(if any) is selected here should be frozen and handed, unchanged, to a\n"
        "separate script for a single TEST/OOD evaluation."
    )

    rf_model, rf_model_path, rf_config = load_trained_model(RF_EXPERIMENT_NAME, RF_MODEL_FILE_NAME)
    hist_model, hist_model_path, hist_config = load_trained_model(HIST_EXPERIMENT_NAME, HIST_MODEL_FILE_NAME)

    print_section("MODELS")
    print(f"RF   model type : {type(rf_model).__name__}")
    print(f"RF   model path : {rf_model_path}")
    print(f"Hist model type : {type(hist_model).__name__}")
    print(f"Hist model path : {hist_model_path}")
    print(f"Layer           : {rf_config.layer.value}")
    print(f"Target column   : {rf_config.target_column}")
    print(f"Random state    : {rf_config.random_state}")

    val = load_validation_split()
    print_section("VALIDATION SPLIT")
    print(f"n rows: {len(val)}")
    print(f"Feature columns available: {val.feature_columns}")

    rf_preds = rf_model.predict(val.X)
    hist_preds = hist_model.predict(val.X)

    rf_metrics = compute_all_metrics(val.y, rf_preds)
    hist_metrics = compute_all_metrics(val.y, hist_preds)

    print_section("STANDALONE BASELINES (validation)")
    print(f"RF   -> MAE={_fmt(rf_metrics['mae'])} RMSE={_fmt(rf_metrics['rmse'])} R2={_fmt(rf_metrics['r2'])} n={rf_metrics['n']}")
    print(f"Hist -> MAE={_fmt(hist_metrics['mae'])} RMSE={_fmt(hist_metrics['rmse'])} R2={_fmt(hist_metrics['r2'])} n={hist_metrics['n']}")

    baseline_best_mae = min(rf_metrics["mae"], hist_metrics["mae"])

    # ---- Build candidate rules (fit on validation only) ----
    candidate_rules: List[CandidateRule] = []

    congestion_rule = build_congestion_rule(val.X, val.y, rf_preds, hist_preds)
    if congestion_rule is not None:
        candidate_rules.append(congestion_rule)

    signal_rule = build_signal_state_rule(val.X, val.y, rf_preds, hist_preds)
    if signal_rule is not None:
        candidate_rules.append(signal_rule)

    visible_queue_rules = build_visible_queue_threshold_rules(val.X, val.y, rf_preds, hist_preds)
    candidate_rules.extend(visible_queue_rules)

    approach_rule = build_approach_edge_rule(val.keys, val.y, rf_preds, hist_preds)

    # ---- Score every generically-applicable rule (X-only apply_fn) ----
    print_section("CANDIDATE RULES CONSIDERED")
    for rule in candidate_rules:
        print(f"- {rule.name}: {rule.description}")
    if approach_rule is not None:
        print(f"- {approach_rule.name}: {approach_rule.description}")
    if not candidate_rules and approach_rule is None:
        print("(no candidate observable columns were found in this validation split)")

    result_rows = []
    for rule in candidate_rules:
        hybrid_preds = hybrid_predictions_for_rule(rule, val.X, rf_preds, hist_preds)
        m = compute_all_metrics(val.y, hybrid_preds)
        result_rows.append(
            {
                "name": rule.name,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "n_free_parameters": rule.n_free_parameters,
                "beats_both_baselines": m["mae"] < baseline_best_mae,
            }
        )

    # approach_edge rule uses a dedicated scoring path since approach_edge
    # is a key column, not a feature column, in this project's schema.
    if approach_rule is not None:
        hybrid_preds = score_approach_edge_rule(approach_rule, val.keys, rf_preds, hist_preds)
        m = compute_all_metrics(val.y, hybrid_preds)
        result_rows.append(
            {
                "name": approach_rule.name,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "n_free_parameters": approach_rule.n_free_parameters,
                "beats_both_baselines": m["mae"] < baseline_best_mae,
            }
        )

    result_rows.append(
        {
            "name": "RF_alone",
            "mae": rf_metrics["mae"],
            "rmse": rf_metrics["rmse"],
            "r2": rf_metrics["r2"],
            "n_free_parameters": 0,
            "beats_both_baselines": False,
        }
    )
    result_rows.append(
        {
            "name": "Hist_alone",
            "mae": hist_metrics["mae"],
            "rmse": hist_metrics["rmse"],
            "r2": hist_metrics["r2"],
            "n_free_parameters": 0,
            "beats_both_baselines": False,
        }
    )

    print_section("VALIDATION COMPARISON -- RF vs Hist vs candidate rules (ranked by MAE)")
    print_ranked_table(result_rows)

    # ---- Identify the simplest improving rule, if any ----
    improving = [r for r in result_rows if r["beats_both_baselines"]]
    print_section("SIMPLEST IMPROVING RULE")
    if not improving:
        print(
            "No candidate rule beat both standalone models' validation MAE. "
            "Based on this validation split, neither observable feature "
            "investigated here provides a routing signal that helps -- "
            "the recommendation would be to keep the single better "
            f"standalone model ({'RF' if rf_metrics['mae'] <= hist_metrics['mae'] else 'Hist'})."
        )
    else:
        simplest = min(improving, key=lambda r: (r["n_free_parameters"], r["mae"]))
        print(
            f"'{simplest['name']}' is the simplest rule that beat both standalone "
            f"models on validation (MAE={_fmt(simplest['mae'])} vs "
            f"RF={_fmt(rf_metrics['mae'])} / Hist={_fmt(hist_metrics['mae'])}, "
            f"{simplest['n_free_parameters']} free parameter(s))."
        )
        print(
            "This rule is a CANDIDATE only. It has not been evaluated on TEST or "
            "OOD anywhere in this script. Freeze its exact definition and evaluate "
            "it, unchanged, in a separate script against TEST/OOD before drawing "
            "any conclusion about real generalization."
        )

    print_section("NOTE")
    print(
        "Every apply_fn above takes only observable feature/key columns (X or\n"
        "keys) and never true_queue_length_m -- the target is used solely, in\n"
        "aggregate, to calibrate which direction/lookup value a rule assigns on\n"
        "this validation split, and to score the resulting MAE/RMSE/R2. No rule\n"
        "here has been -- or should be -- selected using TEST or OOD data."
    )


if __name__ == "__main__":
    main()