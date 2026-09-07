"""
model_disagreement_analysis.py
====================
Read-only, analysis-only comparison of the two already-trained Layer 2
baseline models (Random Forest and HistGradientBoosting) at the level of
individual rows: which model is closer to the truth on each row, and
whether any observable (camera-derived) condition consistently predicts
that.

This mirrors error_analysis.py / evaluate_hybrid_router.py's loading
conventions exactly (same ExperimentConfig / load_model / DatasetLoader /
compute_all_metrics pattern), applied to a per-row disagreement analysis
instead of a routing rule or a single-model error breakdown.

This script:
    * loads BOTH ALREADY-TRAINED models via persistence.load_model() --
      no retraining, no fitting
    * loads Layer 2 test/ood splits via the existing DatasetLoader --
      no new splitting logic, no dataset modification
    * calls model.predict(X) once per model per split
    * computes each model's absolute error per row against
      true_queue_length_m (used ONLY for this analysis, never to build
      any routing/selection rule)
    * classifies each row as "RF" (Random Forest closer), "Hist"
      (HistGradientBoosting closer), or "Tie" (equal absolute error)
    * reuses metrics.compute_all_metrics() for every MAE/RMSE/R2 number,
      including for the per-row "better model" prediction -- no metric
      formula is reimplemented
    * cross-tabulates the per-row winner against observable conditions:
      queue_reaches_camera_edge, visible_queue_length_m bins,
      approach_edge, signal state (if present), and scenario_id
    * reports true_queue_length_m bins purely descriptively, alongside
      the same win/loss/tie breakdown -- NOT as an input to any decision

IMPORTANT -- NOT A ROUTER:
    This script produces descriptive statistics only. It does not build,
    fit, select, or recommend a routing rule of any kind, and it does not
    use true_queue_length_m or the per-row winner label as a feature for
    anything other than printed, human-readable reporting. Any decision
    about whether an observable feature is "consistent enough" to route
    on is left entirely to the reader.

Nothing here writes to disk. No CSV/JSON is created. No infrastructure
file (data_loader.py, evaluate.py, experiment_runner.py, metrics.py,
persistence.py, model implementations, or either error_analysis*.py /
evaluate_hybrid_router.py) is modified. No model is retrained, re-fit,
or tuned; this is evaluation of two existing artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# models/results/model_disagreement_analysis.py -> models/ is the parent's parent.
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
# Model identity, mirrors error_analysis.py / evaluate_hybrid_router.py
# exactly -- same artifact paths, nothing re-derived or guessed.
# ---------------------------------------------------------------------------
RF_EXPERIMENT_NAME = "random_forest_layer2_p11_baseline"
RF_MODEL_FILE_NAME = "random_forest.joblib"
RF_BASELINE_MODEL_NAME = "random_forest"

HIST_EXPERIMENT_NAME = "hist_gradient_boosting_layer2_p11_baseline"
HIST_MODEL_FILE_NAME = "hist_gradient_boosting.joblib"
HIST_BASELINE_MODEL_NAME = "hist_gradient_boosting"

_RESULTS_DIR = Path(__file__).resolve().parent
_BASELINE_CSV_PATH = _RESULTS_DIR / "baseline_results.csv"

# Observable (camera-derived / metadata) conditions to cross-tabulate the
# per-row winner against. Only columns actually present in a given
# split's X / metadata are used -- nothing here is assumed to exist.
CONGESTION_COLUMN = "queue_reaches_camera_edge"
VISIBLE_QUEUE_COLUMN = "visible_queue_length_m"
APPROACH_COLUMN = "approach_edge"
SCENARIO_COLUMN = "scenario_id"
# Candidate names for a signal-state feature; the manifest for this layer
# may or may not declare one, so every candidate is checked and only
# columns that actually exist in X are used. Nothing is fabricated.
SIGNAL_STATE_CANDIDATES = ["is_green_for_approach", "signal_state", "signal_phase"]

VISIBLE_QUEUE_N_QUANTILES = 4

TRUE_QUEUE_BIN_EDGES = [0, 25, 50, 100, 200, 400, np.inf]
TRUE_QUEUE_BIN_LABELS = ["0-25m", "25-50m", "50-100m", "100-200m", "200-400m", ">400m"]


# ---------------------------------------------------------------------------
# Loading (no retraining, no dataset modification)
# ---------------------------------------------------------------------------

def load_trained_model(experiment_name: str, model_file_name: str):
    """Load one already-trained, already-persisted model artifact.
    Identical mechanism to error_analysis.py / evaluate_hybrid_router.py --
    ExperimentConfig resolves the artifact directory, persistence.load_model()
    deserializes it. No fitting happens here."""
    config = ExperimentConfig(
        layer=LAYER,
        random_state=RANDOM_STATE,
        experiment_name=experiment_name,
    )
    model_path = config.output_dir() / model_file_name
    model = load_model(model_path)
    return model, model_path, config


def load_splits() -> dict:
    loader = DatasetLoader(layer=LAYER)
    return {
        "test": loader.load(Split.TEST),
        "ood": loader.load(Split.OOD),
    }


# ---------------------------------------------------------------------------
# Row-level DataFrame construction (in-memory only, analysis-only)
# ---------------------------------------------------------------------------

def _find_signal_state_column(X: pd.DataFrame) -> Optional[str]:
    for candidate in SIGNAL_STATE_CANDIDATES:
        if candidate in X.columns:
            return candidate
    return None


def build_row_level_df(split_data: SplitData, rf_preds: np.ndarray, hist_preds: np.ndarray) -> pd.DataFrame:
    """Combine SplitData.keys / .metadata / .X / .y with both models'
    predictions into one in-memory analysis DataFrame. Does not mutate
    split_data. true_queue_length_m is included for descriptive analysis
    (absolute error, binning) only -- it is never treated as, or used to
    derive, a feature for any decision rule."""
    df = pd.DataFrame(
        {
            "timestamp": split_data.keys["timestamp"].values,
            "approach_edge": split_data.keys["approach_edge"].values,
            "scenario_id": split_data.metadata["scenario_id"].values,
            "true_queue_length_m": split_data.y.values,
            "rf_prediction": rf_preds,
            "hist_prediction": hist_preds,
        }
    )
    df["rf_abs_error"] = (df["rf_prediction"] - df["true_queue_length_m"]).abs()
    df["hist_abs_error"] = (df["hist_prediction"] - df["true_queue_length_m"]).abs()

    # Winner classification: strictly a comparison of the two already-
    # computed absolute-error columns above. Ties are exact equality
    # (including both models being exactly correct).
    conditions = [
        df["rf_abs_error"] < df["hist_abs_error"],
        df["hist_abs_error"] < df["rf_abs_error"],
    ]
    choices = ["RF", "Hist"]
    df["winner"] = np.select(conditions, choices, default="Tie")

    if CONGESTION_COLUMN in split_data.X.columns:
        df[CONGESTION_COLUMN] = split_data.X[CONGESTION_COLUMN].values
    if VISIBLE_QUEUE_COLUMN in split_data.X.columns:
        df[VISIBLE_QUEUE_COLUMN] = split_data.X[VISIBLE_QUEUE_COLUMN].values
    signal_col = _find_signal_state_column(split_data.X)
    if signal_col is not None:
        df[signal_col] = split_data.X[signal_col].values

    return df, signal_col


# ---------------------------------------------------------------------------
# Win / loss / tie summary
# ---------------------------------------------------------------------------

def win_loss_tie_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    counts = df["winner"].value_counts()
    rf_wins = int(counts.get("RF", 0))
    hist_wins = int(counts.get("Hist", 0))
    ties = int(counts.get("Tie", 0))
    return {
        "n": n,
        "rf_wins": rf_wins,
        "rf_pct": (rf_wins / n * 100.0) if n else float("nan"),
        "hist_wins": hist_wins,
        "hist_pct": (hist_wins / n * 100.0) if n else float("nan"),
        "ties": ties,
        "tie_pct": (ties / n * 100.0) if n else float("nan"),
    }


def better_model_predictions(df: pd.DataFrame) -> np.ndarray:
    """Per-row prediction from whichever model had the lower absolute
    error on that row (ties default to the RF prediction, an arbitrary
    but fixed tie-break used only to compute a single 'best possible
    pointwise selection' metric -- this is NOT a routing rule, since it
    is derived from the already-known errors themselves, not from any
    observable feature)."""
    return np.where(
        df["hist_abs_error"].to_numpy() < df["rf_abs_error"].to_numpy(),
        df["hist_prediction"].to_numpy(),
        df["rf_prediction"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Cross-tabulation of winner against observable conditions
# ---------------------------------------------------------------------------

def crosstab_winner(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """For each value of group_col, report n, RF/Hist/Tie counts and
    percentages. Purely descriptive -- observable-condition columns only,
    never true_queue_length_m or the winner label used as a routing
    input elsewhere."""
    rows = []
    for value, group_df in df.groupby(group_col, observed=True, dropna=False):
        summary = win_loss_tie_summary(group_df)
        rows.append({group_col: value, **summary})
    result = pd.DataFrame(rows).set_index(group_col)
    return result.sort_values("n", ascending=False)


def add_visible_queue_bin(df: pd.DataFrame) -> "tuple[pd.DataFrame, Optional[list]]":
    if VISIBLE_QUEUE_COLUMN not in df.columns:
        return df, None
    df = df.copy()
    try:
        binned, edges = pd.qcut(
            df[VISIBLE_QUEUE_COLUMN],
            q=VISIBLE_QUEUE_N_QUANTILES,
            duplicates="drop",
            retbins=True,
        )
        df["visible_queue_bin"] = binned.astype(str)
        return df, list(edges)
    except ValueError:
        return df, None


def add_true_queue_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive-only binning of true_queue_length_m for reporting.
    NOT used to construct, select, or influence any routing rule."""
    df = df.copy()
    df["true_queue_bin"] = pd.cut(
        df["true_queue_length_m"],
        bins=TRUE_QUEUE_BIN_EDGES,
        labels=TRUE_QUEUE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    return df


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


def print_overall_metrics(label: str, rf_m: dict, hist_m: dict, better_m: dict) -> None:
    print_section(label)
    header = f"{'metric':<8} {'RF':>14} {'HistGradientBoosting':>22} {'Per-row Better':>16}"
    print(header)
    print("-" * len(header))
    for key, title in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")):
        print(
            f"{title:<8} {_fmt(rf_m[key]):>14} "
            f"{_fmt(hist_m[key]):>22} {_fmt(better_m[key]):>16}"
        )
    print(f"{'n':<8} {rf_m['n']:>14} {hist_m['n']:>22} {better_m['n']:>16}")


def print_win_loss_tie(label: str, summary: dict) -> None:
    print_section(label)
    print(f"n total : {summary['n']}")
    print(f"RF wins   : {summary['rf_wins']:>8}  ({_fmt(summary['rf_pct'], 2)}%)")
    print(f"Hist wins : {summary['hist_wins']:>8}  ({_fmt(summary['hist_pct'], 2)}%)")
    print(f"Ties      : {summary['ties']:>8}  ({_fmt(summary['tie_pct'], 2)}%)")


def print_crosstab(label: str, crosstab_df: pd.DataFrame) -> None:
    print_section(label)
    if crosstab_df.empty:
        print("(no data)")
        return
    display = crosstab_df.copy()
    display["rf_pct"] = display["rf_pct"].apply(lambda v: _fmt(v, 2))
    display["hist_pct"] = display["hist_pct"].apply(lambda v: _fmt(v, 2))
    display["tie_pct"] = display["tie_pct"].apply(lambda v: _fmt(v, 2))
    print(display.to_string())


def load_baseline_consistency_row(model_name: str) -> Optional[pd.DataFrame]:
    if not _BASELINE_CSV_PATH.exists():
        return None
    baseline_df = pd.read_csv(_BASELINE_CSV_PATH)
    match = baseline_df[
        (baseline_df["model"] == model_name) & (baseline_df["layer"] == LAYER.value)
    ]
    return match if not match.empty else None


def print_consistency_check(model_label: str, baseline_model_name: str, computed: dict) -> None:
    print_section(f"CONSISTENCY CHECK vs baseline_results.csv ({model_label})")
    row = load_baseline_consistency_row(baseline_model_name)
    if row is None:
        print(f"No matching row found in {_BASELINE_CSV_PATH} for '{baseline_model_name}'.")
        return
    r = row.iloc[0]
    print(
        f"baseline_results.csv -> test_mae={_fmt(r.get('test_mae'))}, "
        f"test_rmse={_fmt(r.get('test_rmse'))}, test_r2={_fmt(r.get('test_r2'))}, "
        f"ood_mae={_fmt(r.get('ood_mae'))}, ood_rmse={_fmt(r.get('ood_rmse'))}, "
        f"ood_r2={_fmt(r.get('ood_r2'))}"
    )
    print("(compare against the standalone RF / Hist rows in the metrics table above)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_split(split_name: str, split: SplitData, rf_model, hist_model) -> None:
    print_section(f"{split_name.upper()} SPLIT")

    # Both models score every row independently; no interaction between
    # them influences either predict() call.
    rf_preds = rf_model.predict(split.X)
    hist_preds = hist_model.predict(split.X)

    df, signal_col = build_row_level_df(split, rf_preds, hist_preds)

    # ---- Overall metrics: RF, Hist, and per-row better-model selection ----
    rf_metrics = compute_all_metrics(df["true_queue_length_m"], df["rf_prediction"])
    hist_metrics = compute_all_metrics(df["true_queue_length_m"], df["hist_prediction"])
    better_preds = better_model_predictions(df)
    better_metrics = compute_all_metrics(df["true_queue_length_m"], better_preds)
    print_overall_metrics(f"{split_name.upper()} -- OVERALL METRICS", rf_metrics, hist_metrics, better_metrics)

    # ---- Win / loss / tie ----
    overall_wlt = win_loss_tie_summary(df)
    print_win_loss_tie(f"{split_name.upper()} -- WIN / LOSS / TIE (overall)", overall_wlt)

    # ---- Cross-tabulations against observable conditions ----
    if CONGESTION_COLUMN in df.columns:
        ct = crosstab_winner(df, CONGESTION_COLUMN)
        print_crosstab(f"{split_name.upper()} -- WINNER BY {CONGESTION_COLUMN}", ct)
    else:
        print_section(f"{split_name.upper()} -- WINNER BY {CONGESTION_COLUMN}")
        print(f"{CONGESTION_COLUMN} not present in feature columns -- skipped.")

    df_vqbin, visible_bin_edges = add_visible_queue_bin(df)
    if visible_bin_edges is not None:
        ct = crosstab_winner(df_vqbin, "visible_queue_bin")
        print_crosstab(f"{split_name.upper()} -- WINNER BY VISIBLE QUEUE LENGTH (quartile bins)", ct)
        print(f"Quartile bin edges ({VISIBLE_QUEUE_COLUMN}): {[round(e, 2) for e in visible_bin_edges]}")
    else:
        print_section(f"{split_name.upper()} -- WINNER BY VISIBLE QUEUE LENGTH")
        print(f"{VISIBLE_QUEUE_COLUMN} not present or insufficiently distinct for quartile bins -- skipped.")

    ct = crosstab_winner(df, APPROACH_COLUMN)
    print_crosstab(f"{split_name.upper()} -- WINNER BY {APPROACH_COLUMN}", ct)

    if signal_col is not None:
        ct = crosstab_winner(df, signal_col)
        print_crosstab(f"{split_name.upper()} -- WINNER BY SIGNAL STATE ({signal_col})", ct)
    else:
        print_section(f"{split_name.upper()} -- WINNER BY SIGNAL STATE")
        print(f"No signal-state column found among candidates {SIGNAL_STATE_CANDIDATES} -- skipped.")

    ct = crosstab_winner(df, SCENARIO_COLUMN)
    print_crosstab(f"{split_name.upper()} -- WINNER BY {SCENARIO_COLUMN}", ct)

    # ---- Descriptive-only: true_queue_length_m bins ----
    df_tqbin = add_true_queue_bin(df)
    ct = crosstab_winner(df_tqbin, "true_queue_bin")
    print_crosstab(
        f"{split_name.upper()} -- WINNER BY TRUE QUEUE LENGTH (descriptive only, not a routing input)",
        ct,
    )

    return rf_metrics, hist_metrics


def main() -> None:
    title = "LAYER 2 MODEL DISAGREEMENT ANALYSIS (Random Forest vs HistGradientBoosting)"
    print(title)
    print("=" * len(title))

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

    splits = load_splits()

    test_rf_metrics, test_hist_metrics = analyze_split("test", splits["test"], rf_model, hist_model)
    analyze_split("ood", splits["ood"], rf_model, hist_model)

    print_consistency_check("Random Forest", RF_BASELINE_MODEL_NAME, test_rf_metrics)
    print_consistency_check("HistGradientBoosting", HIST_BASELINE_MODEL_NAME, test_hist_metrics)

    print_section("NOTE")
    print(
        "true_queue_length_m is used only to compute each model's absolute error and\n"
        "to label the per-row winner, both strictly for reporting. The cross-tabulations\n"
        "above show whether any single observable condition lines up consistently with\n"
        "one model winning; this script draws no conclusion and builds no routing rule --\n"
        "that judgment, and any decision to act on it, is left to the reader."
    )


if __name__ == "__main__":
    main()