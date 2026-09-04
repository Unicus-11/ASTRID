"""
baseline_comparison.py
=========================
Read-only analysis script for the existing baseline results CSV.

This script performs ANALYSIS ONLY:
    * it reads models/results/baseline_results.csv (already produced by
      collect_baseline_results.py via the existing run_experiment() /
      results_to_dataframe() infrastructure)
    * it does not rerun, retrain, or load any model
    * it does not touch experiment_runner.py, evaluate.py, metrics.py,
      any model implementation, or any dataset file
    * it does not write baseline_results.csv, and does not create any
      new CSV/JSON output

All numbers printed below come directly from the CSV at run time -- no
metric value is hard-coded in this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Resolve the CSV path relative to this file, so the script works
# regardless of the caller's current working directory.
_RESULTS_DIR = Path(__file__).resolve().parent
_CSV_PATH = _RESULTS_DIR / "baseline_results.csv"

_LAYER_DISPLAY_NAME = {
    "layer1": "LAYER 1",
    "layer2_p11": "LAYER 2 (p11)",
}

# Columns shown per-model, per-layer.
_METRIC_COLUMNS = [
    "validation_mae", "validation_rmse", "validation_r2",
    "test_mae", "test_rmse", "test_r2",
    "ood_mae", "ood_rmse", "ood_r2",
]

# (label, column, "min" or "max") -- defines every "best X" reported.
_BEST_METRIC_SPECS = [
    ("validation MAE", "validation_mae", "min"),
    ("validation R2", "validation_r2", "max"),
    ("test MAE", "test_mae", "min"),
    ("test RMSE", "test_rmse", "min"),
    ("test R2", "test_r2", "max"),
    ("OOD MAE", "ood_mae", "min"),
    ("OOD RMSE", "ood_rmse", "min"),
    ("OOD R2", "ood_r2", "max"),
]

# (label, metric_stem) pairs used in the Layer1 -> Layer2 improvement table.
# MAE/RMSE use percentage reduction; R2 uses absolute improvement.
_IMPROVEMENT_SPECS = [
    ("test_mae", "pct_reduction"),
    ("test_rmse", "pct_reduction"),
    ("test_r2", "abs_improvement"),
    ("ood_mae", "pct_reduction"),
    ("ood_rmse", "pct_reduction"),
    ("ood_r2", "abs_improvement"),
]


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"baseline_results.csv not found at {csv_path}. "
            f"Run collect_baseline_results.py first."
        )
    return pd.read_csv(csv_path)


def _fmt(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def print_layer_section(df: pd.DataFrame, layer: str) -> None:
    layer_df = df[df["layer"] == layer].copy()
    display_name = _LAYER_DISPLAY_NAME.get(layer, layer.upper())

    print(display_name)
    print("-" * len(display_name))

    if layer_df.empty:
        print("(no results found for this layer)\n")
        return

    table = layer_df[["model"] + _METRIC_COLUMNS].copy()
    for col in _METRIC_COLUMNS:
        table[col] = table[col].apply(_fmt)
    table = table.rename(columns={"model": "model"}).set_index("model")
    print(table.to_string())
    print()

    print("Best metrics:")
    for label, col, direction in _BEST_METRIC_SPECS:
        if col not in layer_df.columns or layer_df[col].dropna().empty:
            print(f"  Best {label}: NA")
            continue
        if direction == "min":
            idx = layer_df[col].idxmin()
        else:
            idx = layer_df[col].idxmax()
        best_row = layer_df.loc[idx]
        print(f"  Best {label}: {best_row['model']} ({_fmt(best_row[col])})")
    print()


def print_improvement_section(df: pd.DataFrame) -> None:
    print("LAYER 1 -> LAYER 2 IMPROVEMENT")
    print("-" * len("LAYER 1 -> LAYER 2 IMPROVEMENT"))

    l1 = df[df["layer"] == "layer1"].set_index("model")
    l2 = df[df["layer"] == "layer2_p11"].set_index("model")

    common_models = sorted(set(l1.index) & set(l2.index))
    if not common_models:
        print("(no overlapping models between layer1 and layer2_p11)\n")
        return

    rows = []
    for model in common_models:
        row = {"model": model}
        for metric_col, kind in _IMPROVEMENT_SPECS:
            v1 = l1.loc[model, metric_col]
            v2 = l2.loc[model, metric_col]
            if pd.isna(v1) or pd.isna(v2):
                row[metric_col] = None
                continue
            if kind == "pct_reduction":
                if v1 == 0:
                    row[metric_col] = None
                else:
                    row[metric_col] = (v1 - v2) / v1 * 100.0
            else:  # abs_improvement
                row[metric_col] = v2 - v1
        rows.append(row)

    improvement_df = pd.DataFrame(rows).set_index("model")
    display_df = improvement_df.copy()
    for metric_col, kind in _IMPROVEMENT_SPECS:
        suffix = "% reduction" if kind == "pct_reduction" else " abs change"
        display_df[metric_col] = improvement_df[metric_col].apply(
            lambda v, s=suffix: "NA" if pd.isna(v) else f"{v:+.2f}{s}"
        )
    print(display_df.to_string())
    print()
    print(
        "Note: for MAE/RMSE, positive values are a reduction in error "
        "(Layer 2 better than Layer 1). For R2, positive values are an "
        "absolute increase in R2 (Layer 2 better than Layer 1)."
    )
    print()


def print_model_selection_summary(df: pd.DataFrame) -> None:
    print("MODEL SELECTION SUMMARY")
    print("-" * len("MODEL SELECTION SUMMARY"))
    print(
        "Validation is the primary model-selection set. Test is the final\n"
        "held-out evaluation, reported alongside validation but not used to\n"
        "pick a model here. OOD is robustness/generalization evidence only\n"
        "and must never influence model selection.\n"
    )

    for layer in ["layer1", "layer2_p11"]:
        layer_df = df[df["layer"] == layer]
        if layer_df.empty or layer_df["validation_mae"].dropna().empty:
            continue
        display_name = _LAYER_DISPLAY_NAME.get(layer, layer.upper())
        leader_idx = layer_df["validation_mae"].idxmin()
        leader = layer_df.loc[leader_idx]

        print(f"{display_name}: validation-leading model = {leader['model']} "
              f"(validation MAE = {_fmt(leader['validation_mae'])}, "
              f"validation R2 = {_fmt(leader['validation_r2'])})")

        # Show how the validation leader compares to everyone else on
        # test and OOD, without re-selecting a "winner" from those sets.
        compare_cols = ["model", "validation_mae", "test_mae", "test_r2", "ood_mae", "ood_r2"]
        compare_table = layer_df[compare_cols].copy()
        for col in compare_cols[1:]:
            compare_table[col] = compare_table[col].apply(_fmt)
        compare_table = compare_table.set_index("model")
        print(compare_table.to_string())

        best_test_mae_idx = layer_df["test_mae"].idxmin()
        best_test_mae_model = layer_df.loc[best_test_mae_idx, "model"]
        best_ood_mae_idx = layer_df["ood_mae"].idxmin()
        best_ood_mae_model = layer_df.loc[best_ood_mae_idx, "model"]

        if best_test_mae_model != leader["model"]:
            print(
                f"  Note: {best_test_mae_model} has a better test MAE than "
                f"the validation leader ({leader['model']}) on {display_name}."
            )
        if best_ood_mae_model != leader["model"]:
            print(
                f"  Note: {best_ood_mae_model} has a better OOD MAE than "
                f"the validation leader ({leader['model']}) on {display_name} "
                f"-- shown for robustness context only, not used for selection."
            )
        print()


def print_overall_best_metrics(df: pd.DataFrame) -> None:
    print("OVERALL BEST METRICS (per layer)")
    print("-" * len("OVERALL BEST METRICS (per layer)"))
    for layer in ["layer1", "layer2_p11"]:
        layer_df = df[df["layer"] == layer]
        if layer_df.empty:
            continue
        display_name = _LAYER_DISPLAY_NAME.get(layer, layer.upper())
        print(f"{display_name}:")
        for label, col, direction in _BEST_METRIC_SPECS:
            if col not in layer_df.columns or layer_df[col].dropna().empty:
                print(f"  Best {label}: NA")
                continue
            idx = layer_df[col].idxmin() if direction == "min" else layer_df[col].idxmax()
            best_row = layer_df.loc[idx]
            print(f"  Best {label}: {best_row['model']} ({_fmt(best_row[col])})")
        print()


def main() -> None:
    print(f"Reading baseline results from: {_CSV_PATH}\n")
    df = load_results(_CSV_PATH)

    print("BASELINE MODEL COMPARISON")
    print("=" * len("BASELINE MODEL COMPARISON"))
    print()

    for layer in ["layer1", "layer2_p11"]:
        print_layer_section(df, layer)

    print_improvement_section(df)
    print_model_selection_summary(df)
    print_overall_best_metrics(df)


if __name__ == "__main__":
    main()