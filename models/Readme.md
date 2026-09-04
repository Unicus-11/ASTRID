# ASTRID Modeling Infrastructure

This directory contains the **foundational, model-agnostic infrastructure**
for the modeling stage of the ASTRID project. It does not contain any
trained model implementation (no Random Forest, XGBoost, CNN, etc.) --
only the shared plumbing every future model will use in the same way.

## Files

| File                     | Purpose                                                            |
|---------------------------|---------------------------------------------------------------------|
| `data_loader.py`         | Manifest-driven loading of features (`X`) and target (`y`)         |
| `metrics.py`             | Shared regression metrics (MAE, RMSE, R²)                          |
| `persistence.py`         | Model-agnostic save/load via `joblib`                              |
| `experiment_config.py`   | Central `Layer` / `Split` / target / `ExperimentConfig` definitions |
| `train.py`               | Common training interface (`RegressionModel` protocol)             |
| `evaluate.py`            | Common evaluation interface across validation/test/OOD             |

## The prediction task

`true_queue_length_m` is the single regression target for every model
built in this directory. It is defined once, centrally, in
`experiment_config.TARGET_COLUMN`, and every other module imports it from
there rather than hard-coding the string.

## What the data loader does

`data_loader.py` never re-derives feature/label/metadata membership. For
every `(layer, split)` pair it reads the corresponding `manifest.json`
(written by `dataset/assemble_dataset.py`) and treats its
`feature_columns`, `label_columns`, and `metadata_columns` lists as the
**sole source of truth**.

Loading a split returns a `SplitData` object with four *separate* pieces:

- `X` — the feature matrix, containing **only** the manifest's declared
  `feature_columns`.
- `y` — the target column (`true_queue_length_m`) as a `Series` aligned
  to `X`.
- `metadata` — columns like `scenario_id`, `split`, `design_method`,
  useful for tracing rows back to their scenario, but never usable as a
  model input.
- `keys` — the `timestamp` / `approach_edge` key columns, similarly kept
  out of `X`.

Because `X` is built by explicitly selecting only the declared feature
columns, and the loader raises an error if the manifest ever declares a
column as both a feature and a label/metadata/key column, it is
structurally difficult for a future model implementation to accidentally
train on a forbidden column — there is no "give me everything" method
that returns labels or metadata mixed in with features.

The loader also does not touch the data itself: no imputation, no dtype
coercion beyond what `pandas.read_csv` does by default, no re-splitting,
and no modification of the CSVs on disk. Missing values are preserved
exactly as they were assembled.

## Train / validation / test / OOD

- **train** — used to fit a model.
- **validation** — used for model selection and any tuning.
- **test** — a held-out set for a single, final reported result per
  model. Not used during tuning.
- **ood** (out-of-distribution) — reserved **exclusively** for final
  robustness evaluation, after a model has already been selected using
  validation/test. `evaluate.py`'s `EvaluationReport` keeps `ood` in a
  clearly separate field from `validation`/`test`, and
  `EvaluationReport.selection_metrics()` deliberately excludes it, so
  there is no easy path for OOD results to leak into model selection.

Splits are **scenario-level and already fixed** by the dataset assembly
pipeline (`dataset/assemble_dataset.py`, driven by each scenario's
`scenario.json["split"]` field, itself written by `scenario_builder.py`).
No code in `models/` creates, changes, or re-derives a train/test split —
`data_loader.py` only reads whichever split's CSV/manifest already
exists.

## How features are selected

Feature selection is not a modeling-time decision here — it is inherited
entirely from the manifest produced by dataset assembly. `layer1` and
`layer2_p11` have different feature sets (Layer 2 adds GPS-probe-derived
and traffic-signal-state features on top of Layer 1's camera-only
features); `data_loader.get_feature_columns(layer, split)` reports
exactly which columns will be used for a given layer/split without
having to load the full dataframe.

## How future models plug in

`experiment_config.py` defines the shared vocabulary:

- `Layer` — `LAYER1` / `LAYER2_P11`
- `Split` — `TRAIN` / `VALIDATION` / `TEST` / `OOD`
- `TARGET_COLUMN` — `"true_queue_length_m"`
- `ExperimentConfig` — layer, target column, data/output paths, a shared
  random seed convention, and an experiment name. Contains **no**
  model-specific hyperparameters.

`train.py` defines a minimal `RegressionModel` protocol (`fit(X, y)`,
`predict(X)`), matching the standard scikit-learn estimator interface
already implemented by scikit-learn's own models, XGBoost, LightGBM, and
CatBoost's scikit-learn wrappers. A CNN implementation would only need a
thin wrapper exposing the same two methods. `build_training_inputs()`
turns a loaded training `SplitData` plus an `ExperimentConfig` into a
`TrainingInputs` object, and `train_model()` fits any conforming model on
those inputs and optionally persists it via `persistence.save_model()`.

`evaluate.py` defines `evaluate_model()`, which takes any trained,
conforming model plus its validation/test/OOD `SplitData` and returns an
`EvaluationReport` computed entirely through `metrics.compute_all_metrics()`.

## All models must use the same data and evaluation protocol

Every future model implementation is expected to:

1. Load its data exclusively through `data_loader.DatasetLoader` /
   `load_split()` — never read the assembled CSVs directly.
2. Accept an `ExperimentConfig` for anything that must stay consistent
   across models (layer, target column, paths, seed convention).
3. Conform to the `RegressionModel` protocol in `train.py` and be trained
   through `train_model()`.
4. Be evaluated exclusively through `evaluate.py`'s `evaluate_model()`,
   using `metrics.py` for every reported number.
5. Persist trained artifacts exclusively through `persistence.py`.

This keeps data loading, metrics, and evaluation protocol identical
across Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, and CNN
implementations, so that differences in reported results reflect real
differences between models rather than inconsistencies in how each one
was fed data or scored.

## Explicitly out of scope for this infrastructure

- No model implementations (Random Forest, etc.)
- No hyperparameter search
- No model comparison/selection logic
- No changes to the assembled dataset or its splits
- No dataset regeneration