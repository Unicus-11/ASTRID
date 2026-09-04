# HistGradientBoosting Hyperparameter Tuning and Final Selection

## 1. Objective

After establishing the seven baseline regression models for ASTRID Layer 2, HistGradientBoosting was selected for further hyperparameter investigation because it produced the strongest standalone baseline performance.

The tuning experiment was conducted using the `layer2_p11` dataset with GPS penetration fixed at 11%. GPS penetration was intentionally held constant during model selection so that hyperparameter tuning measured model behavior rather than changes in sensing availability.

Hyperparameter selection used the TRAIN split for model fitting and the VALIDATION split for candidate selection. TEST and OOD were not used to select hyperparameters during the tuning rounds.

## 2. Tuning Methodology

The tuning process was conducted in five progressively narrower rounds.

The hyperparameters investigated were:

- `learning_rate`
- `max_iter`
- `max_leaf_nodes`
- `min_samples_leaf`
- `l2_regularization`
- `max_depth`

`early_stopping=False` and `random_state=42` were kept fixed throughout the valid tuning rounds.

The selection rule was defined before the final tuning decision:

1. Select the configuration with the lowest validation MAE.
2. Treat configurations within 1% of the best validation MAE as a tie group.
3. Within that group, select the lowest validation RMSE.
4. If necessary, use the highest validation R².
5. If still necessary, prefer the simpler configuration.

The tuning process was validation-driven and did not use TEST or OOD to adjust hyperparameters.

## 3. Important Implementation Correction

During Round 5, an implementation issue was discovered.

The project `HistGradientBoostingModel` wrapper did not expose or forward the `early_stopping` parameter to `sklearn.ensemble.HistGradientBoostingRegressor`. This could cause scikit-learn's automatic early-stopping behavior to be used unintentionally.

The affected initial Round 5 execution was therefore treated as invalid and was not used for any model-selection decision.

Round 5 was rerun using `HistGradientBoostingRegressor` directly with:

```text
early_stopping=False
random_state=42
```

The corrected Round 5 results reproduced the earlier valid reference configurations and were used for the final tuning decision.

## 4. Round 1

The first tuning round evaluated 40 configurations.

The strongest configuration was Trial 26:

```text
learning_rate      = 0.0341917
max_iter           = 270
max_leaf_nodes     = 43
min_samples_leaf   = 40
l2_regularization  = 0.942854
max_depth          = 10
early_stopping     = False
```

Validation performance:

```text
MAE  = 5.075884
RMSE = 20.365645
R²   = 0.972108
```

This improved validation MAE relative to the baseline-control configuration, but its subsequent TEST/OOD performance did not outperform the original baseline.

## 5. Rounds 2–4

Further validation-only searches explored a narrower region around strong candidates from Round 1.

The most important configurations emerging from these rounds were:

### Trial 13

```text
learning_rate      = 0.058092
max_iter           = 151
max_leaf_nodes     = 54
min_samples_leaf   = 25
l2_regularization  = 0.470324
max_depth          = 8
```

Validation:

```text
MAE  = 4.984759
RMSE = 20.462688
R²   = 0.971842
```

### Trial 7

```text
learning_rate      = 0.025034
max_iter           = 369
max_leaf_nodes     = 52
min_samples_leaf   = 35
l2_regularization  = 0.502603
max_depth          = 10
```

Validation:

```text
MAE  = 4.986771
RMSE = 20.229983
R²   = 0.972479
```

### Trial 39

```text
learning_rate      = 0.029409
max_iter           = 279
max_leaf_nodes     = 45
min_samples_leaf   = 31
l2_regularization  = 0.782118
max_depth          = 10
```

Validation:

```text
MAE  = 4.992469
RMSE = 20.343517
R²   = 0.972169
```

Although Trial 13 had slightly lower MAE than Trial 7, the difference was within the predefined 1% MAE tie band. Trial 7 therefore won the tie-break because it had the best validation RMSE and R² among the qualifying configurations.

## 6. Round 5

Round 5 was deliberately small and focused on local perturbations around the strongest configurations.

The lowest validation MAE obtained in Round 5 was:

```text
Trial 8
learning_rate      = 0.022500
max_iter           = 400
max_leaf_nodes     = 52
min_samples_leaf   = 36
l2_regularization  = 0.480000
max_depth          = 9
```

Validation:

```text
MAE  = 4.946963
RMSE = 20.344454
R²   = 0.972166
```

This was the lowest validation MAE observed during the tuning process.

However, Trial 8 fell within the 1% MAE tie band around the best candidate. Trial 7 remained the formal winner under the predefined selection rule because it had:

```text
RMSE = 20.229983
R²   = 0.972479
```

which were the strongest values within the qualifying MAE group.

Therefore, Trial 7 was selected as the frozen tuned configuration for held-out evaluation.

## 7. Final Trial 7 Held-Out Evaluation

Trial 7 was then trained on TRAIN only and evaluated on TEST and OOD.

The selected configuration was:

```text
learning_rate      = 0.025034
max_iter           = 369
max_leaf_nodes     = 52
min_samples_leaf   = 35
l2_regularization  = 0.502603
max_depth          = 10
early_stopping     = False
random_state       = 42
```

### TEST

Original HistGradientBoosting baseline:

```text
MAE  = 18.706834
RMSE = 44.630782
R²   = 0.939220
```

Trial 7:

```text
MAE  = 19.152881
RMSE = 46.303357
R²   = 0.934579
```

Relative to the original baseline:

```text
MAE  = 2.38% worse
RMSE = 3.75% worse
R²   = 0.004641 lower
```

Therefore, Trial 7 did not improve held-out TEST performance.

### OOD

Original HistGradientBoosting baseline:

```text
MAE  = 34.757319
RMSE = 63.107135
R²   = 0.903846
```

Trial 7:

```text
MAE  = 34.302741
RMSE = 63.247887
R²   = 0.903417
```

Relative to the original baseline:

```text
MAE  = 1.31% better
RMSE = 0.22% worse
R²   = 0.000429 lower
```

The OOD result was therefore mixed rather than a clear improvement.

## 8. Verification of the Dataset Split

The scenario-level split was explicitly verified before interpreting the held-out results.

```text
TRAIN
scenario_high_demand
scenario_left_turn_heavy
scenario_low_demand
scenario_normal_balanced

VALIDATION
scenario_north_heavy
scenario_straight_heavy

TEST
scenario_east_west_heavy
scenario_south_heavy

OOD
scenario_burst_demand_OOD
scenario_heavy_vehicle_OOD
scenario_north_extreme_OOD
scenario_very_high_demand_OOD
```

The row counts were:

```text
TRAIN       = 11,536
VALIDATION  =  5,768
TEST        =  5,768
OOD         = 11,536
```

The matching row counts are explained by the equal number of rows contributed by each scenario rather than by an accidental split duplication.

## 9. Scenario-Level Analysis

To determine why the validation improvement did not transfer to TEST/OOD, Trial 7 was compared against the exact original baseline artifact separately for every TEST and OOD scenario.

The original baseline artifact was loaded directly rather than reconstructed or retrained.

The aggregate consistency check reproduced the official baseline values exactly, confirming that the scenario-level comparison was performed against the correct baseline.

### TEST scenarios

For `scenario_east_west_heavy`:

```text
MAE change  = -2.55%
RMSE change = -3.87%
R² change   = -0.004575
```

For `scenario_south_heavy`:

```text
MAE change  = -2.20%
RMSE change = -3.64%
R² change   = -0.004860
```

Trial 7 therefore degraded performance on both TEST scenarios.

### OOD scenarios

For `scenario_burst_demand_OOD`:

```text
MAE change  = -1.02%
RMSE change = -1.19%
R² change   = -0.004075
```

For `scenario_heavy_vehicle_OOD`:

```text
MAE change  = +4.68%
RMSE change = +3.21%
R² change   = +0.009113
```

For `scenario_north_extreme_OOD`:

```text
MAE change  = +4.60%
RMSE change = +3.98%
R² change   = +0.001717
```

For `scenario_very_high_demand_OOD`:

```text
MAE change  = -0.02%
RMSE change = -2.32%
R² change   = -0.007236
```

The OOD behavior was therefore mixed: Trial 7 improved performance for heavy-vehicle and north-extreme conditions, but degraded performance for burst-demand and very-high-demand conditions.

## 10. Interpretation

The results provide evidence of a distribution mismatch between validation and held-out scenarios.

Validation contains:

```text
north_heavy
straight_heavy
```

while TEST contains:

```text
east_west_heavy
south_heavy
```

and OOD contains deliberately different stress regimes:

```text
burst_demand_OOD
very_high_demand_OOD
north_extreme_OOD
heavy_vehicle_OOD
```

Consequently, the validation-selected Trial 7 configuration achieved a lower validation MAE but did not generalize better to the TEST scenarios.

However, the results do not support the stronger claim that "heavier traffic causes Trial 7 to fail." Trial 7 actually improved considerably on the `heavy_vehicle_OOD` and `north_extreme_OOD` scenarios.

The more defensible interpretation is:

> The validation-selected hyperparameters improved performance for the validation distribution but did not provide a consistent improvement across different held-out traffic regimes. This indicates sensitivity to scenario distribution rather than a uniformly better model.

## 11. Final Model Decision

Hyperparameter tuning is considered complete for the current HistGradientBoosting experiment.

Trial 7 remains the **winner of validation-based tuning**, but it is **not selected over the original baseline as the final model** because it failed to improve TEST performance and did not provide a consistent improvement on OOD.

The original HistGradientBoosting baseline remains the stronger generalizing model for the current experiment.

This conclusion is based on:

- verified scenario-level train/validation/test/OOD separation;
- exact reproduction of the original baseline artifact's recorded TEST/OOD metrics;
- validation-only hyperparameter selection;
- held-out TEST evaluation showing consistent degradation for Trial 7;
- OOD evaluation showing mixed rather than uniformly improved performance.

Further rounds of hyperparameter tuning will not be performed merely to optimize against the already observed TEST results, because doing so would progressively turn TEST into another tuning set.

## 12. Next Stage

The model-selection stage is now frozen.

The selected model for the next ASTRID experiment is the **original HistGradientBoosting baseline**.

The planned GPS-penetration sensitivity experiment should be conducted separately after model selection. The selected model should remain fixed while GPS penetration is varied to study how sensing availability affects queue-estimation performance.