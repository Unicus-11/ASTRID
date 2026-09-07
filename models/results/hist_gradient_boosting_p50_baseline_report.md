# HistGradientBoosting -- p50-trained baseline (training-distribution mismatch probe)

Generated: 2026-09-04T20:45:59.081343+00:00

## Purpose
Separate experiment from the GPS penetration sensitivity analysis. Trains a new HistGradientBoosting model on p50 TRAIN only (original baseline configuration, unchanged) and evaluates it on p50 VALIDATION/TEST/OOD, to check whether the frozen p11-trained model's weaker p50 result is a training-distribution mismatch effect. Does not change the project's final model selection -- the original p11 HGB remains selected regardless of this experiment's outcome.

## Dataset
- Path: `C:\Users\DISHA\SIH\ASTRID\dataset\assembled\layer2_p50`
- n_features (verified against p11 schema): 23

## Scenario split verification
- **TRAIN**: ['scenario_high_demand', 'scenario_left_turn_heavy', 'scenario_low_demand', 'scenario_normal_balanced']
- **VALIDATION**: ['scenario_north_heavy', 'scenario_straight_heavy']
- **TEST**: ['scenario_east_west_heavy', 'scenario_south_heavy']
- **OOD**: ['scenario_burst_demand_OOD', 'scenario_heavy_vehicle_OOD', 'scenario_north_extreme_OOD', 'scenario_very_high_demand_OOD']

## Model configuration (original baseline, unchanged; only training data is p50)
```python
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 31
min_samples_leaf = 20
l2_regularization = 0.0
max_depth = None
early_stopping = False
random_state = 42
```

## Artifact
- Saved to: `C:\Users\DISHA\SIH\ASTRID\models\artifacts\layer2_p50\hist_gradient_boosting_layer2_p50_baseline\hist_gradient_boosting.joblib`
- p11 baseline artifact (referenced only, never modified): `C:\Users\DISHA\SIH\ASTRID\models\artifacts\layer2_p11\hist_gradient_boosting_layer2_p11_baseline\hist_gradient_boosting.joblib`

## Metrics -- p50-trained model
| split | mae | rmse | r2 | n |
|---|---:|---:|---:|---:|
| VALIDATION | 4.272012 | 18.394906 | 0.977245 | 5768 |
| TEST | 7.033426 | 16.930291 | 0.991254 | 5768 |
| OOD | 21.293812 | 41.514141 | 0.958390 | 11536 |

## Comparison -- A) frozen p11-trained model on p50, vs B) p50-trained model on p50
| split | A: p11-trained MAE | B: p50-trained MAE | MAE change % | A: p11-trained RMSE | B: p50-trained RMSE | RMSE change % | A: p11-trained R2 | B: p50-trained R2 | R2 change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TEST | 16.520118 | 7.033426 | +57.43% | 39.883303 | 16.930291 | +57.55% | 0.951463 | 0.991254 | +0.039791 |
| OOD | 28.930702 | 21.293812 | +26.40% | 57.671497 | 41.514141 | +28.02% | 0.919697 | 0.958390 | +0.038693 |

Positive % / positive R2 change = the p50-trained model is better than the frozen p11-trained model, when both are evaluated on p50.

## Notes
- No hyperparameter tuning was performed.
- TEST/OOD were used only for final reporting, never for selection.
- The p11 baseline artifact was never opened, loaded, or written by this script.
- This experiment does not change the project's current final-model selection; the original p11 HistGradientBoosting remains the selected final model.