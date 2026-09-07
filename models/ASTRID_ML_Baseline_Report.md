# ASTRID — Machine Learning Baseline Development and Model Selection

## 1. Overview

ASTRID estimates traffic queue length from simulated traffic observations with the eventual goal of providing a useful traffic-state estimate for downstream signal control.

The machine-learning target used in this stage is:

- **Target:** `true_queue_length_m`
- **Meaning:** ground-truth queue length in metres.

The objective of this stage was to establish a reliable baseline model, compare several machine-learning approaches, understand their error patterns, investigate whether combining models could improve performance, and select a final baseline model before hyperparameter tuning.

> **Important:** The results in this report are baseline results. Hyperparameter tuning has **not** yet been performed.

---

## 2. Feature Layers

Two feature configurations were evaluated.

### Layer 1

Layer 1 uses information that can be obtained primarily from the camera observation stream, together with past-only temporal features and occupancy information.

The temporal features use historical information only. For example, a 30-second change feature compares the current observation with the observation 30 seconds earlier:

`value(t) - value(t - 30 s)`

This avoids using future information when constructing the current-state estimate.

### Layer 2

Layer 2 extends Layer 1 with additional information:

- GPS/probe observations
- signal phase information
- phase elapsed time
- additional temporal probe features
- physics-derived traffic features

The purpose of Layer 2 is to determine whether combining camera information with probe, signal, and derived traffic-state information produces a more accurate queue estimate.

---

## 3. Dataset and Evaluation Design

The dataset contains multiple traffic scenarios representing different demand and directional conditions.

The scenarios include:

- normal balanced demand
- low demand
- high demand
- north-heavy demand
- south-heavy demand
- east/west-heavy demand
- straight-heavy demand
- left-turn-heavy demand
- burst-demand OOD
- very-high-demand OOD
- north-extreme OOD
- heavy-vehicle OOD

The data were assembled using **scenario-level splits** rather than random row-level splitting. This is important because rows from the same traffic scenario are temporally related. Randomly distributing rows from one scenario across training and testing could allow information from essentially the same traffic realization to appear in both sets.

The final split structure is:

- **TRAIN:** used to fit models
- **VALIDATION:** used for model comparison and selection
- **TEST:** held out for final in-distribution evaluation
- **OOD:** held out for robustness/generalization evaluation

### OOD meaning

**OOD** means **Out-of-Distribution**.

The OOD scenarios intentionally differ from the normal training conditions. They test whether a model can generalize to traffic conditions that were not represented in the ordinary training scenarios.

Examples include burst demand, very high demand, extreme directional demand, and heavy-vehicle conditions.

---

# 4. Seven Baseline Models

Seven machine-learning models were evaluated.

## 4.1 Random Forest

**Random Forest** is an ensemble of decision trees. Multiple trees independently make predictions, and their predictions are combined, typically by averaging for regression.

It is a strong and robust baseline for tabular data.

### Intuition

Instead of trusting one decision tree:

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤ → average → prediction
...     │
Tree N ─┘
```

The averaging process generally makes the prediction more stable than relying on one tree.

---

## 4.2 Extra Trees

**Extra Trees**, or Extremely Randomized Trees, is similar to Random Forest but introduces additional randomness when constructing the trees.

This reduces correlation between individual trees and can sometimes improve generalization.

In simple terms:

> Random Forest builds many different trees; Extra Trees deliberately makes the tree-building process even more random.

---

## 4.3 XGBoost

**XGBoost** is a gradient-boosting algorithm.

Instead of building independent trees and averaging them, gradient boosting builds trees sequentially. Each new tree attempts to correct errors made by the previous trees.

Conceptually:

```text
Initial prediction
       ↓
Tree 1 corrects errors
       ↓
Tree 2 corrects remaining errors
       ↓
Tree 3 corrects remaining errors
       ↓
...
       ↓
Final prediction
```

XGBoost is widely used for structured/tabular machine-learning problems.

---

## 4.4 LightGBM

**LightGBM** is another gradient-boosting framework designed for efficient tree construction.

It uses histogram-based techniques and is designed to train efficiently while maintaining strong predictive performance on tabular datasets.

The project uses the `lightgbm` package.

---

## 4.5 CatBoost

**CatBoost** is a gradient-boosting algorithm designed particularly to handle categorical information effectively.

Although the ASTRID feature set is primarily numerical, CatBoost was included as another established gradient-boosting baseline.

---

## 4.6 HistGradientBoosting

**HistGradientBoosting** is a histogram-based gradient-boosting algorithm available in scikit-learn.

Instead of considering every continuous feature value independently when constructing trees, continuous values are grouped into histogram bins.

This can make training efficient while retaining strong predictive performance.

It also supports missing numerical values natively, which is useful for ASTRID because sparse probe observations naturally produce missing values in some rows.

---

## 4.7 MLP

**MLP** stands for **Multi-Layer Perceptron**.

It is a basic feed-forward neural network. Unlike the tree-based models, it learns a set of interconnected numerical transformations through layers of neurons.

The MLP was included to determine whether a simple neural-network approach could outperform the tree-based baselines on the ASTRID tabular feature representation.

The MLP uses:

- median imputation
- feature standardization
- two hidden layers
- ReLU activation
- Adam optimization
- fixed random state

The preprocessing is fitted using training data only and persisted together with the model.

---

# 5. Evaluation Metrics

Several metrics were used to evaluate queue-length prediction.

## 5.1 MAE — Mean Absolute Error

**MAE** is the average absolute difference between the predicted and true queue lengths.

For example:

> MAE = 18.71 m

means that the model's predictions differ from the true queue length by approximately 18.71 metres on average.

Lower is better.

MAE is particularly easy to interpret because it remains in the same unit as the target: **metres**.

---

## 5.2 RMSE — Root Mean Squared Error

**RMSE** is similar to MAE, but it gives greater weight to large errors because the errors are squared before averaging.

This is useful for ASTRID because a very large queue-estimation error can be more consequential than several small errors.

Lower is better.

---

## 5.3 R² — Coefficient of Determination

**R²** measures how well the model explains variation in the target.

A simplified interpretation is:

- `R² = 1` → perfect prediction
- `R² = 0` → no improvement over predicting the mean
- `R² < 0` → worse than that simple mean-prediction baseline

Higher is better.

---

# 6. Layer 1 vs Layer 2

The addition of Layer 2 information substantially improved model performance.

For the eventual strongest baseline, HistGradientBoosting:

| Metric | Layer 1 | Layer 2 | Change |
|---|---:|---:|---:|
| Test MAE | 29.19 m | **18.71 m** | **35.90% reduction** |
| Test RMSE | 63.01 m | **44.63 m** | **29.17% reduction** |
| Test R² | 0.879 | **0.939** | +0.060 |
| OOD MAE | 64.51 m | **34.76 m** | **46.12% reduction** |
| OOD RMSE | 104.67 m | **63.11 m** | **39.71% reduction** |
| OOD R² | 0.735 | **0.904** | +0.168 |

This indicates that the additional probe/GPS, signal, temporal, and derived traffic-state information provides substantial predictive value.

Layer 2 was therefore selected as the feature configuration for the subsequent model comparison.

---

# 7. Layer 2 Baseline Model Comparison

The seven models were compared using the Layer 2 feature set.

| Model | Validation MAE ↓ | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ | OOD MAE ↓ | OOD R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **4.91** | 20.04 | 49.12 | 0.926 | 36.24 | 0.893 |
| Extra Trees | 5.61 | 20.26 | 48.27 | 0.929 | 38.38 | 0.892 |
| XGBoost | 6.22 | 20.56 | 48.32 | 0.929 | 38.41 | 0.892 |
| LightGBM | 5.78 | 20.09 | 47.68 | 0.931 | 36.90 | 0.897 |
| CatBoost | 7.52 | 20.59 | 47.03 | 0.933 | 38.20 | 0.897 |
| **HistGradientBoosting** | 5.28 | **18.71** | **44.63** | **0.939** | **34.76** | **0.904** |
| MLP | 10.67 | 22.39 | 47.51 | 0.931 | 39.81 | 0.892 |

### Interpretation

Random Forest achieved the best validation MAE.

However, validation performance alone was not used to declare the final model. The held-out TEST and OOD results provide a stronger basis for assessing generalization.

HistGradientBoosting achieved:

- the lowest TEST MAE
- the lowest TEST RMSE
- the highest TEST R²
- the lowest OOD MAE
- the highest OOD R²

Therefore, HistGradientBoosting was the strongest overall baseline model.

The MLP was valid but did not outperform the tree-based models. Its Layer 2 TEST R² of 0.9311 was respectable, but its TEST MAE and RMSE were worse than HistGradientBoosting, and its validation MAE was substantially worse.

No additional tuning was performed on the MLP because the objective at this stage was baseline comparison rather than optimization of every model.

---

# 8. Error Analysis

Overall metrics do not show where a model succeeds or fails. Therefore, additional error analysis was performed on HistGradientBoosting and Random Forest.

A major pattern emerged.

## 8.1 Short queues

For short queues, particularly queues below approximately 25 metres, Random Forest performed extremely well.

On the TEST split for the `0–25 m` true-queue group:

| Model | MAE |
|---|---:|
| Random Forest | **0.030 m** |
| HistGradientBoosting | 0.438 m |

This indicates that Random Forest was very effective when the queue was short and directly observable.

---

## 8.2 Large queues

Performance became much more difficult as the true queue length increased.

For HistGradientBoosting on TEST:

| True queue length | MAE |
|---|---:|
| 0–25 m | 0.44 m |
| 25–50 m | 2.31 m |
| 50–100 m | 8.88 m |
| 100–200 m | 23.63 m |
| 200–400 m | 47.08 m |
| >400 m | 73.34 m |

The model is therefore highly accurate for short queues but has substantially larger errors for long queues.

---

# 9. Camera-Edge Censoring

A key difficulty is represented by:

`queue_reaches_camera_edge`

When this value is `True`, the queue reaches the observable boundary of the camera.

This means that the camera cannot directly observe the complete queue.

### Censoring

**Censoring** means that the quantity of interest exists, but the sensor's observation range prevents direct observation of the entire quantity.

For example:

```text
Actual queue:
|---------------------------------------------->

Camera:
|--------------------|
                     ↑ observation boundary
```

If the queue extends beyond that boundary, the observed queue length is not necessarily the true queue length.

This explains why large queues and camera-edge cases are considerably harder for the ML models.

For HistGradientBoosting on TEST:

| Camera-edge condition | MAE |
|---|---:|
| `False` | 8.79 m |
| `True` | 62.09 m |

This is one of the most important error patterns identified in the project.

---

# 10. Why a Hybrid Model Was Investigated

The error analysis showed that the two strongest models had complementary behavior.

In simplified form:

```text
Short / clearly observable queue
            ↓
      Random Forest
      tends to be stronger

Queue reaches camera edge
            ↓
 HistGradientBoosting
      tends to be stronger
```

This suggested that instead of always using one model, ASTRID could potentially choose the model according to the observed traffic state.

## Hybrid model

A **hybrid model** uses multiple trained models and a routing rule that determines which model produces the prediction for a particular observation.

The idea was:

```text
Observation
    ↓
Routing condition
    ├── condition A → Random Forest
    └── condition B → HistGradientBoosting
```

---

# 11. Validation-Only Hybrid Investigation

Several simple routing rules were investigated using the **VALIDATION split only**.

The rules included conditions based on:

- camera-edge status
- signal state
- visible queue length thresholds
- approach
- combinations of observable traffic conditions

The simplest rule that improved upon both standalone models on validation was named:

## `congestion_flag`

The frozen rule was:

```text
queue_reaches_camera_edge == False
    → Random Forest

queue_reaches_camera_edge == True
    → HistGradientBoosting
```

On VALIDATION:

| Model / Rule | MAE |
|---|---:|
| Random Forest | 4.9125 m |
| HistGradientBoosting | 5.2766 m |
| **Hybrid: congestion_flag** | **4.7364 m** |

The hybrid therefore improved upon both standalone models on validation.

### Why validation was used

The routing rule was selected using validation data so that TEST and OOD could remain untouched.

This is important because using TEST performance to design the routing rule would make the TEST set part of the model-selection process.

The rule was therefore **frozen** after the validation investigation.

---

# 12. Final Hybrid Evaluation on TEST and OOD

After freezing `congestion_flag`, the rule was applied unchanged to the untouched TEST and OOD datasets.

No validation data were loaded by the final evaluation script.

No threshold was re-derived.

No model was retrained.

No recalibration was performed.

No routing rule was changed after seeing TEST/OOD results.

## TEST

| Metric | Random Forest | HistGradientBoosting | Hybrid |
|---|---:|---:|---:|
| MAE | 20.0419 | **18.7068** | 18.8554 |
| RMSE | 49.1248 | **44.6308** | 46.2776 |
| R² | 0.9264 | **0.9392** | 0.9347 |

The hybrid improved over Random Forest:

- MAE improvement: 1.1865 m
- RMSE improvement: 2.8473 m
- R² improvement: 0.0083

However, it did not outperform HistGradientBoosting:

- MAE: hybrid was 0.1485 m worse
- RMSE: hybrid was 1.6468 m worse
- R²: hybrid was 0.0046 lower

---

## OOD

| Metric | Random Forest | HistGradientBoosting | Hybrid |
|---|---:|---:|---:|
| MAE | 36.2399 | **34.7573** | 35.0366 |
| RMSE | 66.5836 | **63.1071** | 64.5130 |
| R² | 0.8930 | **0.9038** | 0.8995 |

Again, the hybrid improved over Random Forest but remained worse than HistGradientBoosting:

- MAE: hybrid was 0.2793 m worse
- RMSE: hybrid was 1.4058 m worse
- R²: hybrid was 0.0043 lower

---

# 13. Why the Hybrid Was Rejected

The hybrid experiment was not unsuccessful. It demonstrated that the models have complementary strengths and that a simple observable traffic-state rule can improve Random Forest.

However, the purpose of model selection is to choose the model that performs best on unseen data while avoiding unnecessary complexity.

The frozen hybrid did not outperform HistGradientBoosting on either untouched TEST or OOD.

Therefore, the hybrid routing architecture was not selected as the final model.

The decision is:

> **Use HistGradientBoosting as the final baseline model rather than adding a routing layer between Random Forest and HistGradientBoosting.**

This is preferable because it:

- performs better on TEST
- performs better on OOD
- avoids additional routing logic
- avoids maintaining two prediction models in the final baseline pipeline
- provides a simpler controller integration path

---

# 14. Final Baseline Model Selection

The final baseline model is:

## HistGradientBoosting — Layer 2

Target:

`true_queue_length_m`

Final baseline TEST performance:

- **MAE:** 18.7068 m
- **RMSE:** 44.6308 m
- **R²:** 0.9392

Final baseline OOD performance:

- **MAE:** 34.7573 m
- **RMSE:** 63.1071 m
- **R²:** 0.9038

The model was selected because it provided the strongest overall held-out performance among the seven baseline models.

It also performed better than the frozen hybrid on both TEST and OOD.

---

# 15. Important Interpretation of the Current Result

The selected HistGradientBoosting model should be described as the **final baseline model**, not yet the final optimized model.

The distinction is:

```text
Baseline model
    ↓
Fixed reasonable hyperparameters
    ↓
Model comparison
    ↓
HistGradientBoosting selected
    ↓
NEXT: hyperparameter tuning
    ↓
Potentially improved HistGradientBoosting
```

The current selection answers:

> Which of the seven baseline approaches is strongest under the current fixed configurations?

It does **not** yet answer:

> What is the best possible HistGradientBoosting configuration for this dataset?

That question belongs to the next stage.

---

# 16. Next Stage: Hyperparameter Tuning

The next machine-learning stage is focused hyperparameter optimization of HistGradientBoosting.

The tuning procedure should preserve the existing evaluation discipline:

- **TRAIN:** fit candidate configurations
- **VALIDATION:** select the best configuration
- **TEST:** remain untouched until final evaluation
- **OOD:** remain untouched until final robustness evaluation

The primary selection metric will be validation MAE, with RMSE and R² used as supporting metrics.

Potential parameters for focused tuning include:

- `learning_rate`
- `max_iter`
- `max_leaf_nodes`
- `min_samples_leaf`
- `l2_regularization`
- `max_depth`

The goal is not to perform unlimited search. The baseline comparison has already identified HistGradientBoosting as the strongest candidate, so tuning should focus computational effort on that model.

After tuning:

```text
Best validation configuration
          ↓
Final tuned model
          ↓
Untouched TEST evaluation
          ↓
Untouched OOD evaluation
          ↓
Final model for controller integration
```

---

# 17. Summary

The machine-learning baseline stage produced the following conclusions:

1. **Layer 2 substantially improves queue-length estimation** compared with Layer 1.
2. Seven baseline models were evaluated:
   - Random Forest
   - Extra Trees
   - XGBoost
   - LightGBM
   - CatBoost
   - HistGradientBoosting
   - MLP
3. **HistGradientBoosting produced the strongest overall TEST and OOD performance.**
4. Error analysis showed that short, directly observable queues are much easier to estimate than long queues that reach the camera boundary.
5. Random Forest was particularly strong for short queues.
6. HistGradientBoosting was stronger in difficult large-queue/camera-edge conditions.
7. A hybrid routing strategy named `congestion_flag` was therefore investigated.
8. `congestion_flag` was selected using VALIDATION data only.
9. The frozen hybrid improved over Random Forest on TEST and OOD.
10. However, the hybrid did not outperform HistGradientBoosting on either TEST or OOD.
11. **HistGradientBoosting was therefore retained as the final baseline model.**
12. **Hyperparameter tuning remains to be performed.**
13. Controller integration should follow the tuning stage.

---

## Current ML Pipeline Status

| Stage | Status |
|---|---|
| Dataset construction | Complete |
| Dataset QA | Complete |
| Train/validation/test/OOD assembly | Complete |
| Seven baseline models | Complete |
| Layer 1 vs Layer 2 comparison | Complete |
| Baseline model comparison | Complete |
| Error analysis | Complete |
| Model disagreement analysis | Complete |
| Validation-only hybrid investigation | Complete |
| Frozen hybrid TEST/OOD evaluation | Complete |
| Baseline model selection | **HistGradientBoosting** |
| Hyperparameter tuning | **Next** |
| Controller integration | Pending |
| Closed-loop SUMO evaluation | Pending |
