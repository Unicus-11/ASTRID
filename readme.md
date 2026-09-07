# ASTRID — Adaptive Signal Traffic Regulation via Intelligent Detection
## Master README (Scenario Generation → Sensors → ML Estimation → Control → Reinforcement Learning)

This document ties together the whole project, end to end, in the order
it was actually built. It is the single reference for what exists, why
each piece exists, and how data flows between pieces.

---

## 0. One-paragraph summary

ASTRID simulates one traffic-signal-controlled intersection in SUMO,
observes it through two physically-limited sensors (a range-limited
camera and a sparse GPS-probe population), estimates the true queue
length from those imperfect observations using machine learning, and
then controls the traffic signal using that estimate — progressing from
a fixed-time baseline, to a Webster-timing controller, to a
supervised-learning control policy, and finally to a reinforcement-
learning (PPO) agent that learns signal control by acting inside the
simulation and being rewarded for real traffic outcomes.

---

## 1. End-to-end architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 1. SCENARIO GENERATION (scenario_builder.py)                         │
│    12 hand-designed scenarios: 8 development + 4 OOD                 │
│    → scenario.json, flow.xml, vtype.xml per scenario                 │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. SUMO SIMULATION (sq.net.xml + flow.xml + vtype.xml → SUMO)        │
│    → raw_output/vehicle_trajectories.csv (per-vehicle, per-second)   │
│    → ground_truth/state_timeseries.csv  (TRUE queue/density/flow)    │
└───────────────┬───────────────────────────────────────┬──────────────┘
                ▼                                        ▼
┌───────────────────────────────┐      ┌─────────────────────────────────┐
│ 3a. CAMERA SENSOR              │      │ 3b. GPS SENSOR                  │
│  range-limited (150 m)         │      │ 11% penetration, count-limited  │
│  camera_timeseries.csv         │      │ gps_p11_timeseries.csv          │
└───────────────┬────────────────┘      └────────────────┬─────────────┘
                └──────────────────┬──────────────────────┘
                                   ▼
                 ┌─────────────────────────────────────┐
                 │ 4. OBSERVATION ASSEMBLY               │
                 │  assembled_observations_p11.csv       │
                 └───────────────────┬───────────────────┘
                                     ▼
                 ┌─────────────────────────────────────┐
                 │ 5. FEATURE ENGINEERING                │
                 │  Layer 1 (camera-only + history)      │
                 │  Layer 2 (+ GPS + signal + physics)   │
                 │  → features_layer2_p11.csv            │
                 │  → labels_layer2_p11.csv (true_queue)  │
                 └───────────────────┬───────────────────┘
                                     ▼
                 ┌─────────────────────────────────────┐
                 │ 6. DATASET ASSEMBLY (manifest.json)   │
                 │  train / validation / test / ood      │
                 └───────────────────┬───────────────────┘
                                     ▼
     ┌───────────────────────────────────────────────────────┐
     │ 7. SUPERVISED MODEL COMPARISON (7 models)              │
     │  RF, ExtraTrees, XGBoost, LightGBM, CatBoost,          │
     │  HistGradientBoosting, MLP                             │
     │  → HistGradientBoosting selected (Layer 2, p11)        │
     └───────────────────────────┬─────────────────────────────┘
                                 ▼
     ┌───────────────────────────────────────────────────────┐
     │ 8. FROZEN ONLINE QUEUE ESTIMATOR                        │
     │  online_hgb_queue_estimator.py (inference-only)         │
     │  raw sensors → estimated_queue_length_m, live, 5s tick  │
     └───────────────────┬─────────────────────────────────────┘
                         ▼
     ┌────────────────────────────────────────────────────────────┐
     │ 9. CONTROLLERS (compared on identical scenario+seed)         │
     │  a) Fixed-time (static program)                              │
     │  b) Webster-timing (one-shot, demand-derived)                │
     │  c) Random-Forest policy (KEEP/SWITCH from live estimate)    │
     │  d) PPO reinforcement-learning agent  ◄── current best        │
     └────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 1 — Scenario generation

**File:** `scenario_builder.py` (v0.4.0)

12 scenarios, **hand-authored**, each with a fixed, deterministic seed —
not randomly sampled. Split:

| Split | Count | Scenarios |
|---|---|---|
| train | 4 | normal_balanced, low_demand, high_demand, left_turn_heavy |
| validation | 2 | north_heavy, straight_heavy |
| test | 2 | south_heavy, east_west_heavy |
| ood | 4 | very_high_demand_OOD, north_extreme_OOD, burst_demand_OOD, heavy_vehicle_OOD |

**Key label:** `design_method` = `"development"` or `"ood"`. OOD
scenarios are *never* used for training, tuning, feature fitting, or
model/checkpoint selection — enforced at multiple layers (dataset
assembly, `EvaluationReport.selection_metrics()`, PPO's
`ValidationCallback`).

Each scenario varies exactly one axis relative to the `normal_balanced`
baseline: demand level, directional imbalance, movement mix (left/
straight-heavy), vehicle composition, or arrival pattern (`constant` vs
`burst` — a 3-segment 0.56x/2.78x/0.56x demand curve across the hour).
Every OOD scenario is a *more extreme* version of its closest
development analogue (e.g. `high_demand` ×1.30 vs `very_high_demand_OOD`
×1.90), so OOD genuinely tests generalization rather than relabeling.

A **Webster Y-ratio** (`Y = y_NS + y_EW`, `y = demand/saturation_flow`)
is computed per scenario as an informational diagnostic only — it does
not drive scenario generation or split assignment.

Validation before writing to disk: every distribution sums to 1.0 *and*
every individual value is a valid probability; well-formed XML;
**flow-ordering regression guard** (SUMO requires non-decreasing `<flow>`
begin times — an earlier bug wrote a scenario's segments movement-by-
movement instead of globally sorted by time, which silently dropped 2 of
3 time-segments' vehicles for burst scenarios).

Output per scenario: `scenario.json`, `flow.xml`, `vtype.xml`.

---

## 3. Stage 2 — SUMO simulation

**Network:** `sq.net.xml` — one 4-way signalized junction ("Squire
Junction"), 3 lanes per approach edge (`1i`=west, `2i`=east, `3i`=south,
`4i`=north — confirmed from junction coordinates), ~485m approach
length. Simulation window: `0 → 3600s` (one hour).

**Vehicle types:** bike (50%), car (30%), hgv (10%), bus (10%), Krauss
car-following model (`sigma=0.0, tau=1.0`).

**The traffic light program** (`<tlLogic id="0">`, static, 8 phases):

| Phase | Duration | Meaning |
|---|---|---|
| 0 | 25s | NS through/right green |
| 1 | 7s | transition (yellow) |
| 2 | 6s | NS protected-left green |
| 3 | 7s | transition (yellow) |
| 4 | 25s | EW through/right green |
| 5 | 7s | transition (yellow) |
| 6 | 6s | EW protected-left green |
| 7 | 7s | transition (yellow) |

This mapping was **independently decoded** from the raw `<connection
... linkIndex>` table in `sq.net.xml` (`controller/signal_config.py`) —
and in doing so we found that `dataset/feature_builder.py`'s
`PHASE_GREEN_GROUP` has NS/EW **backwards** relative to the real
program. That mislabeling is deliberately left alone in the offline
feature pipeline (the frozen HGB model was trained against it — "fixing"
it now would silently shift the model's input distribution) but the
live PPO controller uses `signal_config.py`'s **correct** mapping, since
real signal safety depends on getting this right.

**Output:** `raw_output/vehicle_trajectories.csv` (every vehicle, every
second: position, speed, lane, distance-to-stopline) and
`ground_truth/state_timeseries.csv` (the TRUE per-approach queue length,
density, flow — read in exactly one place downstream, `build_labels()`,
and never exposed to any sensor or controller).

---

## 4. Stage 3 — Sensors (the physical observation limit)

**Anti-leakage rule, enforced by code, everywhere:** every sensor file
carries a `FORBIDDEN_GROUND_TRUTH_COLUMNS` set and raises an error if any
ground-truth-shaped column name ever appears in its output.

### Camera (`camera_simulator.py`)
Hard range limit: **150 m** from the stop line. A vehicle beyond that
does not exist as far as the camera is concerned — same as a real
camera. Key output field: `queue_reaches_camera_edge` — signals "the
visible queue fills my whole field of view, there may be more I can't
see," **never** a claim about the true length.

**Physical concept — censoring:** the true quantity exists, but the
sensor's range prevents observing all of it.
```text
True queue:  |----------------------------------------->
Camera FOV:  |------------------|
                                ↑ observation boundary
```

### GPS (`gps_simulator.py`, v0.6)
11% of vehicles are probes, selected **deterministically**
(`sha256(seed:vehicle_id)`-ranked, exact top-K cut — not an independent
coin-flip per vehicle). Not range-limited (a probe far upstream still
reports), but count-limited. v0.6 preserves each probe's full 1-second
trajectory across approach/internal/outgoing edges (for possible future
Cheng et al. critical-point analysis — **explicitly not implemented**,
since ASTRID's aggregate probe stats don't expose the per-vehicle
trajectory that method's equations require; an earlier substitute was
found to be a materially different, coarser quantity and was removed).

### Queue definition (`trajectory_utils.py`) — one rule, shared everywhere
A vehicle counts as **queued** once at/below 1.0 m/s (`QUEUE_SPEED_
THRESHOLD_MPS`), on an approach edge, for **≥3.0 consecutive seconds**
(`QUEUE_MIN_DURATION_S`). Used identically by ground truth and both
sensors, so every part of the pipeline agrees on what "queued" means.

---

## 5. Stage 4 — Feature engineering and the physics behind it

**File:** `feature_builder.py`. Two feature layers, on a shared
5-second grid (`SAMPLING_INTERVAL_S`):

- **Layer 1** — camera-only: `visible_vehicle_count`, `visible_mean_
  speed_mps`, `visible_queue_length_m`, `visible_occupancy_fraction`,
  plus **past-only** 30-second change features (`value(t) - value(t-30s)`
  — never uses future information).
- **Layer 2** — Layer 1 + GPS probe stats + signal-phase state
  (`current_phase`, `phase_elapsed_s`, `is_green_for_approach`,
  `red_duration_s`) + **physics-derived features**.

### The physics, and exactly what is/isn't literature-sourced

Grounded in **Lighthill & Whitham (1955)**, "On kinematic waves II" —
full text read directly, every citation checked against it:

- `estimated_density_k_veh_per_km` — adapted from LW's density concept
  (§2, eq. 2), but as an *instantaneous* snapshot over the camera's own
  visible region, not LW's time-averaged link-slice quantity.
- `observed_flow_veh_per_hour` — LW's `q = k·v` identity (eq. 3), fed
  with the density estimate above and the camera's mean speed. Explicitly
  an aggregate, mixed-regime quantity (may include both free-flow and
  queued vehicles at once) — kept as a general feature, deliberately
  **not** fed into a shock-speed formula, because it isn't the clean
  upstream state that formula requires.
- `estimated_queue_front_propagation_m_per_s` — the queue-front shock
  speed, measured **empirically** as the rate of change of the visible
  queue length over its verified elapsed window. Stands in for LW §6's
  red-signal shockwave concept without needing LW's clean-upstream-state
  assumption. Valid only while the queue hasn't yet reached the camera
  edge; `NA` once censored.
- `estimated_hidden_queue_extension_m` — once the queue **is** censored,
  the last known pre-censoring propagation rate is held constant and
  extrapolated forward across the red phase's duration. Explicitly a
  first-order, held-rate approximation — not LW's exact eq. (17)
  shock-position solution.
- `compute_k_jam()` (jam density) — LW (p.322) only establishes the
  *qualitative* claim that jam headway relates to vehicle length; the
  specific formula (`k_jam = 1000 / (avg_vehicle_length + effective_gap)`)
  is **ASTRID's own modeling assumption**, built in that spirit but not
  derived from LW's equations.

Two other cited papers (Richards 1956; Rempe, Kessler & Bogenberger
2017) were confirmed to exist but were **paywalled and never obtained in
full text** — nothing in the code implements an equation from either;
they're acknowledged as named references only. A previously-attempted
"estimated_shockwave_w_bf_m_per_s" feature was **removed** (not patched)
because it mixed free-flow and queued vehicles into one flux estimate
and pathologically approached zero exactly when the true shock speed
mattered most.

Ground truth is read in **exactly one function**, `build_labels()`,
writing `true_queue_length_m` / `true_queue_beyond_camera` to a separate
labels file — the feature-building code path never opens `ground_truth/`
otherwise.

---

## 6. Stage 5 — Dataset assembly

`manifest.json` per layer/penetration records, per split: scenario list,
row count, key columns, **feature_columns** (23 for Layer 2 p11 — see
Section 8), label columns, metadata columns. Feature-column lists are
cross-checked identical across all four splits at load time
(`data_loader.py`), and any column declared as both a feature *and* a
label/metadata/key raises `DataLeakageGuardError`.

| Split | Scenarios | Rows (Layer 2 p11) |
|---|---|---|
| train | 4 | 11,536 |
| validation | 2 | 5,768 |
| test | 2 | 5,768 |
| ood | 4 | 11,536 |

---

## 7. Stage 6 — Seven supervised models compared

**Target:** `true_queue_length_m` (meters). Split at the **scenario
level**, never row level — rows from one scenario are temporally
correlated, so random row-splitting would leak information between
train and test.

| Model | Val MAE↓ | Test MAE↓ | Test RMSE↓ | Test R²↑ | OOD MAE↓ | OOD R²↑ |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **4.91** | 20.04 | 49.12 | 0.926 | 36.24 | 0.893 |
| Extra Trees | 5.61 | 20.26 | 48.27 | 0.929 | 38.38 | 0.892 |
| XGBoost | 6.22 | 20.56 | 48.32 | 0.929 | 38.41 | 0.892 |
| LightGBM | 5.78 | 20.09 | 47.68 | 0.931 | 36.90 | 0.897 |
| CatBoost | 7.52 | 20.59 | 47.03 | 0.933 | 38.20 | 0.897 |
| **HistGradientBoosting** | 5.28 | **18.71** | **44.63** | **0.939** | **34.76** | **0.904** |
| MLP | 10.67 | 22.39 | 47.51 | 0.931 | 39.81 | 0.892 |

Random Forest won validation MAE, but **held-out test/OOD performance**
was used to decide the final model — HistGradientBoosting won on every
held-out metric. Layer 2 vs Layer 1 (HGB): test MAE dropped 35.9%
(29.19m → 18.71m), OOD MAE dropped 46.1% (64.51m → 34.76m).

**Error analysis revealed a physically meaningful, complementary
pattern:**

| Condition | Random Forest MAE | HGB MAE |
|---|---:|---:|
| Queue 0–25m (clearly visible) | **0.030 m** | 0.438 m |
| `queue_reaches_camera_edge = False` | — | 8.79 m |
| `queue_reaches_camera_edge = True` (censored) | — | **62.09 m** |

RF is nearly perfect on short, directly observable queues; HGB is
comparatively stronger once the queue is censored past the camera's
edge — exactly the physical distinction Section 4 describes.

**Hybrid investigation (`congestion_flag` routing rule):**
`queue_reaches_camera_edge == False → RF`, `== True → HGB`, frozen using
**validation data only**, then applied unchanged to test/OOD:

| Split | RF MAE | HGB MAE | Hybrid MAE |
|---|---:|---:|---:|
| Validation | 4.91 | 5.28 | **4.74** (best) |
| Test | 20.04 | **18.71** | 18.86 |
| OOD | 36.24 | **34.76** | 35.04 |

The hybrid beat RF everywhere and beat HGB on validation — but **not**
on held-out test/OOD. **Decision: keep HistGradientBoosting alone**,
rather than add a routing layer that doesn't actually outperform the
simpler single model on unseen data.

**GPS penetration sensitivity** (frozen p11-trained model, evaluated at
other penetrations): best aggregate performance was actually at 25%
(TEST MAE 10.54m), non-monotonic at 50% (16.52m) — a follow-up
experiment training a fresh model directly on p50 data cut its own TEST
MAE by 57%, showing the p50 degradation was mostly a **training-
distribution mismatch**, not evidence that more GPS is inherently worse.
The **deployed model remains the original p11 baseline** — these
penetration experiments are diagnostic, not a model-selection change.

---

## 8. Stage 7 — The frozen, live queue estimator

`online_hgb_queue_estimator.py` loads the tuned p11 `.joblib` artifact
**inference-only** (never `.fit()`), reconstructs the same 23 Layer-2
features online from live sensor ticks via `OnlineSensorObserver` +
`OnlineLayer2FeatureState`, and renames the model's raw output
(`true_queue_length_m` at training time) to `estimated_queue_length_m`
the instant it leaves the module — that name is never changed back.
Recomputes only on 5-second-aligned ticks, holding the last value
between.

**The 23 Layer-2 features:**
```
camera_range_m, visible_vehicle_count, visible_mean_speed_mps,
visible_queue_count, visible_queue_length_m, queue_reaches_camera_edge,
probe_count, probe_mean_speed_mps, probe_min/max_distance_to_stopline_m,
visible_queue_length_m_change_30s, visible_mean_speed_mps_change_30s,
visible_occupancy_fraction, probe_count_change_30s,
probe_max_distance_to_stopline_m_change_30s, current_phase,
phase_elapsed_s, is_green_for_approach, red_duration_s,
estimated_density_k_veh_per_km, observed_flow_veh_per_hour,
estimated_queue_front_propagation_m_per_s,
estimated_hidden_queue_extension_m
```

---

## 9. Stage 8 — Controllers, in order of sophistication

All three are compared on **identical scenario + identical seed**, so
any KPI difference is attributable to the controller, not to different
traffic (`comparison.py`/`comparison_2.py`).

### a) Fixed-time
Static program, no adaptation — the industry-standard "dumb" baseline.

### b) Webster-timing (`normal_controller.py`)
Computes cycle length and green split **once**, from the scenario's
known demand rate:
```
y_NS = NS_demand / (525 veh/h/lane × 6 lanes)
y_EW = EW_demand / (525 veh/h/lane × 6 lanes)
Y    = y_NS + y_EW
C    = (1.5·L + 5) / (1 − Y)     if Y < 1     (L = 14s lost time)
     = 120s (max cycle)          if Y ≥ 1
green_NS = (C − L) × y_NS / Y ;  green_EW = (C − L) × y_EW / Y
```
Principled, but **not adaptive mid-run** — it doesn't watch the live
queue and react.

### c) Random-Forest control policy (`forest_controller.py`, "astrid")
A *different* random forest than the queue-estimation one — this one
outputs `ACTION_KEEP` / `ACTION_REQUEST_NEXT` directly from live
`ControllerState(estimated_queue_m, current_phase, phase_elapsed_s)`.
Runs through the shared safety pipeline:
```
policy_fn(state) → action string
        ↓
actions.resolve_action() → TransitionEffect
        ↓
signal_config.is_legal_transition() checked
        ↓
sumo_interface.py issues AT MOST one setPhase() call
```
`sumo_interface.py` treats SUMO itself as the sole source of truth for
phase/timing (`getPhase`/`getSpentDuration`) — an earlier bug re-issued
`setPhase()` every step even when the phase hadn't changed, which
restarts SUMO's own phase timer and corrupts timing; fixed to issue
exactly one call per real transition. `_ensure_stage_duration()`
programs each newly-entered stage to `MAX_GREEN_S` via
`setPhaseDuration()` once per entry, giving `ACTION_KEEP` genuine
authority to hold past the base static duration.

**Reward used for this RL-adjacent evaluation** (`reward.py`):
```
reward = −(w_queue·Σ estimated_queue_m
           + w_switch_requested·[genuine policy switch]
           + w_switch_forced·[safety-cap-forced switch])
```
`w_switch_requested=15.0` vs `w_switch_forced=1.0` — deliberately
asymmetric, so the policy is taught to control traffic well, not merely
to avoid ever letting the safety cap fire.

---

## 10. Stage 9 — PPO: the reinforcement-learning controller

### What PPO is, and why it's a different branch of ML

```text
Machine Learning
    ├── Supervised     : X → known y            (RF, HGB, XGBoost, MLP, ...)
    ├── Unsupervised    : structure in X, no y
    └── Reinforcement   : policy learned by ACTING in an environment
          Learning        and being scored by REWARD, no labels at all
                           ← PPO lives here
```

Everything in Section 7–9(a–c) is either supervised regression or a
policy trained/derived without ever experiencing multi-step
consequences of its own actions inside the simulation. **PPO
(Proximal Policy Optimization)** — an on-policy, actor-critic RL
algorithm — closes that gap: it acts, watches what happens to real
traffic as a result, and updates its policy toward actions that lead to
better long-run outcomes, formalized as maximizing
`J(θ) = E[Σ γ^t · r_t]`.

### Full loop
```text
Real SUMO  →  online sensor features  →  FROZEN HGB estimator (unchanged)
                                                    │
                                    + live signal timing state
                                                    ▼
                                     PPO OBSERVATION (23-dim)
                                                    ▼
                                    PPO MLP actor/critic [128,128]
                                                    ▼
                                          KEEP  or  SWITCH
                                                    ▼
                              SignalSafetyController (ONLY thing that
                               calls setPhase/setPhaseDuration)
                                                    ▼
                                    Real SUMO TLS, 5 more seconds
                                                    ▼
                          REWARD (real queue, waiting, speed, throughput,
                                  switching — computed from ground truth,
                                  which is normal for RL reward, unlike
                                  the observation above)
                                                    ▼
                                        PPO policy updated
```

### Observation (23 values)
Per approach edge (×4): `estimated_queue_length_m` (frozen HGB),
`is_green_for_approach`, `vehicle_count`, `mean_speed_mps`,
`occupancy_frac`. Global (×3): normalized `current_phase`, normalized
`current_phase_elapsed_s` (resets on every SUMO phase change, including
yellows), normalized `time_since_last_switch_s` (tracked separately —
time since *this controller* last acted, not SUMO's own per-phase
timer).

### Action space and safety
`Discrete(2)`: KEEP / SWITCH. Only the four **stage** phases (0,2,4,6)
are controllable at all — yellow phases (1,3,5,7) always run their fixed
duration untouched, and PPO's action has no effect during them.
`MIN_GREEN_S[phase]` equals that phase's own original static duration
(sourced from `signal_config.py`, decoded from the real `<tlLogic>`), so
PPO can never make a stage shorter than the fixed-time program already
would — its real lever is **extending** a green past default, up to
`MAX_GREEN_S = 2× default` (ASTRID's own prototype safety cap).
Extension is implemented via `setPhaseDuration()`, since the underlying
program is static and would otherwise auto-advance regardless of what
PPO wants.

### Reward
```
r = + w_queue      ·(−avg_queue_m/queue_scale)
    + w_waiting    ·(−avg_waiting_s/waiting_scale)
    + w_speed      ·( avg_speed_mps/speed_scale)
    + w_throughput ·( arrived_this_interval/control_interval_s)
    − w_switch     ·( 1 if switched this tick else 0)
```
Queue and waiting are **instantaneous levels**, not deltas — a
`traci.edge.getWaitingTime()` snapshot resets per-vehicle once it moves,
so a delta-of-snapshots wouldn't cleanly mean "waiting incurred this
interval," and a persistently long queue must keep costing reward every
tick it persists. Speed is rewarded only *alongside* queue/throughput,
so a policy can't "win" by starving one approach to inflate the others'
speed. Every term is divided by a scale constant first, so no term
dominates purely from unit scale (the exact same normalization concern
that motivated `w_switch_requested=15.0` in Section 9c's reward).

### Training discipline
- **Warm-up:** 300s, during which the HGB estimator is still ticked
  every second (populating its 30-second rolling history) but no
  reward/training happens — fixes an earlier version where warm-up
  skipped the estimator entirely, leaving it with zero history the
  moment real training began.
- **Episode length is absolute**, not warm-up-plus-episode:
  `episode_seconds` = the scenario's real simulation end (3600s) — an
  earlier bug added warm-up on top, producing a target the real SUMO run
  could never reach.
- **Decision interval:** 5 seconds, matching the HGB estimator's own
  feature grid.
- **Validation:** deterministic, one fixed-seed episode per validation
  scenario (never randomly sampled — an earlier version's random
  sampling could evaluate the same scenario twice and never touch the
  other), scored by a composite (`−queue −0.5·waiting +0.1·speed
  +0.05·throughput`) that mirrors the project's real priority order —
  **not** raw PPO training reward, since "highest reward" and "best
  traffic controller" aren't automatically the same thing.
- Train/validation/test/OOD separation is identical in spirit to
  Section 7 — PPO is never trained or checkpoint-selected on test/OOD.

### Network & hyperparameters
MLP actor/critic, `[128, 128]` each (SB3 `PPO`, `MlpPolicy`) —
deliberately small; this is a low-dimensional single-intersection state
vector, not an image or sequence task.

| Hyperparameter | Default |
|---|---|
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| total_timesteps | 200,000 |
| seed | 42 |

---

## 11. Stage 10 — Fair comparison, and the bug we found and fixed

`comparison.py`/`comparison_2.py` run "normal," "astrid" (RF), and "ppo"
on the **same scenario, same seed**. A real bug was found and fixed
here: the "normal"/"astrid" legs always launched SUMO against one
static, scenario-independent route file (`sq.rou.xml`, ~4800 vehicles,
generated once), regardless of which scenario was requested — while the
PPO leg correctly used each scenario's own `flow.xml`/`vtype.xml`. Every
prior scenario-to-scenario comparison for "normal"/"astrid" was
therefore secretly running identical traffic. Fixed by generating a
per-scenario `.sumo.cfg` (shared network, that scenario's own demand
files and seed) for every controller, the same way PPO's environment
already did it.

---

## 12. Glossary of key labels

| Term | Meaning |
|---|---|
| `true_queue_length_m` | Ground-truth queue length (meters) — the supervised-learning label; never a feature. |
| `estimated_queue_length_m` | The frozen HGB model's live prediction — the name it's given the instant it leaves the estimator. |
| `queue_reaches_camera_edge` | Censoring flag: visible queue fills the camera's whole range; does *not* mean true queue = camera range. |
| `design_method` / `split` | `development` (train/val/test) vs `ood` — OOD is never used for training or selection. |
| `Layer 1` / `Layer 2` | Camera-only features vs. camera+GPS+signal+physics features. |
| `penetration` (e.g. p11) | Fraction of vehicles that are GPS probes (11% is the primary experiment). |
| `STAGE_INDICES` | The 4 controllable green phases (0,2,4,6); yellows (1,3,5,7) are never controllable. |
| `MIN_GREEN_S` / `MAX_GREEN_S` | Per-phase floor (= the static program's own duration) and ceiling (= 2× that) on green time. |
| `KEEP` / `SWITCH` | PPO's only two actions — never a raw phase index. |
| `congestion_flag` | The frozen (validation-only) RF/HGB hybrid routing rule — ultimately rejected in favor of HGB alone. |

---

## 13. Current pipeline status

| Stage | Status |
|---|---|
| Scenario generation (12) | Complete |
| SUMO simulation + sensors | Complete |
| Feature engineering (Layer 1/2) | Complete |
| Dataset assembly + QA | Complete |
| 7-model comparison | Complete |
| Error analysis + hybrid investigation | Complete |
| Final model selection (HistGradientBoosting, p11) | Complete |
| GPS penetration sensitivity | Complete |
| Fixed-time / Webster / RF controllers | Complete |
| PPO reinforcement-learning controller | Complete (baseline trained, validated) |
| Fair 3-way controller comparison | Complete (route-file bug fixed) |
| Extended PPO training / hyperparameter tuning | Pending |
| Full closed-loop TEST/OOD PPO evaluation | Pending |