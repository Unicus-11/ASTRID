# ASTRID — PPO Reinforcement-Learning Signal Controller

## 0. What is PPO, and where does it sit in ML?

Machine learning splits broadly into:

```text
Machine Learning
    │
    ├── Supervised Learning        (RF, XGBoost, HGB, MLP -- Sections above)
    │     learns  X (features) → y (a known label), from a fixed dataset
    │
    ├── Unsupervised Learning
    │     finds structure in X with no label at all
    │
    └── Reinforcement Learning  (PPO -- this document)
          learns a POLICY by interacting with an ENVIRONMENT and
          maximizing a REWARD signal, with no labeled "correct action"
          ever provided
```

Everything in the `ML Baseline Development` README (Random Forest, XGBoost,
LightGBM, CatBoost, HistGradientBoosting, MLP) is **supervised regression**:
each model is handed `(features, true_queue_length_m)` pairs and learns to
predict the label. There is no notion of an "action" or a "consequence" —
just a fixed table of inputs and outputs.

**PPO (Proximal Policy Optimization)** is a **Reinforcement Learning**
algorithm — specifically an **on-policy, actor-critic** method. It does
not receive a labeled queue length to imitate. Instead it is placed inside
a live simulation loop, tries actions, observes what happens to traffic as
a result, and is rewarded or penalized based on the outcome. Over many
episodes it learns a **policy** — a rule for choosing actions from
observations — that tends to produce better outcomes.

```text
SUPERVISED (HGB, RF, ...)          REINFORCEMENT LEARNING (PPO)
------------------------           -----------------------------
features  →  true_queue_length_m   state s_t  →  action a_t
      (a known, fixed answer)              ↓
                                    environment reacts, produces s_{t+1}
                                            ↓
                                    reward r_t (NOT a label -- a score)
                                            ↓
                                    policy updated to prefer actions
                                    that led to higher long-run reward
```

## 1. Where PPO sits in the ASTRID pipeline

ASTRID's supervised stage answers *"how long is the queue right now?"*
(perception). PPO answers a completely different question: *"given what
we currently know about traffic, should the signal stay green or
switch?"* (control/decision-making).

```text
                 Real SUMO simulation
                         │
                 observable traffic state
                         │
        ┌────────────────┴────────────────┐
        │                                  │
 online sensor features             signal timing state
        │                                  │
        ▼                                  ▼
 FROZEN HGB estimator                current_phase,
 (perception -- unchanged,           phase_elapsed_s,
  never retrained by PPO)            time_since_switch
        │                                  │
        └────────────────┬─────────────────┘
                          ▼
                  PPO OBSERVATION (23-dim)
                          ▼
                ┌───────────────────┐
                │   PPO MLP policy  │  actor [128,128] / critic [128,128]
                └─────────┬─────────┘
                          ▼
                    KEEP  or  SWITCH
                          ▼
                ┌────────────────────┐
                │ SignalSafetyController │  <- the ONLY thing allowed to
                │ (min/max green,        │     call setPhase / setPhaseDuration
                │  legal phase sequence) │
                └─────────┬──────────┘
                          ▼
                    Real SUMO TLS
                          │
                  traffic evolves for
                  one 5-second interval
                          │
                          ▼
              REWARD (queue, waiting, speed,
                      throughput, switching)
                          │
                          ▼
                 PPO updates its policy
```

PPO **never** touches the HGB model's weights, and the HGB model **never**
sees PPO's actions before they happen — the estimator only reports what
sensors currently show, exactly as it would for any other controller.

## 2. The Markov Decision Process (MDP) formulation

Formally, PPO is trying to learn a policy `π_θ(a_t | s_t)` that maximizes:

```
J(θ) = E[ Σ_{t=0}^{T} γ^t · r_t ]
```

where `γ` (gamma) discounts future reward relative to immediate reward,
and the expectation is over trajectories the policy itself generates
by acting in the environment (as opposed to supervised learning's
expectation over a fixed dataset).

| Symbol | Meaning in ASTRID |
|---|---|
| `s_t` | the 23-dim observation at decision tick `t` (Section 3) |
| `a_t` | KEEP or SWITCH (Section 4) |
| `s_{t+1}` | the observation 5 simulated seconds later, after `a_t` and 5s of real SUMO traffic evolution |
| `r_t` | the reward computed from that 5-second interval's real traffic outcome (Section 5) |
| `π_θ` | the PPO actor network — outputs a probability over {KEEP, SWITCH} |
| `V_θ(s_t)` | the PPO critic network — estimates expected future reward from `s_t`, used to reduce training variance (advantage estimation via GAE) |

## 3. Observation — what PPO is allowed to see

23 real numbers, all legitimately available at decision time (no future
information, no ground truth):

| # | Feature | Per | Source |
|---|---|---|---|
| 1 | `estimated_queue_length_m` | each of 4 approach edges | frozen HGB estimator |
| 2 | `is_green_for_approach` (0/1) | each of 4 approach edges | current SUMO phase vs. `signal_config.APPROACH_STAGE` |
| 3 | `vehicle_count` | each of 4 approach edges | `traci.edge.getLastStepVehicleNumber` |
| 4 | `mean_speed_mps` | each of 4 approach edges | `traci.edge.getLastStepMeanSpeed` |
| 5 | `occupancy_frac` | each of 4 approach edges | `traci.edge.getLastStepOccupancy` |
| 6 | `current_phase / n_phases` | global | normalized SUMO phase index |
| 7 | `current_phase_elapsed_s / GLOBAL_MAX_GREEN_S` | global | time in the *current SUMO phase* (resets on every phase change, including yellows) |
| 8 | `time_since_last_switch_s / GLOBAL_MAX_GREEN_S` | global | time since *this controller* last issued KEEP→SWITCH, tracked separately from #7 |

(4 edges × 5 per-edge features = 20, + 3 global = 23.)

`estimated_queue_length_m` and `is_green_for_approach` are exactly the
kind of information the frozen HGB pipeline already produces for every
other controller — PPO does not get privileged sensor access.

## 4. Action space and the safety layer

**Action space: `Discrete(2)`** — `0 = KEEP`, `1 = SWITCH`. PPO never
calls `setPhase()`/`setPhaseDuration()` itself.

`sq.net.xml`'s `<tlLogic>` is a **static** program: 8 phases, cycling
`0→1→2→...→7→0`. Phases `{0, 2, 4, 6}` are steady "stage" greens (NS
through/right, NS protected-left, EW through/right, EW protected-left);
phases `{1, 3, 5, 7}` are mandatory fixed-duration yellows in between.

`ppo_controller.SignalSafetyController` is the only code path allowed to
touch the traffic light:

```
if phase not in STAGE_INDICES:
    # inside a yellow -- PPO's action this tick has NO effect,
    # the fixed transition duration always runs to completion
    return

elapsed = time in current phase
min_g   = MIN_GREEN_S[phase]   # == that phase's own sq.net.xml duration
max_g   = MAX_GREEN_S[phase]   # == 2x that, ASTRID's own prototype safety cap

if elapsed >= max_g:
    SWITCH now, regardless of the action (safety cap, "forced")
elif action == SWITCH and elapsed >= min_g:
    SWITCH now (a genuine, PPO-requested switch)
else:
    EXTEND the phase (setPhaseDuration) so SUMO's static program
    does not auto-advance before the next 5s decision tick
```

Because `MIN_GREEN_S` equals the phase's own original static duration,
PPO can never make a stage *shorter* than the fixed-time program would
have — its only real lever is **extending a green past its default
duration**, up to `MAX_GREEN_S`. This means PPO physically cannot produce
an illegal or unsafe signal sequence; the worst it can do is hold a green
too long or switch "on time."

## 5. Reward — what PPO is optimizing

```
r_t =  + w_queue      · ( -avg_queue_m       / queue_scale_m  )
       + w_waiting    · ( -avg_waiting_s     / waiting_scale_s)
       + w_speed      · (  avg_speed_mps     / speed_scale_mps)
       + w_throughput · (  arrived_this_interval / control_interval_s)
       - w_switch     · ( 1 if a switch happened this tick else 0 )
```

Default weights: `w_queue=1.0, w_waiting=1.0, w_speed=0.5,
w_throughput=0.5, w_switch=0.3` (`ppo_config.RewardWeights`).

**Physics/measurement reasoning behind each term:**

- **Queue** and **waiting** are used as *instantaneous levels*, averaged
  over the 5-second control interval — **not deltas**. `traci.edge.
  getWaitingTime()` returns a per-step snapshot (the sum of each
  currently-present vehicle's *currently accumulated* waiting time,
  which itself resets once that vehicle starts moving again) — it is
  not a monotonically-growing counter, so subtracting consecutive
  snapshots doesn't cleanly mean "waiting incurred this interval." A
  persistently long queue must keep costing reward every tick it
  persists, so a level is the more defensible reading of both metrics.
- **Speed** is rewarded, but only alongside queue/waiting/throughput —
  rewarding speed alone would let a policy "win" by starving one
  approach so the others flow freely.
- **Throughput** (`arrived_this_interval`, real vehicles that completed
  their trip) is rewarded per-second-of-interval so it's on a comparable
  scale to the other terms regardless of `control_interval_s`.
- **Switching penalty** discourages pathological rapid oscillation
  between stages — without it, nothing in the other four terms directly
  penalizes flapping the signal every tick.
- Each raw quantity is divided by a **scale constant**
  (`queue_scale_m=100, waiting_scale_s=200, speed_scale_mps=13.9 ≈
  50 km/h`) before weighting, so no single term dominates purely because
  its raw units happen to be numerically larger (e.g. waiting time in
  seconds vs. speed in m/s) — this mirrors the same normalization
  concern raised for the earlier RF-controller `reward.py`
  (`w_switch_requested=15.0` there existed for exactly this reason: to
  stop one term's raw scale from silently drowning the others).
- Reward is computed from **real traci ground truth** (true waiting
  time, true speed, true arrivals) — this is *not* observation leakage.
  The observation (what the policy sees before acting) stays limited to
  Section 3's sensor-legitimate features; the reward (the training
  signal, computed *after* the action's consequences have played out) is
  allowed to use the full simulation outcome, exactly as standard RL
  practice requires.

## 6. Decision frequency and warm-up

- **Decision interval: 5 seconds** (`CONTROL_INTERVAL_S`), matching
  `SAMPLING_INTERVAL_S` — the same 5-second grid the HGB estimator's own
  features are built on, so PPO's decisions and its perception input are
  synchronized.
- SUMO is advanced by **simulation time**, not a fixed
  `traci.simulationStep()` call count — correct regardless of the
  configured SUMO step length.
- **Warm-up: 300 seconds by default** (`warmup_seconds`), because every
  scenario starts from an empty network — an empty-network queue
  estimate is meaningless and would poison early training. During
  warm-up, SUMO is stepped **and** the HGB estimator's `estimate()` is
  called every second exactly as it will be during the real episode, so
  its rolling/past-only feature history (30-second change features etc.)
  is properly populated before the first reward-bearing observation —
  only the *return value* is discarded, never the estimator's internal
  state.
- **Episode length is absolute**, not "warm-up + episode": `episode_
  seconds` defaults to `signal_config.SIMULATION_END_S` (3600s, matching
  the real `sq.sumo.cfg`). The reward-bearing control window runs from
  `warmup_seconds` to `episode_seconds`, never past the scenario's real
  simulated hour.

## 7. Network architecture

Standard Stable-Baselines3 `PPO` with `"MlpPolicy"`:

```
observation (23) → [128] → [128] → ┬── actor head  → Discrete(2) logits
                                    └── critic head → V(s) scalar
```

A small, plain MLP was chosen deliberately — this is a low-dimensional
state-vector control problem for one intersection, not an image or
sequence task, so there is no justification for a CNN/LSTM/Transformer
here.

## 8. Hyperparameters (defaults, `ppo_config.PPOHyperparams`)

| Hyperparameter | Value |
|---|---|
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma (discount) | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| net_arch | [128, 128] |
| total_timesteps | 200,000 (default; configurable) |
| seed | 42 |

All overridable via `train_ppo.py` CLI flags.

## 9. Scenario split — identical philosophy to the supervised models

| Split | Scenarios | Used for |
|---|---|---|
| train (4) | normal_balanced, low_demand, high_demand, left_turn_heavy | PPO gradient updates |
| validation (2) | north_heavy, straight_heavy | checkpoint selection only |
| test (2) | south_heavy, east_west_heavy | final held-out evaluation only |
| ood (4) | very_high_demand_OOD, north_extreme_OOD, burst_demand_OOD, heavy_vehicle_OOD | generalization evaluation only |

PPO is **never** trained or checkpoint-selected on test/ood — the same
rule enforced throughout the supervised pipeline (`EvaluationReport.
selection_metrics()` excluding OOD, the manifest's OOD-never-gates rule).

## 10. Validation / checkpoint selection

Deliberately **not** Stable-Baselines3's default `EvalCallback`, which
samples a validation scenario at random each call — with only 2
validation scenarios that can (and did) evaluate the same one twice and
never touch the other.

`ValidationCallback` instead, every `eval_every_timesteps`:

1. Runs **exactly one deterministic episode per validation scenario**
   (`north_heavy` and `straight_heavy`, every time — never sampled),
   using **fixed, reproducible seeds** derived via
   `zlib.crc32(f"{scenario_id}_{base_seed}")` (not Python's built-in
   `hash()`, which is randomized per process via `PYTHONHASHSEED` and
   would silently break reproducibility run-to-run).
2. Averages the resulting traffic metrics.
3. Scores them with a composite metric, **not raw PPO episode reward**:

```
composite_score = -1.00 · avg_queue_m
                  -0.50 · avg_waiting_s
                  +0.10 · avg_speed_mps
                  +0.05 · avg_throughput
```

This matters because "highest reward" and "best traffic controller" are
not automatically the same thing — the reward's component weights are a
training convenience, while checkpoint selection should track the
project's actual, documented priority order (reduce congestion/delay
first, maintain speed/throughput second — Section 5's physical
reasoning, re-applied here as a selection criterion instead of a
training signal).

4. Saves the model only when this composite score improves — never
   silently overwriting a better checkpoint with a worse one, including
   across a `--resume` run (the resumed run reloads `best_score` from the
   checkpoint's own saved `validation_metrics.json` rather than
   restarting from `-inf`).

## 11. Evaluation (`evaluate_ppo.py`)

Loads a frozen checkpoint, runs it deterministically (no `.learn()` call)
across a chosen split (`validation` / `test` / `ood`), with the same
CRC32-seeded reproducibility as validation. Reports per-scenario and
averaged: `avg_queue_m`, `max_queue_m`, `avg_waiting_s`, `avg_speed_mps`,
`throughput_vehicles`, `switch_count`, `forced_switch_count`,
`collisions`, `teleports`.

## 12. Comparison against the fixed-time and Random-Forest controllers

`comparison_2.py` runs "normal" (fixed-time Webster/placeholder policy),
"astrid" (Random-Forest `policy_fn`, via the shared `SumoInterface`/
`ControllerState`/`actions.py` pipeline), and "ppo" (driving
`ASTRIDSignalEnv` directly, since PPO's 23-dim observation is richer
than `ControllerState`'s 3 fields) on the **same scenario, same seed**,
so any KPI difference is attributable to the controller, not to
different traffic. A separate, now-fixed bug had the "normal"/"astrid"
legs always replaying one static, scenario-independent route file
(`sq.rou.xml`) regardless of `--scenario-dirs` — patched to build a
per-scenario SUMO config (scenario's own `flow.xml`/`vtype.xml`/seed) the
same way the PPO leg already did.

## 13. Files

```
ppo/
├── ppo_config.py       -- paths, signal facts (imported from controller/signal_config.py,
│                          NOT re-derived), reward weights, PPO hyperparameters, scenario splits
├── ppo_controller.py    -- SignalSafetyController: the only code allowed to touch traci.trafficlight
├── ppo_env.py           -- Gymnasium env: observation/reward/warm-up/episode-timing logic
├── train_ppo.py         -- trains on TRAIN, validates deterministically, supports --resume
├── evaluate_ppo.py      -- no-training evaluation on validation/test/ood, seed-reproducible
└── ppo_models/<run_name>/
    ├── best_model/model.zip + validation_metrics.json
    ├── final_model.zip
    ├── validation_history.json
    └── tb/               (tensorboard logs)
```

## 14. Points still requiring verification against your real deployment

These are explicitly flagged in code comments (`ADAPT`), not silently
assumed:

- `TLS_ID`, `PHASE_NEXT`/`TRANSITION_AFTER_STAGE`, per-phase `MIN_GREEN_S`
  / `MAX_GREEN_S` are sourced from `controller/signal_config.py` (itself
  independently decoded from `sq.net.xml`'s `<tlLogic>`) — if
  `signal_config.py` ever changes, `ppo_config.py` picks that up
  automatically rather than duplicating the values.
- `CAMERA_RANGE_M` / `GPS_PENETRATION_RATE` must match the scenario
  config and the HGB model's own training penetration (11%) — mismatched
  values would silently shift the estimator's feature distribution away
  from what it was trained on.
- Reward scale constants (`queue_scale_m`, `waiting_scale_s`,
  `speed_scale_mps`) are reasonable defaults, not calibrated against your
  own logged value ranges — recommended before trusting absolute reward
  magnitudes across long training runs.