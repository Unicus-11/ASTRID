"""
rl_policy.py
====================
Training-time counterpart to nn_controller.NNPolicy. NNPolicy itself
stays untouched -- reused directly for deterministic evaluation (see
train_rl.py's run_validation), by constructing
NNPolicy(model=<a TrainableMLP instance>), bypassing only its
default_factory (which would otherwise load the warm-start .npz --
NOT used anywhere in this file).

Algorithm: Actor-Critic with Generalized Advantage Estimation (GAE).

TrainableMLP subclasses nn_model.TrafficControllerMLP -- same
architecture (14 -> 64 -> 64 -> 2), same forward pass, same .npz
save/load format, random Xavier init inherited unmodified. No warm-start
checkpoint is loaded anywhere in this file.

MLPValueBaseline is a small critic (14 -> 32 -> 1) used only to compute
GAE advantages; it never influences which action is taken.

No hand-written queue-aware label or heuristic exists anywhere in this
file. Queue-sensitive behavior comes only from the policy-gradient
update driven by reward.py's existing, unmodified queue-cost reward.

RETURN NORMALIZATION (this revision -- diagnostic-driven fix)
-----------------------------------------------------------------
A 3-epoch real-SUMO diagnostic run showed MLPValueBaseline's value_mse
diverging by ~4 orders of magnitude in 2 epochs (1.40e9 -> 1.15e13).
Root cause: ASTRID's queue-cost-dominated rewards are unnormalized and
can reach the hundreds of thousands per episode; fitting a plain-SGD
critic directly against targets of that raw magnitude at any reasonable
learning rate is numerically unstable, independent of whether the
policy's own gradient step is otherwise correct.

RunningMeanStd (added below) is a standard, self-adapting fix (the same
technique OpenAI Baselines' RunningMeanStd / VecNormalize use): track a
running mean/variance of GAE returns, fit MLPValueBaseline against
NORMALIZED targets (well-conditioned, O(1) scale, regardless of the raw
reward magnitude), and denormalize its predictions back to raw units
before they are ever used inside compute_gae. GAE itself still operates
entirely in raw reward units -- only the critic's internal fitting
target changes. This introduces no new hyperparameter that has to be
hand-tuned to the reward scale; it adapts automatically as rewards are
observed.

Nothing else in this file (TrainableMLP, MLPValueBaseline's own
architecture/update rule, DecisionRecord, TrainablePolicy, compute_gae)
is modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from actions import ACTION_KEEP, ACTION_REQUEST_NEXT
from controller_state import ControllerState
from nn_features import build_nn_features
from nn_model import TrafficControllerMLP
from signal_config import STAGE_INDICES
from reward import RewardConfig, compute_step_components

DECISION_INTERVAL_S = 5.0
INDEX_TO_ACTION = {0: ACTION_KEEP, 1: ACTION_REQUEST_NEXT}


class TrainableMLP(TrafficControllerMLP):
    """Adds a policy-gradient update (entropy bonus + optional gradient
    clipping) on top of TrafficControllerMLP's unmodified architecture,
    forward pass, and random Xavier initialization."""

    def policy_gradient_step(
        self,
        x_batch: np.ndarray,
        action_indices: np.ndarray,
        advantages: np.ndarray,
        lr: float = 1e-3,
        entropy_coef: float = 0.0,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        probs, (x, z1, h1, z2, h2, _) = self._forward_with_cache(x_batch)
        n = x_batch.shape[0]
        if advantages.shape != (n,):
            raise ValueError(f"advantages must have shape ({n},), got {advantages.shape}")
        if action_indices.shape != (n,):
            raise ValueError(f"action_indices must have shape ({n},), got {action_indices.shape}")

        logp = np.log(np.clip(probs, 1e-9, 1.0))
        entropy = -(probs * logp).sum(axis=1)

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), action_indices] = 1.0

        pg_loss = float(-np.mean(advantages * logp[np.arange(n), action_indices]))
        mean_entropy = float(entropy.mean())

        d_logits_pg = (probs - y_onehot) * advantages[:, None]
        d_logits_entropy = entropy_coef * probs * (logp + entropy[:, None])
        d_logits = (d_logits_pg + d_logits_entropy) / n

        d_w3 = h2.T @ d_logits
        d_b3 = d_logits.sum(axis=0)
        d_h2 = d_logits @ self.w3.T
        d_z2 = d_h2 * (z2 > 0)
        d_w2 = h1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (z1 > 0)
        d_w1 = x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        grad_norm = None
        if max_grad_norm is not None:
            total_sq = sum(float((g ** 2).sum()) for g in (d_w1, d_b1, d_w2, d_b2, d_w3, d_b3))
            total_norm = float(np.sqrt(total_sq))
            grad_norm = total_norm
            if total_norm > max_grad_norm:
                scale = max_grad_norm / (total_norm + 1e-8)
                d_w1 *= scale; d_b1 *= scale
                d_w2 *= scale; d_b2 *= scale
                d_w3 *= scale; d_b3 *= scale

        self.w3 -= lr * d_w3
        self.b3 -= lr * d_b3
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1

        return {"policy_loss": pg_loss, "mean_entropy": mean_entropy, "grad_norm": grad_norm}


@dataclass
class MLPValueBaseline:
    """Critic: 14 -> 32 -> 1, ReLU hidden. Used only for GAE advantages.

    Unchanged architecture/update rule from the prior revision. Callers
    (train_rl.py) are now expected to fit this against NORMALIZED
    return targets (via RunningMeanStd below) rather than raw returns --
    this class itself has no opinion about normalization; it just fits
    whatever targets it is given, as before.
    """

    dim: int
    hidden: int = 32
    seed: int = 0
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        bound1 = np.sqrt(6.0 / (self.dim + self.hidden))
        self.w1 = rng.uniform(-bound1, bound1, (self.dim, self.hidden)).astype(np.float64)
        self.b1 = np.zeros(self.hidden, dtype=np.float64)
        bound2 = np.sqrt(6.0 / (self.hidden + 1))
        self.w2 = rng.uniform(-bound2, bound2, (self.hidden, 1)).astype(np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        h = np.maximum(x_batch @ self.w1 + self.b1, 0.0)
        return (h @ self.w2 + self.b2).ravel()

    def update(self, x_batch: np.ndarray, targets: np.ndarray, lr: float = 1e-3) -> float:
        h_pre = x_batch @ self.w1 + self.b1
        h = np.maximum(h_pre, 0.0)
        preds = (h @ self.w2 + self.b2).ravel()
        n = x_batch.shape[0]
        error = preds - targets
        d_out = (error / n)[:, None]
        d_w2 = h.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h = d_out @ self.w2.T
        d_pre = d_h * (h_pre > 0)
        d_w1 = x_batch.T @ d_pre
        d_b1 = d_pre.sum(axis=0)

        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        return float(np.mean(error ** 2))


class RunningMeanStd:
    """Tracks a running mean/variance using Welford's parallel-variance
    combination formula (the same approach OpenAI Baselines' own
    RunningMeanStd / VecNormalize use).

    Used by train_rl.py to normalize GAE return targets before fitting
    MLPValueBaseline (so the critic's own SGD fit stays numerically
    well-conditioned regardless of ASTRID's raw reward magnitude), and
    to denormalize the critic's predictions back to raw reward units
    before they are used inside compute_gae -- GAE itself always
    operates on raw rewards/returns; only the critic's internal fitting
    target is ever normalized.

    Starts at mean=0.0, var=1.0 (i.e. denormalize is the identity
    function until the first update()), so the very first epoch of
    training -- before any return has been observed -- behaves exactly
    as it did before this normalizer existed.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = max(new_var, 1e-8)
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * (np.sqrt(self.var) + 1e-8) + self.mean


def compute_gae(
    rewards: List[float], values: List[float], next_value: float,
    gamma: float = 0.99, lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-episode GAE. next_value=0.0 for episodes ending at
    SIMULATION_END_S (they end, not truncate).

    `values` must already be in RAW reward units (i.e. if the caller's
    critic was fit on normalized targets, `values` here must already be
    denormalized -- see RunningMeanStd above and train_rl.py's usage).
    This function itself is unchanged from the prior revision and has
    no awareness of normalization."""
    advantages = np.zeros(len(rewards))
    gae = 0.0
    values_ext = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + np.array(values)
    return advantages, returns


@dataclass
class DecisionRecord:
    features: np.ndarray
    action_index: int
    accumulated_reward: float = 0.0
    queue_cost: float = 0.0
    requested_switch_cost: float = 0.0
    forced_switch_cost: float = 0.0
    requested_transitions: int = 0
    forced_transitions: int = 0
    shaping_reward: float = 0.0


@dataclass
class TrainablePolicy:
    """policy_fn for training. Samples actions stochastically from the
    actor's softmax. Deterministic argmax evaluation is handled
    separately by nn_controller.NNPolicy in train_rl.py's
    run_validation -- never by this class."""

    model: TrainableMLP
    decision_interval_s: float = DECISION_INTERVAL_S
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    trajectory: List[DecisionRecord] = field(default_factory=list, init=False)
    _last_phase_seen: Optional[int] = field(default=None, init=False)
    _last_decision_bucket: int = field(default=-1, init=False)
    _last_queue_estimate: Optional[dict] = field(default=None, init=False)
    _prev_total_queue_m: Optional[float] = field(default=None, init=False)

    def reset_episode(self) -> None:
        self.trajectory = []
        self._last_phase_seen = None
        self._last_decision_bucket = -1
        self._last_queue_estimate = None
        self._prev_total_queue_m = None

    def __call__(self, state: ControllerState) -> str:
        self._last_queue_estimate = state.estimated_queue_m

        if state.current_phase not in STAGE_INDICES:
            self._last_phase_seen = state.current_phase
            self._last_decision_bucket = -1
            return ACTION_KEEP

        if self._last_phase_seen != state.current_phase:
            self._last_phase_seen = state.current_phase
            self._last_decision_bucket = -1

        bucket = int(state.phase_elapsed_s // self.decision_interval_s)
        if bucket <= self._last_decision_bucket:
            return ACTION_KEEP

        self._last_decision_bucket = bucket
        features = build_nn_features(state)
        probs = self.model.action_probs(features)
        action_index = int(self.rng.choice(len(probs), p=probs))
        self.trajectory.append(DecisionRecord(features=features, action_index=action_index))
        return INDEX_TO_ACTION[action_index]

    def add_step_reward(self, resolved_action_value: str) -> None:
        if not self.trajectory or self._last_queue_estimate is None:
            return
        comp = compute_step_components(
            self._last_queue_estimate, resolved_action_value, self._prev_total_queue_m, self.reward_config,
        )
        self._prev_total_queue_m = comp.queue_m

        rec = self.trajectory[-1]
        rec.accumulated_reward += comp.reward
        rec.queue_cost += comp.queue_cost
        rec.requested_switch_cost += comp.requested_switch_cost
        rec.forced_switch_cost += comp.forced_switch_cost
        rec.requested_transitions += int(comp.is_requested_transition)
        rec.forced_transitions += int(comp.is_forced_transition)
        rec.shaping_reward += comp.shaping_reward