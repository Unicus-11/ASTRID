"""
train_rl.py
====================
RL training driver for TrainableMLP (see rl_policy.py). Policy gradient
(REINFORCE) + a learned linear value baseline, stochastic action
sampling during training. Trains against --train-scenario-dirs only;
--val-scenario-dirs are used solely to pick the best checkpoint
(deterministic evaluation via eval_common.run_evaluation_episode, the
SAME function eval_rl.py uses for the final held-out comparison). Test
scenarios are never referenced by this script at all -- pass them only
to eval_rl.py.

Per epoch: run every (scenario, episode) combination, collect each
episode's DecisionRecords (features, sampled action, reward broken into
components), compute discounted per-episode returns-to-go, pool
everything collected this epoch, compute advantages against the value
baseline, normalize them, then take exactly one TrainableMLP
policy_gradient_step() (entropy bonus + optional grad-norm clipping)
and one LinearValueBaseline update.

Uses only SumoInterface's existing public start()/step()/close() (never
run(), since this needs to call policy.add_step_reward() after each
step) and never calls traci.* directly itself. actions.py,
sumo_interface.py, nn_model.py, nn_features.py, controller_state.py,
signal_config.py, and online_hgb_queue_estimator.py are all imported
and used unmodified. The policy only ever returns "KEEP"/"REQUEST_NEXT"
strings and every transition still passes through
actions.resolve_action() inside SumoInterface.step() -- no new path to
TraCI is introduced anywhere in this file.

This experiment is scoped to the current p11 (11%) GPS penetration
(--penetration default 0.11). A future penetration-sensitivity sweep is
a separate experiment and must not be mixed into this training run or
into checkpoint selection.

Has not been run against a real SUMO install in the environment this
was written in -- treat as a carefully-reasoned first draft; watch the
printed per-epoch log (component breakdown + validation score) once you
run it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from nn_controller import NNPolicy  # noqa: E402 (unmodified -- reused as-is for deterministic validation)
from nn_features import FEATURE_DIM  # noqa: E402
from signal_config import SIMULATION_END_S  # noqa: E402
from sumo_interface import LoopConfig, SumoInterface  # noqa: E402
from reward import RewardConfig  # noqa: E402
from rl_policy import TrainableMLP, TrainablePolicy, LinearValueBaseline, compute_returns  # noqa: E402
from eval_common import build_estimator, build_episode_sumo_config, run_evaluation_episode  # noqa: E402
import nn_features
import nn_model

print("nn_features loaded from:", nn_features.__file__)
print("FEATURE_DIM =", nn_features.FEATURE_DIM)
print("nn_model loaded from:", nn_model.__file__)

def run_training_episode(scenario_dir: Path, args: argparse.Namespace, policy: TrainablePolicy) -> dict:
    """One training episode: fresh HGB estimator, fresh SumoInterface,
    stochastic policy, reward attached after every step. Inlines
    SumoInterface.run()'s loop (not editing sumo_interface.py) purely so
    policy.add_step_reward() can be called after each trace."""
    policy.reset_episode()
    estimator = build_estimator(
        scenario_dir=scenario_dir,
        sumo_config_json=Path(args.sumo_config_json),
        model_path=Path(args.model_path),
        manifest_path=Path(args.manifest_path),
        penetration=args.penetration,
    )
    interface = SumoInterface(LoopConfig(
        sumo_binary=args.sumo_binary,
        config_path=str(build_episode_sumo_config(scenario_dir)),
        max_steps=args.max_steps,
        queue_estimator=estimator,
        policy_fn=policy,
        print_every_s=float("inf"),
    ))
    interface.start()
    n_steps = 0
    try:
        while True:
            if args.max_steps is not None and n_steps >= args.max_steps:
                break
            if interface.traci.simulation.getTime() >= SIMULATION_END_S:
                break
            trace = interface.step()
            policy.add_step_reward(trace.resolved_action)
            n_steps += 1
    finally:
        interface.close()

    return {
        "scenario": scenario_dir.name,
        "n_steps": n_steps,
        "n_decisions": len(policy.trajectory),
        "episode_reward": sum(r.accumulated_reward for r in policy.trajectory),
        "queue_cost": sum(r.queue_cost for r in policy.trajectory),
        "requested_switch_cost": sum(r.requested_switch_cost for r in policy.trajectory),
        "forced_switch_cost": sum(r.forced_switch_cost for r in policy.trajectory),
        "requested_transitions": sum(r.requested_transitions for r in policy.trajectory),
        "forced_transitions": sum(r.forced_transitions for r in policy.trajectory),
        "shaping_reward": sum(r.shaping_reward for r in policy.trajectory),
    }


def run_validation(model: TrainableMLP, val_scenario_dirs: List[Path], args: argparse.Namespace,
                    reward_config: RewardConfig) -> float:
    """Deterministic (argmax) evaluation on every val scenario, via
    NNPolicy(model=...) -- the SAME class production uses -- bypassing
    only its default_factory (which would otherwise reload the
    hardcoded warm-start path). Returns mean per-step cost across
    scenarios (lower is better); this is the checkpoint-selection
    metric."""
    costs = []
    for scenario_dir in val_scenario_dirs:
        eval_policy = NNPolicy(model=model)
        metrics = run_evaluation_episode(
            policy_fn=eval_policy,
            scenario_dir=scenario_dir,
            sumo_config_json=Path(args.sumo_config_json),
            model_path=Path(args.model_path),
            manifest_path=Path(args.manifest_path),
            penetration=args.penetration,
            sumo_binary=args.sumo_binary,
            sumo_cfg_name=args.sumo_cfg_name,
            max_steps=args.max_steps,
            reward_config=reward_config,
        )
        costs.append(metrics.mean_cost)
    return float(np.mean(costs))


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training for ASTRID's NNPolicy.")
    parser.add_argument("--train-scenario-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--val-scenario-dirs", type=str, nargs="+", required=True,
                         help="Used ONLY for checkpoint selection, never for gradient updates.")
    parser.add_argument("--sumo-config-json", type=str, required=True)
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str, default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11, help="p11 for this experiment; do not sweep here.")
    parser.add_argument("--sumo-binary", type=str, default="sumo")
    parser.add_argument("--sumo-cfg-name", type=str, default="sq.sumo.cfg")
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--init-weights", type=str, default="controller/nn_controller_warmstart.npz")
    parser.add_argument("--output-dir", type=str, default="controller")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--validate-every", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-policy", type=float, default=1e-3)
    parser.add_argument("--lr-value", type=float, default=1e-2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=5.0,
                         help="Pass a negative value to disable clipping.")

    parser.add_argument("--w-queue", type=float, default=1.0)
    parser.add_argument("--w-switch-requested", type=float, default=15.0)
    parser.add_argument("--w-switch-forced", type=float, default=1.0)
    parser.add_argument("--enable-potential-shaping", action="store_true")
    parser.add_argument("--w-shaping", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reward_config = RewardConfig(
        w_queue=args.w_queue,
        w_switch_requested=args.w_switch_requested,
        w_switch_forced=args.w_switch_forced,
        enable_potential_shaping=args.enable_potential_shaping,
        w_shaping=args.w_shaping,
        shaping_gamma=args.gamma,
    )
    max_grad_norm = None if args.max_grad_norm is not None and args.max_grad_norm < 0 else args.max_grad_norm
    rng = np.random.default_rng(args.seed)

    model = TrainableMLP(seed=args.seed)
    init_path = Path(args.init_weights)
    if init_path.exists():
        model.load(str(init_path))
        print(f"[init] loaded starting weights from {init_path}")
    else:
        print(f"[init] {init_path} not found -- starting from random Xavier init.")

    value_fn = LinearValueBaseline(dim=FEATURE_DIM)
    train_dirs = [Path(s) for s in args.train_scenario_dirs]
    val_dirs = [Path(s) for s in args.val_scenario_dirs]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "nn_controller_rl_last.npz"
    best_path = out_dir / "nn_controller_rl_best.npz"
    best_val_cost = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_features, epoch_actions, epoch_returns = [], [], []
        epoch_stats = []

        for scenario_dir in train_dirs:
            for _ in range(args.episodes_per_scenario):
                policy = TrainablePolicy(model=model, reward_config=reward_config, rng=rng)
                stats = run_training_episode(scenario_dir, args, policy)
                epoch_stats.append(stats)

                episode_rewards = [r.accumulated_reward for r in policy.trajectory]
                episode_returns = compute_returns(episode_rewards, args.gamma)
                for rec, g in zip(policy.trajectory, episode_returns):
                    epoch_features.append(rec.features)
                    epoch_actions.append(rec.action_index)
                    epoch_returns.append(g)

        if not epoch_features:
            print(f"[epoch {epoch}] no decisions collected -- check scenario dirs / max-steps.")
            continue

        X = np.stack(epoch_features).astype(np.float64)
        A = np.array(epoch_actions, dtype=np.int64)
        G = np.array(epoch_returns, dtype=np.float64)

        advantages = G - value_fn.predict(X)
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        update_info = model.policy_gradient_step(
            X, A, advantages, lr=args.lr_policy, entropy_coef=args.entropy_coef, max_grad_norm=max_grad_norm,
        )
        value_mse = value_fn.update(X, G, lr=args.lr_value)

        mean_reward = float(np.mean([s["episode_reward"] for s in epoch_stats]))
        mean_queue_cost = float(np.mean([s["queue_cost"] for s in epoch_stats]))
        mean_req_switch_cost = float(np.mean([s["requested_switch_cost"] for s in epoch_stats]))
        mean_forced_switch_cost = float(np.mean([s["forced_switch_cost"] for s in epoch_stats]))
        mean_req_transitions = float(np.mean([s["requested_transitions"] for s in epoch_stats]))
        mean_forced_transitions = float(np.mean([s["forced_transitions"] for s in epoch_stats]))
        mean_shaping = float(np.mean([s["shaping_reward"] for s in epoch_stats]))

        print(
            f"[epoch {epoch:4d}] episodes={len(epoch_stats):3d} decisions={len(epoch_returns):4d} "
            f"reward={mean_reward:9.2f} queue_cost={mean_queue_cost:8.2f} "
            f"req_switch_cost={mean_req_switch_cost:7.2f}(n={mean_req_transitions:4.1f}) "
            f"forced_switch_cost={mean_forced_switch_cost:6.2f}(n={mean_forced_transitions:4.1f}) "
            f"shaping={mean_shaping:7.2f} pg_loss={update_info['policy_loss']:8.4f} "
            f"entropy={update_info['mean_entropy']:.4f} grad_norm={update_info['grad_norm']} "
            f"value_mse={value_mse:10.2f}"
        )

        model.save(str(last_path))

        if epoch % args.validate_every == 0 or epoch == args.epochs:
            val_cost = run_validation(model, val_dirs, args, reward_config)
            print(f"[epoch {epoch}] validation mean_cost={val_cost:.4f} (best so far={best_val_cost:.4f})")
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                model.save(str(best_path))
                print(f"[epoch {epoch}] new best checkpoint saved to {best_path}")

        if epoch % args.checkpoint_every == 0:
            print(f"[epoch {epoch}] periodic checkpoint (last) saved to {last_path}")

    print(f"[done] best validation mean_cost={best_val_cost:.4f} -> {best_path}")


if __name__ == "__main__":
    main()