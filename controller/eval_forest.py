"""
eval_forest.py
====================
Held-out comparison of the trained classifier candidates (from
controller/forest_models/, produced by train_classifiers.py) plus
astrid_controller.placeholder_policy as a fixed baseline, on
--test-scenario-dirs -- scenarios that must be disjoint from whatever
collect_dataset.py used to build the train/val datasets.

Uses eval_common.run_evaluation_episode -- the same evaluation function
used for the earlier RL comparison -- so results are directly
comparable to any earlier numbers you have.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from astrid_controller import placeholder_policy  # noqa: E402 (unmodified)
from forest_controller import load_forest_policy  # noqa: E402
from reward import RewardConfig  # noqa: E402
from eval_common import EpisodeMetrics, run_evaluation_episode  # noqa: E402


def print_table(name: str, per_scenario: list) -> None:
    print(f"\n=== {name} ===")
    print(f"{'scenario':<28}{'steps':>8}{'mean_q_m':>10}{'max_q_m':>10}{'req_sw':>8}{'forced_sw':>10}{'mean_cost':>11}")
    for m in per_scenario:  # type: EpisodeMetrics
        print(
            f"{m.scenario:<28}{m.n_steps:>8}{m.mean_total_queue_m:>10.2f}{m.max_total_queue_m:>10.2f}"
            f"{m.requested_transitions:>8}{m.forced_transitions:>10}{m.mean_cost:>11.2f}"
        )
    n = len(per_scenario)
    if n:
        print(
            f"{'MEAN':<28}{'':>8}"
            f"{sum(m.mean_total_queue_m for m in per_scenario) / n:>10.2f}"
            f"{sum(m.max_total_queue_m for m in per_scenario) / n:>10.2f}"
            f"{sum(m.requested_transitions for m in per_scenario) / n:>8.1f}"
            f"{sum(m.forced_transitions for m in per_scenario) / n:>10.1f}"
            f"{sum(m.mean_cost for m in per_scenario) / n:>11.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out comparison for candidate ASTRID forest controllers.")
    parser.add_argument("--test-scenario-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--sumo-config-json", type=str, required=True)
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str, default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11)
    parser.add_argument("--sumo-binary", type=str, default="sumo")
    parser.add_argument("--sumo-cfg-name", type=str, default="sq.sumo.cfg")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--forest-models-dir", type=str, default="controller/forest_models")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--w-queue", type=float, default=1.0)
    parser.add_argument("--w-switch-requested", type=float, default=15.0)
    parser.add_argument("--w-switch-forced", type=float, default=1.0)
    args = parser.parse_args()

    reward_config = RewardConfig(
        w_queue=args.w_queue, w_switch_requested=args.w_switch_requested, w_switch_forced=args.w_switch_forced,
    )
    test_dirs = [Path(s) for s in args.test_scenario_dirs]
    models_dir = Path(args.forest_models_dir)

    controllers = []
    for model_path in sorted(models_dir.glob("*.joblib")):
        if model_path.name == "best_model.joblib":
            continue  # duplicate of whichever candidate won; candidates already listed individually
        controllers.append((model_path.stem, model_path))

    if not args.skip_baseline:
        controllers.append(("baseline (placeholder_policy)", None))

    for name, model_path in controllers:
        per_scenario = []
        for scenario_dir in test_dirs:
            policy_fn = load_forest_policy(model_path) if model_path is not None else placeholder_policy
            metrics = run_evaluation_episode(
                policy_fn=policy_fn,
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
            per_scenario.append(metrics)
        print_table(name, per_scenario)


if __name__ == "__main__":
    main()