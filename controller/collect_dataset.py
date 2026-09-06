"""
collect_dataset.py
====================
Runs the rule-based teacher through real SUMO, via the SAME unmodified
SumoInterface/LoopConfig/OnlineHGBQueueEstimator pipeline, and saves the
resulting (features, label, scenario) dataset to an .npz file for
train_classifiers.py.

One pass per scenario dir is sufficient: both the SUMO scenario and the
teacher are deterministic given a fixed seed, so repeated episodes on
the same scenario would just duplicate rows.

Run this once for your TRAIN scenarios and once for your VAL scenarios
(different --output-path each time) to keep the same train/val
separation discipline as before -- train_classifiers.py never sees the
val scenarios' underlying SUMO runs, only the labeled rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from signal_config import SIMULATION_END_S  # noqa: E402
from sumo_interface import LoopConfig, SumoInterface  # noqa: E402
from eval_common import build_estimator  # noqa: E402
from rule_teacher import RuleTeacherConfig  # noqa: E402
from teacher_policy import TeacherDataCollector  # noqa: E402


def collect_scenario(scenario_dir: Path, args, teacher_config: RuleTeacherConfig):
    collector = TeacherDataCollector(teacher_config=teacher_config)
    collector.reset_episode()
    estimator = build_estimator(
        scenario_dir=scenario_dir,
        sumo_config_json=Path(args.sumo_config_json),
        model_path=Path(args.model_path),
        manifest_path=Path(args.manifest_path),
        penetration=args.penetration,
    )
    interface = SumoInterface(LoopConfig(
        sumo_binary=args.sumo_binary,
        config_path=str(scenario_dir / args.sumo_cfg_name),
        max_steps=args.max_steps,
        queue_estimator=estimator,
        policy_fn=collector,
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
            interface.step()
            n_steps += 1
    finally:
        interface.close()
    return collector.dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a rule-teacher-labeled dataset from real SUMO.")
    parser.add_argument("--scenario-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--sumo-config-json", type=str, required=True)
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str, default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11)
    parser.add_argument("--sumo-binary", type=str, default="sumo")
    parser.add_argument("--sumo-cfg-name", type=str, default="sq.sumo.cfg")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    teacher_config = RuleTeacherConfig(ratio=args.ratio, margin=args.margin)
    all_X, all_y, all_scenario = [], [], []
    for scenario_dir in [Path(s) for s in args.scenario_dirs]:
        rows = collect_scenario(scenario_dir, args, teacher_config)
        print(f"[{scenario_dir.name}] collected {len(rows)} decisions")
        for features, label in rows:
            all_X.append(features)
            all_y.append(label)
            all_scenario.append(scenario_dir.name)

    X = np.stack(all_X).astype(np.float64)
    y = np.array(all_y, dtype=np.int64)
    scenario = np.array(all_scenario)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y, scenario=scenario)
    print(f"[done] saved {X.shape[0]} rows ({(y == 1).sum()} REQUEST_NEXT / {(y == 0).sum()} KEEP) to {out_path}")


if __name__ == "__main__":
    main()