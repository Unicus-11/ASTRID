"""
run_online_hgb_demo.py
========================
Part 20 real-SUMO smoke test: wire OnlineHGBQueueEstimator into the
EXISTING, unmodified SumoInterface/LoopConfig, and demonstrate the
StepTrace changing from queues[1i=NA,...] to queues[1i=<float>,...].

Does not modify actuator behavior, actions.py, or the safety layer in
any way -- only supplies a different queue_estimator and policy_fn to
LoopConfig, exactly the injection points that already existed.

POLICY: uses rule_based_policy (threshold rule on HGB estimated
queues) instead of the NN -- see rule_based_policy.py's docstring for
why, for this prototype stage.

NOTE ON --max-steps: this is simulated SECONDS, not training epochs --
nothing here trains or can "fail" from a larger value. A small value
(e.g. 5) ends the run before MIN_GREEN_S has even elapsed on the first
stage, so the rule will never get a legal chance to fire. Use ~60 for a
quick smoke test, 200+ to see real behavior across multiple stages.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("controller", "sensors", "dataset", "models", "models/results"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import json

from controller_state import PlaceholderQueueEstimator
from signal_config import APPROACH_EDGES, TLS_ID
from sumo_interface import LoopConfig, SumoInterface

from online_hgb_queue_estimator import OnlineHGBQueueEstimator  # from models/
from rule_based_policy import rule_based_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=str, required=True,
                         help="e.g. sumo/generated_scenarios/scenario_normal_balanced")
    parser.add_argument("--sumo-config-json", type=str, required=True,
                         help="Path to scenario_config.json (network.camera_range_m)")
    parser.add_argument("--model-path", type=str,
                         default="models/artifacts/layer2_p11/hist_gradient_boosting_layer2_p11_tuned/hist_gradient_boosting.joblib")
    parser.add_argument("--manifest-path", type=str,
                         default="dataset/assembled/layer2_p11/manifest.json")
    parser.add_argument("--penetration", type=float, default=0.11)
    parser.add_argument("--sumo-binary", type=str, default="sumo")
    parser.add_argument("--sumo-cfg", type=str, default="sq.sumo.cfg")
    parser.add_argument("--max-steps", type=int, default=200,
                         help="Simulated SECONDS to run, not epochs -- nothing trains here. "
                              "Use ~60 for a fast smoke test, 200+ for real behavior.")
    parser.add_argument("--baseline", action="store_true",
                         help="Run with PlaceholderQueueEstimator instead, for A/B comparison.")
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    with open(scenario_dir / "scenario.json", "r", encoding="utf-8") as f:
        scenario = json.load(f)
    with open(args.sumo_config_json, "r", encoding="utf-8") as f:
        network_cfg = json.load(f)

    camera_range_m = network_cfg["network"]["camera_range_m"]
    scenario_seed = int(scenario["seed"])
    sim_begin_s = int(scenario["simulation_begin"])

    if args.baseline:
        estimator = PlaceholderQueueEstimator()
    else:
        # traci_module is resolved lazily inside SumoInterface; the
        # estimator needs the SAME module instance, so we import it here
        # too (both end up importing the single installed `traci` package).
        import traci
        estimator = OnlineHGBQueueEstimator(
            traci_module=traci,
            model_path=Path(args.model_path),
            manifest_path=Path(args.manifest_path),
            approach_edges=APPROACH_EDGES,
            tls_id=TLS_ID,
            camera_range_m=camera_range_m,
            gps_penetration_rate=args.penetration,
            scenario_seed=scenario_seed,
            sim_begin_s=sim_begin_s,
        )

    interface = SumoInterface(LoopConfig(
        sumo_binary=args.sumo_binary,
        config_path=args.sumo_cfg,
        max_steps=args.max_steps,
        queue_estimator=estimator,
        policy_fn=rule_based_policy,
    ))
    interface.start()
    try:
        interface.run()
    finally:
        interface.close()


if __name__ == "__main__":
    main()