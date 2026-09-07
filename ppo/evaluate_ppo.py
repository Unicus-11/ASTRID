"""
ppo/evaluate_ppo.py  (PATCHED)
=======================
Same seed-determinism fix as train_ppo.py: every scenario is now run
with a fixed sumo_seed/estimator_seed (derived from --seed via CRC32),
so test/OOD evaluation is reproducible run-to-run and comparable
against the RF baseline on identical traffic. Also reports collisions
and teleports per scenario.

Run:
    python evaluate_ppo.py --model ppo_models/ppo_astrid_v1/final_model.zip --split test
    python evaluate_ppo.py --model ppo_models/ppo_astrid_v1/final_model.zip --split ood
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import ppo_config as cfg
from ppo_env import ASTRIDSignalEnv


def _deterministic_seed(scenario_id: str, base_seed: int) -> int:
    return zlib.crc32(f"{scenario_id}_{base_seed}".encode("utf-8")) % (2**31 - 1)


def run_episode(env: ASTRIDSignalEnv, model: PPO, scenario_id: str, base_seed: int) -> dict:
    obs, info = env.reset(
        options={
            "scenario_id": scenario_id,
            "sumo_seed": _deterministic_seed(scenario_id, base_seed),
            "estimator_seed": _deterministic_seed(scenario_id, base_seed + 1),
        }
    )
    done = False
    queues, waitings, speeds, arrivals = [], [], [], 0
    collisions, teleports = 0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(int(action))
        done = terminated or truncated
        queues.append(step_info["avg_queue_m"])
        waitings.append(step_info["avg_waiting_s"])
        speeds.append(step_info["avg_speed_mps"])
        arrivals += step_info["arrived_this_interval"]
        collisions += step_info["collisions"]
        teleports += step_info["teleports"]

    return {
        "scenario_id": scenario_id,
        "avg_queue_m": float(np.mean(queues)),
        "max_queue_m": float(np.max(queues)),
        "avg_waiting_s": float(np.mean(waitings)),
        "avg_speed_mps": float(np.mean(speeds)),
        "throughput_vehicles": arrivals,
        "switch_count": env._controller.switch_count,
        "forced_switch_count": env._controller.forced_switch_count,
        "collisions": collisions,
        "teleports": teleports,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model (no training).")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--split", type=str, choices=["validation", "test", "ood"], default="test")
    p.add_argument("--sumo-binary", type=str, default="sumo")
    p.add_argument("--seed", type=int, default=42, help="Base seed for deterministic per-scenario seeding.")
    p.add_argument("--out", type=str, default=None, help="Optional path to write JSON results")
    args = p.parse_args()

    run_cfg = cfg.PPORunConfig()
    run_cfg.sumo_binary = args.sumo_binary
    scenarios = {
        "validation": run_cfg.validation_scenarios,
        "test": run_cfg.test_scenarios,
        "ood": run_cfg.ood_scenarios,
    }[args.split]

    model = PPO.load(args.model)
    env = ASTRIDSignalEnv(run_cfg, scenarios, seed=0)

    results = [run_episode(env, model, s, args.seed) for s in scenarios]
    env.close()

    for r in results:
        print(json.dumps(r, indent=2))

    summary = {
        "split": args.split,
        "avg_queue_m": float(np.mean([r["avg_queue_m"] for r in results])),
        "avg_waiting_s": float(np.mean([r["avg_waiting_s"] for r in results])),
        "avg_speed_mps": float(np.mean([r["avg_speed_mps"] for r in results])),
        "avg_throughput": float(np.mean([r["throughput_vehicles"] for r in results])),
        "avg_switch_count": float(np.mean([r["switch_count"] for r in results])),
        "avg_collisions": float(np.mean([r["collisions"] for r in results])),
        "avg_teleports": float(np.mean([r["teleports"] for r in results])),
        "per_scenario": results,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()