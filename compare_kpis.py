"""
compare_kpis.py
====================
Reads the JSON files comparison.py / comparison_2.py already wrote:

    frontend/output/<scenario>.json           -- has "normal" and "astrid" keys
    frontend/output/ppo_model/<scenario>.json -- has "ppo" key

and prints/writes one side-by-side KPI table: astrid vs ppo, per
scenario, plus averages overall and split by TRAIN-distribution vs OOD
scenarios (any scenario name ending in "_OOD").

Only reads existing files -- runs no simulation, calls no controller,
touches no PPO/RF code. Safe to re-run anytime.

Note: collisions/teleports are NOT included in this table. The
astrid/normal leg (sumo_interface.py path) never logs that data at all,
so there is nothing on the astrid side to compare it against -- adding
it here would silently compare PPO's real collision count against an
implicit (and wrong) zero for astrid.

Usage:
    python compare_kpis.py --output-dir frontend/output
    python compare_kpis.py --output-dir frontend/output --csv-out compare_kpis.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


METRICS = [
    ("avg_queue_m", "Queue (m)", False),      # (json key, display label, higher_is_better)
    ("avg_wait_s", "Wait (s)", False),
    ("avg_speed_kmh", "Speed (km/h)", True),
    ("throughput_veh_per_hr", "Throughput (veh/hr)", True),
]


def load_scenario_kpis(output_dir: Path, ppo_dir: Path, scenario_name: str) -> Optional[Dict[str, dict]]:
    astrid_path = output_dir / f"{scenario_name}.json"
    ppo_path = ppo_dir / f"{scenario_name}.json"

    if not astrid_path.is_file():
        print(f"[skip] {scenario_name}: no astrid/normal JSON at {astrid_path}")
        return None
    if not ppo_path.is_file():
        print(f"[skip] {scenario_name}: no ppo JSON at {ppo_path}")
        return None

    with open(astrid_path, "r", encoding="utf-8") as f:
        astrid_payload = json.load(f)
    with open(ppo_path, "r", encoding="utf-8") as f:
        ppo_payload = json.load(f)

    if "astrid" not in astrid_payload or "kpis" not in astrid_payload["astrid"]:
        print(f"[skip] {scenario_name}: astrid JSON missing ['astrid']['kpis']")
        return None
    if "ppo" not in ppo_payload or "kpis" not in ppo_payload["ppo"]:
        print(f"[skip] {scenario_name}: ppo JSON missing ['ppo']['kpis']")
        return None

    return {
        "astrid": astrid_payload["astrid"]["kpis"],
        "ppo": ppo_payload["ppo"]["kpis"],
    }


def discover_scenarios(output_dir: Path, ppo_dir: Path) -> List[str]:
    """Scenarios present in BOTH indexes -- anything only on one side is
    reported and skipped, not silently dropped."""
    astrid_index_path = output_dir / "index.json"
    ppo_index_path = ppo_dir / "index.json"

    astrid_names, ppo_names = set(), set()
    if astrid_index_path.is_file():
        with open(astrid_index_path, "r", encoding="utf-8") as f:
            astrid_names = set(json.load(f).get("scenarios", []))
    if ppo_index_path.is_file():
        with open(ppo_index_path, "r", encoding="utf-8") as f:
            ppo_names = set(json.load(f).get("scenarios", []))

    only_astrid = astrid_names - ppo_names
    only_ppo = ppo_names - astrid_names
    if only_astrid:
        print(f"[note] scenarios with astrid results but no ppo results yet: {sorted(only_astrid)}")
    if only_ppo:
        print(f"[note] scenarios with ppo results but no astrid results: {sorted(only_ppo)}")

    return sorted(astrid_names & ppo_names)


def fmt_delta(astrid_val: float, ppo_val: float, higher_is_better: bool) -> str:
    diff = ppo_val - astrid_val
    if higher_is_better:
        better = diff > 0
    else:
        better = diff < 0
    arrow = "better" if better else "worse"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f} ({arrow})"


def print_group(title: str, rows: List[dict]) -> None:
    if not rows:
        return
    print(f"\n=== {title} ({len(rows)} scenario(s)) ===")
    header = f"{'scenario':<32}" + "".join(f"{label:>16}{'(ppo)':>10}{'delta':>18}" for _, label, _ in METRICS)
    print(header)
    for row in rows:
        line = f"{row['scenario']:<32}"
        for key, _, higher_is_better in METRICS:
            a = row["astrid"][key]
            p = row["ppo"][key]
            line += f"{a:>16.1f}{p:>10.1f}{fmt_delta(a, p, higher_is_better):>18}"
        print(line)

    # Group average
    avg_line = f"{'AVERAGE':<32}"
    for key, _, higher_is_better in METRICS:
        a_avg = sum(r["astrid"][key] for r in rows) / len(rows)
        p_avg = sum(r["ppo"][key] for r in rows) / len(rows)
        avg_line += f"{a_avg:>16.1f}{p_avg:>10.1f}{fmt_delta(a_avg, p_avg, higher_is_better):>18}"
    print(avg_line)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare astrid vs ppo KPIs from already-written comparison JSONs.")
    p.add_argument("--output-dir", type=str, default="frontend/output")
    p.add_argument("--scenarios", type=str, nargs="*", default=None,
                    help="Optional explicit scenario name list. Default: every scenario present in BOTH "
                         "frontend/output/index.json and frontend/output/ppo_model/index.json.")
    p.add_argument("--csv-out", type=str, default=None, help="Optional path to also write a CSV.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    ppo_dir = output_dir / "ppo_model"

    scenario_names = args.scenarios or discover_scenarios(output_dir, ppo_dir)
    if not scenario_names:
        print("No scenarios found with BOTH astrid and ppo results. Nothing to compare.")
        return

    rows = []
    for name in scenario_names:
        kpis = load_scenario_kpis(output_dir, ppo_dir, name)
        if kpis is None:
            continue
        rows.append({"scenario": name, **kpis})

    if not rows:
        print("No comparable scenarios found.")
        return

    train_rows = [r for r in rows if not r["scenario"].endswith("_OOD")]
    ood_rows = [r for r in rows if r["scenario"].endswith("_OOD")]

    print_group("ALL SCENARIOS", rows)
    print_group("TRAIN-DISTRIBUTION SCENARIOS", train_rows)
    print_group("OOD SCENARIOS", ood_rows)

    if args.csv_out:
        csv_path = Path(args.csv_out)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["scenario", "group"]
            for key, label, _ in METRICS:
                header += [f"astrid_{key}", f"ppo_{key}", f"delta_{key}"]
            writer.writerow(header)
            for r in rows:
                group = "ood" if r["scenario"].endswith("_OOD") else "train"
                line = [r["scenario"], group]
                for key, _, _ in METRICS:
                    a = r["astrid"][key]
                    p = r["ppo"][key]
                    line += [f"{a:.3f}", f"{p:.3f}", f"{p - a:.3f}"]
                writer.writerow(line)
        print(f"\n[done] wrote {csv_path}")


if __name__ == "__main__":
    main()