"""
scenario_builder.py
====================
ASTRID Prototype -- V0 Scenario Generator  (v0.2)

RESPONSIBILITY (and only this):
    load scenario configuration
    generate a controlled, reproducible V0 scenario batch
    validate every scenario before writing it
    write scenario metadata + SUMO-ready flow/vType XML per scenario
    write a manifest

This script does NOT run SUMO, does NOT compute queue/density/speed/
shockwave values, does NOT train models, and does NOT control traffic
lights. A scenario is CAUSE-layer data only; nothing here may be
back-filled with an emergent traffic-state quantity.

v0.2 changes vs v0.1 (see chat for the full audit):
  - routes are now the real 3-edge [in_edge, via_edge, out_edge] chain,
    not a direct 2-edge shortcut
  - approach_name_to_edge corrected to the real compass directions
  - scenario.json now includes explicit normal_controller.py-compatible
    fields (name, demand, demand_rate) confirmed against its source
  - expected_regime_hint is now grounded in a Webster Y-ratio diagnostic
    computed with normal_controller.py's own saturation-flow constants,
    not an arbitrary demand-multiplier heuristic

Reads:
    scenario_config.json

Writes:
    generated_scenarios/manifest.json
    generated_scenarios/scenario_XXXX/scenario.json
    generated_scenarios/scenario_XXXX/flow.xml
    generated_scenarios/scenario_XXXX/vtype.xml

Run:
    python scenario_builder.py --config scenario_config.json --out generated_scenarios

Never touches sq.net.xml / sq.vtype.xml / sq.flow.xml / sq.rou.xml.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Data model
# ============================================================================

@dataclass
class Scenario:
    scenario_id: str
    seed: int
    split: str                      # "train" | "val" | "test" | "ood"
    design_method: str              # "hand_designed" | "stratified_lhs" | "seed_replication" | "ood"

    demand_class: str
    demand_rate_veh_per_hour: float

    approach_pattern: str
    approach_distribution: Dict[str, float]   # keys MUST be north/south/east/west (controller requirement)

    movement_pattern: str
    movement_distribution: Dict[str, float]   # keys: left/straight/right

    composition_pattern: str
    vehicle_composition: Dict[str, float]

    arrival_pattern: str

    simulation_begin: int
    simulation_end: int
    camera_range_m: float

    expected_regime_hint: str = ""   # heuristic PLANNING label only -- never measured data
    replicates: Optional[str] = None
    notes: str = ""


# ============================================================================
# Config
# ============================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Validation gate -- runs BEFORE anything is written to disk
# ============================================================================

def _sums_to_one(d: Dict[str, float], tol: float) -> bool:
    return abs(sum(d.values()) - 1.0) <= tol


def validate_scenario(s: Scenario, cfg: dict) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    tol = cfg["validation"]["proportion_sum_tolerance"]
    min_demand = cfg["validation"]["min_demand_veh_per_hour"]

    if s.demand_rate_veh_per_hour < min_demand:
        errors.append(f"demand_rate_veh_per_hour={s.demand_rate_veh_per_hour} below minimum {min_demand}")

    required_approach_keys = {"north", "south", "east", "west"}
    if set(s.approach_distribution.keys()) != required_approach_keys:
        errors.append(
            f"approach_distribution keys {set(s.approach_distribution.keys())} != "
            f"{required_approach_keys} (normal_controller.py requires exactly these keys)"
        )
    if not _sums_to_one(s.approach_distribution, tol):
        errors.append(f"approach_distribution sums to {sum(s.approach_distribution.values())}, expected 1.0")

    required_movement_keys = {"left", "straight", "right"}
    if set(s.movement_distribution.keys()) != required_movement_keys:
        errors.append(f"movement_distribution keys {set(s.movement_distribution.keys())} != {required_movement_keys}")
    if not _sums_to_one(s.movement_distribution, tol):
        errors.append(f"movement_distribution sums to {sum(s.movement_distribution.values())}, expected 1.0")

    if not _sums_to_one(s.vehicle_composition, tol):
        errors.append(f"vehicle_composition sums to {sum(s.vehicle_composition.values())}, expected 1.0")

    required_types = set(cfg["vehicle_types"].keys())
    if set(s.vehicle_composition.keys()) != required_types:
        errors.append(f"vehicle_composition keys {set(s.vehicle_composition.keys())} != required {required_types}")

    required_approach_edges = set(cfg["network"]["approaches"])
    mapped_edges = set(cfg["approach_name_to_edge"].values())
    if mapped_edges != required_approach_edges:
        errors.append("approach_name_to_edge does not cover exactly the network's approach edges")

    for in_edge, moves in cfg["network"]["movement_map"].items():
        if set(moves.keys()) != required_movement_keys:
            errors.append(f"movement_map[{in_edge}] must define exactly left/straight/right")
            continue
        for mv, chain in moves.items():
            if "via" not in chain or "out" not in chain:
                errors.append(f"movement_map[{in_edge}][{mv}] must define 'via' and 'out' edges")

    if s.simulation_begin >= s.simulation_end:
        errors.append(f"simulation_begin ({s.simulation_begin}) must be < simulation_end ({s.simulation_end})")

    if s.seed is None:
        errors.append("seed is missing")

    if s.arrival_pattern not in cfg["arrival_patterns_implemented"]:
        errors.append(
            f"arrival_pattern '{s.arrival_pattern}' is not implemented in V0 "
            f"(implemented: {cfg['arrival_patterns_implemented']})"
        )

    return (len(errors) == 0, errors)


# ============================================================================
# Latin Hypercube sampling (1-D) -- for the one continuous variable: demand
# ============================================================================

def lhs_1d(n: int, low: float, high: float, rng: random.Random) -> List[float]:
    """Latin Hypercube sample of n points in [low, high]: split the interval
    into n equal strata, draw one uniform sample per stratum, then shuffle
    so scenario index carries no implicit demand ranking."""
    width = (high - low) / n
    samples = []
    for i in range(n):
        lo = low + i * width
        hi = lo + width
        samples.append(rng.uniform(lo, hi))
    rng.shuffle(samples)
    return samples


def demand_class_for(multiplier: float, demand_bands: Dict[str, List[float]]) -> str:
    for name, (lo, hi) in demand_bands.items():
        if lo <= multiplier <= hi:
            return name
    return "OUT_OF_BAND"


# ============================================================================
# Webster Y-ratio diagnostic -- mirrors normal_controller.py's own capacity
# model so scenario selection is grounded in real numbers, not guesswork.
# THIS IS A PLANNING DIAGNOSTIC, NOT A MEASUREMENT.
# ============================================================================

def estimate_webster_y(scenario: Scenario, cfg: dict) -> dict:
    wref = cfg["_webster_reference"]
    sat_ns = wref["saturation_flow_per_lane_veh_per_hour"] * wref["lanes_ns"]
    sat_ew = wref["saturation_flow_per_lane_veh_per_hour"] * wref["lanes_ew"]

    ad = scenario.approach_distribution
    ns_demand = scenario.demand_rate_veh_per_hour * (ad["north"] + ad["south"])
    ew_demand = scenario.demand_rate_veh_per_hour * (ad["east"] + ad["west"])

    y_ns = ns_demand / sat_ns
    y_ew = ew_demand / sat_ew
    Y = y_ns + y_ew

    if Y < 0.85:
        regime = "undersaturated"
    elif Y < 1.0:
        regime = "near_saturation"
    else:
        regime = "oversaturated_by_webster_Y"

    return {
        "ns_demand_veh_per_hour": round(ns_demand, 1),
        "ew_demand_veh_per_hour": round(ew_demand, 1),
        "y_ns": round(y_ns, 3),
        "y_ew": round(y_ew, 3),
        "Y": round(Y, 3),
        "regime": regime,
        "_caveat": "Aggregate NS/EW capacity ratio only, using normal_controller.py's own saturation-flow "
                    "constants. Ignores per-movement (left/straight/right) capacity, stochastic arrival "
                    "variance, and signal cycling detail -- it is a planning diagnostic, not a queue prediction.",
    }


def expected_regime_hint(webster: dict, approach_pattern: str, arrival_pattern: str) -> str:
    """Heuristic PLANNING label, now grounded in the Webster Y-ratio diagnostic
    rather than an arbitrary multiplier. Still not a measurement."""
    base = {
        "undersaturated": "likely free-flow to moderate",
        "near_saturation": "near capacity by Webster Y -- congestion plausible",
        "oversaturated_by_webster_Y": "oversaturated by Webster Y -- queue growth likely, "
                                       "possible queue beyond camera range",
    }[webster["regime"]]

    if approach_pattern == "single_heavy":
        base += " (demand concentrated on one approach)"
    if arrival_pattern == "burst":
        base += " (burst arrival -- expect transient shockwave growth)"
    return base


# ============================================================================
# Demand allocation: total -> approach -> movement (12 (in_edge,movement) cells)
# ============================================================================

def allocate_demand(
    total_veh_per_hour: float,
    approach_distribution: Dict[str, float],
    movement_distribution: Dict[str, float],
    approach_name_to_edge: Dict[str, str],
    duration_hours: float,
) -> Tuple[Dict[Tuple[str, str], float], dict]:
    """Cascade total demand -> approach -> movement. Keyed by (in_edge, movement)
    rather than (in_edge, out_edge): with the real 3-edge route structure,
    'movement' (left/straight/right) is the natural key -- the via/out edges
    are looked up from movement_map only when building the route XML."""
    od: Dict[Tuple[str, str], float] = {}

    for approach_name, approach_share in approach_distribution.items():
        in_edge = approach_name_to_edge[approach_name]
        approach_demand = total_veh_per_hour * approach_share
        for movement, movement_share in movement_distribution.items():
            od[(in_edge, movement)] = approach_demand * movement_share

    allocated_total = sum(od.values())
    report = {
        "requested_total_veh_per_hour": total_veh_per_hour,
        "allocated_total_veh_per_hour": round(allocated_total, 4),
        "veh_per_hour_difference": round(total_veh_per_hour - allocated_total, 6),
        "requested_vehicle_count_over_window": round(total_veh_per_hour * duration_hours),
        "expected_vehicle_count_over_window": round(allocated_total * duration_hours),
    }
    return od, report


def route_edges_for(in_edge: str, movement: str, cfg: dict) -> List[str]:
    """Canonical route = [in_edge, via, out], sourced directly from the real
    network structure now confirmed in movement_map. route_edge_override is
    an escape hatch only, not the default path."""
    override = cfg["network"].get("route_edge_override") or {}
    if in_edge in override and movement in override[in_edge]:
        return override[in_edge][movement]
    chain = cfg["network"]["movement_map"][in_edge][movement]
    return [in_edge, chain["via"], chain["out"]]


# ============================================================================
# SUMO XML generation
# ============================================================================

def build_vtype_xml(scenario: Scenario, cfg: dict) -> str:
    dist_id = f"mix_{scenario.scenario_id}"
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]
    lines.append(f'    <vTypeDistribution id="{dist_id}">')
    for vt_id, share in scenario.vehicle_composition.items():
        p = cfg["vehicle_types"][vt_id]
        lines.append(
            f'        <vType id="{vt_id}" length="{p["length"]}" '
            f'maxSpeed="{p["maxSpeed"]}" accel="{p["accel"]}" decel="{p["decel"]}" '
            f'probability="{share}"/>'
        )
    lines.append("    </vTypeDistribution>")
    lines.append("</additional>")
    return "\n".join(lines) + "\n"


def build_flow_xml(scenario: Scenario, od: Dict[Tuple[str, str], float], cfg: dict) -> str:
    dist_id = f"mix_{scenario.scenario_id}"
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]

    route_ids: Dict[Tuple[str, str], str] = {}
    for (in_edge, movement) in od:
        rid = f"rt_{in_edge}_{movement}"
        route_ids[(in_edge, movement)] = rid
        edges = " ".join(route_edges_for(in_edge, movement, cfg))
        lines.append(f'    <route id="{rid}" edges="{edges}"/>')

    begin, end = scenario.simulation_begin, scenario.simulation_end
    duration = end - begin

    if scenario.arrival_pattern == "constant":
        segments = [(begin, end, 1.0)]
    elif scenario.arrival_pattern == "burst":
        raw = [(0.0, 0.4, 0.6), (0.4, 0.6, 3.0), (0.6, 1.0, 0.6)]
        avg = sum(w * (b - a) for a, b, w in raw)
        segments = [
            (begin + round(a * duration), begin + round(b * duration), w / avg)
            for a, b, w in raw
        ]
    else:
        raise NotImplementedError(f"arrival_pattern '{scenario.arrival_pattern}' is not implemented in V0")

    idx = 0
    for (in_edge, movement), veh_per_hour in od.items():
        rid = route_ids[(in_edge, movement)]
        for seg_begin, seg_end, multiplier in segments:
            rate = round(veh_per_hour * multiplier, 4)
            if rate <= 0 or seg_end <= seg_begin:
                continue
            fid = f"f_{in_edge}_{movement}_{idx}"
            idx += 1
            lines.append(
                f'    <flow id="{fid}" type="{dist_id}" route="{rid}" '
                f'begin="{seg_begin}" end="{seg_end}" vehsPerHour="{rate}" '
                f'departLane="best" departSpeed="max"/>'
            )

    lines.append("</routes>")
    return "\n".join(lines) + "\n"


# ============================================================================
# V0 scenario design (hybrid strategy -- unchanged from v0.1, see chat)
# ============================================================================

def generate_v0_scenarios(cfg: dict) -> List[Scenario]:
    rng = random.Random(cfg["v0_design"]["base_seed"])
    base_seed = cfg["v0_design"]["base_seed"]
    baseline_demand = cfg["baseline"]["total_demand_veh_per_hour"]
    demand_bands = cfg["demand_bands"]
    sim_begin = cfg["simulation"]["begin_s"]
    sim_end = cfg["simulation"]["end_s"]
    camera_range = cfg["network"]["camera_range_m"]

    scenarios: List[Scenario] = []
    idx = 1

    def make(
        design_method: str, split: str, mult: float,
        approach_pattern: str, movement_pattern: str, composition_pattern: str,
        arrival_pattern: str, seed: int, notes: str = "", replicates: Optional[str] = None,
    ) -> Scenario:
        nonlocal idx
        sid = f"scenario_{idx:04d}"
        idx += 1
        demand_rate = round(baseline_demand * mult, 2)
        s = Scenario(
            scenario_id=sid,
            seed=seed,
            split=split,
            design_method=design_method,
            demand_class=demand_class_for(mult, demand_bands),
            demand_rate_veh_per_hour=demand_rate,
            approach_pattern=approach_pattern,
            approach_distribution=cfg["approach_patterns"][approach_pattern],
            movement_pattern=movement_pattern,
            movement_distribution=cfg["movement_patterns"][movement_pattern],
            composition_pattern=composition_pattern,
            vehicle_composition=cfg["composition_patterns"][composition_pattern],
            arrival_pattern=arrival_pattern,
            simulation_begin=sim_begin,
            simulation_end=sim_end,
            camera_range_m=camera_range,
            notes=notes,
            replicates=replicates,
        )
        webster = estimate_webster_y(s, cfg)
        s.expected_regime_hint = expected_regime_hint(webster, approach_pattern, arrival_pattern)
        return s

    # -- Phase A: hand-designed edge cases --
    edge_cases = [
        make("hand_designed", "train", 1.90, "single_heavy", "straight_heavy",
             "baseline_heterogeneous", "constant", base_seed + 1,
             notes="Designed to plausibly push queue on the heavy approach past the 150m camera range."),
        make("hand_designed", "train", 1.40, "balanced", "balanced",
             "baseline_heterogeneous", "burst", base_seed + 2,
             notes="Tests transient shockwave growth from a sudden arrival surge."),
        make("hand_designed", "train", 1.00, "east_west_heavy", "left_heavy",
             "baseline_heterogeneous", "constant", base_seed + 3,
             notes="Moderate total volume, structurally concentrated by approach+turn -- interaction-effect test."),
        make("hand_designed", "train", 1.35, "balanced", "balanced",
             "hgv_heavy", "constant", base_seed + 4,
             notes="Tests effective-capacity loss from a larger/slower vehicle mix."),
    ]
    scenarios.extend(edge_cases)

    # -- Phase B: stratified categorical scaffold + 1-D LHS demand sampling --
    composition_list = list(cfg["composition_patterns"].keys())
    movement_list = list(cfg["movement_patterns"].keys())
    approach_list = list(cfg["approach_patterns"].keys())
    n_strat = cfg["v0_design"]["stratified_lhs_scenarios"]

    demand_mults = lhs_1d(n_strat, demand_bands["VERY_LOW"][0], demand_bands["VERY_HIGH"][1], rng)
    strat_splits = ["train", "train", "train", "train", "val", "test"]
    assert len(strat_splits) == n_strat

    stratified: List[Scenario] = []
    for i in range(n_strat):
        comp = composition_list[i % len(composition_list)]
        move = movement_list[(i + 1) % len(movement_list)]
        appr = approach_list[(i + 2) % len(approach_list)]
        s = make(
            "stratified_lhs", strat_splits[i], demand_mults[i],
            appr, move, comp, "constant", base_seed + 100 + i,
            notes="Categorical archetype via stratified round-robin; demand via 1-D LHS.",
        )
        stratified.append(s)
    scenarios.extend(stratified)

    # -- Phase C: seed replication --
    original = stratified[0]
    replication = make(
        "seed_replication", original.split, demand_mults[0],
        original.approach_pattern, original.movement_pattern, original.composition_pattern,
        original.arrival_pattern, base_seed + 900,
        notes=f"Identical parameters to {original.scenario_id}, different seed only.",
        replicates=original.scenario_id,
    )
    scenarios.append(replication)

    # -- Phase D: explicit OOD holdout --
    ood = make(
        "ood", "ood", 1.75, "north_heavy", "left_heavy", "bike_heavy", "burst", base_seed + 999,
        notes="Deliberately unusual combination, reserved for generalization testing. "
              "Must never be used for training/validation/test-set model selection.",
    )
    scenarios.append(ood)

    return scenarios


# ============================================================================
# Writing scenarios to disk
# ============================================================================

def _od_diagnostics(od: Dict[Tuple[str, str], float], cfg: dict) -> Dict[str, float]:
    out = {}
    for (in_edge, movement), v in od.items():
        chain = cfg["network"]["movement_map"][in_edge][movement]
        label = f"{in_edge}[{movement}]->{chain['via']}->{chain['out']}"
        out[label] = round(v, 4)
    return out


def write_scenario(out_dir: Path, scenario: Scenario, cfg: dict) -> dict:
    duration_hours = (scenario.simulation_end - scenario.simulation_begin) / 3600.0
    od, alloc_report = allocate_demand(
        scenario.demand_rate_veh_per_hour,
        scenario.approach_distribution,
        scenario.movement_distribution,
        cfg["approach_name_to_edge"],
        duration_hours,
    )

    scenario_dir = out_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    webster = estimate_webster_y(scenario, cfg)

    scenario_dict = asdict(scenario)
    # -- normal_controller.py compatibility fields (confirmed against its source) --
    scenario_dict["name"] = scenario.scenario_id
    scenario_dict["demand"] = scenario.demand_class          # label, matches controller's print usage
    scenario_dict["demand_rate"] = scenario.demand_rate_veh_per_hour   # numeric, matches controller's math
    # -- diagnostics (planning aids, never traffic-state ground truth) --
    scenario_dict["_demand_allocation_diagnostics"] = alloc_report
    scenario_dict["_od_diagnostics"] = _od_diagnostics(od, cfg)
    scenario_dict["_webster_diagnostic"] = webster

    with open(scenario_dir / "scenario.json", "w", encoding="utf-8") as f:
        json.dump(scenario_dict, f, indent=2)

    with open(scenario_dir / "vtype.xml", "w", encoding="utf-8") as f:
        f.write(build_vtype_xml(scenario, cfg))

    with open(scenario_dir / "flow.xml", "w", encoding="utf-8") as f:
        f.write(build_flow_xml(scenario, od, cfg))

    return {
        "scenario_id": scenario.scenario_id,
        "split": scenario.split,
        "design_method": scenario.design_method,
        "seed": scenario.seed,
        "demand_class": scenario.demand_class,
        "demand_rate_veh_per_hour": scenario.demand_rate_veh_per_hour,
        "approach_pattern": scenario.approach_pattern,
        "movement_pattern": scenario.movement_pattern,
        "composition_pattern": scenario.composition_pattern,
        "arrival_pattern": scenario.arrival_pattern,
        "webster_Y": webster["Y"],
        "webster_regime": webster["regime"],
        "expected_regime_hint": scenario.expected_regime_hint,
        "output_dir": str(scenario_dir),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRID V0 scenario generator")
    parser.add_argument("--config", default="scenario_config.json")
    parser.add_argument("--out", default="generated_scenarios")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = generate_v0_scenarios(cfg)

    manifest_entries = []
    rejected = []

    for s in scenarios:
        ok, errors = validate_scenario(s, cfg)
        if not ok:
            rejected.append({"scenario_id": s.scenario_id, "errors": errors})
            continue
        entry = write_scenario(out_dir, s, cfg)
        manifest_entries.append(entry)

    manifest = {
        "generator": "scenario_builder.py",
        "config_version": cfg["meta"]["config_version"],
        "strategy": cfg["v0_design"]["strategy"],
        "requested_count": cfg["v0_design"]["target_scenario_count"],
        "written_count": len(manifest_entries),
        "rejected_count": len(rejected),
        "scenarios": manifest_entries,
        "rejected": rejected,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    header = (f"{'scenario_id':13} {'split':6} {'method':17} {'seed':6} {'demand':22} "
              f"{'approach':15} {'movement':15} {'arrival':9} {'Y':6} regime")
    print(header)
    print("-" * len(header))
    for e in manifest_entries:
        demand_str = f"{e['demand_class']}({e['demand_rate_veh_per_hour']:.0f}/h)"
        print(
            f"{e['scenario_id']:13} {e['split']:6} {e['design_method']:17} {e['seed']:<6} "
            f"{demand_str:22} {e['approach_pattern']:15} {e['movement_pattern']:15} "
            f"{e['arrival_pattern']:9} {e['webster_Y']:<6} {e['expected_regime_hint']}"
        )

    if rejected:
        print("\nREJECTED (failed validation, NOT written to disk):")
        for r in rejected:
            print(f"  {r['scenario_id']}: {r['errors']}")

    print(f"\nWrote {len(manifest_entries)}/{len(scenarios)} scenarios to {out_dir.resolve()}")
    print(f"Manifest: {(out_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()