"""
scenario_builder.py
====================
ASTRID Prototype -- Scenario Generator  (v0.4.0)

v0.4.0 IS A DELIBERATE REDESIGN, NOT AN INCREMENTAL FIX.
The earlier v0.2-v0.6 versions generated 12-40 scenarios via Latin
Hypercube demand sampling stratified by Webster-Y regime. That machinery
is REMOVED ENTIRELY in this version (not capped, not disabled behind a
flag -- deleted: lhs_1d, compute_regime_multiplier_bounds,
assign_splits_by_regime, and the hand-designed-edge-case /
seed-replication phases are all gone).

In their place: exactly 12 hand-authored, individually-named scenarios --
8 development + 4 OOD -- defined once in SCENARIO_DEFINITIONS below. This
matches the project's Phase 1 research goal directly: compare several
queue-length-estimation models on 8 development scenarios, then check
which ones still work on 4 deliberately out-of-distribution scenarios that
never touch training, tuning, or model selection.

Each OOD scenario is deliberately MORE EXTREME than its closest
development-scenario analogue, not just a relabeling:
    high_demand        (mult=1.30, HIGH band)       vs very_high_demand_OOD (mult=1.90, VERY_HIGH band)
    north_heavy         (north 45%)                  vs north_extreme_OOD    (north 70%, a NEW approach_pattern)
    (no development burst scenario)                  vs burst_demand_OOD     (arrival_pattern=burst, unseen during dev)
    (baseline ~20% heavy vehicles)                    vs heavy_vehicle_OOD    (60% heavy vehicles, a NEW composition_pattern)

RESPONSIBILITY (and only this):
    load scenario configuration
    generate exactly the 12 defined scenarios
    validate every scenario before writing it (including per-value
    probability-range checks, not just that each distribution sums to 1)
    write scenario metadata + SUMO-ready flow/vType XML per scenario
    verify the written flow.xml is actually begin-time-ordered (regression
    guard for the burst-ordering bug fixed in a prior version)
    write a manifest that explicitly labels every scenario development/OOD

This script does NOT run SUMO, does NOT compute queue/density/speed/
shockwave values, does NOT train models, and does NOT control traffic
lights. A scenario is CAUSE-layer data only.

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
import sys
import xml.etree.ElementTree as ET
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
    design_method: str              # "development" | "ood"

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
    webster_regime: str = ""         # informational only in v0.4.0 -- does NOT drive split assignment
    notes: str = ""


# ============================================================================
# Config
# ============================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


REQUIRED_CONFIG_KEYS = {
    "network": ["approaches", "camera_range_m", "movement_map"],
    "approach_name_to_edge": None,
    "vehicle_types": None,
    "baseline": ["total_demand_veh_per_hour"],
    "demand_bands": ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
    "approach_patterns": ["balanced", "north_heavy", "south_heavy", "east_west_heavy", "north_extreme"],
    "movement_patterns": ["balanced", "straight_heavy", "left_heavy"],
    "composition_patterns": ["baseline_heterogeneous", "hgv_heavy", "heavy_vehicle_extreme"],
    "arrival_patterns_implemented": None,
    "simulation": ["begin_s", "end_s"],
    "v0_design": ["target_scenario_count", "development_scenario_count", "ood_scenario_count", "base_seed"],
    "validation": ["proportion_sum_tolerance", "min_demand_veh_per_hour"],
    "_webster_reference": ["saturation_flow_per_lane_veh_per_hour", "lanes_ns", "lanes_ew"],
}


def validate_config(cfg: dict) -> None:
    """v0.4.0 addition: previously, a missing config key (e.g. demand_bands
    without VERY_LOW/VERY_HIGH) surfaced as a raw KeyError deep inside
    generation with no context. Checked explicitly, up front, with a
    message naming exactly what's missing -- SCENARIO_DEFINITIONS below
    references specific approach/movement/composition pattern names, so a
    missing one would otherwise fail confusingly partway through generation."""
    problems: List[str] = []
    for key, subkeys in REQUIRED_CONFIG_KEYS.items():
        if key not in cfg:
            problems.append(f"missing top-level key '{key}'")
            continue
        if subkeys is None:
            continue
        for subkey in subkeys:
            if subkey not in cfg[key]:
                problems.append(f"missing '{subkey}' under '{key}'")
    if problems:
        raise ValueError(
            "scenario_config.json is missing required keys:\n  " + "\n  ".join(problems)
        )


# ============================================================================
# Validation gate -- runs BEFORE anything is written to disk
# ============================================================================

def _valid_distribution(d: Dict[str, float], required_keys: set, tol: float) -> Tuple[bool, str]:
    """v0.4.0 fix: the previous validator only checked that a distribution's
    values SUMMED to 1.0 -- {"north": 1.2, "south": -0.2, "east": 0, "west": 0}
    would have passed. Now also checks every individual value is a valid
    probability (0 <= v <= 1)."""
    if set(d.keys()) != required_keys:
        return False, f"keys {set(d.keys())} != {required_keys}"
    out_of_range = {k: v for k, v in d.items() if not (0.0 <= v <= 1.0)}
    if out_of_range:
        return False, f"value(s) out of [0,1] range: {out_of_range}"
    total = sum(d.values())
    if abs(total - 1.0) > tol:
        return False, f"sums to {total}, expected 1.0"
    return True, ""


def validate_scenario(s: Scenario, cfg: dict) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    tol = cfg["validation"]["proportion_sum_tolerance"]
    min_demand = cfg["validation"]["min_demand_veh_per_hour"]

    if s.demand_rate_veh_per_hour < min_demand:
        errors.append(f"demand_rate_veh_per_hour={s.demand_rate_veh_per_hour} below minimum {min_demand}")

    ok, msg = _valid_distribution(s.approach_distribution, {"north", "south", "east", "west"}, tol)
    if not ok:
        errors.append(f"approach_distribution invalid: {msg} (normal_controller.py requires exactly north/south/east/west)")

    ok, msg = _valid_distribution(s.movement_distribution, {"left", "straight", "right"}, tol)
    if not ok:
        errors.append(f"movement_distribution invalid: {msg}")

    required_types = set(cfg["vehicle_types"].keys())
    ok, msg = _valid_distribution(s.vehicle_composition, required_types, tol)
    if not ok:
        errors.append(f"vehicle_composition invalid: {msg}")

    required_approach_edges = set(cfg["network"]["approaches"])
    mapped_edges = set(cfg["approach_name_to_edge"].values())
    if mapped_edges != required_approach_edges:
        errors.append("approach_name_to_edge does not cover exactly the network's approach edges")

    required_movement_keys = {"left", "straight", "right"}
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

    if s.split not in {"train", "val", "test", "ood"}:
        errors.append(f"split '{s.split}' is not one of train/val/test/ood")

    if s.design_method not in {"development", "ood"}:
        errors.append(f"design_method '{s.design_method}' is not 'development' or 'ood'")

    if s.design_method == "ood" and s.split != "ood":
        errors.append(f"design_method='ood' but split='{s.split}' -- OOD scenarios must have split='ood'")
    if s.design_method == "development" and s.split == "ood":
        errors.append("design_method='development' but split='ood' -- development scenarios must not use split='ood'")

    return (len(errors) == 0, errors)


# ============================================================================
# Webster Y-ratio diagnostic -- INFORMATIONAL ONLY in v0.4.0.
# It does NOT drive scenario generation or split assignment (that
# machinery was removed -- see module docstring). Kept only because it's a
# useful per-scenario diagnostic to have in scenario.json/the report.
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
        "_caveat": "Informational diagnostic only in v0.4.0 -- does not drive scenario generation or "
                    "split assignment. Aggregate NS/EW capacity ratio only; ignores per-movement "
                    "capacity, stochastic arrival variance, and signal cycling detail.",
    }


def demand_class_for(mult: float, demand_bands: Dict[str, List[float]]) -> str:
    for name, (lo, hi) in demand_bands.items():
        if lo <= mult <= hi:
            return name
    return "OUT_OF_BAND"


def expected_regime_hint(webster: dict, notes_extra: str = "") -> str:
    base = {
        "undersaturated": "likely free-flow to moderate",
        "near_saturation": "near capacity by Webster Y -- congestion plausible",
        "oversaturated_by_webster_Y": "oversaturated by Webster Y -- queue growth likely, "
                                       "possible queue beyond camera range",
    }[webster["regime"]]
    return base + (f" ({notes_extra})" if notes_extra else "")


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
    """Flow entries are collected for every (in_edge, movement) pair FIRST,
    then sorted globally by (begin, end, flow_id) before being written.

    WHY THIS MATTERS: SUMO requires <flow> elements in a route file to
    appear in non-decreasing 'begin' order. Writing all segments for one
    movement before moving to the next (the original approach) works fine
    for 'constant' (one segment per movement) but breaks for 'burst'
    (three segments per movement: 0-1440, 1440-2160, 2160-3600) -- the
    file's begin values would go 0,1440,2160 for the first movement, then
    back to 0 for the second movement's first segment. Observed in earlier
    generated runs: that non-monotonic ordering resulted in only each
    movement's FINAL segment producing any vehicles (the other two
    segments' vehicle IDs never appeared in the corresponding SUMO run's
    trajectory output). That is project-level empirical evidence, not a
    claim about SUMO's internals verified against its source/documentation
    -- phrased that way deliberately."""
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

    flow_entries: List[Tuple[float, float, str, str]] = []
    idx = 0
    for (in_edge, movement), veh_per_hour in od.items():
        rid = route_ids[(in_edge, movement)]
        for seg_begin, seg_end, multiplier in segments:
            rate = round(veh_per_hour * multiplier, 4)
            if rate <= 0 or seg_end <= seg_begin:
                continue
            fid = f"f_{in_edge}_{movement}_{idx}"
            idx += 1
            xml_line = (
                f'    <flow id="{fid}" type="{dist_id}" route="{rid}" '
                f'begin="{seg_begin}" end="{seg_end}" vehsPerHour="{rate}" '
                f'departLane="best" departSpeed="max"/>'
            )
            flow_entries.append((seg_begin, seg_end, fid, xml_line))

    flow_entries.sort(key=lambda e: (e[0], e[1], e[2]))
    lines.extend(entry[3] for entry in flow_entries)

    lines.append("</routes>")
    return "\n".join(lines) + "\n"


# ============================================================================
# Post-write structural validation (v0.4.0 addition)
# ============================================================================

def validate_flow_ordering(flow_xml_text: str) -> Tuple[bool, str]:
    """Regression guard for the burst-ordering bug: parses the ACTUAL
    written flow.xml and asserts every <flow> element's begin time is
    non-decreasing in file order. This is exactly the invariant SUMO
    requires -- checking it here catches a regression immediately, before
    an expensive SUMO run, rather than discovering it later via a
    suspiciously low departed-vehicle count."""
    root = ET.fromstring(flow_xml_text)
    begins = [float(f.get("begin", 0)) for f in root.findall("flow")]
    for i in range(1, len(begins)):
        if begins[i] < begins[i - 1]:
            return False, f"flow #{i} has begin={begins[i]} < previous begin={begins[i-1]} -- non-monotonic"
    return True, f"{len(begins)} flow(s), begin times non-decreasing"


def validate_generated_xml(flow_xml_text: str, vtype_xml_text: str) -> List[str]:
    """Lightweight structural validation (well-formed XML + internal
    consistency) WITHOUT invoking SUMO itself -- this file's job is scenario
    generation, not simulation (see module docstring). Full SUMO-level
    validation happens naturally the first time sumo/run_scenarios.py loads
    these files."""
    problems: List[str] = []
    try:
        flow_root = ET.fromstring(flow_xml_text)
    except ET.ParseError as exc:
        return [f"flow.xml is not well-formed XML: {exc}"]
    try:
        vtype_root = ET.fromstring(vtype_xml_text)
    except ET.ParseError as exc:
        return [f"vtype.xml is not well-formed XML: {exc}"]

    route_ids = {r.get("id") for r in flow_root.findall("route")}
    for flow in flow_root.findall("flow"):
        if flow.get("route") not in route_ids:
            problems.append(f"flow '{flow.get('id')}' references undefined route '{flow.get('route')}'")
        for edge in flow_root.find(f".//route[@id='{flow.get('route')}']").get("edges", "").split():
            if not edge:
                problems.append(f"route '{flow.get('route')}' has an empty edge token")

    vtype_dist = vtype_root.find("vTypeDistribution")
    if vtype_dist is None:
        problems.append("vtype.xml has no <vTypeDistribution> element")
    else:
        probs = [float(v.get("probability", 0)) for v in vtype_dist.findall("vType")]
        if abs(sum(probs) - 1.0) > 1e-4:
            problems.append(f"vTypeDistribution probabilities sum to {sum(probs)}, expected ~1.0")

    ok, msg = validate_flow_ordering(flow_xml_text)
    if not ok:
        problems.append(f"flow ordering: {msg}")

    return problems


# ============================================================================
# v0.4.0: the fixed 12-scenario definition
# ============================================================================
#
# Split assignment for the 8 development scenarios (explicit, documented,
# not automatically balanced -- there are too few scenarios for meaningful
# per-regime stratification, so this is a plain, readable assignment):
#
#   train (4): normal_balanced, low_demand, high_demand, left_turn_heavy
#   val   (2): north_heavy, straight_heavy
#   test  (2): south_heavy, east_west_heavy
#
# All 4 OOD scenarios get split="ood" and design_method="ood", and MUST
# NEVER be used for training, feature fitting, tuning, or model selection
# (enforced downstream by dataset/assemble_dataset.py + the OOD-never-gates
# rule in models/tree_model.py -- this file only labels them correctly).

SCENARIO_DEFINITIONS: List[dict] = [
    # ---- Development (8) ----
    dict(name="normal_balanced", split="train", mult=1.00,
         approach="balanced", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="Baseline reference scenario: normal demand, balanced approaches, normal composition, steady arrivals."),
    dict(name="low_demand", split="train", mult=0.50,
         approach="balanced", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="Substantially lower demand than normal; otherwise normal traffic."),
    dict(name="high_demand", split="train", mult=1.30,
         approach="balanced", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="Higher demand than normal, steady arrivals -- deliberately NOT as extreme as very_high_demand_OOD (mult 1.30 vs 1.90)."),
    dict(name="north_heavy", split="val", mult=1.00,
         approach="north_heavy", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="North approach carries substantially more traffic (45%) -- deliberately less extreme than north_extreme_OOD (70%)."),
    dict(name="south_heavy", split="test", mult=1.00,
         approach="south_heavy", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="South approach carries substantially more traffic (55%)."),
    dict(name="east_west_heavy", split="test", mult=1.00,
         approach="east_west_heavy", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="East/west approaches carry substantially more traffic; north/south lighter."),
    dict(name="left_turn_heavy", split="train", mult=1.00,
         approach="balanced", movement="left_heavy", composition="baseline_heterogeneous", arrival="constant",
         notes="Unusually high proportion of left-turn movements; total demand stays in the normal development range."),
    dict(name="straight_heavy", split="val", mult=1.00,
         approach="balanced", movement="straight_heavy", composition="baseline_heterogeneous", arrival="constant",
         notes="Unusually high proportion of straight movements; total demand stays in the normal development range."),

    # ---- OOD (4) -- never used for training/tuning/model selection ----
    dict(name="very_high_demand_OOD", split="ood", mult=1.90,
         approach="balanced", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="Demand substantially outside the normal training range (mult 1.90, VERY_HIGH band) -- "
               "intentionally more congested than high_demand (mult 1.30, HIGH band)."),
    dict(name="north_extreme_OOD", split="ood", mult=1.00,
         approach="north_extreme", movement="balanced", composition="baseline_heterogeneous", arrival="constant",
         notes="Extremely imbalanced directional demand (north 70%) -- much more extreme than north_heavy (45%). "
               "Total demand held equal to north_heavy so the comparison isolates approach-imbalance, not demand level."),
    dict(name="burst_demand_OOD", split="ood", mult=1.20,
         approach="balanced", movement="balanced", composition="baseline_heterogeneous", arrival="burst",
         notes="Strongly time-varying demand with a clear burst (0.56x/2.78x/0.56x across the hour) -- "
               "no development scenario uses the burst arrival pattern, so this is genuinely unseen during development."),
    dict(name="heavy_vehicle_OOD", split="ood", mult=1.00,
         approach="balanced", movement="balanced", composition="heavy_vehicle_extreme", arrival="constant",
         notes="Unusually high proportion of heavy vehicles (60% hgv+bus via heavy_vehicle_extreme) -- "
               "substantially outside the normal composition range (baseline is 20% hgv+bus)."),
]

assert len(SCENARIO_DEFINITIONS) == 12
assert sum(1 for d in SCENARIO_DEFINITIONS if d["split"] != "ood") == 8
assert sum(1 for d in SCENARIO_DEFINITIONS if d["split"] == "ood") == 4


def generate_v0_scenarios(cfg: dict) -> List[Scenario]:
    validate_config(cfg)

    base_seed = cfg["v0_design"]["base_seed"]
    baseline_demand = cfg["baseline"]["total_demand_veh_per_hour"]
    demand_bands = cfg["demand_bands"]
    sim_begin = cfg["simulation"]["begin_s"]
    sim_end = cfg["simulation"]["end_s"]
    camera_range = cfg["network"]["camera_range_m"]

    scenarios: List[Scenario] = []
    for i, d in enumerate(SCENARIO_DEFINITIONS):
        seed = base_seed + i + 1  # fixed, deterministic, unique per scenario
        demand_rate = round(baseline_demand * d["mult"], 2)
        design_method = "ood" if d["split"] == "ood" else "development"

        s = Scenario(
            scenario_id=f"scenario_{d['name']}",
            seed=seed,
            split=d["split"],
            design_method=design_method,
            demand_class=demand_class_for(d["mult"], demand_bands),
            demand_rate_veh_per_hour=demand_rate,
            approach_pattern=d["approach"],
            approach_distribution=cfg["approach_patterns"][d["approach"]],
            movement_pattern=d["movement"],
            movement_distribution=cfg["movement_patterns"][d["movement"]],
            composition_pattern=d["composition"],
            vehicle_composition=cfg["composition_patterns"][d["composition"]],
            arrival_pattern=d["arrival"],
            simulation_begin=sim_begin,
            simulation_end=sim_end,
            camera_range_m=camera_range,
            notes=d["notes"],
        )
        webster = estimate_webster_y(s, cfg)
        s.webster_regime = webster["regime"]
        s.expected_regime_hint = expected_regime_hint(webster, d["notes"][:40])
        scenarios.append(s)

    # -- Enforced bookkeeping (not decorative): the config's declared counts
    # must match what SCENARIO_DEFINITIONS actually produces. --
    configured_total = cfg["v0_design"]["target_scenario_count"]
    configured_dev = cfg["v0_design"]["development_scenario_count"]
    configured_ood = cfg["v0_design"]["ood_scenario_count"]
    actual_dev = sum(1 for s in scenarios if s.design_method == "development")
    actual_ood = sum(1 for s in scenarios if s.design_method == "ood")

    mismatches = []
    if configured_total != len(scenarios):
        mismatches.append(f"target_scenario_count={configured_total} != actual {len(scenarios)}")
    if configured_dev != actual_dev:
        mismatches.append(f"development_scenario_count={configured_dev} != actual {actual_dev}")
    if configured_ood != actual_ood:
        mismatches.append(f"ood_scenario_count={configured_ood} != actual {actual_ood}")
    if mismatches:
        raise ValueError(
            "scenario_config.json's v0_design counts do not match SCENARIO_DEFINITIONS:\n  "
            + "\n  ".join(mismatches)
        )

    seeds = [s.seed for s in scenarios]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds generated: {seeds}")
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scenario_id generated: {ids}")

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
    scenario_dict["name"] = scenario.scenario_id
    scenario_dict["demand"] = scenario.demand_class
    scenario_dict["demand_rate"] = scenario.demand_rate_veh_per_hour
    scenario_dict["_demand_allocation_diagnostics"] = alloc_report
    scenario_dict["_od_diagnostics"] = _od_diagnostics(od, cfg)
    scenario_dict["_webster_diagnostic"] = webster

    with open(scenario_dir / "scenario.json", "w", encoding="utf-8") as f:
        json.dump(scenario_dict, f, indent=2)

    vtype_xml = build_vtype_xml(scenario, cfg)
    flow_xml = build_flow_xml(scenario, od, cfg)

    with open(scenario_dir / "vtype.xml", "w", encoding="utf-8") as f:
        f.write(vtype_xml)
    with open(scenario_dir / "flow.xml", "w", encoding="utf-8") as f:
        f.write(flow_xml)

    xml_problems = validate_generated_xml(flow_xml, vtype_xml)
    ordering_ok, ordering_msg = validate_flow_ordering(flow_xml)

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
        "xml_valid": len(xml_problems) == 0,
        "xml_problems": xml_problems,
        "flow_ordering_ok": ordering_ok,
        "flow_ordering_detail": ordering_msg,
        "output_dir": str(scenario_dir),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRID scenario generator -- fixed 12-scenario set (v0.4.0)")
    parser.add_argument("--config", default="scenario_config.json")
    parser.add_argument("--out", default="generated_scenarios")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

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

    dev_entries = [e for e in manifest_entries if e["design_method"] == "development"]
    ood_entries = [e for e in manifest_entries if e["design_method"] == "ood"]
    all_seeds = [e["seed"] for e in manifest_entries]
    all_ids = [e["scenario_id"] for e in manifest_entries]
    all_xml_valid = all(e["xml_valid"] for e in manifest_entries)
    all_ordering_ok = all(e["flow_ordering_ok"] for e in manifest_entries)

    manifest = {
        "generator": "scenario_builder.py",
        "config_version": cfg["meta"]["config_version"],
        "strategy": cfg["v0_design"]["strategy"],
        "written_count": len(manifest_entries),
        "development_count": len(dev_entries),
        "ood_count": len(ood_entries),
        "rejected_count": len(rejected),
        "no_duplicate_scenario_ids": len(all_ids) == len(set(all_ids)),
        "no_duplicate_seeds": len(all_seeds) == len(set(all_seeds)),
        "all_xml_valid": all_xml_valid,
        "all_flow_ordering_ok": all_ordering_ok,
        "_ood_rule": (
            "The 4 design_method='ood' / split='ood' scenarios must NEVER be used for training, "
            "feature fitting, model tuning, hyperparameter selection, or model selection -- reserved "
            "for the final generalization test only. dataset/assemble_dataset.py writes them to a "
            "separate ood_features.csv/ood_labels.csv; models/tree_model.py evaluates but never "
            "gates approval on OOD performance."
        ),
        "scenarios": manifest_entries,
        "rejected": rejected,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    header = (f"{'scenario_id':28} {'split':6} {'method':12} {'seed':6} {'demand':22} "
              f"{'approach':15} {'movement':15} {'arrival':9} {'Y':6} xml  order")
    print(header)
    print("-" * len(header))
    for e in manifest_entries:
        demand_str = f"{e['demand_class']}({e['demand_rate_veh_per_hour']:.0f}/h)"
        print(
            f"{e['scenario_id']:28} {e['split']:6} {e['design_method']:12} {e['seed']:<6} "
            f"{demand_str:22} {e['approach_pattern']:15} {e['movement_pattern']:15} "
            f"{e['arrival_pattern']:9} {e['webster_Y']:<6} "
            f"{'OK' if e['xml_valid'] else 'FAIL':4} {'OK' if e['flow_ordering_ok'] else 'FAIL'}"
        )

    if rejected:
        print("\nREJECTED (failed validation, NOT written to disk):")
        for r in rejected:
            print(f"  {r['scenario_id']}: {r['errors']}")

    print()
    print("=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    print(f"Total scenarios generated : {len(manifest_entries)}  {'OK' if len(manifest_entries) == 12 else 'FAIL (expected 12)'}")
    print(f"Development scenarios     : {len(dev_entries)}  {'OK' if len(dev_entries) == 8 else 'FAIL (expected 8)'}")
    print(f"OOD scenarios             : {len(ood_entries)}  {'OK' if len(ood_entries) == 4 else 'FAIL (expected 4)'}")
    print(f"No duplicate scenario IDs : {'OK' if manifest['no_duplicate_scenario_ids'] else 'FAIL'}")
    print(f"No duplicate seeds        : {'OK' if manifest['no_duplicate_seeds'] else 'FAIL'}")
    print(f"All XML structurally valid: {'OK' if all_xml_valid else 'FAIL'}")
    print(f"All flow.xml time-ordered : {'OK' if all_ordering_ok else 'FAIL'}")
    ood_in_dev_split = any(e["design_method"] == "ood" and e["split"] != "ood" for e in manifest_entries)
    print(f"No OOD scenario in dev split: {'OK' if not ood_in_dev_split else 'FAIL'}")

    print()
    print("Development scenarios (8) -- train/val/test:")
    for e in dev_entries:
        print(f"  {e['scenario_id']:28} split={e['split']:5} seed={e['seed']}")
    print("\nOOD scenarios (4) -- held out entirely:")
    for e in ood_entries:
        print(f"  {e['scenario_id']:28} seed={e['seed']}")

    print(f"\nWrote {len(manifest_entries)}/{len(scenarios)} scenarios to {out_dir.resolve()}")
    print(f"Manifest: {(out_dir / 'manifest.json').resolve()}")

    if len(manifest_entries) != 12 or len(dev_entries) != 8 or len(ood_entries) != 4:
        sys.exit(1)


if __name__ == "__main__":
    main()